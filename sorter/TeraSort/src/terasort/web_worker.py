"""Run one queued sort independently of the dashboard process."""

import json
import math
import sys
import time
import traceback
from pathlib import Path


def write_progress_snapshot(path: Path, payload: dict) -> bool:
    """Publish advisory telemetry without making sorting depend on file locks.

    Windows readers/antivirus can briefly deny atomic replacement. Preserve
    the last complete snapshot if retries fail; the next update tries again.
    Sort checkpoints are independent and retain their strict error handling.
    """
    temporary = path.with_suffix(".json.partial")
    encoded = json.dumps(payload)
    for attempt in range(3):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            return True
        except PermissionError:
            if attempt < 2:
                time.sleep(.01 * (attempt + 1))
        except OSError:
            return False
    return False


def bounded_stream_sizes(channels: int, rate_hz: float) -> tuple[float, int]:
    """Keep prefetched raw cores small even for dense flexible probes."""
    if channels < 1 or not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("Valid channel count and sample rate required")
    seconds = min(2., max(.1, 128 * 1024**2 / (channels * rate_hz * 2)))
    core_samples = max(1, round(seconds * rate_hz))
    halo_samples = round(.1 * rate_hz)
    needed_mb = math.ceil((core_samples + 2 * halo_samples) * channels * 2 / 1024**2)
    if needed_mb > 1024:
        raise ValueError("One core plus halos exceeds the 1 GiB read-buffer limit")
    return seconds, max(64, needed_mb + 2)


def effective_request(job: dict) -> dict:
    request = job["request"]
    if not job.get("retry_of") or request.get("backend") in ("bounded_session", "bounded_ks1"):
        return request
    from .read_cache import retry_with_current_cache_defaults
    return retry_with_current_cache_defaults(request)


def build_args(request: dict) -> list[str]:
    args = ["sort", "--settings", request["settings"],
            "--results-dir", request["results_dir"], "--backend", request["backend"]]
    for filename in request.get("filenames", [request["filename"]]):
        args += ["--filename", filename]
    if request.get("probe_json"):
        args += ["--probe-json", request["probe_json"]]
    elif request.get("probe_name"):
        args += ["--probe-name", request["probe_name"]]
    for key, flag in (("skip_drift_correction", "--skip-drift-correction"),
                      ("no_fast_int16", "--no-fast-int16")):
        if request.get(key):
            args.append(flag)
    if request.get("lfp_output"):
        args += ["--lfp-output", request["lfp_output"]]
    if request.get("stage_dir"):
        args += ["--stage-dir", request["stage_dir"]]
    if request.get("read_cache_dir"):
        args += ["--read-cache-dir",request["read_cache_dir"],
                 "--read-cache-mb",str(request["read_cache_mb"]),
                 "--read-cache-slots",str(request["read_cache_slots"]),
                 "--read-cache-workers",str(request.get("read_cache_workers", 4))]
        if request.get("read_cache_retain"):
            args.append("--read-cache-retain")
            args += ["--read-cache-retain-gb", str(request.get("read_cache_retain_gb", 0.))]
    return args


def main() -> int:
    job_file = Path(sys.argv[1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    request = effective_request(job)
    code = 1
    try:
        if request['backend'] == 'bounded_ks1':
            from .session_ks1_sort import run_ks1_session
            from .session_manifest import load_session
            session=load_session(request['session_manifest'])
            total_samples=max(p.stop_sample for p in session.probes)
            first_core_stop=[0];sort_started_at=[0.];last_stop=[0];last_write=[0.]
            def progress(event):
                now=time.monotonic()
                if event['stage']=='core_complete' and now-last_write[0]<2.:return
                stop=int(event.get('stop_sample',last_stop[0]));last_stop[0]=max(stop,last_stop[0])
                if event['stage']=='core_complete' and not first_core_stop[0]:
                    first_core_stop[0]=stop;sort_started_at[0]=time.time()
                payload=dict(
                    stage=event['stage'],stop_sample=last_stop[0],total_samples=total_samples,
                    first_core_stop=first_core_stop[0],sort_started_at=sort_started_at[0],updated_at=time.time())
                for key in ('completed_shards','total_shards','probe_id','day_id'):
                    if key in event:payload[key]=event[key]
                write_progress_snapshot(job_file.with_name('session_progress.json'),payload)
                last_write[0]=now
            result=run_ks1_session(request['session_manifest'],request['results_dir'],
                kilosort_path=request['ks1_path'],matlab=request['matlab'],
                model_path=request.get('ks1_model'),resume=bool(request.get('resume')),progress=progress)
            print(json.dumps(result),flush=True);code=0
        elif request["backend"] == "bounded_session":
            import cupy as cp
            from .session_hybrid import session_sort_profile, manifest_for_session_run
            from .session_sort import run_session
            from .session_export import ensure_session_export

            free_vram, total_vram = cp.cuda.runtime.memGetInfo()
            if free_vram < 12 * 1024**3:
                raise RuntimeError(
                    f"Bounded session sorter needs 12 GiB free VRAM before launch; "
                    f"{free_vram / 1024**3:.1f} GiB free on a "
                    f"{total_vram / 1024**3:.1f} GiB GPU. Wait for other GPU jobs to finish.")
            with Path(request["session_manifest"]).open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            final_segment = manifest["probes"][0]["segments"][-1]
            total_samples = final_segment["start_sample"] + final_segment["n_samples"]
            probe = manifest["probes"][0]
            core_seconds, read_buffer_mb = bounded_stream_sizes(
                len(probe["geometry"]["x_um"]), float(probe["sample_rate_hz"]))
            progress_path = job_file.with_name("session_progress.json")
            last_write = [0.]
            first_core_stop = [0]
            sort_started_at = [0.]
            last_stop = [0]
            progress_io_warned = [False]

            def progress(event):
                now = time.monotonic()
                if event["stage"] == "core_complete" and now - last_write[0] < 2.:
                    return
                stop = int(event.get("stop_sample", last_stop[0]))
                last_stop[0] = max(last_stop[0], stop)
                if event["stage"] == "core_complete" and not first_core_stop[0]:
                    first_core_stop[0] = stop
                    sort_started_at[0] = time.time()
                payload = {"stage": event["stage"],
                           "stop_sample": last_stop[0],
                           "first_core_stop": first_core_stop[0],
                           "sort_started_at": sort_started_at[0],
                           "total_samples": total_samples, "updated_at": time.time()}
                for key in ("completed_shards", "total_shards", "probe_id", "day_id"):
                    if key in event:
                        payload[key] = event[key]
                if not write_progress_snapshot(progress_path, payload) and not progress_io_warned[0]:
                    print("Progress snapshot temporarily unavailable; sorting continues. "
                          "Dashboard updates will resume when the file is writable.",
                          flush=True)
                    progress_io_warned[0] = True
                last_write[0] = now

            print(f"TeraSort bounded session worker started; {core_seconds:.3f}s cores, "
                  f"{read_buffer_mb} MiB read buffer, 10 GiB GPU pool cap.", flush=True)
            session_manifest, resume_sort = manifest_for_session_run(
                request["session_manifest"], request["results_dir"],
                resume=bool(request.get("resume")))
            legacy_resume = resume_sort and Path(session_manifest) == Path(
                request["session_manifest"]).resolve()
            sort_profile = {} if legacy_resume else session_sort_profile(
                request["results_dir"], resume=bool(request.get("resume")))
            sort_profile.update(
                backend="cuda", resume=resume_sort,
                core_seconds=core_seconds, read_buffer_mb=read_buffer_mb,
                prefetch_depth=4, shard_seconds=60.,
                vram_limit_gb=10.,
                progress=progress)
            if legacy_resume:
                sort_profile.update(
                    freeze_templates=True, novelty='off', detector_mode="smooth3",
                    score_floor=.76, fit_amplitude_min=.7,
                    rescue_floor_snr=3.5, rescue_passes=2,
                    rescue_last_score_floor=.8,
                    template_merge_cosine=.93,
                    template_merge_radius_um=32.,
                    waveform_cache_mode="balanced")
            result = run_session(
                session_manifest, request["results_dir"],
                **sort_profile)
            print("Sorting complete; exporting Kilosort/Phy arrays...", flush=True)
            result["phy_export"] = ensure_session_export(request["results_dir"], progress=progress)
            progress({"stage": "completed", "stop_sample": total_samples})
            print(json.dumps(result), flush=True)
            code = 0
        else:
            args = build_args(request)
            print("TeraSort worker started; initializing Kilosort and CUDA...", flush=True)
            if job.get("retry_of"):
                print("TERASORT_RETRY_CURRENT_CODE " + json.dumps({
                    "backend": request["backend"],
                    "read_cache_slots": request.get("read_cache_slots"),
                }), flush=True)
            from .cli import main as cli_main
            code = cli_main(args)
    except BaseException:
        traceback.print_exc()
    finally:
        done = job_file.with_name("done.json")
        temporary = done.with_suffix(".tmp")
        temporary.write_text(json.dumps({"exit_code": code, "finished_at": time.time()}),
                             encoding="utf-8")
        temporary.replace(done)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
