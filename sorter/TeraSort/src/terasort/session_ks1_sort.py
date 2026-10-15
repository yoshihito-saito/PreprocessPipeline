"""Selectable KS1 engine: bounded native learning and streamed CUDA fitting."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import time

import numpy as np
from scipy.io import savemat

from .session_ks1 import KS1Core, RULE, SOURCE_HASHES, device_source, load_model
from .session_manifest import load_session
from .session_models import LocalModels, Match
from .session_signal import Core, assess_quality, iter_cores, prefetch_cores
from .session_store import RunJournal, ShardWriter

STRIDE = 32768
SHARD_CORES = 32


def read_buffer_mb(probe):
    if not 3<=probe.n_channels<=1024:
        raise ValueError('Bounded KS1 currently supports 3-1024 raw channels per probe; split larger arrays into physical probes')
    return max(64,int(np.ceil((STRIDE+640)*probe.n_channels*2/1024**2)))


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.items()):
        array = np.asarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_probe(model, probe):
    channels = model['channels']
    if (np.any(channels >= probe.n_channels) or
            not np.isclose(float(model['sample_rate']), probe.sample_rate_hz, rtol=0, atol=1e-6) or
            not np.allclose(model['positions'], probe.geometry[channels], rtol=0, atol=1e-4) or
            np.intersect1d(channels, probe.bad_channels).size):
        raise ValueError('KS1 model acquisition geometry/rate/bad contacts differ from manifest')


def local_models(model, probe):
    """Small physical templates for checkpoints and Phy; native factors fit spikes."""
    nf = len(model['mu']); width = min(16, len(model['channels']))
    waves = np.zeros((nf, 61, width), np.float32)
    contacts = np.empty((nf, width), np.int64)
    for unit in range(nf):
        spatial = model['WrotInv'].T @ model['U'][:,unit,:]
        physical = model['W'][:,unit,:] @ spatial.T * float(model['mu'][unit])
        selected = np.argsort(-np.max(np.abs(physical), axis=0), kind='stable')[:width]
        contacts[unit] = model['channels'][selected]
        # Native KS1 center is index 20; the common curation window uses 30.
        waves[unit,10:] = physical[:51,selected] * probe.gain_uv_per_count
    return LocalModels(waves, contacts, contacts[:,0], np.zeros(nf,np.int64))


def learn_model(probe, day_id, root, kilosort_path, matlab, progress=None):
    """Native KS1 learning, limited to <=60s and <=1 GiB raw preview.

    Only bounded previews reach MATLAB. Dense fitting never creates a full
    whitened recording, st3, cProj or cProjPC in MATLAB.
    """
    from .session_calibration import select_calibration_windows
    from .session_seed import stage_seed_previews
    root = Path(root); source = Path(kilosort_path).resolve(strict=True)
    active = np.setdiff1d(np.arange(probe.n_channels), probe.bad_channels)
    if not 3 <= len(active) <= 1024:
        raise ValueError('Bounded KS1 calibration currently supports 3-1024 active channels per probe')
    for filename in SOURCE_HASHES:
        device_source(source/'CUDA'/filename)
    for name in ('mexMPregMU','mexWtW2'):
        if not list((source/'CUDA').glob(name+'.mex*')):
            raise FileNotFoundError('KS1 calibration requires compatible compiled native MEX files')
    budget = min(60., (1024**3-16*1024**2)/(probe.sample_rate_hz*probe.n_channels*2))
    if budget < 6:
        raise ValueError('Acquisition too wide for the bounded KS1 calibration budget')
    windows = select_calibration_windows(probe,day_id,window_seconds=budget/3,target_windows=3)
    root.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    if progress:progress(dict(stage='calibration_start',probe_id=probe.probe_id,day_id=day_id))
    previews, io = stage_seed_previews(probe,windows,root,scratch_limit_bytes=1024**3)
    raw = root/'calibration.dat'
    # Join only the small local preview files. No source file is copied whole.
    with raw.open('xb') as out:
        for preview in previews:
            with preview.path.open('rb') as handle:
                while block := handle.read(8*1024**2):
                    out.write(block)
            for path in (preview.path, preview.prefix, preview.suffix):
                path.unlink()
    chanmap=root/'channels.mat'
    savemat(chanmap,dict(chanMap=np.arange(1,probe.n_channels+1)[:,None],
        connected=np.isin(np.arange(probe.n_channels),active)[:,None],
        xcoords=probe.x_um[:,None],ycoords=probe.y_um[:,None],
        kcoords=(probe.shank+1)[:,None],fs=probe.sample_rate_hz))
    quote=lambda p:"'"+Path(p).resolve().as_posix().replace("'","''")+"'"
    script=root/'learn.m'
    # Native learning state and preview size are fixed; no dense final pass.
    script.write_text(f"""addpath(genpath({quote(source)}));
run({quote(source/'configFiles/StandardConfig_MOVEME.m')});
rng(1); ops.showfigures=0;
ops.fbinary={quote(raw)};ops.fproc={quote(root/'whitened.dat')};
ops.root={quote(root)};ops.chanMap={quote(chanmap)};
ops.fs={probe.sample_rate_hz:.17g};ops.NchanTOT={probe.n_channels};ops.Nchan={len(active)};
ops.Nfilt={min(768,32*((2*len(active)+31)//32))};
ops.nNeigh=[];ops.nNeighPC=[];
ops.momentum=1./[2 32];
[rez,DATA,uproj]=preprocessData(ops);
assert(rez.temp.Nbatch_buff==rez.temp.Nbatch,'Not enough RAM for bounded preview cache');
rez=fitTemplates(rez,DATA,uproj);clear DATA uproj;
[W,U,mu]=decompose_dWU(rez.ops,rez.dWU,3,rez.ops.kcoords);
model.ops=rez.ops;model.ops.chanMap={matlab_vector(active+1)}';
model.W=gather_try(W);model.U=gather_try(U);model.mu=gather_try(mu);
model.Wrot=gather_try(rez.Wrot);model.WrotInv=inv(model.Wrot)*rez.ops.scaleproc;
model.xcoords=rez.xc;model.ycoords=rez.yc;model.temp=rez.temp;
clear rez;rez=model;save({quote(root/'model.mat')},'rez','-v7.3');
""",encoding='utf-8')
    subprocess.run([str(matlab),'-wait','-batch',f'run({quote(script)})'],cwd=root,check=True)
    model=load_model(root/'model.mat');validate_probe(model,probe)
    np.savez(root/'model.npz',**model)
    report=dict(rule=RULE,day_id=day_id,windows=windows,preview_budget_seconds=budget,
                model_sha256=model_digest(model),wall_seconds=time.perf_counter()-start,
                learning_source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(source.rglob('*.m'))},**io)
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for path in (raw,root/'whitened.dat',root/'model.mat'):
        if path.exists():path.unlink()
    if progress:progress(dict(stage='calibration_complete',probe_id=probe.probe_id,day_id=day_id,
                              units=int(np.count_nonzero(model['mu'])),wall_seconds=report['wall_seconds']))
    return model


def matlab_vector(values):
    return '['+' '.join(str(int(v)) for v in values)+']'


def find_matlab():
    import shutil
    executable=os.environ.get('TERASORT_MATLAB') or shutil.which('matlab')
    if not executable and os.name=='nt':
        versions=sorted((Path(os.environ.get('ProgramFiles','C:/Program Files'))/'MATLAB').glob('R*/bin/matlab.exe'))
        if versions:executable=versions[-1]
    return Path(executable) if executable else None


def batch_frame(core, segment, model):
    """One native batch and its exact source origin, never float source clocks."""
    first = core.core_start == segment.start_sample
    raw_start = segment.start_sample if first else core.core_start-128
    offset = 0 if first else 64
    begin = raw_start-core.data_start
    raw = core.raw[begin:begin+33088]
    if len(raw)<33088:
        raw=np.pad(raw,((0,33088-len(raw)),(0,0)),mode='edge')
    origin=raw_start+offset
    # Native overlap ownership: later batches cover the 64 samples before
    # their nominal stride. This also avoids losing the final 44 samples.
    owned_start = segment.start_sample if first else core.core_start-44
    owned_stop = min(segment.stop_sample,core.core_stop-44) if core.core_stop<segment.stop_sample else segment.stop_sample
    if first:owned_stop=min(segment.stop_sample,core.core_stop+20)
    view=Core(core.probe_id,core.day_id,core.segment_path,owned_start,owned_stop,
              origin,raw[offset:offset+int(model['NT'])],core.source_bytes_read)
    return raw,offset,view


def run_ks1_session(manifest, output_root, *, kilosort_path, matlab=None,
                    model_path=None, resume=False, progress=None, _core_class=KS1Core):
    """Fixed model per probe/day; immutable shards, bounded source/feature buffers."""
    all_started=time.perf_counter()
    session=load_session(manifest); output=Path(output_root).resolve()
    if any(p.input_is_preprocessed or p.channel_map is not None for p in session.probes):
        raise ValueError("Explicit source-column/preprocessed input is supported by session-sort, not ks1-sort")
    buffers={p.probe_id:read_buffer_mb(p) for p in session.probes}
    source=Path(kilosort_path).resolve(strict=True)
    for filename in SOURCE_HASHES:device_source(source/'CUDA'/filename)
    keys=[(p,day) for p in session.probes for day in sorted({s.day_id for s in p.segments})]
    if model_path and len(keys)!=1:
        raise ValueError('An imported KS1 model requires exactly one probe/day')
    if not resume and output.exists():raise FileExistsError(output)
    if resume and not (output/'run.json').is_file():raise ValueError('No published KS1 run to resume')
    banks={}; hashes={}
    source_record=session.as_source_record()
    calibration=output.with_name(output.name+'.ks1-calibration')
    expected=dict(source=source_record,rule=RULE)
    if not model_path and not resume:
        calibration.mkdir(parents=True,exist_ok=True)
        marker=calibration/'source.json'
        if marker.exists() and json.loads(marker.read_text())!=expected:
            raise ValueError('Calibration directory belongs to different sources')
        marker.write_text(json.dumps(expected,indent=2),encoding='utf-8')
    for probe,day in keys:
        key=probe.probe_id+'/'+day
        if resume:
            model=load_model(output/'models'/probe.probe_id/(day+'.npz'))
        elif model_path:
            model=load_model(model_path)
        else:
            root=calibration/probe.probe_id/day
            if (root/'report.json').is_file():
                model=load_model(root/'model.npz')
                if model_digest(model)!=json.loads((root/'report.json').read_text())['model_sha256']:
                    raise ValueError('Calibration model changed')
            else:
                if not matlab:raise ValueError('MATLAB executable is required for bounded KS1 calibration')
                # Publish only a completed bank; failed scratch is bounded and disposable.
                root.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=day+'-',dir=root.parent) as attempt:
                    learned=Path(attempt)/'learn'
                    model=learn_model(probe,day,learned,source,matlab,progress)
                    os.replace(learned,root)
        validate_probe(model,probe);hashes[key]=model_digest(model)
        banks[key]=(output/'models'/probe.probe_id/(day+'.npz') if resume else
                    Path(model_path) if model_path else calibration/probe.probe_id/day/'model.npz')
        del model
    config=dict(engine=RULE,models_sha256=hashes,cuda_sources=SOURCE_HASHES,
                stride_samples=STRIDE,shard_cores=SHARD_CORES,read_buffer_mb=64,prefetch_depth=4,
                candidate_coverage='accepted KS1 template events; no independent threshold candidate bank',
                phy_feature_source='resident KS1 high-pass voltage',cross_day_tracking='unresolved')
    if any(value!=64 for value in buffers.values()):config['read_buffer_mb_by_probe']=buffers
    journal=RunJournal(output,source_record,config,resume=resume)
    if not resume:
        for probe,day in keys:
            folder=output/'models'/probe.probe_id;folder.mkdir(parents=True,exist_ok=True)
            model=load_model(banks[probe.probe_id+'/'+day])
            np.savez(folder/(day+'.npz'),**model)
            banks[probe.probe_id+'/'+day]=folder/(day+'.npz')
            del model
    started=time.perf_counter();total_spikes=0;read_bytes=0;completed=0
    phases=dict(calibration_seconds=started-all_started,preprocessing_seconds=0.,
                fitting_seconds=0.,qc_seconds=0.,shard_writing_seconds=0.)
    def emit(stage,**kw):
        event=dict(stage=stage,**kw)
        if progress:progress(event)
    for probe in session.probes:
        engine=None;current_key=None;writer=None
        # Reuse the existing sequential two-block reader and asynchronous core
        # queue across shard boundaries. RAM is independent of source duration.
        start_sample=resume_start(probe,journal)
        if start_sample is None:continue
        cores=prefetch_cores(iter_cores(probe,core_seconds=STRIDE/probe.sample_rate_hz,
              halo_samples=320,read_buffer_mb=buffers[probe.probe_id],start_sample=start_sample),depth=4)
        try:
            for core in cores:
                segment=next(s for s in probe.segments if s.path==core.segment_path and s.start_sample<=core.core_start<s.stop_sample)
                batch=(core.core_start-segment.start_sample)//STRIDE
                first=segment.start_sample+(batch//SHARD_CORES)*SHARD_CORES*STRIDE
                stop=min(segment.stop_sample,first+SHARD_CORES*STRIDE)
                # Match ownership gives internal shard boundaries -44 samples.
                shard_start=first if first==segment.start_sample else first-44
                shard_stop=stop if stop==segment.stop_sample else stop-44
                path=journal.shard_path(probe.probe_id,shard_start,shard_stop)
                if journal.completed(path):
                    if batch%SHARD_CORES==0:completed+=1
                    continue
                key=probe.probe_id+'/'+core.day_id
                if key!=current_key:
                    if engine is not None:
                        del engine
                    model=load_model(banks[key])
                    engine=_core_class(model,source);current_key=key
                    models=local_models(model,probe)
                    checkpoint=journal.latest_model(probe.probe_id,shard_start)
                    if checkpoint is not None and checkpoint[1][3].shape==models.assigned.shape:
                        # Counts are diagnostic; no across-day identity merge.
                        previous_day=next((s.day_id for s in probe.segments
                            if s.start_sample<checkpoint[0]<=s.stop_sample),None)
                        if previous_day==core.day_id:models.assigned[:]=checkpoint[1][3]
                    from .session_phy_capture import curation_basis
                    basis=curation_basis(models)
                    channel_map=np.full((probe.n_channels,models.channels.shape[1]),-1,np.int64)
                    # Cache uses anchor-local geometry, not a template ID.
                    from .candidates.waveforms import geometry_channel_map
                    channel_map=geometry_channel_map(probe.geometry,probe.shank,75.,models.channels.shape[1],excluded_channels=probe.bad_channels)
                if writer is None:
                    partial=path.with_suffix('.h5.partial')
                    if resume and partial.exists():partial.unlink()
                    writer=ShardWriter(path,probe=probe,day_id=core.day_id,start_sample=shard_start,
                        stop_sample=shard_stop,config_sha256=journal.record['config_sha256'],
                        phy_feature_basis=basis,phy_feature_width=models.channels.shape[1])
                    writer.handle.attrs['candidate_coverage']=config['candidate_coverage']
                    writer.handle.attrs['engine']=RULE
                    writer.handle.attrs['score_semantics']='native KS1 squared regularized template score'
                raw,offset,view=batch_frame(core,segment,model)
                before=time.perf_counter()
                whitened,filtered=engine.preprocess(raw,offset=offset)
                phases['preprocessing_seconds']+=time.perf_counter()-before
                before=time.perf_counter()
                st,ids,amplitudes,scores,passes=engine.match(whitened)
                phases['fitting_seconds']+=time.perf_counter()-before
                centers=st.astype(np.int64)+int(model['nt0min'])
                absolute=centers+view.data_start
                keep=(absolute>=view.core_start)&(absolute<view.core_stop)
                centers,ids,amplitudes,scores,passes=(v[keep] for v in (centers,ids,amplitudes,scores,passes))
                voltage=np.zeros((int(model['NT']),probe.n_channels),np.float32)
                voltage[:,model['channels']]=filtered*probe.gain_uv_per_count
                before=time.perf_counter()
                quality=assess_quality(view,voltage,probe)
                phases['qc_seconds']+=time.perf_counter()-before
                matches=[Match(int(t),int(models.anchors[u]),int(u),float(c),0.,float(a/model['mu'][u]),float(c),int(p))
                         for t,u,a,c,p in zip(centers,ids,amplitudes,scores,passes)]
                events=np.c_[centers,models.anchors[ids],np.sqrt(np.maximum(scores,0))]
                before=time.perf_counter()
                writer.append_core(core=view,voltage_uv=voltage,quality=quality,
                    threshold_events=events,matches=matches,channel_map=channel_map,
                    cache_fraction=.05*(core.core_stop-core.core_start)/(view.core_stop-view.core_start),
                    model_channels=models.channels)
                phases['shard_writing_seconds']+=time.perf_counter()-before
                models.assigned+=np.bincount(ids,minlength=len(models.assigned))
                total_spikes+=len(ids);read_bytes+=core.source_bytes_read
                emit('core_complete',probe_id=probe.probe_id,day_id=core.day_id,
                     stop_sample=view.core_stop,spikes=len(ids))
                if core.core_stop==stop:
                    writer.finish(models,telemetry=dict(engine=RULE))
                    writer=None;completed+=1
                    emit('shard_complete',probe_id=probe.probe_id,stop_sample=shard_stop)
        finally:
            cores.close()
            if writer is not None:writer.abort()
    phases['dense_seconds']=time.perf_counter()-started
    result=dict(engine=RULE,spikes_written=total_spikes,source_bytes_read=read_bytes,
                completed_shards=completed)
    from .session_export import ensure_session_export
    before=time.perf_counter()
    result['phy_export']=ensure_session_export(output,progress=progress)
    phases['export_seconds']=time.perf_counter()-before
    result.update(phases=phases,wall_seconds=time.perf_counter()-all_started)
    emit('completed')
    return result


def resume_start(probe,journal):
    """Find the first unfinished shard before opening any voltage source."""
    for segment in probe.segments:
        for first in range(segment.start_sample,segment.stop_sample,SHARD_CORES*STRIDE):
            stop=min(segment.stop_sample,first+SHARD_CORES*STRIDE)
            lo=first if first==segment.start_sample else first-44
            hi=stop if stop==segment.stop_sample else stop-44
            if not journal.completed(journal.shard_path(probe.probe_id,lo,hi)):
                return first
    return None
