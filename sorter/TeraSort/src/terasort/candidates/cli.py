"""Inspect or query a candidate bank without CUDA or raw-voltage access."""
import argparse
import json
from pathlib import Path
import numpy as np
from .bank import CandidateBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--snr', type=float, help='Strict SNR > threshold; cannot be below saved floor')
    parser.add_argument('--channels', type=int, nargs='+', help='Zero-based recording channel indices')
    parser.add_argument('--start-sample', type=int, help='Inclusive original-source sample index')
    parser.add_argument('--stop-sample', type=int, help='Exclusive original-source sample index')
    parser.add_argument('--polarity', choices=['both','neg','pos'], default='both')
    parser.add_argument('--waveforms', action='store_true', help='Include local multichannel cached waveforms')
    parser.add_argument('--batch-size', type=int, default=512, help='Maximum waveform source rows per read')
    parser.add_argument('--output', type=Path, help='Optional .npy events or .npz waveforms; materializes selection in RAM')
    args = parser.parse_args()
    bank = CandidateBank(args.bank)
    if args.snr is None:
        if args.output or args.waveforms:
            parser.error('--output and --waveforms require --snr')
        print(json.dumps(dict(events=bank.n_events, channels=bank.n_channels,
                              blocks=len(bank.blocks), manifest=bank.manifest), indent=2))
        return
    kwargs = dict(channels=args.channels, start_sample=args.start_sample,
                  stop_sample=args.stop_sample, polarity=args.polarity)
    if args.waveforms:
        if bank.waveform_spec is None:
            parser.error('This file has no cached waveforms')
        spec = bank.waveform_spec
        batches = bank.iter_waveforms(args.snr, batch_size=args.batch_size, **kwargs)
        if args.output:
            if args.output.suffix != '.npz':
                parser.error('Waveform export path must end in .npz')
            with args.output.open('xb') as f:
                chunks = list(batches)
                shapes = dict(row_indices=(0,), events=(0,), waveforms_uv=(0, spec.n_samples, spec.width),
                              channel_indices=(0, spec.width), valid_start=(0,), valid_stop=(0,))
                from .bank import EVENT_DTYPE
                dtypes = dict(row_indices=np.int64, events=EVENT_DTYPE, waveforms_uv=np.float32,
                              channel_indices=np.int32, valid_start=np.int32, valid_stop=np.int32)
                payload = {key: np.concatenate([getattr(b, key) for b in chunks]) if chunks
                           else np.empty(shape, dtypes[key]) for key, shape in shapes.items()}
                count = len(payload['events'])
                np.savez_compressed(f, **payload,
                                    sample_offsets=np.arange(-spec.pre_samples, spec.post_samples+1),
                                    sample_rate_hz=bank.manifest['sample_rate_hz'],
                                    waveform_metadata_json=json.dumps(spec.metadata(), sort_keys=True))
        else:
            count = sum(len(batch.events) for batch in batches)
    elif args.output:
        if args.output.suffix != '.npy':
            parser.error('Export path must end in .npy')
        selected = bank.query(args.snr, **kwargs)
        # Exclusive creation avoids silently replacing an earlier selection.
        with args.output.open('xb') as f:
            np.save(f, selected, allow_pickle=False)
        count = len(selected)
    else:
        count = sum(len(events) for events in bank.iter_query(args.snr, **kwargs))
    print(json.dumps(dict(candidates=count, threshold_snr=args.snr,
                          polarity=args.polarity, waveforms=args.waveforms,
                          output=str(args.output) if args.output else None)))


if __name__ == '__main__':
    main()
