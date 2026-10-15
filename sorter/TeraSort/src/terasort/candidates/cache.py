"""Add all-candidate waveform caches to an immutable SCB ledger."""
from itertools import zip_longest
from pathlib import Path
import h5py
from .bank import BankWriter, CandidateBank, sha256
from .waveforms import extract_waveforms


def cache_waveforms(source_path, output_path, spec, voltage_blocks, provenance):
    """Write a new bank from the original ledger and matching filtered blocks.

    Each voltage block is (core_start, core_stop, voltage_start, voltage), with
    absolute source-sample coordinates. Match the original detection block's
    preprocessing and provide enough halo for the requested waveform window.
    Memory is bounded by one detection block and its local snippets.
    """
    bank = CandidateBank(source_path)
    if not len(bank.blocks):
        raise ValueError('Source bank needs indexed recording bounds')
    if Path(output_path).resolve() == Path(source_path).resolve():
        raise ValueError('Write waveform extension to a new file')
    manifest = dict(bank.manifest)
    manifest['waveform_provenance'] = dict(provenance)
    manifest['waveform_provenance'].update(parent_bank_sha256=sha256(source_path),
                                          parent_bank_name=Path(source_path).name)
    with h5py.File(source_path, 'r') as original:
        with BankWriter(output_path, manifest, original['noise_uv'][:],
                        original['channel_positions_um'][:], waveform_spec=spec) as writer:
            for block, payload in zip_longest(bank.blocks, voltage_blocks):
                if block is None or payload is None:
                    raise ValueError('Voltage iterator must cover every bank block exactly once')
                start, stop, voltage_start, voltage = payload
                if (start, stop) != (block['start_sample'], block['stop_sample']):
                    raise ValueError('Voltage core does not match bank block')
                events = original['events'][int(block['row_start']):int(block['row_stop'])]
                waveforms, valid_start, valid_stop = extract_waveforms(
                    events, voltage, voltage_start, spec,
                    bank.blocks['start_sample'][0], bank.blocks['stop_sample'][-1])
                writer.append(events, int(start), int(stop), waveforms, valid_start, valid_stop)
    return CandidateBank(output_path)
