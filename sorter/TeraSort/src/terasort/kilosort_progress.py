"""Scoped, pinned-source loop counters. Does not change numerical operations."""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import textwrap

from .job_progress import JobProgress

SOURCES = {
    ('preprocessing','get_whitening_matrix'): '150bdf6d33a32e3dd9a0627de607b70d8bd6d819bd5e963aae10c7b0c251c6f6',
    ('spikedetect','extract_wPCA_wTEMP'): '6d579eb4e0843d2e10dbab7d2bbcabb1333a789a0dbefa0be328c379dc7b7fd7',
    ('spikedetect','run'): 'b75f6e421a3adc6898dbd2b50bf00646dcf38dfba65fa405ef9e2054b819e889',
    ('template_matching','extract'): '51fde663961cd26d7dea7fab9bb96020a9d0d1ec8ef6528cf6a3de4ae5db067a',
    ('clustering_qr','run'): 'c81e475281e755212e15a1b4b49f38eeed4e01d041a1fd84cddbbab4756aff2a',
    ('template_matching','merging_function'): '2d15fe4c78c97def13e2a2f0e0b562362edda9e8e64ee70809496b947e2c996a',
    ('postprocessing','make_pc_features'): '8374977c95a1dc28c3ef6b70bc9e52fbc7d22f2f5f38b9a84a52ab02e4571dc9',
    ('CCG','refract'): '9d14ccd48ae345d147261288958c1547c49b6572740ba0ee2a15e93b9d9389c1',
}
MARKERS = {
    'Computing preprocessing variables.': 'preprocess',
    'Computing drift correction.': 'drift',
    'Extracting spikes using templates': 'detect',
    'First clustering': 'cluster1',
    'Extracting spikes using cluster waveforms': 'learned',
    'Final clustering': 'cluster2',
    'Merging clusters': 'merge',
    'Saving to phy and computing refractory periods': 'export',
    'Generating spike position plot ...': 'finalize',
}


def instrument(reference, module_name, progress):
    """Annotate only the outer work loops in verified upstream functions."""
    source = inspect.getsource(reference)
    key = (module_name, reference.__name__)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCES.get(key):
        raise ValueError(f'Unverified Kilosort source: {key}')
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.For):
            iterator = ast.unparse(node.iter)
            arguments = None
            if key == ('preprocessing','get_whitening_matrix') and iterator == 'range(0, f.n_batches - 1, nskip)':
                arguments = "'Whitening batches', 0., .98, ('preprocess',)"
            elif key == ('spikedetect','extract_wPCA_wTEMP') and iterator == 'range(0, bfile.n_batches, nskip)':
                arguments = "'Template-learning batches (upper bound)', 0., .1, ('drift','detect')"
            elif key == ('spikedetect','run') and iterator == 'prog':
                arguments = "'Detection batches', .1, .85, ('drift','detect')"
            elif key == ('template_matching','extract') and iterator == 'prog':
                arguments = "'Learned-detection batches', 0., .98, ('learned',)"
            elif key == ('postprocessing','make_pc_features') and iterator == 'np.unique(spike_clusters)':
                arguments = "'Export feature clusters', .35, .45, ('export',)"
            elif key == ('CCG','refract') and iterator == 'range(Nfilt)':
                # Called during merging as well; export-only avoids backward progress.
                arguments = "'Refractory checks', .8, .15, ('export',)"
            elif key == ('clustering_qr','run') and iterator == 'np.arange(len(ycent))':
                node.body.insert(0, ast.parse("_ts_progress.counter(jj*len(ycent)+kk, len(xcent)*len(ycent), 'Clustering regions', 0., .98, ('cluster1','cluster2'))").body[0])
                changed += 1
            if arguments:
                node.iter = ast.parse(f'_ts_progress.track({iterator}, {arguments})', mode='eval').body
                changed += 1
        elif key == ('template_matching','merging_function') and isinstance(node, ast.While) and ast.unparse(node.test) == 't < NN':
            node.body.insert(0, ast.parse("_ts_progress.counter(t, NN, 'Merge candidates', 0., .98, ('merge',))").body[0])
            changed += 1
    if changed != 1:
        raise ValueError(f'Expected one progress loop in {key}, found {changed}')
    ast.fix_missing_locations(tree)
    namespace = dict(reference.__globals__, _ts_progress=progress)
    exec(compile(tree, inspect.getsourcefile(reference), 'exec'), namespace)
    return namespace[reference.__name__]


class StageHandler(logging.Handler):
    def __init__(self, progress, lfp, on_stage=None):
        super().__init__(logging.INFO)
        self.progress, self.lfp, self.on_stage = progress, lfp, on_stage

    def emit(self, record):
        message = record.getMessage().strip()
        stage = MARKERS.get(message)
        if stage:
            previous = self.progress.active
            self.progress.stage(stage)
            if self.on_stage is not None and previous != stage and self.progress.active == stage:
                self.on_stage(stage)
        elif message == 'Sorting finished.' and self.lfp:
            self.progress.stage('lfp')


@contextmanager
def kilosort_progress(output=None, *, skip_drift=False, staging_seconds=0.,
                      lfp=False, on_stage=None, on_task_start=None):
    progress = JobProgress(output, skip_drift=skip_drift,
                           staging_seconds=staging_seconds, lfp=lfp,
                           on_task_start=on_task_start)
    originals, replacements = [], {}
    handler = StageHandler(progress, lfp, on_stage)
    logger = logging.getLogger('kilosort.run_kilosort')
    try:
        if importlib.metadata.version('kilosort') == '4.1.7':
            for module_name, name in SOURCES:
                module = importlib.import_module('kilosort.'+module_name)
                original = getattr(module, name)
                try:
                    replacement = instrument(original, module_name, progress)
                except (ValueError, OSError, TypeError, SyntaxError) as exc:
                    logging.getLogger(__name__).warning('Progress counter unavailable: %s', exc)
                    continue
                originals.append((module, name, original))
                replacements[original] = replacement
                setattr(module, name, replacement)
            # Retain aliases imported by value, especially io.make_pc_features.
            io = importlib.import_module('kilosort.io')
            alias = io.make_pc_features
            if alias in replacements:
                originals.append((io, 'make_pc_features', alias))
                io.make_pc_features = replacements[alias]
            for replacement in replacements.values():
                namespace = replacement.__globals__
                for name, value in list(namespace.items()):
                    if inspect.isfunction(value) and value in replacements:
                        namespace[name] = replacements[value]
        logger.addHandler(handler)
        yield progress
    finally:
        progress.publish(force=True)
        logger.removeHandler(handler)
        for module, name, original in reversed(originals): setattr(module, name, original)
