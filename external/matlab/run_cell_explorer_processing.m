function run_cell_explorer_processing(basepath, sourceBasepath, sortingFolder, cellExplorerRoot, varargin)
% Run the PreprocessPipeline CellExplorer post-Phy workflow.
%
% This wrapper intentionally uses the vendored CellExplorer path passed by
% the Python GUI. It keeps the selected Phy folder explicit and avoids
% relying on dir('Kilosort*') discovery.

p = inputParser;
addRequired(p, 'basepath', @ischar);
addRequired(p, 'sourceBasepath', @ischar);
addRequired(p, 'sortingFolder', @ischar);
addRequired(p, 'cellExplorerRoot', @ischar);
addParameter(p, 'preferMergePointsDat', true, @islogical);
addParameter(p, 'prePhy', false, @islogical);
addParameter(p, 'spikeLabels', {'good'}, @iscell);
parse(p, basepath, sourceBasepath, sortingFolder, cellExplorerRoot, varargin{:});

basepath = char(p.Results.basepath);
sourceBasepath = char(p.Results.sourceBasepath);
sortingFolder = char(p.Results.sortingFolder);
cellExplorerRoot = char(p.Results.cellExplorerRoot);
preferMergePointsDat = p.Results.preferMergePointsDat;
prePhy = p.Results.prePhy;
spikeLabels = p.Results.spikeLabels;

if ~isfolder(basepath)
    error('Local CellExplorer basepath does not exist: %s', basepath);
end
if ~isfolder(sourceBasepath)
    error('Source basepath does not exist: %s', sourceBasepath);
end
if ~isfolder(sortingFolder)
    error('Sorting folder does not exist: %s', sortingFolder);
end
if ~isfolder(cellExplorerRoot)
    error('Vendored CellExplorer root does not exist: %s', cellExplorerRoot);
end

addpath(genpath(cellExplorerRoot));
ensure_cell_explorer_mex(cellExplorerRoot);
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

originalDir = pwd;
cleanupObj = onCleanup(@() cd(originalDir));
cd(basepath);

[~, basename] = fileparts(basepath);
sessionFile = fullfile(basepath, [basename, '.session.mat']);
if exist(sessionFile, 'file')
    loaded = load(sessionFile, 'session');
    session = loaded.session;
else
    xmlFile = fullfile(basepath, [basename, '.xml']);
    if exist(xmlFile, 'file') ~= 2
        error(['Missing session metadata and XML metadata.\n', ...
            'Expected session: %s\n', ...
            'Expected XML: %s\n', ...
            'Run preprocessing first or make sure basename.xml exists in the local output directory.'], ...
            sessionFile, xmlFile);
    end
    session = sessionTemplate(basepath, 'showGUI', false);
    save(sessionFile, 'session');
end

session = configure_session_for_cell_explorer(session, basepath, basename, sortingFolder);
if preferMergePointsDat
    normalize_mergepoints_to_source(basepath, sourceBasepath, basename);
    session.extracellular.fileName = fullfile(basepath, [basename, '.force_mergepoints.dat']);
end
session = apply_anatomical_map_csv(session, fullfile(basepath, 'anatomical_map.csv'));
save(sessionFile, 'session');

[session, ~, statusExit] = gui_session(session);
if exist('statusExit', 'var') && isequal(statusExit, 0)
    error('gui_session was cancelled. CellExplorer processing was not run.');
end

session = configure_session_for_cell_explorer(session, basepath, basename, sortingFolder);
if preferMergePointsDat
    session.extracellular.fileName = fullfile(basepath, [basename, '.force_mergepoints.dat']);
end
session = apply_anatomical_map_csv(session, fullfile(basepath, 'anatomical_map.csv'));
save(sessionFile, 'session');

relativeSortingPath = session.spikeSorting{1}.relativePath;
if prePhy
    spikes = loadSpikes( ...
        'session', session, ...
        'clusteringpath', relativeSortingPath, ...
        'labelsToRead', spikeLabels, ...
        'getWaveformsFromDat', false, ...
        'forceReload', true, ...
        'saveMat', true);
    cell_metrics = ProcessCellMetrics( ...
        'session', session, ...
        'spikes', spikes, ...
        'getWaveformsFromDat', false, ...
        'forceReload', true, ...
        'forceReloadSpikes', true, ...
        'manualAdjustMonoSyn', false, ...
        'excludeMetrics', {'deepSuperficial', 'monoSynaptic_connections'}, ...
        'saveAs', 'unsorted.cell_metrics');
else
    spikes = loadSpikes( ...
        'session', session, ...
        'clusteringpath', relativeSortingPath, ...
        'labelsToRead', spikeLabels, ...
        'forceReload', true, ...
        'saveMat', true);
    cell_metrics = ProcessCellMetrics( ...
        'session', session, ...
        'spikes', spikes, ...
        'getWaveformsFromDat', false, ...
        'manualAdjustMonoSyn', false);
end

save(sessionFile, 'session');
launch_cell_explorer_gui(basepath, basename, cell_metrics);
end

function ensure_cell_explorer_mex(cellExplorerRoot)
mexDir = fullfile(cellExplorerRoot, 'calc_CellMetrics', 'mex');
if isfolder(mexDir)
    addpath(mexDir);
else
    error('CellExplorer MEX folder is missing: %s', mexDir);
end

mexFunctions = {'CCGHeart', 'FindInInterval'};
mexSources = {'CCGHeart.c', 'FindInInterval.c'};
needsCompile = false;
for i = 1:numel(mexFunctions)
    if exist(mexFunctions{i}, 'file') ~= 3
        needsCompile = true;
    end
end
if ~needsCompile
    return
end

originalDir = pwd;
cleanupObj = onCleanup(@() cd(originalDir));
cd(mexDir);
for i = 1:numel(mexFunctions)
    if exist(mexFunctions{i}, 'file') == 3
        continue
    end
    sourceFile = fullfile(mexDir, mexSources{i});
    if exist(sourceFile, 'file') ~= 2
        error('Required CellExplorer MEX source is missing: %s', sourceFile);
    end
    disp(['Compiling CellExplorer MEX helper: ', mexSources{i}]);
    mex('-O', mexSources{i});
end
end

function launch_cell_explorer_gui(basepath, basename, cell_metrics)
if ~isfield(cell_metrics, 'general') || ~isfield(cell_metrics.general, 'cellCount')
    error('CellExplorer metrics are missing general.cellCount; cannot launch manual curation GUI.');
end
if cell_metrics.general.cellCount < 1
    error('CellExplorer metrics contain zero cells; manual curation GUI was not launched.');
end

metricsFile = fullfile(basepath, [basename, '.cell_metrics.cellinfo.mat']);
if exist(metricsFile, 'file') ~= 2
    warning('PreprocessPipeline:CellMetricsFileMissing', ...
        'Expected cell_metrics file was not found before launching CellExplorer: %s', metricsFile);
else
    disp(['Launching CellExplorer manual curation GUI from: ', metricsFile]);
end
disp(['CellExplorer cell count: ', num2str(cell_metrics.general.cellCount)]);
drawnow;
cell_metrics = CellExplorer('basepath', basepath); %#ok<NASGU>
end

function session = configure_session_for_cell_explorer(session, basepath, basename, sortingFolder)
session.general.basePath = basepath;
session.general.name = basename;

relativeSortingPath = relative_path(basepath, sortingFolder);
entry.relativePath = relativeSortingPath;
entry.format = 'Phy';
entry.method = 'KiloSort';
entry.manuallyCurated = 1;
entry.notes = '';
if isfield(session, 'spikeSorting') && ~isempty(session.spikeSorting)
    try
        prior = session.spikeSorting{1};
        fields = fieldnames(prior);
        for i = 1:numel(fields)
            if ~isfield(entry, fields{i})
                entry.(fields{i}) = prior.(fields{i});
            end
        end
    catch
    end
end
session.spikeSorting = {entry};
end

function rel = relative_path(basepath, target)
basepath = char(java.io.File(basepath).getCanonicalPath());
target = char(java.io.File(target).getCanonicalPath());
prefix = [basepath, filesep];
if startsWith(target, prefix)
    rel = target(numel(prefix) + 1:end);
else
    rel = target;
end
end

function normalize_mergepoints_to_source(basepath, sourceBasepath, basename)
mergeFile = fullfile(basepath, [basename, '.MergePoints.events.mat']);
if exist(mergeFile, 'file') ~= 2
    warning('MergePoints file is missing; waveform extraction will not be able to use sub-epoch dat files: %s', mergeFile);
    return
end

loaded = load(mergeFile, 'MergePoints');
if ~isfield(loaded, 'MergePoints')
    warning('MergePoints struct is missing from %s', mergeFile);
    return
end
MergePoints = loaded.MergePoints;
if ~isfield(MergePoints, 'foldernames')
    warning('MergePoints.foldernames is missing from %s', mergeFile);
    return
end

foldernames = MergePoints.foldernames;
if isstring(foldernames)
    foldernames = cellstr(foldernames);
end

changed = false;
for i = 1:numel(foldernames)
    foldername = foldernames{i};
    if isfolder(foldername)
        continue
    end
    localEpoch = fullfile(basepath, foldername);
    sourceEpoch = fullfile(sourceBasepath, foldername);
    if isfolder(localEpoch)
        continue
    end
    if isfolder(sourceEpoch)
        foldernames{i} = sourceEpoch;
        changed = true;
    end
end

if changed
    MergePoints.foldernames = foldernames;
    save(mergeFile, 'MergePoints', '-append');
    disp(['Updated MergePoints foldernames to source basepath: ', sourceBasepath]);
end
end

function session = apply_anatomical_map_csv(session, csvPath)
if exist(csvPath, 'file') ~= 2
    return
end
try
    rows = readcell(csvPath, 'Delimiter', ',');
catch ME
    warning('PreprocessPipeline:AnatomicalMapReadFailed', ...
        'Could not read anatomical_map.csv: %s', ME.message);
    return
end
if isempty(rows)
    return
end
if ~isfield(session, 'extracellular') || ~isfield(session.extracellular, 'electrodeGroups')
    return
end

groups = session.extracellular.electrodeGroups.channels;
brainRegions = struct();
for groupIdx = 1:min(numel(groups), size(rows, 2))
    channels = groups{groupIdx};
    for rowIdx = 1:min(numel(channels), size(rows, 1))
        label = rows{rowIdx, groupIdx};
        if isempty(label)
            continue
        end
        if isstring(label) && ismissing(label)
            continue
        end
        if isnumeric(label)
            label = num2str(label);
        end
        label = strtrim(char(label));
        if isempty(label) || strcmpi(label, 'Unknown')
            continue
        end
        fieldName = matlab.lang.makeValidName(label);
        if ~isfield(brainRegions, fieldName)
            brainRegions.(fieldName).channels = [];
        end
        brainRegions.(fieldName).channels(end + 1) = channels(rowIdx);
    end
end

regions = fieldnames(brainRegions);
for i = 1:numel(regions)
    fieldName = regions{i};
    brainRegions.(fieldName).channels = unique(brainRegions.(fieldName).channels);
end
if ~isempty(regions)
    session.brainRegions = brainRegions;
end
end
