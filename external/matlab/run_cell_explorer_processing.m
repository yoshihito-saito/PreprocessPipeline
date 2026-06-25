function run_cell_explorer_processing(basepath, sourceBasepath, sortingFolder, cellExplorerRoot, varargin)
% Run the PreprocessPipeline CellExplorer post-Phy workflow.
%
% This wrapper intentionally uses the vendored CellExplorer path passed by
% the Python GUI. It keeps the selected Phy folder explicit and avoids
% relying on dir('Kilosort*') discovery.

p = inputParser;
addRequired(p, 'basepath', @ischar);
addRequired(p, 'sourceBasepath', @ischar);
addRequired(p, 'sortingFolder', @(x) ischar(x) || isstring(x) || iscell(x));
addRequired(p, 'cellExplorerRoot', @ischar);
addParameter(p, 'preferMergePointsDat', true, @islogical);
addParameter(p, 'prePhy', false, @islogical);
addParameter(p, 'spikeLabels', {'good'}, @iscell);
parse(p, basepath, sourceBasepath, sortingFolder, cellExplorerRoot, varargin{:});

basepath = char(p.Results.basepath);
sourceBasepath = char(p.Results.sourceBasepath);
sortingFolders = normalize_sorting_folders(p.Results.sortingFolder);
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
for i = 1:numel(sortingFolders)
    if ~isfolder(sortingFolders{i})
        error('Sorting folder does not exist: %s', sortingFolders{i});
    end
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

session = configure_session_for_cell_explorer(session, basepath, basename, sortingFolders);
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

session = configure_session_for_cell_explorer(session, basepath, basename, sortingFolders);
if preferMergePointsDat
    session.extracellular.fileName = fullfile(basepath, [basename, '.force_mergepoints.dat']);
end
session = apply_anatomical_map_csv(session, fullfile(basepath, 'anatomical_map.csv'));
save(sessionFile, 'session');

if prePhy
    cellMetricsSaveAs = 'unsorted.cell_metrics';
    spikes = load_spikes_from_sorting_folders( ...
        'session', session, ...
        'sortingFolders', sortingFolders, ...
        'basepath', basepath, ...
        'labelsToRead', spikeLabels, ...
        'showWaveforms', true, ...
        'getWaveformsFromDat', true, ...
        'forceReload', true, ...
        'saveMat', false);
    save(fullfile(basepath, [basename, '.unsorted.spikes.cellinfo.mat']), 'spikes');
    keep_or_remove_existing_mono_res(basepath, basename, cellMetricsSaveAs, numel(spikes.times));
    cell_metrics = ProcessCellMetrics( ...
        'session', session, ...
        'spikes', spikes, ...
        'getWaveformsFromDat', false, ...
        'forceReload', true, ...
        'forceReloadSpikes', false, ...
        'manualAdjustMonoSyn', false, ...
        'excludeMetrics', {'deepSuperficial'}, ...
        'saveAs', cellMetricsSaveAs);
else
    cellMetricsSaveAs = 'cell_metrics';
    spikes = load_spikes_from_sorting_folders( ...
        'session', session, ...
        'sortingFolders', sortingFolders, ...
        'basepath', basepath, ...
        'labelsToRead', spikeLabels, ...
        'showWaveforms', true, ...
        'getWaveformsFromDat', true, ...
        'forceReload', true, ...
        'saveMat', false);
    save(fullfile(basepath, [basename, '.spikes.cellinfo.mat']), 'spikes');
    keep_or_remove_existing_mono_res(basepath, basename, cellMetricsSaveAs, numel(spikes.times));
    cell_metrics = ProcessCellMetrics( ...
        'session', session, ...
        'spikes', spikes, ...
        'getWaveformsFromDat', false, ...
        'forceReloadSpikes', false, ...
        'manualAdjustMonoSyn', false, ...
        'excludeMetrics', {'deepSuperficial'});
end

cell_metrics = ensure_cell_explorer_gui_fields(cell_metrics);
save_cell_metrics_for_gui(basepath, basename, cellMetricsSaveAs, cell_metrics);
save(sessionFile, 'session');
launch_cell_explorer_gui(basepath, basename, cell_metrics);
end

function sortingFolders = normalize_sorting_folders(value)
if ischar(value) || isstring(value)
    sortingFolders = cellstr(value);
elseif iscell(value)
    sortingFolders = value;
else
    error('sortingFolder must be a char, string, or cell array.');
end
sortingFolders = sortingFolders(:)';
sortingFolders = sortingFolders(~cellfun(@isempty, sortingFolders));
for i = 1:numel(sortingFolders)
    sortingFolders{i} = char(sortingFolders{i});
end
if isempty(sortingFolders)
    error('At least one sorting folder is required.');
end
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

function keep_or_remove_existing_mono_res(basepath, basename, saveAs, cellCount)
monoFile = fullfile(basepath, [basename, '.mono_res', erase(saveAs, 'cell_metrics'), '.cellinfo.mat']);
if exist(monoFile, 'file') ~= 2
    return
end
try
    loaded = load(monoFile, 'mono_res');
    if isfield(loaded, 'mono_res') && mono_res_is_compatible(loaded.mono_res, cellCount)
        disp(['Using existing compatible MonoSynaptic file: ', monoFile]);
        return
    end
catch ME
    warning('PreprocessPipeline:MonoResReadFailed', ...
        'Could not validate existing MonoSynaptic file %s: %s', monoFile, ME.message);
end
delete(monoFile);
disp(['Removed incompatible MonoSynaptic file before recomputing: ', monoFile]);
end

function ok = mono_res_is_compatible(mono_res, cellCount)
ok = true;
connectionFields = {'sig_con', 'sig_con_excitatory', 'sig_con_inhibitory'};
for i = 1:numel(connectionFields)
    fieldName = connectionFields{i};
    if isfield(mono_res, fieldName) && ~isempty(mono_res.(fieldName))
        connections = mono_res.(fieldName);
        if size(connections, 2) < 2 || any(connections(:, 1:2) < 1, 'all') || any(connections(:, 1:2) > cellCount, 'all')
            ok = false;
            return
        end
    end
end
if isfield(mono_res, 'ccgR') && ~isempty(mono_res.ccgR)
    ccgSize = size(mono_res.ccgR);
    if numel(ccgSize) < 3 || ccgSize(2) < cellCount || ccgSize(3) < cellCount
        ok = false;
        return
    end
end
end

function cell_metrics = ensure_cell_explorer_gui_fields(cell_metrics)
if ~isfield(cell_metrics, 'general') || ~isfield(cell_metrics.general, 'cellCount')
    return
end
cellCount = cell_metrics.general.cellCount;
if ~isfield(cell_metrics, 'deepSuperficial') || numel(cell_metrics.deepSuperficial) ~= cellCount
    cell_metrics.deepSuperficial = repmat({'Unknown'}, 1, cellCount);
end
if ~isfield(cell_metrics, 'deepSuperficialDistance') || numel(cell_metrics.deepSuperficialDistance) ~= cellCount
    cell_metrics.deepSuperficialDistance = nan(1, cellCount);
end
if ~isfield(cell_metrics, 'deepSuperficial_num') || numel(cell_metrics.deepSuperficial_num) ~= cellCount
    cell_metrics.deepSuperficial_num = ones(1, cellCount);
end
end

function save_cell_metrics_for_gui(basepath, basename, saveAs, cell_metrics)
metricsFile = fullfile(basepath, [basename, '.', saveAs, '.cellinfo.mat']);
if exist(metricsFile, 'file') == 2
    save(metricsFile, 'cell_metrics', '-append');
else
    save(metricsFile, 'cell_metrics');
end
end

function spikes = load_spikes_from_sorting_folders(varargin)
p = inputParser;
addParameter(p, 'session', struct(), @isstruct);
addParameter(p, 'sortingFolders', {}, @iscell);
addParameter(p, 'basepath', '', @ischar);
addParameter(p, 'labelsToRead', {'good'}, @iscell);
addParameter(p, 'getWaveformsFromDat', [], @(x) isempty(x) || islogical(x));
addParameter(p, 'showWaveforms', false, @islogical);
addParameter(p, 'forceReload', true, @islogical);
addParameter(p, 'saveMat', false, @islogical);
parse(p, varargin{:});

session = p.Results.session;
sortingFolders = p.Results.sortingFolders;
basepath = p.Results.basepath;
labelsToRead = p.Results.labelsToRead;
getWaveformsFromDat = p.Results.getWaveformsFromDat;
showWaveforms = p.Results.showWaveforms;
forceReload = p.Results.forceReload;
saveMat = p.Results.saveMat;

for i = 1:numel(sortingFolders)
    clusteringPath = relative_path(basepath, sortingFolders{i});
    loadArgs = {
        'session', session, ...
        'clusteringpath', clusteringPath, ...
        'labelsToRead', labelsToRead, ...
        'forceReload', forceReload, ...
        'showWaveforms', showWaveforms, ...
        'saveMat', saveMat ...
    };
    if ~isempty(getWaveformsFromDat)
        loadArgs = [loadArgs, {'getWaveformsFromDat', getWaveformsFromDat}]; %#ok<AGROW>
    end
    tempSpikes = loadSpikes(loadArgs{:});
    if i == 1
        spikes = tempSpikes;
    else
        spikes = append_spikes_struct(spikes, tempSpikes);
    end
end
end

function spikes = append_spikes_struct(spikes, nextSpikes)
fieldsBase = fieldnames(spikes);
fieldsNext = fieldnames(nextSpikes);
prevUID = length(spikes.UID);
for j = 1:numel(fieldsBase)
    currentField = fieldsBase{j};
    if ~any(strcmp(fieldsNext, currentField))
        spikes = rmfield(spikes, currentField);
        continue
    end
    if strcmp(currentField, 'spindices')
        nextUnitIds = nextSpikes.spindices(:, 2) + prevUID;
        spikes.spindices = [spikes.spindices; [nextSpikes.spindices(:, 1), nextUnitIds]];
    elseif strcmp(currentField, 'UID')
        spikes.UID = [spikes.UID, nextSpikes.UID + prevUID];
    elseif strcmp(currentField, 'basename')
        if ~isequal(spikes.basename, nextSpikes.basename)
            error('Incompatible basenames across Kilosort folders.');
        end
    elseif strcmp(currentField, 'numcells')
        spikes.numcells = spikes.numcells + nextSpikes.numcells;
    elseif strcmp(currentField, 'sr')
        if ~isequal(spikes.sr, nextSpikes.sr)
            error('Incompatible sampling rates across Kilosort folders.');
        end
    elseif strcmp(currentField, 'processinginfo')
        disp('Processing info assumed to be the same across Kilosort folders.');
    elseif iscell(spikes.(currentField))
        spikes.(currentField) = cat(2, spikes.(currentField), nextSpikes.(currentField));
    else
        spikes.(currentField) = [spikes.(currentField), nextSpikes.(currentField)];
    end
end
end

function session = configure_session_for_cell_explorer(session, basepath, basename, sortingFolders)
session.general.basePath = basepath;
session.general.name = basename;

entries = cell(1, numel(sortingFolders));
for sortingIdx = 1:numel(sortingFolders)
    relativeSortingPath = relative_path(basepath, sortingFolders{sortingIdx});
    entry.relativePath = relativeSortingPath;
    entry.format = 'Phy';
    entry.method = 'KiloSort';
    entry.manuallyCurated = 1;
    entry.notes = '';
    entries{sortingIdx} = entry;
end
if isfield(session, 'spikeSorting') && ~isempty(session.spikeSorting)
    try
        prior = session.spikeSorting{1};
        fields = fieldnames(prior);
        for sortingIdx = 1:numel(entries)
            entry = entries{sortingIdx};
            for i = 1:numel(fields)
                if ~isfield(entry, fields{i})
                    entry.(fields{i}) = prior.(fields{i});
                end
            end
            entries{sortingIdx} = entry;
        end
    catch
    end
end
session.spikeSorting = entries;
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
