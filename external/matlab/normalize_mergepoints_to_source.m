function normalize_mergepoints_to_source(basepath, sourceBasepath, basename)
% Keep epoch labels portable; store binary source locations separately.
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
if isstring(foldernames) || ischar(foldernames)
    foldernames = cellstr(foldernames);
end

folderpaths = cell(size(foldernames));
if isfield(MergePoints, 'folderpaths') && numel(MergePoints.folderpaths) == numel(foldernames)
    folderpaths = MergePoints.folderpaths;
    if isstring(folderpaths) || ischar(folderpaths)
        folderpaths = cellstr(folderpaths);
    end
end
for i = 1:numel(foldernames)
    originalName = char(foldernames{i});
    parts = regexp(strrep(originalName, '\', '/'), '[^/]+', 'match');
    if isempty(parts)
        continue
    end
    foldername = parts{end};
    foldernames{i} = foldername;
    sourceEpoch = fullfile(sourceBasepath, foldername);
    % Preserve legacy paths even if their source is temporarily offline.
    if ~strcmp(originalName, foldername)
        folderpaths{i} = originalName;
    elseif isfolder(sourceEpoch)
        folderpaths{i} = sourceEpoch;
    end
end

if ~iscell(MergePoints.foldernames) || ~isequal(MergePoints.foldernames, foldernames) || ...
        ~isfield(MergePoints, 'folderpaths') || ~isequal(MergePoints.folderpaths, folderpaths)
    MergePoints.foldernames = foldernames;
    MergePoints.folderpaths = folderpaths;
    save(mergeFile, 'MergePoints', '-append');
    disp(['Updated MergePoints source paths; preserved folder labels: ', mergeFile]);
end
end
