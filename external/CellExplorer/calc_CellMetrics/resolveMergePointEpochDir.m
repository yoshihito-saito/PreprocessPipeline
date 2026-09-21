function epochDir = resolveMergePointEpochDir(basepath, foldername, sourcePath)
% Resolve portable labels, with compatibility for legacy path-valued labels.
if nargin < 3
    sourcePath = '';
end
foldername = char(foldername);
parts = regexp(strrep(foldername, '\', '/'), '[^/]+', 'match');
localFolderName = foldername;
if ~isempty(parts)
    localFolderName = parts{end};
end
candidates = {fullfile(basepath, localFolderName), ...
    fullfile(basepath, foldername), char(sourcePath), foldername};
for i = 1:numel(candidates)
    if isfolder(candidates{i})
        epochDir = candidates{i};
        return
    end
end
epochDir = fullfile(basepath, localFolderName);
end
