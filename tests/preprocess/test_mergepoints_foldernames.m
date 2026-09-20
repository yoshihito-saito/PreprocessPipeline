function tests = test_mergepoints_foldernames
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
testCase.TestData.oldPath = path;
addpath(fullfile(root, 'external', 'matlab'));
addpath(fullfile(root, 'external', 'CellExplorer', 'calc_CellMetrics'));
addpath(fullfile(root, 'external', 'CellExplorer'));
end

function teardownOnce(testCase)
path(testCase.TestData.oldPath);
end

function setup(testCase)
root = tempname;
mkdir(root);
testCase.TestData.root = root;
testCase.TestData.local = fullfile(root, 'local');
testCase.TestData.source = fullfile(root, 'source');
mkdir(testCase.TestData.local);
mkdir(testCase.TestData.source);
testCase.TestData.file = fullfile(testCase.TestData.local, 'session.MergePoints.events.mat');
end

function teardown(testCase)
rmdir(testCase.TestData.root, 's');
end

function testSourceOnlyAndLocalPreference(testCase)
names = {'epoch1', 'epoch2'};
original = writeMergePoints(testCase, names);
for i = 1:numel(names)
    mkdir(fullfile(testCase.TestData.source, names{i}));
end
normalize(testCase);
loaded = load(testCase.TestData.file);
verifyEqual(testCase, loaded.MergePoints.foldernames, names);
verifyEqual(testCase, rmfield(loaded.MergePoints, 'folderpaths'), original);
verifyEqual(testCase, loaded.unrelated, 42);
for i = 1:numel(names)
    source = fullfile(testCase.TestData.source, names{i});
    verifyEqual(testCase, loaded.MergePoints.folderpaths{i}, source);
    verifyEqual(testCase, resolveMergePointEpochDir(testCase.TestData.local, names{i}, source), source);
end
local = fullfile(testCase.TestData.local, names{1});
mkdir(local);
verifyEqual(testCase, resolveMergePointEpochDir(testCase.TestData.local, names{1}, loaded.MergePoints.folderpaths{1}), local);
normalize(testCase);
verifyEqual(testCase, load(testCase.TestData.file), loaded);
end

function testLegacyAbsolutePaths(testCase)
source = fullfile(testCase.TestData.source, 'epoch1');
mkdir(source);
writeMergePoints(testCase, {source});
% Old files still resolve before migration.
verifyEqual(testCase, resolveMergePointEpochDir(testCase.TestData.local, source), source);
normalize(testCase);
loaded = load(testCase.TestData.file);
verifyEqual(testCase, loaded.MergePoints.foldernames, {'epoch1'});
verifyEqual(testCase, loaded.MergePoints.folderpaths, {source});
verifyEqual(testCase, resolveMergePointEpochDir(testCase.TestData.local, 'epoch1', source), source);
end

function testOfflineAndCrossPlatformPaths(testCase)
names = {'\\132.236.112.15\ayadataB1\data\hpc_ctx_project\HP19\hp19_day3_20260917\hp19_probe_260917_095747'; ...
    'C:\data\epoch2\'; '/mnt/data/epoch3/'};
original = writeMergePoints(testCase, names);
normalize(testCase);
loaded = load(testCase.TestData.file);
verifyEqual(testCase, loaded.MergePoints.foldernames, {'hp19_probe_260917_095747'; 'epoch2'; 'epoch3'});
verifyEqual(testCase, loaded.MergePoints.folderpaths, names);
verifyEqual(testCase, loaded.MergePoints.timestamps_samples, original.timestamps_samples);
end

function testStringLabelsAndExistingSourcePaths(testCase)
writeMergePoints(testCase, ["epoch1", "epoch2"]);
loaded = load(testCase.TestData.file);
MergePoints = loaded.MergePoints;
MergePoints.folderpaths = {fullfile(testCase.TestData.root, 'offline1'), fullfile(testCase.TestData.root, 'offline2')};
save(testCase.TestData.file, 'MergePoints', '-append');
normalize(testCase);
loaded = load(testCase.TestData.file);
verifyEqual(testCase, loaded.MergePoints.foldernames, {'epoch1', 'epoch2'});
verifyEqual(testCase, loaded.MergePoints.folderpaths, MergePoints.folderpaths);
end

function testLegacyRelativeLookup(testCase)
local = fullfile(testCase.TestData.local, 'epoch1');
mkdir(local);
verifyEqual(testCase, resolveMergePointEpochDir(testCase.TestData.local, 'epoch1'), local);
verifyEqual(testCase, resolveMergePointEpochDir(testCase.TestData.local, 'missing'), fullfile(testCase.TestData.local, 'missing'));
end

function testWaveformsUnchangedForIntanAndOpenEphys(testCase)
for openEphys = [false, true]
    source = fullfile(testCase.TestData.source, 'epoch1');
    mkdir(source);
    if openEphys
        binaryDir = fullfile(source, 'Record Node 101', 'experiment1', 'recording1', 'continuous', 'stream');
        mkdir(binaryDir);
        binaryFile = fullfile(binaryDir, 'continuous.dat');
    else
        binaryFile = fullfile(source, 'amplifier.dat');
    end
    samples = int16((1000 * exp(-(1:4)' / 2)) * sin((1:2000) * 0.2));
    fid = fopen(binaryFile, 'w');
    fwrite(fid, samples, 'int16');
    fclose(fid);
    MergePoints = writeMergePoints(testCase, {source});
    MergePoints.timestamps_samples = [0, 2000];
    MergePoints.timestamps = [0, 0.1];
    MergePoints.firstlasttimpoints_samples = [0, 2000];
    save(testCase.TestData.file, 'MergePoints', '-append');
    session.general.basePath = testCase.TestData.local;
    session.general.name = 'session';
    session.extracellular = struct('leastSignificantBit', 1, 'nChannels', 4, ...
        'sr', 20000, 'nElectrodeGroups', 1, 'precision', 'int16');
    session.extracellular.electrodeGroups.channels = {1:4};
    spikes.times = {[500, 1000, 1500]' / 20000};
    args = {'showWaveforms', false, 'saveMat', false, 'getBadChannelsFromDat', false};
    before = getWaveformsFromDat(spikes, session, args{:});
    normalize(testCase);
    after = getWaveformsFromDat(spikes, session, args{:});
    verifyEqual(testCase, after.rawWaveform_all, before.rawWaveform_all);
    verifyEqual(testCase, after.filtWaveform_all, before.filtWaveform_all);
    verifyEqual(testCase, after.maxWaveformCh1, before.maxWaveformCh1);
    verifyEqual(testCase, after.processinginfo, before.processinginfo);
    delete(binaryFile);
end
end

function MergePoints = writeMergePoints(testCase, names)
n = numel(names);
MergePoints.foldernames = names;
MergePoints.timestamps_samples = [(0:n-1)' * 100, (1:n)' * 100];
MergePoints.timestamps = MergePoints.timestamps_samples / 20000;
MergePoints.firstlasttimpoints_samples = [zeros(n, 1), repmat(100, n, 1)];
MergePoints.detectorinfo.detectorname = 'preprocessSession.py';
unrelated = 42;
save(testCase.TestData.file, 'MergePoints', 'unrelated');
end

function normalize(testCase)
normalize_mergepoints_to_source(testCase.TestData.local, testCase.TestData.source, 'session');
end
