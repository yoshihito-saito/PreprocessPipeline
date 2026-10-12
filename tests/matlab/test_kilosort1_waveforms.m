function tests = test_kilosort1_waveforms
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
testCase.TestData.oldPath = path;
addpath(genpath(fullfile(root, 'sorter', 'KiloSort1')));
end

function teardownOnce(testCase)
path(testCase.TestData.oldPath);
end

function testHistoricalWaveformsMatchLegacyProjection(testCase)
rng(1);
data = single(randn(200, 5));
rows = [50; 100];
cols = [2; 4];
for nt0 = [33 61]
    wPCA = single(randn(nt0, 3));
    expected = zeros(5, 2, 3, 'single');
    for i = 1:numel(rows)
        clip = data(rows(i) + (-20:nt0-21), :);
        clip(:, abs((1:5) - cols(i)) >= 2) = 0;
        projection = wPCA' * clip;
        expected(:,i,:) = reshape(projection', 5, 1, 3);
    end
    verifyEqual(testCase, get_PCproj(data, rows, cols, wPCA, 2), expected, 'AbsTol', single(1e-5));
end
end

function testNoInitializationSpikesHasActionableError(testCase)
verifyError(testCase, @() optimizePeaks(struct('nt0', 33), zeros(0, 9)), ...
    'KiloSort:NoInitializationSpikes');
end

function testEmptyFinalMatchingPassHasActionableError(testCase)
rez = makeRez;
[W, ~] = decompose_dWU(rez.ops, rez.dWU, 1, rez.ops.kcoords);
W0 = zeros(rez.ops.NT, 2, 'single');
W0(1:rez.ops.nt0,:) = W;
rez.fW = conj(fft(W0));
verifyError(testCase, @() fullMPMU(rez, zeros(128,3,1,'int16')), ...
    'KiloSort:NoSpikesFound');
end

function testNoActualTrainingDetectionsHasActionableError(testCase)
rez = makeRez;
file = [tempname '.mat'];
cleanup = onCleanup(@() delete(file));
WUinit = rez.dWU;
save(file, 'WUinit');
rez.ops.initialize = 'no';
rez.ops.initFilePath = file;
rez.ops.saveInitTemps = false;
rez.ops.nfullpasses = 2;
rez.ops.nannealpasses = 1;
rez.ops.shuffle_clusters = false;
rez.ops.showfigures = false;
rez.temp = struct('Nbatch', 2, 'Nbatch_buff', 2);
% The first pass must warn and continue; only the completed fit fails.
oldWarning = warning('query', 'KiloSort:NoTemplateMatchesYet');
warnCleanup = onCleanup(@() warning(oldWarning.state, 'KiloSort:NoTemplateMatchesYet'));
warning('off', 'KiloSort:NoTemplateMatchesYet');
try
    fitTemplates(rez, zeros(128,3,2,'int16'), []);
    verifyFail(testCase, 'Expected the completed empty fitting run to fail.');
catch ME
    verifyEqual(testCase, ME.identifier, 'KiloSort:NoTemplateMatches');
    verifySubstring(testCase, ME.message, 'finished without detecting');
end
end

function rez = makeRez
ops = struct('Nfilt', 2, 'Nrank', 1, 'Nchan', 3, 'NT', 128, ...
    'Th', [100 100 100], 'lam', [5 5 5], 'momentum', [.05 .00125], ...
    'epu', Inf, 'nt0', 9, 'nt0min', 3, 'ntbuff', 4, 'maxFR', 10, ...
    'scaleproc', 200, 'GPU', false, 'parfor', false, 'verbose', false, ...
    'nNeigh', [], 'nNeighPC', [], 'kcoords', ones(3,1));
rng(1);
rez.ops = ops;
rez.dWU = single(randn(9,3,2));
rez.temp = struct('Nbatch', 1, 'Nbatch_buff', 1);
end
