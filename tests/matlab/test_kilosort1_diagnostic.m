function tests = test_kilosort1_diagnostic
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
testCase.TestData.root = root;
testCase.TestData.ks = fullfile(root,'sorter','KiloSort1');
testCase.TestData.oldPath = path;
addpath(fullfile(root,'tools'));
end

function teardownOnce(testCase)
path(testCase.TestData.oldPath);
end

function setup(testCase)
folder = tempname;
mkdir(folder);
testCase.TestData.folder = folder;
ops = struct('NT',8192,'Nchan',4,'Nfilt',32,'Nrank',3,'nt0',33, ...
    'Th',[6 10 10],'lam',[12 40 40],'momentum',[.05 .00125], ...
    'epu',Inf,'maxFR',40000,'scaleproc',200,'loc_range',[3 1], ...
    'long_range',[30 6],'spkTh',-4,'maskMaxChannels',8,'criterionNoiseChannels',.2);
save(fullfile(folder,'ops.mat'),'ops');
connected = true(4,1);
kcoords = ones(4,1);
save(fullfile(folder,'chanMap.mat'),'connected','kcoords');
% Distinct temporal ramps and channel offsets catch an interleaved read.
data = int16((1:8192)' + 1000*(1:4));
fid = fopen(fullfile(folder,'temp_wh.dat'),'w');
fwrite(fid,data,'int16');
fwrite(fid,data+int16(3000),'int16');
fclose(fid);
testCase.TestData.data = data;
end

function teardown(testCase)
rmdir(testCase.TestData.folder,'s');
end

function testKnownSignalAndColumnMajorBatch(testCase)
folder = testCase.TestData.folder;
file = fullfile(folder,'report.json');
before = readBytes(fullfile(folder,'temp_wh.dat'));
report = diagnose_kilosort1(folder,testCase.TestData.ks,'UseGPU',false,'OutputFile',file);
verifyTrue(testCase, report.synthetic.cpu_known_signal_passed);
verifyEqual(testCase, report.synthetic.cpu_training, 2);
verifyEqual(testCase, report.synthetic.cpu_final, 2);
verifyEqual(testCase, report.batch.first_row, double(testCase.TestData.data(1,:)));
verifyEqual(testCase, report.batch.size, [8192 4]);
verifyEqual(testCase, readBytes(fullfile(folder,'temp_wh.dat')), before);
verifyTrue(testCase, isfile(file));
loaded = jsondecode(fileread(file));
verifyEqual(testCase, loaded.batch.layout, 'MATLAB column-major NT-by-Nchan blocks');
end

function testSecondBatchSeek(testCase)
report = diagnose_kilosort1(testCase.TestData.folder,testCase.TestData.ks, ...
    'UseGPU',false,'BatchIndex',1,'OutputFile',fullfile(testCase.TestData.folder,'second.json'));
verifyEqual(testCase, report.batch.first_row, double(testCase.TestData.data(1,:)+int16(3000)));
end

function testOutsideBatchIsReported(testCase)
report = diagnose_kilosort1(testCase.TestData.folder,testCase.TestData.ks, ...
    'UseGPU',false,'BatchIndex',2,'OutputFile',fullfile(testCase.TestData.folder,'outside.json'));
verifyEqual(testCase, report.recorded_error.identifier, 'Diagnostic:ShortBatch');
end

function testExistingFileIsPreserved(testCase)
file = fullfile(testCase.TestData.folder,'ops.mat');
before = readBytes(file);
verifyError(testCase, @() diagnose_kilosort1(testCase.TestData.folder,testCase.TestData.ks, ...
    'UseGPU',false,'OutputFile',file), 'Diagnostic:ExistingReport');
verifyEqual(testCase, readBytes(file), before);
end

function bytes = readBytes(file)
fid = fopen(file,'r');
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid,Inf,'*uint8');
end
