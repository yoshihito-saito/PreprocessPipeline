function report = diagnose_kilosort1(sorterOutput, kilosortRoot, varargin)
% DIAGNOSE_KILOSORT1 Check matchers without running a pipeline or full sort.
% sorterOutput contains ops.mat, chanMap.mat and temp_wh.dat from a failed run.
% Writes only a new JSON report (default: a temporary file). Preserves source
% files, recordings and parameters; does not rebuild MEX files, reset the GPU,
% change its memory policy or submit/cancel jobs.
%
% diagnose_kilosort1(out, ks, 'OutputFile', '/tmp/ks-diagnostic.json')
% Optional: 'UseGPU', false; 'BatchIndex', 0; 'TemplateLimit', 64;
% 'LegacyRoot', path to a known-working KiloSort1 source tree;
% 'MexDirectory', a private directory containing all three compiled MEX files.
% The recorded test uses one disk-stored batch and a reduced template bank;
% it does not validate full-size allocations, training or preprocessing.
p = inputParser;
addParameter(p, 'OutputFile', [tempname '.json']);
addParameter(p, 'UseGPU', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'BatchIndex', 0, @(x) isscalar(x) && x >= 0 && x == floor(x));
addParameter(p, 'TemplateLimit', 64, @(x) isscalar(x) && isfinite(x) && x >= 32);
addParameter(p, 'LegacyRoot', '');
addParameter(p, 'MexDirectory', '');
parse(p, varargin{:});
options = p.Results;
if exist(options.OutputFile, 'file')
    error('Diagnostic:ExistingReport', 'Choose a new report path: %s', options.OutputFile);
end
savedPath = path;
pathCleanup = onCleanup(@() path(savedPath));
addpath(genpath(kilosortRoot));
loaded = load(fullfile(sorterOutput, 'ops.mat'), 'ops');
ops = loaded.ops;
report = struct('matlab', version, 'platform', computer, ...
    'sorter_output', sorterOutput, 'kilosort_root', kilosortRoot, ...
    'recorded_test_scope', 'One saved batch; no preprocessing or iterative training', ...
    'run_ops', ops, 'options', options);
report.run_ops_note = 'JSON encodes Inf/NaN as null; ops.mat remains authoritative.';
gpuFns = struct;
if options.UseGPU
    try
        report.forward_compatibility = parallel.gpu.enableCUDAForwardCompatibility;
        device = gpuDevice; % no reset and no device change
        report.gpu = struct('name', device.Name, ...
            'compute_capability', device.ComputeCapability, ...
            'available_memory', device.AvailableMemory, 'total_memory', device.TotalMemory);
        names = {'mexMPregMU', 'mexMPmuFEAT', 'mexWtW2'};
        if ~isempty(options.MexDirectory)
            if ~isfolder(options.MexDirectory)
                error('Diagnostic:MexDirectory','Missing MEX directory: %s',options.MexDirectory);
            end
            addpath(options.MexDirectory,'-begin');
        end
        for i = 1:numel(names)
            report.mex.(names{i}) = which(names{i}, '-all');
            if exist(names{i}, 'file') ~= 3
                error('Diagnostic:MissingMex', 'Compiled %s is missing; inspect MATLAB path.', names{i});
            end
            if ~isempty(options.MexDirectory)
                expected = canonicalPath(fullfile(options.MexDirectory,[names{i} '.' mexext]));
                selected = canonicalPath(which(names{i}));
                if ~strcmp(expected,selected)
                    error('Diagnostic:WrongMex','Expected %s, but MATLAB selected %s.',expected,selected);
                end
            end
        end
        % Bind the current server binaries before any legacy-source comparison.
        gpuFns.reg = @mexMPregMU;
        gpuFns.final = @mexMPmuFEAT;
        gpuFns.correlation = @mexWtW2;
    catch ME
        report.gpu_error = exceptionInfo(ME);
    end
else
    report.gpu_note = 'GPU explicitly disabled; CPU validation only.';
end
report.synthetic = syntheticCheck(ops, gpuFns);
try
    nT = double(ops.NT);
    nChan = double(ops.Nchan);
    file = fullfile(sorterOutput, 'temp_wh.dat');
    info = dir(file);
    if isempty(info) || mod(info.bytes, 2*nT*nChan) ~= 0
        error('Diagnostic:BatchLayout', 'temp_wh.dat is missing or has incomplete NT-by-Nchan blocks.');
    end
    fid = fopen(file, 'r', 'ieee-le');
    if fid < 0, error('Diagnostic:OpenData', 'Cannot open %s', file); end
    fileCleanup = onCleanup(@() fclose(fid));
    if fseek(fid, options.BatchIndex * 2*nT*nChan, 'bof') ~= 0
        error('Diagnostic:SeekData', 'Cannot seek to requested disk batch.');
    end
    % Exact layout used by fitTemplates/fullMPMU: NOT interleaved Nchan-by-NT.
    [stored, count] = fread(fid, [nT nChan], '*int16');
    if count ~= nT*nChan
        error('Diagnostic:ShortBatch', 'Requested disk batch is outside temp_wh.dat.');
    end
    data = single(stored) / ops.scaleproc;
    report.batch = struct('index_zero_based', options.BatchIndex, ...
        'size', size(stored), 'layout', 'MATLAB column-major NT-by-Nchan blocks', ...
        'first_row', double(stored(1,:)), 'min', double(min(stored(:))), ...
        'max', double(max(stored(:))), 'scaled_channel_std', double(std(data,0,1)), ...
        'file_bytes', info.bytes);
    clear fileCleanup stored;
    report.current = recordedCheck(data, ops, sorterOutput, kilosortRoot, options.TemplateLimit, gpuFns);
    if ~isempty(options.LegacyRoot)
        report.legacy = recordedCheck(data, ops, sorterOutput, options.LegacyRoot, options.TemplateLimit, gpuFns);
        report.legacy_note = ['Same saved data, effective options, random seed and current server MEX binaries; ' ...
            'compares initialization/matching only, not raw preprocessing or full-session sampling.'];
    end
catch ME
    report.recorded_error = exceptionInfo(ME);
end
fid = fopen(options.OutputFile, 'w');
if fid < 0, error('Diagnostic:WriteReport', 'Cannot write %s', options.OutputFile); end
outputCleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(report, 'PrettyPrint', true));
fprintf('KiloSort1 diagnostic report: %s\n', options.OutputFile);
disp(report.synthetic);
if isfield(report, 'gpu_error'), disp(report.gpu_error); end
if isfield(report, 'recorded_error'), disp(report.recorded_error); end
if isfield(report, 'current'), disp(report.current); end
end

function result = syntheticCheck(runOps, gpuFns)
% Two isolated, known template events test the backend, independent of PCA.
ops = runOps;
ops.NT = 8192;
ops.Nchan = 4;
ops.Nfilt = 32;
ops.Nrank = 3;
wave = single(-exp(-((1:ops.nt0)' - ceil(ops.nt0/2)).^2 / 4));
wave = wave / norm(wave);
W = zeros(ops.nt0,32,3,'single');
U = zeros(4,32,3,'single');
W(:,:,1) = repmat(wave,1,32);
U(1,:,1) = 1;
mu = single(10*ones(32,1));
dWU = zeros(ops.nt0,4,32,'single');
dWU(:,1,:) = reshape(10*W(:,:,1), ops.nt0,1,32);
Wdiff = cat(1,W,zeros(2,32,3,'single')) - cat(1,zeros(2,32,3,'single'),W);
nu = reshape(sum(sum(Wdiff.^2,1),3),[],1);
data = zeros(8192,4,'single');
for start = [1024 4096]
    data(start + (1:ops.nt0),1) = 40*wave;
end
result = matchChecks(data, ops, W,U,mu,true(32),nu,dWU,gpuFns);
result.expected_events = 2;
result.cpu_known_signal_passed = result.cpu_training == 2 && result.cpu_final == 2;
if isfield(result, 'gpu_training') && isfield(result, 'gpu_final')
    result.gpu_known_signal_passed = result.gpu_training == 2 && result.gpu_final == 2 ...
        && isfield(result,'gpu_correlation_max_error') && result.gpu_correlation_max_error < 1e-3;
end
end

function result = recordedCheck(data, runOps, sorterOutput, root, limit, gpuFns)
oldPath = path;
cleanup = onCleanup(@() path(oldPath));
addpath(genpath(root));
clear get_PCproj optimizePeaks isolated_peaks decompose_dWU get_svds zeroOutKcoords mexMPregMUcpu cpuMPmuFEAT;
ops = runOps;
ops.GPU = false;
ops.parfor = false;
ops.Nfilt = min(runOps.Nfilt, 32*floor(limit/32));
result = struct('root', root, 'original_templates', runOps.Nfilt, ...
    'test_templates', ops.Nfilt, 'template_limit_is_a_diagnostic_reduction', ops.Nfilt ~= runOps.Nfilt);
try
    dd = load(fullfile(root,'configFiles','PCspikes2.mat'));
    ops.wPCA = dd.Wi(round(linspace(1,size(dd.Wi,1),ops.nt0)),1:3);
    [row,col] = isolated_peaks(data,ops.loc_range,ops.long_range,ops.spkTh);
    result.isolated_peaks = numel(row);
    sourceNames = {'get_PCproj','optimizePeaks','decompose_dWU','mexMPregMUcpu','cpuMPmuFEAT'};
    for i = 1:numel(sourceNames)
        f = which(sourceNames{i});
        result.sources.(sourceNames{i}) = struct('file',f,'sha256',fileHash(f));
    end
    if isempty(row), result.note = 'No isolated peaks in this saved batch.'; return; end
    projected = get_PCproj(data,row,col,ops.wPCA,ops.maskMaxChannels);
    projected = reshape(permute(projected,[2 1 3]),numel(row),ops.Nchan*3);
    oldRng = rng;
    rngCleanup = onCleanup(@() rng(oldRng));
    rng(1);
    dWU = optimizePeaks(ops,projected);
    map = load(fullfile(sorterOutput,'chanMap.mat'));
    kcoords = map.kcoords(map.connected > 1e-6);
    if numel(kcoords) ~= ops.Nchan
        error('Diagnostic:Channels', 'Connected map channels do not match ops.Nchan.');
    end
    [W,U,mu,UtU,nu] = decompose_dWU(ops,dWU,ops.Nrank,kcoords);
    result.finite_positive_template_amplitudes = nnz(isfinite(mu) & mu > 0);
    result.nonfinite_W_elements = nnz(~isfinite(W));
    result.matches = matchChecks(data,ops,W,U,mu,UtU,nu,dWU,gpuFns);
    result.matches_note = 'CPU-initialized templates supplied identically to CPU and GPU matchers.';
    if ~isempty(fieldnames(gpuFns))
        try
            % Also exercise GPU PCA/initialization, since backend parity with
            % CPU templates alone cannot establish that GPU initialization works.
            gpuOps = ops;
            gpuOps.GPU = true;
            gProjected = get_PCproj(gpuArray(data),row,col,ops.wPCA,ops.maskMaxChannels);
            gProjected = gather(reshape(permute(gProjected,[2 1 3]),numel(row),ops.Nchan*3));
            result.gpu_initialization.pca_max_abs_difference = double(max(abs(gProjected(:)-projected(:))));
            rng(1);
            gDwu = optimizePeaks(gpuOps,gProjected);
            [gW,gU,gMu,gUtU,gNu] = decompose_dWU(gpuOps,gDwu,ops.Nrank,kcoords);
            result.gpu_initialization.finite_positive_template_amplitudes = nnz(isfinite(gMu)&gMu>0);
            result.gpu_initialization.nonfinite_W_elements = nnz(~isfinite(gW));
            result.gpu_initialization.matches = matchChecks(data,ops,gW,gU,gMu,gUtU,gNu,gDwu,gpuFns);
            result.gpu_initialization.note = 'Same peak indices and RNG seed; native GPU PCA/optimization.';
        catch ME
            result.gpu_initialization.error = exceptionInfo(ME);
        end
    end
catch ME
    result.error = exceptionInfo(ME);
end
end

function result = matchChecks(data,ops,W,U,mu,UtU,nu,dWU,gpuFns)
ops.GPU = false;
ops.parfor = false;
ops.NT = size(data,1);
W0 = zeros(ops.NT,ops.Nfilt,ops.Nrank,'single');
W0(1:ops.nt0,:,:) = W;
fW = conj(fft(W0,[],1));
projection = data * U(:,:);
params = double([ops.NT ops.Nfilt ops.Th(1) ops.maxFR 10 ops.Nchan ops.Nrank ...
    exp(-ops.momentum(1)) ops.epu ops.nt0]);
lam = single(ops.lam(1)*(20./mu).^2);
[~,st] = mexMPregMUcpu(params,data,fW,projection,UtU,mu,lam,dWU,nu,ops);
result.cpu_training = numel(st);
wtw = zeros(2*ops.nt0-1,ops.Nfilt,ops.Nfilt,'single');
for i = 1:ops.Nrank
    for j = 1:ops.Nrank
        wtw = wtw + getWtW2(params,W(:,:,i),W(:,:,j),U(:,:,i)'*U(:,:,j));
    end
end
paramsFinal = params;
paramsFinal(3:5) = [ops.Th(3) 50000 50];
lamFinal = single(ops.lam(3)*(20./mu).^2);
st = cpuMPmuFEAT(paramsFinal,projection,fW,wtw,mu,lamFinal,nu,ops);
result.cpu_final = numel(st);
result.gpu_argument_convention = 'Original runtime: GPU data/projections; CPU W, mu, lam, nu and UtU.';
if isempty(fieldnames(gpuFns)), result.gpu_note = 'GPU test unavailable or disabled.'; return; end
try
    gData = gpuArray(data);
    gU = gpuArray(U);
    gProjection = gData*gU(:,:);
catch ME
    result.gpu_setup_error = exceptionInfo(ME);
    return;
end
try
    % Legacy MEX entry points write every output slot unconditionally.
    % Request the same six/five outputs as the production caller.
    [~,st,~,~,~,~] = gpuFns.reg(params,gData,W,gProjection,UtU,mu,lam,dWU,nu);
    wait(gpuDevice);
    result.gpu_training = numel(st);
catch ME
    result.gpu_training_error = exceptionInfo(ME);
end
try
    gWtw = gpuArray.zeros(size(wtw),'single');
    for i = 1:ops.Nrank
        for j = 1:ops.Nrank
            v = gpuFns.correlation(params,W(:,:,i),W(:,:,j),gU(:,:,i)'*gU(:,:,j));
            gWtw = gWtw + permute(v,[3 1 2]);
        end
    end
    wait(gpuDevice);
    difference = gather(gWtw) - wtw;
    result.gpu_correlation_max_error = double(max(abs(difference(:))));
    [st,~,~,~,~] = gpuFns.final(paramsFinal,gProjection,W,gWtw,mu,lamFinal,nu);
    wait(gpuDevice);
    result.gpu_final = numel(st);
catch ME
    result.gpu_final_error = exceptionInfo(ME);
end
end

function out = exceptionInfo(ME)
out = struct('identifier',ME.identifier,'message',ME.message, ...
    'stack',getReport(ME,'extended','hyperlinks','off'));
end

function result = canonicalPath(file)
if ~java.io.File(file).isAbsolute, file = fullfile(pwd,file); end
result = char(java.io.File(file).getCanonicalPath());
end

function sha = fileHash(file)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(file,'r');
cleanup = onCleanup(@() fclose(fid));
md.update(fread(fid,Inf,'*uint8'));
sha = lower(reshape(dec2hex(typecast(int8(md.digest()),'uint8'),2)',1,[]));
end
