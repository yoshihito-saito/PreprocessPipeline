function info = build_kilosort1_cuda(kilosortRoot, buildDirectory, varargin)
% Build checked KiloSort1 MEX files in a NEW directory using MATLAB's CUDA.
% Includes compute_90 PTX for newer GPUs. Does not change MATLAB's search path,
% replace installed binaries, select/reset a GPU or run a sorting job.
% Optional: 'HostCompiler', '/usr/bin/g++-11'. The default searches for GCC
% 6--12, the host compiler versions supported by MATLAB R2024b's CUDA 12.2.
p = inputParser;
addParameter(p,'HostCompiler','');
parse(p,varargin{:});
if ~strcmp(computer('arch'),'glnxa64')
    error('KiloSort:BuildPlatform','This controlled build currently supports Linux x86-64.');
end
source = fullfile(kilosortRoot,'CUDA');
config = fullfile(matlabroot,'toolbox','parallel','gpu','extern','src','mex','glnxa64','nvcc_g++.xml');
compiler = fullfile(matlabroot,'sys','cuda','glnxa64','cuda','bin');
runtime = fullfile(matlabroot,'bin','glnxa64');
if ~isfile(config) || ~isfile(fullfile(compiler,'nvcc'))
    error('KiloSort:BuildCompiler','MATLAB bundled CUDA compiler/configuration is unavailable.');
end
if ~isfile(fullfile(source,'ks_cuda_checks.h'))
    error('KiloSort:BuildSource','Use the checked CUDA sources supplied with this build helper.');
end
hostCompiler = selectHostCompiler(p.Results.HostCompiler);
if isfolder(buildDirectory) || isfile(buildDirectory)
    error('KiloSort:BuildExists','Choose a new build directory: %s',buildDirectory);
end
names = {'mexMPregMU','mexMPmuFEAT','mexWtW2'};
for i = 1:numel(names)
    if ~isfile(fullfile(source,[names{i} '.cu']))
        error('KiloSort:BuildSource','Missing CUDA source: %s',names{i});
    end
end
mkdir(buildDirectory);
buildDirectory = canonicalPath(buildDirectory);
source = canonicalPath(source);
variables = {'MW_NVCC_PATH','MW_ALLOW_ANY_CUDA'};
oldEnvironment = cellfun(@getenv,variables,'UniformOutput',false);
envCleanup = onCleanup(@() restoreEnvironment(variables,oldEnvironment));
setenv('MW_NVCC_PATH',compiler);
setenv('MW_ALLOW_ANY_CUDA','');
info = struct('matlab',version,'compiler_directory',compiler, ...
    'host_compiler',hostCompiler, ...
    'compiler_config',config,'source_directory',source,'output_directory',buildDirectory, ...
    'native_target','sm_90','forward_target','compute_90','kernel_synchronization',true);
logFile = fullfile(buildDirectory,'build.log');
diary(logFile);
logCleanup = onCleanup(@() diary('off'));
disp(info);
for i = 1:numel(names)
    fprintf('Building %s with MATLAB bundled CUDA and PTX...\n',names{i});
    mexcuda('-v','-f',config,'-largeArrayDims', ...
        ['CUDA_ROOT=' fileparts(compiler)], ...
        ['CUDA_LIBS=' runtime], ...
        ['GCC=' hostCompiler], ...
        'NVCCFLAGS=-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90', ...
        ['NVCC_FLAGS=-ccbin=' hostCompiler], ...
        '-DKS_CUDA_DEBUG_SYNCHRONIZE=1',['-I' source], ...
        '-outdir',buildDirectory,'-output',names{i},fullfile(source,[names{i} '.cu']));
end
fid = fopen(fullfile(buildDirectory,'build-info.json'),'w');
if fid < 0, error('KiloSort:BuildReport','Cannot write build information.'); end
fileCleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(info,'PrettyPrint',true));
fprintf('Built three MEX files in %s. Validate them with diagnose_kilosort1 before use.\n',buildDirectory);
end

function selected = selectHostCompiler(requested)
if isempty(requested)
    candidates = {'g++','g++-12','g++-11','g++-10','g++-9','g++-8','g++-7','g++-6'};
else
    candidates = {char(requested)};
end
for i = 1:numel(candidates)
    candidate = candidates{i};
    % These paths become compiler command arguments; reject shell syntax.
    if isempty(regexp(candidate,'^[A-Za-z0-9_./+-]+$','once'))
        error('KiloSort:HostCompiler','HostCompiler must be an executable path without shell syntax or spaces.');
    end
    [status,resolved] = system(['command -v ' candidate]);
    resolved = strtrim(resolved);
    if status ~= 0 || ~isfile(resolved), continue; end
    if isempty(regexp(resolved,'^[A-Za-z0-9_./+-]+$','once'))
        error('KiloSort:HostCompiler','Resolved compiler path contains unsupported characters.');
    end
    [status,release] = system([resolved ' -dumpversion']);
    major = str2double(regexp(strtrim(release),'^\d+','match','once'));
    if status == 0 && major >= 6 && major <= 12
        selected = resolved;
        return;
    end
end
error('KiloSort:HostCompiler', ...
    'No GCC 6--12 host compiler found. Supply HostCompiler pointing to a supported g++ executable.');
end

function restoreEnvironment(names,values)
for i = 1:numel(names), setenv(names{i},values{i}); end
end

function result = canonicalPath(file)
% Java's user.dir does not follow MATLAB cd; anchor relative paths explicitly.
if ~java.io.File(file).isAbsolute, file = fullfile(pwd,file); end
result = char(java.io.File(file).getCanonicalPath());
end
