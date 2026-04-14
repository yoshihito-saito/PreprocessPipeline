% mexGPUall. For these to complete succesfully, you need to configure the
% Matlab GPU library first (see README files for platform-specific
% information)
    mexcuda -largeArrayDims mexMPmuFEAT.cu NVCC_FLAGS="-allow-unsupported-compiler"
    mexcuda -largeArrayDims mexMPregMU.cu NVCC_FLAGS="-allow-unsupported-compiler"
    mexcuda -largeArrayDims mexWtW2.cu NVCC_FLAGS="-allow-unsupported-compiler"

%    mex -largeArrayDims mexMPmuFEAT.cu
%    mex -largeArrayDims mexMPregMU.cu
%    mex -largeArrayDims mexWtW2.cu

% If you get uninterpretable errors like "An unexpected error occurred during CUDA execution", and if you are using Pascal GPUs 
% (GTX 10xx series), it might be necessary to upgrade to Matlab 2017a / CUDA 8.0. 


