// Six 61-tap filters with shared input/weights and coalesced output stores.
// Output layout is selected to avoid a later whole-batch permutation copy.
extern "C" __global__ void temporal_6x61_f32(
        const float* __restrict__ x, const float* __restrict__ weights,
        float* __restrict__ y, int channels, int samples, int pca_major) {
    __shared__ float signal[256+60];
    __shared__ float w[6*61];
    const int c = blockIdx.y, start = blockIdx.x * 256;
    for (int i=threadIdx.x; i<316; i+=256) {
        const int t=start+i-30;
        signal[i]=(t>=0 && t<samples) ? x[(long long)c*samples+t] : 0.0f;
    }
    for (int i=threadIdx.x; i<366; i+=256) w[i]=weights[i];
    __syncthreads();
    const int t=start+threadIdx.x;
    if (t>=samples) return;
    #pragma unroll 1
    for (int p=0; p<6; ++p) {
        float sum=0.0f;
        #pragma unroll
        for (int k=0; k<61; ++k)
            sum=__fmaf_rn(signal[threadIdx.x+k], w[p*61+k], sum);
        const long long row=pca_major ? (long long)p*channels+c : (long long)c*6+p;
        y[row*samples+t]=sum;
    }
}

// Separate 41-tap specialization: keep the measured 61-tap kernel unchanged.
extern "C" __global__ void temporal_6x41_f32(
        const float* __restrict__ x, const float* __restrict__ weights,
        float* __restrict__ y, int channels, int samples, int pca_major) {
    __shared__ float signal[256+40];
    __shared__ float w[6*41];
    const int c = blockIdx.y, start = blockIdx.x * 256;
    for (int i=threadIdx.x; i<296; i+=256) {
        const int t=start+i-20;
        signal[i]=(t>=0 && t<samples) ? x[(long long)c*samples+t] : 0.0f;
    }
    for (int i=threadIdx.x; i<246; i+=256) w[i]=weights[i];
    __syncthreads();
    const int t=start+threadIdx.x;
    if (t>=samples) return;
    #pragma unroll 1
    for (int p=0; p<6; ++p) {
        float sum=0.0f;
        #pragma unroll
        for (int k=0; k<41; ++k)
            sum=__fmaf_rn(signal[threadIdx.x+k], w[p*41+k], sum);
        const long long row=pca_major ? (long long)p*channels+c : (long long)c*6+p;
        y[row*samples+t]=sum;
    }
}
