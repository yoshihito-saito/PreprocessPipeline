// Fused selection/reduction kernels. No template arithmetic or thresholds change.

extern "C" __global__ void neighbor_max_f32(
    const float* __restrict__ values, const long long* __restrict__ neighbors,
    float* __restrict__ output, long long filters, long long samples,
    long long count, long long value_stride0, long long value_stride1,
    long long neighbor_stride0, long long neighbor_stride1) {
    const long long flat = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= filters * samples) return;
    const long long f = flat / samples, t = flat % samples;
    float best = -__int_as_float(0x7f800000);
    for (long long n = 0; n < count; ++n) {
        const long long channel = neighbors[n * neighbor_stride0 + f * neighbor_stride1];
        const float value = values[channel * value_stride0 + t * value_stride1];
        if (value != value || value > best) best = value;
    }
    output[flat] = best;
}

// Each block owns one template center and a contiguous tile in time. Neighbor
// indices are reused by all threads, instead of being fetched per output.
extern "C" __global__ void neighbor_max_shared_f32(
    const float* __restrict__ values, const long long* __restrict__ neighbors,
    float* __restrict__ output, long long filters, long long samples,
    long long count, long long value_stride0, long long value_stride1,
    long long neighbor_stride0, long long neighbor_stride1) {
    __shared__ long long ids[256];
    const long long f = blockIdx.y;
    for (int n=threadIdx.x; n<count; n+=blockDim.x)
        ids[n] = neighbors[n*neighbor_stride0+f*neighbor_stride1];
    __syncthreads();
    const long long t = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (t>=samples) return;
    float best = -__int_as_float(0x7f800000);
    for (int n=0; n<count; ++n) {
        const float value = values[ids[n]*value_stride0+t*value_stride1];
        if (value != value || value>best) best=value;
    }
    output[f*samples+t]=best;
}

// Four lanes cooperate on each time sample. Warp shuffles reduce partial
// maxima; adjacent groups access adjacent time samples. This is a benchmark
// candidate, not an assumed improvement over contiguous-thread tiling.
extern "C" __global__ void neighbor_max_warp_f32(
    const float* __restrict__ values, const long long* __restrict__ neighbors,
    float* __restrict__ output, long long filters, long long samples,
    long long count, long long value_stride0, long long value_stride1,
    long long neighbor_stride0, long long neighbor_stride1) {
    __shared__ long long ids[256];
    const long long f=blockIdx.y;
    for (int n=threadIdx.x; n<count; n+=blockDim.x)
        ids[n]=neighbors[n*neighbor_stride0+f*neighbor_stride1];
    __syncthreads();
    const int lane=threadIdx.x&3;
    const long long t=(long long)blockIdx.x*(blockDim.x/4)+threadIdx.x/4;
    float best=-__int_as_float(0x7f800000);
    if(t<samples) {
        for(int n=lane; n<count; n+=4) {
            const float value=values[ids[n]*value_stride0+t*value_stride1];
            if(value!=value || value>best) best=value;
        }
    }
    for(int delta=2; delta; delta>>=1) {
        const float value=__shfl_down_sync(0xffffffff,best,delta,4);
        if(value!=value || value>best) best=value;
    }
    if(lane==0 && t<samples) output[f*samples+t]=best;
}

// Default Kilosort geometry: 10 local channels, 5 spatial widths, arbitrary
// temporal-template count. Cache weights and channel indices cooperatively.
// One thread owns one time sample and reuses a voltage load across five sums.
// No gathered tensor, einsum output or abs tensor is materialized.
extern "C" __global__ void template_scores_shared_f32(
    const float* __restrict__ B, const long long* __restrict__ channels,
    const float* __restrict__ weights, float* __restrict__ maxima,
    long long* __restrict__ signed_indices, float* __restrict__ diagnostic,
    long long filters, long long total_samples, long long samples, long long offset,
    long long nk, long long b0, long long b1, long long b2,
    long long c0, long long c1, long long w0, long long w1, long long w2,
    int arithmetic) {
    __shared__ long long ids[10];
    __shared__ float w[5][10];
    const long long f=blockIdx.y;
    if(threadIdx.x<10) ids[threadIdx.x]=channels[threadIdx.x*c0+f*c1];
    if(threadIdx.x<50) {
        const int s=threadIdx.x/10, j=threadIdx.x%10;
        w[s][j]=weights[s*w0+j*w1+f*w2];
    }
    __syncthreads();
    const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=samples) return;
    float best=-1.0f,winner=0.0f;
    long long winner_index=0;
    for(int k=0; k<nk; ++k) {
        float sums[5]={0,0,0,0,0};
        #pragma unroll
        for(int j=0; j<10; ++j) {
            const float value=B[ids[j]*b0+k*b1+(offset+t)*b2];
            #pragma unroll
            for(int s=0; s<5; ++s) {
                sums[s]=arithmetic ? __fmaf_rn(w[s][j],value,sums[s]) : sums[s]+w[s][j]*value;
            }
        }
        #pragma unroll
        for(int s=0; s<5; ++s) {
            const long long index=s*nk+k;
            const float value=sums[s];
            if(diagnostic) diagnostic[(index*filters+f)*samples+t]=value;
            const float magnitude=__uint_as_float(__float_as_uint(value)&0x7fffffffU);
            const bool nan=magnitude!=magnitude, best_nan=best!=best;
            if((nan && !best_nan) || (!best_nan && magnitude>best) ||
               (((nan && best_nan) || magnitude==best) && index<winner_index)) {
                best=magnitude; winner=value; winner_index=index;
            }
        }
    }
    maxima[f*samples+t]=best;
    signed_indices[f*samples+t]=(winner_index+1)*((winner>0.0f)-(winner<0.0f));
}

extern "C" __global__ void abs_max_signed_f32(
    const float* __restrict__ values, float* __restrict__ maxima,
    long long* __restrict__ signed_indices, long long choices,
    long long filters, long long samples, long long stride0,
    long long stride1, long long stride2) {
    const long long flat = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= filters * samples) return;
    const long long f = flat / samples, t = flat % samples;
    const long long base = f * stride1 + t * stride2;
    float winner = values[base];
    float best = __uint_as_float(__float_as_uint(winner) & 0x7fffffffU);
    long long index = 0;
    for (long long k = 1; k < choices; ++k) {
        const float value = values[k * stride0 + base];
        const float magnitude = __uint_as_float(__float_as_uint(value) & 0x7fffffffU);
        // Torch max chooses the first index in a tie, including the first NaN.
        if ((magnitude != magnitude && best == best) || magnitude > best) {
            best = magnitude;
            winner = value;
            index = k;
        }
    }
    maxima[flat] = best;
    const long long sign = (winner > 0.0f) - (winner < 0.0f);
    signed_indices[flat] = (index + 1) * sign;
}
