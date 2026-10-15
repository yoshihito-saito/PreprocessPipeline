// Additional CUDA stages for learned-template matching. No fast-math.

__device__ __forceinline__ bool better(float value, long long index,
                                      float best, long long best_index) {
    const bool nan=value!=value, best_nan=best!=best;
    return (nan && !best_nan) || (!best_nan && value>best) ||
           (((nan && best_nan) || value==best) && index<best_index);
}

extern "C" __global__ void learned_max_f32(
    const float* __restrict__ B, const float* __restrict__ norm,
    float* __restrict__ maxima, long long* __restrict__ indices,
    long long units, long long samples, long long b0, long long b1,
    long long n0, long long edge) {
    const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=samples) return;
    float best=-__int_as_float(0x7f800000);
    long long winner=0;
    if(t<edge || t>=samples-edge) { maxima[t]=0; indices[t]=0; return; }
    for(long long u=0; u<units; ++u) {
        const float b=B[u*b0+t*b1];
        const float positive=b<0.0f ? 0.0f : b;
        const float score=__fdiv_rn(__fmul_rn(positive,positive),norm[u*n0]);
        if(better(score,u,best,winner)) { best=score; winner=u; }
    }
    maxima[t]=best; indices[t]=winner;
}

// Four warps read distinct units, each with contiguous accesses along time.
// Shared memory combines their partial max/argmax while preserving first ties.
extern "C" __global__ void learned_max_parallel_f32(
    const float* __restrict__ B, const float* __restrict__ norm,
    float* __restrict__ maxima, long long* __restrict__ indices,
    long long units, long long samples, long long b0, long long b1,
    long long n0, long long edge) {
    __shared__ float partial[4][32];
    __shared__ long long choices[4][32];
    const int lane=threadIdx.x&31, group=threadIdx.x>>5;
    const long long t=(long long)blockIdx.x*32+lane;
    float best=-__int_as_float(0x7f800000);
    long long winner=0x7fffffffffffffffLL;
    const bool active=t<samples && t>=edge && t<samples-edge;
    if(active) {
        for(long long u=group; u<units; u+=4) {
            const float b=B[u*b0+t*b1];
            const float positive=b<0.0f ? 0.0f : b;
            const float score=__fdiv_rn(__fmul_rn(positive,positive),norm[u*n0]);
            if(better(score,u,best,winner)) { best=score; winner=u; }
        }
    }
    partial[group][lane]=best; choices[group][lane]=winner;
    __syncthreads();
    if(group==0 && t<samples) {
        #pragma unroll
        for(int g=1; g<4; ++g) {
            const float v=partial[g][lane];
            const long long i=choices[g][lane];
            if(better(v,i,best,winner)) { best=v; winner=i; }
        }
        maxima[t]=active ? best : 0.0f;
        indices[t]=active ? winner : 0;
    }
}

// Each accepted event owns a disjoint target interval within this partition.
// The host adapter validates that invariant and falls back for overlaps.
extern "C" __global__ void validate_events(
    const long long* __restrict__ times, const long long* __restrict__ units,
    int* __restrict__ result, long long events, long long samples,
    long long unit_count, long long radius, long long ts, long long us) {
    __shared__ int warp_flags[8];
    int flag=0;
    for(long long i=threadIdx.x; i<events; i+=blockDim.x) {
        const long long t=times[i*ts],u=units[i*us];
        if(t<radius || t>=samples-radius) flag|=1;
        if(i && t-times[(i-1)*ts]<=2*radius) flag|=2;
        if(u<0 || u>=unit_count) flag|=4;
    }
    #pragma unroll
    for(int d=16;d;d>>=1) flag|=__shfl_down_sync(0xffffffff,flag,d);
    if((threadIdx.x&31)==0) warp_flags[threadIdx.x>>5]=flag;
    __syncthreads();
    if(threadIdx.x==0) {
        int combined=0;
        #pragma unroll
        for(int w=0;w<8;++w) combined|=warp_flags[w];
        result[0]=combined;
    }
}

extern "C" __global__ void subtract_correlations_f32(
    float* __restrict__ B, const float* __restrict__ ctc,
    const long long* __restrict__ times, const long long* __restrict__ units,
    const float* __restrict__ amps, long long output_units, long long events,
    long long radius, long long b0, long long b1, long long c0,
    long long c1, long long c2, long long ts, long long us, long long as) {
    const long long width=2*radius+1;
    const long long flat=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(flat>=output_units*events*width) return;
    const long long d=flat%width;
    const long long event=(flat/width)%events;
    const long long u=flat/(width*events);
    const long long target=u*b0+(times[event*ts]-radius+d)*b1;
    const float delta=__fmul_rn(amps[event*as],ctc[u*c0+units[event*us]*c1+d*c2]);
    B[target]=__fsub_rn(B[target],delta);
}
