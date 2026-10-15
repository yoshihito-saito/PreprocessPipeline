// Experimental INT16 temporal proposals; INT32 FIR accumulation and response.
// Four-byte responses stay within a fixed caller-checked GPU allocation budget.
// 17 taps * 32768 * 1024 is below INT32_MAX for every possible INT16 input.

extern "C" __global__ void matched_fir17(const short* raw, const short* coefficients,
    long long nt, int nc, int* response) {
    int c = blockIdx.x*32+threadIdx.x;
    int ty = threadIdx.y;
    long long first = (long long)blockIdx.y*8;
    __shared__ short tile[24][32];
    for (int r=ty; r<24; r+=8) {
        long long t=first+r-8;
        tile[r][threadIdx.x] = c<nc && t>=0 && t<nt ? raw[t*nc+c] : 0;
    }
    __syncthreads();
    long long t=first+ty;
    if (c<nc && t<nt) {
        int dot=0;
        // 17 * 32768 * 1024 < INT32_MAX: no wrap at any INT16 input.
        #pragma unroll
        for (int r=0;r<17;++r) dot += (int)coefficients[r] * (int)tile[ty+r][threadIdx.x];
        response[t*nc+c]=dot;
    }
}
extern "C" __global__ void matched_peaks(const int* response, const float* noise,
    const int* neighbors, int neighbor_width,
    long long nt, int nc, float floor, unsigned long long* count,
    long long* output, unsigned long long capacity) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    bool keep=false;
    if (i<nt*nc) {
        int c=i%nc; long long t=i/nc;
        float value=fabsf((float)response[i]);
        keep=t>=8 && t+8<nt && isfinite(noise[c]) && value>noise[c]*floor;
        if (keep) keep=value>fabsf((float)response[i-nc]) && value>=fabsf((float)response[i+nc]);
        if (keep && neighbor_width>0) {
            const float q=value/noise[c];
            for (int slot=0;slot<neighbor_width && keep;++slot) {
                int other=neighbors[c*neighbor_width+slot];
                if (other<0 || other==c) continue;
                for (int dt=-1;dt<=1;++dt) {
                    float otherq=fabsf((float)response[(t+dt)*nc+other])/noise[other];
                    if (otherq>q || (otherq==q && other<c)) {keep=false;break;}
                }
            }
        }
    }
    unsigned int bits=__ballot_sync(0xffffffff,keep);
    int lane=threadIdx.x&31;
    unsigned long long base=0;
    if (lane==0 && bits) base=atomicAdd(count,(unsigned long long)__popc(bits));
    base=__shfl_sync(0xffffffff,base,0);
    if (keep) {
        unsigned int before=lane==0 ? 0 : bits&((1u<<lane)-1u);
        unsigned long long slot=base+__popc(before);
        if (slot<capacity) output[slot]=i;
    }
}

// Normalize candidate metadata in one physical smooth3 SNR frame. FIR trigger
// scores remain separate from this value. No full-core FP32 array is created.
extern "C" __global__ void smooth3_candidate_strengths(
    const short* residual, const long long* indices, const float* scales,
    const float* noise, long long nt, int nc, int events, float* scores)
{
    const int event=blockIdx.x*blockDim.x+threadIdx.x;
    if (event>=events) return;
    const long long index=indices[event];
    const long long t=index/nc;
    const int c=index%nc;
    float value=(float)residual[index]*scales[c];
    if (t>0 && t+1<nt) {
        const float before=(float)residual[index-nc]*scales[c];
        const float after=(float)residual[index+nc]*scales[c];
        value=.25f*(before+2.f*value+after);
    }
    scores[event]=fabsf(value)/noise[c];
}
