// Fixed-point SOS frame v2. Bulk waveform buffers are short; Q12 states
// and Q28 coefficient products are evaluated with checked INT64 arithmetic.
__device__ long long rounded(long long x, int shift) {
    if (!shift) return x;
    long long a = x < 0 ? -x : x;
    long long y = (a + (1LL << (shift-1))) >> shift;
    return x < 0 ? -y : y;
}
__device__ bool step(long long &x, long long* z, const int* coef) {
    if (x > 2147483647LL || x < -2147483647LL) return false;
    for (int s=0; s<3; ++s) {
        const int* b=coef+s*5;
        long long y=rounded((long long)b[0]*x,28)+z[s*2];
        // Check y before the feedback products: two signed INT32-range
        // products can then be subtracted without overflowing INT64.
        if (y>2147483647LL || y< -2147483647LL) return false;
        long long a=rounded((long long)b[1]*x-(long long)b[3]*y,28)+z[s*2+1];
        long long d=rounded((long long)b[2]*x-(long long)b[4]*y,28);
        if (a>2147483647LL || a< -2147483647LL ||
            d>2147483647LL || d< -2147483647LL) return false;
        z[s*2]=a; z[s*2+1]=d; x=y;
    }
    return true;
}
__device__ int referenced(const short* raw, const int* refs, const int* bias,
                         int rid, int t, int c, int nt, int nc) {
    return 2*(int)raw[(long long)t*nc+c]-refs[(long long)rid*nt+t]-bias[c];
}
__device__ int reflected(const short* raw, const int* refs, const int* bias,
                        int rid, int t, int c, int nt, int nc) {
    if(t<0) return 2*referenced(raw,refs,bias,rid,0,c,nt,nc)-
                    referenced(raw,refs,bias,rid,-t,c,nt,nc);
    if(t>=nt) return 2*referenced(raw,refs,bias,rid,nt-1,c,nt,nc)-
                       referenced(raw,refs,bias,rid,2*nt-2-t,c,nt,nc);
    return referenced(raw,refs,bias,rid,t,c,nt,nc);
}
extern "C" __global__ void peaks_and_qc(const short* raw, const int* refs,
    const int* ids, const int* bias, int* peak, int* counts,
    int nt, int nc, int begin, int finish) {
    int c=blockIdx.x*32+threadIdx.x;
    if(c>=nc) return;
    int first=blockIdx.y*256, m=0, saturated=0, zero=0, flat=0;
    for(int t=first+threadIdx.y;t<nt && t<first+256;t+=8) {
        int value=raw[(long long)t*nc+c];
        m=max(m,abs(referenced(raw,refs,bias,ids[c],t,c,nt,nc)));
        if(t>=begin && t<finish) {
            saturated += value>=32760 || value<=-32760;
            zero += value==0;
            flat += t>begin && value==raw[(long long)(t-1)*nc+c];
        }
    }
    atomicMax(peak+c,m);
    atomicAdd(counts+c,saturated);
    atomicAdd(counts+nc+c,zero);
    atomicAdd(counts+2*nc+c,flat);
}
extern "C" __global__ void tiled_sos(const short* raw, short* data, short* scratch,
    const int* refs, const int* ids, const int* bias, const int* powers,
    const int* coef, const int* zi, const unsigned char* active, int* failure,
    int nt, int nc, int tile, int halo) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc || !active[c]) return;
    int begin=blockIdx.y*tile,finish=min(nt,begin+tile);
    int left=max(-21,begin-halo),right=min(nt+21,finish+halo);
    int rid=ids[c],shift=12+powers[c];
    long long base=(long long)blockIdx.y*(tile+2*halo)*nc+c;
    long long first=(long long)reflected(raw,refs,bias,rid,left,c,nt,nc)*2048;
    if(first>2147483647LL || first< -2147483647LL) {atomicOr(failure+c,1);return;}
    long long z[6];
    for(int k=0;k<6;++k) z[k]=rounded((long long)zi[k]*first,28);
    for(int t=left;t<right;++t) {
        long long x=(long long)reflected(raw,refs,bias,rid,t,c,nt,nc)*2048;
        if(!step(x,z,coef)) {atomicOr(failure+c,1);return;}
        long long q=rounded(x,shift);
        if(q< -32768 || q>32767) {atomicOr(failure+c,2);return;}
        scratch[base+(long long)(t-left)*nc]=(short)q;
    }
    first=(long long)scratch[base+(long long)(right-left-1)*nc]*(1LL<<shift);
    if(first>2147483647LL || first< -2147483647LL) {atomicOr(failure+c,1);return;}
    for(int k=0;k<6;++k) z[k]=rounded((long long)zi[k]*first,28);
    for(int t=right-1;t>=left;--t) {
        long long x=(long long)scratch[base+(long long)(t-left)*nc]*(1LL<<shift);
        if(!step(x,z,coef)) {atomicOr(failure+c,1);return;}
        long long q=rounded(x,shift);
        if(q< -32768 || q>32767) {atomicOr(failure+c,2);return;}
        if(t>=begin && t<finish) data[(long long)t*nc+c]=(short)q;
    }
}
extern "C" __global__ void reserve_headroom(short* data, const int* shifts,
    long long count, int nc) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) data[i]=(short)rounded((long long)data[i],shifts[i%nc]);
}
extern "C" __global__ void synchronous_artifact(const short* data,
    const float* scales, const float* noise, const unsigned char* usable,
    int* flag, int nc, int begin, int finish, int good_count) {
    int lane=threadIdx.x%32;
    int t=begin+blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(t>=finish) return;
    int count=0;
    for(int c=lane;c<nc;c+=32)
        count += usable[c] && fabsf((float)data[(long long)t*nc+c]*scales[c]) > 10.f*noise[c];
    for(int offset=16;offset;offset/=2)
        count+=__shfl_down_sync(0xffffffff,count,offset);
    if(lane==0 && count*10>good_count*3) atomicOr(flag,1);
}
