// Fuse ADC decoding, boundary replication and sample/channel transpose.
// Input loads and output stores are coalesced. Padding avoids shared-bank
// conflicts during the transpose; 256 threads cover a 32x32 tile.
extern "C" __global__ void decode_transpose_pad_i16(
    const short* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ scale, const float* __restrict__ offset,
    long long samples, long long channels, long long output_samples,
    long long left_pad, long long valid_stop) {
    __shared__ short tile[32][34];
    const long long channel=(long long)blockIdx.x*32+threadIdx.x;
    const long long time_base=(long long)blockIdx.y*32;
    #pragma unroll
    for(int j=0;j<32;j+=8) {
        const long long t=time_base+threadIdx.y+j;
        long long source=t-left_pad;
        source=source<0 ? 0 : (source>=samples ? samples-1 : source);
        tile[threadIdx.y+j][threadIdx.x]=(channel<channels && t<output_samples)
            ? input[source*channels+channel] : 0;
    }
    __syncthreads();
    const long long t=time_base+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8) {
        const long long c=(long long)blockIdx.x*32+threadIdx.y+j;
        if(c<channels && t<output_samples)
            output[c*output_samples+t]=t>=valid_stop ? 0.f : __fadd_rn(
                __fmul_rn((float)tile[threadIdx.x][threadIdx.y+j],scale[c]),offset[c]);
    }
}
