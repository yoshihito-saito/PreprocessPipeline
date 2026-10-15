// Positive spatial pooling of the fixed INT32 temporal response.
// width <= 16 and INT16 weights bound every sum below 2^50, within INT64.
extern "C" __global__ void spatial_temporal_average(
    const int* response, const int* neighbors, const short* weights,
    const int* denominators, long long count, int channels, int width,
    int* output)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int c = i % channels;
    long long row = i - c;
    int denominator = denominators[c];
    if (denominator <= 0) { output[i] = 0; return; }
    long long sum = 0;
    for (int slot = 0; slot < width; ++slot) {
        int neighbor = neighbors[c * width + slot];
        int weight = weights[c * width + slot];
        if (neighbor >= 0 && weight > 0)
            sum += (long long)response[row + neighbor] * weight;
    }
    // Symmetric nearest rounding; positive averages stay within INT32 bounds.
    sum += sum >= 0 ? denominator / 2 : -denominator / 2;
    output[i] = (int)(sum / denominator);
}
