// Apply the chosen k-means++ proposal without boolean-index allocations or
// copying the scalar winner to the host. Scoring and RNG stay in Kilosort.
extern "C" __global__ void seed_cluster_update(
    const float* explained, const float* improvement, const long long* winner,
    float* best, int* labels, int rows, int proposals, int iteration)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    long long offset = (long long)row*proposals+winner[0];
    if (improvement[offset] > 0.f) {
        labels[row] = iteration;
        best[row] = explained[offset];
    }
}
