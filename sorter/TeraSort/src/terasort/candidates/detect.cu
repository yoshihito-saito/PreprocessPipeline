// Candidate detector v0.1. Compile with NVRTC via CuPy; no PyTorch operators.
// Input is a finite, float32, time-major score array (amplitude/noise).
// Temporal ties select the first maximum. Spatial ties are retained.
extern "C" __global__ void temporal_flags(
    const float* q, unsigned char* flags, long long nt, int nc,
    int radius, float floor) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt * nc) return;
    long long t = i / nc;
    float v = q[i];
    bool keep = v > floor;
    if (keep) {
        for (int d = 1; d <= radius; ++d) {
            if (t >= d && q[i - (long long)d * nc] >= v) keep = false;
            if (t + d < nt && q[i + (long long)d * nc] > v) keep = false;
            if (!keep) break;
        }
    }
    flags[i] = keep;
}

extern "C" __global__ void pack_candidates(
    const float* q, const unsigned char* flags, const int* neighbors,
    long long nt, int nc, int nneigh, int spatial_radius,
    long long valid_start, long long valid_stop,
    unsigned long long* count, long long* output, unsigned long long capacity) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt * nc || !flags[i]) return;
    long long t = i / nc;
    int c = i % nc;
    if (t < valid_start || t >= valid_stop) return;
    float v = q[i];
    if (nneigh) {
        for (int d = -spatial_radius; d <= spatial_radius; ++d) {
            long long s = t + d;
            if (s < 0 || s >= nt) continue;
            for (int k = 0; k < nneigh; ++k) {
                int c2 = neighbors[c * nneigh + k];
                if (c2 < 0) continue;
                long long j = s * nc + c2;
                if (flags[j] && !(v >= q[j] - 1.0e-8f)) return;
            }
        }
    }
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < capacity) output[slot] = i;
    // count still increments on overflow: the host MUST reject truncated output.
}

extern "C" __global__ void temporal_pack(
    const float* q, long long nt, int nc, int radius, float floor,
    long long valid_start, long long valid_stop,
    unsigned long long* count, long long* output, unsigned long long capacity) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt * nc) return;
    long long t = i / nc;
    if (t < valid_start || t >= valid_stop) return;
    float v = q[i];
    if (!(v > floor)) return;
    for (int d = 1; d <= radius; ++d) {
        if (t >= d && q[i - (long long)d * nc] >= v) return;
        if (t + d < nt && q[i + (long long)d * nc] > v) return;
    }
    unsigned long long slot = atomicAdd(count, 1ULL);
    if (slot < capacity) output[slot] = i;
}
