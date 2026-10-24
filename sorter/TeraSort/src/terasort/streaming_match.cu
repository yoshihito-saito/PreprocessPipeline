// One warp per event/template pair. Channels are sparse, time remains dense.
// This is a reference CUDA implementation: no Torch tensors or dense KxK table.

__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}

extern "C" __global__ void score_pairs(
    const float* __restrict__ events, const int* __restrict__ event_channels,
    const float* __restrict__ templates, const int* __restrict__ template_channels,
    const int* __restrict__ pair_event, const int* __restrict__ pair_template,
    int n_pairs, int n_time, int width,
    float* __restrict__ scores, float* __restrict__ amplitudes,
    int* __restrict__ shifts)
{
    const int p = blockIdx.x;
    const int lane = threadIdx.x;
    if (p >= n_pairs || lane >= 32) return;
    const int e = pair_event[p], u = pair_template[p];
    const int* ec = event_channels + e * width;
    const int* tc = template_channels + u * width;
    const float* ew = events + e * n_time * width;
    const float* tw = templates + u * n_time * width;
    const int channel = lane < width ? tc[lane] : -1;
    int event_slot = -1;
    if (channel >= 0) {
        for (int k = 0; k < width; ++k)
            if (ec[k] == channel) { event_slot = k; break; }
    }
    float total_a = 0.f, total_b = 0.f, shared_a = 0.f, shared_b = 0.f;
    float dot[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    float anorm[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    float bnorm[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    float valid[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    float possible[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll 1
    for (int t = 0; t < n_time; ++t) {
        const float a = channel >= 0 ? tw[t * width + lane] : 0.f;
        if (channel >= 0) total_a += a * a;
        if (event_slot >= 0) shared_a += a * a;
        if (lane < width && ec[lane] >= 0) {
            const float b0 = ew[t * width + lane];
            if (isfinite(b0)) total_b += b0 * b0;
        }
        if (event_slot >= 0) {
            const float b0 = ew[t * width + event_slot];
            if (isfinite(b0)) shared_b += b0 * b0;
            #pragma unroll 1
            for (int s = -2; s <= 2; ++s) {
                const int et = t + s;
                if (et < 0 || et >= n_time) continue;
                const int k = s + 2;
                possible[k] += 1.f;
                const float b = ew[et * width + event_slot];
                if (!isfinite(b)) continue;
                valid[k] += 1.f;
                dot[k] += a * b;
                anorm[k] += a * a;
                bnorm[k] += b * b;
            }
        }
    }
    total_a = warp_sum(total_a);
    total_b = warp_sum(total_b);
    shared_a = warp_sum(shared_a);
    shared_b = warp_sum(shared_b);
    #pragma unroll 1
    for (int k = 0; k < 5; ++k) {
        dot[k] = warp_sum(dot[k]);
        anorm[k] = warp_sum(anorm[k]);
        bnorm[k] = warp_sum(bnorm[k]);
        valid[k] = warp_sum(valid[k]);
        possible[k] = warp_sum(possible[k]);
    }
    if (lane == 0) {
        float best = -1.f, amplitude = __int_as_float(0x7fc00000);
        int best_shift = 0;
        if (total_a > 0.f && total_b > 0.f &&
            shared_a >= .7f * total_a && shared_b >= .7f * total_b) {
            #pragma unroll 1
            for (int k = 0; k < 5; ++k) {
                if (possible[k] <= 0.f || valid[k] < .9f * possible[k] ||
                    anorm[k] <= 0.f || bnorm[k] <= 0.f) continue;
                const float score = dot[k] * rsqrtf(anorm[k] * bnorm[k]);
                if (score > best) {
                    best = score;
                    amplitude = dot[k] / anorm[k];
                    best_shift = k - 2;
                }
            }
        }
        scores[p] = best;
        amplitudes[p] = amplitude;
        shifts[p] = best_shift;
    }
}

// A local candidate list is short (<=512). One lane scans each list so only
// the best and runner-up scores cross PCIe. Matching dominates this reduction.
extern "C" __global__ void best_two(
    const int* __restrict__ offsets, const int* __restrict__ pair_template,
    const float* __restrict__ scores, const float* __restrict__ amplitudes,
    const int* __restrict__ shifts, int n_events,
    int* __restrict__ best_template, float* __restrict__ best_score,
    float* __restrict__ runner_score, float* __restrict__ best_amplitude,
    int* __restrict__ best_shift)
{
    const int e = blockIdx.x;
    if (e >= n_events || threadIdx.x != 0) return;
    float first = -1.f, second = -1.f, amp = __int_as_float(0x7fc00000);
    int unit = -1, shift = 0;
    for (int p = offsets[e]; p < offsets[e+1]; ++p) {
        if (!isfinite(amplitudes[p]) || amplitudes[p] <= 0.f) continue;
        const float score = scores[p];
        if (score > first) {
            second = first;
            first = score;
            unit = pair_template[p];
            amp = amplitudes[p];
            shift = shifts[p];
        } else if (score > second) {
            second = score;
        }
    }
    best_template[e] = unit;
    best_score[e] = first;
    runner_score[e] = second;
    best_amplitude[e] = amp;
    best_shift[e] = shift;
}
