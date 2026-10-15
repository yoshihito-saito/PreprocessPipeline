#ifndef SESSION_NT
#define SESSION_NT 61
#endif
#define SESSION_HALF_NT (SESSION_NT / 2)

// Sparse direct-residual matcher. Four warps share a block; each warp owns
// one event/template proposal. No cross-warp synchronization is required.
// Optional scaled INT16 resident storage; score math always uses FP32.
#ifdef SESSION_INT16
typedef short session_sample_t;
#define SESSION_EXTRA , const float* residual_scale, const float* template_scale, int* overflow
#define RV(index, channel) (float(residual[index]) * residual_scale[channel])
#define TV(value, unit) (float(value) * template_scale[unit])

// Two neighboring shorts share a CAS word. Preserve the other half even if
// another warp updates it concurrently. The allocation includes end padding.
__device__ __forceinline__ void session_subtract_int16(
    short* address, float delta, int* overflow)
{
    if (!isfinite(delta) || fabsf(delta) > 65535.f) {
        atomicExch(overflow, 1);
        return;
    }
    const int change = __float2int_rn(delta);
    if (change == 0) return;
    const unsigned long long pointer = (unsigned long long)address;
    unsigned int* word = (unsigned int*)(pointer & ~3ull);
    const int shift = (pointer & 2ull) ? 16 : 0;
    const unsigned int mask = 0xffffu << shift;
    unsigned int old = atomicCAS(word, 0u, 0u), expected;
    do {
        expected = old;
        const int previous = (short)((expected >> shift) & 0xffffu);
        const int next = previous - change;
        if (next < -32768 || next > 32767) {
            atomicExch(overflow, 1);
            return;  // Never silently wrap or accept a clipped result.
        }
        const unsigned int updated = (expected & ~mask)
            | ((unsigned int)(unsigned short)next << shift);
        old = atomicCAS(word, expected, updated);
    } while (old != expected);
}
#else
typedef float session_sample_t;
#define SESSION_EXTRA
#define RV(index, channel) residual[index]
#define TV(value, unit) value
#endif

// Expand a bounded event batch using the resident channel/template CSR index.
// Pair order matches channel-list order so equal-gain winners stay stable.
extern "C" __global__ void expand_template_pairs(
    const int* event_channels, const int* offsets, const int* channel_offsets,
    const int* channel_units, int first_event, int* pair_event, int* pair_unit)
{
    int event = blockIdx.x;
    int start = offsets[event], stop = offsets[event+1];
    if (start == stop) return;
    int source = channel_offsets[event_channels[first_event+event]];
    for (int p = start+threadIdx.x; p < stop; p += blockDim.x) {
        pair_event[p] = first_event+event;
        pair_unit[p] = channel_units[source+p-start];
    }
}

// Pair order is deterministic. Strict comparisons preserve the first pair
// on equal gains, including ties between templates belonging to one identity.
extern "C" __global__ void select_identity_winners(
    const int* offsets, const int* units, const long long* identities,
    const float* scores, const float* amplitudes, const float* gains,
    const int* shifts, int events, float* selected)
{
    int event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= events) return;
    float* out = selected + (long long)event * 8;
    for (int k = 0; k < 8; ++k) out[k] = 0.f;
    int best = -1, runner = -1;
    float best_gain = 0.f, runner_gain = 0.f;
    for (int p = offsets[event]; p < offsets[event+1]; ++p) {
        if (gains[p] > best_gain) { best_gain = gains[p]; best = p; }
    }
    if (best < 0) return;
    long long identity = identities[units[best]];
    for (int p = offsets[event]; p < offsets[event+1]; ++p) {
        if (identities[units[p]] != identity && gains[p] > runner_gain) {
            runner_gain = gains[p]; runner = p;
        }
    }
    out[0] = scores[best];
    out[1] = best_gain;
    out[2] = __int_as_float(units[best]);
    out[3] = amplitudes[best];
    out[4] = runner < 0 ? -1.f : scores[runner];
    out[5] = __int_as_float(shifts[best]);
    out[6] = runner_gain;
    out[7] = 1.f;
}

// Fuse normalization, temporal peak detection and bounded packing. Adjacent
// lanes access adjacent contacts; no full time-by-channel score array.
extern "C" __global__ void detect_residual_peaks(
    const session_sample_t* residual, const float* noise, long long nt, int nc,
    float floor, unsigned long long* count, long long* output,
    unsigned long long capacity SESSION_EXTRA)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long size = nt * nc;
    bool keep = false;
    if (i < size) {
        const int c = i % nc;
        const long long t = i / nc;
        const float q = fabsf(RV(i, c)) / noise[c];
        keep = q > floor;
        if (keep && t > 0) keep = q > fabsf(RV(i-nc, c)) / noise[c];
        if (keep && t+1 < nt) keep = q >= fabsf(RV(i+nc, c)) / noise[c];
    }
    const unsigned int bits = __ballot_sync(0xffffffff, keep);
    const int lane = threadIdx.x & 31;
    unsigned long long base = 0;
    if (lane == 0 && bits) base = atomicAdd(count, (unsigned long long)__popc(bits));
    base = __shfl_sync(0xffffffff, base, 0);
    if (keep) {
        const unsigned int before = lane == 0 ? 0 : bits & ((1u << lane)-1u);
        const unsigned long long slot = base + __popc(before);
        if (slot < capacity) output[slot] = i;
    }
}

// Three-tap temporal smoothing and local-peak detection in one pass. A block
// cooperatively stages an 8-sample time tile plus two halo samples per side.
// Warps own one time row, so lanes remain coalesced over neighboring contacts.
extern "C" __global__ void detect_residual_peaks_smooth3_shared(
    const session_sample_t* residual, const float* noise, long long nt, int nc,
    float floor, unsigned long long* count, long long* output,
    unsigned long long capacity SESSION_EXTRA)
{
    const int cx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c = (int)blockIdx.x * 32 + cx;
    const long long base = (long long)blockIdx.y * 8;
    __shared__ float tile[12][32];

    for (int row = ty; row < 12; row += 8) {
        long long t = base + row - 2;
        if (t < 0) t = 0;
        if (t >= nt) t = nt - 1;
        tile[row][cx] = c < nc ? RV(t * (long long)nc + c, c) : 0.f;
    }
    __syncthreads();

    const long long t = base + ty;
    bool keep = false;
    if (t < nt && c < nc && isfinite(noise[c]) && noise[c] > 0.f) {
        const float center_raw = tile[ty + 2][cx];
        const float center = (t == 0 || t == nt - 1)
            ? center_raw
            : 0.25f * (tile[ty + 1][cx] + 2.f * center_raw + tile[ty + 3][cx]);
        const float q = fabsf(center) / noise[c];
        keep = q > floor;
        if (keep && t > 0) {
            const long long previous_t = t - 1;
            const float previous_raw = tile[ty + 1][cx];
            const float previous = (previous_t == 0 || previous_t == nt - 1)
                ? previous_raw
                : 0.25f * (tile[ty][cx] + 2.f * previous_raw + tile[ty + 2][cx]);
            keep = q > fabsf(previous) / noise[c];
        }
        if (keep && t + 1 < nt) {
            const long long next_t = t + 1;
            const float next_raw = tile[ty + 3][cx];
            const float next = (next_t == 0 || next_t == nt - 1)
                ? next_raw
                : 0.25f * (tile[ty + 2][cx] + 2.f * next_raw + tile[ty + 4][cx]);
            keep = q >= fabsf(next) / noise[c];
        }
    }

    const unsigned int bits = __ballot_sync(0xffffffff, keep);
    const int lane = cx;
    unsigned long long base_out = 0;
    if (lane == 0 && bits)
        base_out = atomicAdd(count, (unsigned long long)__popc(bits));
    base_out = __shfl_sync(0xffffffff, base_out, 0);
    if (keep) {
        const unsigned int before = lane == 0 ? 0 : bits & ((1u << lane) - 1u);
        const unsigned long long slot = base_out + __popc(before);
        if (slot < capacity)
            output[slot] = t * (long long)nc + c;
    }
}

__device__ __forceinline__ float session_warp_sum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}

// Exact center-first rejection. Integer full-window sums reuse the unmasked
// center products; failing center proposals do not read the waveform tails.
extern "C" __global__ void score_residual_pairs(
    const session_sample_t* __restrict__ residual, const int* __restrict__ event_time,
    const session_sample_t* __restrict__ templates, const int* __restrict__ channels,
    const int* __restrict__ pair_event, const int* __restrict__ pair_unit,
    const float* __restrict__ noise_weight,
    int n_pairs, int n_samples, int n_channels, int width, int half_width,
    const int* __restrict__ shift_radii,
    float score_floor, float amplitude_min, float amplitude_max,
    float* __restrict__ scores, float* __restrict__ amplitudes,
    float* __restrict__ gains, int* __restrict__ shifts SESSION_EXTRA)
{
    const int p = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (p >= n_pairs) return;
    const int event = pair_event[p], unit = pair_unit[p];
    const int shift_radius = shift_radii[unit], center = event_time[event];
    const int channel = lane < width ? channels[unit * width + lane] : -1;
    const session_sample_t* model = templates + unit * SESSION_NT * width;
    const int begin = max(-SESSION_HALF_NT, -half_width), end = min(SESSION_HALF_NT, half_width);
    float best_gain = 0.f, best_score = -1.f, best_amplitude = 0.f;
    int best_shift = 0;
    for (int shift = -shift_radius; shift <= shift_radius; ++shift) {
        const int t0 = center + shift;
        if (t0 < SESSION_HALF_NT || t0 + SESSION_HALF_NT >= n_samples) continue;
        float dot = 0.f, aa = 0.f, bb = 0.f;
#if defined(SESSION_INT16) && defined(SESSION_INTEGER_DOT)
        long long dot_q = 0, aa_q = 0, bb_q = 0;
#endif
        if (channel >= 0) {
            const float weight = noise_weight[channel];
            #pragma unroll 1
            for (int dt = begin; dt <= end; ++dt) {
#if defined(SESSION_INT16) && defined(SESSION_INTEGER_DOT)
                const int a = model[(SESSION_HALF_NT + dt) * width + lane];
                const int b = residual[(long long)(t0 + dt) * n_channels + channel];
                dot_q += (long long)a * b;
                aa_q += (long long)a * a;
                bb_q += (long long)b * b;
#else
                const float a = TV(model[(SESSION_HALF_NT + dt) * width + lane], unit);
                const float b = RV((long long)(t0 + dt) * n_channels + channel, channel);
                dot += a * b * weight;
                aa += a * a * weight;
                bb += b * b * weight;
#endif
            }
#if defined(SESSION_INT16) && defined(SESSION_INTEGER_DOT)
            const float a_scale = template_scale[unit], b_scale = residual_scale[channel];
            dot = (float)dot_q * (a_scale * b_scale * weight);
            aa = (float)aa_q * (a_scale * a_scale * weight);
            bb = (float)bb_q * (b_scale * b_scale * weight);
#endif
        }
#ifdef SESSION_CONTACT_SCORE
        float maximum_contact_energy = aa;
        for (int d = 16; d; d >>= 1)
            maximum_contact_energy = fmaxf(maximum_contact_energy,
                __shfl_xor_sync(0xffffffff, maximum_contact_energy, d));
        if (aa < maximum_contact_energy * .05f) { dot = 0.f; aa = 0.f; bb = 0.f; }
#endif
        dot = session_warp_sum(dot); aa = session_warp_sum(aa); bb = session_warp_sum(bb);
        const float score = dot * rsqrtf(aa * bb);
        int eligible = lane == 0 && aa > 1e-9f && bb > 1e-9f && score >= score_floor;
        eligible = __shfl_sync(0xffffffff, eligible, 0);
        if (!eligible) continue;
        float full_dot = 0.f, full_aa = 0.f;
        if (channel >= 0) {
            const float weight = noise_weight[channel];
#if defined(SESSION_INT16) && defined(SESSION_INTEGER_DOT)
            long long full_dot_q = dot_q, full_aa_q = aa_q;
            // Include the remaining waveform taps and masked contacts in the full fit.
            #pragma unroll 1
            for (int dt = -SESSION_HALF_NT; dt < begin; ++dt) {
                const int a = model[(SESSION_HALF_NT + dt) * width + lane];
                const int b = residual[(long long)(t0 + dt) * n_channels + channel];
                full_dot_q += (long long)a * b; full_aa_q += (long long)a * a;
            }
            #pragma unroll 1
            for (int dt = end + 1; dt <= SESSION_HALF_NT; ++dt) {
                const int a = model[(SESSION_HALF_NT + dt) * width + lane];
                const int b = residual[(long long)(t0 + dt) * n_channels + channel];
                full_dot_q += (long long)a * b; full_aa_q += (long long)a * a;
            }
            const float a_scale = template_scale[unit], b_scale = residual_scale[channel];
            full_dot = (float)full_dot_q * (a_scale * b_scale * weight);
            full_aa = (float)full_aa_q * (a_scale * a_scale * weight);
#else
            // Preserve the original FP32 accumulation order exactly.
            #pragma unroll 1
            for (int dt = -SESSION_HALF_NT; dt <= SESSION_HALF_NT; ++dt) {
                const float a = TV(model[(SESSION_HALF_NT + dt) * width + lane], unit);
                const float b = RV((long long)(t0 + dt) * n_channels + channel, channel);
                full_dot += a * b * weight; full_aa += a * a * weight;
            }
#endif
        }
        full_dot = session_warp_sum(full_dot); full_aa = session_warp_sum(full_aa);
        if (lane == 0 && full_aa > 1e-9f) {
            const float amp = full_dot / full_aa;
            const float gain = 2.f * amp * full_dot - amp * amp * full_aa;
            if (amp >= amplitude_min && amp <= amplitude_max &&
                (gain > best_gain || (gain == best_gain && abs(shift) < abs(best_shift)))) {
                best_gain = gain; best_score = score;
                best_amplitude = amp; best_shift = shift;
            }
        }
    }
    if (lane == 0) {
        scores[p] = best_score; amplitudes[p] = best_amplitude;
        gains[p] = best_gain; shifts[p] = best_shift;
    }
}

// Proposals selected on the host are disjoint on any common contact within
// one waveform span. Atomic subtraction remains safe if a caller relaxes that
// selection policy in the future.
extern "C" __global__ void subtract_fits(
    session_sample_t* __restrict__ residual, const session_sample_t* __restrict__ templates,
    const int* __restrict__ channels, const int* __restrict__ times,
    const int* __restrict__ units, const float* __restrict__ amplitudes,
    int n_fits, int n_samples, int n_channels, int width SESSION_EXTRA)
{
    const int fit = blockIdx.x;
    if (fit >= n_fits) return;
    const int center = times[fit], unit = units[fit];
    const float amplitude = amplitudes[fit];
    for (int k = threadIdx.x; k < SESSION_NT * width; k += blockDim.x) {
        const int dt = k / width - SESSION_HALF_NT;
        const int slot = k % width;
        const int channel = channels[unit * width + slot];
        const int t = center + dt;
        if (channel >= 0 && t >= 0 && t < n_samples) {
#ifdef SESSION_INT16
            const float delta = amplitude * TV(templates[unit * SESSION_NT * width + k], unit)
                                / residual_scale[channel];
            session_subtract_int16(residual + (long long)t * n_channels + channel,
                                   delta, overflow);
#else
            atomicAdd(residual + (long long)t * n_channels + channel,
                      -amplitude * templates[unit * SESSION_NT * width + k]);
#endif
        }
    }
}
