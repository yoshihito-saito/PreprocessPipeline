#ifndef SESSION_NT
#define SESSION_NT 61
#endif
#define SESSION_HALF_NT (SESSION_NT / 2)

// Experimental joint-pair scorer. Compiled separately and only on first use.
// Scaled INT16 arrays are expanded only in registers; no FP32 core allocation.
#ifdef SESSION_INT16
typedef short joint_sample_t;
#define JOINT_EXTRA , const float* residual_scale, const float* template_scale, const int* overflow
#define JOINT_PASS , residual_scale, template_scale, overflow
#define JOINT_R(index, contact) (float(residual[index]) * residual_scale[contact])
#define JOINT_T(value, unit) (float(value) * template_scale[unit])
#else
typedef float joint_sample_t;
#define JOINT_EXTRA
#define JOINT_PASS
#define JOINT_R(index, contact) residual[index]
#define JOINT_T(value, unit) value
#endif
__device__ __forceinline__ float joint_warp_sum(float value) {
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffff, value, delta);
    return value;
}

// One warp fits a pair of local templates on their union of physical
// contacts. Each lane handles its A contact and any B-only contact in the
// same slot. Shared contacts are visited once. Accumulators stay in registers;
// five warp reductions produce the two-by-two weighted normal equations.
__device__ __forceinline__ void joint_channel_terms(
    const joint_sample_t* residual, const joint_sample_t* templates, int n_channels, int width,
    int center_a, int unit_a, int center_b, int unit_b,
    int channel, int slot_a, int slot_b, int first, int stop, float weight,
    float* gaa, float* gab, float* gbb, float* dot_a, float* dot_b JOINT_EXTRA)
{
    const joint_sample_t* model_a = templates + unit_a * SESSION_NT * width;
    const joint_sample_t* model_b = templates + unit_b * SESSION_NT * width;
    for (int t = first; t < stop; ++t) {
        const float a = (slot_a >= 0 && t >= center_a - SESSION_HALF_NT && t <= center_a + SESSION_HALF_NT)
            ? JOINT_T(model_a[(t - center_a + SESSION_HALF_NT) * width + slot_a], unit_a) : 0.f;
        const float b = (slot_b >= 0 && t >= center_b - SESSION_HALF_NT && t <= center_b + SESSION_HALF_NT)
            ? JOINT_T(model_b[(t - center_b + SESSION_HALF_NT) * width + slot_b], unit_b) : 0.f;
        const float observed = JOINT_R((long long)t * n_channels + channel, channel);
        *gaa += a * a * weight;
        *gab += a * b * weight;
        *gbb += b * b * weight;
        *dot_a += a * observed * weight;
        *dot_b += b * observed * weight;
    }
}

__device__ __forceinline__ float joint_gain(
    float a, float b, float gaa, float gab, float gbb, float dot_a, float dot_b)
{
    return 2.f * a * dot_a + 2.f * b * dot_b - gaa * a * a
        - 2.f * gab * a * b - gbb * b * b;
}

__device__ __forceinline__ void joint_consider(
    float a, float b, float gaa, float gab, float gbb, float dot_a, float dot_b,
    float* best_gain, float* best_a, float* best_b)
{
    const float gain = joint_gain(a, b, gaa, gab, gbb, dot_a, dot_b);
    if (gain > *best_gain) {
        *best_gain = gain;
        *best_a = a;
        *best_b = b;
    }
}

extern "C" __global__ void score_joint_template_pairs(
    const joint_sample_t* __restrict__ residual,
    const joint_sample_t* __restrict__ templates, const int* __restrict__ channels,
    const int* __restrict__ centers_a, const int* __restrict__ units_a,
    const int* __restrict__ centers_b, const int* __restrict__ units_b,
    const float* __restrict__ noise_weight,
    const float* __restrict__ blocker_baseline_amplitude,
    int n_pairs, int n_samples, int n_channels, int width,
    float amplitude_min, float amplitude_max,
    float minimum_incremental_gain, float minimum_incremental_fraction,
    int min_center_separation, float max_template_correlation,
    float* __restrict__ out_a, float* __restrict__ out_b,
    float* __restrict__ out_gain, float* __restrict__ out_increment,
    float* __restrict__ out_correlation JOINT_EXTRA)
{
    const int p = blockIdx.x;
    const int lane = threadIdx.x;
    if (p >= n_pairs) return;
    const int ta = centers_a[p], ua = units_a[p];
    const int tb = centers_b[p], ub = units_b[p];
    const int separation = abs(ta - tb);
    if (ta < SESSION_HALF_NT || tb < SESSION_HALF_NT || ta + SESSION_HALF_NT >= n_samples || tb + SESSION_HALF_NT >= n_samples
            || separation < min_center_separation || separation > (SESSION_NT - 1)) {
        if (lane == 0) {
            out_a[p] = 0.f; out_b[p] = 0.f; out_gain[p] = 0.f;
            out_increment[p] = 0.f; out_correlation[p] = 2.f;
        }
        return;
    }
    const int first = min(ta, tb) - SESSION_HALF_NT;
    const int stop = max(ta, tb) + (SESSION_HALF_NT + 1);
    float gaa = 0.f, gab = 0.f, gbb = 0.f, dot_a = 0.f, dot_b = 0.f;
    if (lane < width) {
        const int ca = channels[ua * width + lane];
        if (ca >= 0) {
            int slot_b = -1;
            for (int k = 0; k < width; ++k)
                if (channels[ub * width + k] == ca) { slot_b = k; break; }
            joint_channel_terms(residual, templates, n_channels, width,
                ta, ua, tb, ub, ca, lane, slot_b, first, stop, noise_weight[ca],
                &gaa, &gab, &gbb, &dot_a, &dot_b JOINT_PASS);
        }
        const int cb = channels[ub * width + lane];
        if (cb >= 0) {
            bool already_visited = false;
            for (int k = 0; k < width; ++k)
                if (channels[ua * width + k] == cb) { already_visited = true; break; }
            if (!already_visited)
                joint_channel_terms(residual, templates, n_channels, width,
                    ta, ua, tb, ub, cb, -1, lane, first, stop, noise_weight[cb],
                    &gaa, &gab, &gbb, &dot_a, &dot_b JOINT_PASS);
        }
    }
    gaa = joint_warp_sum(gaa);
    gab = joint_warp_sum(gab);
    gbb = joint_warp_sum(gbb);
    dot_a = joint_warp_sum(dot_a);
    dot_b = joint_warp_sum(dot_b);
    if (lane != 0) return;
    // The shared input may be the residual after *all* accepted templates
    // were subtracted. Restore this pair's accepted blocker analytically so
    // joint and single fits compare on the same conditional local patch.
    const float baseline = blocker_baseline_amplitude[p];
    dot_a += baseline * gab;
    dot_b += baseline * gbb;
    float a = 0.f, b = 0.f, best = 0.f, increment = 0.f, corr = 2.f;
    if (gaa > 1e-9f && gbb > 1e-9f) {
        corr = gab * rsqrtf(gaa * gbb);
        if (fabsf(corr) < max_template_correlation) {
            const float determinant = gaa * gbb - gab * gab;
            float best_gain = -1e30f, best_a = 0.f, best_b = 0.f;
            if (determinant > 1e-12f * gaa * gbb) {
                const float x = (dot_a * gbb - dot_b * gab) / determinant;
                const float y = (dot_b * gaa - dot_a * gab) / determinant;
                if (x >= amplitude_min && x <= amplitude_max &&
                    y >= amplitude_min && y <= amplitude_max)
                    joint_consider(x, y, gaa, gab, gbb, dot_a, dot_b,
                                   &best_gain, &best_a, &best_b);
            }
            for (int edge = 0; edge < 2; ++edge) {
                const float x = edge ? amplitude_max : amplitude_min;
                const float y = fminf(amplitude_max,
                    fmaxf(amplitude_min, (dot_b - gab * x) / gbb));
                joint_consider(x, y, gaa, gab, gbb, dot_a, dot_b,
                               &best_gain, &best_a, &best_b);
                const float z = edge ? amplitude_max : amplitude_min;
                const float w = fminf(amplitude_max,
                    fmaxf(amplitude_min, (dot_a - gab * z) / gaa));
                joint_consider(w, z, gaa, gab, gbb, dot_a, dot_b,
                               &best_gain, &best_a, &best_b);
            }
            const float single_a = fminf(amplitude_max,
                fmaxf(amplitude_min, dot_a / gaa));
            const float single_b = fminf(amplitude_max,
                fmaxf(amplitude_min, dot_b / gbb));
            const float best_single = fmaxf(0.f, fmaxf(
                joint_gain(single_a, 0.f, gaa, gab, gbb, dot_a, dot_b),
                joint_gain(0.f, single_b, gaa, gab, gbb, dot_a, dot_b)));
            const float extra = best_gain - best_single;
            if (best_gain > 0.f && extra >= minimum_incremental_gain &&
                    extra / best_gain >= minimum_incremental_fraction) {
                a = best_a; b = best_b; best = best_gain; increment = extra;
            }
        }
    }
    out_a[p] = a; out_b[p] = b; out_gain[p] = best;
    out_increment[p] = increment; out_correlation[p] = corr;
}
