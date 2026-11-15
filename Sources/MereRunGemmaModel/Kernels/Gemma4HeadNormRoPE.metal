// RMS reduction and staged BF16 scaling adapted from MLX, Copyright 2024 Apple Inc.
// MIT license: see the MLX runtime bundle entry in THIRD_PARTY_NOTICES.md.
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint row = threadgroup_position_in_grid.y;
const uint head = row % (QH + 2 * KH);
const uint token = (row / (QH + 2 * KH)) % T;
const uint batch = row / ((QH + 2 * KH) * T);
const bool query = head < QH;
const bool key = head >= QH && head < QH + KH;
const uint h = query ? head : (head - QH) % KH;
const device bfloat16_t* source = query ? q_raw + ((batch * T + token) * QH + h) * D
    : (key ? k_raw : v_raw) + ((batch * T + token) * KH + h) * D;
threadgroup float sums[32];
threadgroup float inv_mean[1];
threadgroup bfloat16_t normalized[D];
float values[4];
float acc = 0.0f;
for (uint i = 0; i < 4; ++i) {
    values[i] = float(source[lid * 4 + i]);
    acc += values[i] * values[i];
}
acc = simd_sum(acc);
if (sg == 0) sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    acc = simd_sum(sums[lane]);
    if (lane == 0) inv_mean[0] = metal::precise::rsqrt(acc / float(D) + eps[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0; i < 4; ++i) {
    uint d = lid * 4 + i;
    bfloat16_t n = bfloat16_t(values[i] * inv_mean[0]);
    if (query) n = bfloat16_t(float(q_weight[d]) * float(n));
    else if (key) n = bfloat16_t(float(k_weight[d]) * float(n));
    normalized[d] = n;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0; i < 4; ++i) {
    uint d = lid * 4 + i;
    bfloat16_t value = normalized[d];
    if (query || key) {
        uint pair = d % (D / 2);
        if (pair < RotHalf) {
            float inv_freq;
            if (Proportional) inv_freq = 1.0f / freqs[pair];
            else inv_freq = metal::exp2(-float(pair) / float(D / 2) * log_base[0]);
            float theta = float(offset[0] + token) * inv_freq;
            float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
            float a = float(normalized[pair]), b = float(normalized[pair + D / 2]);
            value = bfloat16_t(d < D / 2 ? a * c - b * s : a * s + b * c);
        }
        if (query) q_out[((batch * QH + h) * T + token) * D + d] = value;
        else k_out[((batch * KH + h) * T + token) * D + d] = value;
    } else v_out[((batch * KH + h) * T + token) * D + d] = value;
}
