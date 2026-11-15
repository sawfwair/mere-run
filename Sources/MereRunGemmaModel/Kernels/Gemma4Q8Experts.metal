// Affine Q8 GEMV with two sorted routes sharing each weight load.
// FP32 partial sums follow MLX's qmv_fast/qmv reduction order.
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint r0 = RouteFirst ? threadgroup_position_in_grid.x : threadgroup_position_in_grid.y;
const uint expert = uint(indices[r0]);
uint start = r0;
while (start > 0 && uint(indices[start - 1]) == expert) --start;
if ((r0 - start) % Rows != 0) return;
const uint column = RouteFirst ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x;
const uint n0 = column * SG * O + sg * O;
float result[Rows][O] = {};
for (uint base = 0; base < K; base += 32 * VPT) {
    const uint k = base + lane * VPT;
    if (k >= K) continue;
    float x[Rows][VPT], sum[Rows] = {};
    for (uint r = 0; r < Rows; ++r) {
        const bool valid = r0 + r < R && uint(indices[r0 + r]) == expert;
        for (uint j = 0; j < VPT; ++j) {
            x[r][j] = valid ? float(input[(r0 + r) * K + k + j]) : 0.0f;
            sum[r] += x[r][j];
        }
    }
    for (uint o = 0; o < O; ++o) {
        const uint n = n0 + o;
        const uint meta = (expert * N + n) * (K / 64) + k / 64;
        const float s = float(scales[meta]), b = float(biases[meta]);
        uint packs[VPT/4];
        uchar4 bytes[VPT/4];
        for (uint j = 0; j < VPT/4; ++j) packs[j] = weights[(expert * N + n) * (K/4) + k/4 + j];
        if (ByteLoads) {
            for (uint j = 0; j < VPT/4; ++j) bytes[j] = as_type<uchar4>(packs[j]);
        }
        for (uint r = 0; r < Rows; ++r) {
            float dot = 0.0f;
            for (uint j = 0; j < VPT; ++j) {
                const float q = ByteLoads ? float(bytes[j/4][j%4]) : float((packs[j/4] >> (8*(j%4))) & 255);
                dot += x[r][j] * q;
            }
            result[r][o] += s * dot + sum[r] * b;
        }
    }
}
for (uint r = 0; r < Rows; ++r) {
    for (uint o = 0; o < O; ++o) {
        const float value = simd_sum(result[r][o]);
        if (lane == 0 && r0 + r < R && uint(indices[r0 + r]) == expert)
            output[(r0 + r) * N + n0 + o] = bfloat16_t(value);
    }
}
