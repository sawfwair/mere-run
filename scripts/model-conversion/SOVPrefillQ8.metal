// Research-only grouped prefill TensorOps; not a runtime resource.
const uint row = jobs[threadgroup_position_in_grid.x];
const uint col = threadgroup_position_in_grid.y * BN;
const uint lane = thread_index_in_simdgroup;
const uint expert = uint(indices[row]);
threadgroup bfloat16_t padded[BM * (BK + 8)];
threadgroup bfloat16_t decoded[BN * (BK + 8)];
threadgroup float input_sum[BM];
auto av = tensor(padded, extents<int, BK, BM>{}, array<int, 2>{1, BK + 8});
auto bv = tensor(decoded, extents<int, BK, BN>{}, array<int, 2>{1, BK + 8});
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<1>> operation;
auto acc = operation.template get_destination_cooperative_tensor<decltype(av), decltype(bv), float>();
float totals[BM * BN / 32] = {0.0f};
for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
for (int block = 0; block < K; block += BK) {
    for (uint i = lane; i < BM * BK; i += 32) {
        const uint r = row + i / BK;
        padded[(i / BK) * (BK + 8) + i % BK] = r < R && uint(indices[r]) == expert
            ? input[r * K + block + i % BK] : bfloat16_t(0);
    }
    for (uint i = lane; i < BN * BK / 4; i += 32) {
        const uint n = col + (i * 4) / BK;
        const uint k = block + (i * 4) % BK;
        const uint packed = weights[(expert * N + n) * (K / 4) + k / 4];
        const uint meta = (expert * N + n) * (K / 64) + k / 64;
        for (uint j = 0; j < 4; ++j) {
            const float q = float((packed >> (8 * j)) & 255);
            decoded[((i * 4) / BK) * (BK + 8) + (i * 4) % BK + j] =
                GroupAffine ? bfloat16_t(q) : bfloat16_t(q * float(scales[meta]) + float(biases[meta]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (GroupAffine) {
        for (uint r = 0; r < BM; ++r) {
            const float sum = simd_sum(float(padded[r * (BK + 8) + lane]) + float(padded[r * (BK + 8) + lane + 32]));
            if (lane == 0) input_sum[r] = sum;
        }
        for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(av, bv, acc);
    if constexpr (GroupAffine) {
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto index = acc.get_multidimensional_index(i);
            const uint meta = (expert * N + col + index[0]) * (K / 64) + block / 64;
            totals[i] += float(acc[i]) * float(scales[meta]) + input_sum[index[1]] * float(biases[meta]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint r = row + index[1];
    if (r < R && uint(indices[r]) == expert)
        output[r * N + col + index[0]] = bfloat16_t(GroupAffine ? totals[i] : float(acc[i]));
}
