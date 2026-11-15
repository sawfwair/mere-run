// Research kernel: block-scaled dynamic int8 on M5 GPU TensorOps.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#pragma clang fp reassociate(off)

// ROTATE_BODY
const uint group = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
float values[BK/32];
#pragma unroll
for (uint i=0; i<BK/32; ++i) values[i] = float(input[group*BK+i*32+lane]);
#pragma unroll
for (uint distance=1; distance<32; distance*=2) {
    #pragma unroll
    for (uint i=0; i<BK/32; ++i) {
        const float other = simd_shuffle_xor(values[i], ushort(distance));
        values[i] = (lane & distance) ? other-values[i] : values[i]+other;
    }
}
#pragma unroll
for (uint distance=1; distance<BK/32; distance*=2) {
    #pragma unroll
    for (uint i=0; i<BK/32; ++i) {
        if ((i & distance) == 0) {
            const float a=values[i], b=values[i+distance];
            values[i]=a+b; values[i+distance]=a-b;
        }
    }
}
#pragma unroll
for (uint i=0; i<BK/32; ++i) rotated[group*BK+i*32+lane] = values[i]*rsqrt(float(BK));

// FUSED_PACK_BODY
const uint row = threadgroup_position_in_grid.x;
const uint block = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
threadgroup float maxima[32];
threadgroup float shared_scale[1];
float values[BK/32];
#pragma unroll
for (uint i=0; i<BK/32; ++i) values[i] = float(input[row*K+block*BK+i*32+lane]);
if constexpr (Rotate) {
    #pragma unroll
    for (uint distance=1; distance<32; distance*=2) {
        #pragma unroll
        for (uint i=0; i<BK/32; ++i) {
            const float other = simd_shuffle_xor(values[i], ushort(distance));
            values[i] = (lane & distance) ? other-values[i] : values[i]+other;
        }
    }
    #pragma unroll
    for (uint distance=1; distance<BK/32; distance*=2) {
        #pragma unroll
        for (uint i=0; i<BK/32; ++i) if ((i & distance)==0) {
            const float a=values[i], b=values[i+distance];
            values[i]=a+b; values[i+distance]=a-b;
        }
    }
    #pragma unroll
    for (uint i=0; i<BK/32; ++i) values[i]*=rsqrt(float(BK));
}
float maximum=0.0f;
#pragma unroll
for (uint i=0; i<BK/32; ++i) maximum=max(maximum, abs(values[i]));
maximum=simd_max(maximum);
if (lane==0) maxima[block]=maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (block==0) {
    const float row_max=simd_max(lane<K/BK ? maxima[lane] : 0.0f);
    if (lane==0) { shared_scale[0]=max(row_max/127.0f,1e-12f); row_scale[row]=shared_scale[0]; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float scale=shared_scale[0];
const int exponent=maximum>0 ? clamp(int(floor(log2(127.0f*scale/maximum))),0,MaxExponent) : 0;
if (lane==0) exponents[row*(K/BK)+block]=uchar(exponent);
const float inverse=exp2(float(exponent))/scale;
const uint offset=TileMajor ? ((row/64)*(K/BK)+block)*64*BK+(row%64)*BK : row*K+block*BK;
#pragma unroll
for (uint i=0; i<BK/32; ++i) codes[offset+i*32+lane]=char(clamp(int(rint(values[i]*inverse)),-127,127));

// PACK_BODY
const uint group = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint row = group / (K / BK), block = group % (K / BK);
float values[BK / 32];
float maximum = 0.0f;
#pragma unroll
for (uint i = 0; i < BK / 32; ++i) {
    values[i] = float(input[row * K + block * BK + i * 32 + lane]);
    maximum = max(maximum, abs(values[i]));
}
maximum = simd_max(maximum);
const float scale = row_scale[row];
const int exponent = maximum > 0 ? clamp(int(floor(log2(127.0f * scale / maximum))), 0, MaxExponent) : 0;
if (lane == 0) exponents[group] = uchar(exponent);
const float inverse = exp2(float(exponent)) / scale;
#pragma unroll
for (uint i = 0; i < BK / 32; ++i) {
    const int code = clamp(int(rint(values[i] * inverse)), -127, 127);
    const uint offset = TileMajor ? ((row/64)*(K/BK)+block)*64*BK+(row%64)*BK : row*K+block*BK;
    codes[offset + i * 32 + lane] = char(code);
}

// MATMUL_BODY
const uint tile = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint base_row = (tile / (N / 64)) * 64 + (sg / 2) * 32;
const uint base_col = (tile % (N / 64)) * 64 + (sg % 2) * 32;
auto av = tensor(const_cast<device int8_t*>(reinterpret_cast<const device int8_t*>(aq)),
    extents<int, BK, 32>{}, array<int, 2>{1, TileMajor ? BK : K});
auto bv = tensor(const_cast<device int8_t*>(reinterpret_cast<const device int8_t*>(bq)),
    extents<int, BK, 32>{}, array<int, 2>{1, TileMajor ? BK : K});
constexpr auto descriptor = matmul2d_descriptor(32, 32, BK, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<descriptor, execution_simdgroups<1>> operation;
auto acc = operation.template get_destination_cooperative_tensor<decltype(av), decltype(bv), int>();
float totals[32] = {0.0f};
int previous[32] = {0};
#pragma unroll
for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
for (int block = 0; block < K / BK; ++block) {
    if constexpr (MaxExponent > 0) {
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto index = acc.get_multidimensional_index(i);
        const int exponent = int(a_exp[(base_row+index[1])*(K/BK)+block])
            + int(b_exp[(base_col+index[0])*(K/BK)+block]);
        if constexpr (IntegerAccumulation) {
            const int change = exponent - previous[i];
            if (change > 0) acc[i] = as_type<int>(as_type<uint>(int(acc[i])) << change);
            else if (change < 0) acc[i] = (acc[i] + (1 << (-change-1))) >> -change;
        } else acc[i] = 0;
        previous[i] = exponent;
    }
    } else if constexpr (!IntegerAccumulation) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    }
    const uint a_offset = TileMajor ? ((base_row/64)*(K/BK)+block)*64*BK+(base_row%64)*BK : base_row*K+block*BK;
    const uint b_offset = TileMajor ? ((base_col/64)*(K/BK)+block)*64*BK+(base_col%64)*BK : base_col*K+block*BK;
    auto at = tensor(const_cast<device int8_t*>(reinterpret_cast<const device int8_t*>(aq))+a_offset,
        extents<int, BK, 32>{}, array<int, 2>{1, TileMajor ? BK : K});
    auto bt = tensor(const_cast<device int8_t*>(reinterpret_cast<const device int8_t*>(bq))+b_offset,
        extents<int, BK, 32>{}, array<int, 2>{1, TileMajor ? BK : K});
    operation.run(at, bt, acc);
    if constexpr (!IntegerAccumulation) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) totals[i] += float(acc[i]) * exp2(-float(previous[i]));
        }
    }
}
#pragma unroll
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint row = base_row + index[1], col = base_col + index[0];
    float value;
    if constexpr (IntegerAccumulation && MaxExponent == 0) value = float(acc[i]);
    else if constexpr (IntegerAccumulation) value = float(acc[i]) * exp2(-float(previous[i]));
    else value = totals[i];
    output[row*N+col] = Out(value * a_scale[row] * b_scale[col] + float(bias[col]));
}
