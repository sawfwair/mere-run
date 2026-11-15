// Affine Q8 GEMV with a distributed 16-output reduction and stores.
// Automatic on qualified M5 Max GPUs; other devices require an explicit override.
static_assert(Rows == 2 && O == 8, "Requires sixteen outputs.");
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
float values[Rows*O];
for(uint r=0;r<Rows;++r) for(uint o=0;o<O;++o) values[r*O+o]=result[r][o];
float next16[8];
for(uint i=0;i<8;++i) {
 const float a=(lane&1) ? values[i+8] : values[i];
 const float b=(lane&1) ? values[i] : values[i+8];
 next16[i]=a+simd_shuffle_xor(b,1);
}
for(uint i=0;i<8;++i) values[i]=next16[i];
float next8[4];
for(uint i=0;i<4;++i) {
 const float a=(lane&2) ? values[i+4] : values[i];
 const float b=(lane&2) ? values[i] : values[i+4];
 next8[i]=a+simd_shuffle_xor(b,2);
}
for(uint i=0;i<4;++i) values[i]=next8[i];
float next4[2];
for(uint i=0;i<2;++i) {
 const float a=(lane&4) ? values[i+2] : values[i];
 const float b=(lane&4) ? values[i] : values[i+2];
 next4[i]=a+simd_shuffle_xor(b,4);
}
for(uint i=0;i<2;++i) values[i]=next4[i];
float next2[1];
for(uint i=0;i<1;++i) {
 const float a=(lane&8) ? values[i+1] : values[i];
 const float b=(lane&8) ? values[i] : values[i+1];
 next2[i]=a+simd_shuffle_xor(b,8);
}
for(uint i=0;i<1;++i) values[i]=next2[i];
values[0]+=simd_shuffle_xor(values[0],16);
// The first four shuffle choices assign outputs in reversed lane-bit order.
const uint output_index = ((lane&1)!=0)*8 + ((lane&2)!=0)*4 + ((lane&4)!=0)*2 + ((lane&8)!=0);
const uint row=output_index/O, col=output_index%O;
if((lane&16)==0 && r0+row<R && uint(indices[r0+row])==expert)
 output[(r0+row)*N+n0+col]=bfloat16_t(values[0]);
