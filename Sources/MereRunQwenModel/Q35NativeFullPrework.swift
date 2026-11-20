#if os(macOS)
import Foundation
import Metal
import MereRunTensor

/// Qwen 27B query/key RMSNorm and partial RoPE, preserving both BF16 boundaries.
@available(macOS 26.0, *)
final class Q35NativeFullPrework {
    enum Failure: Error { case missingFunction }
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        guard let function = library.makeFunction(name: "q35_native_full_prework") else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(into encoder: any MTLComputeCommandEncoder, qkv: any MTLBuffer,
                queryWeight: M5NativeArrayBuffer, keyWeight: M5NativeArrayBuffer,
                queries: any MTLBuffer, keys: any MTLBuffer, values: any MTLBuffer,
                plan: Q35NativeTreePlan, prefix: Int, epsilon: Float, keysOffset: Int = 0, valuesOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition(queryWeight.shape == [256] && keyWeight.shape == [256]
            && queryWeight.dtype == .bfloat16 && keyWeight.dtype == .bfloat16)
        precondition(prefix >= 0 && prefix <= Int(Int32.max) - plan.count)
        hazards?.prepare(encoder: encoder, reads: [.init(qkv, count: plan.count * 14336 * 2)],
            writes: [.init(queries, count: plan.count * 6144 * 2),
                     .init(keys, offset: keysOffset, count: 4 * plan.branchRows * 256 * 2), .init(values, offset: valuesOffset, count: 4 * plan.branchRows * 256 * 2)])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(queryWeight.buffer, offset: queryWeight.offset, index: 1)
        encoder.setBuffer(keyWeight.buffer, offset: keyWeight.offset, index: 2)
        let offsets = [0, keysOffset, valuesOffset, 0]
        for (index, buffer) in [queries, keys, values, plan.depths].enumerated() {
            encoder.setBuffer(buffer, offset: offsets[index], index: index + 3)
        }
        var prefix = Int32(prefix), epsilon = epsilon, base = log2(Float(10_000_000))
        encoder.setBytes(&prefix, length: 4, index: 7)
        encoder.setBytes(&epsilon, length: 4, index: 8)
        encoder.setBytes(&base, length: 4, index: 9)
        var branchRows = Int32(plan.branchRows)
        encoder.setBytes(&branchRows, length: 4, index: 10)
        encoder.dispatchThreadgroups(MTLSize(width: 28, height: plan.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }

    // Reduction and rotary arithmetic follow Apple's MIT-licensed rms_norm.metal
    // and rope.metal. Shared storage preserves the intermediate BF16 rounding.
    static let source = """
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp reassociate(off)
        kernel void q35_native_full_prework(const device bfloat* qkv [[buffer(0)]],
            const device bfloat* qw [[buffer(1)]], const device bfloat* kw [[buffer(2)]],
            device bfloat* queries [[buffer(3)]], device bfloat* keys [[buffer(4)]], device bfloat* values [[buffer(5)]],
            const device int* depths [[buffer(6)]], constant int& prefix [[buffer(7)]], constant float& eps [[buffer(8)]],
            constant float& base [[buffer(9)]], constant int& branchRows [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
            uint lid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            const uint head = group.x, row = group.y;
            const bool query = head < 24;
            const uint physical = query ? head : head - 24;
            const device bfloat* x = qkv + row * 14336 + (query ? head * 512 : 12288 + physical * 256);
            const device bfloat* w = query ? qw : kw;
            device bfloat* out = query ? queries + (row * 24 + head) * 256 : keys + (physical * branchRows + row) * 256;
            threadgroup float local_sums[32], inverse[1];
            threadgroup bfloat normalized[256];
            float xi[4], acc = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                xi[i] = float(x[lid * 4 + i]);
                acc += xi[i] * xi[i];
            }
            acc = simd_sum(acc);
            if (simd == 0) local_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[simd] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                acc = simd_sum(local_sums[lane]);
                if (lane == 0) inverse[0] = metal::precise::rsqrt(acc / 256.0f + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4; ++i) {
                const uint column = lid * 4 + i;
                normalized[column] = w[column] * bfloat(xi[i] * inverse[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < 32) {
                const float d = float(lid) / 32.0f;
                const float inv_freq = metal::exp2(-d * base);
                const float theta = float(prefix + depths[row]) * inv_freq;
                const float costheta = metal::fast::cos(theta), sintheta = metal::fast::sin(theta);
                const float x1 = float(normalized[lid]), x2 = float(normalized[lid + 32]);
                out[lid] = bfloat(x1 * costheta - x2 * sintheta);
                out[lid + 32] = bfloat(x1 * sintheta + x2 * costheta);
            }
            for (uint column = 64 + lid; column < 256; column += 64) out[column] = normalized[column];
            if (!query) {
                for (uint i = 0; i < 4; ++i) {
                    const uint column = lid * 4 + i;
                    values[(physical * branchRows + row) * 256 + column] = qkv[row * 14336 + 13312 + physical * 256 + column];
                }
            }
        }
        """
}
#endif
