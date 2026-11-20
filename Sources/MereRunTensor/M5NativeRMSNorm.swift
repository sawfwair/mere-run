#if os(macOS)
import Foundation
import Metal

/// Matches MLX's BF16 5120-wide looped RMSNorm reduction. The caller owns
/// command ordering and buffer lifetimes; no submission or wait occurs here.
@available(macOS 26.0, *)
package final class M5NativeRMSNorm {
    enum Failure: Error { case missingFunction, threadgroupGeometry }
    private let pipeline: any MTLComputePipelineState

    package init(device: any MTLDevice, smallProjectionSums: Bool = false) throws {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version4_0
        let specialization = "#define MERE_QMV_SUMS " + (smallProjectionSums ? "1" : "0") + "\n"
        let library = try device.makeLibrary(source: specialization + Self.source, options: options)
        guard let function = library.makeFunction(name: "mere_native_residual_rms_sums") else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
        // Changing this geometry changes the floating-point reduction tree.
        guard pipeline.maxTotalThreadsPerThreadgroup >= 1_024 else { throw Failure.threadgroupGeometry }
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                residual: (any MTLBuffer)?, weight: any MTLBuffer, combined: any MTLBuffer,
                normalized: any MTLBuffer, sums: any MTLBuffer, rows: Int, epsilon: Float, inputOffset: Int = 0, weightOffset: Int = 0,
                combinedOffset: Int = 0, normalizedOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0)
        var reads: [M5NativeHazards.Access] = [.init(input, offset: inputOffset, count: rows * 5120 * 2)]
        if let residual { reads.append(.init(residual, count: rows * 5120 * 2)) }
        hazards?.prepare(encoder: encoder, reads: reads,
            writes: [.init(combined, offset: combinedOffset, count: rows * 5120 * 2),
                     .init(normalized, offset: normalizedOffset, count: rows * 5120 * 2), .init(sums)])
        encoder.setComputePipelineState(pipeline)
        let offsets = [inputOffset, 0, weightOffset, combinedOffset, normalizedOffset, 0]
        for (index, buffer) in [input, residual ?? input, weight, combined, normalized, sums].enumerated() {
            encoder.setBuffer(buffer, offset: offsets[index], index: index)
        }
        var epsilon = epsilon, count = Int32(rows), includeResidual = UInt32(residual == nil ? 0 : 1)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 7)
        encoder.setBytes(&includeResidual, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.dispatchThreadgroups(MTLSize(width: ((rows + 15) / 16) * 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }

    // Reduction and BF16 output boundaries follow Apple's MIT-licensed
    // rms_looped kernel (mlx/backend/metal/kernels/rms_norm.metal).
    // Producer sums keep the current group-affine serial 64-value sum order.
    static let source = """
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp reassociate(off)
        kernel void mere_native_residual_rms_sums(
            const device bfloat* x [[buffer(0)]], const device bfloat* residual [[buffer(1)]],
            const device bfloat* weight [[buffer(2)]], device bfloat* combined [[buffer(3)]],
            device bfloat* normalized [[buffer(4)]], device float* sums [[buffer(5)]],
            constant float& eps [[buffer(6)]], constant int& rows [[buffer(7)]],
            constant uint& includeResidual [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
            uint lid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            constexpr uint K = 5120, Reads = 4, Threads = 1024;
            if (row >= uint(rows)) {
                #if MERE_QMV_SUMS
                if (lid < K / 16) sums[((row / 4) * (K / 16) + lid) * 4 + row % 4] = 0.0f;
                #else
                if (lid < K / 64) sums[lid * (((rows + 15) / 16) * 16) + row] = 0.0f;
                #endif
                return;
            }
            threadgroup float local_sums[32];
            threadgroup float inverse[1];
            threadgroup bfloat values[K];
            float acc = 0.0f;
            for (uint base = 0; base < K; base += Threads * Reads) {
                for (uint i = 0; i < Reads; ++i) {
                    const uint column = base + lid * Reads + i;
                    if (column < K) {
                        const ulong at = ulong(row) * K + column;
                        const bfloat value = includeResidual ? bfloat(x[at] + residual[at]) : x[at];
                        const float xi = float(value);
                        acc += xi * xi;
                    }
                }
            }
            acc = simd_sum(acc);
            if (simd == 0) local_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[simd] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                acc = simd_sum(local_sums[lane]);
                if (lane == 0) inverse[0] = metal::precise::rsqrt(acc / K + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint base = 0; base < K; base += Threads * Reads) {
                for (uint i = 0; i < Reads; ++i) {
                    const uint column = base + lid * Reads + i;
                    if (column < K) {
                        const ulong at = ulong(row) * K + column;
                        const bfloat value = includeResidual ? bfloat(x[at] + residual[at]) : x[at];
                        const bfloat result = weight[column] * bfloat(float(value) * inverse[0]);
                        combined[at] = value;
                        normalized[at] = result;
                        values[column] = result;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #if MERE_QMV_SUMS
            if (lid < K / 16) {
                float sum = 0.0f;
                for (uint step = 0; step < 16; step += 4) {
                    const uint i = 16 * lid + step;
                    sum += float(values[i]) + float(values[i + 1]) + float(values[i + 2]) + float(values[i + 3]);
                }
                sums[((row / 4) * (K / 16) + lid) * 4 + row % 4] = sum;
            }
            #else
            if (lid < K / 64) {
                float sum = 0.0f;
                for (uint i = 0; i < 64; ++i) sum += float(values[lid * 64 + i]);
                sums[lid * (((rows + 15) / 16) * 16) + row] = sum;
            }
            #endif
        }
        """
}
#endif
