#if os(macOS)
import Foundation
import Metal
import MereRunTensor

/// Readback for checkpoint qualification. Each GPU group reduces one vocabulary
/// tile. The host combines its stable partial sums in FP64.
@available(macOS 26.0, *)
final class Q35NativeTeacherHead {
    private static let chunk = 2_048
    private let vocabulary: Int
    private let chunks: Int
    private let pipeline: any MTLComputePipelineState
    private let partials: any MTLBuffer

    init(device: any MTLDevice, vocabulary: Int) throws {
        let chunks = (vocabulary + Self.chunk - 1) / Self.chunk
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        guard let function = library.makeFunction(name: "q35_teacher_logsumexp"),
              let partials = device.makeBuffer(length: (chunks * 2 + 1) * 4, options: .storageModeShared)
        else { throw Q35NativeVerification.Failure.allocation }
        self.vocabulary = vocabulary
        self.chunks = chunks
        self.partials = partials
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(logits: any MTLBuffer, label: Int, encoder: any MTLComputeCommandEncoder, hazards: M5NativeHazards) {
        precondition((0..<vocabulary).contains(label))
        hazards.prepare(encoder: encoder, reads: [.init(logits, count: vocabulary * 2)], writes: [.init(partials)])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(partials, offset: 0, index: 1)
        [UInt32(label), UInt32(vocabulary), UInt32(chunks), UInt32(Self.chunk)].withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
        }
        encoder.dispatchThreadgroups(MTLSize(width: chunks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func loss() -> Double {
        let values = partials.contents().assumingMemoryBound(to: Float.self)
        let maximum = (0..<chunks).map { Double(values[2 * $0]) }.max()!
        let sum = (0..<chunks).reduce(0.0) { previous, i in
            previous + Double(values[2 * i + 1]) * exp(Double(values[2 * i]) - maximum)
        }
        return maximum + log(sum) - Double(values[2 * chunks])
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void q35_teacher_logsumexp(
            const device bfloat* logits [[buffer(0)]], device float* partials [[buffer(1)]],
            constant uint4& config [[buffer(2)]], uint tile [[threadgroup_position_in_grid]],
            uint id [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            const uint first = tile * config.w, end = min(first + config.w, config.y);
            threadgroup float groupMax[8], groupSum[8], maximum[1];
            float best = -INFINITY;
            for (uint at = first + id; at < end; at += 256) best = max(best, float(logits[at]));
            best = simd_max(best);
            if (lane == 0) groupMax[simd] = best;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                best = simd_max(lane < 8 ? groupMax[lane] : -INFINITY);
                if (lane == 0) maximum[0] = best;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sum = 0;
            for (uint at = first + id; at < end; at += 256)
                sum += metal::precise::exp(float(logits[at]) - maximum[0]);
            sum = simd_sum(sum);
            if (lane == 0) groupSum[simd] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                sum = simd_sum(lane < 8 ? groupSum[lane] : 0);
                if (lane == 0) {
                    partials[tile * 2] = maximum[0];
                    partials[tile * 2 + 1] = sum;
                    if (config.x >= first && config.x < end)
                        partials[config.z * 2] = float(logits[config.x]);
                }
            }
        }
        """
}
#endif
