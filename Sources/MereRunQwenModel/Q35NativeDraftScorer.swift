#if os(macOS)
import MereRunTensor
import Metal
import MLX
import MLXNN

@available(macOS 26.0, *)
final class Q35NativeDraftScorer {
    private static let inputSize = 5120
    private static let rank = 256
    private static let maxRows = 16
    private static let count = M5NativeTopK.count
    private let projection: M5NativeSmallProjection
    private let weights: M5NativeSmallProjection.Weights
    private let predecessors: M5NativeArrayBuffer
    private let successors: M5NativeArrayBuffer
    private let input: any MTLBuffer
    private let projected: any MTLBuffer
    private let scores: any MTLBuffer
    private let score: any MTLComputePipelineState

    private enum Failure: Error { case missingFunction }

    init?(projection: Linear, predecessors: MLXArray, successors: MLXArray, device: any MTLDevice) throws {
        guard let weights = try PortableQuantizedLinear.nativeSmallProjectionWeights(projection, device: device),
              weights.inputSize == Self.inputSize, weights.outputSize == Self.rank,
              predecessors.ndim == 2, predecessors.dim(1) == Self.rank, predecessors.shape == successors.shape,
              predecessors.dtype == .bfloat16, successors.dtype == .bfloat16,
              let input = device.makeBuffer(length: Self.maxRows * Self.inputSize * 2, options: .storageModePrivate),
              let projected = device.makeBuffer(length: Self.maxRows * Self.rank * 2, options: .storageModePrivate),
              let scores = device.makeBuffer(length: Self.maxRows * Self.count * Self.count * 4, options: .storageModeShared)
        else { return nil }
        self.weights = weights
        self.predecessors = try M5NativeArrayBuffer(contiguous(predecessors), device: device)
        self.successors = try M5NativeArrayBuffer(contiguous(successors), device: device)
        self.input = input
        self.projected = projected
        self.scores = scores
        self.projection = try M5NativeSmallProjection(device: device, inputSize: Self.inputSize, outputSize: Self.rank)
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        guard let function = library.makeFunction(name: "qwen_draft_tree_scores") else { throw Failure.missingFunction }
        score = try device.makeComputePipelineState(function: function)
    }

    /// The serial-arithmetic rank projection reads even row pairs. Pad the
    /// last pair here and leave the caller's sliced MLX storage untouched.
    func stageInput(_ hidden: M5NativeArrayBuffer, on command: any MTLCommandBuffer, rows: Int) -> Bool {
        guard let blit = command.makeBlitCommandEncoder() else { return false }
        let byteCount = rows * Self.inputSize * 2
        blit.copy(from: hidden.buffer, sourceOffset: hidden.offset, to: input, destinationOffset: 0, size: byteCount)
        if !rows.isMultiple(of: 2) {
            blit.fill(buffer: input, range: byteCount..<(byteCount + Self.inputSize * 2), value: 0)
        }
        blit.endEncoding()
        return true
    }

    func encode(into encoder: any MTLComputeCommandEncoder, topK: M5NativeTopK, anchor: Int, rows: Int,
                nativeInput: (any MTLBuffer)? = nil, calibration: DFlash2TreeCalibration = .reference) {
        projection.encode(into: encoder, input: nativeInput ?? input, output: projected, weights: weights, rows: rows)
        encoder.setComputePipelineState(score)
        encoder.setBuffer(topK.output, offset: 0, index: 0)
        encoder.setBuffer(projected, offset: 0, index: 1)
        encoder.setBuffer(predecessors.buffer, offset: predecessors.offset, index: 2)
        encoder.setBuffer(successors.buffer, offset: successors.offset, index: 3)
        encoder.setBuffer(scores, offset: 0, index: 4)
        var token = UInt32(anchor)
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 5)
        var parameters = SIMD2<Float>(calibration.edgeWeight, calibration.temperature)
        encoder.setBytes(&parameters, length: MemoryLayout<SIMD2<Float>>.stride, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: Self.count, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.count * 32, height: 1, depth: 1))
    }

    func read(rows: Int) -> [Float]? {
        let count = rows * Self.count * Self.count
        let values = UnsafeBufferPointer(start: scores.contents().assumingMemoryBound(to: Float.self), count: count)
        guard values.allSatisfy(\.isFinite) else { return nil }
        return Array(values)
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void qwen_draft_tree_scores(
            const device uint* shortlist [[buffer(0)]], const device bfloat* hidden [[buffer(1)]],
            const device bfloat* predecessors [[buffer(2)]], const device bfloat* successors [[buffer(3)]],
            device float* scores [[buffer(4)]], constant uint& anchor [[buffer(5)]], constant float2& calibration [[buffer(6)]],
            uint2 group [[threadgroup_position_in_grid]], uint child [[simdgroup_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]]) {
            constexpr uint Rank = \(rank), K = \(count), Record = \(M5NativeTopK.recordWords);
            const uint depth = group.y, parent = group.x;
            const device uint* current = shortlist + depth * Record;
            const device uint* previous = depth == 0 ? current : current - Record;
            if (current[0] || current[1] != K || previous[0] || previous[1] != K) return;
            const uint parentToken = depth == 0 ? anchor : previous[2 + parent * 2];
            const uint childToken = current[2 + child * 2];
            const float unary = as_type<float>(current[3 + child * 2]);

            float partial = 0.0f;
            for (uint position = lane; position < Rank; position += 32) {
                const float left = float(predecessors[parentToken * Rank + position]) * float(hidden[depth * Rank + position]);
                const float right = float(successors[childToken * Rank + position]);
                partial += calibration.x == 1.0f ? float(bfloat(float(bfloat(left)) * right)) : left * right;
            }
            const float edge = simd_sum(partial);
            threadgroup float conditional[K];
            if (lane == 0) conditional[child] = calibration.x == 1.0f && calibration.y == 1.0f
                ? float(bfloat(float(bfloat(edge)) + unary)) : (unary + calibration.x * edge) / calibration.y;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (child == 0) {
                const float value = lane < K ? conditional[lane] : -INFINITY;
                const float maximum = simd_max(value);
                const float denominator = simd_sum(exp(value - maximum));
                if (lane < K) scores[(depth * K + parent) * K + lane] = value - (maximum + log(denominator));
            }
        }
        """
}
#endif
