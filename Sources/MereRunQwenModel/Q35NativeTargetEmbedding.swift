#if os(macOS)
import Metal
import MereRunTensor
import MLX
import MLXNN

@available(macOS 26.0, *)
final class Q35NativeTargetEmbedding {
    private let words: M5NativeArrayBuffer
    private let scales: M5NativeArrayBuffer
    private let biases: M5NativeArrayBuffer
    private let pipeline: any MTLComputePipelineState

    init?(embedding: Embedding, device: any MTLDevice) throws {
        guard let embedding = embedding as? PreQuantizedEmbedding,
              embedding.bits == 4, embedding.groupSize == 64, embedding.mode == .affine,
              embedding.weight.ndim == 2, embedding.weight.dim(1) == 640, embedding.weight.dtype == .uint32,
              embedding.scales.shape == [embedding.weight.dim(0), 80], embedding.scales.dtype == .bfloat16,
              let biases = embedding.biases, biases.shape == embedding.scales.shape, biases.dtype == .bfloat16
        else { return nil }
        words = try M5NativeArrayBuffer(embedding.weight, device: device)
        scales = try M5NativeArrayBuffer(embedding.scales, device: device)
        self.biases = try M5NativeArrayBuffer(biases, device: device)
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        guard let kernel = library.makeFunction(name: "q35_target_embedding") else {
            throw Q35NativeVerification.Failure.geometry
        }
        pipeline = try device.makeComputePipelineState(function: kernel)
    }

    func supports(_ tokens: [Int]) -> Bool {
        let vocabulary = words.shape[0]
        return tokens.allSatisfy { (0..<vocabulary).contains($0) }
    }

    func encode(
        _ tokens: [Int], into output: any MTLBuffer, encoder: any MTLComputeCommandEncoder,
        hazards: M5NativeHazards
    ) {
        hazards.prepare(encoder: encoder, reads: [], writes: [.init(output)])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(words.buffer, offset: words.offset, index: 0)
        encoder.setBuffer(scales.buffer, offset: scales.offset, index: 1)
        encoder.setBuffer(biases.buffer, offset: biases.offset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        tokens.map(UInt32.init).withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
        }
        encoder.dispatchThreads(MTLSize(width: 640, height: tokens.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void q35_target_embedding(
            const device uint* words [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
            const device bfloat* biases [[buffer(2)]], device bfloat* output [[buffer(3)]],
            constant uint* tokens [[buffer(4)]], uint2 rowWord [[thread_position_in_grid]]) {
            uint word = rowWord.x, row = rowWord.y, token = tokens[row];
            uint packed = words[token * 640 + word];
            uint group = token * 80 + word / 8;
            float scale = float(scales[group]), bias = float(biases[group]);
            for (uint i = 0; i < 8; ++i) {
                float code = float((packed >> (i * 4)) & 15);
                output[row * 5120 + word * 8 + i] = bfloat(fma(scale, code, bias));
            }
        }
        """
}
#endif
