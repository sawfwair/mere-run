import Foundation
import MLX
import MLXFast
import MLXNN

/// Rotary phases for the row and column halves of each visual attention head.
struct L24SQPositions {
    let cosine: MLXArray
    let sine: MLXArray

    init(rows: Int, columns: Int, headWidth: Int, dtype: DType) {
        let quarter = headWidth / 4
        let frequencies = MLXArray((0..<quarter).map { pow(Float(10_000), -Float($0) / Float(quarter)) })
        let row = MLX.repeated(MLXArray(0..<rows), count: columns)
        let column = MLX.tiled(MLXArray(0..<columns), repetitions: [rows])
        let anglesY = row.asType(.float32).reshaped(-1, 1) * frequencies
        let anglesX = column.asType(.float32).reshaped(-1, 1) * frequencies
        let phases = concatenated([anglesY, anglesY, anglesX, anglesX], axis: -1)
        cosine = cos(phases).asType(dtype)
        sine = sin(phases).asType(dtype)
    }

    func rotate(_ heads: MLXArray) -> MLXArray {
        let parts = heads.split(parts: 4, axis: -1)
        let rotated = concatenated([-parts[1], parts[0], -parts[3], parts[2]], axis: -1)
        return heads * cosine + rotated * sine
    }
}

private final class L24SQAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        _query.wrappedValue = Linear(width, width, bias: false)
        _key.wrappedValue = Linear(width, width, bias: false)
        _value.wrappedValue = Linear(width, width, bias: false)
        _output.wrappedValue = Linear(width, width, bias: false)
    }

    func callAsFunction(_ input: MLXArray, positions: L24SQPositions) -> MLXArray {
        let headWidth = input.dim(-1) / heads
        func split(_ value: MLXArray) -> MLXArray {
            value.reshaped(1, -1, heads, headWidth).transposed(0, 2, 1, 3)
        }
        let result = MLXFast.scaledDotProductAttention(
            queries: positions.rotate(split(query(input))), keys: positions.rotate(split(key(input))),
            values: split(value(input)), scale: 1 / sqrt(Float(headWidth)), mask: .none)
        return output(result.transposed(0, 2, 1, 3).reshaped(input.shape))
    }
}

private final class L24SQMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(width: Int) {
        let hidden = (width * 8 / 3 + 63) / 64 * 64
        _gate.wrappedValue = Linear(width, hidden, bias: false)
        _up.wrappedValue = Linear(width, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, width, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        down(silu(gate(input)) * up(input))
    }
}

private final class L24SQBlock: Module {
    let norm1: RMSNorm
    let attn: L24SQAttention
    let norm2: RMSNorm
    let mlp: L24SQMLP

    init(width: Int, heads: Int) {
        norm1 = RMSNorm(dimensions: width, eps: 1e-6)
        attn = L24SQAttention(width: width, heads: heads)
        norm2 = RMSNorm(dimensions: width, eps: 1e-6)
        mlp = L24SQMLP(width: width)
    }

    func callAsFunction(_ input: MLXArray, positions: L24SQPositions) -> MLXArray {
        let hidden = input + attn(norm1(input), positions: positions)
        return hidden + mlp(norm2(hidden))
    }
}

/// The inference path from L24SQ lattice indices to interleaved RGB pixels.
public final class L24SQDecoder: Module {
    private let embedding: Embedding
    @ModuleInfo(key: "post_quant_embed") private var projection: Linear
    private let blocks: [L24SQBlock]
    private let norm: RMSNorm
    private let ffn: [Linear]
    @ModuleInfo(key: "conv_out") private var patchOutput: Linear
    private let headWidth: Int
    private let patchSize: Int

    public override convenience init() {
        self.init(width: 1_024, heads: 16, layers: 24, codeCount: 196_560,
                  embeddingWidth: 24, outputWidth: 3_072, patchSize: 16)
    }

    init(width: Int, heads: Int, layers: Int, codeCount: Int, embeddingWidth: Int, outputWidth: Int, patchSize: Int) {
        self.headWidth = width / heads
        self.patchSize = patchSize
        embedding = Embedding(embeddingCount: codeCount, dimensions: embeddingWidth)
        _projection.wrappedValue = Linear(embeddingWidth, width)
        blocks = (0..<layers).map { _ in L24SQBlock(width: width, heads: heads) }
        norm = RMSNorm(dimensions: width, eps: 1e-6)
        ffn = [Linear(width, outputWidth)]
        _patchOutput.wrappedValue = Linear(outputWidth, 3 * patchSize * patchSize)
    }

    public func decode(_ codes: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let vectors = embedding(codes)
        let floats = vectors.asType(.float32)
        let lengths = sqrt(square(floats).sum(axis: -1, keepDims: true)).asType(vectors.dtype)
        var hidden = projection(vectors / maximum(lengths, MLXArray(Float(1e-12)).asType(vectors.dtype)))
        let positions = L24SQPositions(rows: rows, columns: columns, headWidth: headWidth, dtype: hidden.dtype)
        for block in blocks {
            hidden = block(hidden, positions: positions)
        }
        let patches = patchOutput(tanh(ffn[0](norm(hidden)))).asType(.float32)
        let rgb = patches.reshaped(rows, columns, 3, patchSize, patchSize)
            .transposed(0, 3, 1, 4, 2).reshaped(rows * patchSize, columns * patchSize, 3)
        return MLX.round((clip(rgb, min: -1, max: 1) + 1) * 127.5).asType(.uint8)
    }
}
