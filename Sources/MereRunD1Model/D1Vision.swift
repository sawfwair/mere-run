#if !os(iOS)
// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MLX
import MLXFast
import MLXNN

final class D1VLVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear

    private let headCount: Int
    private let headDimension: Int
    private let scale: Float

    init(config: D1VLVisionConfig) {
        headCount = config.numAttentionHeads
        headDimension = config.hiddenSize / max(1, config.numAttentionHeads)
        scale = 1 / sqrt(Float(max(1, headDimension)))
        _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _output.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let queries = query(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let keys = key(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = value(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        return output(MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, sequence, headCount * headDimension))
    }
}

final class D1VLVisionMLP: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

    init(config: D1VLVisionConfig) {
        _input.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _output.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(geluApproximate(input(value)))
    }
}

final class D1VLVisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: D1VLVisionAttention
    @ModuleInfo(key: "mlp") var mlp: D1VLVisionMLP
    @ModuleInfo(key: "layer_norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "layer_norm2") var mlpNorm: LayerNorm

    init(config: D1VLVisionConfig) {
        _attention.wrappedValue = D1VLVisionAttention(config: config)
        _mlp.wrappedValue = D1VLVisionMLP(config: config)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(attentionNorm(input))
        return attended + mlp(mlpNorm(attended))
    }
}

final class D1VLVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    private let config: D1VLVisionConfig
    private let sourceGridSize: Int

    init(config: D1VLVisionConfig) {
        self.config = config
        sourceGridSize = Int(Double(config.numPatches).squareRoot())
        _patchEmbedding.wrappedValue = Linear(
            config.numChannels * config.patchSize * config.patchSize,
            config.hiddenSize,
            bias: true
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numPatches,
            dimensions: config.hiddenSize
        )
        super.init()
    }

    func callAsFunction(pixelValues: MLXArray, grids: [D1VLImageGrid]) -> MLXArray {
        precondition(pixelValues.dim(0) == grids.count)
        let projected = patchEmbedding(pixelValues.asType(patchEmbedding.weight.dtype))
        let positions = grids.map { resizedPositions(grid: $0, maxLength: pixelValues.dim(1)) }
        return projected + MLX.concatenated(positions, axis: 0)
    }

    private func resizedPositions(grid: D1VLImageGrid, maxLength: Int) -> MLXArray {
        let source = positionEmbedding.weight
            .reshaped(1, sourceGridSize, sourceGridSize, config.hiddenSize)
        let resized: MLXArray
        if grid.rows == sourceGridSize, grid.columns == sourceGridSize {
            resized = source
        } else {
            // SigLIP2 NaFlex resizes learned positions with antialiased bilinear interpolation.
            func coefficients(target: Int) -> MLXArray {
                let scale = Double(sourceGridSize) / Double(target), support = max(1, scale)
                let values = (0..<target).flatMap { row -> [Float] in
                    let center = (Double(row) + 0.5) * scale
                    let weights = (0..<sourceGridSize).map { column in
                        max(0, 1 - abs((Double(column) + 0.5 - center) / support))
                    }
                    let total = weights.reduce(0, +)
                    return weights.map { Float($0 / total) }
                }
                return MLXArray(values).reshaped(target, sourceGridSize)
            }
            let columns = matmul(source.asType(.float32).reshaped(sourceGridSize, sourceGridSize, config.hiddenSize)
                .transposed(0, 2, 1), coefficients(target: grid.columns).T).transposed(0, 2, 1)
            resized = matmul(coefficients(target: grid.rows), columns.reshaped(sourceGridSize, -1))
                .reshaped(1, grid.rows, grid.columns, config.hiddenSize).asType(source.dtype)
        }
        let actual = resized.reshaped(1, grid.patchCount, config.hiddenSize)
        guard grid.patchCount < maxLength else { return actual }
        let padding = MLX.tiled(
            actual[0..., 0..<1, 0...],
            repetitions: [1, maxLength - grid.patchCount, 1]
        )
        return MLX.concatenated([actual, padding], axis: 1)
    }
}

final class D1VLVisionTower: Module {
    @ModuleInfo(key: "embeddings") var embeddings: D1VLVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: D1VLVisionEncoder
    @ModuleInfo(key: "post_layernorm") var outputNorm: LayerNorm

    init(config: D1VLVisionConfig) {
        _embeddings.wrappedValue = D1VLVisionEmbeddings(config: config)
        _encoder.wrappedValue = D1VLVisionEncoder(config: config)
        _outputNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(pixelValues: MLXArray, grids: [D1VLImageGrid]) -> MLXArray {
        outputNorm(encoder(embeddings(pixelValues: pixelValues, grids: grids)))
    }
}

final class D1VLVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [D1VLVisionLayer]

    init(config: D1VLVisionConfig) {
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            D1VLVisionLayer(config: config)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers.reduce(input) { hidden, layer in layer(hidden) }
    }
}


public struct D1VLImageGrid: Sendable, Hashable {
    public let rows: Int
    public let columns: Int
    public var patchCount: Int { rows * columns }
    public init(rows: Int, columns: Int) { self.rows = rows; self.columns = columns }
}

final class D1Projector: Module {
    @ModuleInfo(key: "linear_1") var input: Linear
    @ModuleInfo(key: "linear_2") var output: Linear
    init(_ config: D1Configuration) {
        _input.wrappedValue = Linear(4 * config.vision_config.hiddenSize, config.projector_hidden_size)
        _output.wrappedValue = Linear(config.projector_hidden_size, config.text_config.hidden_size)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let unshuffled = x.reshaped(1, h, w / 2, 2 * c).transposed(0, 2, 1, 3)
            .reshaped(1, w / 2, h / 2, 4 * c).transposed(0, 2, 1, 3)
        return output(gelu(input(unshuffled))).reshaped(1, -1, output.weight.dim(0))
    }
}

public final class D1Vision: Module {
    @ModuleInfo var tower: D1VLVisionTower
    @ModuleInfo var projector: D1Projector
    public init(_ config: D1Configuration) {
        _tower.wrappedValue = D1VLVisionTower(config: config.vision_config)
        _projector.wrappedValue = D1Projector(config)
        super.init()
    }
    public func callAsFunction(pixels: [MLXArray], grids: [D1VLImageGrid]) -> MLXArray {
        let features = zip(pixels, grids).map { pixel, grid in
            let hidden = tower(pixelValues: pixel, grids: [grid])
            return projector(hidden.reshaped(1, grid.rows, grid.columns, -1))
        }
        return concatenated(features, axis: 1)
    }
}
#endif
