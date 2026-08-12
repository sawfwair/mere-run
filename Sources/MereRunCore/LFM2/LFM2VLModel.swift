import Foundation
import MLX
import MLXFast
import MLXNN

final class LFM2VLVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear

    private let headCount: Int
    private let headDimension: Int
    private let scale: Float

    init(config: LFM2VLVisionConfig) {
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

final class LFM2VLVisionMLP: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

    init(config: LFM2VLVisionConfig) {
        _input.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _output.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(geluApproximate(input(value)))
    }
}

final class LFM2VLVisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM2VLVisionAttention
    @ModuleInfo(key: "mlp") var mlp: LFM2VLVisionMLP
    @ModuleInfo(key: "layer_norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "layer_norm2") var mlpNorm: LayerNorm

    init(config: LFM2VLVisionConfig) {
        _attention.wrappedValue = LFM2VLVisionAttention(config: config)
        _mlp.wrappedValue = LFM2VLVisionMLP(config: config)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(attentionNorm(input))
        return attended + mlp(mlpNorm(attended))
    }
}

final class LFM2VLVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    private let config: LFM2VLVisionConfig
    private let sourceGridSize: Int

    init(config: LFM2VLVisionConfig) {
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

    func callAsFunction(pixelValues: MLXArray, grids: [LFM2VLImageGrid]) -> MLXArray {
        precondition(pixelValues.dim(0) == grids.count)
        let projected = patchEmbedding(pixelValues.asType(patchEmbedding.weight.dtype))
        let positions = grids.map { resizedPositions(grid: $0, maxLength: pixelValues.dim(1)) }
        return projected + MLX.concatenated(positions, axis: 0)
    }

    private func resizedPositions(grid: LFM2VLImageGrid, maxLength: Int) -> MLXArray {
        let source = positionEmbedding.weight
            .reshaped(1, sourceGridSize, sourceGridSize, config.hiddenSize)
        let resized: MLXArray
        if grid.rows == sourceGridSize, grid.columns == sourceGridSize {
            resized = source
        } else {
            resized = Upsample(
                scaleFactor: [
                    Float(grid.rows) / Float(sourceGridSize),
                    Float(grid.columns) / Float(sourceGridSize),
                ],
                mode: .cubic(alignCorners: false)
            )(source.asType(.float32)).asType(source.dtype)
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

final class LFM2VLVisionTower: Module {
    @ModuleInfo(key: "embeddings") var embeddings: LFM2VLVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: LFM2VLVisionEncoder
    @ModuleInfo(key: "post_layernorm") var outputNorm: LayerNorm

    init(config: LFM2VLVisionConfig) {
        _embeddings.wrappedValue = LFM2VLVisionEmbeddings(config: config)
        _encoder.wrappedValue = LFM2VLVisionEncoder(config: config)
        _outputNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(pixelValues: MLXArray, grids: [LFM2VLImageGrid]) -> MLXArray {
        outputNorm(encoder(embeddings(pixelValues: pixelValues, grids: grids)))
    }
}

final class LFM2VLVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [LFM2VLVisionLayer]

    init(config: LFM2VLVisionConfig) {
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            LFM2VLVisionLayer(config: config)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers.reduce(input) { hidden, layer in layer(hidden) }
    }
}

final class LFM2VLMultiModalProjector: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "linear_1") var input: Linear
    @ModuleInfo(key: "linear_2") var output: Linear

    init(config: LFM2VLConfig) {
        let inputSize = config.visionConfig.hiddenSize * config.downsampleFactor * config.downsampleFactor
        _layerNorm.wrappedValue = config.projectorUseLayerNorm
            ? LayerNorm(dimensions: inputSize)
            : nil
        _input.wrappedValue = Linear(
            inputSize,
            config.projectorHiddenSize,
            bias: config.projectorBias
        )
        _output.wrappedValue = Linear(
            config.projectorHiddenSize,
            config.textConfig.hiddenSize,
            bias: config.projectorBias
        )
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let normalized = layerNorm?(value) ?? value
        return output(gelu(input(normalized)))
    }
}

public final class LFM2VLModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "vision_tower") var visionTower: LFM2VLVisionTower
    @ModuleInfo(key: "multi_modal_projector") var projector: LFM2VLMultiModalProjector
    @ModuleInfo(key: "language_model") var languageModel: LFM2Model

    let config: LFM2VLConfig

    public init(config: LFM2VLConfig) {
        self.config = config
        _visionTower.wrappedValue = LFM2VLVisionTower(config: config.visionConfig)
        _projector.wrappedValue = LFM2VLMultiModalProjector(config: config)
        _languageModel.wrappedValue = LFM2Model(config: config.textConfig)
        super.init()
    }

    func inputEmbeddings(
        inputTokens: [Int],
        pixelValues: MLXArray,
        grids: [LFM2VLImageGrid]
    ) throws -> MLXArray {
        let inputIds = MLXArray(inputTokens.map(Int32.init)).reshaped(1, inputTokens.count)
        let tokenEmbeddings = languageModel.embeddings(for: inputIds)
        let hidden = visionTower(pixelValues: pixelValues, grids: grids)
        var projectedImages: [MLXArray] = []
        for (index, grid) in grids.enumerated() {
            let features = hidden[index..<(index + 1), 0..<grid.patchCount, 0...]
                .reshaped(1, grid.rows, grid.columns, config.visionConfig.hiddenSize)
            projectedImages.append(projector(pixelUnshuffle(features)).reshaped(-1, config.textConfig.hiddenSize))
        }
        let imageFeatures = MLX.concatenated(projectedImages, axis: 0)
        let imagePositions = inputTokens.indices.filter { inputTokens[$0] == config.imageTokenIndex }
        guard imagePositions.count == imageFeatures.dim(0) else {
            throw LFM2Error.generationFailed(
                "LFM2-VL image features and prompt tokens do not match: \(imageFeatures.dim(0)) features for \(imagePositions.count) tokens."
            )
        }

        var featureIndex = 0
        let rows = inputTokens.indices.map { tokenIndex -> MLXArray in
            guard inputTokens[tokenIndex] == config.imageTokenIndex else {
                return tokenEmbeddings[0..., tokenIndex..<(tokenIndex + 1), 0...]
            }
            defer { featureIndex += 1 }
            return imageFeatures[featureIndex..<(featureIndex + 1), 0...].expandedDimensions(axis: 0)
        }
        return MLX.concatenated(rows, axis: 1)
    }

    private func pixelUnshuffle(_ input: MLXArray) -> MLXArray {
        let factor = max(1, config.downsampleFactor)
        guard factor > 1 else { return input }
        let rows = input.dim(1)
        let columns = input.dim(2)
        let rowPadding = (factor - (rows % factor)) % factor
        let columnPadding = (factor - (columns % factor)) % factor
        let paddedInput = padded(
            input,
            widths: [[0, 0], [0, rowPadding], [0, columnPadding], [0, 0]],
            value: MLXArray(0).asType(input.dtype)
        )
        let paddedRows = rows + rowPadding
        let paddedColumns = columns + columnPadding
        let channels = input.dim(3)
        return paddedInput
            .reshaped(1, paddedRows, paddedColumns / factor, channels * factor)
            .transposed(0, 2, 1, 3)
            .reshaped(1, paddedColumns / factor, paddedRows / factor, channels * factor * factor)
            .transposed(0, 2, 1, 3)
    }
}
