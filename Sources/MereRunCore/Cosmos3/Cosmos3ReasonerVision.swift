import Foundation
import MLX
import MLXFast
import MLXNN

final class Cosmos3ReasonerVisionAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "out_proj") var output: Linear

    init(configuration: Cosmos3ReasonerConfiguration.Vision) {
        headCount = configuration.attentionHeadCount
        headDimension = configuration.hiddenSize / configuration.attentionHeadCount
        scale = 1 / sqrt(Float(headDimension))
        _key.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        _value.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        _query.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        _output.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
    }

    func callAsFunction(_ input: MLXArray, segmentLengths: [Int]) -> MLXArray {
        precondition(segmentLengths.reduce(0, +) == input.dim(0))
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(segmentLengths.count)
        var offset = 0
        for length in segmentLengths {
            let segment = input[offset..<(offset + length)]
            let queries = query(segment)
                .reshaped(length, headCount, headDimension)
                .transposed(1, 0, 2)
                .expandedDimensions(axis: 0)
            let keys = key(segment)
                .reshaped(length, headCount, headDimension)
                .transposed(1, 0, 2)
                .expandedDimensions(axis: 0)
            let values = value(segment)
                .reshaped(length, headCount, headDimension)
                .transposed(1, 0, 2)
                .expandedDimensions(axis: 0)
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            ).squeezed(axis: 0)
                .transposed(1, 0, 2)
                .reshaped(length, headCount * headDimension)
            outputs.append(output(attended))
            offset += length
        }
        return MLX.concatenated(outputs, axis: 0)
    }
}

final class Cosmos3ReasonerVisionMLP: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

    init(configuration: Cosmos3ReasonerConfiguration.Vision) {
        _input.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: true
        )
        _output.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.hiddenSize,
            bias: true
        )
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(geluApproximate(input(value)))
    }
}

final class Cosmos3ReasonerVisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Cosmos3ReasonerVisionAttention
    @ModuleInfo(key: "mlp") var mlp: Cosmos3ReasonerVisionMLP
    @ModuleInfo(key: "layer_norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "layer_norm2") var mlpNorm: LayerNorm

    init(configuration: Cosmos3ReasonerConfiguration.Vision) {
        _attention.wrappedValue = Cosmos3ReasonerVisionAttention(configuration: configuration)
        _mlp.wrappedValue = Cosmos3ReasonerVisionMLP(configuration: configuration)
        _attentionNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
        _mlpNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
    }

    func callAsFunction(_ input: MLXArray, segmentLengths: [Int]) -> MLXArray {
        let attended = input + attention(attentionNorm(input), segmentLengths: segmentLengths)
        return attended + mlp(mlpNorm(attended))
    }
}

final class Cosmos3ReasonerVisionEmbeddings: Module {
    let configuration: Cosmos3ReasonerConfiguration.Vision
    let sourceGridSize: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(configuration: Cosmos3ReasonerConfiguration.Vision) {
        self.configuration = configuration
        sourceGridSize = Int(Double(configuration.patchCount).squareRoot())
        _patchEmbedding.wrappedValue = Linear(
            configuration.channelCount * configuration.patchSize * configuration.patchSize,
            configuration.hiddenSize,
            bias: true
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.patchCount,
            dimensions: configuration.hiddenSize
        )
    }

    func callAsFunction(
        patches: MLXArray,
        grids: [Cosmos3ReasonerVisionGrid]
    ) -> MLXArray {
        precondition(patches.dim(0) == grids.reduce(0) { $0 + $1.patchCount })
        let positions = grids.map { resizedPosition(grid: $0, dtype: patches.dtype) }
        return patchEmbedding(patches) + MLX.concatenated(positions, axis: 0)
    }

    private func resizedPosition(
        grid: Cosmos3ReasonerVisionGrid,
        dtype: DType
    ) -> MLXArray {
        let source = positionEmbedding.weight
            .reshaped(1, sourceGridSize, sourceGridSize, configuration.hiddenSize)
        let spatial: MLXArray
        if grid.height == sourceGridSize, grid.width == sourceGridSize {
            spatial = source.asType(dtype)
        } else {
            spatial = Upsample(
                scaleFactor: [
                    Float(grid.height) / Float(sourceGridSize),
                    Float(grid.width) / Float(sourceGridSize),
                ],
                mode: .linear(alignCorners: false)
            )(source.asType(.float32)).asType(dtype)
        }
        let flattened = spatial.reshaped(
            grid.height * grid.width,
            configuration.hiddenSize
        )
        return grid.time == 1
            ? flattened
            : MLX.tiled(flattened, repetitions: [grid.time, 1])
    }
}

final class Cosmos3ReasonerVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Cosmos3ReasonerVisionLayer]

    init(configuration: Cosmos3ReasonerConfiguration.Vision) {
        _layers.wrappedValue = (0..<configuration.layerCount).map { _ in
            Cosmos3ReasonerVisionLayer(configuration: configuration)
        }
    }

    func callAsFunction(_ input: MLXArray, segmentLengths: [Int]) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden, segmentLengths: segmentLengths)
        }
        return hidden
    }
}

final class Cosmos3ReasonerVisionTransformer: Module {
    let configuration: Cosmos3ReasonerConfiguration.Vision

    @ModuleInfo(key: "embeddings") var embeddings: Cosmos3ReasonerVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: Cosmos3ReasonerVisionEncoder
    @ModuleInfo(key: "post_layernorm") var outputNorm: LayerNorm

    init(configuration: Cosmos3ReasonerConfiguration.Vision) {
        self.configuration = configuration
        _embeddings.wrappedValue = Cosmos3ReasonerVisionEmbeddings(configuration: configuration)
        _encoder.wrappedValue = Cosmos3ReasonerVisionEncoder(configuration: configuration)
        _outputNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
    }

    func callAsFunction(
        patches: MLXArray,
        grids: [Cosmos3ReasonerVisionGrid]
    ) -> MLXArray {
        let segments = grids.flatMap { grid in
            Array(repeating: grid.height * grid.width, count: grid.time)
        }
        return outputNorm(encoder(
            embeddings(patches: patches, grids: grids),
            segmentLengths: segments
        ))
    }
}

final class Cosmos3ReasonerProjector: Module {
    let configuration: Cosmos3ReasonerConfiguration.Projector

    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var input: Linear
    @ModuleInfo(key: "linear_fc2") var output: Linear

    init(configuration: Cosmos3ReasonerConfiguration.Projector) {
        self.configuration = configuration
        let mergeDimension = configuration.inputHiddenSize
            * configuration.spatialMergeSize
            * configuration.spatialMergeSize
        _norm.wrappedValue = LayerNorm(
            dimensions: configuration.usesPostShuffleNorm
                ? mergeDimension
                : configuration.inputHiddenSize,
            eps: 1e-6
        )
        _input.wrappedValue = Linear(
            mergeDimension,
            configuration.intermediateSize,
            bias: true
        )
        _output.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.outputHiddenSize,
            bias: true
        )
    }

    func callAsFunction(_ mergedPatches: MLXArray) -> MLXArray {
        let mergeDimension = configuration.inputHiddenSize
            * configuration.spatialMergeSize
            * configuration.spatialMergeSize
        let normalized = configuration.usesPostShuffleNorm
            ? norm(mergedPatches.reshaped(-1, mergeDimension))
            : norm(mergedPatches).reshaped(-1, mergeDimension)
        return output(gelu(input(normalized)))
    }
}

public struct Cosmos3ReasonerVisionGrid: Hashable, Sendable {
    public let time: Int
    public let height: Int
    public let width: Int

    public init(time: Int, height: Int, width: Int) {
        precondition(time > 0 && height > 0 && width > 0)
        self.time = time
        self.height = height
        self.width = width
    }

    public var patchCount: Int { time * height * width }
}

public final class Cosmos3ReasonerVisionModel: Module {
    public let configuration: Cosmos3ReasonerConfiguration

    @ModuleInfo(key: "visual") var visual: Cosmos3ReasonerVisionTransformer
    @ModuleInfo(key: "projector") var projector: Cosmos3ReasonerProjector

    public init(configuration: Cosmos3ReasonerConfiguration) {
        self.configuration = configuration
        _visual.wrappedValue = Cosmos3ReasonerVisionTransformer(
            configuration: configuration.vision
        )
        _projector.wrappedValue = Cosmos3ReasonerProjector(
            configuration: configuration.projector
        )
    }

    public func callAsFunction(
        patches: MLXArray,
        grids: [Cosmos3ReasonerVisionGrid]
    ) -> (embeddings: MLXArray, mergedGrids: [Cosmos3ReasonerVisionGrid]) {
        let hidden = visual(patches: patches, grids: grids)
        let merge = configuration.projector.spatialMergeSize
        var merged: [MLXArray] = []
        var mergedGrids: [Cosmos3ReasonerVisionGrid] = []
        var offset = 0
        for grid in grids {
            precondition(grid.height.isMultiple(of: merge))
            precondition(grid.width.isMultiple(of: merge))
            let item = hidden[offset..<(offset + grid.patchCount)]
                .reshaped(
                    grid.time,
                    grid.height / merge,
                    merge,
                    grid.width / merge,
                    merge,
                    configuration.vision.hiddenSize
                )
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(
                    -1,
                    merge * merge,
                    configuration.vision.hiddenSize
                )
            merged.append(projector(item))
            mergedGrids.append(Cosmos3ReasonerVisionGrid(
                time: grid.time,
                height: grid.height / merge,
                width: grid.width / merge
            ))
            offset += grid.patchCount
        }
        return (MLX.concatenated(merged, axis: 0), mergedGrids)
    }
}
