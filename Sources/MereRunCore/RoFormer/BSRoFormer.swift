import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

private final class RoFormerRMSNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int) {
        self._gamma.wrappedValue = MLX.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let meanSquare = MLX.mean(input.square(), axis: -1, keepDims: true)
        return input * MLX.rsqrt(meanSquare + MLXArray(1e-12).asType(input.dtype))
            * gamma.asType(input.dtype)
    }
}

private final class RoFormerRotaryEmbedding: Module {
    @ParameterInfo(key: "freqs") var frequencies: MLXArray

    init(dimensions: Int) {
        self._frequencies.wrappedValue = MLX.zeros([dimensions / 2])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let heads = input.dim(2)
        let dimensions = input.dim(3)
        let positions = MLXArray((0..<sequence).map(Float.init))
        let angles = positions.expandedDimensions(axis: -1) * frequencies.expandedDimensions(axis: 0)
        let cosine = MLX.cos(angles).reshaped(1, sequence, 1, dimensions / 2).asType(input.dtype)
        let sine = MLX.sin(angles).reshaped(1, sequence, 1, dimensions / 2).asType(input.dtype)
        let paired = input.reshaped(batch, sequence, heads, dimensions / 2, 2)
        let even = paired[.ellipsis, 0]
        let odd = paired[.ellipsis, 1]
        return MLX.stacked([
            even * cosine - odd * sine,
            odd * cosine + even * sine,
        ], axis: -1).reshaped(batch, sequence, heads, dimensions)
    }
}

private final class RoFormerAttentionOutput: Module {
    @ModuleInfo(key: "0") var projection: Linear

    init(dimensions: Int) {
        self._projection.wrappedValue = Linear(dimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { projection(input) }
}

private final class RoFormerAttention: Module {
    @ModuleInfo(key: "norm") var norm: RoFormerRMSNorm
    @ModuleInfo(key: "rotary_embed") var rotary: RoFormerRotaryEmbedding
    @ModuleInfo(key: "to_qkv") var qkv: Linear
    @ModuleInfo(key: "to_gates") var gates: Linear
    @ModuleInfo(key: "to_out") var output: RoFormerAttentionOutput

    private let heads: Int
    private let headDimension: Int
    private let scale: Float

    init(dimensions: Int, heads: Int, headDimension: Int) {
        self.heads = heads
        self.headDimension = headDimension
        self.scale = 1 / sqrt(Float(headDimension))
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: dimensions)
        self._rotary.wrappedValue = RoFormerRotaryEmbedding(dimensions: headDimension)
        self._qkv.wrappedValue = Linear(dimensions, 3 * heads * headDimension, bias: false)
        self._gates.wrappedValue = Linear(dimensions, heads, bias: true)
        self._output.wrappedValue = RoFormerAttentionOutput(dimensions: dimensions)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let normalized = norm(input)
        let projected = qkv(normalized).reshaped(batch, sequence, 3, heads, headDimension)
        let parts = MLX.split(projected, parts: 3, axis: 2).map { $0.squeezed(axis: 2) }
        let queries = rotary(parts[0]).transposed(0, 2, 1, 3)
        let keys = rotary(parts[1]).transposed(0, 2, 1, 3)
        let values = parts[2].transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        ).transposed(0, 2, 1, 3)
        let gated = attended * MLX.sigmoid(gates(normalized)).expandedDimensions(axis: -1)
        return output(gated.reshaped(batch, sequence, heads * headDimension))
    }
}

private final class RoFormerFeedForwardNet: Module {
    @ModuleInfo(key: "0") var norm: RoFormerRMSNorm
    @ModuleInfo(key: "1") var inputProjection: Linear
    @ModuleInfo(key: "4") var outputProjection: Linear

    init(dimensions: Int, expansionFactor: Int) {
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: dimensions)
        self._inputProjection.wrappedValue = Linear(dimensions, dimensions * expansionFactor, bias: true)
        self._outputProjection.wrappedValue = Linear(dimensions * expansionFactor, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = inputProjection(norm(input))
        let exactGELU = hidden * MLXArray(0.5).asType(hidden.dtype)
            * (MLXArray(1).asType(hidden.dtype) + MLX.erf(hidden / sqrt(Float(2))))
        return outputProjection(exactGELU)
    }
}

private final class RoFormerFeedForward: Module {
    @ModuleInfo(key: "net") var net: RoFormerFeedForwardNet

    init(dimensions: Int, expansionFactor: Int) {
        self._net.wrappedValue = RoFormerFeedForwardNet(
            dimensions: dimensions,
            expansionFactor: expansionFactor
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { net(input) }
}

private final class RoFormerTransformerLayer: Module {
    @ModuleInfo(key: "0") var attention: RoFormerAttention
    @ModuleInfo(key: "1") var feedForward: RoFormerFeedForward

    init(configuration: RoFormerConfiguration) {
        self._attention.wrappedValue = RoFormerAttention(
            dimensions: configuration.dim,
            heads: configuration.heads,
            headDimension: configuration.dimHead
        )
        self._feedForward.wrappedValue = RoFormerFeedForward(
            dimensions: configuration.dim,
            expansionFactor: configuration.mlpExpansionFactor
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(input)
        return attended + feedForward(attended)
    }
}

private final class RoFormerTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [RoFormerTransformerLayer]

    init(configuration: RoFormerConfiguration, depth: Int) {
        self._layers.wrappedValue = (0..<depth).map { _ in
            RoFormerTransformerLayer(configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers.reduce(input) { hidden, layer in layer(hidden) }
    }
}

private final class RoFormerAxialLayer: Module {
    @ModuleInfo(key: "0") var timeTransformer: RoFormerTransformer
    @ModuleInfo(key: "1") var frequencyTransformer: RoFormerTransformer

    init(configuration: RoFormerConfiguration) {
        self._timeTransformer.wrappedValue = RoFormerTransformer(
            configuration: configuration,
            depth: configuration.timeTransformerDepth
        )
        self._frequencyTransformer.wrappedValue = RoFormerTransformer(
            configuration: configuration,
            depth: configuration.frequencyTransformerDepth
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let bands = input.dim(2)
        let dimensions = input.dim(3)
        let overTime = timeTransformer(
            input.transposed(0, 2, 1, 3).reshaped(batch * bands, frames, dimensions)
        ).reshaped(batch, bands, frames, dimensions).transposed(0, 2, 1, 3)
        return frequencyTransformer(
            overTime.reshaped(batch * frames, bands, dimensions)
        ).reshaped(batch, frames, bands, dimensions)
    }
}

private final class RoFormerBandProjection: Module {
    @ModuleInfo(key: "0") var norm: RoFormerRMSNorm
    @ModuleInfo(key: "1") var projection: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: inputDimensions)
        self._projection.wrappedValue = Linear(inputDimensions, outputDimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { projection(norm(input)) }
}

private final class RoFormerBandSplit: Module {
    @ModuleInfo(key: "to_features") var projections: [RoFormerBandProjection]
    private let inputDimensions: [Int]

    init(configuration: RoFormerConfiguration) {
        self.inputDimensions = configuration.frequencyBandInputDimensions
        self._projections.wrappedValue = inputDimensions.map {
            RoFormerBandProjection(inputDimensions: $0, outputDimensions: configuration.dim)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let splitPoints = inputDimensions.dropLast().reduce(into: [Int]()) { result, width in
            result.append((result.last ?? 0) + width)
        }
        let bands = MLX.split(input, indices: splitPoints, axis: -1)
        return MLX.stacked(zip(projections, bands).map { projection, band in
            projection(band)
        }, axis: -2)
    }
}

private final class RoFormerMaskMLP: Module {
    @ModuleInfo(key: "0") var inputProjection: Linear
    @ModuleInfo(key: "2") var outputProjection: Linear

    init(dimensions: Int, outputDimensions: Int, expansionFactor: Int) {
        self._inputProjection.wrappedValue = Linear(
            dimensions,
            dimensions * expansionFactor,
            bias: true
        )
        self._outputProjection.wrappedValue = Linear(
            dimensions * expansionFactor,
            outputDimensions * 2,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let projected = outputProjection(MLX.tanh(inputProjection(input)))
        let parts = MLX.split(projected, parts: 2, axis: -1)
        return parts[0] * MLX.sigmoid(parts[1])
    }
}

private final class RoFormerMaskBand: Module {
    @ModuleInfo(key: "0") var mlp: RoFormerMaskMLP

    init(configuration: RoFormerConfiguration, outputDimensions: Int) {
        self._mlp.wrappedValue = RoFormerMaskMLP(
            dimensions: configuration.dim,
            outputDimensions: outputDimensions,
            expansionFactor: configuration.mlpExpansionFactor
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { mlp(input) }
}

private final class RoFormerMaskEstimator: Module {
    @ModuleInfo(key: "to_freqs") var bands: [RoFormerMaskBand]

    init(configuration: RoFormerConfiguration) {
        self._bands.wrappedValue = configuration.frequencyBandInputDimensions.map {
            RoFormerMaskBand(configuration: configuration, outputDimensions: $0)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let values = zip(bands, 0..<bands.count).map { band, index in
            band(input[0..., 0..., index, 0...])
        }
        return MLX.concatenated(values, axis: -1)
    }
}

public final class BSRoFormer: Module {
    @ModuleInfo(key: "layers") private var layers: [RoFormerAxialLayer]
    @ModuleInfo(key: "final_norm") private var finalNorm: RoFormerRMSNorm
    @ModuleInfo(key: "band_split") private var bandSplit: RoFormerBandSplit
    @ModuleInfo(key: "mask_estimators") private var maskEstimators: [RoFormerMaskEstimator]

    public let configuration: RoFormerConfiguration

    public init(configuration: RoFormerConfiguration) {
        self.configuration = configuration
        self._layers.wrappedValue = (0..<configuration.depth).map { _ in
            RoFormerAxialLayer(configuration: configuration)
        }
        self._finalNorm.wrappedValue = RoFormerRMSNorm(dimensions: configuration.dim)
        self._bandSplit.wrappedValue = RoFormerBandSplit(configuration: configuration)
        self._maskEstimators.wrappedValue = (0..<configuration.numStems).map { _ in
            RoFormerMaskEstimator(configuration: configuration)
        }
        super.init()
    }

    public static func load(checkpoint: RoFormerCheckpoint, dtype: DType = .float16) throws -> BSRoFormer {
        let model = BSRoFormer(configuration: checkpoint.configuration)
        try model.validateCheckpoint(at: checkpoint.weightsURL)
        let arrays = try SafetensorsStreamingLoader.loadArrays(
            url: checkpoint.weightsURL,
            dtype: dtype
        )
        let parameters = try model.parameters().mapValues { key, _ in
            guard let value = arrays[key] else {
                throw RoFormerError.checkpointKeyMismatch(missing: [key], unexpected: [])
            }
            return value
        }
        try model.update(parameters: parameters, verify: .all)
        MLX.eval(model)
        return model
    }

    public func validateCheckpoint(at url: URL) throws {
        let metadata = try SafetensorsStreamingLoader.metadata(url: url)
        guard metadata.count == RoFormerResources.expectedTensorCount else {
            throw RoFormerError.invalidCheckpointInventory(
                expected: RoFormerResources.expectedTensorCount,
                actual: metadata.count
            )
        }
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        let sourceKeys = Set(metadata.keys)
        let expectedKeys = Set(expected.keys)
        guard sourceKeys == expectedKeys else {
            throw RoFormerError.checkpointKeyMismatch(
                missing: Array(expectedKeys.subtracting(sourceKeys)).sorted(),
                unexpected: Array(sourceKeys.subtracting(expectedKeys)).sorted()
            )
        }
        for key in sourceKeys.sorted() {
            guard let expectedShape = expected[key], let actualShape = metadata[key]?.shape else { continue }
            guard expectedShape == actualShape else {
                throw RoFormerError.checkpointShapeMismatch(
                    key: key,
                    expected: expectedShape,
                    actual: actualShape
                )
            }
        }
        let scalars = metadata.values.reduce(0) { count, tensor in
            count + tensor.shape.reduce(1, *)
        }
        guard scalars == RoFormerResources.expectedScalarCount else {
            throw RoFormerError.invalidConfiguration("checkpoint scalar count is \(scalars)")
        }
    }

    /// Runs one fixed-size channel-major chunk shaped `[batch, 2, samples]`.
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        precondition(audio.ndim == 3)
        let batch = audio.dim(0)
        let sampleCount = audio.dim(2)
        let window = RoFormerDSP.periodicHannWindow(
            length: configuration.stftWindowLength,
            dtype: audio.dtype
        )
        let representation = RoFormerDSP.stft(
            audio,
            nFFT: configuration.stftNFFT,
            hopLength: configuration.stftHopLength,
            window: window
        )
        let frames = representation.dim(2)
        let frequencyChannels = representation.dim(1)
        let features = representation.transposed(0, 2, 1, 3)
            .reshaped(batch, frames, frequencyChannels * 2)
        var hidden = bandSplit(features)
        for layer in layers {
            hidden = layer(hidden)
        }
        hidden = finalNorm(hidden)
        let masks = MLX.stacked(maskEstimators.map { estimator in
            estimator(hidden)
        }, axis: 1).reshaped(batch, configuration.numStems, frames, frequencyChannels, 2)
            .transposed(0, 1, 3, 2, 4)
        let sourceReal = representation[.ellipsis, 0].expandedDimensions(axis: 1)
        let sourceImaginary = representation[.ellipsis, 1].expandedDimensions(axis: 1)
        let maskReal = masks[.ellipsis, 0]
        let maskImaginary = masks[.ellipsis, 1]
        let masked = MLX.stacked([
            sourceReal * maskReal - sourceImaginary * maskImaginary,
            sourceReal * maskImaginary + sourceImaginary * maskReal,
        ], axis: -1)
        return RoFormerDSP.istft(
            masked,
            channels: configuration.audioChannels,
            length: sampleCount,
            nFFT: configuration.stftNFFT,
            hopLength: configuration.stftHopLength,
            window: window,
            zeroDC: configuration.zeroDC
        )
    }
}
