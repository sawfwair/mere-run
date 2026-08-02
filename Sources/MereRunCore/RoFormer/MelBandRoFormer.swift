import Foundation
@preconcurrency import MLX
import MLXNN

private final class MelBandTransformerLayer: Module {
    @ModuleInfo(key: "0") var attention: RoFormerAttention
    @ModuleInfo(key: "1") var feedForward: RoFormerFeedForward

    init(configuration: MelBandRoFormerConfiguration) {
        self._attention.wrappedValue = RoFormerAttention(
            dimensions: configuration.dim,
            heads: configuration.heads,
            headDimension: configuration.dimHead
        )
        self._feedForward.wrappedValue = RoFormerFeedForward(
            dimensions: configuration.dim,
            expansionFactor: configuration.transformerExpansionFactor
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(input)
        return attended + feedForward(attended)
    }
}

private final class MelBandTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MelBandTransformerLayer]
    @ModuleInfo(key: "norm") var norm: RoFormerRMSNorm

    init(configuration: MelBandRoFormerConfiguration, depth: Int) {
        self._layers.wrappedValue = (0..<depth).map { _ in
            MelBandTransformerLayer(configuration: configuration)
        }
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: configuration.dim)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        norm(layers.reduce(input) { hidden, layer in layer(hidden) })
    }
}

private final class MelBandAxialLayer: Module {
    @ModuleInfo(key: "0") var timeTransformer: MelBandTransformer
    @ModuleInfo(key: "1") var frequencyTransformer: MelBandTransformer

    init(configuration: MelBandRoFormerConfiguration) {
        self._timeTransformer.wrappedValue = MelBandTransformer(
            configuration: configuration,
            depth: configuration.timeTransformerDepth
        )
        self._frequencyTransformer.wrappedValue = MelBandTransformer(
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

private final class MelBandProjection: Module {
    @ModuleInfo(key: "0") var norm: RoFormerRMSNorm
    @ModuleInfo(key: "1") var projection: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: inputDimensions)
        self._projection.wrappedValue = Linear(inputDimensions, outputDimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { projection(norm(input)) }
}

private final class MelBandSplit: Module {
    @ModuleInfo(key: "to_features") var projections: [MelBandProjection]
    private let inputDimensions: [Int]

    init(configuration: MelBandRoFormerConfiguration) {
        self.inputDimensions = configuration.frequencyLayout.inputDimensions
        self._projections.wrappedValue = inputDimensions.map {
            MelBandProjection(inputDimensions: $0, outputDimensions: configuration.dim)
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

private final class MelBandMaskMLP: Module {
    @ModuleInfo(key: "0") var inputProjection: Linear
    @ModuleInfo(key: "2") var middleProjection: Linear
    @ModuleInfo(key: "4") var outputProjection: Linear

    init(dimensions: Int, outputDimensions: Int, expansionFactor: Int) {
        let hiddenDimensions = dimensions * expansionFactor
        self._inputProjection.wrappedValue = Linear(dimensions, hiddenDimensions, bias: true)
        self._middleProjection.wrappedValue = Linear(hiddenDimensions, hiddenDimensions, bias: true)
        self._outputProjection.wrappedValue = Linear(
            hiddenDimensions,
            outputDimensions * 2,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = MLX.tanh(inputProjection(input))
        let projected = outputProjection(MLX.tanh(middleProjection(hidden)))
        let parts = MLX.split(projected, parts: 2, axis: -1)
        return parts[0] * MLX.sigmoid(parts[1])
    }
}

private final class MelBandMaskBand: Module {
    @ModuleInfo(key: "0") var mlp: MelBandMaskMLP

    init(configuration: MelBandRoFormerConfiguration, outputDimensions: Int) {
        self._mlp.wrappedValue = MelBandMaskMLP(
            dimensions: configuration.dim,
            outputDimensions: outputDimensions,
            expansionFactor: configuration.mlpExpansionFactor
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { mlp(input) }
}

private final class MelBandMaskEstimator: Module {
    @ModuleInfo(key: "to_freqs") var bands: [MelBandMaskBand]

    init(configuration: MelBandRoFormerConfiguration) {
        self._bands.wrappedValue = configuration.frequencyLayout.inputDimensions.map {
            MelBandMaskBand(configuration: configuration, outputDimensions: $0)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLX.concatenated(zip(bands, 0..<bands.count).map { band, index in
            band(input[0..., 0..., index, 0...])
        }, axis: -1)
    }
}

public final class MelBandRoFormer: Module {
    @ModuleInfo(key: "layers") private var layers: [MelBandAxialLayer]
    @ModuleInfo(key: "band_split") private var bandSplit: MelBandSplit
    @ModuleInfo(key: "mask_estimators") private var maskEstimators: [MelBandMaskEstimator]

    public let configuration: MelBandRoFormerConfiguration
    private let frequencyChannelIndexValues: [Int]
    private let maskDenominatorValues: [Float]

    public init(configuration: MelBandRoFormerConfiguration) {
        self.configuration = configuration
        let layout = configuration.frequencyLayout
        self.frequencyChannelIndexValues = layout.interleavedFrequencyChannelIndices
        let channelCounts = layout.bandsPerFrequency.flatMap { count in
            [Int](repeating: count, count: configuration.audioChannels)
        }
        self.maskDenominatorValues = channelCounts.map(Float.init)
        self._layers.wrappedValue = (0..<configuration.depth).map { _ in
            MelBandAxialLayer(configuration: configuration)
        }
        self._bandSplit.wrappedValue = MelBandSplit(configuration: configuration)
        self._maskEstimators.wrappedValue = (0..<configuration.numStems).map { _ in
            MelBandMaskEstimator(configuration: configuration)
        }
        super.init()
    }

    public static func load(
        checkpoint: MelBandRoFormerCheckpoint,
        dtype: DType = .float16
    ) throws -> MelBandRoFormer {
        let model = MelBandRoFormer(configuration: checkpoint.configuration)
        try model.validateCheckpoint(checkpoint: checkpoint)
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

    public func validateCheckpoint(checkpoint: MelBandRoFormerCheckpoint) throws {
        let metadata = try SafetensorsStreamingLoader.metadata(url: checkpoint.weightsURL)
        guard metadata.count == checkpoint.profile.expectedTensorCount else {
            throw RoFormerError.invalidCheckpointInventory(
                expected: checkpoint.profile.expectedTensorCount,
                actual: metadata.count
            )
        }
        let expected = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) }
        )
        let sourceKeys = Set(metadata.keys)
        let expectedKeys = Set(expected.keys)
        guard sourceKeys == expectedKeys else {
            throw RoFormerError.checkpointKeyMismatch(
                missing: Array(expectedKeys.subtracting(sourceKeys)).sorted(),
                unexpected: Array(sourceKeys.subtracting(expectedKeys)).sorted()
            )
        }
        for key in sourceKeys.sorted() {
            guard let expectedShape = expected[key], let actualShape = metadata[key]?.shape else {
                continue
            }
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
        guard scalars == checkpoint.profile.expectedScalarCount else {
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
        let frequencyChannels = representation.dim(1)
        let frames = representation.dim(2)
        let frequencyChannelIndices = MLXArray(frequencyChannelIndexValues)
        let gathered = MLX.take(representation, frequencyChannelIndices, axis: 1)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, frames, -1)
        var hidden = bandSplit(gathered)
        for layer in layers {
            hidden = layer(hidden)
        }
        let masks = MLX.stacked(maskEstimators.map { estimator in
            estimator(hidden)
        }, axis: 1).reshaped(
            batch,
            configuration.numStems,
            frames,
            frequencyChannelIndexValues.count,
            2
        ).transposed(0, 1, 3, 2, 4)
        let summedMasks = MLX.zeros(
            [batch, configuration.numStems, frequencyChannels, frames, 2],
            dtype: masks.dtype
        ).at[0..., 0..., frequencyChannelIndices, 0..., 0...].add(masks)
        let maskDenominator = MLXArray(maskDenominatorValues).reshaped(
            1,
            1,
            frequencyChannels,
            1,
            1
        )
        let averagedMasks = summedMasks / maskDenominator.asType(summedMasks.dtype)
        let sourceReal = representation[.ellipsis, 0].expandedDimensions(axis: 1)
        let sourceImaginary = representation[.ellipsis, 1].expandedDimensions(axis: 1)
        let maskReal = averagedMasks[.ellipsis, 0]
        let maskImaginary = averagedMasks[.ellipsis, 1]
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
