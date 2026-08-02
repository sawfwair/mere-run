import Foundation
@preconcurrency import MLX
import MLXNN

private func univerSRReflectPad2D(_ input: MLXArray, amount: Int) -> MLXArray {
    guard amount > 0 else { return input }
    precondition(input.dim(1) > amount && input.dim(2) > amount)
    let height = input.dim(1)
    let width = input.dim(2)
    let top = input[0..., 1..<(amount + 1), 0..., 0...][0..., .stride(by: -1), 0..., 0...]
    let bottom = input[0..., (height - amount - 1)..<(height - 1), 0..., 0...][
        0..., .stride(by: -1), 0..., 0...
    ]
    let vertical = MLX.concatenated([top, input, bottom], axis: 1)
    let left = vertical[0..., 0..., 1..<(amount + 1), 0...][0..., 0..., .stride(by: -1), 0...]
    let right = vertical[0..., 0..., (width - amount - 1)..<(width - 1), 0...][
        0..., 0..., .stride(by: -1), 0...
    ]
    return MLX.concatenated([left, vertical, right], axis: 2)
}

private func univerSRReflectPadFrames(_ input: MLXArray, amount: Int) -> MLXArray {
    guard amount > 0 else { return input }
    let width = input.dim(2)
    precondition(width > amount)
    let reflected = input[0..., 0..., (width - amount - 1)..<(width - 1), 0...][
        0..., 0..., .stride(by: -1), 0...
    ]
    return MLX.concatenated([input, reflected], axis: 2)
}

final class UniverSRGRN: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(dimensions: Int) {
        self._gamma.wrappedValue = MLX.zeros([1, 1, 1, dimensions], dtype: .float32)
        self._beta.wrappedValue = MLX.zeros([1, 1, 1, dimensions], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let response = MLX.sqrt(MLX.sum(input.square(), axes: [1, 2], keepDims: true))
        let normalized = response / (response.mean(axis: -1, keepDims: true) + 1e-6)
        return gamma * (input * normalized) + beta + input
    }
}

final class UniverSRConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var depthwiseConvolution: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var inputProjection: Linear
    @ModuleInfo(key: "grn") var grn: UniverSRGRN
    @ModuleInfo(key: "pwconv2") var outputProjection: Linear

    init(dimensions: Int) {
        self._depthwiseConvolution.wrappedValue = Conv2d(
            inputChannels: dimensions,
            outputChannels: dimensions,
            kernelSize: IntOrPair(7),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            dilation: IntOrPair(1),
            groups: dimensions,
            bias: true
        )
        self._norm.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        self._inputProjection.wrappedValue = Linear(dimensions, dimensions * 4, bias: true)
        self._grn.wrappedValue = UniverSRGRN(dimensions: dimensions * 4)
        self._outputProjection.wrappedValue = Linear(dimensions * 4, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = depthwiseConvolution(univerSRReflectPad2D(input, amount: 3))
        hidden = norm(hidden)
        hidden = inputProjection(hidden)
        hidden = MLXNN.gelu(hidden)
        hidden = grn(hidden)
        return input + outputProjection(hidden)
    }
}

final class UniverSRTimeAdapter: Module {
    @ModuleInfo(key: "0") var input: Linear
    @ModuleInfo(key: "2") var output: Linear

    init(timeDimensions: Int, outputDimensions: Int) {
        self._input.wrappedValue = Linear(timeDimensions, timeDimensions, bias: true)
        self._output.wrappedValue = Linear(timeDimensions, outputDimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(MLXNN.silu(input(value)))
    }
}

final class UniverSRBlockWithEmbedding: Module {
    @ModuleInfo(key: "block") var block: UniverSRConvNeXtBlock
    @ModuleInfo(key: "time_adapter") var timeAdapter: UniverSRTimeAdapter

    init(dimensions: Int, timeDimensions: Int) {
        self._block.wrappedValue = UniverSRConvNeXtBlock(dimensions: dimensions)
        self._timeAdapter.wrappedValue = UniverSRTimeAdapter(
            timeDimensions: timeDimensions,
            outputDimensions: dimensions
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, time: MLXArray) -> MLXArray {
        block(input + timeAdapter(time).expandedDimensions(axis: 1).expandedDimensions(axis: 1))
    }
}

final class UniverSRDownsampler: Module {
    @ModuleInfo(key: "0") var norm: LayerNorm
    @ModuleInfo(key: "1") var convolution: Conv2d

    init(inputDimensions: Int, outputDimensions: Int) {
        self._norm.wrappedValue = LayerNorm(dimensions: inputDimensions, eps: 1e-6)
        self._convolution.wrappedValue = Conv2d(
            inputChannels: inputDimensions,
            outputChannels: outputDimensions,
            kernelSize: IntOrPair(2),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        convolution(norm(input))
    }
}

final class UniverSREncoderBlock: Module {
    @ModuleInfo(key: "blocks") var blocks: [UniverSRBlockWithEmbedding]
    @ModuleInfo(key: "downsampler") var downsampler: UniverSRDownsampler

    init(inputDimensions: Int, outputDimensions: Int, depth: Int, timeDimensions: Int) {
        self._blocks.wrappedValue = (0..<depth).map { _ in
            UniverSRBlockWithEmbedding(dimensions: inputDimensions, timeDimensions: timeDimensions)
        }
        self._downsampler.wrappedValue = UniverSRDownsampler(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, time: MLXArray) -> MLXArray {
        var hidden = input
        for block in blocks {
            hidden = block(hidden, time: time)
        }
        return downsampler(hidden)
    }
}

final class UniverSRMidcoder: Module {
    @ModuleInfo(key: "blocks") var blocks: [UniverSRBlockWithEmbedding]

    init(dimensions: Int, depth: Int, timeDimensions: Int) {
        self._blocks.wrappedValue = (0..<depth).map { _ in
            UniverSRBlockWithEmbedding(dimensions: dimensions, timeDimensions: timeDimensions)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray, time: MLXArray) -> MLXArray {
        var hidden = input
        for block in blocks {
            hidden = block(hidden, time: time)
        }
        return hidden
    }
}

final class UniverSRDecoderBlock: Module {
    @ModuleInfo(key: "upsampler") var upsampler: ConvTransposed2d
    @ModuleInfo(key: "blocks") var blocks: [UniverSRBlockWithEmbedding]

    init(inputDimensions: Int, outputDimensions: Int, depth: Int, timeDimensions: Int) {
        self._upsampler.wrappedValue = ConvTransposed2d(
            inputChannels: inputDimensions,
            outputChannels: outputDimensions,
            kernelSize: IntOrPair(2),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            bias: true
        )
        self._blocks.wrappedValue = (0..<depth).map { _ in
            UniverSRBlockWithEmbedding(dimensions: outputDimensions, timeDimensions: timeDimensions)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray, time: MLXArray) -> MLXArray {
        var hidden = upsampler(input)
        for block in blocks {
            hidden = block(hidden, time: time)
        }
        return hidden
    }
}

final class UniverSRSRAdapter: Module {
    @ModuleInfo(key: "0") var input: Linear
    @ModuleInfo(key: "2") var output: Linear

    init(dimensions: Int) {
        self._input.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._output.wrappedValue = Linear(dimensions, dimensions * 2, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(MLXNN.gelu(input(value)))
    }
}

final class UniverSRConditioningEncoder: Module {
    @ModuleInfo(key: "film_generator") var filmGenerator: Linear
    @ModuleInfo(key: "head") var head: Conv2d
    @ModuleInfo(key: "sr_adapter") var sampleRateAdapter: UniverSRSRAdapter
    @ModuleInfo(key: "blocks") var blocks: [UniverSRConvNeXtBlock]

    init(dimensions: Int, layerCount: Int) {
        self._filmGenerator.wrappedValue = Linear(dimensions, 4, bias: true)
        self._head.wrappedValue = Conv2d(
            inputChannels: 2,
            outputChannels: dimensions,
            kernelSize: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        self._sampleRateAdapter.wrappedValue = UniverSRSRAdapter(dimensions: dimensions)
        self._blocks.wrappedValue = (0..<layerCount).map { _ in
            UniverSRConvNeXtBlock(dimensions: dimensions)
        }
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        frequencyEmbedding: MLXArray,
        sampleRateEmbedding: MLXArray
    ) -> MLXArray {
        let frequencyFilm = MLX.split(filmGenerator(frequencyEmbedding), parts: 2, axis: -1)
        let gamma = frequencyFilm[0].expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let beta = frequencyFilm[1].expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        var hidden = head(input * gamma + beta)

        let sampleRateFilm = MLX.split(sampleRateAdapter(sampleRateEmbedding), parts: 2, axis: -1)
        let sampleRateGamma = sampleRateFilm[0].expandedDimensions(axis: 1).expandedDimensions(axis: 1)
        let sampleRateBeta = sampleRateFilm[1].expandedDimensions(axis: 1).expandedDimensions(axis: 1)
        hidden = hidden * sampleRateGamma + sampleRateBeta
        for block in blocks {
            hidden = block(hidden)
        }
        return hidden.mean(axis: 1)
    }
}

final class UniverSRTimeEmbedding: Module {
    @ParameterInfo(key: "weights") var weights: MLXArray

    init(dimensions: Int) {
        self._weights.wrappedValue = MLX.zeros([1, dimensions / 2], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ time: MLXArray) -> MLXArray {
        let frequencies = time.reshaped(-1, 1) * weights * (2 * Float.pi)
        return MLX.concatenated([MLX.sin(frequencies), MLX.cos(frequencies)], axis: -1) * sqrt(Float(2))
    }
}

final class UniverSRFrequencyPositionEmbedding: Module {
    @ParameterInfo(key: "pe") var value: MLXArray

    init(binCount: Int, dimensions: Int) {
        self._value.wrappedValue = MLX.zeros([binCount, dimensions], dtype: .float32)
        super.init()
    }
}

final class UniverSRInitialConvolution: Module {
    @ModuleInfo(key: "0") var convolution: Conv2d
    @ModuleInfo(key: "1") var norm: LayerNorm

    init(inputChannels: Int, outputChannels: Int) {
        self._convolution.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        self._norm.wrappedValue = LayerNorm(dimensions: outputChannels, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        norm(convolution(input))
    }
}

public final class UniverSRModel: Module {
    @ParameterInfo(key: "uncond_emb") var unconditionalEmbedding: MLXArray
    @ModuleInfo(key: "time_embedder") var timeEmbedder: UniverSRTimeEmbedding
    @ModuleInfo(key: "sr_embedder") var sampleRateEmbedder: Embedding
    @ModuleInfo(key: "sr_projector") var sampleRateProjector: Linear
    @ModuleInfo(key: "freq_pos_enc") var frequencyPosition: UniverSRFrequencyPositionEmbedding
    @ModuleInfo(key: "film_generator") var filmGenerator: Linear
    @ModuleInfo(key: "conditioning_encoder") var conditioningEncoder: UniverSRConditioningEncoder
    @ModuleInfo(key: "init_conv") var initialConvolution: UniverSRInitialConvolution
    @ModuleInfo(key: "encoders") var encoders: [UniverSREncoderBlock]
    @ModuleInfo(key: "decoders") var decoders: [UniverSRDecoderBlock]
    @ModuleInfo(key: "midcoder") var midcoder: UniverSRMidcoder
    @ModuleInfo(key: "final_conv") var finalConvolution: Conv2d

    public let configuration: UniverSRConfiguration

    public init(configuration: UniverSRConfiguration) {
        self.configuration = configuration
        let graph = configuration.model
        self._unconditionalEmbedding.wrappedValue = MLX.zeros(
            [graph.conditioningDimension],
            dtype: .float32
        )
        self._timeEmbedder.wrappedValue = UniverSRTimeEmbedding(dimensions: graph.timeDimension)
        self._sampleRateEmbedder.wrappedValue = Embedding(
            embeddingCount: graph.inputRateBins.count,
            dimensions: graph.conditioningDimension
        )
        self._sampleRateProjector.wrappedValue = Linear(
            graph.conditioningDimension,
            graph.timeDimension,
            bias: true
        )
        self._frequencyPosition.wrappedValue = UniverSRFrequencyPositionEmbedding(
            binCount: graph.totalFrequencyBins,
            dimensions: graph.conditioningDimension
        )
        self._filmGenerator.wrappedValue = Linear(
            graph.conditioningDimension,
            graph.conditioningDimension * 2,
            bias: true
        )
        self._conditioningEncoder.wrappedValue = UniverSRConditioningEncoder(
            dimensions: graph.conditioningDimension,
            layerCount: graph.conditioningLayers
        )
        self._initialConvolution.wrappedValue = UniverSRInitialConvolution(
            inputChannels: graph.inputChannels + graph.conditioningDimension,
            outputChannels: graph.dimensions[0]
        )
        self._encoders.wrappedValue = graph.depths.indices.map { index in
            UniverSREncoderBlock(
                inputDimensions: graph.dimensions[index],
                outputDimensions: index + 1 < graph.dimensions.count
                    ? graph.dimensions[index + 1]
                    : graph.dimensions[index],
                depth: graph.depths[index],
                timeDimensions: graph.timeDimension
            )
        }
        self._midcoder.wrappedValue = UniverSRMidcoder(
            dimensions: graph.dimensions.last ?? 768,
            depth: graph.depths.last ?? 2,
            timeDimensions: graph.timeDimension
        )
        self._decoders.wrappedValue = graph.depths.indices.reversed().map { index in
            UniverSRDecoderBlock(
                inputDimensions: index + 1 < graph.dimensions.count
                    ? graph.dimensions[index + 1]
                    : graph.dimensions[index],
                outputDimensions: graph.dimensions[index],
                depth: graph.depths[index],
                timeDimensions: graph.timeDimension
            )
        }
        self._finalConvolution.wrappedValue = Conv2d(
            inputChannels: graph.dimensions[0],
            outputChannels: graph.outputChannels,
            kernelSize: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    public static func load(
        checkpoint: UniverSRCheckpoint,
        dtype: DType = .float32
    ) throws -> UniverSRModel {
        let model = UniverSRModel(configuration: checkpoint.configuration)
        let archive = try PyTorchStateDictArchive(url: checkpoint.weightsURL.resolvingSymlinksInPath())
        try model.validateCheckpoint(archive)

        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(archive.tensors.count)
        var pending: [MLXArray] = []
        var pendingBytes = 0
        for descriptor in archive.tensors {
            var value = try archive.loadArray(for: descriptor, dtype: dtype)
            if Self.isTransposedConvolutionWeight(descriptor.name) {
                value = Self.contiguous(value.transposed(1, 2, 3, 0))
            } else if descriptor.shape.count == 4 && descriptor.name.hasSuffix(".weight") {
                value = Self.contiguous(value.transposed(0, 2, 3, 1))
            }
            mapped[descriptor.name] = value
            pending.append(value)
            pendingBytes += value.size * Self.byteCount(dtype)
            if pendingBytes >= 64 * 1_024 * 1_024 {
                MLX.eval(pending)
                pending.removeAll(keepingCapacity: true)
                pendingBytes = 0
            }
        }
        if !pending.isEmpty { MLX.eval(pending) }
        let parameters = try model.parameters().mapValues { key, _ in
            guard let value = mapped[key] else {
                throw UniverSRError.checkpointKeyMismatch(missing: [key], unexpected: [])
            }
            return value
        }
        try model.update(parameters: parameters, verify: .all)
        MLX.eval(model)
        return model
    }

    public func validateCheckpoint(_ archive: PyTorchStateDictArchive) throws {
        let scalars = archive.tensors.reduce(0) { $0 + $1.elementCount }
        guard archive.tensors.count == UniverSRResources.expectedTensorCount,
              scalars == UniverSRResources.expectedScalarCount,
              archive.tensors.allSatisfy({ $0.dataType == .float32 }) else {
            throw UniverSRError.invalidCheckpointInventory(
                tensors: archive.tensors.count,
                scalars: scalars
            )
        }

        let expected = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) }
        )
        let source = Dictionary(
            uniqueKeysWithValues: archive.tensors.map { descriptor in
                (descriptor.name, Self.mappedShape(for: descriptor))
            }
        )
        let expectedKeys = Set(expected.keys)
        let sourceKeys = Set(source.keys)
        guard expectedKeys == sourceKeys else {
            throw UniverSRError.checkpointKeyMismatch(
                missing: Array(expectedKeys.subtracting(sourceKeys)).sorted(),
                unexpected: Array(sourceKeys.subtracting(expectedKeys)).sorted()
            )
        }
        for key in expectedKeys.sorted() {
            guard let expectedShape = expected[key], let actualShape = source[key] else { continue }
            guard expectedShape == actualShape else {
                throw UniverSRError.checkpointShapeMismatch(
                    key: key,
                    expected: expectedShape,
                    actual: actualShape
                )
            }
        }
    }

    /// Evaluates the conditional vector field on NHWC complex-spectral features.
    public func callAsFunction(
        _ input: MLXArray,
        time: MLXArray,
        condition: MLXArray?,
        inputRateKHz: Int
    ) -> MLXArray {
        let graph = configuration.model
        guard let lowBinCount = graph.frequencyBins(for: inputRateKHz) else {
            preconditionFailure("Unsupported UniverSR input rate: \(inputRateKHz) kHz")
        }
        let originalFrameCount = input.dim(2)
        let stride = 1 << graph.dimensions.count
        let padding = (stride - originalFrameCount % stride) % stride
        var hidden = univerSRReflectPadFrames(input, amount: padding)
        let paddedCondition = condition.map { univerSRReflectPadFrames($0, amount: padding) }
        let batch = hidden.dim(0)
        let frameCount = hidden.dim(2)

        let rateIndex = Self.sampleRateIndex(inputRateKHz)
        let rateEmbedding = sampleRateEmbedder(MLXArray([Int32(rateIndex)]))
        let tiledRate = MLX.tiled(rateEmbedding, repetitions: [batch, 1])
        let timeEmbedding = timeEmbedder(time) + sampleRateProjector(tiledRate)
        let fullFrequencyEmbedding = frequencyPosition.value
        let lowFrequencyEmbedding = fullFrequencyEmbedding[0..<lowBinCount, 0...]
        let highStart = graph.totalFrequencyBins - graph.highFrequencyBins
        let highFrequencyEmbedding = fullFrequencyEmbedding[highStart..<graph.totalFrequencyBins, 0...]

        let encodedCondition: MLXArray
        if let paddedCondition {
            encodedCondition = conditioningEncoder(
                paddedCondition,
                frequencyEmbedding: lowFrequencyEmbedding,
                sampleRateEmbedding: tiledRate
            )
        } else {
            encodedCondition = MLX.tiled(
                unconditionalEmbedding.reshaped(1, 1, graph.conditioningDimension),
                repetitions: [batch, frameCount, 1]
            )
        }

        let highFilm = MLX.split(filmGenerator(highFrequencyEmbedding), parts: 2, axis: -1)
        let gamma = highFilm[0].expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let beta = highFilm[1].expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let spatialCondition = encodedCondition.expandedDimensions(axis: 1) * gamma + beta
        hidden = initialConvolution(MLX.concatenated([hidden, spatialCondition], axis: -1))

        var skips = [hidden]
        for encoder in encoders {
            hidden = encoder(hidden, time: timeEmbedding)
            skips.append(hidden)
            MLX.eval(hidden)
        }
        hidden = midcoder(hidden, time: timeEmbedding)
        MLX.eval(hidden)
        for decoder in decoders {
            guard let skip = skips.popLast() else { preconditionFailure("UniverSR skip underflow") }
            precondition(hidden.shape == skip.shape)
            hidden = decoder(hidden + skip, time: timeEmbedding)
            MLX.eval(hidden)
        }
        guard let finalSkip = skips.popLast() else { preconditionFailure("UniverSR skip underflow") }
        hidden = finalConvolution(hidden + finalSkip)
        return padding > 0 ? hidden[0..., 0..., 0..<originalFrameCount, 0...] : hidden
    }

    private static func sampleRateIndex(_ inputRateKHz: Int) -> Int {
        switch inputRateKHz {
        case 8: 0
        case 12: 1
        case 16: 2
        case 24: 3
        default: preconditionFailure("Unsupported UniverSR input rate: \(inputRateKHz) kHz")
        }
    }

    private static func mappedShape(for descriptor: PyTorchTensorDescriptor) -> [Int] {
        if isTransposedConvolutionWeight(descriptor.name) {
            return [descriptor.shape[1], descriptor.shape[2], descriptor.shape[3], descriptor.shape[0]]
        }
        if descriptor.shape.count == 4 && descriptor.name.hasSuffix(".weight") {
            return [descriptor.shape[0], descriptor.shape[2], descriptor.shape[3], descriptor.shape[1]]
        }
        return descriptor.shape
    }

    private static func isTransposedConvolutionWeight(_ key: String) -> Bool {
        key.hasPrefix("decoders.") && key.hasSuffix(".upsampler.weight")
    }

    private static func contiguous(_ value: MLXArray) -> MLXArray {
        value.reshaped(-1).reshaped(value.shape)
    }

    private static func byteCount(_ dtype: DType) -> Int {
        switch dtype {
        case .bool, .int8, .uint8: 1
        case .float16, .bfloat16, .int16, .uint16: 2
        case .float32, .int32, .uint32: 4
        case .float64, .int64, .uint64, .complex64: 8
        }
    }
}
