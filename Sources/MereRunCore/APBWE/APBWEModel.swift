import Foundation
@preconcurrency import MLX
import MLXNN

final class APBWEConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var depthwiseConvolution: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var inputProjection: Linear
    @ModuleInfo(key: "pwconv2") var outputProjection: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int, layerScale: Float) {
        self._depthwiseConvolution.wrappedValue = Conv1d(
            inputChannels: dimensions,
            outputChannels: dimensions,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: dimensions,
            bias: true
        )
        self._norm.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        self._inputProjection.wrappedValue = Linear(
            dimensions,
            dimensions * 3,
            bias: true
        )
        self._outputProjection.wrappedValue = Linear(
            dimensions * 3,
            dimensions,
            bias: true
        )
        self._gamma.wrappedValue = MLXArray.ones([dimensions]) * layerScale
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let residual = input
        var hidden = depthwiseConvolution(input)
        hidden = norm(hidden)
        hidden = inputProjection(hidden)
        hidden = Self.exactGELU(hidden)
        hidden = outputProjection(hidden)
        return residual + hidden * gamma
    }

    private static func exactGELU(_ input: MLXArray) -> MLXArray {
        input * MLXArray(0.5).asType(input.dtype)
            * (MLXArray(1).asType(input.dtype) + MLX.erf(input / sqrt(Float(2))))
    }
}

public final class APBWEModel: Module {
    @ModuleInfo(key: "conv_pre_mag") var magnitudeInputConvolution: Conv1d
    @ModuleInfo(key: "norm_pre_mag") var magnitudeInputNorm: LayerNorm
    @ModuleInfo(key: "conv_pre_pha") var phaseInputConvolution: Conv1d
    @ModuleInfo(key: "norm_pre_pha") var phaseInputNorm: LayerNorm
    @ModuleInfo(key: "convnext_mag") var magnitudeBlocks: [APBWEConvNeXtBlock]
    @ModuleInfo(key: "convnext_pha") var phaseBlocks: [APBWEConvNeXtBlock]
    @ModuleInfo(key: "norm_post_mag") var magnitudeOutputNorm: LayerNorm
    @ModuleInfo(key: "norm_post_pha") var phaseOutputNorm: LayerNorm
    @ModuleInfo(key: "linear_post_mag") var magnitudeOutput: Linear
    @ModuleInfo(key: "linear_post_pha_r") var phaseRealOutput: Linear
    @ModuleInfo(key: "linear_post_pha_i") var phaseImaginaryOutput: Linear

    public let configuration: APBWEConfiguration

    public init(configuration: APBWEConfiguration) {
        self.configuration = configuration
        let bins = configuration.frequencyBins
        let dimensions = configuration.channels
        let layerScale = 1 / Float(configuration.layers)
        self._magnitudeInputConvolution.wrappedValue = Conv1d(
            inputChannels: bins,
            outputChannels: dimensions,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self._magnitudeInputNorm.wrappedValue = LayerNorm(
            dimensions: dimensions,
            eps: 1e-6
        )
        self._phaseInputConvolution.wrappedValue = Conv1d(
            inputChannels: bins,
            outputChannels: dimensions,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self._phaseInputNorm.wrappedValue = LayerNorm(
            dimensions: dimensions,
            eps: 1e-6
        )
        self._magnitudeBlocks.wrappedValue = (0..<configuration.layers).map { _ in
            APBWEConvNeXtBlock(dimensions: dimensions, layerScale: layerScale)
        }
        self._phaseBlocks.wrappedValue = (0..<configuration.layers).map { _ in
            APBWEConvNeXtBlock(dimensions: dimensions, layerScale: layerScale)
        }
        self._magnitudeOutputNorm.wrappedValue = LayerNorm(
            dimensions: dimensions,
            eps: 1e-6
        )
        self._phaseOutputNorm.wrappedValue = LayerNorm(
            dimensions: dimensions,
            eps: 1e-6
        )
        self._magnitudeOutput.wrappedValue = Linear(dimensions, bins, bias: true)
        self._phaseRealOutput.wrappedValue = Linear(dimensions, bins, bias: true)
        self._phaseImaginaryOutput.wrappedValue = Linear(dimensions, bins, bias: true)
        super.init()
    }

    public static func load(
        checkpoint: APBWECheckpoint,
        dtype: DType = .float32
    ) throws -> APBWEModel {
        let model = APBWEModel(configuration: checkpoint.configuration)
        let archive = try PyTorchStateDictArchive(
            url: checkpoint.weightsURL.resolvingSymlinksInPath(),
            verifyEntryChecksums: false
        )
        try model.validateCheckpoint(archive)

        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(archive.tensors.count)
        var pending: [MLXArray] = []
        var pendingBytes = 0
        for descriptor in archive.tensors {
            var value = try archive.loadArray(for: descriptor, dtype: dtype)
            if Self.isConvolutionWeight(descriptor.name) {
                value = value.transposed(0, 2, 1)
                value = value.reshaped(-1).reshaped(value.shape)
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
        if !pending.isEmpty {
            MLX.eval(pending)
        }
        try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        MLX.eval(model)
        return model
    }

    public func validateCheckpoint(_ archive: PyTorchStateDictArchive) throws {
        let scalars = archive.tensors.reduce(0) { $0 + $1.elementCount }
        guard archive.tensors.count == APBWEResources.expectedTensorCount,
              scalars == APBWEResources.expectedScalarCount,
              archive.tensors.allSatisfy({ $0.dataType == .float32 }) else {
            throw APBWEError.invalidCheckpointInventory(
                tensors: archive.tensors.count,
                scalars: scalars
            )
        }

        let expected = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) }
        )
        let source = Dictionary(
            uniqueKeysWithValues: archive.tensors.map { descriptor in
                let shape = Self.isConvolutionWeight(descriptor.name)
                    ? [descriptor.shape[0], descriptor.shape[2], descriptor.shape[1]]
                    : descriptor.shape
                return (descriptor.name, shape)
            }
        )
        let expectedKeys = Set(expected.keys)
        let sourceKeys = Set(source.keys)
        guard expectedKeys == sourceKeys else {
            throw APBWEError.checkpointKeyMismatch(
                missing: Array(expectedKeys.subtracting(sourceKeys)).sorted(),
                unexpected: Array(sourceKeys.subtracting(expectedKeys)).sorted()
            )
        }
        for key in expectedKeys.sorted() {
            guard let expectedShape = expected[key], let actualShape = source[key] else { continue }
            guard expectedShape == actualShape else {
                throw APBWEError.checkpointShapeMismatch(
                    key: key,
                    expected: expectedShape,
                    actual: actualShape
                )
            }
        }
    }

    /// Enhances one 48 kHz mono, channel-major chunk shaped `[batch, 1, samples]`.
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        precondition(audio.ndim == 3 && audio.dim(1) == 1)
        let sampleCount = audio.dim(2)
        let window = Self.paddedPeriodicHannWindow(
            nFFT: configuration.nFFT,
            winSize: configuration.winSize,
            dtype: audio.dtype
        )
        let spectrum = RoFormerDSP.stft(
            audio,
            nFFT: configuration.nFFT,
            hopLength: configuration.hopSize,
            window: window
        )
        let real = spectrum[.ellipsis, 0]
        let imaginary = spectrum[.ellipsis, 1]
        let logMagnitude = MLX.log(
            MLX.sqrt(real.square() + imaginary.square())
                + MLXArray(1e-4).asType(real.dtype)
        ).transposed(0, 2, 1)
        let phase = MLX.atan2(imaginary, real).transposed(0, 2, 1)

        var magnitudeHidden = magnitudeInputNorm(magnitudeInputConvolution(logMagnitude))
        var phaseHidden = phaseInputNorm(phaseInputConvolution(phase))
        for index in magnitudeBlocks.indices {
            magnitudeHidden = magnitudeHidden + phaseHidden
            phaseHidden = phaseHidden + magnitudeHidden
            magnitudeHidden = magnitudeBlocks[index](magnitudeHidden)
            phaseHidden = phaseBlocks[index](phaseHidden)
        }

        let widebandMagnitude = logMagnitude
            + magnitudeOutput(magnitudeOutputNorm(magnitudeHidden))
        let normalizedPhase = phaseOutputNorm(phaseHidden)
        let widebandPhase = MLX.atan2(
            phaseImaginaryOutput(normalizedPhase),
            phaseRealOutput(normalizedPhase)
        )
        let amplitude = MLX.exp(widebandMagnitude)
        let reconstructed = MLX.stacked([
            amplitude * MLX.cos(widebandPhase),
            amplitude * MLX.sin(widebandPhase),
        ], axis: -1).transposed(0, 2, 1, 3).expandedDimensions(axis: 1)
        return RoFormerDSP.istft(
            reconstructed,
            channels: 1,
            length: sampleCount,
            nFFT: configuration.nFFT,
            hopLength: configuration.hopSize,
            window: window,
            zeroDC: false
        )
    }

    private static func paddedPeriodicHannWindow(
        nFFT: Int,
        winSize: Int,
        dtype: DType
    ) -> MLXArray {
        let side = (nFFT - winSize) / 2
        return MLX.concatenated([
            MLX.zeros([side], dtype: dtype),
            RoFormerDSP.periodicHannWindow(length: winSize, dtype: dtype),
            MLX.zeros([nFFT - winSize - side], dtype: dtype),
        ])
    }

    private static func isConvolutionWeight(_ key: String) -> Bool {
        key == "conv_pre_mag.weight"
            || key == "conv_pre_pha.weight"
            || key.contains(".dwconv.weight")
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
