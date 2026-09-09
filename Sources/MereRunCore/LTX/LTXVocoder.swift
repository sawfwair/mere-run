import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

final class LTXVocoder: LTXAudioVocoderBase {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resBlocks: [LTXVocoderResidualBlock]
    @ModuleInfo(key: "act_post") var actPost: LTXVocoderActivation1d?
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    let numKernels: Int
    let blockKind: LTXVocoderResBlockKind
    let applyFinalActivation: Bool
    let useTanhAtFinal: Bool

    init(
        blockKind: LTXVocoderResBlockKind = .legacy,
        outputSamplingRate: Int = LTXAudioSampleRate,
        activation: LTXVocoderActivationKind = .leaky,
        applyFinalActivation: Bool = true,
        useTanhAtFinal: Bool = true,
        useBiasAtFinal: Bool = true,
        inputChannels: Int = 128,
        outputChannels: Int = 2,
        upsampleInitialChannels: Int = 1024,
        upsampleRates: [Int] = [6, 5, 2, 2, 2],
        upsampleKernelSizes: [Int] = [16, 15, 8, 4, 4],
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ) {
        self.blockKind = blockKind
        self.applyFinalActivation = applyFinalActivation
        self.useTanhAtFinal = useTanhAtFinal
        self.numKernels = resblockKernelSizes.count

        self._convPre.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: upsampleInitialChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: true
        )

        let upLayers = upsampleRates.enumerated().map { idx, stride in
            let kernel = upsampleKernelSizes[idx]
            let inCh = upsampleInitialChannels / (1 << idx)
            let outCh = upsampleInitialChannels / (1 << (idx + 1))
            return ConvTransposed1d(
                inputChannels: inCh,
                outputChannels: outCh,
                kernelSize: kernel,
                stride: stride,
                padding: (kernel - stride) / 2,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }
        self._ups.wrappedValue = upLayers

        var blocks: [LTXVocoderResidualBlock] = []
        for i in 0..<upLayers.count {
            let channels = upsampleInitialChannels / (1 << (i + 1))
            for (kernelIndex, kernel) in resblockKernelSizes.enumerated() {
                let dilations = kernelIndex < resblockDilationSizes.count
                    ? resblockDilationSizes[kernelIndex]
                    : [1, 3, 5]
                switch blockKind {
                case .legacy:
                    blocks.append(LTXVocoderResBlock1(channels: channels, kernelSize: kernel, dilations: dilations))
                case .amp:
                    blocks.append(LTXVocoderAMPBlock1(
                        channels: channels,
                        kernelSize: kernel,
                        dilations: dilations,
                        activation: activation
                    ))
                }
            }
        }
        self._resBlocks.wrappedValue = blocks

        let finalChannels = upsampleInitialChannels / (1 << upLayers.count)
        self._actPost.wrappedValue = blockKind == .amp
            ? LTXVocoderActivation1d(channels: finalChannels, kind: activation)
            : nil
        self._convPost.wrappedValue = Conv1d(
            inputChannels: finalChannels,
            outputChannels: outputChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: useBiasAtFinal
        )
        super.init(outputSamplingRate: outputSamplingRate)
    }

    convenience init(
        architecture: LTXVocoderArchitectureConfig,
        outputSamplingRate: Int
    ) {
        self.init(
            blockKind: architecture.blockKind,
            outputSamplingRate: outputSamplingRate,
            activation: architecture.activation,
            applyFinalActivation: architecture.applyFinalActivation,
            useTanhAtFinal: architecture.useTanhAtFinal,
            useBiasAtFinal: architecture.useBiasAtFinal,
            inputChannels: architecture.inputChannels,
            outputChannels: architecture.outputChannels,
            upsampleInitialChannels: architecture.upsampleInitialChannels,
            upsampleRates: architecture.upsampleRates,
            upsampleKernelSizes: architecture.upsampleKernelSizes,
            resblockKernelSizes: architecture.resblockKernelSizes,
            resblockDilationSizes: architecture.resblockDilationSizes
        )
    }

    override func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed(0, 1, 3, 2) // [B, 2, 64, T]
        let b = x.dim(0)
        let stereo = x.dim(1)
        let melBins = x.dim(2)
        let time = x.dim(3)
        x = x.reshaped(b, stereo * melBins, time).transposed(0, 2, 1) // [B, T, 128]

        x = convPre(x)

        for i in 0..<ups.count {
            if blockKind == .legacy {
                x = ltxLeakyRelu(x, slope: 0.1)
            }
            x = ups[i](x)
            var blockOutputs: [MLXArray] = []
            blockOutputs.reserveCapacity(numKernels)
            let start = i * numKernels
            for j in 0..<numKernels {
                blockOutputs.append(resBlocks[start + j](x))
            }
            x = MLX.mean(MLX.stacked(blockOutputs, axis: 0), axis: 0)
        }

        if let actPost {
            x = actPost(x)
        } else {
            x = ltxLeakyRelu(x, slope: 0.01)
        }
        x = convPost(x)
        if applyFinalActivation {
            x = useTanhAtFinal
                ? tanh(x)
                : MLX.clip(x, min: MLXArray(-1.0).asType(x.dtype), max: MLXArray(1.0).asType(x.dtype))
        }
        return x.transposed(0, 2, 1) // [B, 2, samples]
    }
}

final class LTXSTFTFn: Module {
    @ModuleInfo(key: "forward_basis") var forwardBasis: MLXArray
    @ModuleInfo(key: "inverse_basis") var inverseBasis: MLXArray
    let hopLength: Int
    let winLength: Int

    init(filterLength: Int, hopLength: Int, winLength: Int) {
        let frequencies = filterLength / 2 + 1
        self.hopLength = hopLength
        self.winLength = winLength
        self._forwardBasis.wrappedValue = MLXArray.zeros([frequencies * 2, filterLength, 1])
        self._inverseBasis.wrappedValue = MLXArray.zeros([frequencies * 2, filterLength, 1])
    }

    func magnitude(_ y: MLXArray) -> MLXArray {
        var input = y
        if input.ndim == 2 {
            input = input.expandedDimensions(axis: 2)
        }
        let leftPad = max(0, winLength - hopLength)
        input = padded(input, widths: [[0, 0], [leftPad, 0], [0, 0]])
        let spec = MLX.conv1d(input, forwardBasis.asType(input.dtype), stride: hopLength, padding: 0)
        let frequencies = spec.dim(2) / 2
        let real = spec[0..., 0..., 0..<frequencies]
        let imag = spec[0..., 0..., frequencies..<(frequencies * 2)]
        return MLX.sqrt(real * real + imag * imag)
    }
}

final class LTXMelSTFT: Module {
    @ModuleInfo(key: "stft_fn") var stftFn: LTXSTFTFn
    @ModuleInfo(key: "mel_basis") var melBasis: MLXArray

    init(filterLength: Int = 1024, hopLength: Int = 60, winLength: Int = 1024, melChannels: Int = 128) {
        self._stftFn.wrappedValue = LTXSTFTFn(
            filterLength: filterLength,
            hopLength: hopLength,
            winLength: winLength
        )
        self._melBasis.wrappedValue = MLXArray.zeros([melChannels, filterLength / 2 + 1])
    }

    func melSpectrogram(_ y: MLXArray) -> MLXArray {
        let magnitude = stftFn.magnitude(y)
        let basis = melBasis.asType(magnitude.dtype)
        let mel = MLX.matmul(magnitude, basis.transposed())
        return MLX.log(MLX.maximum(mel, MLXArray(1e-5).asType(mel.dtype)))
    }
}

final class LTXVocoderWithBWE: LTXAudioVocoderBase {
    @ModuleInfo(key: "vocoder") var vocoder: LTXVocoder
    @ModuleInfo(key: "bwe_generator") var bweGenerator: LTXVocoder
    @ModuleInfo(key: "mel_stft") var melSTFT: LTXMelSTFT
    let inputSamplingRate: Int
    let hopLength: Int
    let resampler: LTXSincUpsample1d

    convenience init(config: LTXBWEVocoderRuntimeConfig) {
        self.init(
            inputSamplingRate: config.inputSamplingRate,
            outputSamplingRate: config.outputSamplingRate,
            hopLength: config.hopLength,
            filterLength: config.filterLength,
            melChannels: config.melChannels,
            baseVocoder: config.baseVocoder,
            bandwidthExtensionVocoder: config.bandwidthExtensionVocoder
        )
    }

    init(
        inputSamplingRate: Int = LTXAudioSampleRate,
        outputSamplingRate: Int = 48_000,
        hopLength: Int = 60,
        filterLength: Int = 1024,
        melChannels: Int = 128,
        baseVocoder: LTXVocoderArchitectureConfig = .defaultBWEBase,
        bandwidthExtensionVocoder: LTXVocoderArchitectureConfig = .defaultBWEGenerator
    ) {
        self.inputSamplingRate = inputSamplingRate
        self.hopLength = hopLength
        self._vocoder.wrappedValue = LTXVocoder(architecture: baseVocoder, outputSamplingRate: inputSamplingRate)
        self._bweGenerator.wrappedValue = LTXVocoder(
            architecture: bandwidthExtensionVocoder,
            outputSamplingRate: outputSamplingRate
        )
        self._melSTFT.wrappedValue = LTXMelSTFT(
            filterLength: filterLength,
            hopLength: hopLength,
            winLength: filterLength,
            melChannels: melChannels
        )
        self.resampler = LTXSincUpsample1d(
            ratio: max(1, outputSamplingRate / inputSamplingRate),
            windowType: "hann"
        )
        super.init(outputSamplingRate: outputSamplingRate)
    }

    override func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let inputDType = mel.dtype
        var lowRate = vocoder(mel.asType(.float32))
        let lowRateLength = lowRate.dim(2)
        let outputLength = lowRateLength * outputSamplingRate / inputSamplingRate
        saveLTXAVDebugAudio(lowRate, suffix: "bwe_low_rate", sampleRate: inputSamplingRate)

        let remainder = lowRateLength % hopLength
        if remainder != 0 {
            lowRate = padded(lowRate, widths: [[0, 0], [0, 0], [0, hopLength - remainder]])
        }

        let batch = lowRate.dim(0)
        let channels = lowRate.dim(1)
        let flattened = lowRate.reshaped(batch * channels, lowRate.dim(2))
        let computedMel = melSTFT.melSpectrogram(flattened)
            .reshaped(batch, channels, -1, melSTFT.melBasis.dim(0))
        saveLTXAVDebugArray(computedMel, suffix: "bwe_computed_mel")
        let residual = bweGenerator(computedMel)
        saveLTXAVDebugAudio(residual, suffix: "bwe_residual", sampleRate: outputSamplingRate)
        let skip = resampler(lowRate.transposed(0, 2, 1)).transposed(0, 2, 1)
        saveLTXAVDebugAudio(skip, suffix: "bwe_skip", sampleRate: outputSamplingRate)

        let mixedLength = min(residual.dim(2), skip.dim(2))
        var mixed = residual[0..., 0..., 0..<mixedLength] + skip[0..., 0..., 0..<mixedLength]
        mixed = MLX.clip(mixed, min: MLXArray(-1.0).asType(mixed.dtype), max: MLXArray(1.0).asType(mixed.dtype))
        saveLTXAVDebugAudio(mixed, suffix: "bwe_mixed", sampleRate: outputSamplingRate)

        let cropLength = min(outputLength, mixed.dim(2))
        mixed = mixed[0..., 0..., 0..<cropLength]
        if cropLength < outputLength {
            mixed = padded(mixed, widths: [[0, 0], [0, 0], [0, outputLength - cropLength]])
        }
        return mixed.asType(inputDType)
    }
}
