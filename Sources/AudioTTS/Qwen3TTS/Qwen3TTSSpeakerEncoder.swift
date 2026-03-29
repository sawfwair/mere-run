import Foundation
import MLX
import MLXNN

private final class SpeakerTimeDelayNetBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    private let pad: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
        self.pad = ((kernelSize - 1) * dilation) / 2
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation,
            groups: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        if pad > 0 {
            out = padded(out, widths: [[0, 0], [pad, pad], [0, 0]])
        }
        out = conv(out)
        out = out.transposed(0, 2, 1)
        return MLXNN.relu(out)
    }
}

private final class SpeakerRes2NetBlock: Module {
    let scale: Int
    @ModuleInfo(key: "blocks") var blocks: [SpeakerTimeDelayNetBlock]

    init(inChannels: Int, outChannels: Int, scale: Int, kernelSize: Int, dilation: Int) {
        self.scale = scale
        let inChunk = max(1, inChannels / scale)
        let hiddenChunk = max(1, outChannels / scale)
        self._blocks.wrappedValue = (0..<(scale - 1)).map { _ in
            SpeakerTimeDelayNetBlock(
                inChannels: inChunk,
                outChannels: hiddenChunk,
                kernelSize: kernelSize,
                dilation: dilation
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(1)
        let chunk = max(1, channels / scale)

        var chunks: [MLXArray] = []
        for idx in 0..<scale {
            let start = idx * chunk
            let end = idx == scale - 1 ? channels : min(channels, start + chunk)
            chunks.append(x[0..., start..<end, 0...])
        }

        var outputs: [MLXArray] = []
        var running: MLXArray?
        for idx in 0..<chunks.count {
            let part = chunks[idx]
            if idx == 0 {
                running = part
                outputs.append(part)
                continue
            }

            let input = (running != nil && idx > 1) ? (part + running!) : part
            let out = blocks[idx - 1](input)
            running = out
            outputs.append(out)
        }

        return MLX.concatenated(outputs, axis: 1)
    }
}

private final class SpeakerSqueezeExcitationBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(inChannels: Int, seChannels: Int, outChannels: Int) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: seChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: seChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xMean = MLX.mean(x, axis: 2, keepDims: true)
        xMean = xMean.transposed(0, 2, 1)
        xMean = conv1(xMean)
        xMean = MLXNN.relu(xMean)
        xMean = conv2(xMean)
        xMean = MLX.sigmoid(xMean)
        xMean = xMean.transposed(0, 2, 1)
        return x * xMean
    }
}

private final class SpeakerSERes2NetBlock: Module {
    @ModuleInfo(key: "tdnn1") var tdnn1: SpeakerTimeDelayNetBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: SpeakerRes2NetBlock
    @ModuleInfo(key: "tdnn2") var tdnn2: SpeakerTimeDelayNetBlock
    @ModuleInfo(key: "se_block") var seBlock: SpeakerSqueezeExcitationBlock

    init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int,
        seChannels: Int,
        kernelSize: Int,
        dilation: Int
    ) {
        self._tdnn1.wrappedValue = SpeakerTimeDelayNetBlock(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._res2netBlock.wrappedValue = SpeakerRes2NetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            scale: res2netScale,
            kernelSize: kernelSize,
            dilation: dilation
        )
        self._tdnn2.wrappedValue = SpeakerTimeDelayNetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._seBlock.wrappedValue = SpeakerSqueezeExcitationBlock(
            inChannels: outChannels,
            seChannels: seChannels,
            outChannels: outChannels
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = tdnn1(x)
        out = res2netBlock(out)
        out = tdnn2(out)
        out = seBlock(out)
        return out + residual
    }
}

private final class SpeakerAttentiveStatisticsPooling: Module {
    @ModuleInfo(key: "tdnn") var tdnn: SpeakerTimeDelayNetBlock
    @ModuleInfo(key: "conv") var conv: Conv1d
    private let eps: Float = 1e-12

    init(channels: Int, attentionChannels: Int) {
        self._tdnn.wrappedValue = SpeakerTimeDelayNetBlock(
            inChannels: channels * 3,
            outChannels: attentionChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._conv.wrappedValue = Conv1d(
            inputChannels: attentionChannels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let channels = x.dim(1)
        let sequence = x.dim(2)

        let mean = MLX.mean(x, axis: 2, keepDims: true)
        let variance = MLX.mean(MLX.square(x - mean), axis: 2, keepDims: true)
        let std = MLX.sqrt(variance + eps)

        let meanExpanded = broadcast(mean, to: [batch, channels, sequence])
        let stdExpanded = broadcast(std, to: [batch, channels, sequence])

        var attention = MLX.concatenated([x, meanExpanded, stdExpanded], axis: 1)
        attention = tdnn(attention)
        attention = MLX.tanh(attention)
        attention = attention.transposed(0, 2, 1)
        attention = conv(attention)
        attention = attention.transposed(0, 2, 1)
        attention = MLX.softmax(attention, axis: 2)

        let pooledMean = MLX.sum(attention * x, axis: 2, keepDims: true)
        let pooledVar = MLX.sum(attention * MLX.square(x - pooledMean), axis: 2, keepDims: true)
        let pooledStd = MLX.sqrt(MLX.maximum(pooledVar, MLXArray(eps)))
        return MLX.concatenated([pooledMean, pooledStd], axis: 1)
    }
}

public final class Qwen3TTSSpeakerEncoder: Module {
    public let config: Qwen3TTSSpeakerEncoderConfig

    @ModuleInfo(key: "blocks") var blocks: [Module]
    @ModuleInfo(key: "mfa") fileprivate var mfa: SpeakerTimeDelayNetBlock
    @ModuleInfo(key: "asp") fileprivate var asp: SpeakerAttentiveStatisticsPooling
    @ModuleInfo(key: "fc") var fc: Conv1d

    public init(config: Qwen3TTSSpeakerEncoderConfig) {
        self.config = config

        var layers: [Module] = []
        layers.append(
            SpeakerTimeDelayNetBlock(
                inChannels: config.melDim,
                outChannels: config.encChannels[0],
                kernelSize: config.encKernelSizes[0],
                dilation: config.encDilations[0]
            )
        )

        for idx in 1..<(config.encChannels.count - 1) {
            layers.append(
                SpeakerSERes2NetBlock(
                    inChannels: config.encChannels[idx - 1],
                    outChannels: config.encChannels[idx],
                    res2netScale: config.encRes2netScale,
                    seChannels: config.encSeChannels,
                    kernelSize: config.encKernelSizes[idx],
                    dilation: config.encDilations[idx]
                )
            )
        }

        self._blocks.wrappedValue = layers
        self._mfa.wrappedValue = SpeakerTimeDelayNetBlock(
            inChannels: config.encChannels.last ?? config.encDim,
            outChannels: config.encChannels.last ?? config.encDim,
            kernelSize: config.encKernelSizes.last ?? 1,
            dilation: config.encDilations.last ?? 1
        )
        self._asp.wrappedValue = SpeakerAttentiveStatisticsPooling(
            channels: config.encChannels.last ?? config.encDim,
            attentionChannels: config.encAttentionChannels
        )
        self._fc.wrappedValue = Conv1d(
            inputChannels: (config.encChannels.last ?? config.encDim) * 2,
            outputChannels: config.encDim,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
    }

    public func callAsFunction(_ mels: MLXArray) -> MLXArray {
        var x = mels.transposed(0, 2, 1)
        var hiddenStates: [MLXArray] = []

        for layer in blocks {
            if let tdnn = layer as? SpeakerTimeDelayNetBlock {
                x = tdnn(x)
            } else if let se = layer as? SpeakerSERes2NetBlock {
                x = se(x)
            }
            hiddenStates.append(x)
        }

        let aggregated: MLXArray
        if hiddenStates.count > 1 {
            aggregated = MLX.concatenated(Array(hiddenStates.dropFirst()), axis: 1)
        } else {
            aggregated = x
        }

        x = mfa(aggregated)
        x = asp(x)
        x = x.transposed(0, 2, 1)
        x = fc(x)
        x = x.transposed(0, 2, 1)
        return x.squeezed(axis: 2)
    }

    public func extractEmbedding(audio: [Float]) -> MLXArray {
        let mels = Qwen3TTSAudioPreprocessor.melSpectrogram(
            samples: audio,
            nMels: config.melDim,
            frameSize: 1024,
            hopSize: 256,
            sampleRate: config.sampleRate
        )
        let embedding = callAsFunction(mels)
        MLX.eval(embedding)
        return embedding
    }

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard key.hasPrefix("speaker_encoder.") else { continue }
            let mappedKey = String(key.dropFirst("speaker_encoder.".count))

            if mappedKey.hasSuffix("weight"), value.ndim == 3 {
                sanitized[mappedKey] = needsConvTranspose(value) ? value.transposed(0, 2, 1) : value
            } else {
                sanitized[mappedKey] = value
            }
        }
        return sanitized
    }

    private static func needsConvTranspose(_ value: MLXArray) -> Bool {
        let dim2 = value.dim(1)
        let dim3 = value.dim(2)
        if dim2 == 1 {
            return !(dim3 > 64)
        }
        if dim3 == 1 {
            return dim2 > 64
        }
        return dim2 >= dim3
    }
}
