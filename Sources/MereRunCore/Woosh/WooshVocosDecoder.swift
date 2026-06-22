import Foundation
import MLX
import MLXNN

final class WooshVocosConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int, intermediateDim: Int, layerScale: Float) {
        self._dwconv.wrappedValue = Conv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: dim,
            bias: true
        )
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim, bias: true)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim, bias: true)
        self._gamma.wrappedValue = MLXArray.ones([dim]) * layerScale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var hidden = dwconv(x)
        hidden = norm(hidden)
        hidden = pwconv1(hidden)
        hidden = WooshTensorOps.geluTanh(hidden)
        hidden = pwconv2(hidden)
        return residual + hidden * gamma
    }
}

final class WooshVocosBackbone: Module {
    @ModuleInfo(key: "embed") var embed: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "convnext") var convnext: [WooshVocosConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: WooshVocosConfig, inputChannels: Int? = nil) {
        let inputChannels = inputChannels ?? config.zDim
        self._embed.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: config.dModel,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self._norm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-6)
        let scale = 1.0 / Float(max(1, config.numLayers))
        self._convnext.wrappedValue = (0..<config.numLayers).map { _ in
            WooshVocosConvNeXtBlock(dim: config.dModel, intermediateDim: config.intermediateDim, layerScale: scale)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = embed(x)
        hidden = norm(hidden)
        for layer in convnext {
            hidden = layer(hidden)
        }
        return finalLayerNorm(hidden)
    }
}

enum WooshSTFT {
    static func complexEmbedding(samples: [Float], config: WooshVocosConfig) throws -> MLXArray {
        guard !samples.isEmpty else {
            throw WooshError.invalidAudioShape([0])
        }

        let nFFT = config.nFFT
        let hopLength = config.hopLength
        let bins = nFFT / 2 + 1
        let padded = reflectPadded(samples, padding: nFFT / 2)
        guard padded.count >= nFFT else {
            throw WooshError.invalidAudioShape([samples.count])
        }

        let frames = ((padded.count - nFFT) / hopLength) + 1
        let window = WooshISTFT.hannWindow(count: nFFT)
        let tables = WooshISTFT.trigTables(nFFT: nFFT)
        var values = [Float](repeating: 0, count: frames * bins * 2)

        for frame in 0..<frames {
            let offset = frame * hopLength
            for bin in 0..<bins {
                var real: Float = 0
                var imaginary: Float = 0
                let tableOffset = bin * nFFT
                for sampleIndex in 0..<nFFT {
                    let sample = padded[offset + sampleIndex] * window[sampleIndex]
                    real += sample * tables.cos[tableOffset + sampleIndex]
                    imaginary -= sample * tables.sin[tableOffset + sampleIndex]
                }
                values[(frame * bins * 2) + bin] = real
                values[(frame * bins * 2) + bins + bin] = imaginary
            }
        }

        return MLXArray(values).reshaped(1, frames, bins * 2).asType(.float32)
    }

    private static func reflectPadded(_ samples: [Float], padding: Int) -> [Float] {
        guard padding > 0 else { return samples }
        guard samples.count > 1 else {
            return [Float](repeating: samples.first ?? 0, count: samples.count + padding * 2)
        }

        let last = samples.count - 1
        var padded = [Float]()
        padded.reserveCapacity(samples.count + padding * 2)
        for index in 0..<padding {
            padded.append(samples[reflectedIndex(padding - index, upperBound: last)])
        }
        padded.append(contentsOf: samples)
        for index in 0..<padding {
            padded.append(samples[reflectedIndex(last - 1 - index, upperBound: last)])
        }
        return padded
    }

    private static func reflectedIndex(_ index: Int, upperBound: Int) -> Int {
        guard upperBound > 0 else { return 0 }
        var value = index
        let period = upperBound * 2
        value %= period
        if value < 0 {
            value += period
        }
        if value > upperBound {
            value = period - value
        }
        return value
    }
}

final class WooshVocosEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: WooshVocosBackbone
    @ModuleInfo(key: "proj") var proj: Conv1d

    private let config: WooshVocosConfig

    init(config: WooshVocosConfig = WooshVocosConfig()) {
        self.config = config
        let bins = config.nFFT / 2 + 1
        self._backbone.wrappedValue = WooshVocosBackbone(config: config, inputChannels: bins * 2)
        self._proj.wrappedValue = Conv1d(
            inputChannels: config.dModel,
            outputChannels: config.zDim,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        )
    }

    func encode(samples: [Float]) throws -> MLXArray {
        let features = try WooshSTFT.complexEmbedding(samples: samples, config: config)
        let hidden = backbone(features)
        return proj(hidden).transposed(0, 2, 1)
    }
}

final class WooshISTFTCircleHead: Module {
    @ModuleInfo(key: "out") var out: Conv1d

    private let config: WooshVocosConfig

    init(config: WooshVocosConfig) {
        self.config = config
        let outDim = ((config.nFFT + 2) * 3) / 2
        self._out.wrappedValue = Conv1d(
            inputChannels: config.dModel,
            outputChannels: outDim,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
    }

    func predictSpectrogram(_ x: MLXArray) -> MLXArray {
        out(x)
    }

    func decode(_ x: MLXArray) throws -> [Float] {
        let predicted = predictSpectrogram(x).transposed(0, 2, 1)
        MLX.eval(predicted)
        guard predicted.ndim == 3, predicted.dim(0) == 1 else {
            throw WooshError.invalidAudioShape(predicted.shape)
        }
        let channels = predicted.dim(1)
        let frames = predicted.dim(2)
        let bins = config.nFFT / 2 + 1
        guard channels == bins * 3 else {
            throw WooshError.invalidAudioShape(predicted.shape)
        }
        let values = predicted.asArray(Float.self)
        var magnitudes = [Float](repeating: 0, count: bins * frames)
        var realUnit = [Float](repeating: 0, count: bins * frames)
        var imagUnit = [Float](repeating: 0, count: bins * frames)
        for frame in 0..<frames {
            for bin in 0..<bins {
                let magRaw = values[(bin * frames) + frame]
                let realRaw = values[((bins + bin) * frames) + frame]
                let imagRaw = values[((2 * bins + bin) * frames) + frame]
                let mag = log1pf(expf(-abs(magRaw))) + max(magRaw, 0)
                let phaseNorm = sqrt(max(1e-8, min(1e3, realRaw * realRaw + imagRaw * imagRaw)))
                magnitudes[frame * bins + bin] = mag
                realUnit[frame * bins + bin] = realRaw / phaseNorm
                imagUnit[frame * bins + bin] = imagRaw / phaseNorm
            }
        }
        return WooshISTFT.reconstruct(
            magnitudes: magnitudes,
            realUnit: realUnit,
            imagUnit: imagUnit,
            frames: frames,
            nFFT: config.nFFT,
            hopLength: config.hopLength
        )
    }
}

final class WooshVocosDecoder: Module {
    @ModuleInfo(key: "backbone") var backbone: WooshVocosBackbone
    @ModuleInfo(key: "head") var head: WooshISTFTCircleHead

    init(config: WooshVocosConfig = WooshVocosConfig()) {
        self._backbone.wrappedValue = WooshVocosBackbone(config: config)
        self._head.wrappedValue = WooshISTFTCircleHead(config: config)
    }

    func decode(_ latents: MLXArray) throws -> [Float] {
        let transposed = latents.transposed(0, 2, 1)
        let hidden = backbone(transposed)
        return try head.decode(hidden)
    }
}

public final class WooshAudioAutoEncoder: Module {
    @ParameterInfo(key: "z_mean") var zMean: MLXArray
    @ParameterInfo(key: "z_std") var zStd: MLXArray
    @ModuleInfo(key: "autoencoder") var autoencoder: WooshVocosAutoEncoder

    private let config: WooshVocosConfig

    public init(config: WooshVocosConfig = WooshVocosConfig()) {
        self.config = config
        self._zMean.wrappedValue = MLXArray.zeros([config.zDim])
        self._zStd.wrappedValue = MLXArray.ones([config.zDim])
        self._autoencoder.wrappedValue = WooshVocosAutoEncoder(config: config)
    }

    public static func load(resources: WooshModelResources, config: WooshVocosConfig = WooshVocosConfig()) throws -> WooshAudioAutoEncoder {
        let model = WooshAudioAutoEncoder(config: config)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.autoencoderWeightsURL,
            to: model,
            dtype: .float32,
            verify: .none,
            mapper: { key, value in
                guard key == "z_mean"
                    || key == "z_std"
                    || key.hasPrefix("autoencoder.encoder.")
                    || key.hasPrefix("autoencoder.decoder.") else {
                    return []
                }
                guard key != "autoencoder.encoder.spec_embed.window" else {
                    return []
                }
                let mappedValue = value.ndim == 3 ? WooshTensorOps.conv1dWeightOIToOKI(value) : value
                return [(key, mappedValue)]
            }
        )
        return model
    }

    public func encode(samples: [Float]) throws -> MLXArray {
        let latents = try autoencoder.encode(samples: samples)
        return (latents - zMean.reshaped(1, config.zDim, 1)) / zStd.reshaped(1, config.zDim, 1)
    }

    public func decode(_ normalizedLatents: MLXArray) throws -> [Float] {
        let latents = normalizedLatents * zStd.reshaped(1, config.zDim, 1) + zMean.reshaped(1, config.zDim, 1)
        return try autoencoder.decode(latents)
    }
}

public final class WooshVocosAutoEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: WooshVocosEncoder
    @ModuleInfo(key: "decoder") var decoder: WooshVocosDecoder

    init(config: WooshVocosConfig) {
        self._encoder.wrappedValue = WooshVocosEncoder(config: config)
        self._decoder.wrappedValue = WooshVocosDecoder(config: config)
    }

    func encode(samples: [Float]) throws -> MLXArray {
        try encoder.encode(samples: samples)
    }

    func decode(_ latents: MLXArray) throws -> [Float] {
        try decoder.decode(latents)
    }
}

enum WooshISTFT {
    static func reconstruct(
        magnitudes: [Float],
        realUnit: [Float],
        imagUnit: [Float],
        frames: Int,
        nFFT: Int,
        hopLength: Int
    ) -> [Float] {
        let bins = nFFT / 2 + 1
        let window = hannWindow(count: nFFT)
        let tables = trigTables(nFFT: nFFT)
        let fullCount = max(0, (frames - 1) * hopLength + nFFT)
        var output = [Float](repeating: 0, count: fullCount)
        var envelope = [Float](repeating: 0, count: fullCount)

        for frame in 0..<frames {
            let offset = frame * hopLength
            for sample in 0..<nFFT {
                var value = magnitudes[frame * bins] * realUnit[frame * bins]
                let nyquist = frame * bins + (bins - 1)
                value += magnitudes[nyquist] * realUnit[nyquist] * (sample.isMultiple(of: 2) ? 1 : -1)
                if bins > 2 {
                    for bin in 1..<(bins - 1) {
                        let index = frame * bins + bin
                        let tableIndex = bin * nFFT + sample
                        value += 2 * magnitudes[index] * (
                            realUnit[index] * tables.cos[tableIndex]
                                - imagUnit[index] * tables.sin[tableIndex]
                        )
                    }
                }
                value /= Float(nFFT)
                let win = window[sample]
                output[offset + sample] += value * win
                envelope[offset + sample] += win * win
            }
        }

        let trim = nFFT / 2
        guard output.count > trim * 2 else {
            return []
        }
        var trimmed = Array(output[trim..<(output.count - trim)])
        let trimmedEnvelope = Array(envelope[trim..<(envelope.count - trim)])
        for index in trimmed.indices where trimmedEnvelope[index] > 1e-11 {
            trimmed[index] /= trimmedEnvelope[index]
        }
        return trimmed
    }

    static func hannWindow(count: Int) -> [Float] {
        (0..<count).map { index in
            0.5 - 0.5 * cosf((2 * Float.pi * Float(index)) / Float(count))
        }
    }

    static func trigTables(nFFT: Int) -> (cos: [Float], sin: [Float]) {
        let bins = nFFT / 2 + 1
        var cosTable = [Float](repeating: 0, count: bins * nFFT)
        var sinTable = [Float](repeating: 0, count: bins * nFFT)
        for bin in 0..<bins {
            for sample in 0..<nFFT {
                let angle = 2 * Float.pi * Float(bin * sample) / Float(nFFT)
                cosTable[bin * nFFT + sample] = cosf(angle)
                sinTable[bin * nFFT + sample] = sinf(angle)
            }
        }
        return (cosTable, sinTable)
    }
}
