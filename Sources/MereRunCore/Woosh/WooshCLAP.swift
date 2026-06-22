import Foundation
import MLX
import MLXFast
import MLXNN

#if canImport(Accelerate)
import Accelerate
#endif

public struct WooshCLAPScore: Sendable, Hashable {
    public let score: Float

    public init(score: Float) {
        self.score = score
    }
}

public final class WooshCLAP {
    private let textTower: WooshCLAPTextTower
    private let audioTower: WooshCLAPAudioTower

    public init(resources: WooshCLAPResources) throws {
        let tokenizer = try WooshRobertaTokenizer.load(
            from: resources.tokenizerRootURL,
            maxLength: WooshRobertaConfig().maxSentenceTokens
        )
        self.textTower = WooshCLAPTextTower(tokenizer: tokenizer)
        self.audioTower = WooshCLAPAudioTower()

        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.textWeightsURL,
            to: textTower,
            dtype: .float32,
            verify: .none,
            mapper: Self.mapTextWeights
        )
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.audioWeightsURL,
            to: audioTower,
            dtype: .float32,
            verify: .none,
            mapper: Self.mapAudioWeights
        )
    }

    public func textEmbedding(_ text: String) -> MLXArray {
        textTower.embed(text)
    }

    public func audioEmbedding(samples: [Float]) throws -> MLXArray {
        try audioTower.embed(samples: samples)
    }

    public func score(text: String, audioSamples: [Float]) throws -> WooshCLAPScore {
        let textEmbedding = textEmbedding(text)
        let audioEmbedding = try audioEmbedding(samples: audioSamples)
        let score = MLX.sum(textEmbedding * audioEmbedding, axis: -1)
        MLX.eval(score)
        return WooshCLAPScore(score: score.asArray(Float.self).first ?? 0)
    }

    private static func mapTextWeights(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix("sentence_head.0.") {
            return [(key.replacingOccurrences(of: "sentence_head.0.", with: "sentence_head."), value)]
        }
        return [(key, value)]
    }

    private static func mapAudioWeights(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix("audio_frontend.model.head.")
            || key.hasPrefix("audio_frontend.model.head_dist.") {
            return []
        }

        var mappedKey = key
        var mappedValue = value
        if key.hasPrefix("audio_head.0.") {
            mappedKey = key.replacingOccurrences(of: "audio_head.0.", with: "audio_head.")
        }
        if mappedKey.hasSuffix("patch_embed.proj.weight") {
            mappedValue = HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
        } else if mappedKey.hasSuffix("freq_new_pos_embed")
            || mappedKey.hasSuffix("time_new_pos_embed") {
            mappedValue = value.transposed(0, 2, 3, 1)
        }
        return [(mappedKey, mappedValue)]
    }
}

private final class WooshCLAPTextTower: Module {
    @ModuleInfo(key: "sentence_frontend") var sentenceFrontend: WooshRobertaModel
    @ModuleInfo(key: "sentence_head") var sentenceHead: Linear

    private let tokenizer: WooshRobertaTokenizer

    init(tokenizer: WooshRobertaTokenizer) {
        self.tokenizer = tokenizer
        self._sentenceFrontend.wrappedValue = WooshRobertaModel(config: WooshRobertaConfig(lhsIndex: -1))
        self._sentenceHead.wrappedValue = Linear(1024, 1024, bias: true)
    }

    func embed(_ text: String) -> MLXArray {
        let batch = tokenizer.encode([text])
        let hidden = sentenceFrontend(inputIds: batch.inputIds, attentionMask: batch.attentionMask)
        let pooled = hidden[0..., 0, 0...]
        return WooshTensorOps.l2Normalize(sentenceHead(pooled))
    }
}

private final class WooshCLAPAudioTower: Module {
    @ModuleInfo(key: "audio_frontend") var audioFrontend: WooshCLAPAudioFrontend
    @ModuleInfo(key: "audio_head") var audioHead: Linear

    private let config = WooshCLAPAudioConfig()

    override init() {
        self._audioFrontend.wrappedValue = WooshCLAPAudioFrontend()
        self._audioHead.wrappedValue = Linear(768, 1024, bias: true)
    }

    func embed(samples: [Float]) throws -> MLXArray {
        guard !samples.isEmpty else {
            throw WooshError.invalidAudioShape([0])
        }

        let maxSamples = min(samples.count, config.evalMaxSeconds * config.sampleRate)
        let usedSamples = Array(samples.prefix(maxSamples))
        let segmentSamples = config.segmentSeconds * config.sampleRate
        let segmentCount = max(1, Int(ceil(Float(usedSamples.count) / Float(segmentSamples))))

        var accumulated: MLXArray?
        for index in 0..<segmentCount {
            let start = index * segmentSamples
            let end = min(start + segmentSamples, usedSamples.count)
            var segment = start < end ? Array(usedSamples[start..<end]) : []
            if segment.count < segmentSamples {
                segment.append(contentsOf: Array(repeating: 0, count: segmentSamples - segment.count))
            }
            let features = try audioFrontend.embed(samples: segment, config: config)
            accumulated = accumulated.map { $0 + features } ?? features
        }

        let averaged = (accumulated ?? MLXArray.zeros([1, 768], dtype: .float32)) / MLXArray(Float(segmentCount))
        return WooshTensorOps.l2Normalize(audioHead(averaged))
    }
}

private struct WooshCLAPAudioConfig {
    let sampleRate = 32_000
    let segmentSeconds = 5
    let evalMaxSeconds = 60
    let nMels = 128
    let nFFT = 1024
    let winLength = 800
    let hopLength = 320
    let fMin: Float = 0
    let fMax: Float = 15_000
}

private final class WooshCLAPAudioFrontend: Module {
    @ModuleInfo(key: "model") var model: WooshPaSSTModel

    override init() {
        self._model.wrappedValue = WooshPaSSTModel()
    }

    func embed(samples: [Float], config: WooshCLAPAudioConfig) throws -> MLXArray {
        let mel = try WooshPaSSTMelSpectrogram(config: config).extract(samples: samples)
        return model(mel)
    }
}

private final class WooshPaSSTModel: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: WooshPaSSTPatchEmbed
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "dist_token") var distToken: MLXArray
    @ParameterInfo(key: "new_pos_embed") var newPosEmbed: MLXArray
    @ParameterInfo(key: "freq_new_pos_embed") var freqNewPosEmbed: MLXArray
    @ParameterInfo(key: "time_new_pos_embed") var timeNewPosEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [WooshPaSSTBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    private let hiddenSize = 768

    override init() {
        let hiddenSize = 768
        self._patchEmbed.wrappedValue = WooshPaSSTPatchEmbed()
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, hiddenSize], dtype: .float32)
        self._distToken.wrappedValue = MLXArray.zeros([1, 1, hiddenSize], dtype: .float32)
        self._newPosEmbed.wrappedValue = MLXArray.zeros([1, 2, hiddenSize], dtype: .float32)
        self._freqNewPosEmbed.wrappedValue = MLXArray.zeros([1, 8, 1, hiddenSize], dtype: .float32)
        self._timeNewPosEmbed.wrappedValue = MLXArray.zeros([1, 1, 62, hiddenSize], dtype: .float32)
        self._blocks.wrappedValue = (0..<12).map { _ in WooshPaSSTBlock(hiddenSize: hiddenSize) }
        self._norm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var hidden = patchEmbed(mel)
        let batch = hidden.dim(0)
        let freqPatches = hidden.dim(1)
        let timePatches = min(hidden.dim(2), timeNewPosEmbed.dim(2))
        if hidden.dim(2) != timePatches {
            hidden = hidden[0..., 0..., 0..<timePatches, 0...]
        }

        hidden = hidden
            + freqNewPosEmbed[0..., 0..<freqPatches, 0..., 0...]
            + timeNewPosEmbed[0..., 0..., 0..<timePatches, 0...]
        hidden = hidden.reshaped(batch, freqPatches * timePatches, hiddenSize)

        let cls = MLX.broadcast(
            clsToken + newPosEmbed[0..., 0..<1, 0...],
            to: [batch, 1, hiddenSize]
        )
        let dist = MLX.broadcast(
            distToken + newPosEmbed[0..., 1..<2, 0...],
            to: [batch, 1, hiddenSize]
        )
        hidden = MLX.concatenated([cls, dist, hidden], axis: 1)

        for block in blocks {
            hidden = block(hidden)
        }
        hidden = norm(hidden)
        return (hidden[0..., 0, 0...] + hidden[0..., 1, 0...]) / MLXArray(2.0).asType(hidden.dtype)
    }
}

private final class WooshPaSSTPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    override init() {
        self._proj.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: 768,
            kernelSize: IntOrPair(16),
            stride: IntOrPair(16),
            bias: true
        )
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        proj(mel.expandedDimensions(axis: -1))
    }
}

private final class WooshPaSSTAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    private let numHeads = 12
    private let headDim = 64
    private let scale: Float

    init(hiddenSize: Int) {
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: true)
        self._proj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let sequence = hidden.dim(1)
        let parts = MLX.split(qkv(hidden), parts: 3, axis: -1)
        let q = parts[0].reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        return proj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, numHeads * headDim))
    }
}

private final class WooshPaSSTMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int) {
        self._fc1.wrappedValue = Linear(hiddenSize, hiddenSize * 4, bias: true)
        self._fc2.wrappedValue = Linear(hiddenSize * 4, hiddenSize, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        fc2(gelu(fc1(hidden)))
    }
}

private final class WooshPaSSTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: WooshPaSSTAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: WooshPaSSTMlp

    init(hiddenSize: Int) {
        self._norm1.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._attn.wrappedValue = WooshPaSSTAttention(hiddenSize: hiddenSize)
        self._norm2.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._mlp.wrappedValue = WooshPaSSTMlp(hiddenSize: hiddenSize)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var current = hidden + attn(norm1(hidden))
        current = current + mlp(norm2(current))
        return current
    }
}

private struct WooshPaSSTMelSpectrogram {
    let config: WooshCLAPAudioConfig

    func extract(samples: [Float]) throws -> MLXArray {
        let emphasized = preemphasize(samples)
        let padded = reflectPad(emphasized, padding: config.nFFT / 2)
        guard padded.count >= config.nFFT else {
            throw WooshError.invalidAudioShape([samples.count])
        }

        let frameCount = ((padded.count - config.nFFT) / config.hopLength) + 1
        let fftPlan = try WooshRealFFTPlan(size: config.nFFT)
        let window = centeredWindow()
        let melBasis = melFilterBank()
        let fftBins = (config.nFFT / 2) + 1
        var mel = [Float](repeating: 0, count: config.nMels * frameCount)
        var frameBuffer = [Float](repeating: 0, count: config.nFFT)

        for frame in 0..<frameCount {
            let start = frame * config.hopLength
            for index in 0..<config.nFFT {
                frameBuffer[index] = padded[start + index] * window[index]
            }
            let power = fftPlan.powerSpectrum(frameBuffer)
            for melIndex in 0..<config.nMels {
                let basisOffset = melIndex * fftBins
                var energy: Float = 0
                for bin in 0..<fftBins {
                    energy += melBasis[basisOffset + bin] * power[bin]
                }
                mel[melIndex * frameCount + frame] = (logf(energy + 0.00001) + 4.5) / 5.0
            }
        }

        return MLXArray(mel).reshaped(1, config.nMels, frameCount).asType(.float32)
    }

    private func preemphasize(_ samples: [Float]) -> [Float] {
        guard samples.count > 1 else { return samples }
        var output = [Float](repeating: 0, count: samples.count - 1)
        for index in 0..<output.count {
            output[index] = -0.97 * samples[index] + samples[index + 1]
        }
        return output
    }

    private func centeredWindow() -> [Float] {
        let offset = max(0, (config.nFFT - config.winLength) / 2)
        var window = [Float](repeating: 0, count: config.nFFT)
        guard config.winLength > 1 else { return window }
        let denominator = Float(config.winLength - 1)
        for index in 0..<config.winLength {
            window[offset + index] = 0.5 - 0.5 * cosf(2 * Float.pi * Float(index) / denominator)
        }
        return window
    }

    private func melFilterBank() -> [Float] {
        let fftBins = (config.nFFT / 2) + 1
        let fftBinWidth = Float(config.sampleRate) / Float(config.nFFT)
        let melLow = melScale(config.fMin)
        let melHigh = melScale(config.fMax)
        let melDelta = (melHigh - melLow) / Float(config.nMels + 1)
        var filters = [Float](repeating: 0, count: config.nMels * fftBins)

        for melIndex in 0..<config.nMels {
            let left = melLow + Float(melIndex) * melDelta
            let center = melLow + Float(melIndex + 1) * melDelta
            let right = melLow + Float(melIndex + 2) * melDelta
            for bin in 0..<(fftBins - 1) {
                let mel = melScale(fftBinWidth * Float(bin))
                let up = (mel - left) / (center - left)
                let down = (right - mel) / (right - center)
                filters[melIndex * fftBins + bin] = max(0, min(up, down))
            }
        }
        return filters
    }

    private func melScale(_ frequency: Float) -> Float {
        1127.0 * logf(1.0 + frequency / 700.0)
    }

    private func reflectPad(_ samples: [Float], padding: Int) -> [Float] {
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

    private func reflectedIndex(_ index: Int, upperBound: Int) -> Int {
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

private struct WooshRealFFTPlan {
    let size: Int

    #if canImport(Accelerate)
    private let acceleratePlan: vDSP.FFT<DSPSplitComplex>?
    #endif

    init(size: Int) throws {
        guard size > 1 else {
            throw WooshError.invalidAudioShape([size])
        }
        self.size = size
        #if canImport(Accelerate)
        if size.nonzeroBitCount == 1 {
            let log2n = vDSP_Length(log2(Double(size)))
            self.acceleratePlan = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)
        } else {
            self.acceleratePlan = nil
        }
        #endif
    }

    func powerSpectrum(_ samples: [Float]) -> [Float] {
        var frame = samples
        if frame.count < size {
            frame += [Float](repeating: 0, count: size - frame.count)
        } else if frame.count > size {
            frame = Array(frame.prefix(size))
        }

        #if canImport(Accelerate)
        if let acceleratePlan {
            return acceleratePowerSpectrum(frame, plan: acceleratePlan)
        }
        #endif

        return discretePowerSpectrum(frame)
    }

    #if canImport(Accelerate)
    private func acceleratePowerSpectrum(
        _ frame: [Float],
        plan: vDSP.FFT<DSPSplitComplex>
    ) -> [Float] {
        var real = [Float](repeating: 0, count: size / 2)
        var imag = [Float](repeating: 0, count: size / 2)
        frame.withUnsafeBufferPointer { inputPtr in
            real.withUnsafeMutableBufferPointer { realPtr in
                imag.withUnsafeMutableBufferPointer { imagPtr in
                    var split = DSPSplitComplex(
                        realp: realPtr.baseAddress!,
                        imagp: imagPtr.baseAddress!
                    )
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                        2,
                        &split,
                        1,
                        vDSP_Length(size / 2)
                    )
                    plan.forward(input: split, output: &split)
                }
            }
        }

        var magnitudes = [Float](repeating: 0, count: (size / 2) + 1)
        magnitudes[0] = real[0] * real[0]
        for index in 1..<(size / 2) {
            magnitudes[index] = (real[index] * real[index]) + (imag[index] * imag[index])
        }
        magnitudes[size / 2] = imag[0] * imag[0]
        return magnitudes
    }
    #endif

    private func discretePowerSpectrum(_ frame: [Float]) -> [Float] {
        var magnitudes = [Float](repeating: 0, count: (size / 2) + 1)
        let denominator = Double(size)
        for frequency in 0...(size / 2) {
            var real = 0.0
            var imaginary = 0.0
            for sampleIndex in 0..<size {
                let angle = -2.0 * Double.pi * Double(frequency * sampleIndex) / denominator
                let value = Double(frame[sampleIndex])
                real += value * cos(angle)
                imaginary += value * sin(angle)
            }
            magnitudes[frequency] = Float((real * real) + (imaginary * imaginary))
        }
        return magnitudes
    }
}
