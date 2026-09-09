import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

class LTXAudioVocoderBase: Module {
    let outputSamplingRate: Int

    init(outputSamplingRate: Int) {
        self.outputSamplingRate = outputSamplingRate
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        preconditionFailure("LTXAudioVocoderBase must be subclassed.")
    }
}

class LTXVocoderResidualBlock: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        preconditionFailure("LTXVocoderResidualBlock must be subclassed.")
    }
}

final class LTXVocoderSnake: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray
    @ModuleInfo(key: "beta") var beta: MLXArray?

    init(channels: Int, hasBeta: Bool) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = hasBeta ? MLXArray.zeros([channels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaValue = MLX.exp(alpha.asType(x.dtype)).reshaped(1, 1, -1)
        let betaValue = (beta.map { MLX.exp($0.asType(x.dtype)).reshaped(1, 1, -1) } ?? alphaValue)
        let sine = MLX.sin(x * alphaValue)
        return x + (sine * sine) / (betaValue + MLXArray(1e-9).asType(x.dtype))
    }
}

func ltxBesselI0(_ value: Double) -> Double {
    var sum = 1.0
    var term = 1.0
    let scaled = (value * value) / 4.0
    for index in 1...24 {
        term *= scaled / Double(index * index)
        sum += term
        if term < 1e-12 {
            break
        }
    }
    return sum
}

func ltxSinc(_ value: Float) -> Float {
    if abs(value) < 1e-8 {
        return 1.0
    }
    return sin(Float.pi * value) / (Float.pi * value)
}

func ltxKaiserWindow(kernelSize: Int, beta: Float) -> [Float] {
    guard kernelSize > 1 else { return [1.0] }
    let denominator = ltxBesselI0(Double(beta))
    return (0..<kernelSize).map { index in
        let ratio = (2.0 * Double(index)) / Double(kernelSize - 1) - 1.0
        let value = Double(beta) * sqrt(max(0.0, 1.0 - ratio * ratio))
        return Float(ltxBesselI0(value) / denominator)
    }
}

func ltxKaiserSincFilter1d(cutoff: Float, halfWidth: Float, kernelSize: Int) -> [Float] {
    let even = kernelSize.isMultiple(of: 2)
    let halfSize = kernelSize / 2
    let deltaF = 4.0 * halfWidth
    let amplitude = 2.285 * Float(halfSize - 1) * Float.pi * deltaF + 7.95
    let beta: Float
    if amplitude > 50.0 {
        beta = 0.1102 * (amplitude - 8.7)
    } else if amplitude >= 21.0 {
        beta = 0.5842 * pow(amplitude - 21.0, 0.4) + 0.07886 * (amplitude - 21.0)
    } else {
        beta = 0.0
    }

    let window = ltxKaiserWindow(kernelSize: kernelSize, beta: beta)
    guard cutoff != 0 else {
        return [Float](repeating: 0, count: kernelSize)
    }

    var filter = [Float](repeating: 0, count: kernelSize)
    for index in 0..<kernelSize {
        let time = even ? Float(index - halfSize) + 0.5 : Float(index - halfSize)
        filter[index] = 2.0 * cutoff * window[index] * ltxSinc(2.0 * cutoff * time)
    }

    let total = filter.reduce(0, +)
    if total != 0 {
        filter = filter.map { $0 / total }
    }
    return filter
}

func ltxHannSincFilter1d(ratio: Int) -> [Float] {
    let rolloff: Float = 0.99
    let lowpassFilterWidth: Float = 6.0
    let width = Int(ceil(lowpassFilterWidth / rolloff))
    let kernelSize = 2 * width * ratio + 1
    return (0..<kernelSize).map { index in
        let timeAxis = (Float(index) / Float(ratio) - Float(width)) * rolloff
        let clamped = min(lowpassFilterWidth, max(-lowpassFilterWidth, timeAxis))
        let window = pow(cos(clamped * Float.pi / lowpassFilterWidth / 2.0), 2.0)
        return ltxSinc(timeAxis) * window * rolloff / Float(ratio)
    }
}

final class LTXLowPassFilter1d: Module {
    @ModuleInfo(key: "filter") var filter: MLXArray
    let kernelSize: Int
    let padLeft: Int
    let padRight: Int
    let stride: Int

    init(ratio: Int) {
        self.kernelSize = Int(6 * ratio / 2) * 2
        self.padLeft = kernelSize / 2 - (kernelSize.isMultiple(of: 2) ? 1 : 0)
        self.padRight = kernelSize / 2
        self.stride = ratio
        let values = ltxKaiserSincFilter1d(
            cutoff: 0.5 / Float(ratio),
            halfWidth: 0.6 / Float(ratio),
            kernelSize: kernelSize
        )
        self._filter.wrappedValue = MLXArray(values).reshaped(1, kernelSize, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(2)
        let paddedInput = padded(
            x,
            widths: [[0, 0], [padLeft, padRight], [0, 0]],
            mode: .edge
        )
        let weights = broadcast(filter.asType(x.dtype), to: [channels, kernelSize, 1])
        return MLX.conv1d(paddedInput, weights, stride: stride, padding: 0, groups: channels)
    }
}

final class LTXSincUpsample1d: Module {
    @ModuleInfo(key: "filter") var filter: MLXArray
    let ratio: Int
    let kernelSize: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int

    init(ratio: Int, windowType: String = "kaiser") {
        self.ratio = ratio
        let values: [Float]
        if windowType == "hann" {
            values = ltxHannSincFilter1d(ratio: ratio)
            self.kernelSize = values.count
            let width = Int(ceil(6.0 / 0.99))
            self.pad = width
            self.padLeft = 2 * width * ratio
            self.padRight = kernelSize - ratio
        } else {
            self.kernelSize = Int(6 * ratio / 2) * 2
            self.pad = kernelSize / ratio - 1
            self.padLeft = pad * ratio + (kernelSize - ratio) / 2
            self.padRight = pad * ratio + (kernelSize - ratio + 1) / 2
            values = ltxKaiserSincFilter1d(
                cutoff: 0.5 / Float(ratio),
                halfWidth: 0.6 / Float(ratio),
                kernelSize: kernelSize
            )
        }
        self._filter.wrappedValue = MLXArray(values).reshaped(1, kernelSize, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(2)
        let paddedInput = padded(
            x,
            widths: [[0, 0], [pad, pad], [0, 0]],
            mode: .edge
        )
        let paddedLength = paddedInput.dim(1)
        let upsampledLength = (paddedLength - 1) * ratio + 1
        let zeroTail = MLX.zeros([paddedInput.dim(0), paddedLength, max(0, ratio - 1), channels], dtype: x.dtype)
        let expanded = MLX.concatenated([paddedInput.expandedDimensions(axis: 2), zeroTail], axis: 2)
            .reshaped(paddedInput.dim(0), paddedLength * ratio, channels)
        let upsampled = expanded[0..., 0..<upsampledLength, 0...]

        let convInput = padded(
            upsampled,
            widths: [[0, 0], [kernelSize - 1, kernelSize - 1], [0, 0]]
        )
        let weights = broadcast(filter.asType(x.dtype), to: [channels, kernelSize, 1])
        let filtered = MLX.conv1d(
            convInput,
            weights,
            stride: 1,
            padding: 0,
            groups: channels
        ) * MLXArray(Float(ratio)).asType(x.dtype)
        let end = max(padLeft, filtered.dim(1) - padRight)
        return filtered[0..., padLeft..<end, 0...]
    }
}

final class LTXVocoderActivation1d: Module {
    @ModuleInfo(key: "act") var act: LTXVocoderSnake?
    @ModuleInfo(key: "upsample") var upsample: LTXSincUpsample1d
    @ModuleInfo(key: "downsample") var downsample: LTXLowPassFilter1d
    let kind: LTXVocoderActivationKind

    init(channels: Int, kind: LTXVocoderActivationKind) {
        self.kind = kind
        switch kind {
        case .leaky:
            self._act.wrappedValue = nil
        case .snake:
            self._act.wrappedValue = LTXVocoderSnake(channels: channels, hasBeta: false)
        case .snakeBeta:
            self._act.wrappedValue = LTXVocoderSnake(channels: channels, hasBeta: true)
        }
        self._upsample.wrappedValue = LTXSincUpsample1d(ratio: 2)
        self._downsample.wrappedValue = LTXLowPassFilter1d(ratio: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch kind {
        case .leaky:
            return ltxLeakyRelu(x, slope: 0.1)
        case .snake, .snakeBeta:
            guard let act else { return x }
            var h = upsample(x)
            h = act(h)
            return downsample(h)
        }
    }
}

final class LTXVocoderResBlock1: LTXVocoderResidualBlock {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        self._convs1.wrappedValue = dilations.map { dilation in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: ((kernelSize - 1) * dilation) / 2,
                dilation: dilation,
                groups: 1,
                bias: true
            )
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for idx in 0..<convs1.count {
            var h = ltxLeakyRelu(out, slope: 0.1)
            h = convs1[idx](h)
            h = ltxLeakyRelu(h, slope: 0.1)
            h = convs2[idx](h)
            out = out + h
        }
        return out
    }
}

final class LTXVocoderAMPBlock1: LTXVocoderResidualBlock {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "acts1") var acts1: [LTXVocoderActivation1d]
    @ModuleInfo(key: "acts2") var acts2: [LTXVocoderActivation1d]

    init(channels: Int, kernelSize: Int, dilations: [Int], activation: LTXVocoderActivationKind) {
        self._convs1.wrappedValue = dilations.map { dilation in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: ((kernelSize - 1) * dilation) / 2,
                dilation: dilation,
                groups: 1,
                bias: true
            )
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }
        self._acts1.wrappedValue = dilations.map { _ in
            LTXVocoderActivation1d(channels: channels, kind: activation)
        }
        self._acts2.wrappedValue = dilations.map { _ in
            LTXVocoderActivation1d(channels: channels, kind: activation)
        }
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for idx in 0..<convs1.count {
            var h = acts1[idx](out)
            h = convs1[idx](h)
            h = acts2[idx](h)
            h = convs2[idx](h)
            out = out + h
        }
        return out
    }
}
