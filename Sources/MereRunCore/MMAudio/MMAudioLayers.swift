import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

enum MMAudioTensorOps {
    static func layerNorm(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let promoted = x.asType(.float32)
        let mean = promoted.mean(axis: -1, keepDims: true)
        let centered = promoted - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return (centered / MLX.sqrt(variance + eps)).asType(x.dtype)
    }

    static func rmsNorm(_ x: MLXArray) -> MLXArray {
        let eps: Float = x.dtype == .float16 ? 0.000_976_562_5 : 0.000_000_119_209_29
        let promoted = x.asType(.float32)
        let variance = (promoted * promoted).mean(axis: -1, keepDims: true)
        return (promoted / MLX.sqrt(variance + eps)).asType(x.dtype)
    }

    static func nearestInterpolateSequence(_ x: MLXArray, length: Int) -> MLXArray {
        let sourceLength = x.dim(1)
        let indices = (0..<length).map { index in
            Int32(min(
                sourceLength - 1,
                Int(floor((Float(index) + 0.5) * Float(sourceLength) / Float(length)))
            ))
        }
        return MLX.take(x, MLXArray(indices), axis: 1)
    }
}

final class MMAudioDenseOrConv1D: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    private let kernelSize: Int
    private let padding: Int

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, padding: Int, bias: Bool) {
        self.kernelSize = kernelSize
        self.padding = padding
        if kernelSize == 1 {
            self._weight.wrappedValue = MLXRandom.normal([outputChannels, inputChannels]) * 0.02
        } else {
            self._weight.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels]) * 0.02
        }
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if kernelSize == 1 {
            var output = matmul(x, weight.transposed())
            if let bias {
                output = output + bias
            }
            return output
        }
        var output = MLX.conv1d(x, weight, stride: 1, padding: padding)
        if let bias {
            output = output + bias
        }
        return output
    }
}

final class MMAudioRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MMAudioTensorOps.rmsNorm(x) * weight
    }
}

final class MMAudioMLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dimensions: Int, requestedHiddenDimensions: Int) {
        let hidden = Self.roundedHiddenDimension(requestedHiddenDimensions)
        self._w1.wrappedValue = Linear(dimensions, hidden, bias: false)
        self._w2.wrappedValue = Linear(hidden, dimensions, bias: false)
        self._w3.wrappedValue = Linear(dimensions, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(MLXNN.silu(w1(x)) * w3(x))
    }

    private static func roundedHiddenDimension(_ requested: Int) -> Int {
        let reduced = 2 * requested / 3
        return 256 * ((reduced + 255) / 256)
    }
}

final class MMAudioConvMLP: Module {
    @ModuleInfo(key: "w1") var w1: MMAudioDenseOrConv1D
    @ModuleInfo(key: "w2") var w2: MMAudioDenseOrConv1D
    @ModuleInfo(key: "w3") var w3: MMAudioDenseOrConv1D

    init(dimensions: Int, requestedHiddenDimensions: Int, kernelSize: Int, padding: Int) {
        let reduced = 2 * requestedHiddenDimensions / 3
        let hidden = 256 * ((reduced + 255) / 256)
        self._w1.wrappedValue = MMAudioDenseOrConv1D(
            inputChannels: dimensions,
            outputChannels: hidden,
            kernelSize: kernelSize,
            padding: padding,
            bias: false
        )
        self._w2.wrappedValue = MMAudioDenseOrConv1D(
            inputChannels: hidden,
            outputChannels: dimensions,
            kernelSize: kernelSize,
            padding: padding,
            bias: false
        )
        self._w3.wrappedValue = MMAudioDenseOrConv1D(
            inputChannels: dimensions,
            outputChannels: hidden,
            kernelSize: kernelSize,
            padding: padding,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(MLXNN.silu(w1(x)) * w3(x))
    }
}

struct MMAudioRoPE {
    let cos: MLXArray
    let sin: MLXArray

    static func make(length: Int, dimensions: Int, frequencyScaling: Float) -> MMAudioRoPE {
        let positions = MLXArray((0..<length).map(Float.init)).expandedDimensions(axis: 1)
        let frequencies = MLXArray(stride(from: 0, to: dimensions, by: 2).map { index in
            frequencyScaling / pow(10_000, Float(index) / Float(dimensions))
        }).expandedDimensions(axis: 0)
        let angles = positions * frequencies
        return MMAudioRoPE(cos: MLX.cos(angles), sin: MLX.sin(angles))
    }

    func apply(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let heads = x.dim(1)
        let sequence = x.dim(2)
        let dimensions = x.dim(3)
        let paired = x.reshaped(batch, heads, sequence, dimensions / 2, 2)
        let first = paired[0..., 0..., 0..., 0..., 0]
        let second = paired[0..., 0..., 0..., 0..., 1]
        let cosValues = cos[0..<sequence, 0...].expandedDimensions(axes: [0, 1])
        let sinValues = sin[0..<sequence, 0...].expandedDimensions(axes: [0, 1])
        let rotatedFirst = first * cosValues - second * sinValues
        let rotatedSecond = first * sinValues + second * cosValues
        return MLX.stacked([rotatedFirst, rotatedSecond], axis: -1)
            .reshaped(x.shape)
            .asType(x.dtype)
    }
}

final class MMAudioSelfAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MMAudioRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MMAudioRMSNorm

    private let dimensions: Int
    private let heads: Int
    private let headDimensions: Int

    init(dimensions: Int, heads: Int) {
        self.dimensions = dimensions
        self.heads = heads
        self.headDimensions = dimensions / heads
        self._qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: true)
        self._qNorm.wrappedValue = MMAudioRMSNorm(dimensions: dimensions / heads)
        self._kNorm.wrappedValue = MMAudioRMSNorm(dimensions: dimensions / heads)
    }

    func project(_ x: MLXArray, rope: MMAudioRoPE?) -> (MLXArray, MLXArray, MLXArray) {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let packed = qkv(x).reshaped(batch, sequence, heads, headDimensions, 3)
        var query = packed[0..., 0..., 0..., 0..., 0].transposed(0, 2, 1, 3)
        var key = packed[0..., 0..., 0..., 0..., 1].transposed(0, 2, 1, 3)
        let value = packed[0..., 0..., 0..., 0..., 2].transposed(0, 2, 1, 3)
        query = qNorm(query)
        key = kNorm(key)
        if let rope {
            query = rope.apply(query)
            key = rope.apply(key)
        }
        return (query, key, value)
    }

    func attend(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
        let output = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / sqrt(Float(headDimensions)),
            mask: .none
        )
        return output.transposed(0, 2, 1, 3).reshaped(output.dim(0), output.dim(2), dimensions)
    }
}
