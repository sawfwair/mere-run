import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Snake1d Activation

/// Snake activation: x + (1/alpha) * sin^2(alpha * x)
final class Snake1d: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, channels, 1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let sinTerm = MLX.sin(alpha * x)
        return x + (sinTerm * sinTerm) / alpha
    }
}

// MARK: - Weight-Normalized Layers

/// Weight-normalized 1D convolution
final class WNConv1d: Module, @unchecked Sendable {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let stride: Int
    let padding: Int
    let dilation: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        // Weight normalization: weight = g * (v / ||v||)
        // g: [outputChannels, 1, 1], v: [outputChannels, inputChannels, kernelSize]
        self._weightG.wrappedValue = MLXArray.ones([outputChannels, 1, 1])
        self._weightV.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels]) * 0.02

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        } else {
            self._bias.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute normalized weight: g * (v / ||v||)
        // weightV shape: [out, kernel, in] for MLX conv1d
        let vNorm = MLX.sqrt(MLX.sum(weightV * weightV, axes: [1, 2], keepDims: true) + 1e-12)
        let weight = weightG * (weightV / vNorm)

        // MLX conv1d expects: input [N, L, C_in], weight [C_out, K, C_in]
        var result = MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)

        if let bias {
            result = result + bias
        }

        return result
    }
}

/// Weight-normalized transposed 1D convolution
final class WNConvTranspose1d: Module, @unchecked Sendable {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let stride: Int
    let padding: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding

        self._weightG.wrappedValue = MLXArray.ones([outputChannels, 1, 1])
        self._weightV.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels]) * 0.02

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        } else {
            self._bias.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let vNorm = MLX.sqrt(MLX.sum(weightV * weightV, axes: [1, 2], keepDims: true) + 1e-12)
        let weight = weightG * (weightV / vNorm)

        var result = MLX.convTransposed1d(x, weight, stride: stride, padding: padding)

        if let bias {
            result = result + bias
        }

        return result
    }
}

// MARK: - Residual Unit

/// Residual unit with dilated convolutions and Snake activation
final class ResidualUnit: Module, @unchecked Sendable {
    @ModuleInfo(key: "block") var block: Sequential

    init(channels: Int, dilation: Int) {
        // Dilated residual block: snake -> conv (dilated) -> snake -> conv (1x1)
        let padding = (7 - 1) * dilation / 2
        let layers: [Module] = [
            Snake1d(channels: channels),
            WNConv1d(inputChannels: channels, outputChannels: channels, kernelSize: 7, padding: padding, dilation: dilation),
            Snake1d(channels: channels),
            WNConv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        ]
        self._block.wrappedValue = Sequential(layers)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + block(x)
    }
}

// MARK: - Decoder Block

/// Decoder block with upsampling via transposed conv + residual units
final class DecoderBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "block") var block: Sequential

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        let kernelSize = stride * 2
        let padding = stride / 2

        var layers: [Module] = [
            Snake1d(channels: inputChannels),
            WNConvTranspose1d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: kernelSize,
                stride: stride,
                padding: padding
            )
        ]

        // Add residual units with increasing dilation
        for i in 0..<3 {
            layers.append(ResidualUnit(channels: outputChannels, dilation: Int(pow(3.0, Double(i)))))
        }

        self._block.wrappedValue = Sequential(layers)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        block(x)
    }
}

// MARK: - Local Multi-Head Attention

/// Local multi-head attention with sliding window
final class LocalMHA: Module, @unchecked Sendable {
    @ModuleInfo(key: "to_qkv") var toQKV: WNConv1d
    @ModuleInfo(key: "to_out") var toOut: WNConv1d

    let heads: Int
    let windowSize: Int
    let headDim: Int

    init(dim: Int, heads: Int = 8, windowSize: Int = 32) {
        self.heads = heads
        self.windowSize = windowSize
        self.headDim = dim / heads

        self._toQKV.wrappedValue = WNConv1d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1, bias: false)
        self._toOut.wrappedValue = WNConv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))

        // Project to Q, K, V
        let qkv = toQKV(x)
        let qkvSplit = qkv.split(parts: 3, axis: -1)
        var q = qkvSplit[0]
        var k = qkvSplit[1]
        var v = qkvSplit[2]

        // Reshape for multi-head attention: [B, L, H, D] -> [B, H, L, D]
        q = q.reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)

        // Use MLXFast scaled dot product attention
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )

        // Reshape back: [B, H, L, D] -> [B, L, H*D]
        let output = attnOutput.transposed(0, 2, 1, 3).reshaped(batch, length, dim)

        return toOut(output)
    }
}

// MARK: - Sequential Helper

/// Simple sequential container for modules
final class Sequential: Module, @unchecked Sendable {
    // Use @ModuleInfo to expose layers to the Module system
    @ModuleInfo var layers: [Module]

    init(_ layers: [Module]) {
        self._layers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        for layer in layers {
            if let unary = layer as? any UnaryLayer {
                result = unary.callAsFunction(result)
            }
        }
        return result
    }
}
