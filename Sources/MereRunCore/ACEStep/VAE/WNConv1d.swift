import Foundation
import MLX
import MLXNN
import MLXRandom

/// Weight-normalized 1D convolution used by Oobleck components.
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

        var result = MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        if let bias {
            result = result + bias
        }
        return result
    }
}
