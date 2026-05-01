import Foundation
import MLX
import MLXNN

final class OobleckSnakeBeta: Module, UnaryLayer {
    @ModuleInfo(key: "alpha") var alpha: MLXArray
    @ModuleInfo(key: "beta") var beta: MLXArray

    private let eps: Float = 1e-9

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([1, 1, channels])
        self._beta.wrappedValue = MLXArray.zeros([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaExp = MLX.exp(alpha)
        let betaExp = MLX.exp(beta)
        return x + (1.0 / (betaExp + eps)) * MLX.pow(MLX.sin(x * alphaExp), 2)
    }
}

final class OobleckResUnit: Module {
    @ModuleInfo(key: "snake1") var snake1: OobleckSnakeBeta
    @ModuleInfo(key: "conv1") var conv1: WNConv1d
    @ModuleInfo(key: "snake2") var snake2: OobleckSnakeBeta
    @ModuleInfo(key: "conv2") var conv2: WNConv1d

    init(channels: Int, dilation: Int) {
        self._snake1.wrappedValue = OobleckSnakeBeta(channels: channels)
        self._conv1.wrappedValue = WNConv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 7,
            stride: 1,
            padding: 3 * dilation,
            dilation: dilation,
            bias: true
        )

        self._snake2.wrappedValue = OobleckSnakeBeta(channels: channels)
        self._conv2.wrappedValue = WNConv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return x + h
    }
}
