import MLX
import MLXNN

/// Symmetrically padded 3D convolution, unlike the Wan VAE's causal convolution.
final class FourDAnyonePoseConvolution: Module {
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray
    let kernel: [Int]
    let stride: [Int]

    init(input: Int, output: Int, kernel: [Int], stride: [Int]) {
        self.kernel = kernel
        self.stride = stride
        self._weight.wrappedValue = MLX.zeros([output, kernel[0], kernel[1], kernel[2], input])
        self._bias.wrappedValue = MLX.zeros([output])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let padded = MLX.padded(input.asType(weight.dtype), widths: [
            [0, 0], [1, 1], [1, 1], [1, 1], [0, 0],
        ])
        let frames = (padded.dim(1) - kernel[0]) / stride[0] + 1
        var slices: [MLXArray] = []
        for index in 0..<kernel[0] {
            let indices = MLXArray((0..<frames).map { Int32($0 * stride[0] + index) })
            let images = MLX.take(padded, indices, axis: 1)
                .reshaped(padded.dim(0) * frames, padded.dim(2), padded.dim(3), padded.dim(4))
            slices.append(MLX.convGeneral(
                images, weight[0..., index, 0..., 0..., 0...], strides: [stride[1], stride[2]]
            ))
        }
        let convolved = slices.dropFirst().reduce(slices[0], +) + bias
        return convolved.reshaped(
            input.dim(0), frames, convolved.dim(1), convolved.dim(2), convolved.dim(3)
        )
    }
}

/// Encodes skeleton RGB videos and the learned all-minus-one null condition.
public final class FourDAnyonePoseEncoder: Module {
    @ModuleInfo var convolutions: [FourDAnyonePoseConvolution]
    @ModuleInfo(key: "final_proj") var projection: Linear
    @ModuleInfo var scale: MLXArray
    public let outputChannels: Int

    public init(outputChannels: Int = 3_072) {
        precondition(outputChannels > 0)
        self.outputChannels = outputChannels
        let channels = [3, 3, 16, 16, 32, 32, 64, 64, 128, 128, 256]
        self._convolutions.wrappedValue = (0..<10).map { index in
            let downsample = !index.isMultiple(of: 2)
            return FourDAnyonePoseConvolution(
                input: channels[index], output: channels[index + 1],
                kernel: downsample ? [3, 4, 4] : [3, 3, 3],
                stride: downsample ? [index >= 7 ? 2 : 1, 2, 2] : [1, 1, 1]
            )
        }
        self._projection.wrappedValue = Linear(256, outputChannels)
        self._scale.wrappedValue = MLXArray([Float(2)])
    }

    /// Input is [views, 3, 4n+1, 32h, 32w]; output is [views, channels, n+1, h, w].
    public func callAsFunction(
        _ videos: MLXArray,
        checkCancellation: () throws -> Void = { try Task.checkCancellation() }
    ) throws -> MLXArray {
        guard videos.ndim == 5, videos.shape.allSatisfy({ $0 > 0 }),
              videos.dim(1) == 3, (videos.dim(2) - 1).isMultiple(of: 4),
              videos.dim(3).isMultiple(of: 32), videos.dim(4).isMultiple(of: 32) else {
            throw FourDAnyoneError.invalidInput("Skeleton videos must have shape [views, 3, 4n+1, 32h, 32w].")
        }
        let prefix = MLX.repeated(videos[0..., 0..., 0..<1, 0..., 0...], count: 3, axis: 2)
        var hidden = MLX.concatenated([prefix, videos], axis: 2).transposed(0, 2, 3, 4, 1)
        for convolution in convolutions {
            try checkCancellation()
            hidden = FourDAnyoneActivation.silu(convolution(hidden))
            eval(hidden)
        }
        return (projection(hidden.asType(projection.weight.dtype)) * scale).transposed(0, 4, 1, 2, 3)
    }
}
