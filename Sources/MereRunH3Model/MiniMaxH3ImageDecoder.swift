import Foundation
import MLX
import MLXNN

/// The image VAE rotates two spatial axes; the video VAE's three-axis clock does not apply here.
public final class MiniMaxH3ImageDecoder: Module {
    @ModuleInfo(key: "proj_in") var input: Linear
    @ParameterInfo(key: "register_tokens") var registers: MLXArray
    @ModuleInfo(key: "transformer_blocks") var blocks: [MiniMaxH3VideoDecoderBlock]
    @ModuleInfo(key: "norm_out") var norm: LayerNorm
    @ModuleInfo(key: "proj_out") var output: Linear
    let configuration: MiniMaxH3VideoDecoderConfiguration

    public init(configuration: MiniMaxH3VideoDecoderConfiguration = .init(
        temporalPatchSize: 1, layerCount: 24, headCount: 16, rotaryDimensionRatio: 0.5
    )) {
        self.configuration = configuration
        _input.wrappedValue = Linear(configuration.latentChannels, configuration.hiddenSize)
        _registers.wrappedValue = MLXArray.zeros([1, configuration.registerTokenCount, configuration.hiddenSize])
        _blocks.wrappedValue = (0..<configuration.layerCount).map { _ in MiniMaxH3VideoDecoderBlock(configuration: configuration) }
        _norm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _output.wrappedValue = Linear(configuration.hiddenSize, configuration.outputChannels * configuration.patchSize * configuration.patchSize)
    }

    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0), height = latents.dim(2), width = latents.dim(3)
        var hidden = input(latents.transposed(0, 2, 3, 1).reshaped(batch, height * width, -1))
        hidden = MLX.concatenated([hidden, MLX.tiled(registers, repetitions: [batch, 1, 1]).asType(hidden.dtype),
                                  MLXArray.zeros([batch, 1, configuration.hiddenSize], dtype: hidden.dtype)], axis: 1)
        let rope = rotaryEmbedding(height: height, width: width)
        for block in blocks {
            hidden = block(hidden, cosine: rope.0, sine: rope.1)
            MLX.eval(hidden)
        }
        hidden = output(norm(hidden))[0..., 0..<(height * width), 0...]
        let patch = configuration.patchSize
        return hidden.reshaped(batch, height, width, configuration.outputChannels, patch, patch)
            .transposed(0, 3, 1, 4, 2, 5).reshaped(batch, configuration.outputChannels, height * patch, width * patch)
    }

    package func rotaryEmbedding(height: Int, width: Int) -> (MLXArray, MLXArray) {
        let axisDimension = configuration.rotaryDimension / 4
        let inverse = MLX.pow(MLXArray(configuration.rotaryTheta),
            -MLXArray((0..<axisDimension).map { Float($0) / Float(axisDimension) }))
        func grid(_ size: Int) -> MLXArray {
            (2 * ((MLXArray(0..<size).asType(.float32) + 0.5) / Float(size)) - 1) * (2 * Float.pi)
        }
        let y = grid(height).reshaped(height, 1, 1) * inverse
        let x = grid(width).reshaped(1, width, 1) * inverse
        let angles = MLX.concatenated([MLX.broadcast(y, to: [height, width, axisDimension]),
                                       MLX.broadcast(x, to: [height, width, axisDimension])], axis: -1)
            .reshaped(1, height * width, 1, configuration.rotaryDimension / 2)
        let doubled = MLX.concatenated([angles, angles], axis: -1)
        let complete = MLX.concatenated([doubled, MLXArray.zeros([1, configuration.registerTokenCount + 1, 1, configuration.rotaryDimension])], axis: 1)
        return (MLX.cos(complete), MLX.sin(complete))
    }
}
