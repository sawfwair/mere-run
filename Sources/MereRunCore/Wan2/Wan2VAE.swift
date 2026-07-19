import Foundation
import MLX
import MLXFast
import MLXNN

private let wan2VAE22MeanValues: [Float] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]

private let wan2VAE22StandardDeviationValues: [Float] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

private let wan2VAE21MeanValues: [Float] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]

private let wan2VAE21StandardDeviationValues: [Float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

public struct Wan2VAEConfiguration: Hashable, Sendable {
    public let latentChannels: Int
    public let encoderDimensions: Int
    public let decoderDimensions: Int
    public let imagePatchSize: Int
    public let blockResampleShortcut: Bool
    public let decoderResampleReducesChannels: Bool
    public let latentMean: [Float]
    public let latentStandardDeviation: [Float]

    public init(
        latentChannels: Int,
        encoderDimensions: Int,
        decoderDimensions: Int,
        imagePatchSize: Int,
        blockResampleShortcut: Bool = true,
        decoderResampleReducesChannels: Bool = false,
        latentMean: [Float],
        latentStandardDeviation: [Float]
    ) {
        precondition(latentChannels > 0)
        precondition(encoderDimensions > 0 && decoderDimensions > 0)
        precondition(imagePatchSize > 0)
        precondition(latentMean.count == latentChannels)
        precondition(latentStandardDeviation.count == latentChannels)
        precondition(latentStandardDeviation.allSatisfy { $0 > 0 })
        self.latentChannels = latentChannels
        self.encoderDimensions = encoderDimensions
        self.decoderDimensions = decoderDimensions
        self.imagePatchSize = imagePatchSize
        self.blockResampleShortcut = blockResampleShortcut
        self.decoderResampleReducesChannels = decoderResampleReducesChannels
        self.latentMean = latentMean
        self.latentStandardDeviation = latentStandardDeviation
    }

    public static let wan22TI2V = Wan2VAEConfiguration(
        latentChannels: 48,
        encoderDimensions: 160,
        decoderDimensions: 256,
        imagePatchSize: 2,
        blockResampleShortcut: true,
        latentMean: wan2VAE22MeanValues,
        latentStandardDeviation: wan2VAE22StandardDeviationValues
    )

    public static let wan21 = Wan2VAEConfiguration(
        latentChannels: 16,
        encoderDimensions: 96,
        decoderDimensions: 96,
        imagePatchSize: 1,
        blockResampleShortcut: false,
        decoderResampleReducesChannels: true,
        latentMean: wan2VAE21MeanValues,
        latentStandardDeviation: wan2VAE21StandardDeviationValues
    )
}

final class Wan2VAECausalConv3D: Module {
    let kernel: (temporal: Int, height: Int, width: Int)
    let stride: (temporal: Int, height: Int, width: Int)
    let temporalPadding: Int
    let heightPadding: Int
    let widthPadding: Int
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernel: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (0, 0, 0)
    ) {
        self.kernel = kernel
        self.stride = stride
        self.temporalPadding = 2 * padding.0
        self.heightPadding = padding.1
        self.widthPadding = padding.2
        self._weight.wrappedValue = MLX.zeros([
            outputChannels, kernel.0, kernel.1, kernel.2, inputChannels,
        ])
        self._bias.wrappedValue = MLX.zeros([outputChannels])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.ndim == 5)
        let batch = input.dim(0)
        var hidden = input
        if temporalPadding > 0 {
            let prefix = MLX.zeros(
                [batch, temporalPadding, input.dim(2), input.dim(3), input.dim(4)],
                dtype: input.dtype
            )
            hidden = MLX.concatenated([prefix, hidden], axis: 1)
        }
        if heightPadding > 0 || widthPadding > 0 {
            hidden = MLX.padded(hidden, widths: [
                [0, 0], [0, 0],
                [heightPadding, heightPadding],
                [widthPadding, widthPadding],
                [0, 0],
            ])
        }
        let outputFrames = (hidden.dim(1) - kernel.temporal) / stride.temporal + 1
        var frames: [MLXArray] = []
        frames.reserveCapacity(outputFrames)
        for outputIndex in 0..<outputFrames {
            let start = outputIndex * stride.temporal
            var accumulated: MLXArray?
            for kernelIndex in 0..<kernel.temporal {
                let frame = hidden[0..., start + kernelIndex]
                let kernel2D = weight[0..., kernelIndex]
                let convolved = MLX.convGeneral(
                    frame,
                    kernel2D,
                    strides: [stride.height, stride.width]
                )
                accumulated = accumulated.map { $0 + convolved } ?? convolved
            }
            frames.append(accumulated! + bias)
        }
        return MLX.stacked(frames, axis: 1)
    }
}

final class Wan2VAERMSNorm: Module {
    let scale: Float
    @ModuleInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int) {
        self.scale = Float(dimensions).squareRoot()
        self._gamma.wrappedValue = MLX.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let squaredNorm = MLX.sum(input * input, axis: -1, keepDims: true)
        return input * MLX.rsqrt(MLX.maximum(squaredNorm, MLXArray(1e-24))) * scale * gamma
    }
}

final class Wan2VAEResidualLayers: Module {
    @ModuleInfo(key: "layer_0") var norm1: Wan2VAERMSNorm
    @ModuleInfo(key: "layer_2") var conv1: Wan2VAECausalConv3D
    @ModuleInfo(key: "layer_3") var norm2: Wan2VAERMSNorm
    @ModuleInfo(key: "layer_6") var conv2: Wan2VAECausalConv3D

    init(inputChannels: Int, outputChannels: Int) {
        self._norm1.wrappedValue = Wan2VAERMSNorm(dimensions: inputChannels)
        self._conv1.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernel: (3, 3, 3),
            padding: (1, 1, 1)
        )
        self._norm2.wrappedValue = Wan2VAERMSNorm(dimensions: outputChannels)
        self._conv2.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: outputChannels,
            outputChannels: outputChannels,
            kernel: (3, 3, 3),
            padding: (1, 1, 1)
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = conv1(MLXNN.silu(norm1(input)))
        eval(hidden)
        hidden = conv2(MLXNN.silu(norm2(hidden)))
        return hidden
    }
}

final class Wan2VAEResidualBlock: Module {
    @ModuleInfo(key: "residual") var residual: Wan2VAEResidualLayers
    @ModuleInfo(key: "shortcut") var shortcut: Wan2VAECausalConv3D?

    init(inputChannels: Int, outputChannels: Int) {
        self._residual.wrappedValue = Wan2VAEResidualLayers(
            inputChannels: inputChannels,
            outputChannels: outputChannels
        )
        self._shortcut.wrappedValue = inputChannels == outputChannels ? nil : Wan2VAECausalConv3D(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernel: (1, 1, 1)
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        residual(input) + (shortcut?(input) ?? input)
    }
}

final class Wan2VAEAttentionBlock: Module {
    let dimensions: Int
    @ModuleInfo(key: "norm") var norm: Wan2VAERMSNorm
    @ModuleInfo(key: "to_qkv_weight") var qkvWeight: MLXArray
    @ModuleInfo(key: "to_qkv_bias") var qkvBias: MLXArray
    @ModuleInfo(key: "proj_weight") var projectionWeight: MLXArray
    @ModuleInfo(key: "proj_bias") var projectionBias: MLXArray

    init(dimensions: Int) {
        self.dimensions = dimensions
        self._norm.wrappedValue = Wan2VAERMSNorm(dimensions: dimensions)
        self._qkvWeight.wrappedValue = MLX.zeros([3 * dimensions, 1, 1, dimensions])
        self._qkvBias.wrappedValue = MLX.zeros([3 * dimensions])
        self._projectionWeight.wrappedValue = MLX.zeros([dimensions, 1, 1, dimensions])
        self._projectionBias.wrappedValue = MLX.zeros([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        let flattened = norm(input.reshaped(batch * frames, height, width, dimensions))
        let qkv = (MLX.convGeneral(flattened, qkvWeight) + qkvBias)
            .reshaped(batch * frames, height * width, 3 * dimensions)
        let parts = MLX.split(qkv, parts: 3, axis: -1)
        let queries = parts[0].expandedDimensions(axis: 1)
        let keys = parts[1].expandedDimensions(axis: 1)
        let values = parts[2].expandedDimensions(axis: 1)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / Float(dimensions).squareRoot(),
            mask: .none
        ).squeezed(axis: 1).reshaped(batch * frames, height, width, dimensions)
        let projected = MLX.convGeneral(attended, projectionWeight) + projectionBias
        return input + projected.reshaped(batch, frames, height, width, dimensions)
    }
}

final class Wan2VAEDuplicateUpsample: Module {
    let outputChannels: Int
    let temporalFactor: Int
    let spatialFactor: Int
    let repeats: Int

    init(inputChannels: Int, outputChannels: Int, temporalFactor: Int, spatialFactor: Int) {
        self.outputChannels = outputChannels
        self.temporalFactor = temporalFactor
        self.spatialFactor = spatialFactor
        self.repeats = outputChannels * temporalFactor * spatialFactor * spatialFactor / inputChannels
    }

    func callAsFunction(_ input: MLXArray, firstChunk: Bool) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        var hidden = MLX.repeated(input, count: repeats, axis: -1)
            .reshaped(
                batch, frames, height, width, outputChannels,
                temporalFactor, spatialFactor, spatialFactor
            )
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(
                batch, frames * temporalFactor,
                height * spatialFactor, width * spatialFactor,
                outputChannels
            )
        if firstChunk && temporalFactor > 1 {
            hidden = hidden[0..., (temporalFactor - 1)...]
        }
        return hidden
    }
}

final class Wan2VAEAverageDownsample: Module {
    let outputChannels: Int
    let temporalFactor: Int
    let spatialFactor: Int
    let groupSize: Int

    init(inputChannels: Int, outputChannels: Int, temporalFactor: Int, spatialFactor: Int) {
        self.outputChannels = outputChannels
        self.temporalFactor = temporalFactor
        self.spatialFactor = spatialFactor
        self.groupSize = inputChannels * temporalFactor * spatialFactor * spatialFactor / outputChannels
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        var frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        let channels = input.dim(4)
        var hidden = input
        let temporalPadding = (temporalFactor - frames % temporalFactor) % temporalFactor
        if temporalPadding > 0 {
            hidden = MLX.padded(hidden, widths: [
                [0, 0], [temporalPadding, 0], [0, 0], [0, 0], [0, 0],
            ])
            frames += temporalPadding
        }
        return hidden
            .reshaped(
                batch, frames / temporalFactor, temporalFactor,
                height / spatialFactor, spatialFactor,
                width / spatialFactor, spatialFactor, channels
            )
            .transposed(0, 1, 3, 5, 7, 2, 4, 6)
            .reshaped(
                batch, frames / temporalFactor,
                height / spatialFactor, width / spatialFactor,
                outputChannels, groupSize
            )
            .mean(axis: -1)
    }
}

final class Wan2VAEResample: Module {
    enum Mode { case up2D, up3D, down2D, down3D }
    let dimensions: Int
    let outputDimensions: Int
    let mode: Mode
    @ModuleInfo(key: "resample_weight") var spatialWeight: MLXArray
    @ModuleInfo(key: "resample_bias") var spatialBias: MLXArray
    @ModuleInfo(key: "time_conv") var temporalConv: Wan2VAECausalConv3D?

    init(dimensions: Int, outputDimensions: Int? = nil, mode: Mode) {
        self.dimensions = dimensions
        self.outputDimensions = outputDimensions ?? dimensions
        self.mode = mode
        self._spatialWeight.wrappedValue = MLX.zeros([self.outputDimensions, 3, 3, dimensions])
        self._spatialBias.wrappedValue = MLX.zeros([self.outputDimensions])
        switch mode {
        case .up3D:
            self._temporalConv.wrappedValue = Wan2VAECausalConv3D(
                inputChannels: dimensions,
                outputChannels: dimensions * 2,
                kernel: (3, 1, 1),
                padding: (1, 0, 0)
            )
        case .down3D:
            self._temporalConv.wrappedValue = Wan2VAECausalConv3D(
                inputChannels: dimensions,
                outputChannels: dimensions,
                kernel: (3, 1, 1),
                stride: (2, 1, 1),
                padding: (0, 0, 0)
            )
        case .up2D, .down2D:
            self._temporalConv.wrappedValue = nil
        }
    }

    func callAsFunction(_ input: MLXArray, firstChunk: Bool = false) -> MLXArray {
        let batch = input.dim(0)
        var frames = input.dim(1)
        var height = input.dim(2)
        var width = input.dim(3)
        var hidden = input
        if mode == .up3D, let temporalConv {
            if firstChunk && frames > 1 {
                let first = hidden[0..., 0..<1]
                let rest = temporalConv(hidden[0..., 1...])
                    .reshaped(batch, frames - 1, height, width, 2, dimensions)
                let interleaved = MLX.stacked(
                    [rest[0..., 0..., 0..., 0..., 0, 0...], rest[0..., 0..., 0..., 0..., 1, 0...]],
                    axis: 2
                ).reshaped(batch, (frames - 1) * 2, height, width, dimensions)
                hidden = MLX.concatenated([first, interleaved], axis: 1)
            } else {
                let temporal = temporalConv(hidden).reshaped(batch, frames, height, width, 2, dimensions)
                hidden = MLX.stacked(
                    [temporal[0..., 0..., 0..., 0..., 0, 0...], temporal[0..., 0..., 0..., 0..., 1, 0...]],
                    axis: 2
                ).reshaped(batch, frames * 2, height, width, dimensions)
            }
            eval(hidden)
            frames = hidden.dim(1)
        }

        switch mode {
        case .up2D, .up3D:
            var flattened = hidden.reshaped(batch * frames, height, width, dimensions)
            flattened = MLX.repeated(flattened, count: 2, axis: 1)
            flattened = MLX.repeated(flattened, count: 2, axis: 2)
            flattened = MLX.padded(flattened, widths: [[0, 0], [1, 1], [1, 1], [0, 0]])
            flattened = MLX.convGeneral(flattened, spatialWeight) + spatialBias
            height = flattened.dim(1)
            width = flattened.dim(2)
            hidden = flattened.reshaped(batch, frames, height, width, outputDimensions)
        case .down2D, .down3D:
            var flattened = hidden.reshaped(batch * frames, height, width, dimensions)
            flattened = MLX.padded(flattened, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
            flattened = MLX.convGeneral(flattened, spatialWeight, strides: [2, 2]) + spatialBias
            height = flattened.dim(1)
            width = flattened.dim(2)
            hidden = flattened.reshaped(batch, frames, height, width, dimensions)
        }

        if mode == .down3D, frames > 1, let temporalConv {
            let first = hidden[0..., 0..<1]
            let downsampled = temporalConv(hidden)
            hidden = MLX.concatenated([first, downsampled], axis: 1)
            eval(hidden)
        }
        return hidden
    }
}

final class Wan2VAEUpBlock: Module {
    @ModuleInfo(key: "avg_shortcut") var shortcut: Wan2VAEDuplicateUpsample?
    @ModuleInfo(key: "upsamples") var layers: [Module]

    init(
        inputChannels: Int,
        outputChannels: Int,
        temporalUpsample: Bool,
        upsample: Bool,
        useBlockShortcut: Bool,
        resampleOutputChannels: Int
    ) {
        self._shortcut.wrappedValue = upsample && useBlockShortcut ? Wan2VAEDuplicateUpsample(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            temporalFactor: temporalUpsample ? 2 : 1,
            spatialFactor: 2
        ) : nil
        var modules: [Module] = []
        var input = inputChannels
        for _ in 0..<3 {
            modules.append(Wan2VAEResidualBlock(inputChannels: input, outputChannels: outputChannels))
            input = outputChannels
        }
        if upsample {
            modules.append(Wan2VAEResample(
                dimensions: outputChannels,
                outputDimensions: resampleOutputChannels,
                mode: temporalUpsample ? .up3D : .up2D
            ))
        }
        self._layers.wrappedValue = modules
    }

    func callAsFunction(_ input: MLXArray, firstChunk: Bool) -> MLXArray {
        var hidden = input
        for layer in layers {
            if let residual = layer as? Wan2VAEResidualBlock {
                hidden = residual(hidden)
            } else if let resample = layer as? Wan2VAEResample {
                hidden = resample(hidden, firstChunk: firstChunk)
            }
            eval(hidden)
        }
        guard let shortcut else { return hidden }
        let skip = shortcut(input, firstChunk: firstChunk)
        eval(skip)
        return hidden + skip
    }
}

final class Wan2VAEDownBlock: Module {
    @ModuleInfo(key: "avg_shortcut") var shortcut: Wan2VAEAverageDownsample?
    @ModuleInfo(key: "downsamples") var layers: [Module]

    init(
        inputChannels: Int,
        outputChannels: Int,
        temporalDownsample: Bool,
        downsample: Bool,
        useBlockShortcut: Bool
    ) {
        self._shortcut.wrappedValue = useBlockShortcut ? Wan2VAEAverageDownsample(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            temporalFactor: temporalDownsample ? 2 : 1,
            spatialFactor: downsample ? 2 : 1
        ) : nil
        var modules: [Module] = []
        var input = inputChannels
        for _ in 0..<2 {
            modules.append(Wan2VAEResidualBlock(inputChannels: input, outputChannels: outputChannels))
            input = outputChannels
        }
        if downsample {
            modules.append(Wan2VAEResample(
                dimensions: outputChannels,
                mode: temporalDownsample ? .down3D : .down2D
            ))
        }
        self._layers.wrappedValue = modules
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let skip = shortcut?(input)
        if let skip { eval(skip) }
        var hidden = input
        for layer in layers {
            if let residual = layer as? Wan2VAEResidualBlock {
                hidden = residual(hidden)
            } else if let resample = layer as? Wan2VAEResample {
                hidden = resample(hidden)
            }
            eval(hidden)
        }
        return skip.map { hidden + $0 } ?? hidden
    }
}

final class Wan2VAEHead: Module {
    @ModuleInfo(key: "layer_0") var norm: Wan2VAERMSNorm
    @ModuleInfo(key: "layer_2") var conv: Wan2VAECausalConv3D

    init(inputChannels: Int, outputChannels: Int = 12) {
        self._norm.wrappedValue = Wan2VAERMSNorm(dimensions: inputChannels)
        self._conv.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernel: (3, 3, 3),
            padding: (1, 1, 1)
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        conv(MLXNN.silu(norm(input)))
    }
}

final class Wan2VAEDecoder3D: Module {
    @ModuleInfo(key: "conv1") var inputConv: Wan2VAECausalConv3D
    @ModuleInfo(key: "middle") var middle: [Module]
    @ModuleInfo(key: "upsamples") var upsampleBlocks: [Wan2VAEUpBlock]
    @ModuleInfo(key: "head") var head: Wan2VAEHead

    init(
        baseDimensions: Int = 256,
        latentChannels: Int = 48,
        outputChannels: Int = 12,
        useBlockShortcuts: Bool = true,
        resampleReducesChannels: Bool = false
    ) {
        let dimensions = [baseDimensions * 4, baseDimensions * 4, baseDimensions * 4, baseDimensions * 2, baseDimensions]
        self._inputConv.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: latentChannels,
            outputChannels: dimensions[0],
            kernel: (3, 3, 3),
            padding: (1, 1, 1)
        )
        self._middle.wrappedValue = [
            Wan2VAEResidualBlock(inputChannels: dimensions[0], outputChannels: dimensions[0]),
            Wan2VAEAttentionBlock(dimensions: dimensions[0]),
            Wan2VAEResidualBlock(inputChannels: dimensions[0], outputChannels: dimensions[0]),
        ]
        var currentDimensions = dimensions[0]
        self._upsampleBlocks.wrappedValue = (0..<4).map { index in
            let block = Wan2VAEUpBlock(
                inputChannels: currentDimensions,
                outputChannels: dimensions[index + 1],
                temporalUpsample: index < 2,
                upsample: index < 3,
                useBlockShortcut: useBlockShortcuts,
                resampleOutputChannels: resampleReducesChannels && index < 3
                    ? dimensions[index + 1] / 2
                    : dimensions[index + 1]
            )
            currentDimensions = resampleReducesChannels && index < 3
                ? dimensions[index + 1] / 2
                : dimensions[index + 1]
            return block
        }
        self._head.wrappedValue = Wan2VAEHead(
            inputChannels: dimensions.last!,
            outputChannels: outputChannels
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = inputConv(input)
        for layer in middle {
            if let residual = layer as? Wan2VAEResidualBlock {
                hidden = residual(hidden)
            } else if let attention = layer as? Wan2VAEAttentionBlock {
                hidden = attention(hidden)
            }
        }
        eval(hidden)
        for block in upsampleBlocks {
            hidden = block(hidden, firstChunk: true)
            eval(hidden)
        }
        return head(hidden)
    }
}

final class Wan2VAEEncoder3D: Module {
    @ModuleInfo(key: "conv1") var inputConv: Wan2VAECausalConv3D
    @ModuleInfo(key: "downsamples") var downsampleBlocks: [Wan2VAEDownBlock]
    @ModuleInfo(key: "middle") var middle: [Module]
    @ModuleInfo(key: "head") var head: Wan2VAEHead

    init(
        baseDimensions: Int = 160,
        inputChannels: Int = 12,
        outputChannels: Int = 96,
        useBlockShortcuts: Bool = true
    ) {
        let dimensions = [baseDimensions, baseDimensions, baseDimensions * 2, baseDimensions * 4, baseDimensions * 4]
        self._inputConv.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: inputChannels,
            outputChannels: dimensions[0],
            kernel: (3, 3, 3),
            padding: (1, 1, 1)
        )
        self._downsampleBlocks.wrappedValue = (0..<4).map { index in
            Wan2VAEDownBlock(
                inputChannels: dimensions[index],
                outputChannels: dimensions[index + 1],
                temporalDownsample: index == 1 || index == 2,
                downsample: index < 3,
                useBlockShortcut: useBlockShortcuts
            )
        }
        self._middle.wrappedValue = [
            Wan2VAEResidualBlock(inputChannels: dimensions.last!, outputChannels: dimensions.last!),
            Wan2VAEAttentionBlock(dimensions: dimensions.last!),
            Wan2VAEResidualBlock(inputChannels: dimensions.last!, outputChannels: dimensions.last!),
        ]
        self._head.wrappedValue = Wan2VAEHead(inputChannels: dimensions.last!, outputChannels: outputChannels)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = inputConv(input)
        for block in downsampleBlocks {
            hidden = block(hidden)
        }
        for layer in middle {
            if let residual = layer as? Wan2VAEResidualBlock {
                hidden = residual(hidden)
            } else if let attention = layer as? Wan2VAEAttentionBlock {
                hidden = attention(hidden)
            }
        }
        eval(hidden)
        return head(hidden)
    }
}

public final class Wan2VAEModel: Module {
    let latentChannels: Int
    let imagePatchSize: Int
    let latentMean: MLXArray
    let latentStandardDeviation: MLXArray
    @ModuleInfo(key: "conv1") var encoderProjection: Wan2VAECausalConv3D
    @ModuleInfo(key: "encoder") var encoder: Wan2VAEEncoder3D
    @ModuleInfo(key: "conv2") var decoderProjection: Wan2VAECausalConv3D
    @ModuleInfo(key: "decoder") var decoder: Wan2VAEDecoder3D

    public convenience init(
        latentChannels: Int = 48,
        encoderDimensions: Int = 160,
        decoderDimensions: Int = 256
    ) {
        let defaults = Wan2VAEConfiguration.wan22TI2V
        self.init(configuration: Wan2VAEConfiguration(
            latentChannels: latentChannels,
            encoderDimensions: encoderDimensions,
            decoderDimensions: decoderDimensions,
            imagePatchSize: defaults.imagePatchSize,
            blockResampleShortcut: defaults.blockResampleShortcut,
            decoderResampleReducesChannels: defaults.decoderResampleReducesChannels,
            latentMean: Array(defaults.latentMean.prefix(latentChannels)),
            latentStandardDeviation: Array(defaults.latentStandardDeviation.prefix(latentChannels))
        ))
    }

    public init(configuration: Wan2VAEConfiguration) {
        self.latentChannels = configuration.latentChannels
        self.imagePatchSize = configuration.imagePatchSize
        self.latentMean = MLXArray(configuration.latentMean).reshaped(
            1, 1, 1, 1, configuration.latentChannels
        )
        self.latentStandardDeviation = MLXArray(configuration.latentStandardDeviation).reshaped(
            1, 1, 1, 1, configuration.latentChannels
        )
        let imageChannels = 3 * configuration.imagePatchSize * configuration.imagePatchSize
        self._encoderProjection.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: configuration.latentChannels * 2,
            outputChannels: configuration.latentChannels * 2,
            kernel: (1, 1, 1)
        )
        self._encoder.wrappedValue = Wan2VAEEncoder3D(
            baseDimensions: configuration.encoderDimensions,
            inputChannels: imageChannels,
            outputChannels: configuration.latentChannels * 2,
            useBlockShortcuts: configuration.blockResampleShortcut
        )
        self._decoderProjection.wrappedValue = Wan2VAECausalConv3D(
            inputChannels: configuration.latentChannels,
            outputChannels: configuration.latentChannels,
            kernel: (1, 1, 1)
        )
        self._decoder.wrappedValue = Wan2VAEDecoder3D(
            baseDimensions: configuration.decoderDimensions,
            latentChannels: configuration.latentChannels,
            outputChannels: imageChannels,
            useBlockShortcuts: configuration.blockResampleShortcut,
            resampleReducesChannels: configuration.decoderResampleReducesChannels
        )
    }

    public func encodeImage(_ image: MLXArray) -> MLXArray {
        precondition(image.ndim == 5 && image.dim(1) == 1 && image.dim(4) == 3)
        return encodeVideo(image)
    }

    public func encodeVideo(_ video: MLXArray) -> MLXArray {
        precondition(video.ndim == 5 && video.dim(4) == 3)
        precondition(video.dim(1) >= 1 && (video.dim(1) - 1) % 4 == 0)
        precondition(video.dim(2).isMultiple(of: imagePatchSize))
        precondition(video.dim(3).isMultiple(of: imagePatchSize))
        let packed = Self.patchify(video, patchSize: imagePatchSize)
        let moments = encoderProjection(encoder(packed))
        let mean = moments[0..., 0..., 0..., 0..., 0..<latentChannels]
        return normalizeLatents(mean)
    }

    public func decode(_ normalizedLatents: MLXArray) -> MLXArray {
        precondition(normalizedLatents.dim(4) == latentChannels)
        let denormalized = denormalizeLatents(normalizedLatents)
        let decoded = decoder(decoderProjection(denormalized))
        return MLX.clip(Self.unpatchify(decoded, patchSize: imagePatchSize), min: -1, max: 1)
    }

    public func normalizeLatents(_ latents: MLXArray) -> MLXArray {
        precondition(latents.dim(4) == latentChannels)
        return (latents - latentMean) / latentStandardDeviation
    }

    public func denormalizeLatents(_ latents: MLXArray) -> MLXArray {
        precondition(latents.dim(4) == latentChannels)
        return latents * latentStandardDeviation + latentMean
    }

    public static func normalize(_ latents: MLXArray) -> MLXArray {
        let mean = MLXArray(wan2VAE22MeanValues).reshaped(1, 1, 1, 1, 48)
        let standardDeviation = MLXArray(wan2VAE22StandardDeviationValues).reshaped(1, 1, 1, 1, 48)
        return (latents - mean) / standardDeviation
    }

    public static func denormalize(_ latents: MLXArray) -> MLXArray {
        let mean = MLXArray(wan2VAE22MeanValues).reshaped(1, 1, 1, 1, 48)
        let standardDeviation = MLXArray(wan2VAE22StandardDeviationValues).reshaped(1, 1, 1, 1, 48)
        return latents * standardDeviation + mean
    }

    static func patchify(_ input: MLXArray, patchSize: Int = 2) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2) / patchSize
        let width = input.dim(3) / patchSize
        let channels = input.dim(4)
        return input
            .reshaped(batch, frames, height, patchSize, width, patchSize, channels)
            .transposed(0, 1, 2, 4, 6, 5, 3)
            .reshaped(batch, frames, height, width, channels * patchSize * patchSize)
    }

    static func unpatchify(_ input: MLXArray, patchSize: Int = 2) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        let channels = input.dim(4) / (patchSize * patchSize)
        return input
            .reshaped(batch, frames, height, width, channels, patchSize, patchSize)
            .transposed(0, 1, 2, 6, 3, 5, 4)
            .reshaped(batch, frames, height * patchSize, width * patchSize, channels)
    }
}
