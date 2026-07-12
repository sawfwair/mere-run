import Foundation
@preconcurrency import MLX
import MLXNN

public struct MoGe2RawOutput {
    public let points: MLXArray
    public let normals: MLXArray
    public let maskProbability: MLXArray
    public let metricScale: MLXArray
}

private func moge2ReplicatePad(_ input: MLXArray, padding: Int = 1) -> MLXArray {
    guard padding > 0 else { return input }
    let height = input.dim(1)
    let width = input.dim(2)
    precondition(height > 0 && width > 0)
    let top = MLX.repeated(input[0..., 0..<1, 0..., 0...], count: padding, axis: 1)
    let bottom = MLX.repeated(input[0..., (height - 1)..<height, 0..., 0...], count: padding, axis: 1)
    let vertical = MLX.concatenated([top, input, bottom], axis: 1)
    let left = MLX.repeated(vertical[0..., 0..., 0..<1, 0...], count: padding, axis: 2)
    let right = MLX.repeated(vertical[0..., 0..., (width - 1)..<width, 0...], count: padding, axis: 2)
    return MLX.concatenated([left, vertical, right], axis: 2)
}

final class MoGe2ResidualConvBlock: Module {
    @ModuleInfo(key: "first") var first: Conv2d
    @ModuleInfo(key: "second") var second: Conv2d

    init(channels: Int) {
        self._first.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 0,
            bias: true
        )
        self._second.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = relu(input)
        hidden = first(moge2ReplicatePad(hidden))
        hidden = relu(hidden)
        hidden = second(moge2ReplicatePad(hidden))
        return input + hidden
    }
}

final class MoGe2ResidualLevel: Module {
    @ModuleInfo(key: "blocks") var blocks: [MoGe2ResidualConvBlock]

    init(channels: Int, count: Int) {
        self._blocks.wrappedValue = (0..<count).map { _ in MoGe2ResidualConvBlock(channels: channels) }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        blocks.reduce(input) { hidden, block in block(hidden) }
    }
}

final class MoGe2Resampler: Module {
    enum Kind {
        case transposed
        case bilinear
    }

    let kind: Kind
    @ModuleInfo(key: "transpose") var transpose: ConvTransposed2d?
    @ModuleInfo(key: "conv") var convolution: Conv2d

    init(inputChannels: Int, outputChannels: Int, kind: Kind) {
        self.kind = kind
        if kind == .transposed {
            self._transpose.wrappedValue = ConvTransposed2d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: true
            )
        } else {
            self._transpose.wrappedValue = nil
        }
        self._convolution.wrappedValue = Conv2d(
            inputChannels: kind == .transposed ? outputChannels : inputChannels,
            outputChannels: outputChannels,
            kernelSize: 3,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let upsampled: MLXArray
        switch kind {
        case .transposed:
            upsampled = transpose!(input)
        case .bilinear:
            upsampled = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: false))(input)
        }
        return convolution(moge2ReplicatePad(upsampled))
    }
}

final class MoGe2ConvStack: Module {
    let channels: [Int]
    @ModuleInfo(key: "input_blocks") var inputBlocks: [Conv2d]
    @ModuleInfo(key: "resamplers") var resamplers: [MoGe2Resampler]
    @ModuleInfo(key: "res_blocks") var residualLevels: [MoGe2ResidualLevel]
    @ModuleInfo(key: "output") var output: Conv2d?

    init(inputChannels: [Int], channels: [Int], outputChannels: Int? = nil) {
        precondition(inputChannels.count == 5 && channels.count == 5)
        self.channels = channels
        self._inputBlocks.wrappedValue = zip(inputChannels, channels).map { input, output in
            Conv2d(inputChannels: input, outputChannels: output, kernelSize: 1, padding: 0, bias: true)
        }
        self._resamplers.wrappedValue = [
            MoGe2Resampler(inputChannels: channels[0], outputChannels: channels[1], kind: .transposed),
            MoGe2Resampler(inputChannels: channels[1], outputChannels: channels[2], kind: .transposed),
            MoGe2Resampler(inputChannels: channels[2], outputChannels: channels[3], kind: .transposed),
            MoGe2Resampler(inputChannels: channels[3], outputChannels: channels[4], kind: .bilinear),
        ]
        self._residualLevels.wrappedValue = [
            MoGe2ResidualLevel(channels: channels[0], count: 0),
            MoGe2ResidualLevel(channels: channels[1], count: 1),
            MoGe2ResidualLevel(channels: channels[2], count: 1),
            MoGe2ResidualLevel(channels: channels[3], count: 1),
            MoGe2ResidualLevel(channels: channels[4], count: 0),
        ]
        self._output.wrappedValue = outputChannels.map {
            Conv2d(inputChannels: channels[4], outputChannels: $0, kernelSize: 1, padding: 0, bias: true)
        }
        super.init()
    }

    func callAsFunction(_ inputs: [MLXArray]) -> [MLXArray] {
        precondition(inputs.count == 5)
        var outputs: [MLXArray] = []
        var hidden: MLXArray?
        for level in 0..<5 {
            let projected = inputBlocks[level](inputs[level])
            hidden = level == 0 ? projected : hidden! + projected
            hidden = residualLevels[level](hidden!)
            if level == 4, let output {
                outputs.append(output(hidden!))
            } else {
                outputs.append(hidden!)
            }
            if level < 4 { hidden = resamplers[level](hidden!) }
        }
        return outputs
    }
}

final class MoGe2Encoder: Module {
    @ModuleInfo(key: "backbone") var backbone: DINOv2VisionTransformer
    @ModuleInfo(key: "output_projections") var outputProjections: [Conv2d]

    override init() {
        // The pinned deployment graph was exported with MoGe's
        // `onnx_compatible_mode`, which selects explicit output-size bicubic
        // interpolation and therefore omits DINOv2's historical +0.1 offset.
        self._backbone.wrappedValue = DINOv2VisionTransformer(
            configuration: DINOv2Configuration(positionInterpolationOffset: 0)
        )
        self._outputProjections.wrappedValue = (0..<2).map { _ in
            Conv2d(inputChannels: 384, outputChannels: 384, kernelSize: 1, padding: 0, bias: true)
        }
        super.init()
    }

    func callAsFunction(
        _ image: MLXArray,
        tokenRows: Int,
        tokenColumns: Int
    ) -> (features: MLXArray, classToken: MLXArray) {
        let targetHeight = tokenRows * 14
        let targetWidth = tokenColumns * 14
        let resized = Upsample(
            scaleFactor: .array([
                (Float(targetHeight) + 0.001) / Float(image.dim(1)),
                (Float(targetWidth) + 0.001) / Float(image.dim(2)),
            ]),
            mode: .linear(alignCorners: false)
        )(image)
        precondition(resized.dim(1) == targetHeight && resized.dim(2) == targetWidth)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 3).asType(resized.dtype)
        let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 3).asType(resized.dtype)
        let encoded = backbone((resized - mean) / std, intermediateLayers: [5, 11])
        let layer5 = outputProjections[0](encoded.featuresByLayer[5]!)
        let layer11 = outputProjections[1](encoded.featuresByLayer[11]!)
        return (layer5 + layer11, encoded.classToken)
    }
}

final class MoGe2ScaleHead: Module {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "second") var second: Linear
    @ModuleInfo(key: "output") var output: Linear

    override init() {
        self._first.wrappedValue = Linear(384, 384, bias: true)
        self._second.wrappedValue = Linear(384, 384, bias: true)
        self._output.wrappedValue = Linear(384, 1, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        output(relu(second(relu(first(input)))))
    }
}

public final class MoGe2Model: Module {
    @ModuleInfo(key: "encoder") var encoder: MoGe2Encoder
    @ModuleInfo(key: "neck") var neck: MoGe2ConvStack
    @ModuleInfo(key: "points_head") var pointsHead: MoGe2ConvStack
    @ModuleInfo(key: "normal_head") var normalHead: MoGe2ConvStack
    @ModuleInfo(key: "mask_head") var maskHead: MoGe2ConvStack
    @ModuleInfo(key: "scale_head") var scaleHead: MoGe2ScaleHead

    public override init() {
        let channels = [384, 256, 128, 64, 32]
        self._encoder.wrappedValue = MoGe2Encoder()
        self._neck.wrappedValue = MoGe2ConvStack(inputChannels: [386, 2, 2, 2, 2], channels: channels)
        self._pointsHead.wrappedValue = MoGe2ConvStack(inputChannels: channels, channels: channels, outputChannels: 3)
        self._normalHead.wrappedValue = MoGe2ConvStack(inputChannels: channels, channels: channels, outputChannels: 3)
        self._maskHead.wrappedValue = MoGe2ConvStack(inputChannels: channels, channels: channels, outputChannels: 1)
        self._scaleHead.wrappedValue = MoGe2ScaleHead()
        super.init()
    }

    /// Input is NHWC RGB in [0, 1]. This is the authoritative raw forward pass;
    /// focal/shift recovery and camera reprojection are performed separately.
    public func callAsFunction(_ image: MLXArray, tokenCount: Int) -> MoGe2RawOutput {
        precondition(image.ndim == 4 && image.dim(3) == 3)
        let originalHeight = image.dim(1)
        let originalWidth = image.dim(2)
        let aspect = Float(originalWidth) / Float(originalHeight)
        let tokenRows = max(1, Int(round(sqrt(Float(tokenCount) / aspect))))
        let tokenColumns = max(1, Int(round(sqrt(Float(tokenCount) * aspect))))
        let encoded = encoder(image, tokenRows: tokenRows, tokenColumns: tokenColumns)
        var pyramid: [MLXArray] = []
        for level in 0..<5 {
            let height = tokenRows * (1 << level)
            let width = tokenColumns * (1 << level)
            let uv = Self.normalizedViewPlaneUV(
                width: width,
                height: height,
                aspectRatio: aspect,
                batch: image.dim(0),
                dtype: image.dtype
            )
            pyramid.append(level == 0 ? MLX.concatenated([encoded.features, uv], axis: 3) : uv)
        }
        let neckFeatures = neck(pyramid)
        let rawPoints = resize(pointsHead(neckFeatures).last!, height: originalHeight, width: originalWidth)
        let rawNormals = resize(normalHead(neckFeatures).last!, height: originalHeight, width: originalWidth)
        let rawMask = resize(maskHead(neckFeatures).last!, height: originalHeight, width: originalWidth)

        let z = MLX.exp(rawPoints[0..., 0..., 0..., 2..<3])
        let points = MLX.concatenated([rawPoints[0..., 0..., 0..., 0..<2] * z, z], axis: 3)
        let normals = rawNormals / MLX.sqrt(
            MLX.sum(rawNormals * rawNormals, axis: -1, keepDims: true) + MLXArray(Float(1e-12))
        )
        return MoGe2RawOutput(
            points: points,
            normals: normals,
            maskProbability: MLX.sigmoid(rawMask).squeezed(axis: -1),
            metricScale: MLX.exp(scaleHead(encoded.classToken)).squeezed(axis: -1)
        )
    }

    private func resize(_ input: MLXArray, height: Int, width: Int) -> MLXArray {
        if input.dim(1) == height && input.dim(2) == width { return input }
        let resized = Upsample(
            scaleFactor: .array([
                (Float(height) + 0.001) / Float(input.dim(1)),
                (Float(width) + 0.001) / Float(input.dim(2)),
            ]),
            mode: .linear(alignCorners: false)
        )(input)
        precondition(resized.dim(1) == height && resized.dim(2) == width)
        return resized
    }

    private static func normalizedViewPlaneUV(
        width: Int,
        height: Int,
        aspectRatio: Float,
        batch: Int,
        dtype: DType
    ) -> MLXArray {
        let spanX = aspectRatio / sqrt(1 + aspectRatio * aspectRatio)
        let spanY: Float = 1 / sqrt(1 + aspectRatio * aspectRatio)
        var values: [Float] = []
        values.reserveCapacity(width * height * 2)
        for y in 0..<height {
            let v = height == 1
                ? 0
                : -spanY * Float(height - 1) / Float(height) + 2 * spanY * Float(height - 1) / Float(height) * Float(y) / Float(height - 1)
            for x in 0..<width {
                let u = width == 1
                    ? 0
                    : -spanX * Float(width - 1) / Float(width) + 2 * spanX * Float(width - 1) / Float(width) * Float(x) / Float(width - 1)
                values.append(u)
                values.append(v)
            }
        }
        let single = MLXArray(values).reshaped(1, height, width, 2).asType(dtype)
        return MLX.broadcast(single, to: [batch, height, width, 2])
    }
}
