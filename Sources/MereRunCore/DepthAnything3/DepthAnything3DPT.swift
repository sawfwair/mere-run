import Foundation
@preconcurrency import MLX
import MLXNN

private func da3Resize(
    _ input: MLXArray,
    height: Int,
    width: Int,
    alignCorners: Bool
) -> MLXArray {
    if input.dim(1) == height && input.dim(2) == width { return input }
    precondition(height > 0 && width > 0)
    let resized = Upsample(
        scaleFactor: FloatOrArray.array([
            Float(height) / Float(input.dim(1)) + 1e-6,
            Float(width) / Float(input.dim(2)) + 1e-6,
        ]),
        mode: .linear(alignCorners: alignCorners)
    )(input)
    precondition(resized.dim(1) == height && resized.dim(2) == width)
    return resized
}

func depthAnything3UVPositionEmbedding(
    height: Int,
    width: Int,
    channels: Int,
    aspectRatio: Float,
    dtype: DType = .float32
) -> MLXArray {
    precondition(height > 0 && width > 0 && channels.isMultiple(of: 4))
    let diagonal = sqrt(aspectRatio * aspectRatio + 1)
    let spanX = aspectRatio / diagonal
    let spanY: Float = 1 / diagonal
    let left = -spanX * Float(width - 1) / Float(width)
    let right = spanX * Float(width - 1) / Float(width)
    let top = -spanY * Float(height - 1) / Float(height)
    let bottom = spanY * Float(height - 1) / Float(height)

    let halfChannels = channels / 2
    let frequencyCount = halfChannels / 2
    let inverseFrequencies = (0..<frequencyCount).map { index -> Float in
        1 / pow(100, Float(index) / Float(frequencyCount))
    }
    var result = [Float](repeating: 0, count: height * width * channels)
    for y in 0..<height {
        let vertical = height == 1
            ? top
            : top + (bottom - top) * Float(y) / Float(height - 1)
        for x in 0..<width {
            let horizontal = width == 1
                ? left
                : left + (right - left) * Float(x) / Float(width - 1)
            let base = (y * width + x) * channels
            for frequency in 0..<frequencyCount {
                let horizontalAngle = horizontal * inverseFrequencies[frequency]
                let verticalAngle = vertical * inverseFrequencies[frequency]
                result[base + frequency] = sin(horizontalAngle)
                result[base + frequencyCount + frequency] = cos(horizontalAngle)
                result[base + halfChannels + frequency] = sin(verticalAngle)
                result[base + halfChannels + frequencyCount + frequency] = cos(verticalAngle)
            }
        }
    }
    return MLXArray(result).asType(dtype).reshaped(1, height, width, channels)
}

private func da3AddPositionEmbedding(
    _ input: MLXArray,
    imageHeight: Int,
    imageWidth: Int
) -> MLXArray {
    let position = depthAnything3UVPositionEmbedding(
        height: input.dim(1),
        width: input.dim(2),
        channels: input.dim(3),
        aspectRatio: Float(imageWidth) / Float(imageHeight),
        dtype: input.dtype
    )
    return input + position * Float(0.1)
}

private final class DA3ResidualConvUnit: Module {
    @ModuleInfo(key: "conv1") var first: Conv2d
    @ModuleInfo(key: "conv2") var second: Conv2d

    init(channels: Int) {
        self._first.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._second.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(relu(first(relu(input)))) + input
    }
}

private final class DA3FeatureFusionBlock: Module {
    let hasResidual: Bool

    @ModuleInfo(key: "resConfUnit1") var firstResidual: DA3ResidualConvUnit?
    @ModuleInfo(key: "resConfUnit2") var secondResidual: DA3ResidualConvUnit
    @ModuleInfo(key: "out_conv") var output: Conv2d

    init(channels: Int, hasResidual: Bool = true) {
        self.hasResidual = hasResidual
        self._firstResidual.wrappedValue = hasResidual ? DA3ResidualConvUnit(channels: channels) : nil
        self._secondResidual.wrappedValue = DA3ResidualConvUnit(channels: channels)
        self._output.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        residual: MLXArray? = nil,
        targetHeight: Int? = nil,
        targetWidth: Int? = nil
    ) -> MLXArray {
        var hidden = input
        if hasResidual, let residual, let firstResidual {
            hidden = hidden + firstResidual(residual)
        }
        hidden = secondResidual(hidden)
        if let targetHeight, let targetWidth {
            hidden = da3Resize(hidden, height: targetHeight, width: targetWidth, alignCorners: true)
        } else {
            hidden = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: true))(hidden)
        }
        return output(hidden)
    }
}

private final class DA3PrimaryOutputHead: Module {
    @ModuleInfo(key: "first") var first: Conv2d
    @ModuleInfo(key: "second") var second: Conv2d

    init(channels: Int) {
        self._first.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: 32,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._second.wrappedValue = Conv2d(
            inputChannels: 32,
            outputChannels: 2,
            kernelSize: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(relu(first(input)))
    }
}

private final class DA3AuxiliaryPreHead: Module {
    @ModuleInfo(key: "first") var first: Conv2d
    @ModuleInfo(key: "second") var second: Conv2d
    @ModuleInfo(key: "third") var third: Conv2d
    @ModuleInfo(key: "fourth") var fourth: Conv2d
    @ModuleInfo(key: "fifth") var fifth: Conv2d

    init(channels: Int) {
        self._first.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels / 2,
            kernelSize: 3, padding: 1, bias: true
        )
        self._second.wrappedValue = Conv2d(
            inputChannels: channels / 2, outputChannels: channels,
            kernelSize: 3, padding: 1, bias: true
        )
        self._third.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels / 2,
            kernelSize: 3, padding: 1, bias: true
        )
        self._fourth.wrappedValue = Conv2d(
            inputChannels: channels / 2, outputChannels: channels,
            kernelSize: 3, padding: 1, bias: true
        )
        self._fifth.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels / 2,
            kernelSize: 3, padding: 1, bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        fifth(fourth(third(second(first(input)))))
    }
}

private final class DA3AuxiliaryOutputHead: Module {
    @ModuleInfo(key: "first") var first: Conv2d
    /// Upstream reuses one LayerNorm instance across all four Sequential
    /// heads. PyTorch serializes that shared module only at `.0.2`.
    @ModuleInfo(key: "owned_shared_norm") var ownedSharedNorm: LayerNorm?
    @ModuleInfo(key: "second") var second: Conv2d

    init(channels: Int, ownsSharedNorm: Bool) {
        self._first.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: 32,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._ownedSharedNorm.wrappedValue = ownsSharedNorm
            ? LayerNorm(dimensions: 32, eps: 1e-5)
            : nil
        self._second.wrappedValue = Conv2d(
            inputChannels: 32,
            outputChannels: 7,
            kernelSize: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, sharedNorm: LayerNorm) -> MLXArray {
        second(relu(sharedNorm(first(input))))
    }
}

private final class DA3DPTScratch: Module {
    @ModuleInfo(key: "layer1_rn") var layer1: Conv2d
    @ModuleInfo(key: "layer2_rn") var layer2: Conv2d
    @ModuleInfo(key: "layer3_rn") var layer3: Conv2d
    @ModuleInfo(key: "layer4_rn") var layer4: Conv2d
    @ModuleInfo(key: "refinenet1") var refine1: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet2") var refine2: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet3") var refine3: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet4") var refine4: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet1_aux") var auxiliaryRefine1: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet2_aux") var auxiliaryRefine2: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet3_aux") var auxiliaryRefine3: DA3FeatureFusionBlock
    @ModuleInfo(key: "refinenet4_aux") var auxiliaryRefine4: DA3FeatureFusionBlock
    @ModuleInfo(key: "output_conv1") var output1: Conv2d
    @ModuleInfo(key: "output_conv2") var output2: DA3PrimaryOutputHead
    @ModuleInfo(key: "output_conv1_aux") var auxiliaryOutput1: [DA3AuxiliaryPreHead]
    @ModuleInfo(key: "output_conv2_aux") var auxiliaryOutput2: [DA3AuxiliaryOutputHead]

    init(projectedChannels: [Int], featureChannels: Int) {
        self._layer1.wrappedValue = Conv2d(
            inputChannels: projectedChannels[0], outputChannels: featureChannels,
            kernelSize: 3, padding: 1, bias: false
        )
        self._layer2.wrappedValue = Conv2d(
            inputChannels: projectedChannels[1], outputChannels: featureChannels,
            kernelSize: 3, padding: 1, bias: false
        )
        self._layer3.wrappedValue = Conv2d(
            inputChannels: projectedChannels[2], outputChannels: featureChannels,
            kernelSize: 3, padding: 1, bias: false
        )
        self._layer4.wrappedValue = Conv2d(
            inputChannels: projectedChannels[3], outputChannels: featureChannels,
            kernelSize: 3, padding: 1, bias: false
        )
        self._refine1.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels)
        self._refine2.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels)
        self._refine3.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels)
        self._refine4.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels, hasResidual: false)
        self._auxiliaryRefine1.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels)
        self._auxiliaryRefine2.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels)
        self._auxiliaryRefine3.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels)
        self._auxiliaryRefine4.wrappedValue = DA3FeatureFusionBlock(channels: featureChannels, hasResidual: false)
        self._output1.wrappedValue = Conv2d(
            inputChannels: featureChannels,
            outputChannels: featureChannels / 2,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._output2.wrappedValue = DA3PrimaryOutputHead(channels: featureChannels / 2)
        self._auxiliaryOutput1.wrappedValue = (0..<4).map { _ in
            DA3AuxiliaryPreHead(channels: featureChannels)
        }
        self._auxiliaryOutput2.wrappedValue = (0..<4).map {
            DA3AuxiliaryOutputHead(channels: featureChannels / 2, ownsSharedNorm: $0 == 0)
        }
        super.init()
    }

    func fuse(_ pyramid: [MLXArray]) -> (main: MLXArray, auxiliary: MLXArray) {
        precondition(pyramid.count == 4)
        let level1 = layer1(pyramid[0])
        let level2 = layer2(pyramid[1])
        let level3 = layer3(pyramid[2])
        let level4 = layer4(pyramid[3])

        var main = refine4(
            level4,
            targetHeight: level3.dim(1),
            targetWidth: level3.dim(2)
        )
        var auxiliary = auxiliaryRefine4(
            level4,
            targetHeight: level3.dim(1),
            targetWidth: level3.dim(2)
        )
        main = refine3(
            main,
            residual: level3,
            targetHeight: level2.dim(1),
            targetWidth: level2.dim(2)
        )
        auxiliary = auxiliaryRefine3(
            auxiliary,
            residual: level3,
            targetHeight: level2.dim(1),
            targetWidth: level2.dim(2)
        )
        main = refine2(
            main,
            residual: level2,
            targetHeight: level1.dim(1),
            targetWidth: level1.dim(2)
        )
        auxiliary = auxiliaryRefine2(
            auxiliary,
            residual: level2,
            targetHeight: level1.dim(1),
            targetWidth: level1.dim(2)
        )
        main = refine1(main, residual: level1)
        auxiliary = auxiliaryRefine1(auxiliary, residual: level1)
        return (output1(main), auxiliaryOutput1[3](auxiliary))
    }

    func primaryOutput(_ input: MLXArray) -> MLXArray {
        output2(input)
    }

    func auxiliaryOutput(_ input: MLXArray) -> MLXArray {
        guard let norm = auxiliaryOutput2[0].ownedSharedNorm else {
            preconditionFailure("DA3 shared auxiliary LayerNorm is missing")
        }
        return auxiliaryOutput2[3](input, sharedNorm: norm)
    }
}

struct DA3DPTChunkOutput {
    let depth: MLXArray
    let confidence: MLXArray
    let ray: MLXArray
    let rayConfidence: MLXArray
}

final class DepthAnything3DualDPT: Module {
    let configuration: DepthAnything3Configuration

    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "projects") var projects: [Conv2d]
    @ModuleInfo(key: "resize_layers") var resizeLayers: [UnaryLayer]
    @ModuleInfo(key: "scratch") private var scratch: DA3DPTScratch

    init(configuration: DepthAnything3Configuration) {
        self.configuration = configuration
        let inputChannels = configuration.hiddenSize * 2
        self._norm.wrappedValue = LayerNorm(
            dimensions: inputChannels,
            eps: configuration.headLayerNormEpsilon
        )
        self._projects.wrappedValue = configuration.projectedChannels.map {
            Conv2d(
                inputChannels: inputChannels,
                outputChannels: $0,
                kernelSize: 1,
                padding: 0,
                bias: true
            )
        }
        self._resizeLayers.wrappedValue = [
            ConvTransposed2d(
                inputChannels: configuration.projectedChannels[0],
                outputChannels: configuration.projectedChannels[0],
                kernelSize: 4,
                stride: 4,
                padding: 0,
                bias: true
            ),
            ConvTransposed2d(
                inputChannels: configuration.projectedChannels[1],
                outputChannels: configuration.projectedChannels[1],
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: configuration.projectedChannels[3],
                outputChannels: configuration.projectedChannels[3],
                kernelSize: 3,
                stride: 2,
                padding: 1,
                bias: true
            ),
        ]
        self._scratch.wrappedValue = DA3DPTScratch(
            projectedChannels: configuration.projectedChannels,
            featureChannels: configuration.featureChannels
        )
        super.init()
    }

    func callAsFunction(
        features: [MLXArray],
        imageHeight: Int,
        imageWidth: Int,
        batch: Int,
        views: Int
    ) -> DA3DPTChunkOutput {
        precondition(features.count == 4)
        let flattenedCount = batch * views
        let flattened = features.map {
            $0.reshaped(flattenedCount, $0.dim(2), $0.dim(3))
        }
        guard let chunkSize = configuration.headMicroBatchSize,
              flattenedCount > chunkSize
        else {
            let output = forwardChunk(flattened, imageHeight: imageHeight, imageWidth: imageWidth)
            return reshape(output, batch: batch, views: views)
        }

        var chunks: [DA3DPTChunkOutput] = []
        for lower in stride(from: 0, to: flattenedCount, by: chunkSize) {
            let upper = min(lower + chunkSize, flattenedCount)
            let output = forwardChunk(
                flattened.map { $0[lower..<upper, 0..., 0...] },
                imageHeight: imageHeight,
                imageWidth: imageWidth
            )
            MLX.eval(output.depth, output.confidence, output.ray, output.rayConfidence)
            chunks.append(output)
        }
        let combined = DA3DPTChunkOutput(
            depth: MLX.concatenated(chunks.map(\.depth), axis: 0),
            confidence: MLX.concatenated(chunks.map(\.confidence), axis: 0),
            ray: MLX.concatenated(chunks.map(\.ray), axis: 0),
            rayConfidence: MLX.concatenated(chunks.map(\.rayConfidence), axis: 0)
        )
        return reshape(combined, batch: batch, views: views)
    }

    private func forwardChunk(
        _ features: [MLXArray],
        imageHeight: Int,
        imageWidth: Int
    ) -> DA3DPTChunkOutput {
        let count = features[0].dim(0)
        let patchHeight = imageHeight / configuration.patchSize
        let patchWidth = imageWidth / configuration.patchSize
        var pyramid: [MLXArray] = []
        for index in 0..<4 {
            var feature = norm(features[index])
                .reshaped(count, patchHeight, patchWidth, configuration.hiddenSize * 2)
            feature = projects[index](feature)
            feature = da3AddPositionEmbedding(
                feature,
                imageHeight: imageHeight,
                imageWidth: imageWidth
            )
            pyramid.append(resizeLayers[index](feature))
        }

        var fused = scratch.fuse(pyramid)
        fused.main = da3Resize(
            fused.main,
            height: imageHeight,
            width: imageWidth,
            alignCorners: true
        )
        fused.main = da3AddPositionEmbedding(
            fused.main,
            imageHeight: imageHeight,
            imageWidth: imageWidth
        )
        let mainLogits = scratch.primaryOutput(fused.main)
        let depth = MLX.exp(mainLogits[0..., 0..., 0..., 0])
        let confidence = MLX.exp(mainLogits[0..., 0..., 0..., 1]) + 1

        fused.auxiliary = da3AddPositionEmbedding(
            fused.auxiliary,
            imageHeight: imageHeight,
            imageWidth: imageWidth
        )
        let auxiliaryLogits = scratch.auxiliaryOutput(fused.auxiliary)
        let ray = auxiliaryLogits[0..., 0..., 0..., 0..<6]
        let rayConfidence = MLX.exp(auxiliaryLogits[0..., 0..., 0..., 6]) + 1
        return DA3DPTChunkOutput(
            depth: depth,
            confidence: confidence,
            ray: ray,
            rayConfidence: rayConfidence
        )
    }

    private func reshape(
        _ output: DA3DPTChunkOutput,
        batch: Int,
        views: Int
    ) -> DA3DPTChunkOutput {
        DA3DPTChunkOutput(
            depth: output.depth.reshaped(batch, views, output.depth.dim(1), output.depth.dim(2)),
            confidence: output.confidence.reshaped(batch, views, output.confidence.dim(1), output.confidence.dim(2)),
            ray: output.ray.reshaped(batch, views, output.ray.dim(1), output.ray.dim(2), 6),
            rayConfidence: output.rayConfidence.reshaped(
                batch,
                views,
                output.rayConfidence.dim(1),
                output.rayConfidence.dim(2)
            )
        )
    }
}
