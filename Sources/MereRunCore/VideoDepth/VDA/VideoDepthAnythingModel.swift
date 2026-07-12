import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// Exact graph configuration for the Apache-2.0 Video Depth Anything Small
/// checkpoints published by Depth Anything.
public struct VideoDepthAnythingConfiguration: Equatable, Sendable {
    public let backbone: DINOv2Configuration
    public let intermediateLayers: [Int]
    public let featureChannels: Int
    public let projectedChannels: [Int]
    public let temporalFrameCount: Int
    public let temporalHeadCount: Int
    public let temporalTransformerBlockCount: Int
    public let temporalAttentionBlockCount: Int
    public let temporalGroupCount: Int

    public init(
        backbone: DINOv2Configuration = .smallPatch14,
        intermediateLayers: [Int] = [2, 5, 8, 11],
        featureChannels: Int = 64,
        projectedChannels: [Int] = [48, 96, 192, 384],
        temporalFrameCount: Int = 32,
        temporalHeadCount: Int = 8,
        temporalTransformerBlockCount: Int = 1,
        temporalAttentionBlockCount: Int = 2,
        temporalGroupCount: Int = 32
    ) {
        precondition(intermediateLayers.count == 4)
        precondition(Set(intermediateLayers).count == 4)
        precondition(intermediateLayers.allSatisfy { $0 >= 0 && $0 < backbone.layerCount })
        precondition(projectedChannels.count == 4)
        precondition(featureChannels > 0 && projectedChannels.allSatisfy { $0 > 0 })
        precondition(temporalFrameCount > 0)
        precondition(temporalHeadCount > 0)
        precondition(temporalTransformerBlockCount > 0)
        precondition(temporalAttentionBlockCount > 0)
        precondition(temporalGroupCount > 0)
        let temporalChannels = [projectedChannels[2], projectedChannels[3], featureChannels, featureChannels]
        precondition(temporalChannels.allSatisfy { $0.isMultiple(of: temporalHeadCount) })
        precondition(temporalChannels.allSatisfy { $0.isMultiple(of: temporalGroupCount) })
        self.backbone = backbone
        self.intermediateLayers = intermediateLayers
        self.featureChannels = featureChannels
        self.projectedChannels = projectedChannels
        self.temporalFrameCount = temporalFrameCount
        self.temporalHeadCount = temporalHeadCount
        self.temporalTransformerBlockCount = temporalTransformerBlockCount
        self.temporalAttentionBlockCount = temporalAttentionBlockCount
        self.temporalGroupCount = temporalGroupCount
    }

    public static let small = VideoDepthAnythingConfiguration()
}

/// Execution-only memory controls. These values do not change the graph or
/// checkpoint layout and can be overridden per generation.
public struct VideoDepthAnythingMemoryConfiguration: Equatable, Sendable {
    /// Maximum flattened video frames sent through DINOv2 at once. `nil`
    /// retains full-batch execution.
    public let encoderMicroBatchSize: Int?
    /// Maximum flattened frames used by the high-resolution DPT tail. As in
    /// upstream VDA, chunking is used only when the frame batch is divisible.
    public let dptTailMicroBatchSize: Int?

    public init(
        encoderMicroBatchSize: Int? = 4,
        dptTailMicroBatchSize: Int? = 4
    ) {
        precondition(encoderMicroBatchSize.map { $0 > 0 } ?? true)
        precondition(dptTailMicroBatchSize.map { $0 > 0 } ?? true)
        self.encoderMicroBatchSize = encoderMicroBatchSize
        self.dptTailMicroBatchSize = dptTailMicroBatchSize
    }

    /// Conservative defaults for unified-memory Apple Silicon systems.
    public static let appleSilicon = VideoDepthAnythingMemoryConfiguration()
    /// Useful for parity checks and hosts with enough memory for all frames.
    public static let fullBatch = VideoDepthAnythingMemoryConfiguration(
        encoderMicroBatchSize: nil,
        dptTailMicroBatchSize: nil
    )
}

public struct VideoDepthAnythingRawOutput {
    /// `[batch, frames, height, width]`. Relative and metric checkpoints share
    /// the graph; output semantics are selected by the weights, not this type.
    public let depth: MLXArray
}

private func vdaResize(
    _ input: MLXArray,
    height: Int,
    width: Int,
    alignCorners: Bool
) -> MLXArray {
    if input.dim(1) == height && input.dim(2) == width {
        return input
    }
    precondition(height > 1 && width > 1)
    let resized = Upsample(
        scaleFactor: .array([
            Float(height) / Float(input.dim(1)) + 1e-6,
            Float(width) / Float(input.dim(2)) + 1e-6,
        ]),
        mode: .linear(alignCorners: alignCorners)
    )(input)
    precondition(resized.dim(1) == height && resized.dim(2) == width)
    return resized
}

private final class VDAResidualConvUnit: Module {
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

private final class VDAFeatureFusionBlock: Module {
    @ModuleInfo(key: "out_conv") var output: Conv2d
    @ModuleInfo(key: "resConfUnit1") var firstResidual: VDAResidualConvUnit
    @ModuleInfo(key: "resConfUnit2") var secondResidual: VDAResidualConvUnit

    init(channels: Int) {
        self._output.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1,
            padding: 0,
            bias: true
        )
        self._firstResidual.wrappedValue = VDAResidualConvUnit(channels: channels)
        self._secondResidual.wrappedValue = VDAResidualConvUnit(channels: channels)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        residual: MLXArray? = nil,
        targetHeight: Int? = nil,
        targetWidth: Int? = nil
    ) -> MLXArray {
        var hidden = input
        if let residual {
            hidden = hidden + firstResidual(residual)
        }
        hidden = secondResidual(hidden)
        if let targetHeight, let targetWidth {
            hidden = vdaResize(hidden, height: targetHeight, width: targetWidth, alignCorners: true)
        } else {
            hidden = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: true))(hidden)
        }
        return output(hidden)
    }
}

private final class VDATemporalPositionEncoding: Module {
    @ParameterInfo(key: "pe") var position: MLXArray

    init(channels: Int, maximumFrames: Int) {
        precondition(channels.isMultiple(of: 2))
        var values = [Float](repeating: 0, count: maximumFrames * channels)
        for frame in 0..<maximumFrames {
            for channel in stride(from: 0, to: channels, by: 2) {
                let exponent = Float(channel) * (-log(10_000) / Float(channels))
                let angle = Float(frame) * exp(exponent)
                values[frame * channels + channel] = sin(angle)
                values[frame * channels + channel + 1] = cos(angle)
            }
        }
        self._position.wrappedValue = MLXArray(values).reshaped(1, maximumFrames, channels)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dim(1) <= position.dim(1))
        return input + position[0..., 0..<input.dim(1), 0...].asType(input.dtype)
    }
}

private final class VDATemporalAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "to_out") var output: [UnaryLayer]
    @ModuleInfo(key: "pos_encoder") var position: VDATemporalPositionEncoding

    init(channels: Int, headCount: Int, maximumFrames: Int) {
        precondition(channels.isMultiple(of: headCount))
        self.headCount = headCount
        self.headDimension = channels / headCount
        self.scale = 1 / sqrt(Float(headDimension))
        self._query.wrappedValue = Linear(channels, channels, bias: false)
        self._key.wrappedValue = Linear(channels, channels, bias: false)
        self._value.wrappedValue = Linear(channels, channels, bias: false)
        self._output.wrappedValue = [
            Linear(channels, channels, bias: true),
            Identity(),
        ]
        self._position.wrappedValue = VDATemporalPositionEncoding(
            channels: channels,
            maximumFrames: maximumFrames
        )
        super.init()
    }

    /// Input is `[batch * spatial, frames, channels]`.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = position(input)
        let batch = hidden.dim(0)
        let frames = hidden.dim(1)
        let channels = hidden.dim(2)
        let queries = query(hidden)
            .reshaped(batch, frames, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let keys = key(hidden)
            .reshaped(batch, frames, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = value(hidden)
            .reshaped(batch, frames, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        return output.reduce(
            attended.transposed(0, 2, 1, 3).reshaped(batch, frames, channels)
        ) { hidden, layer in
            layer(hidden)
        }
    }
}

private final class VDAGEGLU: Module, UnaryLayer {
    let innerChannels: Int
    @ModuleInfo(key: "proj") var projection: Linear

    init(channels: Int) {
        self.innerChannels = channels * 4
        self._projection.wrappedValue = Linear(channels, channels * 8, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let projected = projection(input)
        let hidden = projected[0..., 0..., 0..<innerChannels]
        let gate = projected[0..., 0..., innerChannels...]
        return hidden * gelu(gate)
    }
}

private final class VDAFeedForward: Module {
    @ModuleInfo(key: "net") var network: [UnaryLayer]

    init(channels: Int) {
        self._network.wrappedValue = [
            VDAGEGLU(channels: channels),
            Identity(),
            Linear(channels * 4, channels, bias: true),
        ]
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        network.reduce(input) { hidden, layer in layer(hidden) }
    }
}

private final class VDATemporalTransformerBlock: Module {
    @ModuleInfo(key: "attention_blocks") var attentions: [VDATemporalAttention]
    @ModuleInfo(key: "norms") var norms: [LayerNorm]
    @ModuleInfo(key: "ff") var feedForward: VDAFeedForward
    @ModuleInfo(key: "ff_norm") var feedForwardNorm: LayerNorm

    init(
        channels: Int,
        headCount: Int,
        attentionBlockCount: Int,
        maximumFrames: Int
    ) {
        self._attentions.wrappedValue = (0..<attentionBlockCount).map { _ in
            VDATemporalAttention(channels: channels, headCount: headCount, maximumFrames: maximumFrames)
        }
        self._norms.wrappedValue = (0..<attentionBlockCount).map { _ in
            LayerNorm(dimensions: channels)
        }
        self._feedForward.wrappedValue = VDAFeedForward(channels: channels)
        self._feedForwardNorm.wrappedValue = LayerNorm(dimensions: channels)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input
        for index in attentions.indices {
            hidden = hidden + attentions[index](norms[index](hidden))
        }
        return hidden + feedForward(feedForwardNorm(hidden))
    }
}

private final class VDATemporalTransformer: Module {
    let channels: Int

    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var inputProjection: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [VDATemporalTransformerBlock]
    @ModuleInfo(key: "proj_out") var outputProjection: Linear

    init(channels: Int, configuration: VideoDepthAnythingConfiguration) {
        self.channels = channels
        self._norm.wrappedValue = GroupNorm(
            groupCount: configuration.temporalGroupCount,
            dimensions: channels,
            eps: 1e-6,
            affine: true,
            pytorchCompatible: true
        )
        self._inputProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._blocks.wrappedValue = (0..<configuration.temporalTransformerBlockCount).map { _ in
            VDATemporalTransformerBlock(
                channels: channels,
                headCount: configuration.temporalHeadCount,
                attentionBlockCount: configuration.temporalAttentionBlockCount,
                maximumFrames: configuration.temporalFrameCount
            )
        }
        self._outputProjection.wrappedValue = Linear(channels, channels, bias: true)
        super.init()
    }

    /// Input and output are `[batch * frames, height, width, channels]`.
    func callAsFunction(_ input: MLXArray, batch: Int, frames: Int) -> MLXArray {
        precondition(input.dim(0) == batch * frames && input.dim(3) == channels)
        let height = input.dim(1)
        let width = input.dim(2)
        let spatial = height * width
        var hidden = inputProjection(norm(input))
            .reshaped(batch, frames, spatial, channels)
            .transposed(0, 2, 1, 3)
            .reshaped(batch * spatial, frames, channels)
        for block in blocks {
            hidden = block(hidden)
        }
        hidden = hidden
            .reshaped(batch, spatial, frames, channels)
            .transposed(0, 2, 1, 3)
            .reshaped(batch * frames, height, width, channels)
        return outputProjection(hidden) + input
    }
}

private final class VDATemporalModule: Module {
    @ModuleInfo(key: "temporal_transformer") var transformer: VDATemporalTransformer

    init(channels: Int, configuration: VideoDepthAnythingConfiguration) {
        self._transformer.wrappedValue = VDATemporalTransformer(
            channels: channels,
            configuration: configuration
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, batch: Int, frames: Int) -> MLXArray {
        transformer(input, batch: batch, frames: frames)
    }
}

private final class VDAScratch: Module {
    @ModuleInfo(key: "layer1_rn") var layer1: Conv2d
    @ModuleInfo(key: "layer2_rn") var layer2: Conv2d
    @ModuleInfo(key: "layer3_rn") var layer3: Conv2d
    @ModuleInfo(key: "layer4_rn") var layer4: Conv2d
    @ModuleInfo(key: "refinenet1") var refine1: VDAFeatureFusionBlock
    @ModuleInfo(key: "refinenet2") var refine2: VDAFeatureFusionBlock
    @ModuleInfo(key: "refinenet3") var refine3: VDAFeatureFusionBlock
    @ModuleInfo(key: "refinenet4") var refine4: VDAFeatureFusionBlock
    @ModuleInfo(key: "output_conv1") var output1: Conv2d
    @ModuleInfo(key: "output_conv2") var output2: [UnaryLayer]

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
        self._refine1.wrappedValue = VDAFeatureFusionBlock(channels: featureChannels)
        self._refine2.wrappedValue = VDAFeatureFusionBlock(channels: featureChannels)
        self._refine3.wrappedValue = VDAFeatureFusionBlock(channels: featureChannels)
        self._refine4.wrappedValue = VDAFeatureFusionBlock(channels: featureChannels)
        self._output1.wrappedValue = Conv2d(
            inputChannels: featureChannels,
            outputChannels: featureChannels / 2,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._output2.wrappedValue = [
            Conv2d(
                inputChannels: featureChannels / 2,
                outputChannels: 32,
                kernelSize: 3,
                padding: 1,
                bias: true
            ),
            ReLU(),
            Conv2d(
                inputChannels: 32,
                outputChannels: 1,
                kernelSize: 1,
                padding: 0,
                bias: true
            ),
            ReLU(),
            Identity(),
        ]
        super.init()
    }
}

private final class VDADPTTemporalHead: Module {
    let configuration: VideoDepthAnythingConfiguration

    @ModuleInfo(key: "projects") var projects: [Conv2d]
    @ModuleInfo(key: "resize_layers") var resizeLayers: [UnaryLayer]
    @ModuleInfo(key: "scratch") var scratch: VDAScratch
    @ModuleInfo(key: "motion_modules") var motionModules: [VDATemporalModule]

    init(configuration: VideoDepthAnythingConfiguration) {
        self.configuration = configuration
        self._projects.wrappedValue = configuration.projectedChannels.map { channels in
            Conv2d(
                inputChannels: configuration.backbone.hiddenSize,
                outputChannels: channels,
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
        self._scratch.wrappedValue = VDAScratch(
            projectedChannels: configuration.projectedChannels,
            featureChannels: configuration.featureChannels
        )
        let temporalChannels = [
            configuration.projectedChannels[2],
            configuration.projectedChannels[3],
            configuration.featureChannels,
            configuration.featureChannels,
        ]
        self._motionModules.wrappedValue = temporalChannels.map { channels in
            VDATemporalModule(channels: channels, configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(
        featuresByLayer: [Int: MLXArray],
        patchHeight: Int,
        patchWidth: Int,
        batch: Int,
        frames: Int,
        dptTailMicroBatchSize: Int?
    ) -> MLXArray {
        var pyramid: [MLXArray] = []
        pyramid.reserveCapacity(4)
        for index in 0..<4 {
            guard let feature = featuresByLayer[configuration.intermediateLayers[index]] else {
                preconditionFailure("Missing requested DINOv2 intermediate layer")
            }
            pyramid.append(resizeLayers[index](projects[index](feature)))
        }

        var layer1 = pyramid[0]
        var layer2 = pyramid[1]
        var layer3 = motionModules[0](pyramid[2], batch: batch, frames: frames)
        var layer4 = motionModules[1](pyramid[3], batch: batch, frames: frames)

        layer1 = scratch.layer1(layer1)
        layer2 = scratch.layer2(layer2)
        layer3 = scratch.layer3(layer3)
        layer4 = scratch.layer4(layer4)

        var path4 = scratch.refine4(
            layer4,
            targetHeight: layer3.dim(1),
            targetWidth: layer3.dim(2)
        )
        path4 = motionModules[2](path4, batch: batch, frames: frames)
        var path3 = scratch.refine3(
            path4,
            residual: layer3,
            targetHeight: layer2.dim(1),
            targetWidth: layer2.dim(2)
        )
        path3 = motionModules[3](path3, batch: batch, frames: frames)

        let flattenedFrameCount = batch * frames
        if let microBatchSize = dptTailMicroBatchSize,
           flattenedFrameCount > microBatchSize,
           flattenedFrameCount.isMultiple(of: microBatchSize)
        {
            // MLX is lazy. Materialize the shared temporal trunk once, then
            // each high-resolution tail chunk, so chunking actually bounds
            // peak live intermediates instead of constructing one large graph.
            MLX.eval(path3, layer2, layer1)
            var chunks: [MLXArray] = []
            chunks.reserveCapacity(flattenedFrameCount / microBatchSize)
            for lowerBound in stride(from: 0, to: flattenedFrameCount, by: microBatchSize) {
                let upperBound = lowerBound + microBatchSize
                let output = renderTail(
                    path3: path3[lowerBound..<upperBound, 0..., 0..., 0...],
                    layer2: layer2[lowerBound..<upperBound, 0..., 0..., 0...],
                    layer1: layer1[lowerBound..<upperBound, 0..., 0..., 0...],
                    patchHeight: patchHeight,
                    patchWidth: patchWidth
                )
                MLX.eval(output)
                chunks.append(output)
            }
            return MLX.concatenated(chunks, axis: 0)
        }
        return renderTail(
            path3: path3,
            layer2: layer2,
            layer1: layer1,
            patchHeight: patchHeight,
            patchWidth: patchWidth
        )
    }

    private func renderTail(
        path3: MLXArray,
        layer2: MLXArray,
        layer1: MLXArray,
        patchHeight: Int,
        patchWidth: Int
    ) -> MLXArray {
        let path2 = scratch.refine2(
            path3,
            residual: layer2,
            targetHeight: layer1.dim(1),
            targetWidth: layer1.dim(2)
        )
        let path1 = scratch.refine1(path2, residual: layer1)

        var output = scratch.output1(path1)
        output = vdaResize(
            output,
            height: patchHeight * configuration.backbone.patchSize,
            width: patchWidth * configuration.backbone.patchSize,
            alignCorners: true
        )
        return scratch.output2.reduce(output) { hidden, layer in layer(hidden) }
    }
}

/// Native MLX graph shared by the relative and metric VDA-S checkpoints.
///
/// Input is `[batch, frames, height, width, 3]` NHWTC RGB after upstream
/// resize and ImageNet normalization. Long-video windowing and affine
/// cross-window alignment intentionally live outside this graph.
private struct VDAEncodedFeatures {
    let featuresByLayer: [Int: MLXArray]
    let gridHeight: Int
    let gridWidth: Int
}

public final class VideoDepthAnythingModel: Module {
    public let configuration: VideoDepthAnythingConfiguration
    public let defaultMemoryConfiguration: VideoDepthAnythingMemoryConfiguration

    @ModuleInfo(key: "pretrained") var pretrained: DINOv2VisionTransformer
    @ModuleInfo(key: "head") fileprivate var head: VDADPTTemporalHead

    public init(
        configuration: VideoDepthAnythingConfiguration = .small,
        memoryConfiguration: VideoDepthAnythingMemoryConfiguration = .appleSilicon
    ) {
        self.configuration = configuration
        self.defaultMemoryConfiguration = memoryConfiguration
        self._pretrained.wrappedValue = DINOv2VisionTransformer(configuration: configuration.backbone)
        self._head.wrappedValue = VDADPTTemporalHead(configuration: configuration)
        super.init()
    }

    public func callAsFunction(
        _ normalizedVideo: MLXArray,
        memoryConfiguration: VideoDepthAnythingMemoryConfiguration? = nil
    ) -> VideoDepthAnythingRawOutput {
        precondition(normalizedVideo.ndim == 5 && normalizedVideo.dim(4) == 3)
        let batch = normalizedVideo.dim(0)
        let frames = normalizedVideo.dim(1)
        let height = normalizedVideo.dim(2)
        let width = normalizedVideo.dim(3)
        precondition(frames > 0 && frames <= configuration.temporalFrameCount)
        precondition(height.isMultiple(of: configuration.backbone.patchSize))
        precondition(width.isMultiple(of: configuration.backbone.patchSize))

        let memoryConfiguration = memoryConfiguration ?? defaultMemoryConfiguration
        let flattened = normalizedVideo.reshaped(batch * frames, height, width, 3)
        let encoded = encode(
            flattened,
            microBatchSize: memoryConfiguration.encoderMicroBatchSize
        )
        var depth = head(
            featuresByLayer: encoded.featuresByLayer,
            patchHeight: encoded.gridHeight,
            patchWidth: encoded.gridWidth,
            batch: batch,
            frames: frames,
            dptTailMicroBatchSize: memoryConfiguration.dptTailMicroBatchSize
        )
        depth = vdaResize(depth, height: height, width: width, alignCorners: true)
        depth = relu(depth).squeezed(axis: -1).reshaped(batch, frames, height, width)
        return VideoDepthAnythingRawOutput(depth: depth)
    }

    private func encode(
        _ flattenedVideo: MLXArray,
        microBatchSize: Int?
    ) -> VDAEncodedFeatures {
        let requestedLayers = Set(configuration.intermediateLayers)
        let flattenedFrameCount = flattenedVideo.dim(0)
        guard let microBatchSize, flattenedFrameCount > microBatchSize else {
            let encoded = pretrained(flattenedVideo, intermediateLayers: requestedLayers)
            return VDAEncodedFeatures(
                featuresByLayer: encoded.featuresByLayer,
                gridHeight: encoded.gridHeight,
                gridWidth: encoded.gridWidth
            )
        }

        var chunksByLayer = Dictionary(
            uniqueKeysWithValues: configuration.intermediateLayers.map { ($0, [MLXArray]()) }
        )
        var gridHeight: Int?
        var gridWidth: Int?
        for lowerBound in stride(from: 0, to: flattenedFrameCount, by: microBatchSize) {
            let upperBound = min(lowerBound + microBatchSize, flattenedFrameCount)
            let encoded = pretrained(
                flattenedVideo[lowerBound..<upperBound, 0..., 0..., 0...],
                intermediateLayers: requestedLayers
            )
            let chunkFeatures = configuration.intermediateLayers.map { layer -> MLXArray in
                guard let feature = encoded.featuresByLayer[layer] else {
                    preconditionFailure("Missing requested DINOv2 intermediate layer")
                }
                return feature
            }
            // Force one encoder chunk at a time; otherwise MLX would retain a
            // lazy graph spanning every nominal micro-batch.
            MLX.eval(chunkFeatures)
            for (layer, feature) in zip(configuration.intermediateLayers, chunkFeatures) {
                chunksByLayer[layer]!.append(feature)
            }
            if let gridHeight, let gridWidth {
                precondition(encoded.gridHeight == gridHeight && encoded.gridWidth == gridWidth)
            } else {
                gridHeight = encoded.gridHeight
                gridWidth = encoded.gridWidth
            }
        }

        var combined: [Int: MLXArray] = [:]
        for layer in configuration.intermediateLayers {
            guard let chunks = chunksByLayer.removeValue(forKey: layer) else {
                preconditionFailure("Missing requested DINOv2 intermediate layer")
            }
            let feature = MLX.concatenated(chunks, axis: 0)
            MLX.eval(feature)
            combined[layer] = feature
        }
        guard let gridHeight, let gridWidth else {
            preconditionFailure("VDA encoder received no frames")
        }
        return VDAEncodedFeatures(
            featuresByLayer: combined,
            gridHeight: gridHeight,
            gridWidth: gridWidth
        )
    }
}
