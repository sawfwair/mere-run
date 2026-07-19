import Foundation
import MLX
import MLXFast
import MLXNN

public enum SCAIL2Mode: String, Codable, CaseIterable, Hashable, Sendable {
    case animation
    case replacement
}

public struct SCAIL2TransformerConfiguration: Hashable, Sendable {
    public let patchSize: [Int]
    public let textLength: Int
    public let inputChannels: Int
    public let maskChannels: Int
    public let hiddenSize: Int
    public let feedForwardSize: Int
    public let timestepFrequencySize: Int
    public let textEmbeddingSize: Int
    public let imageEmbeddingSize: Int
    public let outputChannels: Int
    public let headCount: Int
    public let layerCount: Int
    public let epsilon: Float
    public let ropeTableLength: Int
    public let poseWidthShift: Int
    public let replacementReferenceHeightShift: Int

    public init(
        patchSize: [Int] = [1, 2, 2],
        textLength: Int = 512,
        inputChannels: Int = 20,
        maskChannels: Int = 28,
        hiddenSize: Int = 5_120,
        feedForwardSize: Int = 13_824,
        timestepFrequencySize: Int = 256,
        textEmbeddingSize: Int = 4_096,
        imageEmbeddingSize: Int = 1_280,
        outputChannels: Int = 16,
        headCount: Int = 40,
        layerCount: Int = 40,
        epsilon: Float = 1e-6,
        ropeTableLength: Int = 8_192,
        poseWidthShift: Int = 120,
        replacementReferenceHeightShift: Int = 120
    ) {
        precondition(patchSize.count == 3)
        precondition(hiddenSize.isMultiple(of: headCount))
        precondition((hiddenSize / headCount).isMultiple(of: 2))
        precondition(ropeTableLength > poseWidthShift)
        precondition(ropeTableLength > replacementReferenceHeightShift)
        self.patchSize = patchSize
        self.textLength = textLength
        self.inputChannels = inputChannels
        self.maskChannels = maskChannels
        self.hiddenSize = hiddenSize
        self.feedForwardSize = feedForwardSize
        self.timestepFrequencySize = timestepFrequencySize
        self.textEmbeddingSize = textEmbeddingSize
        self.imageEmbeddingSize = imageEmbeddingSize
        self.outputChannels = outputChannels
        self.headCount = headCount
        self.layerCount = layerCount
        self.epsilon = epsilon
        self.ropeTableLength = ropeTableLength
        self.poseWidthShift = poseWidthShift
        self.replacementReferenceHeightShift = replacementReferenceHeightShift
    }

    public init(_ configuration: SCAIL2Configuration) {
        self.init(
            patchSize: configuration.patchSize,
            textLength: configuration.textLength,
            inputChannels: configuration.inputChannels,
            maskChannels: configuration.maskChannels,
            hiddenSize: configuration.hiddenSize,
            feedForwardSize: configuration.feedForwardSize,
            timestepFrequencySize: configuration.frequencySize,
            textEmbeddingSize: configuration.textEmbeddingSize,
            outputChannels: configuration.outputChannels,
            headCount: configuration.headCount,
            layerCount: configuration.layerCount,
            epsilon: configuration.epsilon
        )
    }
}

public struct SCAIL2TransformerInput {
    public let videoLatent: MLXArray
    public let referenceLatent: MLXArray
    public let referenceMask: MLXArray
    public let drivingLatent: MLXArray
    public let drivingMask: MLXArray
    public let historyMask: MLXArray?
    public let additionalReferenceLatents: [MLXArray]
    public let additionalReferenceMasks: [MLXArray]
    public let textEmbeddings: MLXArray
    public let imageEmbeddings: MLXArray
    public let timestep: MLXArray
    public let mode: SCAIL2Mode

    public init(
        videoLatent: MLXArray,
        referenceLatent: MLXArray,
        referenceMask: MLXArray,
        drivingLatent: MLXArray,
        drivingMask: MLXArray,
        historyMask: MLXArray? = nil,
        additionalReferenceLatents: [MLXArray] = [],
        additionalReferenceMasks: [MLXArray] = [],
        textEmbeddings: MLXArray,
        imageEmbeddings: MLXArray,
        timestep: MLXArray,
        mode: SCAIL2Mode
    ) {
        precondition(additionalReferenceLatents.count == additionalReferenceMasks.count)
        self.videoLatent = videoLatent
        self.referenceLatent = referenceLatent
        self.referenceMask = referenceMask
        self.drivingLatent = drivingLatent
        self.drivingMask = drivingMask
        self.historyMask = historyMask
        self.additionalReferenceLatents = additionalReferenceLatents
        self.additionalReferenceMasks = additionalReferenceMasks
        self.textEmbeddings = textEmbeddings
        self.imageEmbeddings = imageEmbeddings
        self.timestep = timestep
        self.mode = mode
    }
}

struct SCAIL2TokenLayout: Hashable {
    let additionalReferenceGrid: Wan2GridSize?
    let referenceGrid: Wan2GridSize
    let videoGrid: Wan2GridSize
    let drivingGrid: Wan2GridSize

    var additionalReferenceLength: Int { additionalReferenceGrid?.sequenceLength ?? 0 }
    var referenceLength: Int { referenceGrid.sequenceLength }
    var videoLength: Int { videoGrid.sequenceLength }
    var drivingLength: Int { drivingGrid.sequenceLength }
    var totalLength: Int {
        additionalReferenceLength + referenceLength + videoLength + drivingLength
    }
    var videoOffset: Int { additionalReferenceLength + referenceLength }
}

enum SCAIL2RoPE {
    static func prepare(
        layout: SCAIL2TokenLayout,
        frequencies: MLXArray,
        mode: SCAIL2Mode,
        poseWidthShift: Int,
        replacementReferenceHeightShift: Int
    ) -> Wan2RoPE.Cache {
        let additionalCount = layout.additionalReferenceGrid?.frames ?? 0
        let videoTemporalShift = mode == .replacement ? additionalCount : additionalCount + 1
        let referenceHeightShift = mode == .replacement ? replacementReferenceHeightShift : 0
        var cosine: [MLXArray] = []
        var sine: [MLXArray] = []

        if let additionalGrid = layout.additionalReferenceGrid {
            let cache = component(
                grid: additionalGrid,
                frequencies: frequencies,
                shifts: (0, referenceHeightShift, 0),
                downsampleSpatially: false
            )
            cosine.append(cache.cosine)
            sine.append(cache.sine)
        }
        let reference = component(
            grid: layout.referenceGrid,
            frequencies: frequencies,
            shifts: (additionalCount, referenceHeightShift, 0),
            downsampleSpatially: false
        )
        let video = component(
            grid: layout.videoGrid,
            frequencies: frequencies,
            shifts: (videoTemporalShift, 0, 0),
            downsampleSpatially: false
        )
        let driving = component(
            grid: layout.videoGrid,
            frequencies: frequencies,
            shifts: (videoTemporalShift, 0, poseWidthShift),
            downsampleSpatially: true
        )
        cosine.append(contentsOf: [reference.cosine, video.cosine, driving.cosine])
        sine.append(contentsOf: [reference.sine, video.sine, driving.sine])
        return Wan2RoPE.Cache(
            cosine: MLX.concatenated(cosine, axis: 0),
            sine: MLX.concatenated(sine, axis: 0)
        )
    }

    static func apply(_ input: MLXArray, cache: Wan2RoPE.Cache) -> MLXArray {
        precondition(input.ndim == 4)
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let heads = input.dim(2)
        let dimensions = input.dim(3)
        precondition(cache.cosine.dim(0) == sequence)
        let paired = input.asType(.float32).reshaped(batch, sequence, heads, dimensions / 2, 2)
        let real = paired[0..., 0..., 0..., 0..., 0]
        let imaginary = paired[0..., 0..., 0..., 0..., 1]
        let cosine = cache.cosine.expandedDimensions(axis: 0)
        let sine = cache.sine.expandedDimensions(axis: 0)
        return MLX.stacked(
            [real * cosine - imaginary * sine, real * sine + imaginary * cosine],
            axis: -1
        ).reshaped(batch, sequence, heads, dimensions)
    }

    private static func component(
        grid: Wan2GridSize,
        frequencies: MLXArray,
        shifts: (temporal: Int, height: Int, width: Int),
        downsampleSpatially: Bool
    ) -> Wan2RoPE.Cache {
        let typed = frequencies.asType(.float32)
        let halfDimensions = typed.dim(1)
        let temporalDimensions = halfDimensions - 2 * (halfDimensions / 3)
        let heightDimensions = halfDimensions / 3
        let widthDimensions = halfDimensions / 3
        precondition(shifts.temporal + grid.frames <= typed.dim(0))
        precondition(shifts.height + grid.height <= typed.dim(0))
        precondition(shifts.width + grid.width <= typed.dim(0))

        let temporal = MLX.broadcast(
            typed[
                shifts.temporal..<(shifts.temporal + grid.frames),
                0..<temporalDimensions
            ].reshaped(grid.frames, 1, 1, temporalDimensions, 2),
            to: [grid.frames, grid.height, grid.width, temporalDimensions, 2]
        )
        let height = MLX.broadcast(
            typed[
                shifts.height..<(shifts.height + grid.height),
                temporalDimensions..<(temporalDimensions + heightDimensions)
            ].reshaped(1, grid.height, 1, heightDimensions, 2),
            to: [grid.frames, grid.height, grid.width, heightDimensions, 2]
        )
        let width = MLX.broadcast(
            typed[
                shifts.width..<(shifts.width + grid.width),
                (temporalDimensions + heightDimensions)..<halfDimensions
            ].reshaped(1, 1, grid.width, widthDimensions, 2),
            to: [grid.frames, grid.height, grid.width, widthDimensions, 2]
        )
        var combined = MLX.concatenated([temporal, height, width], axis: 3)
        var outputGrid = grid
        if downsampleSpatially {
            precondition(grid.height.isMultiple(of: 2) && grid.width.isMultiple(of: 2))
            combined = MLX.mean(
                combined.reshaped(
                    grid.frames,
                    grid.height / 2, 2,
                    grid.width / 2, 2,
                    halfDimensions, 2
                ),
                axes: [2, 4]
            )
            outputGrid = Wan2GridSize(
                frames: grid.frames,
                height: grid.height / 2,
                width: grid.width / 2
            )
        }
        let flattened = combined.reshaped(outputGrid.sequenceLength, 1, halfDimensions, 2)
        return Wan2RoPE.Cache(
            cosine: flattened[0..., 0..., 0..., 0],
            sine: flattened[0..., 0..., 0..., 1]
        )
    }
}

final class SCAIL2SelfAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float
    @ModuleInfo(key: "q") var query: Linear
    @ModuleInfo(key: "k") var key: Linear
    @ModuleInfo(key: "v") var value: Linear
    @ModuleInfo(key: "o") var output: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: Wan2RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: Wan2RMSNorm

    init(configuration: SCAIL2TransformerConfiguration) {
        self.heads = configuration.headCount
        self.headDimension = configuration.hiddenSize / configuration.headCount
        self.scale = 1 / Float(headDimension).squareRoot()
        self._query.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._key.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._value.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._output.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._queryNorm.wrappedValue = Wan2RMSNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
        self._keyNorm.wrappedValue = Wan2RMSNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
    }

    func callAsFunction(_ input: MLXArray, rope: Wan2RoPE.Cache) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let typed = input.asType(query.weight.dtype)
        var q = queryNorm(query(typed)).reshaped(batch, sequence, heads, headDimension)
        var k = keyNorm(key(typed)).reshaped(batch, sequence, heads, headDimension)
        let v = value(typed).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        q = SCAIL2RoPE.apply(q, cache: rope).transposed(0, 2, 1, 3)
        k = SCAIL2RoPE.apply(k, cache: rope).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDimension))
    }
}

struct SCAIL2CrossAttentionCache {
    let textKey: MLXArray
    let textValue: MLXArray
    let imageKey: MLXArray
    let imageValue: MLXArray
}

public struct SCAIL2PreparedConditioning {
    let text: MLXArray
    let image: MLXArray
    let crossAttentionCaches: [SCAIL2CrossAttentionCache]

    var arrays: [MLXArray] {
        [text, image] + crossAttentionCaches.flatMap {
            [$0.textKey, $0.textValue, $0.imageKey, $0.imageValue]
        }
    }
}

final class SCAIL2CrossAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float
    @ModuleInfo(key: "q") var query: Linear
    @ModuleInfo(key: "k") var key: Linear
    @ModuleInfo(key: "v") var value: Linear
    @ModuleInfo(key: "o") var output: Linear
    @ModuleInfo(key: "k_img") var imageKey: Linear
    @ModuleInfo(key: "v_img") var imageValue: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: Wan2RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: Wan2RMSNorm
    @ModuleInfo(key: "norm_k_img") var imageKeyNorm: Wan2RMSNorm

    init(configuration: SCAIL2TransformerConfiguration) {
        self.heads = configuration.headCount
        self.headDimension = configuration.hiddenSize / configuration.headCount
        self.scale = 1 / Float(headDimension).squareRoot()
        self._query.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._key.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._value.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._output.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._imageKey.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._imageValue.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._queryNorm.wrappedValue = Wan2RMSNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
        self._keyNorm.wrappedValue = Wan2RMSNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
        self._imageKeyNorm.wrappedValue = Wan2RMSNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
    }

    func prepareCache(text: MLXArray, image: MLXArray) -> SCAIL2CrossAttentionCache {
        let batch = text.dim(0)
        let textTyped = text.asType(key.weight.dtype)
        let imageTyped = image.asType(imageKey.weight.dtype)
        return SCAIL2CrossAttentionCache(
            textKey: keyNorm(key(textTyped)).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3),
            textValue: value(textTyped).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3),
            imageKey: imageKeyNorm(imageKey(imageTyped)).reshaped(batch, -1, heads, headDimension)
                .transposed(0, 2, 1, 3),
            imageValue: imageValue(imageTyped).reshaped(batch, -1, heads, headDimension)
                .transposed(0, 2, 1, 3)
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        text: MLXArray,
        image: MLXArray,
        cache: SCAIL2CrossAttentionCache?
    ) -> MLXArray {
        let batch = input.dim(0)
        let q = queryNorm(query(input.asType(query.weight.dtype)))
            .reshaped(batch, -1, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let prepared = cache ?? prepareCache(text: text, image: image)
        let textOutput = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: prepared.textKey,
            values: prepared.textValue,
            scale: scale,
            mask: .none
        )
        let imageOutput = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: prepared.imageKey,
            values: prepared.imageValue,
            scale: scale,
            mask: .none
        )
        let merged = (textOutput + imageOutput)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, -1, heads * headDimension)
        return output(merged)
    }
}

final class SCAIL2TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var selfNorm: Wan2LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: SCAIL2SelfAttention
    @ModuleInfo(key: "norm3") var crossNorm: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: SCAIL2CrossAttention
    @ModuleInfo(key: "norm2") var feedForwardNorm: Wan2LayerNorm
    @ModuleInfo(key: "ffn") var feedForward: Wan2FeedForward
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    init(configuration: SCAIL2TransformerConfiguration) {
        self._selfNorm.wrappedValue = Wan2LayerNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
        self._selfAttention.wrappedValue = SCAIL2SelfAttention(configuration: configuration)
        self._crossNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.epsilon
        )
        self._crossAttention.wrappedValue = SCAIL2CrossAttention(configuration: configuration)
        self._feedForwardNorm.wrappedValue = Wan2LayerNorm(
            dimensions: configuration.hiddenSize,
            epsilon: configuration.epsilon
        )
        self._feedForward.wrappedValue = Wan2FeedForward(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.feedForwardSize
        )
        self._modulation.wrappedValue = MLX.zeros([1, 6, configuration.hiddenSize], dtype: .float32)
    }

    func callAsFunction(
        _ input: MLXArray,
        modulationInput: MLXArray,
        text: MLXArray,
        image: MLXArray,
        rope: Wan2RoPE.Cache,
        crossCache: SCAIL2CrossAttentionCache?
    ) -> MLXArray {
        let parts = MLX.split(modulation.expandedDimensions(axis: 1) + modulationInput, parts: 6, axis: 2)
            .map { $0.squeezed(axis: 2) }
        let hiddenType = input.dtype
        let selfInput = selfNorm(input.asType(.float32)) * (1 + parts[1]) + parts[0]
        let selfOutput = selfAttention(selfInput.asType(hiddenType), rope: rope)
        var hidden = (input.asType(.float32) + selfOutput.asType(.float32) * parts[2]).asType(hiddenType)
        hidden = hidden + crossAttention(
            crossNorm(hidden.asType(.float32)).asType(hiddenType),
            text: text,
            image: image,
            cache: crossCache
        )
        let feedForwardInput = feedForwardNorm(hidden.asType(.float32)) * (1 + parts[4]) + parts[3]
        let feedForwardOutput = feedForward(feedForwardInput.asType(hiddenType))
        return (hidden.asType(.float32) + feedForwardOutput.asType(.float32) * parts[5]).asType(hiddenType)
    }

    func parityTrace(
        _ input: MLXArray,
        modulationInput: MLXArray,
        text: MLXArray,
        image: MLXArray,
        rope: Wan2RoPE.Cache,
        crossCache: SCAIL2CrossAttentionCache?
    ) -> [String: MLXArray] {
        let parts = MLX.split(modulation.expandedDimensions(axis: 1) + modulationInput, parts: 6, axis: 2)
            .map { $0.squeezed(axis: 2) }
        let hiddenType = input.dtype
        let selfInput = selfNorm(input.asType(.float32)) * (1 + parts[1]) + parts[0]
        let selfOutput = selfAttention(selfInput.asType(hiddenType), rope: rope)
        var hidden = (input.asType(.float32) + selfOutput.asType(.float32) * parts[2]).asType(hiddenType)
        let postSelf = hidden
        let crossInput = crossNorm(hidden.asType(.float32)).asType(hiddenType)
        let crossOutput = crossAttention(
            crossInput,
            text: text,
            image: image,
            cache: crossCache
        )
        hidden = hidden + crossOutput
        let postCross = hidden
        let feedForwardInput = feedForwardNorm(hidden.asType(.float32)) * (1 + parts[4]) + parts[3]
        let feedForwardOutput = feedForward(feedForwardInput.asType(hiddenType))
        hidden = (hidden.asType(.float32) + feedForwardOutput.asType(.float32) * parts[5]).asType(hiddenType)
        return [
            "self_input": selfInput,
            "self_output": selfOutput,
            "post_self": postSelf,
            "cross_input": crossInput,
            "cross_output": crossOutput,
            "post_cross": postCross,
            "ffn_input": feedForwardInput,
            "ffn_output": feedForwardOutput,
            "block_0_output": hidden,
        ]
    }
}

final class SCAIL2ImageProjection: Module {
    @ModuleInfo(key: "layer_0") var inputNorm: LayerNorm
    @ModuleInfo(key: "layer_1") var inputProjection: Linear
    @ModuleInfo(key: "layer_3") var outputProjection: Linear
    @ModuleInfo(key: "layer_4") var outputNorm: LayerNorm

    init(configuration: SCAIL2TransformerConfiguration) {
        self._inputNorm.wrappedValue = LayerNorm(
            dimensions: configuration.imageEmbeddingSize,
            eps: configuration.epsilon
        )
        self._inputProjection.wrappedValue = Linear(
            configuration.imageEmbeddingSize,
            configuration.imageEmbeddingSize,
            bias: true
        )
        self._outputProjection.wrappedValue = Linear(
            configuration.imageEmbeddingSize,
            configuration.hiddenSize,
            bias: true
        )
        self._outputNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.epsilon
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        outputNorm(outputProjection(MLXNN.gelu(inputProjection(inputNorm(input)))))
    }
}

public final class SCAIL2TransformerModel: Module {
    public let configuration: SCAIL2TransformerConfiguration
    let inverseTimestepFrequencies: MLXArray
    let ropeFrequencies: MLXArray
    @ModuleInfo(key: "patch_embedding_proj") var patchProjection: Linear
    @ModuleInfo(key: "patch_embedding_pose_proj") var posePatchProjection: Linear
    @ModuleInfo(key: "patch_embedding_mask_proj") var maskPatchProjection: Linear
    @ModuleInfo(key: "text_embedding_0") var textProjectionIn: Linear
    @ModuleInfo(key: "text_embedding_1") var textProjectionOut: Linear
    @ModuleInfo(key: "time_embedding_0") var timestepProjectionIn: Linear
    @ModuleInfo(key: "time_embedding_1") var timestepProjectionOut: Linear
    @ModuleInfo(key: "time_projection") var modulationProjection: Linear
    @ModuleInfo(key: "img_emb") var imageProjection: SCAIL2ImageProjection
    @ModuleInfo(key: "blocks") var blocks: [SCAIL2TransformerBlock]
    @ModuleInfo(key: "head") var outputHead: Wan2OutputHead

    public init(configuration: SCAIL2TransformerConfiguration = SCAIL2TransformerConfiguration()) {
        self.configuration = configuration
        let patchVolume = configuration.patchSize.reduce(1, *)
        self._patchProjection.wrappedValue = Linear(
            configuration.inputChannels * patchVolume,
            configuration.hiddenSize,
            bias: true
        )
        self._posePatchProjection.wrappedValue = Linear(
            configuration.inputChannels * patchVolume,
            configuration.hiddenSize,
            bias: true
        )
        self._maskPatchProjection.wrappedValue = Linear(
            configuration.maskChannels * patchVolume,
            configuration.hiddenSize,
            bias: true
        )
        self._textProjectionIn.wrappedValue = Linear(
            configuration.textEmbeddingSize,
            configuration.hiddenSize,
            bias: true
        )
        self._textProjectionOut.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: true
        )
        self._timestepProjectionIn.wrappedValue = Linear(
            configuration.timestepFrequencySize,
            configuration.hiddenSize,
            bias: true
        )
        self._timestepProjectionOut.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: true
        )
        self._modulationProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize * 6,
            bias: true
        )
        self._imageProjection.wrappedValue = SCAIL2ImageProjection(configuration: configuration)
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            SCAIL2TransformerBlock(configuration: configuration)
        }
        let outputConfiguration = Wan2TransformerConfiguration(
            patchSize: configuration.patchSize,
            textLength: configuration.textLength,
            inputChannels: configuration.inputChannels,
            hiddenSize: configuration.hiddenSize,
            feedForwardSize: configuration.feedForwardSize,
            timestepFrequencySize: configuration.timestepFrequencySize,
            textEmbeddingSize: configuration.textEmbeddingSize,
            outputChannels: configuration.outputChannels,
            headCount: configuration.headCount,
            layerCount: configuration.layerCount,
            epsilon: configuration.epsilon
        )
        self._outputHead.wrappedValue = Wan2OutputHead(configuration: outputConfiguration)
        let frequencyHalf = configuration.timestepFrequencySize / 2
        let frequencyValues: [Float] = (0..<frequencyHalf).map {
            Float(Foundation.pow(10_000, -Double($0) / Double(frequencyHalf)))
        }
        self.inverseTimestepFrequencies = MLXArray(frequencyValues)
        let headDimension = configuration.hiddenSize / configuration.headCount
        let sixth = headDimension / 6
        self.ropeFrequencies = Wan2RoPE.frequencies(
            maxSequence: configuration.ropeTableLength,
            dimensions: [headDimension - 4 * sixth, 2 * sixth, 2 * sixth]
        )
    }

    public func callAsFunction(_ input: SCAIL2TransformerInput) -> MLXArray {
        let conditioning = prepareConditioning(
            textEmbeddings: input.textEmbeddings,
            imageEmbeddings: input.imageEmbeddings
        )
        return callAsFunction(input, conditioning: conditioning)
    }

    public func prepareConditioning(
        textEmbeddings: MLXArray,
        imageEmbeddings: MLXArray
    ) -> SCAIL2PreparedConditioning {
        precondition(textEmbeddings.ndim == 3 && textEmbeddings.dim(0) == 1)
        precondition(textEmbeddings.dim(2) == configuration.textEmbeddingSize)
        precondition(imageEmbeddings.ndim == 3 && imageEmbeddings.dim(0) == 1)
        precondition(imageEmbeddings.dim(2) == configuration.imageEmbeddingSize)
        let text = embedText(textEmbeddings)
        let image = imageProjection(imageEmbeddings.asType(patchProjection.weight.dtype))
        return SCAIL2PreparedConditioning(
            text: text,
            image: image,
            crossAttentionCaches: blocks.map {
                $0.crossAttention.prepareCache(text: text, image: image)
            }
        )
    }

    public func callAsFunction(
        _ input: SCAIL2TransformerInput,
        conditioning: SCAIL2PreparedConditioning
    ) -> MLXArray {
        validate(input)
        precondition(conditioning.crossAttentionCaches.count == blocks.count)
        let assembled = assembleTokens(input)
        var hidden = assembled.tokens
        let timestep = input.timestep.ndim == 0 ? input.timestep.reshaped(1) : input.timestep
        let sinusoid = timestep.asType(.float32).expandedDimensions(axis: -1) * inverseTimestepFrequencies
        let frequencyEmbedding = MLX.concatenated([MLX.cos(sinusoid), MLX.sin(sinusoid)], axis: -1)
            .asType(timestepProjectionIn.weight.dtype)
        let timestepEmbedding = timestepProjectionOut(MLXNN.silu(timestepProjectionIn(frequencyEmbedding)))
        let modulation = modulationProjection(MLXNN.silu(timestepEmbedding)).asType(.float32)
            .reshaped(1, 1, 6, configuration.hiddenSize)
        let rope = SCAIL2RoPE.prepare(
            layout: assembled.layout,
            frequencies: ropeFrequencies,
            mode: input.mode,
            poseWidthShift: configuration.poseWidthShift,
            replacementReferenceHeightShift: configuration.replacementReferenceHeightShift
        )
        for (block, crossCache) in zip(blocks, conditioning.crossAttentionCaches) {
            hidden = block(
                hidden,
                modulationInput: modulation,
                text: conditioning.text,
                image: conditioning.image,
                rope: rope,
                crossCache: crossCache
            )
        }
        let patches = outputHead(hidden, timestepEmbedding: timestepEmbedding)
        return unpatchifyVideo(
            patches,
            grid: assembled.layout.videoGrid,
            offset: assembled.layout.videoOffset
        )
    }

    func parityTrace(
        _ input: SCAIL2TransformerInput,
        assembledTokensOverride: MLXArray? = nil
    ) -> [String: MLXArray] {
        validate(input)
        let conditioning = prepareConditioning(
            textEmbeddings: input.textEmbeddings,
            imageEmbeddings: input.imageEmbeddings
        )
        let assembled = assembleTokens(input)
        var hidden = assembledTokensOverride ?? assembled.tokens
        let timestep = input.timestep.ndim == 0 ? input.timestep.reshaped(1) : input.timestep
        let sinusoid = timestep.asType(.float32).expandedDimensions(axis: -1) * inverseTimestepFrequencies
        let frequencyEmbedding = MLX.concatenated([MLX.cos(sinusoid), MLX.sin(sinusoid)], axis: -1)
            .asType(timestepProjectionIn.weight.dtype)
        let timestepEmbedding = timestepProjectionOut(MLXNN.silu(timestepProjectionIn(frequencyEmbedding)))
        let modulation = modulationProjection(MLXNN.silu(timestepEmbedding)).asType(.float32)
            .reshaped(1, 1, 6, configuration.hiddenSize)
        let rope = SCAIL2RoPE.prepare(
            layout: assembled.layout,
            frequencies: ropeFrequencies,
            mode: input.mode,
            poseWidthShift: configuration.poseWidthShift,
            replacementReferenceHeightShift: configuration.replacementReferenceHeightShift
        )
        let assembledTokens = assembled.tokens
        precondition(blocks.count == 1)
        let blockTrace = blocks[0].parityTrace(
            hidden,
            modulationInput: modulation,
            text: conditioning.text,
            image: conditioning.image,
            rope: rope,
            crossCache: conditioning.crossAttentionCaches[0]
        )
        hidden = blockTrace["block_0_output"]!
        let patches = outputHead(hidden, timestepEmbedding: timestepEmbedding)
        let output = unpatchifyVideo(
            patches,
            grid: assembled.layout.videoGrid,
            offset: assembled.layout.videoOffset
        )
        var trace: [String: MLXArray] = [
            "timestep_embedding": timestepEmbedding,
            "modulation": modulation.reshaped(1, -1),
            "text_conditioning": conditioning.text,
            "image_conditioning": conditioning.image,
            "assembled_tokens": assembledTokens,
            "block_0_output": hidden,
            "head_patches": patches,
            "output": output,
        ]
        for (key, value) in blockTrace {
            trace[key] = value
        }
        return trace
    }

    func parityFullTrace(
        _ input: SCAIL2TransformerInput,
        blockInputs: [MLXArray],
        finalHidden: MLXArray
    ) -> [String: MLXArray] {
        validate(input)
        precondition(blockInputs.count == blocks.count)
        let conditioning = prepareConditioning(
            textEmbeddings: input.textEmbeddings,
            imageEmbeddings: input.imageEmbeddings
        )
        let assembled = assembleTokens(input)
        let timestep = input.timestep.ndim == 0 ? input.timestep.reshaped(1) : input.timestep
        let sinusoid = timestep.asType(.float32).expandedDimensions(axis: -1) * inverseTimestepFrequencies
        let frequencyEmbedding = MLX.concatenated([MLX.cos(sinusoid), MLX.sin(sinusoid)], axis: -1)
            .asType(timestepProjectionIn.weight.dtype)
        let timestepEmbedding = timestepProjectionOut(MLXNN.silu(timestepProjectionIn(frequencyEmbedding)))
        let modulation = modulationProjection(MLXNN.silu(timestepEmbedding)).asType(.float32)
            .reshaped(1, 1, 6, configuration.hiddenSize)
        let rope = SCAIL2RoPE.prepare(
            layout: assembled.layout,
            frequencies: ropeFrequencies,
            mode: input.mode,
            poseWidthShift: configuration.poseWidthShift,
            replacementReferenceHeightShift: configuration.replacementReferenceHeightShift
        )
        var trace: [String: MLXArray] = ["assembled_tokens": assembled.tokens]
        for index in blocks.indices {
            trace["block_\(index)_output"] = blocks[index](
                blockInputs[index],
                modulationInput: modulation,
                text: conditioning.text,
                image: conditioning.image,
                rope: rope,
                crossCache: conditioning.crossAttentionCaches[index]
            )
        }
        let patches = outputHead(finalHidden, timestepEmbedding: timestepEmbedding)
        trace["head_patches"] = patches
        trace["output"] = unpatchifyVideo(
            patches,
            grid: assembled.layout.videoGrid,
            offset: assembled.layout.videoOffset
        )
        return trace
    }

    func embedText(_ embeddings: MLXArray) -> MLXArray {
        precondition(embeddings.ndim == 3 && embeddings.dim(0) == 1)
        let context: MLXArray
        if embeddings.dim(1) < configuration.textLength {
            context = MLX.concatenated([
                embeddings,
                MLX.zeros(
                    [1, configuration.textLength - embeddings.dim(1), embeddings.dim(2)],
                    dtype: embeddings.dtype
                ),
            ], axis: 1)
        } else {
            context = embeddings[0..., 0..<configuration.textLength]
        }
        return textProjectionOut(MLXNN.geluApproximate(textProjectionIn(context)))
            .asType(patchProjection.weight.dtype)
    }

    private func validate(_ input: SCAIL2TransformerInput) {
        let video = input.videoLatent
        let reference = input.referenceLatent
        let driving = input.drivingLatent
        precondition(video.ndim == 4 && video.dim(0) == configuration.outputChannels)
        precondition(reference.shape == [configuration.outputChannels, 1, video.dim(2), video.dim(3)])
        precondition(input.referenceMask.shape == [configuration.maskChannels, 1, video.dim(2), video.dim(3)])
        precondition(driving.shape == [
            configuration.outputChannels, video.dim(1), video.dim(2) / 2, video.dim(3) / 2,
        ])
        precondition(input.drivingMask.shape == [
            configuration.maskChannels, video.dim(1), video.dim(2) / 2, video.dim(3) / 2,
        ])
        precondition(input.historyMask.map { $0.shape == [4, video.dim(1), video.dim(2), video.dim(3)] } ?? true)
        precondition(input.additionalReferenceLatents.allSatisfy {
            $0.shape == [configuration.outputChannels, 1, video.dim(2), video.dim(3)]
        })
        precondition(input.additionalReferenceMasks.allSatisfy {
            $0.shape == [configuration.maskChannels, 1, video.dim(2), video.dim(3)]
        })
        precondition(input.textEmbeddings.ndim == 3 && input.textEmbeddings.dim(0) == 1)
        precondition(input.textEmbeddings.dim(2) == configuration.textEmbeddingSize)
        precondition(input.imageEmbeddings.ndim == 3 && input.imageEmbeddings.dim(0) == 1)
        precondition(input.imageEmbeddings.dim(2) == configuration.imageEmbeddingSize)
        precondition(input.timestep.size == 1)
    }

    private func assembleTokens(_ input: SCAIL2TransformerInput) -> (
        tokens: MLXArray,
        layout: SCAIL2TokenLayout
    ) {
        let video = input.videoLatent
        let onesReference = addControlChannels(input.referenceLatent, value: 1)
        let history = input.historyMask ?? MLX.zeros(
            [4, video.dim(1), video.dim(2), video.dim(3)],
            dtype: video.dtype
        )
        let controlledVideo = MLX.concatenated([video, history.asType(video.dtype)], axis: 0)
        let referenceAndVideo = MLX.concatenated([onesReference, controlledVideo], axis: 1)
        let zeroVideoMask = MLX.zeros(
            [configuration.maskChannels, video.dim(1), video.dim(2), video.dim(3)],
            dtype: input.referenceMask.dtype
        )
        let referenceAndVideoMask = MLX.concatenated([input.referenceMask, zeroVideoMask], axis: 1)
        let primary = project(
            referenceAndVideo,
            with: patchProjection
        ) + project(referenceAndVideoMask, with: maskPatchProjection)
        let primaryGrid = Wan2PatchLayout.flatten(
            referenceAndVideo,
            patchSize: configuration.patchSize
        ).grid
        let referenceGrid = Wan2GridSize(
            frames: 1,
            height: primaryGrid.height,
            width: primaryGrid.width
        )
        let videoGrid = Wan2GridSize(
            frames: primaryGrid.frames - 1,
            height: primaryGrid.height,
            width: primaryGrid.width
        )

        let controlledDriving = addControlChannels(input.drivingLatent, value: 1)
        let driving = project(controlledDriving, with: posePatchProjection)
            + project(input.drivingMask, with: maskPatchProjection)
        let drivingGrid = Wan2PatchLayout.flatten(
            controlledDriving,
            patchSize: configuration.patchSize
        ).grid

        var parts: [MLXArray] = []
        var additionalGrid: Wan2GridSize?
        if !input.additionalReferenceLatents.isEmpty {
            let latents = MLX.concatenated(
                input.additionalReferenceLatents.map { addControlChannels($0, value: 1) },
                axis: 1
            )
            let masks = MLX.concatenated(input.additionalReferenceMasks, axis: 1)
            let additional = project(latents, with: patchProjection) + project(masks, with: maskPatchProjection)
            parts.append(additional)
            additionalGrid = Wan2PatchLayout.flatten(latents, patchSize: configuration.patchSize).grid
        }
        parts.append(contentsOf: [primary, driving])
        return (
            MLX.concatenated(parts, axis: 1),
            SCAIL2TokenLayout(
                additionalReferenceGrid: additionalGrid,
                referenceGrid: referenceGrid,
                videoGrid: videoGrid,
                drivingGrid: drivingGrid
            )
        )
    }

    private func addControlChannels(_ latent: MLXArray, value: Float) -> MLXArray {
        MLX.concatenated([
            latent,
            MLX.full(
                [4, latent.dim(1), latent.dim(2), latent.dim(3)],
                values: MLXArray(value).asType(latent.dtype)
            ),
        ], axis: 0)
    }

    private func project(_ latent: MLXArray, with projection: Linear) -> MLXArray {
        let flattened = Wan2PatchLayout.flatten(latent, patchSize: configuration.patchSize).value
        return projection(flattened.asType(projection.weight.dtype)).expandedDimensions(axis: 0)
    }

    private func unpatchifyVideo(_ patches: MLXArray, grid: Wan2GridSize, offset: Int) -> MLXArray {
        let temporalPatch = configuration.patchSize[0]
        let heightPatch = configuration.patchSize[1]
        let widthPatch = configuration.patchSize[2]
        return patches[0, offset..<(offset + grid.sequenceLength)]
            .reshaped(
                grid.frames, grid.height, grid.width,
                temporalPatch, heightPatch, widthPatch,
                configuration.outputChannels
            )
            .transposed(6, 0, 3, 1, 4, 2, 5)
            .reshaped(
                configuration.outputChannels,
                grid.frames * temporalPatch,
                grid.height * heightPatch,
                grid.width * widthPatch
            )
            .asType(.float32)
    }
}
