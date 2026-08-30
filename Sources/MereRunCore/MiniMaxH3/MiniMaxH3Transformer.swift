import Foundation
import MLX
import MLXFast
import MLXNN
#if DEBUG
import MLXRandom
#endif

@inline(__always)
private func miniMaxH3Linear(_ linear: Linear, _ value: MLXArray) -> MLXArray {
    let activationDType = (linear as? QuantizedLinear)?.scales.dtype ?? linear.weight.dtype
    return linear(value.asType(activationDType))
}

enum MiniMaxH3ExactKernelMode: String, Sendable, Equatable {
    case disabled
    case boundaryLayout = "boundary-layout"
    case affineQ8 = "affine-q8"
    case affineQ8MLP = "affine-q8-mlp"
    case fastH3Metal = "fasth3-metal"

    var usesBoundaryLayout: Bool {
        self == .boundaryLayout || self == .affineQ8 || self == .fastH3Metal
    }

    var usesAffineQ8FeedForward: Bool {
        self == .affineQ8 || self == .affineQ8MLP || self == .fastH3Metal
    }

    var usesTiledAffineQ8FeedForward: Bool {
        self == .affineQ8MLP || self == .fastH3Metal
    }
}

enum MiniMaxH3ExactKernelStage: String, CaseIterable, Sendable, Hashable {
    case attentionAdaLN = "k0-attention-adaln"
    case gateAdaLN = "k1-gate-adaln"
    case qkvLayout = "k2a-qkv-layout"
    case qkvProjection = "k2b-qkv-projection"
    case attentionOutput = "k3-attention-output"
    case feedForwardInput = "k4a-feed-forward-input"
    case feedForwardOutput = "k4b-feed-forward-output"
}

private struct MiniMaxH3AffineQ8Weights {
    let codes: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

struct MiniMaxH3FastH3CompressionGate {
    enum Storage: Sendable, Equatable {
        case dense
        case affineQ8(groupSize: Int, bits: Int)
    }

    let storage: Storage
    let parameters: [MLXArray]

    init(weight: MLXArray) {
        self.storage = .dense
        self.parameters = [weight]
    }

    init(
        codes: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int
    ) {
        precondition(groupSize > 0 && bits == 8)
        self.storage = .affineQ8(groupSize: groupSize, bits: bits)
        self.parameters = [codes, scales, biases]
    }

    func project(_ input: MLXArray) -> MLXArray {
        Self.project(input, storage: storage, parameters: parameters)
    }

    static func project(
        _ input: MLXArray,
        storage: Storage,
        parameters: [MLXArray]
    ) -> MLXArray {
        switch storage {
        case .dense:
            precondition(parameters.count == 1)
            let weight = parameters[0]
            return MLX.matmul(input.asType(weight.dtype), weight.T)
        case .affineQ8(let groupSize, let bits):
            precondition(parameters.count == 3)
            return MLX.quantizedMM(
                input.asType(parameters[1].dtype),
                parameters[0],
                scales: parameters[1],
                biases: parameters[2],
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }
    }
}

private func miniMaxH3AffineQ8Weights(
    _ linear: Linear
) -> MiniMaxH3AffineQ8Weights? {
    guard let quantized = linear as? QuantizedLinear,
          quantized.bits == 8,
          quantized.groupSize == 64,
          quantized.mode == .affine,
          quantized.bias == nil,
          quantized.globalScale == nil,
          let biases = quantized.biases else {
        return nil
    }
    return MiniMaxH3AffineQ8Weights(
        codes: quantized.weight,
        scales: quantized.scales,
        biases: biases
    )
}

func miniMaxH3SplitProjectedQKV(
    _ projected: MLXArray,
    heads: Int,
    headDimension: Int
) -> [MLXArray] {
    precondition(projected.dim(-1) == heads * 3 * headDimension)
    return MLX.split(projected, parts: 3, axis: -1).map {
        $0.reshaped(projected.dim(0), projected.dim(1), heads, headDimension)
    }
}

public struct MiniMaxH3TransformerConfiguration: Hashable, Sendable {
    public let hiddenSize: Int
    public let layerCount: Int
    public let refinerLayerCount: Int
    public let attentionHeadCount: Int
    public let attentionHeadDimension: Int
    public let feedForwardSize: Int
    public let videoLatentChannels: Int
    public let audioLatentChannels: Int
    public let patchSize: [Int]
    public let textDimension: Int
    public let timeFrequencyDimension: Int
    public let timeEmbeddingHiddenSize: Int
    public let timeEmbeddingDimension: Int
    public let ropeFrequencyCount: Int
    public let ropeTheta: Float
    public let normEpsilon: Float
    public let queryKeyNormEpsilon: Float

    public init(
        hiddenSize: Int = 5_376,
        layerCount: Int = 50,
        refinerLayerCount: Int = 2,
        attentionHeadCount: Int = 56,
        attentionHeadDimension: Int = 128,
        feedForwardSize: Int = 14_336,
        videoLatentChannels: Int = 24,
        audioLatentChannels: Int = 32,
        patchSize: [Int] = [1, 2, 2],
        textDimension: Int = 5_120,
        timeFrequencyDimension: Int = 256,
        timeEmbeddingHiddenSize: Int = 5_376,
        timeEmbeddingDimension: Int = 2_688,
        ropeFrequencyCount: Int = 16,
        ropeTheta: Float = 10_000,
        normEpsilon: Float = 1e-5,
        queryKeyNormEpsilon: Float = 1e-5
    ) {
        precondition(hiddenSize > 0 && layerCount > 0 && refinerLayerCount >= 0)
        precondition(attentionHeadCount > 0 && attentionHeadDimension > 0)
        precondition(patchSize.count == 3)
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.refinerLayerCount = refinerLayerCount
        self.attentionHeadCount = attentionHeadCount
        self.attentionHeadDimension = attentionHeadDimension
        self.feedForwardSize = feedForwardSize
        self.videoLatentChannels = videoLatentChannels
        self.audioLatentChannels = audioLatentChannels
        self.patchSize = patchSize
        self.textDimension = textDimension
        self.timeFrequencyDimension = timeFrequencyDimension
        self.timeEmbeddingHiddenSize = timeEmbeddingHiddenSize
        self.timeEmbeddingDimension = timeEmbeddingDimension
        self.ropeFrequencyCount = ropeFrequencyCount
        self.ropeTheta = ropeTheta
        self.normEpsilon = normEpsilon
        self.queryKeyNormEpsilon = queryKeyNormEpsilon
    }

    public init(_ configuration: MiniMaxH3Configuration) {
        self.init(
            hiddenSize: configuration.hiddenSize,
            layerCount: configuration.layerCount,
            refinerLayerCount: configuration.refinerLayerCount,
            attentionHeadCount: configuration.attentionHeadCount,
            attentionHeadDimension: configuration.attentionHeadDimension,
            feedForwardSize: configuration.feedForwardSize,
            videoLatentChannels: configuration.videoLatentChannels,
            audioLatentChannels: configuration.audioLatentChannels,
            patchSize: configuration.patchSize,
            textDimension: configuration.textDimension,
            timeFrequencyDimension: configuration.timeFrequencyDimension,
            timeEmbeddingHiddenSize: configuration.timeEmbeddingHiddenSize,
            timeEmbeddingDimension: configuration.timeEmbeddingDimension
        )
    }

    var videoPatchDimension: Int {
        videoLatentChannels * patchSize.reduce(1, *)
    }
}

public struct MiniMaxH3TransformerOutput {
    public let videoVelocityRows: MLXArray
    public let audioVelocityRows: MLXArray
}

struct MiniMaxH3BlockReuseResult {
    let output: MiniMaxH3TransformerOutput
    let refreshedTailResidual: MLXArray?
}

struct MiniMaxH3FirstBlockChange: Sendable, Equatable {
    let videoGlobal: Float
    let audioGlobal: Float
    let videoTemporalMaximum: Float
    let audioTemporalMaximum: Float

    var isFinite: Bool {
        videoGlobal.isFinite
            && audioGlobal.isFinite
            && videoTemporalMaximum.isFinite
            && audioTemporalMaximum.isFinite
    }

    static func measure(
        current: MLXArray,
        previous: MLXArray,
        layout: MiniMaxH3PackedLayout
    ) -> Self {
        let audioRowCount = layout.targetAudioRows.count
        let videoRowCount = layout.targetVideoRows.count
        let totalRowCount = audioRowCount + videoRowCount
        precondition(current.shape == previous.shape)
        precondition(current.ndim == 3 && current.dim(0) == 1)
        precondition(current.dim(1) == totalRowCount)
        precondition(audioRowCount == layout.audioLatentFrames * 2)
        precondition(videoRowCount.isMultiple(of: layout.videoLatentFrames))

        let currentFloat = current.asType(.float32)
        let previousFloat = previous.asType(.float32)
        let currentAudio = currentFloat[0..., 0..<audioRowCount, 0...]
        let previousAudio = previousFloat[0..., 0..<audioRowCount, 0...]
        let currentVideo = currentFloat[0..., audioRowCount..<totalRowCount, 0...]
        let previousVideo = previousFloat[0..., audioRowCount..<totalRowCount, 0...]

        func relativeGlobal(_ value: MLXArray, _ reference: MLXArray) -> MLXArray {
            let numerator = MLX.mean(MLX.abs(value - reference))
            let denominator = MLX.maximum(
                MLX.mean(MLX.abs(reference)),
                MLXArray(Float(1e-8))
            )
            return numerator / denominator
        }

        let videoRowsPerFrame = videoRowCount / layout.videoLatentFrames
        let videoDifference = MLX.abs(currentVideo - previousVideo).reshaped(
            1,
            layout.videoLatentFrames,
            videoRowsPerFrame,
            current.dim(2)
        )
        let videoReference = MLX.abs(previousVideo).reshaped(
            1,
            layout.videoLatentFrames,
            videoRowsPerFrame,
            current.dim(2)
        )
        let videoTemporal = MLX.mean(videoDifference, axes: [0, 2, 3])
            / MLX.maximum(
                MLX.mean(videoReference, axes: [0, 2, 3]),
                MLXArray(Float(1e-8))
            )

        let audioDifference = MLX.abs(currentAudio - previousAudio).reshaped(
            1,
            2,
            layout.audioLatentFrames,
            current.dim(2)
        )
        let audioReference = MLX.abs(previousAudio).reshaped(
            1,
            2,
            layout.audioLatentFrames,
            current.dim(2)
        )
        let audioTemporal = MLX.mean(audioDifference, axes: [0, 1, 3])
            / MLX.maximum(
                MLX.mean(audioReference, axes: [0, 1, 3]),
                MLXArray(Float(1e-8))
            )

        let metrics = MLX.stacked([
            relativeGlobal(currentVideo, previousVideo),
            relativeGlobal(currentAudio, previousAudio),
            MLX.max(videoTemporal),
            MLX.max(audioTemporal),
        ]).asType(.float32)
        MLX.eval(metrics)
        let values = metrics.asArray(Float.self)
        return .init(
            videoGlobal: values[0],
            audioGlobal: values[1],
            videoTemporalMaximum: values[2],
            audioTemporalMaximum: values[3]
        )
    }
}

struct MiniMaxH3AdaptiveBlockReuseResult {
    let output: MiniMaxH3TransformerOutput
    let refreshedFirstResidual: MLXArray?
    let refreshedTargetTailResidual: MLXArray?
    let reusedTail: Bool
    let change: MiniMaxH3FirstBlockChange?
}

private final class MiniMaxH3Attention: Module {
    let heads: Int
    let headDimension: Int
    let innerDimension: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var queryKeyValue: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "out_proj") var output: Linear
    var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled
    var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases)
    var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)?
    var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)?

    init(configuration: MiniMaxH3TransformerConfiguration) {
        self.heads = configuration.attentionHeadCount
        self.headDimension = configuration.attentionHeadDimension
        self.innerDimension = heads * headDimension
        self.scale = 1 / sqrt(Float(headDimension))
        self._queryKeyValue.wrappedValue = Linear(
            configuration.hiddenSize,
            3 * innerDimension,
            bias: false
        )
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: headDimension,
            eps: configuration.queryKeyNormEpsilon
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: headDimension,
            eps: configuration.queryKeyNormEpsilon
        )
        self._output.wrappedValue = Linear(innerDimension, configuration.hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray, rope: MiniMaxH3RotaryEmbedding?) -> MLXArray {
        let projected = project(input, rope: rope)
        let attended = scaledDotProductAttention(
            queries: projected[0],
            keys: projected[1],
            values: projected[2],
            maximumQueryTokens: nil,
            maximumHeadsPerKernel: nil
        )
        return projectOutput(attended)
    }

    func project(_ input: MLXArray, rope: MiniMaxH3RotaryEmbedding?) -> [MLXArray] {
        if exactKernelMode == .affineQ8,
           enabledExactKernelStages.contains(.qkvProjection) {
            if let rope, let weights = miniMaxH3AffineQ8Weights(queryKeyValue) {
                if let projected = MiniMaxH3FusedKernels.projectHeadMajorQKVAffineInt8(
                    input: input,
                    weightCodes: weights.codes,
                    weightScales: weights.scales,
                    weightBiases: weights.biases,
                    queryNormWeight: queryNorm.weight,
                    keyNormWeight: keyNorm.weight,
                    ropeCosine: rope.cosine,
                    ropeSine: rope.sine,
                    eps: queryNorm.eps
                ) {
                    exactKernelDispatchHandler?(.qkvProjection)
                    return [projected.query, projected.key, projected.value]
                }
                exactKernelFallbackHandler?(
                    .qkvProjection,
                    "input=\(input.dtype):\(input.shape) q_norm=\(queryNorm.weight.dtype) "
                        + "k_norm=\(keyNorm.weight.dtype) rope=\(rope.cosine.dtype):"
                        + "\(rope.cosine.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.qkvProjection, "weight-or-rope-contract")
            }
        }
        let globalProjection = queryKeyValue(input)
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.qkvLayout) {
            if let rope {
                if let projected = MiniMaxH3FusedKernels.prepareHeadMajorQKV(
                    projected: globalProjection,
                    queryNormWeight: queryNorm.weight,
                    keyNormWeight: keyNorm.weight,
                    ropeCosine: rope.cosine,
                    ropeSine: rope.sine,
                    eps: queryNorm.eps
                ) {
                    exactKernelDispatchHandler?(.qkvLayout)
                    return [projected.query, projected.key, projected.value]
                }
                exactKernelFallbackHandler?(
                    .qkvLayout,
                    "projection=\(globalProjection.dtype):\(globalProjection.shape) "
                        + "q_norm=\(queryNorm.weight.dtype) k_norm=\(keyNorm.weight.dtype) "
                        + "rope=\(rope.cosine.dtype):\(rope.cosine.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.qkvLayout, "rope-unavailable")
            }
        }
        // The MLX-Serve artifact deinterleaves the released checkpoint's
        // per-head rows into three global Q/K/V slabs before quantization.
        // Do not apply the raw-checkpoint interleave a second time here.
        let projected = miniMaxH3SplitProjectedQKV(
            globalProjection, heads: heads, headDimension: headDimension
        )
        var query = queryNorm(projected[0])
        var key = keyNorm(projected[1])
        if let rope {
            query = rope.apply(query)
            key = rope.apply(key)
        }
        // Chunked SDPA reuses the complete K/V tensors for every query slice.
        // Materialize the transposed head-major views once so MLX does not
        // repack the same strided K/V storage inside every attention kernel.
        query = query.transposed(0, 2, 1, 3).contiguous()
        key = key.transposed(0, 2, 1, 3).contiguous()
        let value = projected[2].transposed(0, 2, 1, 3).contiguous()
        return [query, key, value]
    }

    func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        maximumQueryTokens: Int?,
        maximumHeadsPerKernel: Int?,
        maximumKernelsPerEvaluation: Int = 1
    ) -> MLXArray {
        if let maximumQueryTokens { precondition(maximumQueryTokens > 0) }
        if let maximumHeadsPerKernel { precondition(maximumHeadsPerKernel > 0) }
        let queryChunkSize = min(maximumQueryTokens ?? queries.dim(2), queries.dim(2))
        let headChunkSize = min(maximumHeadsPerKernel ?? queries.dim(1), queries.dim(1))
        guard queryChunkSize < queries.dim(2) || headChunkSize < queries.dim(1) else {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
        }

        precondition(maximumKernelsPerEvaluation > 0)
        var headOutputs: [MLXArray] = []
        headOutputs.reserveCapacity((queries.dim(1) + headChunkSize - 1) / headChunkSize)
        var pending: [MLXArray] = []
        pending.reserveCapacity(maximumKernelsPerEvaluation)
        for headStart in stride(from: 0, to: queries.dim(1), by: headChunkSize) {
            let headEnd = min(headStart + headChunkSize, queries.dim(1))
            var queryOutputs: [MLXArray] = []
            queryOutputs.reserveCapacity((queries.dim(2) + queryChunkSize - 1) / queryChunkSize)
            for queryStart in stride(from: 0, to: queries.dim(2), by: queryChunkSize) {
                let queryEnd = min(queryStart + queryChunkSize, queries.dim(2))
                let chunk = MLXFast.scaledDotProductAttention(
                    queries: queries[
                        0..., headStart..<headEnd, queryStart..<queryEnd, 0...
                    ],
                    keys: keys[0..., headStart..<headEnd, 0..., 0...],
                    values: values[0..., headStart..<headEnd, 0..., 0...],
                    scale: scale,
                    mask: .none
                )
                queryOutputs.append(chunk)
                pending.append(chunk)
                if pending.count == maximumKernelsPerEvaluation {
                    MLX.eval(pending)
                    pending.removeAll(keepingCapacity: true)
                }
            }
            headOutputs.append(MLX.concatenated(queryOutputs, axis: 2))
        }
        if !pending.isEmpty {
            MLX.eval(pending)
        }
        return MLX.concatenated(headOutputs, axis: 1)
    }

    func projectOutput(_ attended: MLXArray) -> MLXArray {
        if exactKernelMode == .affineQ8,
           enabledExactKernelStages.contains(.attentionOutput) {
            if let weights = miniMaxH3AffineQ8Weights(output) {
                if let projected = MiniMaxH3FusedKernels.projectHeadMajorAttentionAffineInt8(
                    attention: attended,
                    weightCodes: weights.codes,
                    weightScales: weights.scales,
                    weightBiases: weights.biases
                ) {
                    exactKernelDispatchHandler?(.attentionOutput)
                    return projected
                }
                exactKernelFallbackHandler?(
                    .attentionOutput,
                    "attention=\(attended.dtype):\(attended.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.attentionOutput, "weight-contract")
            }
        }
        let batch = attended.dim(0)
        let sequence = attended.dim(2)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, innerDimension))
    }
}

private final class MiniMaxH3FeedForward: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear
    var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled
    var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases)
    var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)?
    var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)?

    init(configuration: MiniMaxH3TransformerConfiguration) {
        self._input.wrappedValue = Linear(
            configuration.hiddenSize,
            2 * configuration.feedForwardSize,
            bias: false
        )
        self._output.wrappedValue = Linear(
            configuration.feedForwardSize,
            configuration.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        projectOutput(project(value))
    }

    func project(_ value: MLXArray) -> MLXArray {
        if exactKernelMode.usesAffineQ8FeedForward,
           enabledExactKernelStages.contains(.feedForwardInput) {
            if let weights = miniMaxH3AffineQ8Weights(input) {
                let projected = exactKernelMode.usesTiledAffineQ8FeedForward
                    ? MiniMaxH3FusedKernels.projectFeedForwardInputAffineInt8SwiGLUTiled(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                    : MiniMaxH3FusedKernels.projectFeedForwardInputAffineInt8SwiGLU(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                if let projected {
                    exactKernelDispatchHandler?(.feedForwardInput)
                    return projected
                }
                exactKernelFallbackHandler?(
                    .feedForwardInput,
                    "input=\(value.dtype):\(value.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.feedForwardInput, "weight-contract")
            }
        }
        let parts = MLX.split(input(value), parts: 2, axis: -1)
        return MLXNN.silu(parts[0]) * parts[1]
    }

    func projectOutput(_ value: MLXArray) -> MLXArray {
        if exactKernelMode.usesAffineQ8FeedForward,
           enabledExactKernelStages.contains(.feedForwardOutput) {
            if let weights = miniMaxH3AffineQ8Weights(output) {
                let projected = exactKernelMode.usesTiledAffineQ8FeedForward
                    ? MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8Tiled(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                    : MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8(
                        input: value,
                        weightCodes: weights.codes,
                        weightScales: weights.scales,
                        weightBiases: weights.biases
                    )
                if let projected {
                    exactKernelDispatchHandler?(.feedForwardOutput)
                    return projected
                }
                exactKernelFallbackHandler?(
                    .feedForwardOutput,
                    "input=\(value.dtype):\(value.shape)"
                )
            } else {
                exactKernelFallbackHandler?(.feedForwardOutput, "weight-contract")
            }
        }
        return output(value)
    }
}

private final class MiniMaxH3TokenRefinerBlock: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxH3Attention
    @ModuleInfo(key: "norm2") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "mlp") var feedForward: MiniMaxH3FeedForward

    init(configuration: MiniMaxH3TransformerConfiguration) {
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._attention.wrappedValue = MiniMaxH3Attention(configuration: configuration)
        self._feedForwardNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._feedForward.wrappedValue = MiniMaxH3FeedForward(configuration: configuration)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let attended = value + attention(attentionNorm(value), rope: nil)
        return attended + feedForward(feedForwardNorm(attended))
    }
}

private final class MiniMaxH3TokenRefiner: Module {
    @ModuleInfo(key: "blocks") var blocks: [MiniMaxH3TokenRefinerBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm

    init(configuration: MiniMaxH3TransformerConfiguration) {
        self._blocks.wrappedValue = (0..<configuration.refinerLayerCount).map { _ in
            MiniMaxH3TokenRefinerBlock(configuration: configuration)
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        finalNorm(blocks.reduce(value) { hidden, block in block(hidden) })
    }
}

private final class MiniMaxH3AdaLNProjection: Module {
    let hiddenSize: Int
    let partCount: Int
    let modalityCount: Int
    @ModuleInfo(key: "linear") var linear: Linear

    init(inputDimension: Int, hiddenSize: Int, partCount: Int, modalityCount: Int) {
        self.hiddenSize = hiddenSize
        self.partCount = partCount
        self.modalityCount = modalityCount
        self._linear.wrappedValue = Linear(
            inputDimension,
            partCount * modalityCount * hiddenSize,
            bias: true
        )
    }

    convenience init(discarded: Void) {
        self.init(inputDimension: 1, hiddenSize: 1, partCount: 1, modalityCount: 1)
    }

    func callAsFunction(_ timeEmbedding: MLXArray) -> [MLXArray] {
        let projected = miniMaxH3Linear(linear, MLXNN.silu(timeEmbedding))
            .reshaped(timeEmbedding.dim(0) * modalityCount, partCount * hiddenSize)
        return MLX.split(projected, parts: partCount, axis: -1)
    }

    func concatenated(_ timeEmbedding: MLXArray) -> MLXArray {
        MLX.concatenated(self(timeEmbedding), axis: -1)
    }
}

private final class MiniMaxH3TimeEmbedding: Module {
    @ModuleInfo(key: "proj_in") var input: Linear
    @ModuleInfo(key: "proj_out") var output: Linear

    init(configuration: MiniMaxH3TransformerConfiguration) {
        self._input.wrappedValue = Linear(
            configuration.timeFrequencyDimension,
            configuration.timeEmbeddingHiddenSize,
            bias: true
        )
        self._output.wrappedValue = Linear(
            configuration.timeEmbeddingHiddenSize,
            configuration.timeEmbeddingDimension,
            bias: true
        )
    }

    init(discarded: Void) {
        self._input.wrappedValue = Linear(1, 1, bias: false)
        self._output.wrappedValue = Linear(1, 1, bias: false)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        miniMaxH3Linear(output, MLXNN.silu(miniMaxH3Linear(input, value)))
    }
}

private final class MiniMaxH3TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxH3Attention
    @ModuleInfo(key: "norm2") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "mlp") var feedForward: MiniMaxH3FeedForward
    @ModuleInfo(key: "adaln_proj") var adaLN: MiniMaxH3AdaLNProjection?
    private var adaLNWeightsAvailable: Bool
    var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled {
        didSet {
            attention.exactKernelMode = exactKernelMode
            feedForward.exactKernelMode = exactKernelMode
        }
    }
    var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases) {
        didSet {
            attention.enabledExactKernelStages = enabledExactKernelStages
            feedForward.enabledExactKernelStages = enabledExactKernelStages
        }
    }
    var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)? {
        didSet {
            attention.exactKernelDispatchHandler = exactKernelDispatchHandler
            feedForward.exactKernelDispatchHandler = exactKernelDispatchHandler
        }
    }
    var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)? {
        didSet {
            attention.exactKernelFallbackHandler = exactKernelFallbackHandler
            feedForward.exactKernelFallbackHandler = exactKernelFallbackHandler
        }
    }

    init(configuration: MiniMaxH3TransformerConfiguration, includeAdaLN: Bool) {
        self.adaLNWeightsAvailable = includeAdaLN
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._attention.wrappedValue = MiniMaxH3Attention(configuration: configuration)
        self._feedForwardNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._feedForward.wrappedValue = MiniMaxH3FeedForward(configuration: configuration)
        self._adaLN.wrappedValue = includeAdaLN
            ? MiniMaxH3AdaLNProjection(
                inputDimension: configuration.timeEmbeddingDimension,
                hiddenSize: configuration.hiddenSize,
                partCount: 6,
                modalityCount: 3
            )
            : nil
    }

    var includesAdaLN: Bool {
        adaLNWeightsAvailable
    }

    var supportsAffineQ8ExactKernels: Bool {
        miniMaxH3AffineQ8Weights(attention.queryKeyValue) != nil
            && miniMaxH3AffineQ8Weights(attention.output) != nil
            && miniMaxH3AffineQ8Weights(feedForward.input) != nil
            && miniMaxH3AffineQ8Weights(feedForward.output) != nil
            && attention.queryNorm.weight.dtype == .bfloat16
            && attention.keyNorm.weight.dtype == .bfloat16
            && feedForwardNorm.weight.dtype == .bfloat16
    }

    #if DEBUG
    func feedForwardForBenchmark(_ value: MLXArray) -> MLXArray {
        feedForward(value)
    }

    func feedForwardInputForBenchmark(_ value: MLXArray) -> MLXArray {
        feedForward.project(value)
    }

    func feedForwardOutputForBenchmark(_ value: MLXArray) -> MLXArray {
        feedForward.projectOutput(value)
    }
    #endif

    func discardAdaLNWeights() {
        guard adaLNWeightsAvailable else { return }
        update(modules: ModuleChildren.unflattened([
            ("adaln_proj", MiniMaxH3AdaLNProjection(discarded: ())),
        ]))
        adaLNWeightsAvailable = false
    }

    private func prepareAttentionInput(
        _ value: MLXArray,
        modulation: MLXArray,
        adaLNIndices: MLXArray
    ) -> MLXArray {
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.attentionAdaLN) {
            if let prepared = MiniMaxH3FusedKernels.prepareAttentionInput(
                input: value,
                normWeight: attentionNorm.weight,
                modulation: modulation,
                rowIndices: adaLNIndices,
                eps: attentionNorm.eps
            ) {
                exactKernelDispatchHandler?(.attentionAdaLN)
                return prepared
            }
            exactKernelFallbackHandler?(
                .attentionAdaLN,
                "input=\(value.dtype):\(value.shape) "
                    + "norm=\(attentionNorm.weight.dtype) "
                    + "modulation=\(modulation.dtype):\(modulation.shape) "
                    + "indices=\(adaLNIndices.dtype):\(adaLNIndices.shape)"
            )
        }
        let parts = MLX.split(modulation, parts: 6, axis: -1)
        let shift = MLX.take(parts[0], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scale = MLX.take(parts[1], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        return attentionNorm(value) * (1 + scale) + shift
    }

    func callAsFunction(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.gateAdaLN),
           let exact = exactBoundaryCall(
               value,
               timeEmbedding: timeEmbedding,
               adaLNIndices: adaLNIndices,
               rope: rope,
               cachedModulation: cachedModulation
           ) {
            return exact
        }
        let attended = attentionResidual(
            value,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            rope: rope,
            cachedModulation: cachedModulation
        )
        return feedForwardResidual(
            attended,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            cachedModulation: cachedModulation
        )
    }

    private func exactBoundaryCall(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray? {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else {
                exactKernelFallbackHandler?(.gateAdaLN, "adaln-unavailable")
                return nil
            }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        let attentionOutput = attention(attentionInput, rope: rope)
        guard let boundary = MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
            residual: value,
            attentionOutput: attentionOutput,
            normWeight: feedForwardNorm.weight,
            modulation: completeModulation,
            rowIndices: adaLNIndices,
            eps: feedForwardNorm.eps
        ) else {
            exactKernelFallbackHandler?(
                .gateAdaLN,
                "residual=\(value.dtype):\(value.shape) "
                    + "attention=\(attentionOutput.dtype):\(attentionOutput.shape) "
                    + "norm=\(feedForwardNorm.weight.dtype) "
                    + "modulation=\(completeModulation.dtype):\(completeModulation.shape) "
                    + "indices=\(adaLNIndices.dtype):\(adaLNIndices.shape)"
            )
            return nil
        }
        exactKernelDispatchHandler?(.gateAdaLN)
        return boundary.residual
            + boundary.feedForwardGate * feedForward(boundary.feedForwardInput)
    }

    func attentionResidual(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let modulation = MLX.split(completeModulation, parts: 6, axis: -1)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        return value + gateAttention * attention(attentionInput, rope: rope)
    }

    func feedForwardResidual(
        _ attended: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 6, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let shiftFeedForward = MLX.take(modulation[3], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scaleFeedForward = MLX.take(modulation[4], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let gateFeedForward = MLX.take(modulation[5], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let feedForwardInput = feedForwardNorm(attended) * (1 + scaleFeedForward) + shiftFeedForward
        return attended + gateFeedForward * feedForward(feedForwardInput)
    }

    func attentionProjection(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> [MLXArray] {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let modulation = MLX.split(completeModulation, parts: 6, axis: -1)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        return attention.project(attentionInput, rope: rope) + [gateAttention]
    }

    func fastH3AttentionProjection(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?,
        compressionGate: MiniMaxH3FastH3CompressionGate
    ) -> [MLXArray] {
        fastH3AttentionProjection(
            value,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            rope: rope,
            cachedModulation: cachedModulation,
            compressionGateStorage: compressionGate.storage,
            compressionGateParameters: compressionGate.parameters
        )
    }

    func fastH3AttentionProjection(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?,
        compressionGateStorage: MiniMaxH3FastH3CompressionGate.Storage,
        compressionGateParameters: [MLXArray]
    ) -> [MLXArray] {
        let completeModulation: MLXArray
        if let cachedModulation {
            completeModulation = cachedModulation
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            completeModulation = adaLN.concatenated(timeEmbedding)
        }
        let modulation = MLX.split(completeModulation, parts: 6, axis: -1)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = prepareAttentionInput(
            value,
            modulation: completeModulation,
            adaLNIndices: adaLNIndices
        )
        let projected = attention.project(attentionInput, rope: rope)
        let compressed = MiniMaxH3FastH3CompressionGate.project(
            attentionInput,
            storage: compressionGateStorage,
            parameters: compressionGateParameters
        ).reshaped(
            attentionInput.dim(0),
            attentionInput.dim(1),
            attention.heads,
            attention.headDimension
        ).transposed(0, 2, 1, 3).contiguous()
        return projected + [gateAttention, compressed]
    }

    func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        maximumQueryTokens: Int,
        maximumHeadsPerKernel: Int?,
        maximumKernelsPerEvaluation: Int,
        dynamicSparseRequest: DynamicSparseAttentionRequest? = nil
    ) -> MLXArray {
        if let dynamicSparseRequest,
           let sparse = DynamicSparseAttention.call(
               queries: queries,
               keys: keys,
               values: values,
               request: dynamicSparseRequest,
               scale: attention.scale,
               maximumQueryTokens: maximumQueryTokens,
               maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
           ) {
            return sparse
        }
        return attention.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            maximumQueryTokens: maximumQueryTokens,
            maximumHeadsPerKernel: maximumHeadsPerKernel,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
        )
    }

    func attentionProjectionResidual(
        _ value: MLXArray,
        attended: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        value + gate * attention.projectOutput(attended)
    }

    func postAttention(
        _ value: MLXArray,
        attended: MLXArray,
        gate: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let projected = postAttentionProjection(
            value,
            attended: attended,
            gate: gate,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            cachedModulation: cachedModulation
        )
        return feedForwardProjectionResidual(
            projected[0],
            projected: projected[1],
            gate: projected[2]
        )
    }

    func postAttentionProjection(
        _ value: MLXArray,
        attended: MLXArray,
        gate: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> [MLXArray] {
        let attentionOutput = attention.projectOutput(attended)
        if exactKernelMode.usesBoundaryLayout,
           enabledExactKernelStages.contains(.gateAdaLN) {
            let completeModulation: MLXArray
            if let cachedModulation {
                completeModulation = cachedModulation
            } else {
                guard let adaLN else {
                    preconditionFailure("MiniMax-H3 AdaLN cache is required")
                }
                completeModulation = adaLN.concatenated(timeEmbedding)
            }
            if let boundary = MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                residual: value,
                attentionOutput: attentionOutput,
                normWeight: feedForwardNorm.weight,
                modulation: completeModulation,
                rowIndices: adaLNIndices,
                eps: feedForwardNorm.eps
            ) {
                exactKernelDispatchHandler?(.gateAdaLN)
                return [
                    boundary.residual,
                    feedForward.project(boundary.feedForwardInput),
                    boundary.feedForwardGate,
                ]
            }
            exactKernelFallbackHandler?(
                .gateAdaLN,
                "residual=\(value.dtype):\(value.shape) "
                    + "attention=\(attentionOutput.dtype):\(attentionOutput.shape) "
                    + "norm=\(feedForwardNorm.weight.dtype) "
                    + "modulation=\(completeModulation.dtype):\(completeModulation.shape) "
                    + "indices=\(adaLNIndices.dtype):\(adaLNIndices.shape)"
            )
        }
        let attentionResidual = value + gate * attentionOutput
        let feedForwardParts = feedForwardProjection(
            attentionResidual,
            timeEmbedding: timeEmbedding,
            adaLNIndices: adaLNIndices,
            cachedModulation: cachedModulation
        )
        return [
            attentionResidual,
            feedForwardParts[0],
            feedForwardParts[1],
        ]
    }

    func feedForwardProjection(
        _ attended: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        cachedModulation: MLXArray?
    ) -> [MLXArray] {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 6, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let shiftFeedForward = MLX.take(modulation[3], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scaleFeedForward = MLX.take(modulation[4], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let gateFeedForward = MLX.take(modulation[5], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let feedForwardInput = feedForwardNorm(attended) * (1 + scaleFeedForward) + shiftFeedForward
        return [feedForward.project(feedForwardInput), gateFeedForward]
    }

    func feedForwardProjectionResidual(
        _ attended: MLXArray,
        projected: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        attended + gate * feedForward.projectOutput(projected)
    }

    func precomputeModulation(timeEmbedding: MLXArray) -> MLXArray {
        guard adaLNWeightsAvailable, let adaLN else {
            preconditionFailure("MiniMax-H3 AdaLN weights are not loaded")
        }
        return adaLN.concatenated(timeEmbedding)
    }
}

private final class MiniMaxH3FinalLayer: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "adaln_proj") var adaLN: MiniMaxH3AdaLNProjection?
    @ModuleInfo(key: "video_out") var videoOutput: Linear
    @ModuleInfo(key: "audio_out") var audioOutput: Linear

    init(configuration: MiniMaxH3TransformerConfiguration, includeAdaLN: Bool) {
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._adaLN.wrappedValue = includeAdaLN
            ? MiniMaxH3AdaLNProjection(
                inputDimension: configuration.timeEmbeddingDimension,
                hiddenSize: configuration.hiddenSize,
                partCount: 2,
                modalityCount: 1
            )
            : nil
        self._videoOutput.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.videoPatchDimension,
            bias: true
        )
        self._audioOutput.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.audioLatentChannels,
            bias: true
        )
    }

    func discardAdaLNWeights() {
        guard adaLN != nil else { return }
        update(modules: ModuleChildren.unflattened([
            ("adaln_proj", MiniMaxH3AdaLNProjection(discarded: ())),
        ]))
    }

    func callAsFunction(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        videoRows: Range<Int>,
        videoTimeIndex: Int32,
        audioRows: Range<Int>,
        audioTimeIndex: Int32,
        cachedModulation: MLXArray?
    ) -> MiniMaxH3TransformerOutput {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 2, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let normalizedVideo = norm(value[0..., videoRows, 0...])
        let videoIndex = MLXArray([videoTimeIndex])
        let videoHidden = normalizedVideo * (1 + MLX.take(modulation[1], videoIndex, axis: 0))
            + MLX.take(modulation[0], videoIndex, axis: 0)
        let normalizedAudio = norm(value[0..., audioRows, 0...])
        let audioIndex = MLXArray([audioTimeIndex])
        let audioHidden = normalizedAudio * (1 + MLX.take(modulation[1], audioIndex, axis: 0))
            + MLX.take(modulation[0], audioIndex, axis: 0)
        return MiniMaxH3TransformerOutput(
            videoVelocityRows: miniMaxH3Linear(videoOutput, videoHidden),
            audioVelocityRows: miniMaxH3Linear(audioOutput, audioHidden)
        )
    }

    func precomputeModulation(timeEmbedding: MLXArray) -> MLXArray {
        guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN weights are not loaded") }
        return adaLN.concatenated(timeEmbedding)
    }
}

struct MiniMaxH3TransformerPreparedContext {
    let text: MLXArray
    let adaLNIndices: MLXArray
    let rope: MiniMaxH3RotaryEmbedding
    let layout: MiniMaxH3PackedLayout
    let fastVSA: MiniMaxH3FastVSAPreparedContext?
}

struct MiniMaxH3TokenReductionState {
    let reducedHidden: MLXArray
    let originalVideo: MLXArray
    let pooledBaseline: MLXArray
}

struct MiniMaxH3TokenReductionMap {
    let targetVideoStart: Int
    let reducedVideoRowCount: Int
    let firstVideoIndices: MLXArray
    let secondVideoIndices: MLXArray
    let parentVideoIndices: MLXArray
    let absoluteSourceIndices: MLXArray

    init(layout: MiniMaxH3PackedLayout) {
        precondition(layout.targetVideoRows.upperBound == layout.sequenceLength)
        let spatialHeight = layout.latentHeight / 2
        let spatialWidth = layout.latentWidth / 2
        let reducedWidth = (spatialWidth + 1) / 2
        precondition(spatialHeight > 0 && spatialWidth > 0)
        precondition(
            layout.targetVideoRows.count
                == layout.videoLatentFrames * spatialHeight * spatialWidth
        )

        var first: [Int32] = []
        var second: [Int32] = []
        first.reserveCapacity(layout.videoLatentFrames * spatialHeight * reducedWidth)
        second.reserveCapacity(first.capacity)
        for temporal in 0..<layout.videoLatentFrames {
            for height in 0..<spatialHeight {
                let rowStart = (temporal * spatialHeight + height) * spatialWidth
                for reducedColumn in 0..<reducedWidth {
                    let firstIndex = rowStart + reducedColumn * 2
                    first.append(Int32(firstIndex))
                    second.append(Int32(min(firstIndex + 1, rowStart + spatialWidth - 1)))
                }
            }
        }
        var parents: [Int32] = []
        parents.reserveCapacity(layout.targetVideoRows.count)
        for temporal in 0..<layout.videoLatentFrames {
            for height in 0..<spatialHeight {
                let reducedRowStart = (temporal * spatialHeight + height) * reducedWidth
                for column in 0..<spatialWidth {
                    parents.append(Int32(reducedRowStart + column / 2))
                }
            }
        }
        let prefix = (0..<layout.targetVideoRows.lowerBound).map(Int32.init)
        let video = first.map { Int32(layout.targetVideoRows.lowerBound) + $0 }
        self.targetVideoStart = layout.targetVideoRows.lowerBound
        self.reducedVideoRowCount = first.count
        self.firstVideoIndices = MLXArray(first)
        self.secondVideoIndices = MLXArray(second)
        self.parentVideoIndices = MLXArray(parents)
        self.absoluteSourceIndices = MLXArray(prefix + video)
    }

    func pool(_ hidden: MLXArray) -> MiniMaxH3TokenReductionState {
        precondition(hidden.ndim == 3)
        let originalVideo = hidden[0..., targetVideoStart..., 0...]
        let first = MLX.take(originalVideo, firstVideoIndices, axis: 1).asType(.float32)
        let second = MLX.take(originalVideo, secondVideoIndices, axis: 1).asType(.float32)
        let pooled = ((first + second) * 0.5).asType(hidden.dtype)
        let reduced = targetVideoStart == 0
            ? pooled
            : MLX.concatenated([hidden[0..., 0..<targetVideoStart, 0...], pooled], axis: 1)
        return MiniMaxH3TokenReductionState(
            reducedHidden: reduced,
            originalVideo: originalVideo,
            pooledBaseline: pooled
        )
    }

    func restore(
        _ reducedHidden: MLXArray,
        state: MiniMaxH3TokenReductionState,
        updateScale: Float
    ) -> MLXArray {
        precondition(reducedHidden.dim(1) == targetVideoStart + reducedVideoRowCount)
        let reducedVideo = reducedHidden[0..., targetVideoStart..., 0...]
        let current = MLX.take(reducedVideo, parentVideoIndices, axis: 1).asType(.float32)
        let baseline = MLX.take(
            state.pooledBaseline,
            parentVideoIndices,
            axis: 1
        ).asType(.float32)
        let restoredVideo = (
            state.originalVideo.asType(.float32) + updateScale * (current - baseline)
        ).asType(reducedHidden.dtype)
        return targetVideoStart == 0
            ? restoredVideo
            : MLX.concatenated([
                reducedHidden[0..., 0..<targetVideoStart, 0...],
                restoredVideo,
            ], axis: 1)
    }
}

struct MiniMaxH3TokenReductionPreparedContext {
    let map: MiniMaxH3TokenReductionMap
    let reducedContext: MiniMaxH3TransformerPreparedContext
}

struct MiniMaxH3AdaLNStep {
    let timeEmbedding: MLXArray
    let blockModulations: [MLXArray]
    let finalModulation: MLXArray
}

private typealias MiniMaxH3CompiledBlockForward = @Sendable ([MLXArray]) -> [MLXArray]

private struct MiniMaxH3CompiledBlockForwards {
    let attentionProjection: MiniMaxH3CompiledBlockForward
    let fastH3AttentionProjection: MiniMaxH3CompiledBlockForward?
    let attentionOutput: MiniMaxH3CompiledBlockForward
    let feedForwardProjection: MiniMaxH3CompiledBlockForward
    let feedForwardOutput: MiniMaxH3CompiledBlockForward
    let postAttentionProjection: MiniMaxH3CompiledBlockForward
    let postAttention: MiniMaxH3CompiledBlockForward
}

#if DEBUG
/// Test-only harness for timing the exact production H3 block schedules at
/// realistic packed-row counts without loading the complete checkpoint.
final class MiniMaxH3BlockScheduleBenchmark {
    enum Schedule {
        case splitPostAttention
        case fusedFeedForward
        case fusedPostAttention
    }

    let rowCount: Int
    private let maximumQueryTokens: Int
    private let maximumHeadsPerKernel: Int?
    private let maximumKernelsPerEvaluation: Int
    private let block: MiniMaxH3TransformerBlock
    private let originalParameters: [(String, MLXArray)]
    private let forwards: MiniMaxH3CompiledBlockForwards
    private let fusedFeedForward: MiniMaxH3CompiledBlockForward
    private let hidden: MLXArray
    private let timeEmbedding: MLXArray
    private let adaLNIndices: MLXArray
    private let rope: MiniMaxH3RotaryEmbedding
    private let cachedModulation: MLXArray

    init(
        rowCount: Int,
        maximumQueryTokens: Int,
        maximumKernelsPerEvaluation: Int,
        maximumHeadsPerKernel: Int? = nil,
        dtype: DType = .bfloat16,
        weightSeed: UInt64? = nil,
        inputSeed: UInt64? = nil
    ) {
        precondition(rowCount > 0)
        precondition(maximumQueryTokens > 0)
        precondition(maximumKernelsPerEvaluation > 0)
        if let maximumHeadsPerKernel { precondition(maximumHeadsPerKernel > 0) }
        self.rowCount = rowCount
        self.maximumQueryTokens = maximumQueryTokens
        self.maximumHeadsPerKernel = maximumHeadsPerKernel
        self.maximumKernelsPerEvaluation = maximumKernelsPerEvaluation

        let configuration = MiniMaxH3TransformerConfiguration()
        if let weightSeed { MLXRandom.seed(weightSeed) }
        let block = MiniMaxH3TransformerBlock(configuration: configuration, includeAdaLN: false)
        block.update(parameters: block.parameters().mapValues { $0.asType(dtype) })
        MLX.eval(block.parameters())
        self.block = block
        self.originalParameters = block.parameters().flattened()

        let attentionProjection = MLX.compile(inputs: [block]) { inputs in
            block.attentionProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                cachedModulation: inputs[5]
            )
        }
        let attentionOutput = MLX.compile(inputs: [block]) { inputs in
            [block.attentionProjectionResidual(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2]
            )]
        }
        let feedForwardProjection = MLX.compile(inputs: [block]) { inputs in
            block.feedForwardProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs[3]
            )
        }
        let feedForwardOutput = MLX.compile(inputs: [block]) { inputs in
            [block.feedForwardProjectionResidual(
                inputs[0],
                projected: inputs[1],
                gate: inputs[2]
            )]
        }
        let feedForward = MLX.compile(inputs: [block]) { inputs in
            [block.feedForwardResidual(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs[3]
            )]
        }
        let postAttention = MLX.compile(inputs: [block]) { inputs in
            [block.postAttention(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs[5]
            )]
        }
        let postAttentionProjection = MLX.compile(inputs: [block]) { inputs in
            block.postAttentionProjection(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs[5]
            )
        }
        self.forwards = MiniMaxH3CompiledBlockForwards(
            attentionProjection: attentionProjection,
            fastH3AttentionProjection: nil,
            attentionOutput: attentionOutput,
            feedForwardProjection: feedForwardProjection,
            feedForwardOutput: feedForwardOutput,
            postAttentionProjection: postAttentionProjection,
            postAttention: postAttention
        )
        self.fusedFeedForward = feedForward

        if let inputSeed { MLXRandom.seed(inputSeed) }
        self.hidden = MLXRandom.normal([1, rowCount, configuration.hiddenSize])
            .asType(dtype)
        self.timeEmbedding = MLXArray.zeros(
            [3, configuration.timeEmbeddingDimension],
            dtype: dtype
        )
        self.adaLNIndices = MLXArray((0..<rowCount).map { Int32($0 % 9) })
        self.rope = MiniMaxH3RotaryEmbedding(
            cosine: MLXArray.ones(
                [1, rowCount, 1, 6 * configuration.ropeFrequencyCount],
                dtype: dtype
            ),
            sine: MLXArray.zeros(
                [1, rowCount, 1, 6 * configuration.ropeFrequencyCount],
                dtype: dtype
            )
        )
        self.cachedModulation = (
            MLXRandom.normal([9, 6 * configuration.hiddenSize]) * Float(0.1)
        ).asType(dtype)
        MLX.eval(
            hidden,
            timeEmbedding,
            adaLNIndices,
            rope.cosine,
            rope.sine,
            cachedModulation
        )
    }

    func useOriginalWeights(from source: MiniMaxH3BlockScheduleBenchmark) {
        block.update(parameters: ModuleParameters.unflattened(source.originalParameters))
    }

    var defaultInput: MLXArray { hidden }

    func callAsFunction(
        schedule: Schedule,
        maximumQueryTokens: Int? = nil,
        maximumHeadsPerKernel: Int? = nil,
        maximumKernelsPerEvaluation: Int? = nil,
        input: MLXArray? = nil
    ) -> MLXArray {
        let hidden = input ?? self.hidden
        let projectedAttention = projectAttention(input: hidden)
        MLX.eval(projectedAttention)
        let attended = attend(
            projectedAttention,
            maximumQueryTokens: maximumQueryTokens,
            maximumHeadsPerKernel: maximumHeadsPerKernel,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
        )
        return postAttention(
            schedule: schedule,
            attended: attended,
            gate: projectedAttention[3],
            input: hidden
        )
    }

    func projectAttention(input: MLXArray? = nil) -> [MLXArray] {
        forwards.attentionProjection([
            input ?? hidden,
            timeEmbedding,
            adaLNIndices,
            rope.cosine,
            rope.sine,
            cachedModulation,
        ])
    }

    func attend(
        _ projectedAttention: [MLXArray],
        maximumQueryTokens: Int? = nil,
        maximumHeadsPerKernel: Int? = nil,
        maximumKernelsPerEvaluation: Int? = nil
    ) -> MLXArray {
        block.scaledDotProductAttention(
            queries: projectedAttention[0],
            keys: projectedAttention[1],
            values: projectedAttention[2],
            maximumQueryTokens: maximumQueryTokens ?? self.maximumQueryTokens,
            maximumHeadsPerKernel: maximumHeadsPerKernel ?? self.maximumHeadsPerKernel,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
                ?? self.maximumKernelsPerEvaluation
        )
    }

    func attentionOutput(
        input: MLXArray,
        attended: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        let output = forwards.attentionOutput([input, attended, gate])[0]
        MLX.eval(output)
        return output
    }

    func splitFeedForward(input: MLXArray) -> MLXArray {
        let projected = forwards.feedForwardProjection([
            input,
            timeEmbedding,
            adaLNIndices,
            cachedModulation,
        ])
        MLX.eval(projected)
        let output = forwards.feedForwardOutput([input, projected[0], projected[1]])[0]
        MLX.eval(output)
        return output
    }

    func makeFusedBoundary(
        to successor: MiniMaxH3BlockScheduleBenchmark
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        MLX.compile(inputs: [block, successor.block]) { inputs in
            let nextHidden = self.block.feedForwardResidual(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs[3]
            )
            return [nextHidden] + successor.block.attentionProjection(
                nextHidden,
                timeEmbedding: inputs[4],
                adaLNIndices: inputs[5],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[6], sine: inputs[7]),
                cachedModulation: inputs[8]
            )
        }
    }

    func fusedBoundaryInputs(
        to successor: MiniMaxH3BlockScheduleBenchmark,
        input: MLXArray
    ) -> [MLXArray] {
        [
            input,
            timeEmbedding,
            adaLNIndices,
            cachedModulation,
            successor.timeEmbedding,
            successor.adaLNIndices,
            successor.rope.cosine,
            successor.rope.sine,
            successor.cachedModulation,
        ]
    }

    func postAttention(
        schedule: Schedule,
        attended: MLXArray,
        gate: MLXArray,
        input: MLXArray? = nil
    ) -> MLXArray {
        let hidden = input ?? self.hidden
        switch schedule {
        case .splitPostAttention:
            let attendedHidden = attentionOutput(input: hidden, attended: attended, gate: gate)
            return splitFeedForward(input: attendedHidden)
        case .fusedFeedForward:
            let attentionOutput = forwards.attentionOutput([
                hidden,
                attended,
                gate,
            ])[0]
            MLX.eval(attentionOutput)
            let output = fusedFeedForward([
                attentionOutput,
                timeEmbedding,
                adaLNIndices,
                cachedModulation,
            ])[0]
            MLX.eval(output)
            return output
        case .fusedPostAttention:
            let output = forwards.postAttention([
                hidden,
                attended,
                gate,
                timeEmbedding,
                adaLNIndices,
                cachedModulation,
            ])[0]
            MLX.eval(output)
            return output
        }
    }
}
#endif

struct MiniMaxH3ResidentBF16Materialization: Sendable, Equatable {
    let linearCount: Int
    let byteCount: UInt64
}

private func miniMaxH3ResidentBF16Linear(
    _ linear: Linear
) -> (linear: Linear, byteCount: UInt64)? {
    guard let quantized = linear as? QuantizedLinear else { return nil }
    var weight = MLX.dequantized(
        quantized.weight,
        scales: quantized.scales,
        biases: quantized.biases,
        groupSize: quantized.groupSize,
        bits: quantized.bits,
        mode: quantized.mode,
        dtype: .bfloat16
    )
    if let globalScale = quantized.globalScale {
        weight = (weight * (globalScale / (448 * 6))).asType(.bfloat16)
    }
    let bias = quantized.bias?.asType(.bfloat16)
    if let bias {
        MLX.eval(weight, bias)
    } else {
        MLX.eval(weight)
    }
    let shape = quantized.shape
    let weightBytes = UInt64(shape.0) * UInt64(shape.1) * 2
    let biasBytes = UInt64(quantized.bias?.size ?? 0) * 2
    let base = Linear(weight: weight, bias: bias)
    if let lora = quantized as? MiniMaxH3RuntimeQuantizedLoRALinear {
        let adapterBytes = UInt64(lora.loraDown.size + lora.loraUp.size) * 2
        return (
            MiniMaxH3RuntimeLoRALinear(
                base: base,
                loraDown: lora.loraDown,
                loraUp: lora.loraUp,
                strength: lora.strength
            ),
            weightBytes + biasBytes + adapterBytes
        )
    }
    if let lora = quantized as? MiniMaxH3RuntimeQuantizedQKVLoRALinear {
        let adapterCount = lora.queryDown.size + lora.queryUp.size
            + lora.keyDown.size + lora.keyUp.size
            + lora.valueDown.size + lora.valueUp.size
        return (
            MiniMaxH3RuntimeQKVLoRALinear(
                base: base,
                queryDown: lora.queryDown,
                queryUp: lora.queryUp,
                keyDown: lora.keyDown,
                keyUp: lora.keyUp,
                valueDown: lora.valueDown,
                valueUp: lora.valueUp,
                strength: lora.strength
            ),
            weightBytes + biasBytes + UInt64(adapterCount) * 2
        )
    }
    return (base, weightBytes + biasBytes)
}

private func miniMaxH3MaterializeResidentBF16(
    in module: Module
) -> MiniMaxH3ResidentBF16Materialization {
    var replacements: [(String, Module)] = []
    var byteCount: UInt64 = 0
    for (path, leaf) in module.leafModules().flattened() {
        guard let linear = leaf as? Linear,
              let resident = miniMaxH3ResidentBF16Linear(linear) else { continue }
        replacements.append((path, resident.linear))
        byteCount += resident.byteCount
    }
    if !replacements.isEmpty {
        module.update(modules: ModuleChildren.unflattened(replacements))
    }
    return .init(linearCount: replacements.count, byteCount: byteCount)
}

private func miniMaxH3ResidentBF16ByteCount(_ linear: Linear) -> UInt64 {
    if let quantized = linear as? QuantizedLinear {
        let shape = quantized.shape
        var byteCount = UInt64(shape.0) * UInt64(shape.1) * 2
            + UInt64(quantized.bias?.size ?? 0) * 2
        if let lora = quantized as? MiniMaxH3RuntimeQuantizedLoRALinear {
            byteCount += UInt64(lora.loraDown.size + lora.loraUp.size) * 2
        } else if let lora = quantized as? MiniMaxH3RuntimeQuantizedQKVLoRALinear {
            byteCount += UInt64(
                lora.queryDown.size + lora.queryUp.size
                    + lora.keyDown.size + lora.keyUp.size
                    + lora.valueDown.size + lora.valueUp.size
            ) * 2
        }
        return byteCount
    }
    return linear.parameters().flattened().reduce(into: UInt64(0)) { total, entry in
        total += UInt64(entry.1.size) * 2
    }
}

private func miniMaxH3EvaluateParameters(in module: Module) {
    let parameters = module.parameters().flattened().map(\.1)
    guard !parameters.isEmpty else { return }
    MLX.eval(parameters)
}

struct MiniMaxH3RotaryEmbedding {
    let cosine: MLXArray
    let sine: MLXArray

    func apply(_ value: MLXArray) -> MLXArray {
        let rotaryDimension = cosine.dim(-1)
        let rotary = value[0..., 0..., 0..., 0..<rotaryDimension]
        let passthrough = value[0..., 0..., 0..., rotaryDimension...]
        let halves = MLX.split(rotary, parts: 2, axis: -1)
        let rotated = MLX.concatenated([-halves[1], halves[0]], axis: -1)
        return MLX.concatenated(
            [rotary * cosine + rotated * sine, passthrough],
            axis: -1
        )
    }
}

public final class MiniMaxH3Transformer: Module {
    public let configuration: MiniMaxH3TransformerConfiguration
    var usesLayerwiseEvaluation = false
    var clearsCacheAfterLayerwiseEvaluation = true
    var usesBlockwiseCompilation = false
    var usesFusedPostAttention = true
    var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled {
        didSet {
            for block in blocks {
                block.exactKernelMode = exactKernelMode
            }
            compiledBlockRunner = nil
            compiledBlockForwards = nil
        }
    }
    var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases) {
        didSet {
            for block in blocks {
                block.enabledExactKernelStages = enabledExactKernelStages
            }
            compiledBlockRunner = nil
            compiledBlockForwards = nil
        }
    }
    var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)? {
        didSet {
            for block in blocks {
                block.exactKernelDispatchHandler = exactKernelDispatchHandler
            }
        }
    }
    var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)? {
        didSet {
            for block in blocks {
                block.exactKernelFallbackHandler = exactKernelFallbackHandler
            }
        }
    }
    private(set) var usesResidentBF16 = false
    var maximumAttentionQueryTokensPerKernel = 1_024
    var maximumAttentionHeadsPerKernel: Int?
    var maximumAttentionKernelsPerEvaluation = 4
    var dynamicSparseAttentionPolicy: DynamicSparseAttentionPolicy?
    var dynamicSparseAttentionStepIndex = 0
    var dynamicSparseAttentionStepCount = 0
    var dynamicSparseAttentionLogHandler: ((String) -> Void)?
    var blockTimingHandler: ((Int, TimeInterval, Memory.Snapshot) -> Void)?
    var fastH3BlockPhaseTimingHandler: ((Int, TimeInterval, TimeInterval, TimeInterval) -> Void)?
    var activeBlockIndices: Set<Int>?
    @ModuleInfo(key: "video_patch_proj") var videoInput: Linear
    @ModuleInfo(key: "audio_patch_proj") var audioInput: Linear
    @ModuleInfo(key: "condition_proj") var textInput: Linear
    @ModuleInfo(key: "time_embedder") private var timeEmbedder: MiniMaxH3TimeEmbedding?
    @ModuleInfo(key: "token_refiner") private var tokenRefiner: MiniMaxH3TokenRefiner
    @ModuleInfo(key: "blocks") private var blocks: [MiniMaxH3TransformerBlock]
    @ModuleInfo(key: "final_layer") private var finalLayer: MiniMaxH3FinalLayer

    private let inverseFrequencies: MLXArray
    private let timeFrequencies: MLXArray
    private var compiledBlockRunner: MiniMaxH3TransformerBlock?
    private var compiledBlockForwards: MiniMaxH3CompiledBlockForwards?
    private var dynamicSparseAttentionGateResults: [String: Bool] = [:]
    private var fastH3CompressionGates: [MiniMaxH3FastH3CompressionGate?]

    var supportsAffineQ8ExactKernels: Bool {
        !blocks.isEmpty && blocks.allSatisfy(\.supportsAffineQ8ExactKernels)
    }

    static func requiresTiledFeedForwardEvaluationBoundary(
        rowCount: Int,
        feedForwardSize: Int,
        itemSize: Int
    ) -> Bool {
        precondition(rowCount > 0 && feedForwardSize > 0 && itemSize > 0)
        return UInt64(rowCount)
            * UInt64(feedForwardSize)
            * UInt64(itemSize) > UInt64(UInt32.max)
    }

    #if DEBUG
    func feedForwardForBenchmark(_ value: MLXArray, blockIndex: Int) -> MLXArray {
        precondition(blocks.indices.contains(blockIndex))
        return blocks[blockIndex].feedForwardForBenchmark(value)
    }

    func feedForwardInputForBenchmark(_ value: MLXArray, blockIndex: Int) -> MLXArray {
        precondition(blocks.indices.contains(blockIndex))
        return blocks[blockIndex].feedForwardInputForBenchmark(value)
    }

    func feedForwardOutputForBenchmark(_ value: MLXArray, blockIndex: Int) -> MLXArray {
        precondition(blocks.indices.contains(blockIndex))
        return blocks[blockIndex].feedForwardOutputForBenchmark(value)
    }
    #endif

    var affineQ8ExactKernelBlockCount: Int {
        blocks.count(where: \.supportsAffineQ8ExactKernels)
    }
    private var adaLNWeightsAvailable: Bool

    public init(
        configuration: MiniMaxH3TransformerConfiguration = .init(),
        includeAdaLN: Bool = true
    ) {
        self.adaLNWeightsAvailable = includeAdaLN
        self.configuration = configuration
        self.fastH3CompressionGates = Array(repeating: nil, count: configuration.layerCount)
        self._videoInput.wrappedValue = Linear(
            configuration.videoPatchDimension,
            configuration.hiddenSize,
            bias: true
        )
        self._audioInput.wrappedValue = Linear(
            configuration.audioLatentChannels,
            configuration.hiddenSize,
            bias: true
        )
        self._textInput.wrappedValue = Linear(
            configuration.textDimension,
            configuration.hiddenSize,
            bias: true
        )
        self._timeEmbedder.wrappedValue = includeAdaLN
            ? MiniMaxH3TimeEmbedding(configuration: configuration)
            : nil
        self._tokenRefiner.wrappedValue = MiniMaxH3TokenRefiner(configuration: configuration)
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            MiniMaxH3TransformerBlock(configuration: configuration, includeAdaLN: includeAdaLN)
        }
        self._finalLayer.wrappedValue = MiniMaxH3FinalLayer(
            configuration: configuration,
            includeAdaLN: includeAdaLN
        )
        let frequencies = (0..<configuration.ropeFrequencyCount).map { index in
            1 / pow(configuration.ropeTheta, Float(index) / Float(configuration.ropeFrequencyCount))
        }
        self.inverseFrequencies = MLXArray(frequencies)
        let halfTimeDimension = configuration.timeFrequencyDimension / 2
        self.timeFrequencies = MLXArray((0..<halfTimeDimension).map { index in
            exp(-log(Float(10_000)) * Float(index) / Float(halfTimeDimension))
        })
    }

    var estimatedResidentBF16ByteCount: UInt64 {
        leafModules().flattened().reduce(into: UInt64(0)) { total, entry in
            guard let linear = entry.1 as? Linear else { return }
            total += miniMaxH3ResidentBF16ByteCount(linear)
        }
    }

    var activeBlockCount: Int {
        activeBlockIndices?.count ?? blocks.count
    }

    var usesFastH3VSA: Bool {
        !fastH3CompressionGates.isEmpty
            && fastH3CompressionGates.allSatisfy { $0 != nil }
    }

    func installFastH3CompressionGate(_ weight: MLXArray, blockIndex: Int) {
        precondition(fastH3CompressionGates.indices.contains(blockIndex))
        precondition(weight.shape == [
            configuration.attentionHeadCount * configuration.attentionHeadDimension,
            configuration.hiddenSize,
        ])
        fastH3CompressionGates[blockIndex] = MiniMaxH3FastH3CompressionGate(weight: weight)
        MLX.eval(weight)
        compiledBlockRunner = nil
        compiledBlockForwards = nil
    }

    func installFastH3QuantizedCompressionGate(
        codes: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int,
        blockIndex: Int
    ) {
        precondition(fastH3CompressionGates.indices.contains(blockIndex))
        let outputDimension = configuration.attentionHeadCount
            * configuration.attentionHeadDimension
        precondition(bits == 8 && groupSize == 64)
        precondition(codes.shape == [outputDimension, configuration.hiddenSize * bits / 32])
        precondition(scales.shape == [outputDimension, configuration.hiddenSize / groupSize])
        precondition(biases.shape == scales.shape)
        fastH3CompressionGates[blockIndex] = MiniMaxH3FastH3CompressionGate(
            codes: codes,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits
        )
        MLX.eval(codes, scales, biases)
        compiledBlockRunner = nil
        compiledBlockForwards = nil
    }

    /// Expands compact quantized storage into resident bf16 linear weights.
    /// Each transformer/refiner block is replaced and the Metal cache is
    /// cleared before moving to the next one, bounding conversion residency
    /// instead of retaining a second whole-model copy.
    func materializeResidentBF16() -> MiniMaxH3ResidentBF16Materialization {
        compiledBlockRunner = nil
        compiledBlockForwards = nil

        func materialize(_ module: Module) {
            _ = miniMaxH3MaterializeResidentBF16(in: module)
            miniMaxH3EvaluateParameters(in: module)
            MLX.Memory.clearCache()
        }

        for block in tokenRefiner.blocks {
            materialize(block)
        }
        for block in blocks {
            materialize(block)
        }
        if let timeEmbedder {
            materialize(timeEmbedder)
        }
        materialize(finalLayer)

        var rootReplacements: [(String, Module)] = []
        for (path, linear) in [
            ("video_patch_proj", videoInput),
            ("audio_patch_proj", audioInput),
            ("condition_proj", textInput),
        ] {
            guard let resident = miniMaxH3ResidentBF16Linear(linear) else { continue }
            rootReplacements.append((path, resident.linear))
        }
        if !rootReplacements.isEmpty {
            update(modules: ModuleChildren.unflattened(rootReplacements))
        }

        // Sharded BF16 checkpoints already contain dense Linear modules, so
        // conversion alone is a no-op. Evaluating the complete parameter tree
        // here makes the requested residency real instead of charging each
        // transformer block to the first denoise pass.
        miniMaxH3EvaluateParameters(in: self)
        MLX.Memory.clearCache()

        let linears = leafModules().flattened().compactMap { $0.1 as? Linear }
        usesResidentBF16 = !linears.isEmpty && !linears.contains { $0 is QuantizedLinear }
        return .init(
            linearCount: linears.count,
            byteCount: linears.reduce(into: UInt64(0)) { total, linear in
                total += miniMaxH3ResidentBF16ByteCount(linear)
            }
        )
    }

    public func callAsFunction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        textStates: MLXArray,
        layout: MiniMaxH3PackedLayout,
        videoTimestep: Float,
        audioTimestep: Float,
        conditionVideoTimestep: Float = 0.999
    ) -> MiniMaxH3TransformerOutput {
        let context = prepare(textStates: textStates, layout: layout)
        return self(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: MLXArray([
                videoTimestep,
                audioTimestep,
                max(videoTimestep, conditionVideoTimestep),
            ]),
            cachedAdaLN: nil
        )
    }

    func prepare(
        textStates: MLXArray,
        layout: MiniMaxH3PackedLayout
    ) -> MiniMaxH3TransformerPreparedContext {
        precondition(textStates.dim(1) == layout.textRows.count)
        let text = tokenRefiner(miniMaxH3Linear(textInput, textStates))
        var timeIndices = Array(repeating: 0, count: layout.sequenceLength)
        for row in layout.conditionRows { timeIndices[row] = 2 }
        for row in layout.targetAudioRows { timeIndices[row] = 1 }
        let adaLNIndices = MLXArray(zip(timeIndices, layout.tokenTags).map { timeIndex, tag in
            Int32(timeIndex * 3) + max(tag, 0)
        })
        let rope = rotaryEmbedding(positions: layout.positions)
        let fastVSA = usesFastH3VSA ? MiniMaxH3FastVSA.prepare(layout: layout) : nil
        MLX.eval(text, adaLNIndices, rope.cosine, rope.sine)
        return MiniMaxH3TransformerPreparedContext(
            text: text,
            adaLNIndices: adaLNIndices,
            rope: rope,
            layout: layout,
            fastVSA: fastVSA
        )
    }

    func prepareTokenReduction(
        context: MiniMaxH3TransformerPreparedContext
    ) -> MiniMaxH3TokenReductionPreparedContext {
        let map = MiniMaxH3TokenReductionMap(layout: context.layout)
        let firstPositions = MLX.take(
            context.layout.positions[context.layout.targetVideoRows, 0...],
            map.firstVideoIndices,
            axis: 0
        ).asType(.float32)
        let secondPositions = MLX.take(
            context.layout.positions[context.layout.targetVideoRows, 0...],
            map.secondVideoIndices,
            axis: 0
        ).asType(.float32)
        let reducedVideoPositions = (firstPositions + secondPositions) * 0.5
        let reducedPositions = map.targetVideoStart == 0
            ? reducedVideoPositions
            : MLX.concatenated([
                context.layout.positions[0..<map.targetVideoStart, 0...],
                reducedVideoPositions,
            ], axis: 0)
        let reducedAdaLNIndices = MLX.take(
            context.adaLNIndices,
            map.absoluteSourceIndices,
            axis: 0
        )
        let absoluteSources = map.absoluteSourceIndices.asArray(Int32.self).map(Int.init)
        let reducedTags = absoluteSources.map { context.layout.tokenTags[$0] }
        let reducedWidth = ((context.layout.latentWidth / 2 + 1) / 2) * 2
        let reducedLayout = MiniMaxH3PackedLayout(
            positions: reducedPositions,
            tokenTags: reducedTags,
            textRows: context.layout.textRows,
            conditionRows: context.layout.conditionRows,
            conditionSegments: context.layout.conditionSegments,
            conditionVideoRowCount: context.layout.conditionVideoRowCount,
            conditionAudioRowCount: context.layout.conditionAudioRowCount,
            targetAudioRows: context.layout.targetAudioRows,
            targetVideoRows: map.targetVideoStart..<(map.targetVideoStart + map.reducedVideoRowCount),
            videoLatentFrames: context.layout.videoLatentFrames,
            latentHeight: context.layout.latentHeight,
            latentWidth: reducedWidth,
            audioLatentFrames: context.layout.audioLatentFrames
        )
        let reducedRope = rotaryEmbedding(positions: reducedPositions)
        MLX.eval(
            map.firstVideoIndices,
            map.secondVideoIndices,
            map.parentVideoIndices,
            map.absoluteSourceIndices,
            reducedAdaLNIndices,
            reducedRope.cosine,
            reducedRope.sine
        )
        return MiniMaxH3TokenReductionPreparedContext(
            map: map,
            reducedContext: MiniMaxH3TransformerPreparedContext(
                text: context.text,
                adaLNIndices: reducedAdaLNIndices,
                rope: reducedRope,
                layout: reducedLayout,
                fastVSA: nil
            )
        )
    }

    func callAsFunction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MiniMaxH3TransformerOutput {
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let hidden = runBlocks(
            prepared.hidden,
            range: blocks.indices,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        return finalize(
            hidden,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
    }

    func callWithTokenReduction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        reduction: MiniMaxH3TokenReductionPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?,
        policy: MiniMaxH3TokenReductionPolicy,
        stepIndex: Int
    ) -> MiniMaxH3TransformerOutput {
        let restoreBeforeBlock = policy.restoreBeforeBlock(stepIndex: stepIndex)
        precondition(policy.beginBlock < blocks.count)
        precondition(restoreBeforeBlock <= blocks.count)
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let fullBeforeReduction = runBlocks(
            prepared.hidden,
            range: 0..<policy.beginBlock,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        let reductionState = reduction.map.pool(fullBeforeReduction)
        let reducedHidden = runBlocks(
            reductionState.reducedHidden,
            range: policy.beginBlock..<restoreBeforeBlock,
            context: reduction.reducedContext,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        let restoredHidden = reduction.map.restore(
            reducedHidden,
            state: reductionState,
            updateScale: policy.updateScale
        )
        let finalHidden = runBlocks(
            restoredHidden,
            range: restoreBeforeBlock..<blocks.count,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        return finalize(
            finalHidden,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
    }

    func callWithBlockResidualReuse(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?,
        warmBlockCount: Int,
        cachedTailResidual: MLXArray?
    ) -> MiniMaxH3BlockReuseResult {
        precondition(warmBlockCount >= 0 && warmBlockCount < blocks.count)
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let warmHidden = runBlocks(
            prepared.hidden,
            range: 0..<warmBlockCount,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )

        let hidden: MLXArray
        let refreshedTailResidual: MLXArray?
        if let cachedTailResidual {
            precondition(cachedTailResidual.shape == warmHidden.shape)
            hidden = warmHidden + cachedTailResidual
            refreshedTailResidual = nil
            MLX.eval(hidden)
        } else {
            hidden = runBlocks(
                warmHidden,
                range: warmBlockCount..<blocks.count,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            )
            let residual = hidden - warmHidden
            MLX.eval(residual)
            refreshedTailResidual = residual
        }

        return MiniMaxH3BlockReuseResult(
            output: finalize(
                hidden,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            ),
            refreshedTailResidual: refreshedTailResidual
        )
    }

    func callWithAdaptiveFirstBlockReuse(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?,
        policy: MiniMaxH3AdaptiveFirstBlockCachePolicy,
        canConsiderReuse: Bool,
        previousFirstResidual: MLXArray?,
        cachedTargetTailResidual: MLXArray?
    ) -> MiniMaxH3AdaptiveBlockReuseResult {
        precondition(blocks.count > 1)
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let firstHidden = runBlocks(
            prepared.hidden,
            range: 0..<1,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        let targetRows = context.layout.targetAudioRows.lowerBound..<context.layout.targetVideoRows.upperBound
        let initialTarget = prepared.hidden[0..., targetRows, 0...]
        let firstTarget = firstHidden[0..., targetRows, 0...]
        let firstResidual = firstTarget - initialTarget

        let change: MiniMaxH3FirstBlockChange?
        let reusesTail: Bool
        if canConsiderReuse,
           let previousFirstResidual,
           let cachedTargetTailResidual,
           previousFirstResidual.shape == firstResidual.shape,
           cachedTargetTailResidual.shape == firstTarget.shape {
            let measured = MiniMaxH3FirstBlockChange.measure(
                current: firstResidual,
                previous: previousFirstResidual,
                layout: context.layout
            )
            change = measured
            reusesTail = policy.shouldReuse(change: measured)
        } else {
            change = nil
            reusesTail = false
        }

        let hidden: MLXArray
        let refreshedFirstResidual: MLXArray?
        let refreshedTargetTailResidual: MLXArray?
        if reusesTail, let cachedTargetTailResidual {
            let completedTarget = firstTarget + cachedTargetTailResidual
            hidden = targetRows.lowerBound == 0
                ? completedTarget
                : MLX.concatenated([
                    firstHidden[0..., 0..<targetRows.lowerBound, 0...],
                    completedTarget,
                ], axis: 1)
            MLX.eval(hidden)
            refreshedFirstResidual = nil
            refreshedTargetTailResidual = nil
        } else {
            hidden = runBlocks(
                firstHidden,
                range: 1..<blocks.count,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            )
            let targetTailResidual = hidden[0..., targetRows, 0...] - firstTarget
            MLX.eval(firstResidual, targetTailResidual)
            refreshedFirstResidual = firstResidual
            refreshedTargetTailResidual = targetTailResidual
        }

        return MiniMaxH3AdaptiveBlockReuseResult(
            output: finalize(
                hidden,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            ),
            refreshedFirstResidual: refreshedFirstResidual,
            refreshedTargetTailResidual: refreshedTargetTailResidual,
            reusedTail: reusesTail,
            change: change
        )
    }

    private func prepareBlockInput(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> (hidden: MLXArray, timeEmbedding: MLXArray) {
        let layout = context.layout
        precondition(videoRows.dim(1) == layout.conditionVideoRowCount + layout.targetVideoRows.count)
        precondition(audioRows.dim(1) == layout.conditionAudioRowCount + layout.targetAudioRows.count)
        precondition(timesteps.shape == [3])
        let video = miniMaxH3Linear(videoInput, videoRows).asType(context.text.dtype)
        let audio = miniMaxH3Linear(audioInput, audioRows).asType(context.text.dtype)
        let targetVideo = video[0..., layout.conditionVideoRowCount..., 0...]
        let targetAudio = audio[0..., layout.conditionAudioRowCount..., 0...]
        var packed: [MLXArray] = [context.text]
        for segment in layout.conditionSegments {
            switch segment.modality {
            case .video:
                packed.append(video[0..., segment.sourceRows, 0...])
            case .audio:
                packed.append(audio[0..., segment.sourceRows, 0...])
            case .text:
                preconditionFailure("condition segments cannot contain text")
            }
        }
        packed.append(targetAudio)
        packed.append(targetVideo)
        let hidden = MLX.concatenated(packed, axis: 1)

        let timeEmbedding = cachedAdaLN?.timeEmbedding ?? embedTimesteps(timesteps)
        if let cachedAdaLN {
            precondition(cachedAdaLN.blockModulations.count == blocks.count)
        }
        return (hidden, timeEmbedding)
    }

    private func runBlocks(
        _ initialHidden: MLXArray,
        range: Range<Int>,
        context: MiniMaxH3TransformerPreparedContext,
        timeEmbedding: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MLXArray {
        precondition(range.lowerBound >= 0 && range.upperBound <= blocks.count)
        var hidden = initialHidden
        for index in range {
            if let activeBlockIndices, !activeBlockIndices.contains(index) { continue }
            let block = blocks[index]
            let blockStarted = blockTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
            if let compressionGate = fastH3CompressionGates[index] {
                let phaseTimingHandler = fastH3BlockPhaseTimingHandler
                let projectionStarted = phaseTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
                let compiled = usesBlockwiseCompilation
                    ? compiledBlockForward(for: block, compressionGate: compressionGate)
                    : nil
                let projectedAttention: [MLXArray]
                if let compiled, let fastH3Projection = compiled.fastH3AttentionProjection {
                    var projectionInputs = [
                        hidden,
                        timeEmbedding,
                        context.adaLNIndices,
                        context.rope.cosine,
                        context.rope.sine,
                    ]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        projectionInputs.append(modulation)
                    }
                    projectionInputs.append(contentsOf: compressionGate.parameters)
                    projectedAttention = fastH3Projection(projectionInputs)
                } else {
                    projectedAttention = block.fastH3AttentionProjection(
                        hidden,
                        timeEmbedding: timeEmbedding,
                        adaLNIndices: context.adaLNIndices,
                        rope: context.rope,
                        cachedModulation: cachedAdaLN?.blockModulations[index],
                        compressionGate: compressionGate
                    )
                }
                MLX.eval(projectedAttention)
                let projectionSeconds = projectionStarted.map {
                    CFAbsoluteTimeGetCurrent() - $0
                }
                guard let fastVSA = context.fastVSA else {
                    preconditionFailure("FastH3 VSA prepared context is missing")
                }
                let attentionStarted = phaseTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
                guard let attended = MiniMaxH3FastVSA.call(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    compressionGate: projectedAttention[4],
                    prepared: fastVSA,
                    kernelMode: .runtimeDefault
                ) else {
                    preconditionFailure("FastH3 VSA requires batch-one BF16 attention on Metal")
                }
                if phaseTimingHandler != nil { MLX.eval(attended) }
                let attentionSeconds = attentionStarted.map {
                    CFAbsoluteTimeGetCurrent() - $0
                }
                let postAttentionStarted = phaseTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
                if let compiled {
                    var postAttentionInputs = [
                        hidden,
                        attended,
                        projectedAttention[3],
                        timeEmbedding,
                        context.adaLNIndices,
                    ]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        postAttentionInputs.append(modulation)
                    }
                    if exactKernelMode.usesTiledAffineQ8FeedForward,
                       Self.requiresTiledFeedForwardEvaluationBoundary(
                        rowCount: hidden.dim(1),
                        feedForwardSize: configuration.feedForwardSize,
                        itemSize: hidden.itemSize
                    ) {
                        // MLX buffers use 32-bit byte offsets. Materialize the
                        // compact SwiGLU result before compiling the FC2 stage
                        // when that single activation exceeds 4 GiB.
                        let feedForwardParts = compiled.postAttentionProjection(
                            postAttentionInputs
                        )
                        MLX.eval(feedForwardParts)
                        hidden = compiled.feedForwardOutput(feedForwardParts)[0]
                    } else {
                        hidden = compiled.postAttention(postAttentionInputs)[0]
                    }
                    MLX.eval(hidden)
                } else {
                    hidden = block.postAttention(
                        hidden,
                        attended: attended,
                        gate: projectedAttention[3],
                        timeEmbedding: timeEmbedding,
                        adaLNIndices: context.adaLNIndices,
                        cachedModulation: cachedAdaLN?.blockModulations[index]
                    )
                }
                if let phaseTimingHandler,
                   let projectionSeconds,
                   let attentionSeconds,
                   let postAttentionStarted {
                    MLX.eval(hidden)
                    phaseTimingHandler(
                        index,
                        projectionSeconds,
                        attentionSeconds,
                        CFAbsoluteTimeGetCurrent() - postAttentionStarted
                    )
                }
            } else if usesBlockwiseCompilation {
                var attentionInputs = [
                    hidden,
                    timeEmbedding,
                    context.adaLNIndices,
                    context.rope.cosine,
                    context.rope.sine,
                ]
                if let modulation = cachedAdaLN?.blockModulations[index] {
                    attentionInputs.append(modulation)
                }
                let compiled = compiledBlockForward(for: block)
                let projectedAttention = compiled.attentionProjection(attentionInputs)
                MLX.eval(projectedAttention)
                let dynamicSparseRequest = qualifiedDynamicSparseAttentionRequest(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    layerIndex: index,
                    layout: context.layout
                )
                let attended = block.scaledDotProductAttention(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    maximumQueryTokens: maximumAttentionQueryTokensPerKernel,
                    maximumHeadsPerKernel: maximumAttentionHeadsPerKernel,
                    maximumKernelsPerEvaluation: maximumAttentionKernelsPerEvaluation,
                    dynamicSparseRequest: dynamicSparseRequest
                )
                if usesFusedPostAttention {
                    var postAttentionInputs = [
                        hidden,
                        attended,
                        projectedAttention[3],
                        timeEmbedding,
                        context.adaLNIndices,
                    ]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        postAttentionInputs.append(modulation)
                    }
                    hidden = compiled.postAttention(postAttentionInputs)[0]
                    MLX.eval(hidden)
                } else {
                    hidden = compiled.attentionOutput([hidden, attended, projectedAttention[3]])[0]
                    MLX.eval(hidden)
                    var feedForwardInputs = [hidden, timeEmbedding, context.adaLNIndices]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        feedForwardInputs.append(modulation)
                    }
                    let projectedFeedForward = compiled.feedForwardProjection(feedForwardInputs)
                    MLX.eval(projectedFeedForward)
                    hidden = compiled.feedForwardOutput([
                        hidden,
                        projectedFeedForward[0],
                        projectedFeedForward[1],
                    ])[0]
                    MLX.eval(hidden)
                }
            } else {
                hidden = block(
                    hidden,
                    timeEmbedding: timeEmbedding,
                    adaLNIndices: context.adaLNIndices,
                    rope: context.rope,
                    cachedModulation: cachedAdaLN?.blockModulations[index]
                )
            }
            if usesLayerwiseEvaluation || blockTimingHandler != nil {
                MLX.eval(hidden)
                if clearsCacheAfterLayerwiseEvaluation {
                    MLX.Memory.clearCache()
                }
            }
            if let blockStarted, let blockTimingHandler {
                blockTimingHandler(
                    index,
                    CFAbsoluteTimeGetCurrent() - blockStarted,
                    Memory.snapshot()
                )
            }
        }
        return hidden
    }

    private func qualifiedDynamicSparseAttentionRequest(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        layerIndex: Int,
        layout: MiniMaxH3PackedLayout
    ) -> DynamicSparseAttentionRequest? {
        guard let request = dynamicSparseAttentionPolicy?.request(
            stepIndex: dynamicSparseAttentionStepIndex,
            stepCount: dynamicSparseAttentionStepCount,
            layerIndex: layerIndex,
            sequenceLength: layout.sequenceLength,
            prefixTokenCount: layout.targetVideoRows.lowerBound
        ) else { return nil }

        let gateShape = "\(queries.dim(1))x\(queries.dim(2))x\(request.prefixTokenCount)"
        let gateKey = "\(gateShape):\(queries.dtype):\(keys.dtype):\(values.dtype)"
        if dynamicSparseAttentionGateResults[gateKey] == nil {
            let gate = DynamicSparseAttention.denseRouteGate(
                queries: queries,
                keys: keys,
                values: values,
                queryStart: request.prefixTokenCount,
                scale: 1 / sqrt(Float(configuration.attentionHeadDimension))
            )
            dynamicSparseAttentionGateResults[gateKey] = gate?.passed ?? false
            if let gate {
                dynamicSparseAttentionLogHandler?(String(
                    format: "dynamic_sparse_gate=%@ shape=%@ max_abs=%.6g mean_abs=%.6g "
                        + "max_rel=%.6g mean_rel=%.6g rel_l2=%.6g",
                    gate.passed ? "pass" : "fail",
                    gateShape,
                    gate.maximumAbsoluteError,
                    gate.meanAbsoluteError,
                    gate.maximumRelativeError,
                    gate.meanRelativeError,
                    gate.relativeL2Error
                ))
            } else {
                dynamicSparseAttentionLogHandler?(
                    "dynamic_sparse_gate=unavailable shape=\(gateShape) "
                        + "q_dtype=\(queries.dtype) k_dtype=\(keys.dtype) "
                        + "v_dtype=\(values.dtype) device="
                        + String(describing: Device.defaultDevice().deviceType)
                )
            }
        }
        return dynamicSparseAttentionGateResults[gateKey] == true ? request : nil
    }

    private func finalize(
        _ hidden: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timeEmbedding: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MiniMaxH3TransformerOutput {
        finalLayer(
            hidden,
            timeEmbedding: timeEmbedding,
            videoRows: context.layout.targetVideoRows,
            videoTimeIndex: 0,
            audioRows: context.layout.targetAudioRows,
            audioTimeIndex: 1,
            cachedModulation: cachedAdaLN?.finalModulation
        )
    }

    private func compiledBlockForward(
        for block: MiniMaxH3TransformerBlock,
        compressionGate: MiniMaxH3FastH3CompressionGate? = nil
    ) -> MiniMaxH3CompiledBlockForwards {
        if let compiledBlockRunner, let compiledBlockForwards {
            updateCompiledBlockRunner(compiledBlockRunner, from: block)
            return compiledBlockForwards
        }

        let runner = makeCompiledBlockRunner(from: block)
        updateCompiledBlockRunner(runner, from: block)
        let attentionProjection = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.attentionProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )
        }
        let fastH3AttentionProjection: MiniMaxH3CompiledBlockForward? = compressionGate.map { gate in
            let gateStorage = gate.storage
            let gateParameterCount = gate.parameters.count
            return MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
                let hasCachedModulation = inputs.count == 6 + gateParameterCount
                let gateStart = hasCachedModulation ? 6 : 5
                return runner.fastH3AttentionProjection(
                    inputs[0],
                    timeEmbedding: inputs[1],
                    adaLNIndices: inputs[2],
                    rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                    cachedModulation: hasCachedModulation ? inputs[5] : nil,
                    compressionGateStorage: gateStorage,
                    compressionGateParameters: Array(inputs[gateStart...])
                )
            }
        }
        let attentionOutput = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            [runner.attentionProjectionResidual(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2]
            )]
        }
        let feedForwardProjection = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.feedForwardProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs.count == 4 ? inputs[3] : nil
            )
        }
        let feedForwardOutput = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            [runner.feedForwardProjectionResidual(
                inputs[0],
                projected: inputs[1],
                gate: inputs[2]
            )]
        }
        let postAttentionProjection = MLX.compile(
            inputs: [runner]
        ) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.postAttentionProjection(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )
        }
        let postAttention = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            [runner.postAttention(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )]
        }
        let forwards = MiniMaxH3CompiledBlockForwards(
            attentionProjection: attentionProjection,
            fastH3AttentionProjection: fastH3AttentionProjection,
            attentionOutput: attentionOutput,
            feedForwardProjection: feedForwardProjection,
            feedForwardOutput: feedForwardOutput,
            postAttentionProjection: postAttentionProjection,
            postAttention: postAttention
        )
        compiledBlockRunner = runner
        compiledBlockForwards = forwards
        return forwards
    }

    private func updateCompiledBlockRunner(
        _ runner: MiniMaxH3TransformerBlock,
        from block: MiniMaxH3TransformerBlock
    ) {
        let parameters = block.parameters().flattened().filter { path, _ in
            block.includesAdaLN || !path.hasPrefix("adaln_proj.")
        }
        runner.update(parameters: ModuleParameters.unflattened(parameters))
    }

    private func makeCompiledBlockRunner(
        from block: MiniMaxH3TransformerBlock
    ) -> MiniMaxH3TransformerBlock {
        let runner = MiniMaxH3TransformerBlock(
            configuration: configuration,
            includeAdaLN: block.includesAdaLN
        )
        runner.exactKernelMode = block.exactKernelMode
        runner.enabledExactKernelStages = block.enabledExactKernelStages
        runner.exactKernelDispatchHandler = block.exactKernelDispatchHandler
        runner.exactKernelFallbackHandler = block.exactKernelFallbackHandler
        let replacements: [(String, Module)] = block.leafModules().flattened().compactMap { path, module in
            if let lora = module as? MiniMaxH3RuntimeQuantizedQKVLoRALinear {
                let base = QuantizedLinear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    scales: MLXArray.zeros(lora.scales.shape, dtype: lora.scales.dtype),
                    biases: lora.biases.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    groupSize: lora.groupSize,
                    bits: lora.bits,
                    mode: lora.mode,
                    globalScale: lora.globalScale.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeQuantizedQKVLoRALinear(
                        base: base,
                        queryDown: MLXArray.zeros(lora.queryDown.shape, dtype: lora.queryDown.dtype),
                        queryUp: MLXArray.zeros(lora.queryUp.shape, dtype: lora.queryUp.dtype),
                        keyDown: MLXArray.zeros(lora.keyDown.shape, dtype: lora.keyDown.dtype),
                        keyUp: MLXArray.zeros(lora.keyUp.shape, dtype: lora.keyUp.dtype),
                        valueDown: MLXArray.zeros(lora.valueDown.shape, dtype: lora.valueDown.dtype),
                        valueUp: MLXArray.zeros(lora.valueUp.shape, dtype: lora.valueUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            if let lora = module as? MiniMaxH3RuntimeQuantizedLoRALinear {
                let base = QuantizedLinear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    scales: MLXArray.zeros(lora.scales.shape, dtype: lora.scales.dtype),
                    biases: lora.biases.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    groupSize: lora.groupSize,
                    bits: lora.bits,
                    mode: lora.mode,
                    globalScale: lora.globalScale.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeQuantizedLoRALinear(
                        base: base,
                        loraDown: MLXArray.zeros(lora.loraDown.shape, dtype: lora.loraDown.dtype),
                        loraUp: MLXArray.zeros(lora.loraUp.shape, dtype: lora.loraUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            if let lora = module as? MiniMaxH3RuntimeQKVLoRALinear {
                let base = Linear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeQKVLoRALinear(
                        base: base,
                        queryDown: MLXArray.zeros(lora.queryDown.shape, dtype: lora.queryDown.dtype),
                        queryUp: MLXArray.zeros(lora.queryUp.shape, dtype: lora.queryUp.dtype),
                        keyDown: MLXArray.zeros(lora.keyDown.shape, dtype: lora.keyDown.dtype),
                        keyUp: MLXArray.zeros(lora.keyUp.shape, dtype: lora.keyUp.dtype),
                        valueDown: MLXArray.zeros(lora.valueDown.shape, dtype: lora.valueDown.dtype),
                        valueUp: MLXArray.zeros(lora.valueUp.shape, dtype: lora.valueUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            if let lora = module as? MiniMaxH3RuntimeLoRALinear {
                let base = Linear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeLoRALinear(
                        base: base,
                        loraDown: MLXArray.zeros(lora.loraDown.shape, dtype: lora.loraDown.dtype),
                        loraUp: MLXArray.zeros(lora.loraUp.shape, dtype: lora.loraUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            guard let quantized = module as? QuantizedLinear else { return nil }
            let weight = MLXArray.zeros(quantized.weight.shape, dtype: quantized.weight.dtype)
            let bias = quantized.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            let scales = MLXArray.zeros(quantized.scales.shape, dtype: quantized.scales.dtype)
            let biases = quantized.biases.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            let globalScale = quantized.globalScale.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            let clone: Module
            if let portable = quantized as? PortableQuantizedLinear {
                let portableClone = PortableQuantizedLinear(
                    weight: weight,
                    bias: bias,
                    scales: scales,
                    biases: biases,
                    groupSize: portable.groupSize,
                    bits: portable.bits,
                    mode: portable.mode,
                    globalScale: globalScale
                )
                portableClone.cacheDenseFallbackWeight = portable.cacheDenseFallbackWeight
                portableClone.useUncachedDenseFallback = portable.useUncachedDenseFallback
                clone = portableClone
            } else {
                clone = QuantizedLinear(
                    weight: weight,
                    bias: bias,
                    scales: scales,
                    biases: biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: globalScale
                )
            }
            return (path, clone)
        }
        runner.update(modules: ModuleChildren.unflattened(replacements))
        return runner
    }

    func precomputeAdaLN(
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        sourceIdentity: String,
        progressHandler: (@Sendable (Int, Int) -> Void)? = nil
    ) -> MiniMaxH3AdaLNCache {
        precondition(videoSchedule.timesteps.count == audioSchedule.timesteps.count)
        let stepTimeEmbeddings = videoSchedule.timesteps.indices.map { index in
            let timesteps = MLXArray([
                videoSchedule.timesteps[index],
                audioSchedule.timesteps[index],
                max(videoSchedule.timesteps[index], 0.999),
            ])
            let value = embedTimesteps(timesteps)
            MLX.eval(value)
            return value
        }
        let timeEmbeddings = MLX.stacked(stepTimeEmbeddings, axis: 0)
        MLX.eval(timeEmbeddings)

        let progressTotal = blocks.count + 2
        progressHandler?(1, progressTotal)
        var blockModulations: [MLXArray] = []
        blockModulations.reserveCapacity(blocks.count)
        for (index, block) in blocks.enumerated() {
            let stepModulations = stepTimeEmbeddings.map { timeEmbedding in
                let value = block.precomputeModulation(timeEmbedding: timeEmbedding)
                MLX.eval(value)
                return value
            }
            let modulation = MLX.stacked(stepModulations, axis: 0)
            MLX.eval(modulation)
            blockModulations.append(modulation)
            progressHandler?(index + 2, progressTotal)
        }
        let finalModulations = MLX.stacked(stepTimeEmbeddings.map { timeEmbedding in
            let value = finalLayer.precomputeModulation(timeEmbedding: timeEmbedding)
            MLX.eval(value)
            return value
        }, axis: 0)
        MLX.eval(finalModulations)
        progressHandler?(progressTotal, progressTotal)
        return MiniMaxH3AdaLNCache(
            timeEmbeddings: timeEmbeddings,
            blockModulations: blockModulations,
            finalModulations: finalModulations,
            videoSigmas: videoSchedule.sigmas,
            audioSigmas: audioSchedule.sigmas,
            sourceIdentity: sourceIdentity
        )
    }

    func precomputeAdaLNStep(timesteps: MLXArray) -> MiniMaxH3AdaLNStep {
        precondition(timesteps.shape == [3])
        let timeEmbedding = embedTimesteps(timesteps)
        MLX.eval(timeEmbedding)
        let blockModulations = blocks.map { block in
            let modulation = block.precomputeModulation(timeEmbedding: timeEmbedding)
            MLX.eval(modulation)
            return modulation
        }
        let finalModulation = finalLayer.precomputeModulation(
            timeEmbedding: timeEmbedding
        )
        MLX.eval(finalModulation)
        return MiniMaxH3AdaLNStep(
            timeEmbedding: timeEmbedding,
            blockModulations: blockModulations,
            finalModulation: finalModulation
        )
    }

    /// Releases the schedule-only projection weights after an exact modulation
    /// table has been built for the current run. The remaining dense core is
    /// the complete denoising transformer.
    func discardAdaLNWeights() {
        guard adaLNWeightsAvailable else { return }
        compiledBlockRunner = nil
        compiledBlockForwards = nil
        update(modules: ModuleChildren.unflattened([
            ("time_embedder", MiniMaxH3TimeEmbedding(discarded: ())),
        ]))
        for block in blocks {
            block.discardAdaLNWeights()
        }
        finalLayer.discardAdaLNWeights()
        adaLNWeightsAvailable = false
    }

    private func embedTimesteps(_ timesteps: MLXArray) -> MLXArray {
        guard adaLNWeightsAvailable, let timeEmbedder else {
            preconditionFailure("MiniMax-H3 AdaLN cache is required")
        }
        let arguments = timesteps.reshaped(-1, 1) * timeFrequencies.reshaped(1, -1)
        let sinusoidal = MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1)
        return timeEmbedder(sinusoidal.asType(.float32))
    }

    private func rotaryEmbedding(positions: MLXArray) -> MiniMaxH3RotaryEmbedding {
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1)
            * inverseFrequencies.reshaped(1, 1, -1)
        let combined = frequencies.reshaped(positions.dim(0), 3 * configuration.ropeFrequencyCount)
        let angles = MLX.concatenated([combined, combined], axis: -1)
            .reshaped(1, positions.dim(0), 1, 6 * configuration.ropeFrequencyCount)
        return MiniMaxH3RotaryEmbedding(cosine: MLX.cos(angles), sine: MLX.sin(angles))
    }

}
