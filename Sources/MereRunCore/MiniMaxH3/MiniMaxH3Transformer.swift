import Foundation
import MLX
import MLXFast
import MLXNN

@inline(__always)
private func miniMaxH3Linear(_ linear: Linear, _ value: MLXArray) -> MLXArray {
    let activationDType = (linear as? QuantizedLinear)?.scales.dtype ?? linear.weight.dtype
    return linear(value.asType(activationDType))
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

private final class MiniMaxH3Attention: Module {
    let heads: Int
    let headDimension: Int
    let innerDimension: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var queryKeyValue: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "out_proj") var output: Linear

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
            maximumQueryTokens: nil
        )
        return projectOutput(attended)
    }

    func project(_ input: MLXArray, rope: MiniMaxH3RotaryEmbedding?) -> [MLXArray] {
        // The MLX-Serve artifact deinterleaves the released checkpoint's
        // per-head rows into three global Q/K/V slabs before quantization.
        // Do not apply the raw-checkpoint interleave a second time here.
        let projected = miniMaxH3SplitProjectedQKV(
            queryKeyValue(input), heads: heads, headDimension: headDimension
        )
        var query = queryNorm(projected[0])
        var key = keyNorm(projected[1])
        if let rope {
            query = rope.apply(query)
            key = rope.apply(key)
        }
        query = query.transposed(0, 2, 1, 3)
        key = key.transposed(0, 2, 1, 3)
        let value = projected[2].transposed(0, 2, 1, 3)
        return [query, key, value]
    }

    func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        maximumQueryTokens: Int?,
        maximumKernelsPerEvaluation: Int = 1
    ) -> MLXArray {
        guard let maximumQueryTokens,
              maximumQueryTokens > 0,
              queries.dim(2) > maximumQueryTokens else {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
        }

        precondition(maximumKernelsPerEvaluation > 0)
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((queries.dim(2) + maximumQueryTokens - 1) / maximumQueryTokens)
        var pending: [MLXArray] = []
        pending.reserveCapacity(maximumKernelsPerEvaluation)
        for start in stride(from: 0, to: queries.dim(2), by: maximumQueryTokens) {
            let end = min(start + maximumQueryTokens, queries.dim(2))
            let chunk = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., start..<end, 0...],
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            chunks.append(chunk)
            pending.append(chunk)
            if pending.count == maximumKernelsPerEvaluation {
                MLX.eval(pending)
                pending.removeAll(keepingCapacity: true)
            }
        }
        if !pending.isEmpty {
            MLX.eval(pending)
        }
        return MLX.concatenated(chunks, axis: 2)
    }

    func projectOutput(_ attended: MLXArray) -> MLXArray {
        let batch = attended.dim(0)
        let sequence = attended.dim(2)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, innerDimension))
    }
}

private final class MiniMaxH3FeedForward: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

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
        output(project(value))
    }

    func project(_ value: MLXArray) -> MLXArray {
        let parts = MLX.split(input(value), parts: 2, axis: -1)
        return MLXNN.silu(parts[0]) * parts[1]
    }

    func projectOutput(_ value: MLXArray) -> MLXArray {
        output(value)
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

    init(configuration: MiniMaxH3TransformerConfiguration, includeAdaLN: Bool) {
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
        adaLN != nil
    }

    func callAsFunction(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray {
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

    func attentionResidual(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        adaLNIndices: MLXArray,
        rope: MiniMaxH3RotaryEmbedding,
        cachedModulation: MLXArray?
    ) -> MLXArray {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 6, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let shiftAttention = MLX.take(modulation[0], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scaleAttention = MLX.take(modulation[1], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = attentionNorm(value) * (1 + scaleAttention) + shiftAttention
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
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 6, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let shiftAttention = MLX.take(modulation[0], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let scaleAttention = MLX.take(modulation[1], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let gateAttention = MLX.take(modulation[2], adaLNIndices, axis: 0).expandedDimensions(axis: 0)
        let attentionInput = attentionNorm(value) * (1 + scaleAttention) + shiftAttention
        return attention.project(attentionInput, rope: rope) + [gateAttention]
    }

    func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        maximumQueryTokens: Int,
        maximumKernelsPerEvaluation: Int
    ) -> MLXArray {
        attention.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            maximumQueryTokens: maximumQueryTokens,
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
        guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN weights are not loaded") }
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
}

struct MiniMaxH3AdaLNStep {
    let timeEmbedding: MLXArray
    let blockModulations: [MLXArray]
    let finalModulation: MLXArray
}

private typealias MiniMaxH3CompiledBlockForward = @Sendable ([MLXArray]) -> [MLXArray]

private struct MiniMaxH3CompiledBlockForwards {
    let attentionProjection: MiniMaxH3CompiledBlockForward
    let attentionOutput: MiniMaxH3CompiledBlockForward
    let feedForwardProjection: MiniMaxH3CompiledBlockForward
    let feedForwardOutput: MiniMaxH3CompiledBlockForward
}

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
    return (Linear(weight: weight, bias: bias), weightBytes + biasBytes)
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
    private(set) var usesResidentBF16 = false
    var maximumAttentionQueryTokensPerKernel = 2_048
    var maximumAttentionKernelsPerEvaluation = 4
    var blockTimingHandler: ((Int, TimeInterval, Memory.Snapshot) -> Void)?
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

    public init(
        configuration: MiniMaxH3TransformerConfiguration = .init(),
        includeAdaLN: Bool = true
    ) {
        self.configuration = configuration
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
            guard let quantized = entry.1 as? QuantizedLinear else { return }
            let shape = quantized.shape
            total += UInt64(shape.0) * UInt64(shape.1) * 2
            total += UInt64(quantized.bias?.size ?? 0) * 2
        }
    }

    /// Expands compact quantized storage into resident bf16 linear weights.
    /// Each transformer/refiner block is replaced and the Metal cache is
    /// cleared before moving to the next one, bounding conversion residency
    /// instead of retaining a second whole-model copy.
    func materializeResidentBF16() -> MiniMaxH3ResidentBF16Materialization {
        compiledBlockRunner = nil
        compiledBlockForwards = nil
        var total = MiniMaxH3ResidentBF16Materialization(linearCount: 0, byteCount: 0)

        func include(_ materialization: MiniMaxH3ResidentBF16Materialization) {
            total = .init(
                linearCount: total.linearCount + materialization.linearCount,
                byteCount: total.byteCount + materialization.byteCount
            )
            MLX.Memory.clearCache()
        }

        for block in tokenRefiner.blocks {
            include(miniMaxH3MaterializeResidentBF16(in: block))
        }
        for block in blocks {
            include(miniMaxH3MaterializeResidentBF16(in: block))
        }
        if let timeEmbedder {
            include(miniMaxH3MaterializeResidentBF16(in: timeEmbedder))
        }
        include(miniMaxH3MaterializeResidentBF16(in: finalLayer))

        var rootReplacements: [(String, Module)] = []
        var rootBytes: UInt64 = 0
        for (path, linear) in [
            ("video_patch_proj", videoInput),
            ("audio_patch_proj", audioInput),
            ("condition_proj", textInput),
        ] {
            guard let resident = miniMaxH3ResidentBF16Linear(linear) else { continue }
            rootReplacements.append((path, resident.linear))
            rootBytes += resident.byteCount
        }
        if !rootReplacements.isEmpty {
            update(modules: ModuleChildren.unflattened(rootReplacements))
            include(.init(linearCount: rootReplacements.count, byteCount: rootBytes))
        }
        usesResidentBF16 = total.linearCount > 0
        return total
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
        MLX.eval(text, adaLNIndices, rope.cosine, rope.sine)
        return MiniMaxH3TransformerPreparedContext(
            text: text,
            adaLNIndices: adaLNIndices,
            rope: rope,
            layout: layout
        )
    }

    func callAsFunction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MiniMaxH3TransformerOutput {
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
        var hidden = MLX.concatenated(packed, axis: 1)

        let timeEmbedding = cachedAdaLN?.timeEmbedding ?? embedTimesteps(timesteps)
        if let cachedAdaLN {
            precondition(cachedAdaLN.blockModulations.count == blocks.count)
        }
        for (index, block) in blocks.enumerated() {
            let blockStarted = blockTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
            if usesBlockwiseCompilation {
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
                let attended = block.scaledDotProductAttention(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    maximumQueryTokens: maximumAttentionQueryTokensPerKernel,
                    maximumKernelsPerEvaluation: maximumAttentionKernelsPerEvaluation
                )
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
        return finalLayer(
            hidden,
            timeEmbedding: timeEmbedding,
            videoRows: layout.targetVideoRows,
            videoTimeIndex: 0,
            audioRows: layout.targetAudioRows,
            audioTimeIndex: 1,
            cachedModulation: cachedAdaLN?.finalModulation
        )
    }

    private func compiledBlockForward(
        for block: MiniMaxH3TransformerBlock
    ) -> MiniMaxH3CompiledBlockForwards {
        if let compiledBlockRunner, let compiledBlockForwards {
            compiledBlockRunner.update(parameters: block.parameters())
            return compiledBlockForwards
        }

        let runner = makeCompiledBlockRunner(from: block)
        runner.update(parameters: block.parameters())
        let attentionProjection = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.attentionProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )
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
        let forwards = MiniMaxH3CompiledBlockForwards(
            attentionProjection: attentionProjection,
            attentionOutput: attentionOutput,
            feedForwardProjection: feedForwardProjection,
            feedForwardOutput: feedForwardOutput
        )
        compiledBlockRunner = runner
        compiledBlockForwards = forwards
        return forwards
    }

    private func makeCompiledBlockRunner(
        from block: MiniMaxH3TransformerBlock
    ) -> MiniMaxH3TransformerBlock {
        let runner = MiniMaxH3TransformerBlock(
            configuration: configuration,
            includeAdaLN: block.includesAdaLN
        )
        let replacements: [(String, Module)] = block.leafModules().flattened().compactMap { path, module in
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
        let stepCount = videoSchedule.timesteps.count
        let flattenedTimesteps = videoSchedule.timesteps.indices.flatMap { index in
            let videoTimestep = videoSchedule.timesteps[index]
            let audioTimestep = audioSchedule.timesteps[index]
            return [videoTimestep, audioTimestep, max(videoTimestep, 0.999)]
        }
        let timeEmbeddings = embedTimesteps(MLXArray(flattenedTimesteps))
            .reshaped(stepCount, 3, configuration.timeEmbeddingDimension)
        MLX.eval(timeEmbeddings)

        let progressTotal = blocks.count + 2
        progressHandler?(1, progressTotal)
        var blockModulations: [MLXArray] = []
        blockModulations.reserveCapacity(blocks.count)
        for (index, block) in blocks.enumerated() {
            let modulation = block.precomputeModulation(
                timeEmbedding: timeEmbeddings.reshaped(stepCount * 3, configuration.timeEmbeddingDimension)
            ).reshaped(stepCount, 3 * 3, 6 * configuration.hiddenSize)
            MLX.eval(modulation)
            blockModulations.append(modulation)
            progressHandler?(index + 2, progressTotal)
        }
        let finalModulations = finalLayer.precomputeModulation(
            timeEmbedding: timeEmbeddings.reshaped(stepCount * 3, configuration.timeEmbeddingDimension)
        ).reshaped(stepCount, 3, 2 * configuration.hiddenSize)
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

    private func embedTimesteps(_ timesteps: MLXArray) -> MLXArray {
        guard let timeEmbedder else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
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
