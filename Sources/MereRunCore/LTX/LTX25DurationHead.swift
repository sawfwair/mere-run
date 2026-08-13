import Foundation
import MLX
import MLXFast
import MLXNN

public struct LTX25AutoDuration: Sendable, Equatable {
    public let minimumSeconds: Double
    public let maximumSeconds: Double

    public init(minimumSeconds: Double = 1, maximumSeconds: Double = 20) {
        precondition(minimumSeconds > 0, "minimumSeconds must be positive")
        precondition(maximumSeconds >= minimumSeconds, "maximumSeconds must be >= minimumSeconds")
        self.minimumSeconds = minimumSeconds
        self.maximumSeconds = maximumSeconds
    }
}

public enum LTX25DurationConditioning: Sendable, Equatable {
    case audioOnly
    case audioVideo
}

public enum LTX25DurationHeadError: LocalizedError {
    case missingWeights(URL)
    case missingInputs
    case invalidBatchSize(Int)
    case invalidFrameRate(Double)

    public var errorDescription: String? {
        switch self {
        case .missingWeights(let url):
            return "Missing LTX 2.5 duration-head weights at \(url.path)."
        case .missingInputs:
            return "LTX 2.5 duration prediction requires video or audio connector tokens."
        case .invalidBatchSize(let value):
            return "LTX 2.5 duration prediction supports a single prompt at a time (got batch \(value))."
        case .invalidFrameRate(let value):
            return "LTX 2.5 duration prediction requires a positive frame rate (got \(value))."
        }
    }
}

public final class LTX25DurationHead: Module {
    @ModuleInfo(key: "video_input_proj") private var videoInputProjection: Linear
    @ModuleInfo(key: "video_modality_emb") private var videoModalityEmbedding: MLXArray
    @ModuleInfo(key: "audio_input_proj") private var audioInputProjection: Linear
    @ModuleInfo(key: "audio_modality_emb") private var audioModalityEmbedding: MLXArray
    @ModuleInfo(key: "attention_pooler") private var attentionPooler: LTX25DurationAttentionPooler
    @ModuleInfo(key: "mlp_hidden") private var hiddenProjection: Linear
    @ModuleInfo(key: "mlp_out") private var outputProjection: Linear

    public override init() {
        self._videoInputProjection.wrappedValue = Linear(4_096, 256, bias: true)
        self._videoModalityEmbedding.wrappedValue = MLX.zeros([256], dtype: .float32)
        self._audioInputProjection.wrappedValue = Linear(2_048, 256, bias: true)
        self._audioModalityEmbedding.wrappedValue = MLX.zeros([256], dtype: .float32)
        self._attentionPooler.wrappedValue = LTX25DurationAttentionPooler()
        self._hiddenProjection.wrappedValue = Linear(256, 256, bias: true)
        self._outputProjection.wrappedValue = Linear(256, 1, bias: true)
        super.init()
    }

    public static func load(
        weightsURL: URL,
        dtype: DType = .bfloat16,
        fileManager: FileManager = .default
    ) throws -> LTX25DurationHead {
        guard fileManager.fileExists(atPath: weightsURL.path) else {
            throw LTX25DurationHeadError.missingWeights(weightsURL)
        }
        let head = LTX25DurationHead()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: weightsURL,
            to: head,
            dtype: dtype,
            verify: .all,
            include: { $0.hasPrefix("duration_head.") },
            mapper: { key, value in
                mapLTX25DurationHeadWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 15
        )
        MLX.eval(head)
        return head
    }

    public func callAsFunction(
        videoTokens: MLXArray?,
        audioTokens: MLXArray?
    ) throws -> MLXArray {
        guard videoTokens != nil || audioTokens != nil else {
            throw LTX25DurationHeadError.missingInputs
        }

        var groups: [MLXArray] = []
        if let videoTokens {
            groups.append(
                videoInputProjection(videoTokens)
                    + videoModalityEmbedding.reshaped(1, 1, -1).asType(videoTokens.dtype)
            )
        }
        if let audioTokens {
            groups.append(
                audioInputProjection(audioTokens)
                    + audioModalityEmbedding.reshaped(1, 1, -1).asType(audioTokens.dtype)
            )
        }

        let tokens = groups.count == 1 ? groups[0] : MLX.concatenated(groups, axis: 1)
        let pooled = attentionPooler(tokens)
        let flattened = pooled.reshaped(pooled.dim(0), -1)
        let hidden = MLXNN.geluApproximate(hiddenProjection(flattened))
        return MLX.exp(outputProjection(hidden).squeezed(axis: -1))
    }

    public func predictFrameCount(
        videoTokens: MLXArray?,
        audioTokens: MLXArray?,
        frameRate: Double,
        range: LTX25AutoDuration = LTX25AutoDuration()
    ) throws -> Int {
        guard frameRate > 0 else {
            throw LTX25DurationHeadError.invalidFrameRate(frameRate)
        }
        let prediction = try callAsFunction(videoTokens: videoTokens, audioTokens: audioTokens)
        guard prediction.shape == [1] else {
            throw LTX25DurationHeadError.invalidBatchSize(prediction.dim(0))
        }
        MLX.eval(prediction)
        return ltx25FrameCount(
            predictedSeconds: Double(prediction.item(Float.self)),
            frameRate: frameRate,
            range: range
        )
    }
}

private final class LTX25DurationAttentionPooler: Module {
    @ModuleInfo(key: "query_tokens") private var queryTokens: MLXArray
    @ModuleInfo(key: "cross_attn") private var crossAttention: LTX25DurationCrossAttention

    override init() {
        self._queryTokens.wrappedValue = MLX.zeros([1, 256], dtype: .float32)
        self._crossAttention.wrappedValue = LTX25DurationCrossAttention()
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let queries = broadcast(
            queryTokens.expandedDimensions(axis: 0).asType(tokens.dtype),
            to: [tokens.dim(0), queryTokens.dim(0), queryTokens.dim(1)]
        )
        return crossAttention(queries: queries, tokens: tokens)
    }
}

private final class LTX25DurationCrossAttention: Module {
    private let hiddenSize = 256
    private let heads = 4
    private let headDim = 64

    @ModuleInfo(key: "in_proj_weight") private var inputProjectionWeight: MLXArray
    @ModuleInfo(key: "in_proj_bias") private var inputProjectionBias: MLXArray
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear

    override init() {
        self._inputProjectionWeight.wrappedValue = MLX.zeros([768, 256], dtype: .float32)
        self._inputProjectionBias.wrappedValue = MLX.zeros([768], dtype: .float32)
        self._outputProjection.wrappedValue = Linear(256, 256, bias: true)
        super.init()
    }

    func callAsFunction(queries: MLXArray, tokens: MLXArray) -> MLXArray {
        let batch = tokens.dim(0)
        let queryCount = queries.dim(1)
        let tokenCount = tokens.dim(1)
        let projectedQueries = MLX.matmul(queries, inputProjectionWeight.T) + inputProjectionBias
        let projectedTokens = MLX.matmul(tokens, inputProjectionWeight.T) + inputProjectionBias

        let query = projectedQueries[0..., 0..., 0..<hiddenSize]
            .reshaped(batch, queryCount, heads, headDim)
            .transposed(0, 2, 1, 3)
        let key = projectedTokens[0..., 0..., hiddenSize..<(hiddenSize * 2)]
            .reshaped(batch, tokenCount, heads, headDim)
            .transposed(0, 2, 1, 3)
        let value = projectedTokens[0..., 0..., (hiddenSize * 2)...]
            .reshaped(batch, tokenCount, heads, headDim)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / Float(headDim).squareRoot(),
            mask: .none
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped(batch, queryCount, hiddenSize)
        )
    }
}

func mapLTX25DurationHeadWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("duration_head.") else { return [] }
    let mapped = String(key.dropFirst("duration_head.".count))
    let casted = value.dtype.isFloatingPoint && value.dtype != dtype
        ? value.asType(dtype)
        : value
    return [(mapped, casted)]
}

public func ltx25FrameCount(
    predictedSeconds: Double,
    frameRate: Double,
    range: LTX25AutoDuration = LTX25AutoDuration()
) -> Int {
    precondition(frameRate > 0, "frameRate must be positive")
    let minimumFrames = Int((range.minimumSeconds * frameRate).rounded(.toNearestOrEven))
    let maximumFrames = Int((range.maximumSeconds * frameRate).rounded(.toNearestOrEven))
    let rawFrames = Int((predictedSeconds * frameRate).rounded(.toNearestOrEven))
    let clamped = min(max(rawFrames, minimumFrames), maximumFrames)
    let snappedDown = ((max(1, clamped) - 1) / 8) * 8 + 1
    if snappedDown >= minimumFrames {
        return snappedDown
    }
    return min((((minimumFrames - 1) + 7) / 8) * 8 + 1, maximumFrames)
}
