import Foundation
import MLX
import MLXFast
import MLXNN

enum Q35AttentionLayerType: String {
    case linear = "linear_attention"
    case full = "full_attention"

    init(_ raw: String) {
        self = Q35AttentionLayerType(rawValue: raw) ?? .linear
    }
}

public enum Q35LayerCache: @unchecked Sendable {
    case linear(Q35LinearCache)
    case full(KVCache)

    func fork() -> Q35LayerCache {
        switch self {
        case .linear(let cache):
            return .linear(cache.fork())
        case .full(let cache):
            return .full(cache.fork())
        }
    }

    func batched(with caches: [Q35LayerCache]) -> Q35LayerCache? {
        guard !caches.isEmpty else { return nil }
        switch self {
        case .linear(let cache):
            let typed = caches.compactMap { entry -> Q35LinearCache? in
                if case .linear(let linear) = entry {
                    return linear
                }
                return nil
            }
            guard typed.count == caches.count,
                  let batched = cache.batched(with: typed) else {
                return nil
            }
            return .linear(batched)
        case .full(let cache):
            let typed = caches.compactMap { entry -> KVCache? in
                if case .full(let full) = entry {
                    return full
                }
                return nil
            }
            guard typed.count == caches.count,
                  let batched = cache.batched(with: typed) else {
                return nil
            }
            return .full(batched)
        }
    }

    func unbatchedRows(count: Int) -> [Q35LayerCache]? {
        switch self {
        case .linear(let cache):
            return cache.unbatchedRows(count: count)?.map(Q35LayerCache.linear)
        case .full(let cache):
            return cache.unbatchedRows(count: count)?.map(Q35LayerCache.full)
        }
    }
}

final class Q35DecoderLayer: Module {
    @ModuleInfo(key: "linear_attn") var linearAttention: Q35LinearAttention
    @ModuleInfo(key: "self_attn") var selfAttention: Q35FullAttention
    @ModuleInfo(key: "mlp") var mlp: Q35FeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Q35RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Q35RMSNorm

    let layerType: Q35AttentionLayerType
    let isMLPOnly: Bool

    init(config: Q35Config, layerIndex: Int) {
        let text = config.textConfig
        self.layerType = Q35AttentionLayerType(text.layerTypes[layerIndex])
        self.isMLPOnly = text.mlpOnlyLayers.contains(layerIndex)

        self._linearAttention.wrappedValue = Q35LinearAttention(config: config)
        self._selfAttention.wrappedValue = Q35FullAttention(config: config)
        self._mlp.wrappedValue = Q35FeedForward(config: config)
        self._inputLayerNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        fullMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: Q35LayerCache?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var h = x
        if !isMLPOnly {
            let normed = inputLayerNorm(x)
            let attentionOut: MLXArray
            switch layerType {
            case .linear:
                let linearCache: Q35LinearCache?
                if case .linear(let cache)? = cache {
                    linearCache = cache
                } else {
                    linearCache = nil
                }
                attentionOut = linearAttention(normed, cache: linearCache)
            case .full:
                let fullCache: KVCache?
                if case .full(let cache)? = cache {
                    fullCache = cache
                } else {
                    fullCache = nil
                }
                attentionOut = selfAttention(normed, mask: fullMask, cache: fullCache, positionIds: positionIds)
            }
            h = x + attentionOut
        }

        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

final class Q35Transformer: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Q35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Q35RMSNorm

    init(config: Q35Config) {
        let text = config.textConfig
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabSize,
            dimensions: text.hiddenSize
        )
        self._layers.wrappedValue = (0..<text.numHiddenLayers).map { index in
            Q35DecoderLayer(config: config, layerIndex: index)
        }
        self._norm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        super.init()
    }

    func embeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return embedTokens(tokenIds)
    }

    private func firstFullCache(from cache: [Q35LayerCache?]?) -> KVCache? {
        guard let cache else { return nil }
        for entry in cache {
            if case .full(let kv)? = entry {
                return kv
            }
        }
        return nil
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embeddings(for: inputIds)

        let fullMask = createAttentionMask(h: hidden, cache: firstFullCache(from: cache))
        Q35DebugLayerDump.record(stage: "embeddings", hidden)

        for (index, layer) in layers.enumerated() {
            let layerCache = cache?[index] ?? nil
            hidden = layer(
                hidden,
                fullMask: fullMask,
                cache: layerCache,
                positionIds: positionIds
            )
            Q35DebugLayerDump.record(stage: "layer\(index)", hidden)
        }

        let normed = norm(hidden)
        Q35DebugLayerDump.record(stage: "final_norm", normed)
        Q35DebugLayerDump.flush()
        return normed
    }
}

struct Q35ForwardOutput {
    let hidden: MLXArray
    let logits: MLXArray
}

#if os(macOS) || os(iOS)
private let q35CompactDraftSelectKernel = MLXFast.metalKernel(
    name: "q35_compact_draft_select",
    inputNames: ["logits"],
    outputNames: ["token_id"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        float best_value = 0.0f;
        uint best_id = 0;
        bool have = false;

        for (uint index = lane; index < REAL_COUNT; index += TG_SIZE) {
            float value = float(logits[index]);
            bool value_nan = isnan(value);
            bool take;
            if (!have) {
                take = true;
            } else if (value_nan != isnan(best_value)) {
                take = !value_nan;
            } else if (value > best_value) {
                take = true;
            } else if (value < best_value) {
                take = false;
            } else {
                take = index < best_id;
            }
            if (take) {
                best_value = value;
                best_id = index;
                have = true;
            }
        }

        threadgroup float scratch_value[TG_SIZE];
        threadgroup uint scratch_id[TG_SIZE];
        scratch_value[lane] = have ? best_value : NAN;
        scratch_id[lane] = have ? best_id : 0xFFFFFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                float lhs_value = scratch_value[lane];
                float rhs_value = scratch_value[lane + stride];
                uint lhs_id = scratch_id[lane];
                uint rhs_id = scratch_id[lane + stride];
                bool lhs_empty = lhs_id == 0xFFFFFFFFu;
                bool rhs_empty = rhs_id == 0xFFFFFFFFu;
                bool take_rhs;
                if (rhs_empty) {
                    take_rhs = false;
                } else if (lhs_empty) {
                    take_rhs = true;
                } else if (isnan(rhs_value) != isnan(lhs_value)) {
                    take_rhs = !isnan(rhs_value);
                } else if (rhs_value > lhs_value) {
                    take_rhs = true;
                } else if (rhs_value < lhs_value) {
                    take_rhs = false;
                } else {
                    take_rhs = rhs_id < lhs_id;
                }
                if (take_rhs) {
                    scratch_value[lane] = rhs_value;
                    scratch_id[lane] = rhs_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint id = scratch_id[0];
            token_id[0] = int(id < PREFIX_COUNT ? id : id + CONTROL_OFFSET);
        }
    """,
    header: "",
    ensureRowContiguous: false
)
#endif

public final class Q35Model: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: Q35Transformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Q35Config

    private var compactDraftHead: Linear?
    private var compactDraftHeadSource: ObjectIdentifier?

    static let compactDraftPrefixCount = 98_304
    static let compactDraftControlStart = 248_044
    static let compactDraftControlEnd = 248_070
    static let compactDraftRealCount = compactDraftPrefixCount
        + compactDraftControlEnd - compactDraftControlStart
    static let compactDraftPaddedCount = 98_336

    public init(config: Q35Config) {
        self.config = config
        self._model.wrappedValue = Q35Transformer(config: config)
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(
                config.textConfig.hiddenSize,
                config.textConfig.vocabSize,
                bias: false
            )
        super.init()
    }

    public func embeddings(for inputIds: MLXArray) -> MLXArray {
        model.embeddings(for: inputIds)
    }

    func logits(from hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    /// Prepare the proposal-only compact projection after the exact target
    /// weights have loaded. This is called only when an MTP head is admitted.
    func prepareGreedyMTPDrafting() {
        guard let head = resolvedCompactDraftHead() else { return }
        MLX.eval(head.weight)
        if let quantized = head as? PortableQuantizedLinear {
            MLX.eval(quantized.scales)
            if let biases = quantized.biases { MLX.eval(biases) }
        }

        let hidden = MLXArray.zeros(
            [1, 1, config.textConfig.hiddenSize],
            dtype: .bfloat16
        )
        MLX.eval(greedyDraftToken(from: hidden))
    }

    /// Proposal-only greedy token selection. Target verification remains the
    /// sole authority for emitted tokens.
    func greedyDraftToken(from hidden: MLXArray) -> MLXArray {
        guard let head = resolvedCompactDraftHead() else {
            return argMax(logits(from: hidden), axis: -1).asType(.int32).reshaped(1, 1)
        }
        return Self.compactDraftTokenID(from: head(hidden))
    }

    static func compactDraftRows(_ array: MLXArray) -> MLXArray {
        let prefix = array[0..<compactDraftPrefixCount]
        let controls = array[compactDraftControlStart..<compactDraftControlEnd]
        let paddingCount = compactDraftPaddedCount - compactDraftRealCount
        let padding = array[0..<paddingCount]
        return MLX.concatenated([prefix, controls, padding], axis: 0)
    }

    static func compactDraftTokenID(from paddedLogits: MLXArray) -> MLXArray {
        #if os(macOS) || os(iOS)
        if Device.defaultDevice().deviceType == .gpu {
            let threadGroupSize = 1_024
            return q35CompactDraftSelectKernel(
                [paddedLogits.reshaped([compactDraftPaddedCount])],
                template: [
                    ("REAL_COUNT", compactDraftRealCount),
                    ("PREFIX_COUNT", compactDraftPrefixCount),
                    ("CONTROL_OFFSET", compactDraftControlStart - compactDraftPrefixCount),
                    ("TG_SIZE", threadGroupSize),
                ],
                grid: (threadGroupSize, 1, 1),
                threadGroup: (threadGroupSize, 1, 1),
                outputShapes: [[1, 1]],
                outputDTypes: [.int32]
            )[0]
        }
        #endif

        let compactID = argMax(
            paddedLogits[0..., 0..., 0..<compactDraftRealCount],
            axis: -1
        ).asType(.int32)
        return MLX.which(
            compactID .< compactDraftPrefixCount,
            compactID,
            compactID + (compactDraftControlStart - compactDraftPrefixCount)
        )
    }

    private func resolvedCompactDraftHead() -> Linear? {
        guard config.textConfig.vocabSize == 248_320,
              let fullHead = lmHead else {
            return nil
        }

        let source = ObjectIdentifier(fullHead)
        if compactDraftHeadSource == source {
            return compactDraftHead
        }

        let resolved: Linear?
        if let quantized = fullHead as? PortableQuantizedLinear {
            resolved = PortableQuantizedLinear(
                weight: Self.compactDraftRows(quantized.weight),
                bias: quantized.bias.map(Self.compactDraftRows),
                scales: Self.compactDraftRows(quantized.scales),
                biases: quantized.biases.map(Self.compactDraftRows),
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            )
        } else if type(of: fullHead) == Linear.self {
            resolved = Linear(
                weight: Self.compactDraftRows(fullHead.weight),
                bias: fullHead.bias.map(Self.compactDraftRows)
            )
        } else {
            // Wrapped or corrected projections may carry behavior that cannot
            // be represented by slicing their stored base arrays.
            resolved = nil
        }

        compactDraftHead = resolved
        compactDraftHeadSource = source
        return resolved
    }

    func forward(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil,
        positionIds: MLXArray? = nil
    ) -> Q35ForwardOutput {
        let hidden = model(
            inputIds,
            cache: cache,
            inputEmbeddings: inputEmbeddings,
            positionIds: positionIds
        )
        return Q35ForwardOutput(hidden: hidden, logits: logits(from: hidden))
    }

    /// Forward for prefill chunks: hidden states flow through every position
    /// (the KV cache needs them all), but the lm_head projects only the
    /// final position. Prefill consumers — first-token sampling, the MTP
    /// hidden seed, and prefix-KV checkpoint restores — read exactly the
    /// last position, while a full-chunk projection is a
    /// [chunk, 150k-vocab] matmul plus its materialization per chunk and
    /// pins full-chunk logits inside stored prefix-cache entries.
    func forwardPrefill(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil,
        positionIds: MLXArray? = nil,
        retainAllHidden: Bool = false
    ) -> Q35ForwardOutput {
        let hidden = model(
            inputIds,
            cache: cache,
            inputEmbeddings: inputEmbeddings,
            positionIds: positionIds
        )
        let sequenceLength = hidden.dim(1)
        let lastHidden: MLXArray
        if sequenceLength > 1 {
            lastHidden = hidden[0..., (sequenceLength - 1)..., 0...]
        } else {
            lastHidden = hidden
        }
        return Q35ForwardOutput(
            hidden: retainAllHidden ? hidden : lastHidden,
            logits: logits(from: lastHidden)
        )
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [Q35LayerCache?]?,
        inputEmbeddings: MLXArray? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        forward(
            inputIds,
            cache: cache,
            inputEmbeddings: inputEmbeddings,
            positionIds: positionIds
        ).logits
    }
}
