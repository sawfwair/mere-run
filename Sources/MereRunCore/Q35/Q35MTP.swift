import Foundation
import MLX
import MLXFast
import MLXNN

@inline(__always)
private func q35MTPSwiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

private final class Q35MTPPositionCache: KVCache {
    public private(set) var offset: Int

    init(offset: Int) {
        self.offset = offset
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(2)
        return (keys, values)
    }

    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n == 1 ? .none : .causal
    }

    func fork() -> KVCache {
        Q35MTPPositionCache(offset: offset)
    }
}

final class Q35MTPExperts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: MLXArray
    @ModuleInfo(key: "down_proj") var downProj: MLXArray

    private let intermediateSize: Int
    let expertCount: Int

    init(config: Q35Config) {
        let text = config.textConfig
        self.intermediateSize = text.moeIntermediateSize
        self.expertCount = text.numExperts
        self._gateUpProj.wrappedValue = MLXArray.zeros(
            [text.numExperts, text.moeIntermediateSize * 2, text.hiddenSize],
            dtype: .bfloat16
        )
        self._downProj.wrappedValue = MLXArray.zeros(
            [text.numExperts, text.hiddenSize, text.moeIntermediateSize],
            dtype: .bfloat16
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)

        var expanded = x.reshaped([batchTokens, 1, inputDim])
        expanded = MLX.expandedDimensions(expanded, axis: 1)
        expanded = MLX.repeated(expanded, count: topK, axis: 1)
        let flatX = expanded.reshaped([batchTokens * topK, 1, inputDim])
        let flatIndices = indices.reshaped([batchTokens * topK])

        let gateUp = q35DenseExpertMM(
            flatX,
            weight: gateUpProj,
            indices: flatIndices,
            sortedIndices: false
        )
        let gate = gateUp[.ellipsis, 0..<intermediateSize]
        let up = gateUp[.ellipsis, intermediateSize...]
        let activated = q35MTPSwiglu(gate, up)

        let output = q35DenseExpertMM(
            activated,
            weight: downProj,
            indices: flatIndices,
            sortedIndices: false
        )
        let outDim = output.dim(2)
        return output.reshaped([x.dim(0), x.dim(1), topK, outDim])
    }
}

final class Q35MTPFeedForward: Module {
    @ModuleInfo(key: "gate") var gate: Linear?
    @ModuleInfo(key: "experts") var experts: Q35MTPExperts?
    @ModuleInfo(key: "shared_expert") var sharedExpert: Q35MLP?
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear?
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "up_proj") var upProj: Linear?
    @ModuleInfo(key: "down_proj") var downProj: Linear?

    private let topK: Int
    private let usesMoE: Bool

    init(config: Q35Config) {
        let text = config.textConfig
        self.usesMoE = text.usesMoE
        self.topK = max(1, text.numExpertsPerTok)

        if text.usesMoE {
            self._gate.wrappedValue = Linear(text.hiddenSize, text.numExperts, bias: false)
            self._experts.wrappedValue = Q35MTPExperts(config: config)
            self._sharedExpert.wrappedValue = Q35MLP(
                hiddenSize: text.hiddenSize,
                intermediateSize: text.sharedExpertIntermediateSize
            )
            self._sharedExpertGate.wrappedValue = Linear(text.hiddenSize, 1, bias: false)
            self._gateProj.wrappedValue = nil
            self._upProj.wrappedValue = nil
            self._downProj.wrappedValue = nil
        } else {
            self._gate.wrappedValue = nil
            self._experts.wrappedValue = nil
            self._sharedExpert.wrappedValue = nil
            self._sharedExpertGate.wrappedValue = nil
            self._gateProj.wrappedValue = Linear(text.hiddenSize, text.intermediateSize, bias: false)
            self._upProj.wrappedValue = Linear(text.hiddenSize, text.intermediateSize, bias: false)
            self._downProj.wrappedValue = Linear(text.intermediateSize, text.hiddenSize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if !usesMoE,
           let gateProj,
           let upProj,
           let downProj {
            return downProj(q35MTPSwiglu(gateProj(x), upProj(x)))
        }

        guard let gate,
              let experts,
              let sharedExpert,
              let sharedExpertGate else {
            return x
        }
        var scores = softmax(gate(x), axis: -1)
        let k = min(topK, scores.dim(-1))
        let indices = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        scores = takeAlong(scores, indices, axis: -1)
        if k > 1 {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let routed = (experts(x, indices: indices) * MLX.expandedDimensions(scores, axis: scores.ndim))
            .sum(axis: -2)
        let shared = sharedExpert(x)
        return routed + MLX.sigmoid(sharedExpertGate(x)) * shared
    }
}

final class Q35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Q35FullAttention
    @ModuleInfo(key: "mlp") var mlp: Q35MTPFeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Q35RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Q35RMSNorm

    init(config: Q35Config) {
        let text = config.textConfig
        self._selfAttention.wrappedValue = Q35FullAttention(config: config)
        self._mlp.wrappedValue = Q35MTPFeedForward(config: config)
        self._inputLayerNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

struct Q35MTPDraftOutput {
    let logitsHidden: MLXArray
    let recurrentHidden: MLXArray
}

protocol Q35MTPDraftModel: AnyObject {
    var diagnosticsID: String { get }

    func forwardDraft(
        inputEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        cache: KVCache
    ) -> Q35MTPDraftOutput

    func draftLogits(
        token: Int,
        previousHidden: MLXArray,
        positionOffset: Int,
        baseModel: Q35Model
    ) -> MLXArray

    func draftBlock(
        lastToken: Int,
        hidden: MLXArray,
        blockSize: Int,
        session: Q35MTPDraftSession,
        baseModel: Q35Model
    ) -> Q35MTPDraftBlock
}

final class Q35MTPModel: Module, Q35MTPDraftModel {
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: Q35RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: Q35RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Q35MTPDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Q35RMSNorm

    let diagnosticsID = "qwen-mtp"

    init(config: Q35Config) {
        let text = config.textConfig
        self._preFCNormEmbedding.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._preFCNormHidden.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._fc.wrappedValue = Linear(text.hiddenSize * 2, text.hiddenSize, bias: false)
        self._layers.wrappedValue = [Q35MTPDecoderLayer(config: config)]
        self._norm.wrappedValue = Q35RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        inputEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        cache: KVCache
    ) -> MLXArray {
        forwardDraft(
            inputEmbeddings: inputEmbeddings,
            hiddenStates: hiddenStates,
            cache: cache
        ).logitsHidden
    }

    func forwardDraft(
        inputEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        cache: KVCache
    ) -> Q35MTPDraftOutput {
        let normalizedEmbeddings = preFCNormEmbedding(inputEmbeddings)
        let normalizedHidden = preFCNormHidden(hiddenStates)
        var hidden = MLX.concatenated([normalizedEmbeddings, normalizedHidden], axis: -1)
        hidden = fc(hidden)

        let mask = cache.makeMask(n: hidden.dim(1))
        for layer in layers {
            hidden = layer(hidden, mask: mask, cache: cache)
        }
        let output = norm(hidden)
        return Q35MTPDraftOutput(logitsHidden: output, recurrentHidden: output)
    }

    func draftLogits(
        token: Int,
        previousHidden: MLXArray,
        positionOffset: Int,
        baseModel: Q35Model
    ) -> MLXArray {
        let inputIds = MLXArray([Int32(token)]).reshaped(1, 1)
        let embeddings = baseModel.embeddings(for: inputIds)
        let cache = Q35MTPPositionCache(offset: positionOffset)
        let output = forwardDraft(
            inputEmbeddings: embeddings,
            hiddenStates: previousHidden,
            cache: cache
        )
        return baseModel.logits(from: output.logitsHidden)
    }

    func draftBlock(
        lastToken: Int,
        hidden: MLXArray,
        blockSize: Int,
        session: Q35MTPDraftSession,
        baseModel: Q35Model
    ) -> Q35MTPDraftBlock {
        session.draftBlock(
            lastToken: lastToken,
            hidden: hidden,
            blockSize: blockSize,
            mtpModel: self,
            baseModel: baseModel
        )
    }
}

/// Device-resident greedy proposals for one target verification round.
///
/// Keeping the ids as an MLX array lets the target graph consume them before
/// the single verification synchronization also makes them available to the
/// CPU acceptance loop.
struct Q35MTPDraftBlock {
    let tokenIDs: MLXArray

    var count: Int {
        tokenIDs.dim(1)
    }

    var tokens: [Int] {
        tokenIDs.asArray(Int32.self).map(Int.init)
    }
}

/// Qwen4Exp's inline one-layer MTP head.
///
/// Unlike the earlier Qwen predictor, this head consumes the target model's
/// four-stream residual state. The token embedding and each residual stream
/// are projected independently, added in stream space, passed through one
/// full GR/QSA/MoE decoder layer, and reduced by the trained final mixer.
final class Q38MTPModel: Module, Q35MTPDraftModel {
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: Q35RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: Q35RMSNorm
    @ModuleInfo(key: "fc_embedding") var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") var fcHidden: Linear
    @ModuleInfo(key: "layers") var layers: [Q35DecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Q38GatedResidual

    let diagnosticsID = "qwen4-exp-mtp"

    private let hiddenSize: Int
    private let streamCount: Int

    init(config: Q35Config) {
        let text = config.textConfig
        let hyperDimensions = text.hiddenSize * text.hyperConnectionCount
        self.hiddenSize = text.hiddenSize
        self.streamCount = text.hyperConnectionCount
        self._preFCNormEmbedding.wrappedValue = Q35RMSNorm(
            dimensions: text.hiddenSize,
            eps: text.rmsNormEps
        )
        self._preFCNormHidden.wrappedValue = Q35RMSNorm(
            dimensions: hyperDimensions,
            eps: text.rmsNormEps
        )
        self._fcEmbedding.wrappedValue = Linear(text.hiddenSize, text.hiddenSize, bias: false)
        self._fcHidden.wrappedValue = Linear(text.hiddenSize, text.hiddenSize, bias: false)
        let rawLayerType = text.mtp?.layerTypes.first ?? Q35AttentionLayerType.full.rawValue
        self._layers.wrappedValue = [
            Q35DecoderLayer(
                config: config,
                layerIndex: 0,
                layerTypeOverride: Q35AttentionLayerType(rawLayerType),
                includesPLE: false
            ),
        ]
        self._hyperConnectionMixer.wrappedValue = Q38GatedResidual(
            config: config,
            combinesBlockOutput: false
        )
        super.init()
    }

    func callAsFunction(
        inputEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        cache: KVCache
    ) -> MLXArray {
        forwardDraft(
            inputEmbeddings: inputEmbeddings,
            hiddenStates: hiddenStates,
            cache: cache
        ).logitsHidden
    }

    func forwardDraft(
        inputEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        cache: KVCache
    ) -> Q35MTPDraftOutput {
        precondition(
            hiddenStates.dim(-1) == hiddenSize * streamCount,
            "Qwen4Exp MTP requires the target model's multi-stream hidden state"
        )
        let shapePrefix = Array(hiddenStates.shape.dropLast())
        let embedding = fcEmbedding(preFCNormEmbedding(inputEmbeddings))
        var hidden = preFCNormHidden(hiddenStates)
            .reshaped(shapePrefix + [streamCount, hiddenSize])
        hidden = fcHidden(hidden) + MLX.expandedDimensions(embedding, axis: -2)
        hidden = hidden.reshaped(shapePrefix + [streamCount * hiddenSize])

        let mask = cache.makeMask(n: hidden.dim(1))
        for layer in layers {
            hidden = layer(
                hidden,
                fullMask: mask,
                cache: .full(cache),
                targetVerify: false
            )
        }
        return Q35MTPDraftOutput(
            logitsHidden: hyperConnectionMixer.combine(hidden),
            recurrentHidden: hidden
        )
    }

    func draftLogits(
        token: Int,
        previousHidden: MLXArray,
        positionOffset: Int,
        baseModel: Q35Model
    ) -> MLXArray {
        let inputIds = MLXArray([Int32(token)]).reshaped(1, 1)
        let embeddings = baseModel.embeddings(for: inputIds)
        let cache = Q35MTPPositionCache(offset: positionOffset)
        let output = forwardDraft(
            inputEmbeddings: embeddings,
            hiddenStates: previousHidden,
            cache: cache
        )
        return baseModel.logits(from: output.logitsHidden)
    }

    func draftBlock(
        lastToken: Int,
        hidden: MLXArray,
        blockSize: Int,
        session: Q35MTPDraftSession,
        baseModel: Q35Model
    ) -> Q35MTPDraftBlock {
        session.draftBlock(
            lastToken: lastToken,
            hidden: hidden,
            blockSize: blockSize,
            mtpModel: self,
            baseModel: baseModel
        )
    }

    @discardableResult
    func prepareFusedSwitchGLU() -> Bool {
        guard let layer = layers.first else { return false }
        let prepared = layer.mlp.prepareFusedSwitchGLU()
        if prepared {
            Stream.gpu.synchronize()
            Memory.clearCache()
        }
        return prepared
    }
}

/// Request-local committed-history cache for greedy MTP proposals.
///
/// The persistent cache contains only target-confirmed transitions. Draft-only
/// rows run on a fork and are discarded after the round, so proposal history
/// can improve acceptance without becoming an output authority.
final class Q35MTPDraftSession {
    private static let historyChunkSize = 256
    private let historyCache: KVCache
    private var backlogHidden: [MLXArray] = []
    private var backlogTokens: [Int] = []

    init(promptTokens: [Int] = [], promptHidden: MLXArray? = nil, historyCache: KVCache = KVCacheSimple()) {
        self.historyCache = historyCache
        guard let promptHidden,
              promptTokens.count > 1,
              promptHidden.dim(1) == promptTokens.count else {
            return
        }
        backlogHidden.append(promptHidden[0..., 0..<(promptTokens.count - 1), 0...])
        backlogTokens.append(contentsOf: promptTokens.dropFirst())
    }

    var committedHistoryCount: Int {
        historyCache.offset + backlogTokens.count
    }

    var pendingHistoryCount: Int { backlogTokens.count }

    /// Prefix checkpoints never share a mutable draft cache with a request.
    func fork() -> Q35MTPDraftSession {
        let copy = Q35MTPDraftSession(historyCache: historyCache.fork())
        copy.backlogHidden = backlogHidden
        copy.backlogTokens = backlogTokens
        return copy
    }

    /// Materialize complete history blocks during target prefill. Retaining the
    /// incomplete block preserves the same 256-token boundaries as cold MTP
    /// priming and keeps fewer than 256 pending transitions, rather than a
    /// prompt-wide hidden history. A tail view can retain its prefill chunk.
    func primeCommittedHistory(mtpModel: any Q35MTPDraftModel, baseModel: Q35Model) {
        let count = backlogTokens.count / Self.historyChunkSize * Self.historyChunkSize
        guard count > 0 else { return }
        let output = flushCommittedHistory(count: count, mtpModel: mtpModel, baseModel: baseModel)
        MLX.eval(output.recurrentHidden)
    }

    func recordCommittedTransitions(hiddenStates: MLXArray, nextTokens: [Int]) {
        guard !nextTokens.isEmpty,
              hiddenStates.dim(1) >= nextTokens.count else {
            return
        }
        backlogHidden.append(hiddenStates[0..., 0..<nextTokens.count, 0...])
        backlogTokens.append(contentsOf: nextTokens)
    }

    /// Reuse the target's accepted prefix after cache rollback. Flash-Next's
    /// predictor consumes the four-stream residual, not the reduced LM hidden.
    func restoredVerificationState(
        from output: Q35ForwardOutput,
        acceptedTokens: [Int]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let hidden = output.mtpHidden ?? output.hidden
        recordCommittedTransitions(hiddenStates: hidden, nextTokens: acceptedTokens)
        let last = acceptedTokens.count
        return (
            output.logits[0..., last..<(last + 1), 0...],
            hidden[0..., last..<(last + 1), 0...]
        )
    }

    func draftBlock(
        lastToken: Int,
        hidden: MLXArray,
        blockSize: Int,
        mtpModel: any Q35MTPDraftModel,
        baseModel: Q35Model
    ) -> Q35MTPDraftBlock {
        let total = max(1, blockSize) - 1
        guard total > 0 else {
            return Q35MTPDraftBlock(tokenIDs: MLXArray.zeros([1, 0], dtype: .int32))
        }

        backlogHidden.append(hidden)
        backlogTokens.append(lastToken)
        let flushed = flushCommittedHistory(
            count: backlogTokens.count, mtpModel: mtpModel, baseModel: baseModel
        )
        var previousHidden = flushed.recurrentHidden[
            0...,
            (flushed.recurrentHidden.dim(1) - 1)...,
            0...
        ]
        let lastLogitsHidden = flushed.logitsHidden[
            0...,
            (flushed.logitsHidden.dim(1) - 1)...,
            0...
        ]
        var tokenArray = baseModel.greedyDraftToken(from: lastLogitsHidden)
        var tokenArrays = [tokenArray]
        tokenArrays.reserveCapacity(total)

        if total > 1 {
            let speculativeCache = historyCache.fork()
            for _ in 1..<total {
                let embeddings = baseModel.embeddings(for: tokenArray)
                let output = mtpModel.forwardDraft(
                    inputEmbeddings: embeddings,
                    hiddenStates: previousHidden,
                    cache: speculativeCache
                )
                previousHidden = output.recurrentHidden
                tokenArray = baseModel.greedyDraftToken(from: output.logitsHidden)
                tokenArrays.append(tokenArray)
            }
        }

        let draftTokens = MLX.concatenated(tokenArrays, axis: 1)
        MLX.asyncEval(draftTokens)
        return Q35MTPDraftBlock(tokenIDs: draftTokens)
    }

    private func flushCommittedHistory(
        count: Int,
        mtpModel: any Q35MTPDraftModel,
        baseModel: Q35Model
    ) -> Q35MTPDraftOutput {
        precondition(count > 0 && count <= backlogTokens.count)
        let flushHidden = backlogHidden.count == 1
            ? backlogHidden[0]
            : MLX.concatenated(backlogHidden, axis: 1)
        let flushTokens = MLXArray(backlogTokens.prefix(count).map(Int32.init)).reshaped(1, count)
        backlogHidden = count < backlogTokens.count ? [flushHidden[0..., count..., 0...]] : []
        backlogTokens = Array(backlogTokens.dropFirst(count))

        // A long prompt must not become one giant MTP MoE/attention graph.
        let chunkSize = Self.historyChunkSize
        var processed = 0
        while count - processed > chunkSize {
            let end = processed + chunkSize
            let partial = mtpModel.forwardDraft(
                inputEmbeddings: baseModel.embeddings(for: flushTokens[0..., processed..<end]),
                hiddenStates: flushHidden[0..., processed..<end, 0...],
                cache: historyCache
            )
            MLX.eval(partial.recurrentHidden)
            processed = end
        }
        let flushEmbeddings = baseModel.embeddings(for: flushTokens[0..., processed...])
        return mtpModel.forwardDraft(
            inputEmbeddings: flushEmbeddings,
            hiddenStates: flushHidden[0..., processed..<count, 0...],
            cache: historyCache
        )
    }
}
