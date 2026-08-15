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

    init(config: Q35Config) {
        let text = config.textConfig
        self.intermediateSize = text.moeIntermediateSize
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

        let gateUp = gatherMM(
            flatX,
            gateUpProj.swappedAxes(-1, -2),
            rhsIndices: flatIndices,
            sortedIndices: false
        )
        let gate = gateUp[.ellipsis, 0..<intermediateSize]
        let up = gateUp[.ellipsis, intermediateSize...]
        let activated = q35MTPSwiglu(gate, up)

        let output = gatherMM(
            activated,
            downProj.swappedAxes(-1, -2),
            rhsIndices: flatIndices,
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

final class Q35MTPModel: Module {
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: Q35RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: Q35RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Q35MTPDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Q35RMSNorm

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
        positionOffset: Int
    ) -> MLXArray {
        let normalizedEmbeddings = preFCNormEmbedding(inputEmbeddings)
        let normalizedHidden = preFCNormHidden(hiddenStates)
        var hidden = MLX.concatenated([normalizedEmbeddings, normalizedHidden], axis: -1)
        hidden = fc(hidden)

        let cache = Q35MTPPositionCache(offset: positionOffset)
        let mask = cache.makeMask(n: hidden.dim(1))
        for layer in layers {
            hidden = layer(hidden, mask: mask, cache: cache)
        }
        return norm(hidden)
    }

    func draftLogits(
        token: Int,
        previousHidden: MLXArray,
        positionOffset: Int,
        baseModel: Q35Model
    ) -> MLXArray {
        let inputIds = MLXArray([Int32(token)]).reshaped(1, 1)
        let embeddings = baseModel.embeddings(for: inputIds)
        let hidden = self(
            inputEmbeddings: embeddings,
            hiddenStates: previousHidden,
            positionOffset: positionOffset
        )
        return baseModel.logits(from: hidden)
    }

    func draftBlock(
        lastToken: Int,
        hidden: MLXArray,
        positionOffset: Int,
        blockSize: Int,
        baseModel: Q35Model
    ) -> [Int] {
        let total = max(1, blockSize) - 1
        guard total > 0 else {
            return []
        }

        var tokenArray = MLXArray([Int32(lastToken)]).reshaped(1, 1)
        var previousHidden = hidden
        var tokenArrays: [MLXArray] = []
        tokenArrays.reserveCapacity(total)
        for index in 0..<total {
            let embeddings = baseModel.embeddings(for: tokenArray)
            previousHidden = self(
                inputEmbeddings: embeddings,
                hiddenStates: previousHidden,
                positionOffset: positionOffset + index
            )
            let draftLogits = baseModel.logits(from: previousHidden)[0, -1, 0...]
            tokenArray = argMax(draftLogits, axis: -1).asType(.int32).reshaped(1, 1)
            tokenArrays.append(tokenArray)
        }

        let draftTokens = MLX.concatenated(tokenArrays, axis: 1)
        MLX.eval(draftTokens)
        return draftTokens.asArray(Int32.self).map(Int.init)
    }
}
