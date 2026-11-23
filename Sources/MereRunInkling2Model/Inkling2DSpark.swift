import Foundation
import MLX

final class Inkling2DSparkFeatures {
    private let visitIDs: [Int]
    private let visits: Set<Int>
    private var states: [Int: MLXArray] = [:]

    init(_ visitIDs: [Int]) {
        self.visitIDs = visitIDs
        visits = Set(visitIDs)
    }

    func record(_ state: MLXArray, at visit: Int) {
        if visits.contains(visit) { states[visit] = state }
    }

    func joined() throws -> MLXArray {
        let ordered = try visitIDs.map { visit in
            guard let state = states[visit] else { throw Inkling2Error.invalid("missing DSpark target feature at visit \(visit)") }
            return state
        }
        return stacked(ordered, axis: -2)
    }
}

struct Inkling2DSparkWeights {
    let arrays: [String: MLXArray]

    func read(_ name: String, _ shape: [Int]) throws -> MLXArray {
        guard let array = arrays[name], array.shape == shape, array.dtype == .bfloat16 else {
            throw Inkling2Error.invalid("invalid DSpark weight \(name)")
        }
        return array
    }
}

struct Inkling2DSparkLayerCache {
    var keys: MLXArray?
    var values: MLXArray?
    var chronologicalBank = false

    mutating func write(_ k: MLXArray, _ v: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let range = offset..<(offset + k.dim(2))
        let newKeys = keys!.reshaped(keys!.shape)
        let newValues = values!.reshaped(values!.shape)
        newKeys[0..., 0..., range, 0...] = k
        newValues[0..., 0..., range, 0...] = v
        keys = newKeys
        values = newValues
        return (newKeys[0..., 0..., ..<range.upperBound, 0...], newValues[0..., 0..., ..<range.upperBound, 0...])
    }
}

struct Inkling2DSparkLayer {
    let queryKeyValue: MLXArray
    let contextKeyValue: MLXArray
    let attentionOutput: MLXArray
    let queryNorm: MLXArray
    let keyNorm: MLXArray
    let contextKeyNorm: MLXArray
    let contextNorm: MLXArray
    let inputNorm: MLXArray
    let postAttentionNorm: MLXArray
    let gateUp: MLXArray
    let down: MLXArray
    let config: Inkling2DSparkConfiguration
    let window: Int?

    init(weights: Inkling2DSparkWeights, index: Int, config: Inkling2DSparkConfiguration) throws {
        self.config = config
        window = config.localLayerIds.contains(index) ? config.slidingWindowSize : nil
        let hidden = config.hiddenSize, head = config.headDim, intermediate = config.intermediateSize
        let qWidth = config.numAttentionHeads * head, kvWidth = config.numKeyValueHeads * head
        let prefix = "layers.\(index)."
        func read(_ suffix: String, _ shape: [Int]) throws -> MLXArray {
            try weights.read(prefix + suffix + ".weight", shape)
        }
        queryKeyValue = try concatenated([read("self_attn.q_proj", [qWidth, hidden]),
                                          read("self_attn.k_proj", [kvWidth, hidden]),
                                          read("self_attn.v_proj", [kvWidth, hidden])], axis: 0)
        contextKeyValue = try concatenated([read("self_attn.k_proj_ctx", [kvWidth, hidden]),
                                            read("self_attn.v_proj_ctx", [kvWidth, hidden])], axis: 0)
        attentionOutput = try read("self_attn.o_proj", [hidden, qWidth])
        queryNorm = try read("self_attn.q_norm", [head])
        keyNorm = try read("self_attn.k_norm", [head])
        contextKeyNorm = try read("self_attn.k_norm_ctx", [head])
        contextNorm = try read("dflare_norm", [hidden])
        inputNorm = try read("input_layernorm", [hidden])
        postAttentionNorm = try read("post_attention_layernorm", [hidden])
        gateUp = try concatenated([read("mlp.gate_proj", [intermediate, hidden]),
                                    read("mlp.up_proj", [intermediate, hidden])], axis: 0)
        down = try read("mlp.down_proj", [hidden, intermediate])
        eval(queryKeyValue, contextKeyValue, gateUp)
    }

    private func heads(_ x: MLXArray, count: Int) -> MLXArray {
        x.reshaped(1, -1, count, config.headDim).transposed(0, 2, 1, 3)
    }

    private func rotate(_ x: MLXArray, norm: MLXArray, count: Int, offset: Int) -> MLXArray {
        let x = MLXFast.rmsNorm(heads(x, count: count), weight: norm, eps: config.rmsNormEps)
        return MLXFast.RoPE(x, dimensions: config.headDim, traditional: false, base: config.ropeTheta, scale: 1, offset: offset)
    }

    func append(_ context: MLXArray, offset: Int, cache: inout Inkling2DSparkLayerCache) {
        let count = config.numKeyValueHeads, width = count * config.headDim
        let kv = matmul(context, contextKeyValue.T)
        let k = rotate(kv[.ellipsis, ..<width], norm: contextKeyNorm, count: count, offset: offset)
        let v = heads(kv[.ellipsis, width...], count: count)
        if cache.chronologicalBank {
            _ = cache.write(k, v, offset: offset)
            return
        }
        let keys = cache.keys.map { concatenated([$0, k], axis: 2) } ?? k
        let values = cache.values.map { concatenated([$0, v], axis: 2) } ?? v
        let start = max(0, keys.dim(2) - (window ?? keys.dim(2)))
        cache.keys = keys[0..., 0..., start..., 0...]
        cache.values = values[0..., 0..., start..., 0...]
    }

    func forward(_ input: MLXArray, offset: Int, cache: inout Inkling2DSparkLayerCache,
                 queries: Range<Int>? = nil) -> MLXArray {
        let h = config.numAttentionHeads, kv = config.numKeyValueHeads, d = config.headDim
        let qEnd = h * d, kEnd = qEnd + kv * d
        let normalized = MLXFast.rmsNorm(input.asType(.float32), weight: inputNorm.asType(.float32), eps: config.rmsNormEps)
        let qkv = matmul(normalized.asType(.bfloat16), queryKeyValue.T)
        let query = queries.map { qkv[0..., $0, ..<qEnd] } ?? qkv[.ellipsis, ..<qEnd]
        let q = rotate(query, norm: queryNorm, count: h, offset: offset + (queries?.lowerBound ?? 0))
        let k = rotate(qkv[.ellipsis, qEnd..<kEnd], norm: keyNorm, count: kv, offset: offset)
        let v = heads(qkv[.ellipsis, kEnd...], count: kv)
        let keys: MLXArray, values: MLXArray
        if cache.chronologicalBank {
            (keys, values) = cache.write(k, v, offset: offset)
        } else {
            keys = cache.keys.map { concatenated([$0, k], axis: 2) } ?? k
            values = cache.values.map { concatenated([$0, v], axis: 2) } ?? v
        }
        let attention = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: 1 / sqrt(Float(d)), mask: .none
        ).transposed(0, 2, 1, 3).reshaped(1, -1, qEnd)
        let residualInput = queries.map { input[0..., $0, 0...] } ?? input
        let residual = residualInput.asType(.bfloat16).asType(.float32) + matmul(attention, attentionOutput.T).asType(.float32)
        let x = MLXFast.rmsNorm(residual, weight: postAttentionNorm.asType(.float32), eps: config.rmsNormEps)
        let gated = matmul(x.asType(.bfloat16), gateUp.T).asType(.float32)
        let mid = config.intermediateSize
        let gate = gated[.ellipsis, ..<mid]
        let activated = (gate * sigmoid(gate) * gated[.ellipsis, mid...]).asType(.bfloat16)
        return residual.asType(.bfloat16).asType(.float32) + matmul(activated, down.T).asType(.float32)
    }
}

private enum Inkling2DSparkProjection {
    case dense(MLXArray)
    case q8(Inkling2Q8Projection)
    case reference(Inkling2ReferenceAffineProjection)

    init(_ weight: MLXArray, useQ8: Bool) {
        if useQ8, weight.dim(-1).isMultiple(of: 64) {
            let (packed, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 8, mode: .affine)
            eval([packed, scales] + (biases.map { [$0] } ?? []))
            self = .q8(Inkling2Q8Projection(weight: packed, scales: scales, biases: biases))
        } else {
            self = .dense(weight)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        switch self {
        case let .dense(weight):
            matmul(input, weight.T)
        case let .q8(projection):
            projection(input)
        case let .reference(projection):
            projection(input)
        }
    }
}

package final class Inkling2DSpark {
    let config: Inkling2DSparkConfiguration
    let target: Inkling2Model
    package var maxProposalCount: Int { config.blockSize - 1 }
    private let layers: [Inkling2DSparkLayer]
    private let mixWeights: MLXArray
    private let maskEmbedding: MLXArray
    private let norm: MLXArray
    private let markovNorm: MLXArray
    private let markovEmbedding: MLXArray
    private let recurrent: MLXArray
    private let recurrentContext: MLXArray
    private let recurrentBias: MLXArray
    private let head: Inkling2DSparkProjection
    private let headBias: MLXArray
    private let baseHead: Inkling2DSparkProjection

    package init(target: Inkling2Model, weightsURL: URL) throws {
        let configURL = weightsURL.deletingLastPathComponent().appendingPathComponent("config.json")
        let config = try Inkling2DSparkConfiguration.decode(Data(contentsOf: configURL))
        try config.validate(against: target.config.textConfig)
        self.config = config
        self.target = target
        let weights = Inkling2DSparkWeights(arrays: try loadArrays(url: weightsURL))
        let hidden = config.hiddenSize, rank = config.markovRank
        let vocab = target.config.textConfig.vocabSize
        layers = try (0..<config.numHiddenLayers).map { try Inkling2DSparkLayer(weights: weights, index: $0, config: config) }
        mixWeights = try softmax(weights.read("fuse_weight", [config.numHiddenLayers + 1, config.targetLayerIds.count])
            .asType(.float32), axis: -1)
        maskEmbedding = try weights.read("mask_embed", [hidden])
        norm = try weights.read("norm.weight", [hidden]).asType(.float32)
        markovNorm = try weights.read("markov_head.dflare_norm.weight", [hidden])
        markovEmbedding = try weights.read("markov_head.markov_w1.weight", [vocab, rank])
        let rnn = try weights.read("markov_head.rnn_proj.weight", [3 * rank, 2 * (rank + hidden)]).asType(.float32)
        recurrent = rnn[0..., ..<(2 * rank)]
        recurrentContext = rnn[0..., (2 * rank)...]
        recurrentBias = try weights.read("markov_head.rnn_proj.bias", [3 * rank]).asType(.float32)
        let outputRows = target.config.textConfig.unpaddedVocabSize
        let headKey = target.prefix + "unembed.weight"
        if let q8OutputHead = target.q8OutputHead {
            baseHead = .q8(q8OutputHead)
            head = try .q8(Inkling2Q8Projection(weights: target.weights, key: Inkling2Q8Projection.draftKey,
                                                rows: outputRows, columns: rank))
            let biasKey = Inkling2Q8Projection.draftBiasKey
            try target.weights.require(biasKey, shape: [outputRows], dtype: "BF16")
            headBias = try target.weights.read(biasKey)
            eval(mixWeights, recurrent, recurrentContext, recurrentBias, headBias)
        } else {
            let base: MLXArray
            if let root = target.weights.speedDraftRoot {
                let draftWeights = try Inkling2Weights(root: root.appendingPathComponent(Inkling2Weights.speedDraftHeadDirectory))
                let foldWeights = try Inkling2Weights(root: root.appendingPathComponent(Inkling2Weights.speedDraftFoldDirectory))
                #if os(macOS)
                draftWeights.readsIntoOwnedBuffers = target.usePackedExperts
                foldWeights.readsIntoOwnedBuffers = target.usePackedExperts
                #endif
                try draftWeights.require(headKey, shape: [vocab, hidden / 8], dtype: "U32")
                for suffix in [".mtplx_scales", ".mtplx_biases"] {
                    try draftWeights.require(headKey + suffix, shape: [vocab, hidden / 64], dtype: "BF16")
                }
                guard let projection = try draftWeights.referenceProjection(headKey),
                      projection.spec.bits == 4, projection.spec.groupSize == 64 else {
                    throw Inkling2Error.invalid("invalid Speed pack draft output projection")
                }
                try foldWeights.require(headKey, shape: [vocab, hidden], dtype: "BF16")
                base = try foldWeights.read(headKey, retain: false)[..<outputRows]
                baseHead = .reference(projection.slice(0..<outputRows))
            } else if let affine = try target.weights.referenceProjection(headKey) {
                let output = affine.slice(0..<outputRows)
                base = output.unpacked()
                baseHead = .reference(output)
            } else {
                base = try target.weights.read(headKey)[..<outputRows]
                baseHead = Inkling2DSparkProjection(base, useQ8: target.usePackedExperts)
            }
            let correction = try matmul(base, weights.read("markov_head.hidden_out.weight", [hidden, rank]))
            headBias = try matmul(base, weights.read("markov_head.hidden_out.bias", [hidden]))
            head = Inkling2DSparkProjection(correction, useQ8: target.usePackedExperts)
            eval(mixWeights, recurrent, recurrentContext, recurrentBias, correction, headBias)
        }
    }

    private func mix(_ features: MLXArray, row: Int, norm: MLXArray) -> MLXArray {
        let alpha = mixWeights[row].reshaped(1, 1, -1, 1)
        let mixed = (features.asType(.float32) * alpha).sum(axis: -2)
        return MLXFast.rmsNorm(mixed, weight: norm.asType(.float32), eps: config.rmsNormEps).asType(.bfloat16)
    }

    func append(_ features: MLXArray, offset: Int, caches: inout [Inkling2DSparkLayerCache]) {
        for (index, layer) in layers.enumerated() {
            layer.append(mix(features, row: index, norm: layer.contextNorm), offset: offset, cache: &caches[index])
        }
    }

    func prepareChronologicalBank(offset: Int, budget: Int, caches: inout [Inkling2DSparkLayerCache]) {
        let cap = ((offset + budget + config.blockSize + 511) / 512) * 512
        for (index, layer) in layers.enumerated() where layer.window == nil {
            guard let keys = caches[index].keys, let values = caches[index].values, keys.dim(2) == offset else { continue }
            let padding = MLXArray.zeros([keys.dim(0), keys.dim(1), cap - offset, keys.dim(3)], dtype: keys.dtype)
            caches[index].keys = concatenated([keys, padding], axis: 2)
            caches[index].values = concatenated([values, padding], axis: 2)
            caches[index].chronologicalBank = true
        }
    }

    func propose(anchor: MLXArray, count: Int, features: MLXArray, offset: Int,
                 caches: inout [Inkling2DSparkLayerCache], distribution: ((MLXArray) -> MLXArray)?) throws
        -> (tokens: MLXArray, probabilities: [MLXArray]) {
        let embedding = try target.embeddings(anchor.reshaped(1, 1))
        let masks = broadcast(maskEmbedding, to: [1, config.blockSize - 1, config.hiddenSize])
        var x = concatenated([embedding, masks], axis: 1)
        for (index, layer) in layers.enumerated() {
            // The final layer still attends to the whole block, but we only use the proposal outputs.
            let queries = index == layers.count - 1 ? 1..<(count + 1) : nil
            x = layer.forward(x, offset: offset, cache: &caches[index], queries: queries)
        }
        let hidden = MLXFast.rmsNorm(x, weight: norm, eps: config.rmsNormEps).asType(.bfloat16)
        let f = mix(features, row: layers.count, norm: markovNorm)[0..., -1, 0...]
        let contextInput = concatenated([hidden, broadcast(f, to: hidden.shape)], axis: -1)
        let context = matmul(contextInput.asType(.float32), recurrentContext.T) + recurrentBias
        let base = baseHead(hidden)
        let rank = config.markovRank
        var state = MLXArray.zeros([1, rank], dtype: .bfloat16)
        var previous = anchor.reshaped(1)
        var proposed = [anchor.reshaped(1)]
        var probabilities: [MLXArray] = []
        for i in 0..<count {
            let recurrentInput = concatenated([state, markovEmbedding[previous]], axis: -1).asType(.float32)
            let raw = (matmul(recurrentInput, recurrent.T) + context[0..., i, 0...]).asType(.bfloat16)
            let gate = sigmoid(raw[0..., ..<rank])
            state = gate * state + (1 - gate) * tanh(raw[0..., rank..<(2 * rank)])
            let latent = tanh(raw[0..., (2 * rank)...])
            let correction = head(latent) + headBias
            let scores = (base[0, i] + correction).asType(.float32)
            let logits = Inkling2Math.softcap(scores, target.config.textConfig.finalLogitSoftcapping).asType(.bfloat16)
            if let distribution {
                let probability = distribution(logits[0])
                previous = MLXRandom.categorical(log(probability)).asType(.int32).reshaped(1)
                probabilities.append(probability)
            } else {
                previous = argMax(logits, axis: -1).asType(.int32)
            }
            proposed.append(previous)
        }
        return (concatenated(proposed, axis: 0).reshaped(1, -1), probabilities)
    }
}
