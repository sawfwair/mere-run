import MLX

package final class UnlimitedOCRDecoder {
    package let config: UnlimitedOCRConfiguration
    package let weights: UnlimitedOCRWeights
    private let experts: [[String: MLXArray]]

    package init(config: UnlimitedOCRConfiguration, tensors: [String: MLXArray]) {
        self.config = config
        weights = UnlimitedOCRWeights(tensors: tensors.filter {
            ($0.key.hasPrefix("model.layers.") && !$0.key.contains(".mlp.experts."))
                || ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"].contains($0.key)
        })
        experts = (0..<config.numHiddenLayers).map { layer in
            guard layer >= config.firstKDenseReplace else { return [:] }
            return Dictionary(uniqueKeysWithValues: ["gate_proj", "up_proj", "down_proj"].map { projection in
                let stacked = stacked((0..<config.nRoutedExperts).map {
                    tensors["model.layers.\(layer).mlp.experts.\($0).\(projection).weight"]!
                })
                eval(stacked)
                return (projection, stacked)
            })
        }
    }

    package func embeddings(_ ids: [Int]) -> MLXArray {
        weights.tensor("model.embed_tokens.weight").take(MLXArray(ids), axis: 0)
    }

    package func forward(
        _ embeddings: MLXArray, state: UnlimitedOCRPageState, cache: [UnlimitedOCRCache],
        trace: ((String, MLXArray) -> Void)? = nil,
        transform: ((String, MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        let length = embeddings.dim(0)
        let offset = state.tokenPositions.count - length
        let tok = MLXArray(Array(state.tokenPositions.suffix(length)))
        let page = MLXArray(Array(state.pagePositions.suffix(length)))
        var hidden = embeddings
        for layer in 0..<config.numHiddenLayers {
            hidden = block(hidden, layer: layer, tok: tok, page: page, state: state, offset: offset, cache: cache[layer])
            hidden = transform?("decoder.block.\(layer)", hidden) ?? hidden
            trace?("decoder.block.\(layer)", hidden)
        }
        var last = weights.rms(hidden[(-1)...], "model.norm")
        last = transform?("decoder.norm", last) ?? last
        trace?("decoder.norm", last)
        let logits = weights.linear(last, "lm_head").asType(.float32)
        trace?("logits", logits)
        return logits.squeezed(axis: 0)
    }

    package func block(
        _ input: MLXArray, layer: Int, tok: MLXArray, page: MLXArray,
        state: UnlimitedOCRPageState, offset: Int, cache: UnlimitedOCRCache,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let prefix = "model.layers.\(layer)"
        let x = weights.rms(input, prefix + ".input_layernorm")
        trace?("input_layernorm", x)
        let attended = attention(x, prefix: prefix + ".self_attn", layer: layer,
                                tok: tok, page: page, state: state, offset: offset, cache: cache)
        trace?("self_attn.0", attended)
        var hidden = input + attended
        let normalized = weights.rms(hidden, prefix + ".post_attention_layernorm")
        trace?("post_attention_layernorm", normalized)
        let feedforward = layer < config.firstKDenseReplace
            ? mlp(normalized, prefix: prefix + ".mlp") : moe(normalized, layer: layer, prefix: prefix + ".mlp")
        trace?("mlp", feedforward)
        hidden = hidden + feedforward
        eval(hidden)
        return hidden
    }

    private func mlp(_ x: MLXArray, prefix: String) -> MLXArray {
        weights.linear(UnlimitedOCRMath.silu(weights.linear(x, prefix + ".gate_proj")) * weights.linear(x, prefix + ".up_proj"), prefix + ".down_proj")
    }

    private func moe(_ x: MLXArray, layer: Int, prefix: String) -> MLXArray {
        let scores = softmax(weights.linear(x.asType(.float32), prefix + ".gate"), axis: -1)
        let indices = argPartition(-scores, kth: config.numExpertsPerTok - 1, axis: -1)[.ellipsis, ..<config.numExpertsPerTok]
        // Released norm_topk_prob=false: preserve the selected softmax mass.
        let routing = takeAlong(scores, indices, axis: -1)
        func projection(_ value: MLXArray, _ name: String) -> MLXArray {
            gatherMM(value, experts[layer][name]!.swappedAxes(-1, -2), rhsIndices: indices).squeezed(axis: -2)
        }
        let input = x.reshaped(x.dim(0), 1, 1, -1)
        let activated = UnlimitedOCRMath.silu(projection(input, "gate_proj")) * projection(input, "up_proj")
        let routed = projection(activated.expandedDimensions(axis: -2), "down_proj")
        return UnlimitedOCRMath.mixExperts(routed, probabilities: routing)
            + mlp(x, prefix: prefix + ".shared_experts")
    }

    private func attention(
        _ x: MLXArray, prefix: String, layer: Int, tok: MLXArray, page: MLXArray,
        state: UnlimitedOCRPageState, offset: Int, cache: UnlimitedOCRCache
    ) -> MLXArray {
        let heads = config.numAttentionHeads
        let dim = config.hiddenSize / heads
        func project(_ name: String) -> MLXArray {
            weights.linear(x, prefix + name).reshaped(-1, heads, dim).transposed(1, 0, 2)
        }
        let q = UnlimitedOCRMath.rope(project(".q_proj"), tokens: tok, pages: page)
        let newK = UnlimitedOCRMath.rope(project(".k_proj"), tokens: tok, pages: page)
        let newV = project(".v_proj")
        let k = cache.keys.map { concatenated([$0, newK], axis: 1) } ?? newK
        let v = cache.values.map { concatenated([$0, newV], axis: 1) } ?? newV
        let router = prefix + ".dash_router"
        func index(_ name: String) -> MLXArray {
            let normalized = UnlimitedOCRMath.routerRMS(weights.linear(x, router + name + "_proj"),
                                                       weight: weights.tensor(router + name + "_norm.weight"))
            return UnlimitedOCRMath.rope(normalized.expandedDimensions(axis: 0), tokens: tok, pages: page)
                .squeezed(axis: 0).asType(.float32)
        }
        let iq = index(".q")
        let newIndex = index(".k")
        let ik = cache.indexKeys.map { concatenated([$0, newIndex], axis: 0) } ?? newIndex
        cache.keys = k
        cache.values = v
        cache.indexKeys = ik
        let sparse = config.sparseAttentionConfig
        let output = UnlimitedOCRAttention.attend(
            q: q, k: k, v: v, indexQ: iq, indexK: ik,
            head: weights.tensor(router + ".head_cls").asType(.float32), state: state, offset: offset,
            scale: sparse.dashScalePerLayer[layer], floor: sparse.dashFloorPerLayer[layer], sigma: sparse.dashSigma
        )
        return weights.linear(output.transposed(1, 0, 2).reshaped(-1, config.hiddenSize), prefix + ".o_proj")
    }
}
