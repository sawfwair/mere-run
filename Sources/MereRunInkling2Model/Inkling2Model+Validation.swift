import MLX

extension Inkling2Model {
    func validateWeights() throws {
        let c = config.textConfig, d = c.hiddenSize
        var expected: [String: [Int]] = [
            prefix + "embed.weight": [c.vocabSize, d], prefix + "embed_norm.weight": [d],
            prefix + "norm.weight": [d], prefix + "unembed.weight": [c.vocabSize, d],
            prefix + "attn_res.output_query": [d]
        ]
        for i in 0..<c.numHiddenLayers {
            let p = prefix + "layers.\(i).", local = c.isLocal(i)
            let heads = local ? c.swaNumAttentionHeads : c.numAttentionHeads
            let kv = local ? c.swaNumKeyValueHeads : c.numKeyValueHeads
            let headDim = local ? c.swaHeadDim : c.headDim
            for name in ["attn_sconv", "mlp_sconv"] { expected[p + name + ".weight"] = [d, 1, c.sconvKernelSize] }
            expected[p + "mlp_norm.weight"] = [d]
            if local { expected[p + "attn_norm.weight"] = [d] }
            else { for name in ["q", "k", "v"] { expected[p + name + "_input_norm.weight"] = [d] } }
            if i > 0 { expected[prefix + "attn_res.attn_queries.\(i - 1)"] = local ? [d] : [3, d] }
            expected[prefix + "attn_res.mlp_queries.\(i)"] = [d]
            expected[p + "mlp.gate.weight"] = [c.nRoutedExperts, d]
            expected[p + "mlp.gate.bias"] = [c.nRoutedExperts]
            let a = p + "attn."
            expected[a + "wq_du.weight"] = [heads * headDim, d]
            expected[a + "wk_dv.weight"] = [kv * headDim, d]
            expected[a + "wv_dv.weight"] = [kv * headDim, d]
            expected[a + "wr_du.weight"] = [heads * c.dRel, d]
            expected[a + "wo_ud.weight"] = [d, heads * headDim]
            expected[a + "gate_attn.weight"] = [heads * headDim, d]
            expected[a + "attn_sink_proj.weight"] = [heads, d]
            expected[a + "q_l2_clip.scale"] = [heads, headDim]
            expected[a + "rel_l2_clip.scale"] = [heads, c.dRel]
            expected[a + "rel_logits_proj.proj"] = [c.dRel, local ? c.slidingWindowSize : c.relExtent]
            for name in ["k_sconv", "v_sconv"] {
                if local { expected[a + name + ".weight"] = [kv * headDim, 1, c.sconvKernelSize] }
                else {
                    expected[a + name + ".static_w"] = [c.sconvKernelSize, kv * headDim]
                    expected[a + name + ".weight_proj"] = [kv * c.sconvKernelSize, d]
                }
            }
            for (name, output, input) in [("w13_weight", 2 * c.intermediateSize, d), ("w2_weight", d, c.intermediateSize)] {
                let key = p + "mlp.experts." + name
                let type = weights.dtype(key)
                let packed = type == "U8"
                let fp8 = type == "F8_E4M3" || type == "F8_E4M3FN"
                if packed || fp8 {
                    let group = packed ? 16 : 32
                    guard input.isMultiple(of: group) else { throw Inkling2Error.invalid("expert input must divide its quantization group") }
                    try weights.require(key, shape: [c.nRoutedExperts, output, packed ? input / 2 : input], bytes: true)
                    try weights.require(key + "_scale", shape: [c.nRoutedExperts, output, input / group], bytes: true)
                    let scaleType = weights.dtype(key + "_scale")
                    guard packed ? ["F8_E4M3", "F8_E4M3FN"].contains(scaleType ?? "") : scaleType == "U8" else {
                        throw Inkling2Error.invalid("incorrect packed expert scale encoding: \(key)")
                    }
                    if packed {
                        try weights.require(key + "_amax", shape: [c.nRoutedExperts])
                        try weights.require(key + "_outer_scale_denominator", shape: [1])
                        expected[key + "_amax"] = [c.nRoutedExperts]
                        expected[key + "_outer_scale_denominator"] = [1]
                    }
                    expected[key] = weights.shape(key)
                    expected[key + "_scale"] = weights.shape(key + "_scale")
                } else { expected[key] = [c.nRoutedExperts, output, input] }
            }
        }
        for (key, shape) in expected {
            try weights.require(key, shape: shape, bytes: key.contains(".experts.") && !key.hasSuffix("_amax") && !key.hasSuffix("_denominator"))
        }
        let textKeys = weights.names.filter { $0.hasPrefix(prefix) }
        guard Set(expected.keys) == Set(textKeys) else { throw Inkling2Error.invalid("unexpected text checkpoint tensors") }
        if let v = config.visionConfig { try validateVision(v) }
    }

    func validateVision(_ v: Inkling2VisionConfiguration) throws {
        let p = "model.visual.", d = v.hiddenSize, intermediate = Int(Float(d) * v.mlpRatio)
        var expected: [String: [Int]] = [
            p + "backbone.embeddings.class_embedding": [d], p + "backbone.embeddings.cls_position_embedding": [d],
            p + "backbone.embeddings.patch_embedding.weight": [d, v.nativePatchSize * v.nativePatchSize * 3],
            p + "backbone.embeddings.position_embedding": [v.posembGridSize * v.posembGridSize, d],
            p + "backbone.ln_pre.weight": [d], p + "backbone.ln_pre.bias": [d],
            p + "proj_in.weight": [v.projectionHiddenDim, v.adapterMergeSize * v.adapterMergeSize * d],
            p + "proj_out.weight": [config.textConfig.hiddenSize, v.projectionHiddenDim]
        ]
        for i in 0..<v.numHiddenLayers {
            let l = p + "backbone.layers.\(i)."
            for n in ["ln_1", "ln_2"] { expected[l + n + ".weight"] = [d]; expected[l + n + ".bias"] = [d] }
            for n in ["ls_1", "ls_2"] { expected[l + n + ".gamma"] = [d] }
            for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                expected[l + "self_attn." + n + ".weight"] = [d, d]
                expected[l + "self_attn." + n + ".bias"] = [d]
            }
            expected[l + "mlp.c_fc.weight"] = [intermediate, d]; expected[l + "mlp.c_fc.bias"] = [intermediate]
            expected[l + "mlp.c_proj.weight"] = [d, intermediate]; expected[l + "mlp.c_proj.bias"] = [d]
        }
        for (key, shape) in expected { try weights.require(key, shape: shape) }
        guard Set(weights.names.filter { $0.hasPrefix(p) }) == Set(expected.keys) else {
            throw Inkling2Error.invalid("unexpected vision checkpoint tensors")
        }
    }
}
