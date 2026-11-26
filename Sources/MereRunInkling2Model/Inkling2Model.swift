import Foundation
import MLX

public final class Inkling2Model {
    public let config: Inkling2Configuration
    let weights: Inkling2Weights
    let prefix = "model.llm."
    let q8OutputHead: Inkling2Q8Projection?
    let loopCDStrength: Float
    package let usePackedExperts: Bool
    private var packedExperts: [Int: Inkling2PackedExperts] = [:]
    private var queryProjections: [Int: MLXArray] = [:]
    private var groupedAffineQueries: [Int: Inkling2ReferenceAffineProjection] = [:]
    private var groupedAffineQueryKeyValues: [Int: Inkling2ReferenceAffineProjection] = [:]
    package var usesSpeedPack: Bool { weights.speedDraftRoot != nil }
    var controllerProjections: [String: MLXArray] = [:]
    #if os(macOS)
    var loopControllerBatches: [String: Inkling2LoopController] = [:]
    #endif
    private lazy var clipThresholds = MLXArray([config.textConfig.qkL2ClipThreshold, config.textConfig.relL2ClipThreshold])
    public var prefillChunkSize: Int { usePackedExperts ? 32 : 8 }

    /// The longest prompt that prefills with 256-token chunks on M5 Max.
    package static let longPrefillMaxTokens = 131_072

    package func prefillChunkSize(for tokenCount: Int) -> Int {
        if let forced = ProcessInfo.processInfo.environment["MERERUN_INKLING2_PREFILL_CHUNK"].flatMap(Int.init),
           (8...512).contains(forced) {
            return forced
        }
        #if os(macOS)
        // Larger chunks reuse routed expert weights. Through 131,072 tokens, 256-token chunks prefill
        // 2.4-2.7x faster than 32-token chunks on M5 Max, and peak memory stays at MLX's cache limit.
        if usePackedExperts, Inkling2PackedExperts.reorderPrefill,
           tokenCount > Inkling2DecodeAttention.maxKeys, tokenCount <= Self.longPrefillMaxTokens {
            return 256
        }
        if usesSpeedPack, usePackedExperts, tokenCount > Inkling2DecodeAttention.maxKeys, tokenCount <= Inkling2MaskedAttention.maxKeys {
            return 64
        }
        #endif
        return prefillChunkSize
    }

    public convenience init(config: Inkling2Configuration, weights: Inkling2Weights) throws {
        try self.init(config: config, weights: weights, loopCDStrength: 0)
    }

    package init(config: Inkling2Configuration, weights: Inkling2Weights, loopCDStrength: Float) throws {
        try config.validate()
        guard loopCDStrength.isFinite, loopCDStrength >= 0, loopCDStrength == 0 || config.textConfig.loopConfig != nil else {
            throw Inkling2Error.invalid("LoopCD requires Inkling 2 Mini and a finite, non-negative strength")
        }
        self.config = config
        self.weights = weights
        self.loopCDStrength = loopCDStrength
        #if os(macOS)
        usePackedExperts = config.textConfig.loopConfig != nil && Device.defaultDevice().deviceType == .gpu
            && ProcessInfo.processInfo.environment["MERERUN_INKLING2_PACKED_EXPERTS"] != "0"
            && weights.storedBytes <= ProcessInfo.processInfo.physicalMemory / 4 * 3
            && (0..<config.textConfig.numHiddenLayers).allSatisfy { layer in
                let prefix = "model.llm.layers.\(layer).mlp.experts."
                return ["w13_weight", "w2_weight"].allSatisfy { name in
                    ["U8", "F8_E4M3", "F8_E4M3FN"].contains(weights.dtype(prefix + name) ?? "")
                }
            }
        #else
        usePackedExperts = false
        #endif
        #if os(macOS)
        weights.readsIntoOwnedBuffers = usePackedExperts
        #endif
        let headKey = Inkling2Q8Projection.outputKey
        let c = config.textConfig
        q8OutputHead = try weights.names.contains(headKey + ".q8_weight")
            ? Inkling2Q8Projection(weights: weights, key: headKey, rows: c.unpaddedVocabSize, columns: c.hiddenSize)
            : nil
        try validateWeights()
    }

    func linear(_ x: MLXArray, _ key: String, bias: Bool = false) throws -> MLXArray {
        let name = key + ".weight"
        let output: MLXArray
        if let affine = try weights.referenceProjection(name) {
            output = affine(x)
        } else {
            let weight = try weights.read(name)
            output = matmul(x.asType(weight.dtype), weight.T)
        }
        return bias ? try output + weights.read(key + ".bias") : output
    }

    func norm(_ x: MLXArray, _ key: String) throws -> MLXArray {
        let weight = try weights.read(key + ".weight").asType(.float32)
        let epsilon = config.textConfig.rmsNormEps
        if usePackedExperts {
            return MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: epsilon).asType(x.dtype)
        }
        return (Inkling2Math.rms(x, epsilon: epsilon) * weight).asType(x.dtype)
    }

    public func embeddings(_ tokens: MLXArray) throws -> MLXArray {
        let key = prefix + "embed.weight"
        let vectors = try weights.referenceProjection(key)?.embeddings(tokens) ?? weights.read(key)[tokens]
        return try norm(vectors, prefix + "embed_norm")
    }

    public func callAsFunction(_ tokens: MLXArray, cache: Inkling2Cache, lastPositionOnly: Bool = true) throws -> MLXArray {
        try forward(embeddings: embeddings(tokens), cache: cache, lastPositionOnly: lastPositionOnly)
    }

    public func makeCache() -> Inkling2Cache {
        Inkling2Cache(layerCount: config.textConfig.cacheLayerCount)
    }

    public func forward(embeddings input: MLXArray, cache: Inkling2Cache, lastPositionOnly: Bool = true) throws -> MLXArray {
        try forward(embeddings: input, cache: cache, lastPositionOnly: lastPositionOnly, capture: nil)
    }

    package func forward(embeddings input: MLXArray, cache: Inkling2Cache, lastPositionOnly: Bool = true,
                         outputDelta: ((MLXArray) -> MLXArray)?) throws -> MLXArray {
        try forward(embeddings: input, cache: cache, lastPositionOnly: lastPositionOnly, capture: nil, outputDelta: outputDelta)
    }

    func forward(embeddings input: MLXArray, cache: Inkling2Cache, lastPositionOnly: Bool,
                 capture: Inkling2DSparkFeatures?, outputDelta: ((MLXArray) -> MLXArray)? = nil) throws -> MLXArray {
        var hidden = try hiddenStates(embeddings: input, cache: cache, capture: capture)
        if lastPositionOnly { hidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...] }
        let output = try makeOutputHead()(hidden, outputDelta)
        if usePackedExperts && cache.verificationPosition == nil {
            // Submit one graph per forward. Cache states may not be outputs of
            // logits, so evaluate those roots too before advancing the cache.
            asyncEval([output] + cache.layers.flatMap(\.arrays))
        }
        return output
    }

    func hiddenStates(embeddings input: MLXArray, cache: Inkling2Cache, capture: Inkling2DSparkFeatures? = nil) throws -> MLXArray {
        let c = config.textConfig
        guard cache.layers.count == c.cacheLayerCount else { throw Inkling2Error.invalid("cache layer count mismatch") }
        cache.beginForward()
        var residual = Inkling2ResidualState(partial: input)
        if let loop = c.loopConfig {
            try runLoop(loop, residual: &residual, cache: cache, capture: capture)
        } else {
            try runLayers(0..<c.numHiddenLayers, residual: &residual, cache: cache)
        }
        let mixed = Inkling2Math.residual(residual.blocks, partial: residual.partial,
                                         query: try weights.read(prefix + "attn_res.output_query"),
                                         fusedDecode: usePackedExperts && (input.dim(1) == 1 || cache.isVerifying))[0]
        let hidden = try norm(mixed, prefix + "norm").asType(input.dtype)
        capture?.record(hidden, at: c.cacheLayerCount)
        return hidden
    }

    func runLayers(
        _ layers: Range<Int>, residual: inout Inkling2ResidualState, cache: Inkling2Cache,
        cacheOffset: Int = 0, previousPass: Inkling2LoopMessages? = nil,
        visitOffset: Int = 0, capture: Inkling2DSparkFeatures? = nil
    ) throws {
        let c = config.textConfig
        let dtype = residual.partial.dtype
        let fusedDecode = usePackedExperts && (residual.partial.dim(1) == 1 || cache.isVerifying)
        for i in layers {
            try Task.checkCancellation()
            let p = prefix + "layers.\(i)."
            let state = cache.layers[i + cacheOffset]
            let incoming = residual.partial
            capture?.record(incoming, at: i + visitOffset)
            let local = c.isLocal(i)
            var attnInputs = [incoming, incoming, incoming]
            if i > 0 {
                let query = try weights.read(prefix + "attn_res.attn_queries.\(i - 1)")
                let mixed = Inkling2Math.residual(residual.blocks, partial: incoming, query: query, fusedDecode: fusedDecode)
                attnInputs = local ? [mixed[0], mixed[0], mixed[0]] : mixed
            }
            let messages = previousPass?.read(layer: i)
            let normalized: [MLXArray]
            if let messages {
                let names = local ? ["attn_norm", "attn_norm", "attn_norm"] : ["q_input_norm", "k_input_norm", "v_input_norm"]
                normalized = try normalizeLoopInputs(attnInputs, messages: Array(messages.prefix(3)), prefix: p,
                                                     controllers: ["attn", "k", "v"], norms: names, dtype: dtype)
            } else if local {
                let x = try norm(attnInputs[0], p + "attn_norm").asType(dtype)
                normalized = [x, x, x]
            } else {
                let normNames = ["q_input_norm", "k_input_norm", "v_input_norm"]
                normalized = try zip(attnInputs, normNames).map { try norm($0.0, p + $0.1).asType(dtype) }
            }
            let attended = try attention(normalized, layer: i, cache: state, sharedCache: cache,
                                         sharesInput: local && messages == nil)
            let boundary = i.isMultiple(of: c.nLayersPerAttnResBlock)
            let partial = try convolutionTail(attended, residual: incoming, prefix: p + "attn_sconv", cache: state,
                                              index: 2, addResidual: !boundary, dtype: .float32)
            if boundary { residual.blocks.append(incoming) }
            let mixed = Inkling2Math.residual(residual.blocks, partial: partial,
                                            query: try weights.read(prefix + "attn_res.mlp_queries.\(i)"),
                                            fusedDecode: fusedDecode)[0]
            let mlpInput: MLXArray
            if let messages {
                mlpInput = try normalizeLoopInputs([mixed], messages: [messages[3]], prefix: p,
                                                  controllers: ["mlp"], norms: ["mlp_norm"], dtype: dtype)[0]
            } else {
                mlpInput = try norm(mixed, p + "mlp_norm").asType(dtype)
            }
            let expertOutput = try moe(mlpInput, layer: i, verifying: cache.isVerifying)
            residual.partial = try convolutionTail(expertOutput, residual: partial, prefix: p + "mlp_sconv",
                                                   cache: state, index: 3, addResidual: true, dtype: dtype)
            if !usePackedExperts { eval([residual.partial] + residual.blocks + state.arrays) }
        }
    }

    private func convolutionTail(_ x: MLXArray, residual: MLXArray, prefix: String, cache: Inkling2LayerCache,
                                 index: Int, addResidual: Bool, dtype: DType) throws -> MLXArray {
        #if os(macOS)
        let taps = config.textConfig.sconvKernelSize
        if usePackedExperts, cache.verification != nil, taps == 4, x.dim(0) == 1,
           x.dtype == .bfloat16, (4...8).contains(x.dim(1)), let prior = cache.convolutions[index] {
            let weights = try self.weights.read(prefix + ".weight")
            let (history, output) = Inkling2StaticResidualConvolution.apply(
                x, prior: prior, weights: weights, residual: residual, taps: taps, addResidual: addResidual, outputDType: dtype
            )
            cache.storeConvolutionContext(history, index: index, history: taps - 1)
            return output
        }
        #endif
        let output = try convolution(x, key: prefix, cache: cache, index: index).asType(.float32)
        return (addResidual ? residual.asType(.float32) + output : output).asType(dtype)
    }

    func convolution(_ x: MLXArray, key: String, cache: Inkling2LayerCache, index: Int,
                     dynamicInput: MLXArray? = nil, heads: Int = 1) throws -> MLXArray {
        let kernel = config.textConfig.sconvKernelSize
        let context = cache.context(x, index: index, kernel: kernel)
        #if os(macOS)
        if usePackedExperts {
            if let dynamicInput {
                let staticWeight = try weights.read(key + ".static_w")
                let projection = try weights.read(key + ".weight_proj")
                let dynamic = matmul(dynamicInput.asType(projection.dtype), projection.T)
                return Inkling2ShortConvolution.apply(x, context: context, taps: staticWeight,
                                                     kernelSize: kernel, dynamic: dynamic, heads: heads)
            }
            return try Inkling2ShortConvolution.apply(x, context: context, taps: weights.read(key + ".weight"),
                                                     kernelSize: kernel)
        }
        #endif
        var output = x.asType(.float32)
        if let dynamicInput {
            let staticWeight = try weights.read(key + ".static_w").asType(.float32)
            let projection = try weights.read(key + ".weight_proj")
            let dynamic = matmul(dynamicInput.asType(projection.dtype), projection.T).asType(.float32)
                .reshaped(x.dim(0), x.dim(1), heads, 1, kernel)
            for lag in 0..<kernel {
                let tap = staticWeight[lag].reshaped(1, 1, heads, x.dim(2) / heads)
                    + dynamic[0..., 0..., 0..., 0..., lag]
                let start = kernel - 1 - lag
                output = output + (context[0..., start..<(start + x.dim(1)), 0...]
                    .reshaped(x.dim(0), x.dim(1), heads, x.dim(2) / heads) * tap).reshaped(x.shape)
            }
        } else {
            let taps = try weights.read(key + ".weight").asType(.float32).reshaped(x.dim(2), kernel)
            for tap in 0..<kernel {
                output = output + context[0..., tap..<(tap + x.dim(1)), 0...] * taps[0..., tap]
            }
        }
        return (output / sqrt(Float((dynamicInput == nil ? 1 : 2) * kernel + 1))).asType(x.dtype)
    }

    func attention(
        _ inputs: [MLXArray], layer: Int, cache: Inkling2LayerCache, sharedCache: Inkling2Cache, sharesInput: Bool
    ) throws -> MLXArray {
        let c = config.textConfig, local = c.isLocal(layer)
        let p = prefix + "layers.\(layer).attn."
        let x = inputs[0], b = x.dim(0), n = x.dim(1)
        let h = local ? c.swaNumAttentionHeads : c.numAttentionHeads
        let kv = local ? c.swaNumKeyValueHeads : c.numKeyValueHeads
        let d = local ? c.swaHeadDim : c.headDim
        let extent = local ? c.slidingWindowSize : c.relExtent
        let offset = cache.offset
        let fusedDecode = usePackedExperts && (n == 1 || sharedCache.isVerifying)
        let projected = try projectQuery(x, layer: layer, prefix: p, heads: h, width: d,
                                         verifying: sharedCache.isVerifying, sharedKeyValueWidth: sharesInput ? kv * d : nil)
        var q = projected.query.reshaped(b, n, h, d)
        let key = try projected.key ?? linear(inputs[1], p + "wk_dv")
        let value = try projected.value ?? linear(inputs[2], p + "wv_dv")
        var k = try convolution(key, key: p + "k_sconv", cache: cache, index: 0,
                                dynamicInput: local ? nil : inputs[1], heads: kv).reshaped(b, n, kv, d)
        var v = try convolution(value, key: p + "v_sconv", cache: cache, index: 1,
                                dynamicInput: local ? nil : inputs[2], heads: kv).reshaped(b, n, kv, d)
        var r = projected.relative.reshaped(b, n, h, c.dRel)
        (q, k, v, r) = try normalizeAttentionHeads(q, key: k, value: v, relative: r, prefix: p,
                                                   dtype: x.dtype, offset: offset, local: local, fusedDecode: fusedDecode,
                                                   queryScale: sharedCache.verificationPosition?.queryScale)
        #if os(macOS)
        if !local, let position = sharedCache.verificationPosition, let keys = cache.keys, let values = cache.values {
            let pending = (k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3))
            cache.verification?.pendingKeysAndValues = pending
            let profiles = try weights.read(p + "rel_logits_proj.proj")
            let attended = Inkling2SplitAttention.apply(
                query: q, keys: keys, values: values, states: r, profiles: profiles,
                sinks: projected.sinks, gate: projected.gate, window: 0, packQueryRows: usesSpeedPack,
                bank: .init(position: position.committed, keys: pending.0, values: pending.1)
            )
            return try linear(attended, p + "wo_ud")
        }
        #endif
        let stored = cache.update(keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
                                  window: local ? c.slidingWindowSize : nil)
        let keyCount = stored.0.dim(2)
        let profiles = try weights.read(p + "rel_logits_proj.proj")
        #if os(macOS)
        if usesSpeedPack, usePackedExperts, sharedCache.isVerifying, b == 1, (4...8).contains(n),
           d == 128, h == 24, kv == 8, q.dtype == .bfloat16,
           stored.0.dtype == q.dtype, stored.1.dtype == q.dtype,
           keyCount > Inkling2DecodeAttention.maxKeys, keyCount <= Inkling2MaskedAttention.maxKeys {
            let attended = Inkling2SplitAttention.apply(
                query: q, keys: stored.0, values: stored.1, states: r, profiles: profiles,
                sinks: projected.sinks, gate: projected.gate, window: local ? extent : 0, packQueryRows: true
            )
            return try linear(attended, p + "wo_ud")
        }
        if fusedDecode, q.dtype == x.dtype, keyCount <= Inkling2DecodeAttention.maxKeys {
            let attended = Inkling2DecodeAttention.apply(query: q, keys: stored.0, values: stored.1, relative: r,
                                                         profiles: profiles, sinks: projected.sinks, gate: projected.gate, local: local)
            return try linear(attended, p + "wo_ud")
        }
        #endif
        let relative = matmul(r.asType(.float32), profiles.asType(.float32))
        #if os(macOS)
        if usesSpeedPack, usePackedExperts, !local, !sharedCache.isVerifying, b == 1, n > 8,
           d == 128, q.dtype == .bfloat16, stored.0.dtype == q.dtype, stored.1.dtype == q.dtype,
           Inkling2BandedPrefill.enabled {
            let attended = Inkling2BandedPrefill.apply(query: q, keys: stored.0, values: stored.1,
                                                      relative: relative, sinks: projected.sinks)
            let gate = sigmoid(projected.gate.asType(.float32))
            return try linear((attended.asType(.float32) * gate).asType(x.dtype), p + "wo_ud")
        }
        if usePackedExperts, keyCount > Inkling2DecodeAttention.maxKeys, keyCount <= Inkling2MaskedAttention.maxKeys {
            let scores = Inkling2Math.groupedAttentionDots(query: q, keys: stored.0)
            let probabilities = Inkling2MaskedAttention.probabilities(
                scores: scores, relative: relative, sinks: projected.sinks, divisor: sqrt(Float(d)), dtype: x.dtype, local: local
            )
            let attended = Inkling2Math.groupedAttentionValues(probabilities, values: stored.1)
            let gate = sigmoid(projected.gate.asType(.float32))
            return try linear((attended.asType(.float32) * gate).asType(x.dtype), p + "wo_ud")
        }
        #endif
        let mask = sharedCache.attentionMask(for: .init(offset: offset, queryCount: n, keyStart: stored.2,
                                                        keyCount: keyCount, extent: extent, local: local))
        let rel = relative.transposed(0, 2, 1, 3)
        let indices = broadcast(mask.relativeIndices, to: [b, h, n, keyCount])
        let bias = MLX.where(mask.withinExtent, takeAlong(rel, indices, axis: -1), MLXArray(Float(0)))
        let attended = Inkling2Math.groupedAttention(
            query: q, keys: stored.0, values: stored.1, bias: bias, allowed: mask.allowed, sinks: projected.sinks
        )
        let gate = sigmoid(projected.gate.asType(.float32))
        return try linear((attended.asType(.float32) * gate).asType(x.dtype), p + "wo_ud")
    }

    private func normalizeAttentionHeads(
        _ query: MLXArray, key: MLXArray, value: MLXArray, relative: MLXArray,
        prefix: String, dtype: DType, offset: Int, local: Bool, fusedDecode: Bool, queryScale: MLXArray?
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let c = config.textConfig, n = query.dim(1)
        let qScale = try weights.read(prefix + "q_l2_clip.scale")
        let rScale = try weights.read(prefix + "rel_l2_clip.scale")
        #if os(macOS)
        if fusedDecode, query.dtype == dtype, key.dtype == dtype,
           value.dtype == dtype, relative.dtype == dtype {
            let tau = local ? MLXArray(Array(repeating: Float(1), count: n))
                : (queryScale ?? attentionQueryScale(offset: offset, count: n))
            return Inkling2HeadNormalization.apply(query: query, key: key, value: value, relative: relative,
                                                   queryScale: qScale, relativeScale: rScale, thresholds: clipThresholds,
                                                   epsilon: c.rmsNormEps, tau: tau)
        }
        #endif
        var q = Inkling2Math.clip(query, scale: qScale, threshold: c.qkL2ClipThreshold)
        let k = Inkling2Math.clip(key, threshold: c.qkL2ClipThreshold)
        var r = Inkling2Math.clip(relative, scale: rScale, threshold: c.relL2ClipThreshold)
        let v = Inkling2Math.rms(value, epsilon: c.rmsNormEps).asType(dtype)
        if !local {
            let positions = MLXArray(Int32(offset + 1)..<Int32(offset + n + 1)).asType(.float32)
            let tau = (1 + c.logScalingAlpha * log(maximum(positions / Float(c.logScalingNFloor), MLXArray(Float(1)))))
                .reshaped(1, n, 1, 1)
            q = (q.asType(.float32) * tau).asType(dtype)
            r = (r.asType(.float32) * tau).asType(dtype)
        }
        return (q, k, v, r)
    }

    func attentionQueryScale(offset: Int, count: Int) -> MLXArray {
        let c = config.textConfig
        let floor = Float(c.logScalingNFloor)
        return MLXArray((1...count).map { index in
            1 + c.logScalingAlpha * log(max(Float(offset + index) / floor, 1))
        })
    }

    private func projectQuery(
        _ input: MLXArray, layer: Int, prefix: String, heads: Int, width: Int,
        verifying: Bool, sharedKeyValueWidth: Int?
    ) throws -> (query: MLXArray, relative: MLXArray, sinks: MLXArray, gate: MLXArray, key: MLXArray?, value: MLXArray?) {
        let keys = ["wq_du", "wr_du", "attn_sink_proj", "gate_attn"]
        let prepared = usesSpeedPack && usePackedExperts ? try combinedAffineQuery(layer: layer, prefix: prefix) : nil
        if verifying, let fused = prepared {
            let combined = sharedKeyValueWidth == nil ? nil : groupedAffineQueryKeyValues[layer]
            let result = (combined ?? fused)(input)
            let qEnd = heads * width
            let rEnd = qEnd + heads * config.textConfig.dRel
            let sinks = try linear(input, prefix + "attn_sink_proj")
            if let keyValueWidth = sharedKeyValueWidth, combined != nil {
                let gEnd = fused.weight.dim(0)
                let kEnd = gEnd + keyValueWidth
                return (result[.ellipsis, ..<qEnd], result[.ellipsis, qEnd..<rEnd], sinks,
                        result[.ellipsis, rEnd..<gEnd], result[.ellipsis, gEnd..<kEnd], result[.ellipsis, kEnd...])
            }
            return (result[.ellipsis, ..<qEnd], result[.ellipsis, qEnd..<rEnd],
                    sinks, result[.ellipsis, rEnd...], nil, nil)
        }
        var projection = queryProjections[layer]
        if usePackedExperts, projection == nil {
            let names = keys.map { prefix + $0 + ".weight" }
            if names.allSatisfy({ weights.referenceAffine[$0] == nil }), Set(names.map { weights.dtype($0) }).count == 1 {
                let fused = try concatenated(names.map { try weights.read($0, retain: false) }, axis: 0)
                eval(fused)
                queryProjections[layer] = fused
                projection = fused
            }
        }
        if let projection {
            let output = matmul(input.asType(projection.dtype), projection.T)
            let qEnd = heads * width, rEnd = qEnd + heads * config.textConfig.dRel
            let sEnd = rEnd + heads
            return (output[.ellipsis, ..<qEnd], output[.ellipsis, qEnd..<rEnd],
                    output[.ellipsis, rEnd..<sEnd], output[.ellipsis, sEnd...], nil, nil)
        }
        let outputs = try keys.map { try linear(input, prefix + $0) }
        return (outputs[0], outputs[1], outputs[2], outputs[3], nil, nil)
    }

    private func combinedAffineQuery(layer: Int, prefix: String) throws -> Inkling2ReferenceAffineProjection? {
        if let prepared = groupedAffineQueries[layer] { return prepared }
        let names = ["wq_du", "wr_du", "gate_attn"].map { prefix + $0 + ".weight" }
        guard let spec = weights.referenceAffine[names[0]],
              names.allSatisfy({
                  weights.referenceAffine[$0]?.bits == spec.bits &&
                  weights.referenceAffine[$0]?.groupSize == spec.groupSize
              }) else { return nil }
        let extra = ["wk_dv", "wv_dv"].map { prefix + $0 + ".weight" }
        let mergeSharedKV = usesSpeedPack && config.textConfig.isLocal(layer) && extra.allSatisfy {
            weights.referenceAffine[$0]?.bits == spec.bits && weights.referenceAffine[$0]?.groupSize == spec.groupSize
        }
        let group = try (mergeSharedKV ? names + extra : names).map { try weights.referenceProjection($0, retain: false)! }
        let bank = Inkling2ReferenceAffineProjection(
            weight: concatenated(group.map(\.weight), axis: 0),
            scales: concatenated(group.map(\.scales), axis: 0),
            biases: concatenated(group.map(\.biases), axis: 0),
            spec: spec
        )
        eval(bank.weight, bank.scales, bank.biases)
        let queries: Inkling2ReferenceAffineProjection
        if mergeSharedKV {
            let queryRows = group.prefix(names.count).reduce(0) { $0 + $1.weight.dim(0) }
            queries = bank.slice(0..<queryRows)
            groupedAffineQueryKeyValues[layer] = bank
            if layer == 0 {
            }
        } else {
            queries = bank
        }
        groupedAffineQueries[layer] = queries
        if layer == 0 {
        }
        return queries
    }

    func moe(_ x: MLXArray, layer: Int, verifying: Bool) throws -> MLXArray {
        let c = config.textConfig, p = prefix + "layers.\(layer).mlp."
        let flat = x.reshaped(-1, c.hiddenSize)
        let router = try weights.read(p + "gate.weight").asType(.float32)
        let logits = matmul(flat.asType(.float32), router.T)
        let (selected, routing) = try routeExperts(logits, bias: weights.read(p + "gate.bias"))
        let expertPrefix = p + "experts."
        if usePackedExperts {
            let experts: Inkling2PackedExperts
            if let resident = packedExperts[layer] {
                experts = resident
            } else {
                experts = try Inkling2PackedExperts(weights: weights, prefix: expertPrefix)
                packedExperts[layer] = experts
            }
            return experts(flat, indices: selected, routing: routing, gateCap: c.mlpGateSoftcap,
                           upCap: c.mlpUpSoftcap, verifying: verifying)
                .reshaped(x.shape)
        }
        let indices = selected.asArray(Int32.self)
        var outputs: [MLXArray] = []
        for token in 0..<flat.dim(0) {
            var result = MLXArray.zeros([1, c.hiddenSize], dtype: .float32)
            for choice in 0..<c.numExpertsPerTok {
                try Task.checkCancellation()
                let expert = Int(indices[token * c.numExpertsPerTok + choice])
                let w13 = try weights.expert(p + "experts.w13_weight", index: expert).asType(x.dtype)
                let projected = matmul(flat[token..<(token + 1)], w13.T).reshaped(1, c.intermediateSize, 2)
                let activated = Inkling2Math.activation(gate: projected[0..., 0..., 0], up: projected[0..., 0..., 1],
                                                       gateCap: c.mlpGateSoftcap, upCap: c.mlpUpSoftcap)
                let w2 = try weights.expert(p + "experts.w2_weight", index: expert).asType(x.dtype)
                result = result + matmul(activated, w2.T).asType(.float32) * routing[token, choice]
                // Bound the live graph to one expert; no full-model dequantization.
                eval(result)
            }
            outputs.append(result)
        }
        return concatenated(outputs, axis: 0).reshaped(x.shape).asType(x.dtype)
    }

    private func routeExperts(_ logits: MLXArray, bias: MLXArray) -> (MLXArray, MLXArray) {
        let c = config.textConfig
        #if os(macOS)
        if usePackedExperts, c.nRoutedExperts == 128, c.numExpertsPerTok == 8, bias.dtype == .float32 {
            return Inkling2Router.select(logits, bias: bias, scale: c.routeScale)
        }
        #endif
        let scores = sigmoid(logits)
        let selected = argSort(-(scores + bias), axis: -1)[0..., ..<c.numExpertsPerTok]
        let routing = takeAlong(scores, selected, axis: -1)
        return (selected, routing / (routing.sum(axis: -1, keepDims: true) + 1e-20) * c.routeScale)
    }
}
