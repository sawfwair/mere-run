import Foundation
import MLX
import MLXFast
import MLXNN

// Owns the transformer stack used by QwenTextEncoder.
// This file intentionally keeps low-level encoder blocks together and leaves
// public text/image orchestration in the main TextEncoder file.

public final class QwenAttention: Module {
  let hiddenSize: Int
  let numAttentionHeads: Int
  let numKeyValueHeads: Int
  let headDim: Int
  let numKeyValueGroups: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  let rotaryEmb: Qwen3VLRotaryEmbedding?
  let legacyRotaryEmb: Qwen2VLRotaryEmbedding?
  let mropeSection: [Int]?
  let rope: RoPE?

  init(configuration: QwenTextEncoderConfiguration) {
    self.hiddenSize = configuration.hiddenSize
    self.numAttentionHeads = configuration.numAttentionHeads
    self.numKeyValueHeads = configuration.numKeyValueHeads
    self.headDim = configuration.headDim
    self.numKeyValueGroups = configuration.numAttentionHeads / configuration.numKeyValueHeads
    self.scale = pow(Float(configuration.headDim), -0.5)

    self._qProj.wrappedValue = Linear(hiddenSize, numAttentionHeads * headDim, bias: false)
    self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
    self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
    self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hiddenSize, bias: false)

    self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
    self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)

    if let mropeSection = configuration.mropeSection, !mropeSection.isEmpty {
      self.mropeSection = mropeSection
      if configuration.mropeInterleaved {
        self.rotaryEmb = Qwen3VLRotaryEmbedding(dim: headDim, base: configuration.ropeTheta, mropeSection: mropeSection)
        self.legacyRotaryEmb = nil
      } else {
        self.rotaryEmb = nil
        self.legacyRotaryEmb = Qwen2VLRotaryEmbedding(dim: headDim, base: configuration.ropeTheta)
      }
      self.rope = nil
    } else {
      self.mropeSection = nil
      self.rotaryEmb = nil
      self.legacyRotaryEmb = nil
      self.rope = RoPE(
        dimensions: headDim,
        traditional: false,
        base: configuration.ropeTheta,
        scale: 1.0
      )
    }
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache? = nil,
    positionIds: MLXArray? = nil,
    debug: Bool = false
  ) -> MLXArray {
    let B = x.dim(0)
    let L = x.dim(1)

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    if debug {
        eval(queries, keys, values)
        print("[Attn] Q pre-norm: std=\(MLX.std(queries).item(Float.self))")
        print("[Attn] K pre-norm: std=\(MLX.std(keys).item(Float.self))")
    }

    queries = qNorm(queries.reshaped(B, L, numAttentionHeads, headDim)).transposed(0, 2, 1, 3)
    keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, headDim)).transposed(0, 2, 1, 3)
    values = values.reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

    if debug {
        eval(queries, keys)
        print("[Attn] Q post-norm: std=\(MLX.std(queries).item(Float.self))")
        print("[Attn] K post-norm: std=\(MLX.std(keys).item(Float.self))")
    }

    let offset = cache?.offset ?? 0
    if let rotaryEmb {
      let posIds: MLXArray
      if let provided = positionIds {
        posIds = provided
      } else {
        let positions = MLXArray(Array(offset..<(offset + L)).map { Int32($0) })
        posIds = broadcast(positions.reshaped(1, L), to: [B, L])
      }
      let (cos, sin) = rotaryEmb.callAsFunction(positionIds: posIds, dtype: queries.dtype)
      (queries, keys) = applyRotaryPosEmb(queries, keys, cos: cos, sin: sin)
    } else if let legacyRotaryEmb, let mropeSection {
      let posIds: MLXArray
      if let provided = positionIds {
        posIds = provided
      } else {
        let positions = MLXArray(Array(offset..<(offset + L)).map { Int32($0) })
        posIds = broadcast(positions.reshaped(1, L), to: [B, L])
      }
      let (cos, sin) = legacyRotaryEmb.callAsFunction(positionIds: posIds, dtype: queries.dtype)
      (queries, keys) = applyMultimodalRoPE(
        queries,
        keys,
        cos: cos,
        sin: sin,
        mropeSection: mropeSection
      )
    } else if let rope {
      queries = rope(queries.asType(.bfloat16), offset: offset)
      keys = rope(keys.asType(.bfloat16), offset: offset)
    }

    if debug {
        eval(queries, keys)
        print("[Attn] Q post-RoPE: std=\(MLX.std(queries.asType(.float32)).item(Float.self))")
        print("[Attn] K post-RoPE: std=\(MLX.std(keys.asType(.float32)).item(Float.self))")
    }

    if let cache = cache {
      let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
      keys = cachedKeys
      values = cachedValues
    }

    if numKeyValueHeads != numAttentionHeads {
      keys = expandKeyValue(keys, repeats: numKeyValueGroups)
      values = expandKeyValue(values, repeats: numKeyValueGroups)
    }

    let queriesF32 = queries.asType(.float32)
    let keysF32 = keys.asType(.float32)
    let valuesF32 = values.asType(.float32)

    var output = MLXFast.scaledDotProductAttention(
      queries: queriesF32,
      keys: keysF32,
      values: valuesF32,
      scale: scale,
      mask: mask
    )

    output = output.asType(queries.dtype)
    output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

    let result = oProj(output)
    if debug {
        eval(result)
        print("[Attn] Output: std=\(MLX.std(result.asType(.float32)).item(Float.self))")
    }
    return result
  }

  private func expandKeyValue(_ x: MLXArray, repeats: Int) -> MLXArray {
    guard repeats > 1 else { return x }
    var expanded = MLX.expandedDimensions(x, axis: 2)
    expanded = MLX.repeated(expanded, count: repeats, axis: 2)
    let shape = x.shape
    return expanded.reshaped(shape[0], shape[1] * repeats, shape[2], shape[3])
  }
}

public final class QwenMLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear

  init(dimensions: Int, hiddenDimensions: Int) {
    self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
  }

  public func callAsFunction(_ x: MLXArray, debug: Bool = false) -> MLXArray {
    if debug {
        eval(x)
        print("[MLP] Input x: std=\(MLX.std(x.asType(.float32)).item(Float.self)), mean=\(MLX.mean(x.asType(.float32)).item(Float.self)), max=\(MLX.max(x).item(Float.self)), min=\(MLX.min(x).item(Float.self))")
    }
    let gate = gateProj(x)
    let up = upProj(x)
    if debug {
        eval(gate, up)
        print("[MLP] gate dtype: \(gate.dtype), x dtype: \(x.dtype)")
        print("[MLP] After gate_proj: std=\(MLX.std(gate.asType(.float32)).item(Float.self)), mean=\(MLX.mean(gate.asType(.float32)).item(Float.self)), max=\(MLX.max(gate).item(Float.self)), min=\(MLX.min(gate).item(Float.self))")
        print("[MLP] After up_proj: std=\(MLX.std(up.asType(.float32)).item(Float.self))")
    }
    let gateF32 = gate.asType(.float32)
    let gateSilu = (gateF32 * sigmoid(gateF32)).asType(gate.dtype)
    if debug {
        eval(gateSilu)
        print("[MLP] After SiLU(gate): std=\(MLX.std(gateSilu.asType(.float32)).item(Float.self))")
    }
    let gated = gateSilu * up
    if debug {
        eval(gated)
        print("[MLP] After gate*up: std=\(MLX.std(gated.asType(.float32)).item(Float.self))")
    }
    let out = downProj(gated)
    if debug {
        eval(out)
        print("[MLP] After down_proj: std=\(MLX.std(out.asType(.float32)).item(Float.self))")
    }
    return out
  }
}

public final class QwenEncoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttention: QwenAttention
  @ModuleInfo(key: "mlp") var mlp: QwenMLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(configuration: QwenTextEncoderConfiguration) {
    self._selfAttention.wrappedValue = QwenAttention(configuration: configuration)
    self._mlp.wrappedValue = QwenMLP(dimensions: configuration.hiddenSize, hiddenDimensions: configuration.intermediateSize)
    self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache? = nil,
    positionIds: MLXArray? = nil,
    debug: Bool = false,
    debugMLP: Bool = false
  ) -> MLXArray {
    if debug {
        eval(x)
        print("[L] Input: std=\(String(format: "%.2f", MLX.std(x.asType(.float32)).item(Float.self))), max=\(String(format: "%.1f", MLX.max(x).item(Float.self))), min=\(String(format: "%.1f", MLX.min(x).item(Float.self)))")
    }
    let normed = inputLayerNorm(x)
    if debug {
        eval(normed)
        print("[L] Normed: std=\(String(format: "%.2f", MLX.std(normed.asType(.float32)).item(Float.self))), max=\(String(format: "%.1f", MLX.max(normed).item(Float.self)))")
    }
    let r = selfAttention(normed, mask: mask, cache: cache, positionIds: positionIds, debug: false)
    let h = x + r
    if debug {
        eval(h)
        print("[L] AfterAttn: std=\(String(format: "%.2f", MLX.std(h.asType(.float32)).item(Float.self))), max=\(String(format: "%.1f", MLX.max(h).item(Float.self))), min=\(String(format: "%.1f", MLX.min(h).item(Float.self)))")
    }

    let postNormed = postAttentionLayerNorm(h)
    if debugMLP {
        eval(postNormed, h)
        let minIdx = MLX.argMin(h).item(Int.self)
        let minIdxNormed = MLX.argMin(postNormed).item(Int.self)
        print("[MLP2] h minIdx=\(minIdx), h[minIdx]=\(h.flattened()[minIdx].item(Float.self))")
        print("[MLP2] postNorm minIdx=\(minIdxNormed), postNorm[minIdx]=\(postNormed.flattened()[minIdxNormed].item(Float.self))")
        let weightIdx = minIdxNormed % 1024
        print("[MLP2] postNormWeight[\(weightIdx)]=\(postAttentionLayerNorm.weight[weightIdx].item(Float.self))")
        print("[MLP2] postNorm: std=\(String(format: "%.2f", MLX.std(postNormed.asType(.float32)).item(Float.self))), max=\(String(format: "%.1f", MLX.max(postNormed).item(Float.self))), min=\(String(format: "%.1f", MLX.min(postNormed).item(Float.self)))")
    }
    let mlpOut = mlp(postNormed, debug: debugMLP)
    let out = h + mlpOut
    if debug {
        eval(out)
        print("[L] AfterMLP: std=\(String(format: "%.2f", MLX.std(out.asType(.float32)).item(Float.self))), max=\(String(format: "%.1f", MLX.max(out).item(Float.self))), min=\(String(format: "%.1f", MLX.min(out).item(Float.self)))")
    }

    return out
  }
}

public final class QwenEncoder: Module {

  public let configuration: QwenTextEncoderConfiguration
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [QwenEncoderLayer]
  @ModuleInfo(key: "norm") var norm: RMSNorm

  public init(configuration: QwenTextEncoderConfiguration) {
    self.configuration = configuration
    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: configuration.vocabSize, dimensions: configuration.hiddenSize)
    self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
      QwenEncoderLayer(configuration: configuration)
    }
    self._norm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXArray {
    forward(inputIds: inputIds, attentionMask: attentionMask).lastHiddenState
  }

  public func forward(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    outputHiddenStates: Bool = false
  ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var h = embedTokens(tokenIds).asType(.bfloat16)

    let mask = createAttentionMask(h: h, attentionMask: attentionMask)

    var allHiddenStates: [MLXArray]? = outputHiddenStates ? [h] : nil

    for layer in layers {
      h = layer(h, mask: mask)
      if outputHiddenStates {
        allHiddenStates?.append(h)
      }
    }

    h = norm(h)

    if outputHiddenStates, var states = allHiddenStates, !states.isEmpty {
      states[states.count - 1] = h
      allHiddenStates = states
    }

    return (h, allHiddenStates)
  }

  public func forward(
    embeddings: MLXArray,
    attentionMask: MLXArray?,
    outputHiddenStates: Bool = false
  ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
    var h = embeddings.dtype == .bfloat16 ? embeddings : embeddings.asType(.bfloat16)

    let mask = createAttentionMask(h: h, attentionMask: attentionMask)

    var allHiddenStates: [MLXArray]? = outputHiddenStates ? [h] : nil

    for layer in layers {
      h = layer(h, mask: mask)
      if outputHiddenStates {
        allHiddenStates?.append(h)
      }
    }

    h = norm(h)

    if outputHiddenStates, var states = allHiddenStates, !states.isEmpty {
      states[states.count - 1] = h
      allHiddenStates = states
    }

    return (h, allHiddenStates)
  }

  private func createAttentionMask(h: MLXArray, attentionMask: MLXArray?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let L = h.dim(1)

    let causalMask = MLXFast.ScaledDotProductAttentionMaskMode.causal

    if let attentionMask = attentionMask {
      let paddingMask = attentionMask.asType(h.dtype)
      let zeros = MLX.zeros(paddingMask.shape, dtype: h.dtype)
      let negInf = MLXArray(-Float.infinity).asType(h.dtype)
      let keepMask = paddingMask .== MLXArray(1).asType(h.dtype)
      var additivePaddingMask = MLX.where(keepMask, zeros, zeros + negInf)

      additivePaddingMask = additivePaddingMask.reshaped(additivePaddingMask.dim(0), 1, 1, L)

      let idx = MLXArray(0..<L)
      let rows = idx.reshaped(L, 1)
      let cols = idx.reshaped(1, L)
      let causalBool = cols .> rows
      var causalAdditive = MLX.zeros([L, L], dtype: h.dtype)
      causalAdditive = MLX.where(causalBool, causalAdditive + negInf, causalAdditive)
      causalAdditive = causalAdditive.reshaped(1, 1, L, L)

      let combinedMask = causalAdditive + additivePaddingMask

      return .array(combinedMask)
    }

    return causalMask
  }
}

extension QwenEncoder {
  public func embed(inputIds: MLXArray) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    return embedTokens(tokenIds).asType(.bfloat16)
  }

  public func forwardCausal(inputIds: MLXArray, cache: [KVCache]?, positionIds: MLXArray? = nil) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var h = embedTokens(tokenIds)
    let mask = createCausalMaskForGeneration(h: h, cache: cache?.first)

    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)
    }

    h = norm(h)
    return embedTokens.asLinear(h)
  }

  private func createCausalMaskForGeneration(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)
    if let cache = cache {
      return cache.makeMask(n: n)
    }
    if n == 1 {
      return .none
    }
    return .causal
  }
}
