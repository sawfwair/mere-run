import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

private func falconPerceptionRMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    let normalized = x32 * rsqrt(variance + MLXArray(eps))
    return (normalized * weight.asType(.float32)).asType(dtype)
}

private func falconPerceptionRepeatAlongHeads(_ x: MLXArray, heads: Int) -> MLXArray {
    MLX.repeated(x, count: heads, axis: 1)
}

private func falconPerceptionContiguous(_ x: MLXArray) -> MLXArray {
    x.reshaped(-1).reshaped(x.shape)
}

private func falconPerceptionShouldForceKernelDecodeAttention() -> Bool {
    ProcessInfo.processInfo.environment["MERERUN_FALCON_FORCE_KERNEL_DECODE_ATTENTION"] == "1"
}

func falconPerceptionAttentionBiasMask(_ mask: MLXArray, dtype: DType) -> MLXArray {
    let keep = mask.asType(.float32) .> MLXArray(0)
    let zeros = MLX.zeros(mask.shape, dtype: dtype)
    let negative = MLXArray(-1e9, dtype: dtype)
    return MLX.where(keep, zeros, zeros + negative)
}

private func falconPerceptionManualAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXArray?,
    sinks: MLXArray?
) -> MLXArray {
    let queriesF32 = queries.asType(.float32)
    let keysF32 = keys.asType(.float32)
    let valuesF32 = values.asType(.float32)
    var scores = MLX.matmul(queriesF32, keysF32.transposed(0, 1, 3, 2)) * MLXArray(scale)
    if let mask {
        scores = scores + mask.asType(.float32)
    }
    let weights: MLXArray
    if let sinks {
        let batchSize = scores.dim(0)
        let queryLength = scores.dim(2)
        var sinkScores = sinks.asType(.float32).reshaped(1, sinks.count, 1, 1)
        if batchSize > 1 {
            sinkScores = MLX.repeated(sinkScores, count: batchSize, axis: 0)
        }
        if queryLength > 1 {
            sinkScores = MLX.repeated(sinkScores, count: queryLength, axis: 2)
        }
        let extendedScores = concatenated([scores, sinkScores], axis: -1)
        weights = softmax(extendedScores, axis: -1)[0..., 0..., 0..., ..<scores.dim(3)]
    } else {
        weights = softmax(scores, axis: -1)
    }
    if weights.dim(1) == valuesF32.dim(1) {
        return MLX.matmul(weights, valuesF32).asType(queries.dtype)
    }

    let queryHeads = weights.dim(1)
    let valueHeads = valuesF32.dim(1)
    precondition(queryHeads.isMultiple(of: valueHeads), "Falcon query heads must group over value heads.")
    let groups = queryHeads / valueHeads
    let groupedWeights = weights.reshaped(
        weights.dim(0),
        valueHeads,
        groups,
        weights.dim(2),
        weights.dim(3)
    )
    let groupedValues = MLX.expandedDimensions(valuesF32, axis: 2)
    return MLX.matmul(groupedWeights, groupedValues)
        .reshaped(weights.dim(0), queryHeads, weights.dim(2), valuesF32.dim(3))
        .asType(queries.dtype)
}

final class FalconPerceptionKVCache: @unchecked Sendable {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset: Int = 0
    private let step = 256

    /// Capacity-padded append: buffers grow in 256-step chunks and new
    /// keys/values write into a slice, so a decode of T tokens copies O(T)
    /// data instead of the O(T²) of re-concatenating the whole cache per
    /// token. The previous implementation also forced an eval per layer per
    /// token; evaluation is now left to the decode loop's readback.
    func updateAndFetch(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let required = previous + newKeys.dim(2)

        let needsGrowth: Bool
        if let currentKeys = keys {
            needsGrowth = required > currentKeys.dim(2)
        } else {
            needsGrowth = true
        }

        if needsGrowth {
            let batch = newKeys.dim(0)
            let keyHeads = newKeys.dim(1)
            let valueHeads = newValues.dim(1)
            let keyDim = newKeys.dim(3)
            let valueDim = newValues.dim(3)
            let chunks = (required + step - 1) / step
            let grownKeys = MLXArray.zeros([batch, keyHeads, chunks * step, keyDim], dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros(
                [batch, valueHeads, chunks * step, valueDim],
                dtype: newValues.dtype
            )
            if let currentKeys = keys, let currentValues = values, previous > 0 {
                grownKeys[.ellipsis, ..<previous, 0...] = currentKeys[.ellipsis, ..<previous, 0...]
                grownValues[.ellipsis, ..<previous, 0...] = currentValues[.ellipsis, ..<previous, 0...]
            }
            keys = grownKeys
            values = grownValues
        }

        keys?[.ellipsis, previous..<required, 0...] = newKeys
        values?[.ellipsis, previous..<required, 0...] = newValues
        offset = required

        guard let cachedKeys = keys, let cachedValues = values else {
            preconditionFailure("FalconPerceptionKVCache should hold keys and values after update.")
        }
        return (
            cachedKeys[.ellipsis, ..<required, 0...],
            cachedValues[.ellipsis, ..<required, 0...]
        )
    }
}

final class FalconPerceptionFourierEncoder: Module {
    @ModuleInfo(key: "embed") var embed: Linear
    @ModuleInfo(key: "transform") var transform: Linear

    init(inDim: Int, featDim: Int, outDim: Int) {
        self._embed.wrappedValue = Linear(inDim, featDim / 2, bias: false)
        self._transform.wrappedValue = Linear(featDim, outDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = (2.0 * Float.pi) * embed(x)
        let features = concatenated([MLX.cos(projected), MLX.sin(projected)], axis: -1)
        return transform(features)
    }
}

final class FalconPerceptionBboxDecoder: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(inDim: Int, hiddenDim: Int, outDim: Int) {
        self._w1.wrappedValue = Linear(inDim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, outDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = relu(w1(x))
        return w2(hidden * hidden)
    }
}

final class FalconPerceptionSegmDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]
    @ModuleInfo(key: "pixel_layer") var pixelLayer: Linear

    init(inDim: Int, outDim: Int, numLayers: Int) {
        self._layers.wrappedValue = (0..<max(0, numLayers - 1)).map { _ in
            Linear(inDim, inDim)
        }
        self._pixelLayer.wrappedValue = Linear(inDim, outDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers {
            let activated = relu(layer(hidden))
            hidden = activated * activated
        }
        return pixelLayer(hidden)
    }
}

struct FalconPerceptionAttentionDebugCapture {
    var input: MLXArray?
    var normalizedInput: MLXArray?
    var qkvProjected: MLXArray?
    var queriesBeforeRoPE: MLXArray?
    var keysBeforeRoPE: MLXArray?
    var valuesBeforeCache: MLXArray?
    var queriesAfterRoPE: MLXArray?
    var keysAfterRoPE: MLXArray?
    var keysAfterCache: MLXArray?
    var valuesAfterCache: MLXArray?
    var attentionMask: MLXArray?
    var attentionOutput: MLXArray?
    var projectedOutput: MLXArray?
}

final class FalconPerceptionAttention: Module {
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ParameterInfo(key: "_norm_w_in") var normWIn: MLXArray
    @ParameterInfo(key: "_norm_w_qk") var normWQK: MLXArray

    private let numHeads: Int
    private let numKVHeads: Int
    private let numRepeats: Int
    private let headDim: Int
    private let qSize: Int
    private let kvSize: Int
    private let scale: Float
    private let eps: Float
    var captureDebugStages = false
    var lastDebugCapture: FalconPerceptionAttentionDebugCapture?

    init(config: FalconPerceptionTextConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.numRepeats = max(1, config.numAttentionHeads / max(1, config.numKeyValueHeads))
        self.headDim = config.headDim
        self.qSize = config.numAttentionHeads * config.headDim
        self.kvSize = config.numKeyValueHeads * config.headDim
        self.scale = 1.0 / sqrt(Float(max(1, config.headDim)))
        self.eps = config.rmsNormEps
        self._wqkv.wrappedValue = Linear(config.hiddenSize, qSize + (2 * kvSize), bias: false)
        self._wo.wrappedValue = Linear(qSize, config.hiddenSize, bias: false)
        self._sinks.wrappedValue = MLX.zeros([config.numAttentionHeads], dtype: .float32)
        self._normWIn.wrappedValue = MLX.ones([config.hiddenSize], dtype: .float32)
        self._normWQK.wrappedValue = MLX.ones([config.headDim], dtype: .float32)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: FalconPerceptionKVCache?,
        cos1D: MLXArray?,
        sin1D: MLXArray?,
        cos2D: MLXArray?,
        sin2D: MLXArray?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let sequenceLength = x.dim(1)
        lastDebugCapture = captureDebugStages ? FalconPerceptionAttentionDebugCapture() : nil
        if captureDebugStages {
            let snapshot = falconPerceptionContiguous(x)
            MLX.eval(snapshot)
            lastDebugCapture?.input = snapshot
        }
        let normalized = falconPerceptionRMSNorm(x, weight: normWIn, eps: eps)
        if captureDebugStages {
            let snapshot = falconPerceptionContiguous(normalized)
            MLX.eval(snapshot)
            lastDebugCapture?.normalizedInput = snapshot
        }
        let qkv = wqkv(normalized)
        if captureDebugStages {
            let snapshot = falconPerceptionContiguous(qkv)
            MLX.eval(snapshot)
            lastDebugCapture?.qkvProjected = snapshot
        }

        var queries = qkv[.ellipsis, 0..<qSize]
        var keys = qkv[.ellipsis, qSize..<(qSize + kvSize)]
        var values = qkv[.ellipsis, (qSize + kvSize)...]

        queries = queries.reshaped(batchSize, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, sequenceLength, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, sequenceLength, numKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = falconPerceptionRMSNorm(queries, weight: normWQK, eps: eps)
        keys = falconPerceptionRMSNorm(keys, weight: normWQK, eps: eps)

        // Falcon's learned 2D RoPE has distinct frequencies for every query
        // head, so keys must be expanded before that rotation. Values have no
        // head-specific transform and remain compact in the persistent cache.
        if numRepeats > 1 {
            keys = falconPerceptionRepeatAlongHeads(keys, heads: numRepeats)
        }

        if captureDebugStages {
            let querySnapshot = falconPerceptionContiguous(queries)
            let keySnapshot = falconPerceptionContiguous(keys)
            let valueSnapshot = falconPerceptionContiguous(values)
            MLX.eval(querySnapshot, keySnapshot, valueSnapshot)
            lastDebugCapture?.queriesBeforeRoPE = querySnapshot
            lastDebugCapture?.keysBeforeRoPE = keySnapshot
            lastDebugCapture?.valuesBeforeCache = valueSnapshot
        }

        if let cos1D, let sin1D {
            let rotated = FalconPerceptionModel.applyRotaryEmbeddings(
                queries: queries,
                keys: keys,
                cos1D: cos1D,
                sin1D: sin1D,
                cos2D: cos2D,
                sin2D: sin2D
            )
            queries = rotated.0
            keys = rotated.1
        }

        if captureDebugStages {
            let querySnapshot = falconPerceptionContiguous(queries)
            let keySnapshot = falconPerceptionContiguous(keys)
            MLX.eval(querySnapshot, keySnapshot)
            lastDebugCapture?.queriesAfterRoPE = querySnapshot
            lastDebugCapture?.keysAfterRoPE = keySnapshot
        }

        if let cache {
            let updated = cache.updateAndFetch(keys: keys, values: values)
            keys = updated.0
            values = updated.1
        }

        if captureDebugStages {
            let keySnapshot = falconPerceptionContiguous(keys)
            let valueSnapshot = falconPerceptionContiguous(values)
            MLX.eval(keySnapshot, valueSnapshot)
            lastDebugCapture?.keysAfterCache = keySnapshot
            lastDebugCapture?.valuesAfterCache = valueSnapshot
        }

        queries = falconPerceptionContiguous(queries)
        keys = falconPerceptionContiguous(keys)
        values = falconPerceptionContiguous(values)

        let attentionMask = mask.map { falconPerceptionAttentionBiasMask($0, dtype: queries.dtype) }
        if captureDebugStages, let attentionMask {
            let snapshot = falconPerceptionContiguous(attentionMask)
            MLX.eval(snapshot)
            lastDebugCapture?.attentionMask = snapshot
        }

        let useManualDecodeAttention = cache != nil
            && sequenceLength == 1
            && !falconPerceptionShouldForceKernelDecodeAttention()
        let attended = if useManualDecodeAttention {
            falconPerceptionManualAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: attentionMask,
                sinks: sinks
            )
        } else {
            let attentionValues = numRepeats > 1
                ? falconPerceptionRepeatAlongHeads(values, heads: numRepeats)
                : values
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: attentionValues,
                scale: scale,
                mask: attentionMask,
                sinks: sinks
            )
        }
        let output = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, qSize)
        if captureDebugStages {
            let snapshot = falconPerceptionContiguous(output)
            MLX.eval(snapshot)
            lastDebugCapture?.attentionOutput = snapshot
        }
        let projected = wo(output)
        if captureDebugStages {
            let snapshot = falconPerceptionContiguous(projected)
            MLX.eval(snapshot)
            lastDebugCapture?.projectedOutput = snapshot
        }
        return projected
    }
}

final class FalconPerceptionMLP: Module {
    @ModuleInfo(key: "w13") var w13: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ParameterInfo(key: "_norm_w") var normW: MLXArray

    private let hiddenDim: Int
    private let eps: Float

    init(config: FalconPerceptionTextConfig) {
        self.hiddenDim = config.intermediateSize
        self.eps = config.rmsNormEps
        self._w13.wrappedValue = Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._normW.wrappedValue = MLX.ones([config.hiddenSize], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = falconPerceptionRMSNorm(x, weight: normW, eps: eps)
        let projected = w13(normalized)
        let gate = projected[.ellipsis, 0..<hiddenDim]
        let up = projected[.ellipsis, hiddenDim...]
        let activated = relu(gate)
        return w2((activated * activated) * up)
    }
}

final class FalconPerceptionDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: FalconPerceptionAttention
    @ModuleInfo(key: "mlp") var mlp: FalconPerceptionMLP

    init(config: FalconPerceptionTextConfig) {
        self._selfAttn.wrappedValue = FalconPerceptionAttention(config: config)
        self._mlp.wrappedValue = FalconPerceptionMLP(config: config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: FalconPerceptionKVCache?,
        cos1D: MLXArray?,
        sin1D: MLXArray?,
        cos2D: MLXArray?,
        sin2D: MLXArray?
    ) -> MLXArray {
        let attended = selfAttn(x, mask: mask, cache: cache, cos1D: cos1D, sin1D: sin1D, cos2D: cos2D, sin2D: sin2D)
        let withAttention = x + attended
        return withAttention + mlp(withAttention)
    }
}

final class FalconPerceptionTransformerModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "img_projector") var imgProjector: Linear
    @ModuleInfo(key: "layers") var layers: [FalconPerceptionDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "freqs_cis_golden") var freqsGolden: MLXArray

    let config: FalconPerceptionModelConfig
    let cos1DTable: MLXArray
    let sin1DTable: MLXArray
    var lastHiddenState: MLXArray?
    var captureLayerOutputs = false
    var capturedLayerOutputs: [MLXArray] = []

    init(config: FalconPerceptionModelConfig) {
        self.config = config
        let text = config.textConfig
        let patchDim = config.visionConfig.temporalPatchSize
            * (config.visionConfig.spatialPatchSize * config.visionConfig.spatialPatchSize)
            * config.visionConfig.channelSize

        self._embedTokens.wrappedValue = Embedding(embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        self._imgProjector.wrappedValue = Linear(patchDim, text.hiddenSize, bias: false)
        self._layers.wrappedValue = (0..<text.numHiddenLayers).map { _ in
            FalconPerceptionDecoderLayer(config: text)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)

        let rotaryDim = max(1, text.headDim / 2)
        let heads = text.numAttentionHeads
        let goldenShape = [heads, max(1, rotaryDim / 2), 2]
        self._freqsGolden.wrappedValue = MLX.zeros(goldenShape, dtype: .float32)

        let halfRotary = max(1, rotaryDim / 2)
        let positions = MLXArray(0..<text.maxPositionEmbeddings)
            .asType(.float32)
            .reshaped(text.maxPositionEmbeddings, 1)
        let dimensions = MLXArray(0..<halfRotary).asType(.float32).reshaped(1, halfRotary)
        let exponents = (dimensions * 2) / Float(max(1, rotaryDim))
        let inverseFrequencies = 1 / MLX.pow(MLXArray(text.ropeTheta), exponents)
        let angles = positions * inverseFrequencies
        self.cos1DTable = MLX.cos(angles)
        self.sin1DTable = MLX.sin(angles)

        super.init()
    }

    func callAsFunction(
        inputIDs: MLXArray,
        inputsEmbeds: MLXArray?,
        mask: MLXArray?,
        caches: [FalconPerceptionKVCache?]?,
        positionIDs: MLXArray?,
        posHW: MLXArray?
    ) -> MLXArray {
        let hidden0 = inputsEmbeds ?? embedTokens(inputIDs)
        let cacheList = caches ?? Array(repeating: nil, count: layers.count)
        capturedLayerOutputs.removeAll(keepingCapacity: captureLayerOutputs)

        let selectedCosSin = FalconPerceptionModel.selectCosSinTables(
            positionIDs: positionIDs,
            cosTable: cos1DTable,
            sinTable: sin1DTable
        )
        let goldenFrequencies = posHW.map {
            FalconPerceptionModel.computeGoldenFrequencies(freqsGolden: freqsGolden, posHW: $0)
        }

        var hidden = hidden0
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                mask: mask,
                cache: cacheList[index],
                cos1D: selectedCosSin.0,
                sin1D: selectedCosSin.1,
                cos2D: goldenFrequencies?.0,
                sin2D: goldenFrequencies?.1
            )
            if captureLayerOutputs {
                let snapshot = falconPerceptionContiguous(hidden)
                MLX.eval(snapshot)
                capturedLayerOutputs.append(snapshot)
            }
        }

        let normalized = norm(hidden)
        lastHiddenState = normalized
        return normalized
    }
}

final class FalconPerceptionLanguageModel: Module {
    @ModuleInfo(key: "model") var model: FalconPerceptionTransformerModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: FalconPerceptionModelConfig
    private(set) var ropeDelta: Int?
    private(set) var prefillPositionIDs: MLXArray?
    private(set) var prefillPosHW: MLXArray?
    private(set) var fullAttentionMask: MLXArray?

    init(config: FalconPerceptionModelConfig) {
        self.config = config
        self._model.wrappedValue = FalconPerceptionTransformerModel(config: config)
        self._lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        super.init()
    }

    func prepareGroundingPrefill(
        positionIDs: MLXArray,
        posHW: MLXArray,
        ropeDelta: Int,
        attentionMask: MLXArray
    ) {
        self.prefillPositionIDs = positionIDs
        self.prefillPosHW = posHW
        self.ropeDelta = ropeDelta
        self.fullAttentionMask = attentionMask
    }

    func finishGroundingPrefill() {
        prefillPositionIDs = nil
        prefillPosHW = nil
    }

    private func makeDecodeAttentionMask(
        sequenceLength: Int,
        cacheOffset: Int
    ) -> MLXArray? {
        guard sequenceLength > 1, cacheOffset > 0 else { return nil }
        let totalKeys = cacheOffset + sequenceLength
        var values = [Bool]()
        values.reserveCapacity(sequenceLength * totalKeys)
        for queryIndex in 0..<sequenceLength {
            let allowedKeys = cacheOffset + queryIndex + 1
            for keyIndex in 0..<totalKeys {
                values.append(keyIndex < allowedKeys)
            }
        }
        return MLXArray(values, [1, 1, sequenceLength, totalKeys])
    }

    func resetGroundingState() {
        model.lastHiddenState = nil
        model.capturedLayerOutputs.removeAll(keepingCapacity: false)
        ropeDelta = nil
        prefillPositionIDs = nil
        prefillPosHW = nil
        fullAttentionMask = nil
    }

    func resolveAttentionInputs(
        sequenceLength: Int,
        cacheOffset: Int,
        mask: MLXArray?,
        positionIDs: MLXArray?,
        posHW: MLXArray?
    ) -> (mask: MLXArray?, positionIDs: MLXArray?, posHW: MLXArray?) {
        var resolvedMask = mask
        var resolvedPositionIDs = positionIDs
        var resolvedPosHW = posHW

        if resolvedPositionIDs == nil, let prefillPositionIDs, sequenceLength > 1 {
            let end = min(prefillPositionIDs.dim(0), cacheOffset + sequenceLength)
            if cacheOffset < end {
                resolvedPositionIDs = prefillPositionIDs[cacheOffset..<end]
            }
        } else if resolvedPositionIDs == nil, let ropeDelta, cacheOffset > 0 {
            let start = cacheOffset + ropeDelta
            let values = (0..<sequenceLength).map { Int32(start + $0) }
            resolvedPositionIDs = MLXArray(values, [sequenceLength])
        }

        if resolvedPosHW == nil, let prefillPosHW, sequenceLength > 1 {
            let end = min(prefillPosHW.dim(1), cacheOffset + sequenceLength)
            if cacheOffset < end {
                resolvedPosHW = prefillPosHW[0..., cacheOffset..<end, 0...]
            }
        }

        if resolvedMask == nil, let fullAttentionMask, sequenceLength > 1 {
            let end = min(fullAttentionMask.dim(2), cacheOffset + sequenceLength)
            if cacheOffset < end {
                resolvedMask = fullAttentionMask[0..., 0..., cacheOffset..<end, 0..<end]
            }
        } else if resolvedMask == nil {
            resolvedMask = makeDecodeAttentionMask(sequenceLength: sequenceLength, cacheOffset: cacheOffset)
        }

        return (resolvedMask, resolvedPositionIDs, resolvedPosHW)
    }

    func callAsFunction(
        inputIDs: MLXArray,
        inputsEmbeds: MLXArray?,
        mask: MLXArray?,
        caches: [FalconPerceptionKVCache?]?,
        positionIDs: MLXArray?,
        posHW: MLXArray?,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        let sequenceLength = inputsEmbeds?.dim(1) ?? inputIDs.dim(1)
        let cacheOffset = caches?.first??.offset ?? 0
        let resolvedInputs = resolveAttentionInputs(
            sequenceLength: sequenceLength,
            cacheOffset: cacheOffset,
            mask: mask,
            positionIDs: positionIDs,
            posHW: posHW
        )
        let hidden = model(
            inputIDs: inputIDs,
            inputsEmbeds: inputsEmbeds,
            mask: resolvedInputs.mask,
            caches: caches,
            positionIDs: resolvedInputs.positionIDs,
            posHW: resolvedInputs.posHW
        )
        let projectedHidden = lastPositionOnly && hidden.dim(1) > 1
            ? hidden[0..., (hidden.dim(1) - 1)..., 0...]
            : hidden
        return lmHead(projectedHidden)
    }
}

public final class FalconPerceptionModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "language_model") var languageModel: FalconPerceptionLanguageModel
    @ModuleInfo(key: "coord_encoder") var coordEncoder: FalconPerceptionFourierEncoder
    @ModuleInfo(key: "coord_decoder") var coordDecoder: FalconPerceptionBboxDecoder
    @ModuleInfo(key: "size_encoder") var sizeEncoder: FalconPerceptionFourierEncoder
    @ModuleInfo(key: "size_decoder") var sizeDecoder: FalconPerceptionBboxDecoder
    @ModuleInfo(key: "proj_segm") var projSegm: FalconPerceptionSegmDecoder?
    @ModuleInfo(key: "conv_segm") var convSegm: Conv2d?
    @ModuleInfo(key: "itok_upsampler") var itokUpsampler: FalconPerceptionAnyUp?

    public let config: FalconPerceptionModelConfig

    public init(config: FalconPerceptionModelConfig) {
        self.config = config
        self._languageModel.wrappedValue = FalconPerceptionLanguageModel(config: config)

        let hidden = config.textConfig.hiddenSize
        self._coordEncoder.wrappedValue = FalconPerceptionFourierEncoder(
            inDim: 2,
            featDim: config.coordEncDim,
            outDim: hidden
        )
        self._coordDecoder.wrappedValue = FalconPerceptionBboxDecoder(
            inDim: hidden,
            hiddenDim: config.coordDecDim,
            outDim: config.coordOutDim
        )
        self._sizeEncoder.wrappedValue = FalconPerceptionFourierEncoder(
            inDim: 2,
            featDim: config.sizeEncDim,
            outDim: hidden
        )
        self._sizeDecoder.wrappedValue = FalconPerceptionBboxDecoder(
            inDim: hidden,
            hiddenDim: config.sizeDecDim,
            outDim: config.sizeOutDim
        )

        if config.doSegmentation {
            self._projSegm.wrappedValue = FalconPerceptionSegmDecoder(
                inDim: hidden,
                outDim: config.segmOutDim,
                numLayers: config.numSegmLayers
            )
            self._convSegm.wrappedValue = Conv2d(
                inputChannels: hidden,
                outputChannels: config.segmOutDim,
                kernelSize: 3,
                padding: 1,
                bias: true
            )
            self._itokUpsampler.wrappedValue = FalconPerceptionAnyUp(inputDim: 3, qkDim: 128, numHeads: 4)
        } else {
            self._projSegm.wrappedValue = nil
            self._convSegm.wrappedValue = nil
            self._itokUpsampler.wrappedValue = nil
        }

        super.init()
    }

    public var lastHiddenState: MLXArray? { languageModel.model.lastHiddenState }

    func makeCaches() -> [FalconPerceptionKVCache?] {
        (0..<languageModel.model.layers.count).map { _ in FalconPerceptionKVCache() }
    }

    func prepareGroundingPrefill(
        positionData: (positionIDs: MLXArray, posHW: MLXArray, ropeDelta: Int, attentionMask: MLXArray)
    ) {
        languageModel.prepareGroundingPrefill(
            positionIDs: positionData.positionIDs,
            posHW: positionData.posHW,
            ropeDelta: positionData.ropeDelta,
            attentionMask: positionData.attentionMask
        )
    }

    func finishGroundingPrefill() {
        languageModel.finishGroundingPrefill()
    }

    func resetGroundingState() {
        languageModel.resetGroundingState()
    }

    public func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
        languageModel.model.embedTokens(inputIDs)
    }

    public func makeInputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray,
        imageGridHW: MLXArray
    ) -> MLXArray {
        let inputsEmbeds = languageModel.model.embedTokens(inputIDs)
        let hiddenStates = patchifyAndProject(pixelValues: pixelValues)
        return mergeImageFeatures(
            imageTokenID: config.imgID,
            imageFeatures: hiddenStates,
            inputsEmbeds: inputsEmbeds,
            inputIDs: inputIDs
        )
    }

    public func encodeCoordinates(into embeds: MLXArray, inputIDs: MLXArray, coordXY: MLXArray?) -> MLXArray {
        guard let coordXY else { return embeds }
        let ids = inputIDs.asArray(Int32.self).map(Int.init)
        guard ids.contains(config.coordTokenID) else { return embeds }
        let encoded = coordEncoder(coordXY.reshaped(-1, 2)).reshaped(1, -1, embeds.dim(-1))
        let output = embeds
        let sequenceLength = min(ids.count, output.dim(1))
        for index in 0..<sequenceLength where ids[index] == config.coordTokenID {
            output[0, index] = encoded[0, 0]
        }
        return output
    }

    public func encodeSizes(into embeds: MLXArray, inputIDs: MLXArray, sizeHW: MLXArray?) -> MLXArray {
        guard let sizeHW else { return embeds }
        let ids = inputIDs.asArray(Int32.self).map(Int.init)
        guard ids.contains(config.sizeTokenID) else { return embeds }
        let encoded = sizeEncoder(sizeHW.reshaped(-1, 2)).reshaped(1, -1, embeds.dim(-1))
        let output = embeds
        let sequenceLength = min(ids.count, output.dim(1))
        for index in 0..<sequenceLength where ids[index] == config.sizeTokenID {
            output[0, index] = encoded[0, 0]
        }
        return output
    }

    public func decodeCoordinates(from hiddenState: MLXArray) -> MLXArray {
        let logits = coordDecoder(hiddenState)
        let half = max(1, config.coordOutDim / 2)
        return logits.reshaped(-1, 2, half)
    }

    public func decodeSizes(from hiddenState: MLXArray) -> MLXArray {
        let logits = sizeDecoder(hiddenState)
        let half = max(1, config.sizeOutDim / 2)
        return logits.reshaped(-1, 2, half)
    }

    public func processSizes(_ logits: MLXArray) -> MLXArray {
        let numBins = logits.dim(-1)
        let predicted = argMax(logits, axis: -1).asType(.float32) / MLXArray(Float(max(1, numBins - 1)))
        let minSize = Float(log2(1.0 / Double(max(1, numBins))))
        let adjusted = predicted * MLXArray(0.0 - minSize) + MLXArray(minSize)
        return MLX.exp(adjusted * MLXArray(Float(log(2.0))))
    }

    public func computeSegmentationFeatures(
        hiddenState: MLXArray,
        inputIDs: MLXArray,
        pixelValues: MLXArray,
        gridH: Int,
        gridW: Int
    ) -> MLXArray? {
        guard let convSegm else { return nil }
        let ids = inputIDs.asArray(Int32.self).map(Int.init)
        let imageTokenPositions = ids.enumerated().compactMap { index, value in
            value == config.imgID ? index : nil
        }
        guard imageTokenPositions.count == gridH * gridW else {
            return nil
        }

        let hiddenSize = hiddenState.dim(-1)
        var imageStates: [MLXArray] = []
        imageStates.reserveCapacity(imageTokenPositions.count)
        for position in imageTokenPositions {
            imageStates.append(hiddenState[0, position].reshaped(1, hiddenSize))
        }
        let stacked = concatenated(imageStates, axis: 0).reshaped(1, gridH, gridW, hiddenSize)
        var segmentationFeatures = convSegm(stacked)

        guard let itokUpsampler else {
            return segmentationFeatures
        }

        let imageHeight = pixelValues.dim(1)
        let imageWidth = pixelValues.dim(2)
        let patchSize = max(1, config.visionConfig.spatialPatchSize)
        let maxImageDimension = max(imageHeight, imageWidth)
        let paddedEdge = ((maxImageDimension + patchSize - 1) / patchSize) * patchSize

        var upsamplerImages = pixelValues
        if paddedEdge != imageHeight || paddedEdge != imageWidth {
            upsamplerImages = padded(
                upsamplerImages,
                widths: [[0, 0], [0, paddedEdge - imageHeight], [0, paddedEdge - imageWidth], [0, 0]]
            )

            let paddedGridH = paddedEdge / patchSize
            let paddedGridW = paddedEdge / patchSize
            segmentationFeatures = padded(
                segmentationFeatures,
                widths: [[0, 0], [0, paddedGridH - gridH], [0, paddedGridW - gridW], [0, 0]]
            )
        }

        var highResolutionFeatures = itokUpsampler(images: upsamplerImages, features: segmentationFeatures)
        if paddedEdge != imageHeight || paddedEdge != imageWidth {
            highResolutionFeatures = highResolutionFeatures[0..., 0..<imageHeight, 0..<imageWidth, 0...]
        }
        return highResolutionFeatures
    }

    public func decodeSegmentationMask(
        segHidden: MLXArray,
        segmentationFeatures: MLXArray,
        outputHeight: Int,
        outputWidth: Int,
        threshold: Float
    ) -> MLXArray? {
        guard let projSegm else { return nil }
        let segToken = projSegm(segHidden)
        let logits = MLX.sum(segmentationFeatures[0] * segToken.reshaped(1, 1, -1), axis: -1)
        if logits.dim(0) == outputHeight && logits.dim(1) == outputWidth {
            return MLX.sigmoid(logits) .> MLXArray(threshold)
        }
        let resized = FalconPerceptionModel.bilinearUpsample(logits, outputHeight: outputHeight, outputWidth: outputWidth)
        return MLX.sigmoid(resized) .> MLXArray(threshold)
    }

    func forward(
        inputIDs: MLXArray,
        pixelValues: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXArray? = nil,
        caches: [FalconPerceptionKVCache?]? = nil,
        positionIDs: MLXArray? = nil,
        posHW: MLXArray? = nil,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        let embeds: MLXArray
        if let inputsEmbeds {
            embeds = inputsEmbeds
        } else if pixelValues != nil {
            preconditionFailure("Use makeInputEmbeddings before calling forward with pixelValues.")
        } else {
            embeds = languageModel.model.embedTokens(inputIDs)
        }

        return languageModel(
            inputIDs: inputIDs,
            inputsEmbeds: embeds,
            mask: mask,
            caches: caches,
            positionIDs: positionIDs,
            posHW: posHW,
            lastPositionOnly: lastPositionOnly
        )
    }

    func patchifyAndProject(pixelValues: MLXArray) -> MLXArray {
        let patchSize = config.visionConfig.spatialPatchSize
        let temporalPatchSize = config.visionConfig.temporalPatchSize
        let channels = config.visionConfig.channelSize

        let batch = pixelValues.dim(0)
        let height = pixelValues.dim(1)
        let width = pixelValues.dim(2)
        let hPatches = height / patchSize
        let wPatches = width / patchSize

        let patches = pixelValues
            .reshaped(batch, hPatches, patchSize, wPatches, patchSize, channels)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch * hPatches * wPatches, patchSize * patchSize * channels * temporalPatchSize)
        return languageModel.model.imgProjector(patches)
    }

    func mergeImageFeatures(
        imageTokenID: Int,
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIDs: MLXArray
    ) -> MLXArray {
        let ids = inputIDs.asArray(Int32.self).map(Int.init)
        let positions = ids.enumerated().compactMap { index, value in
            value == imageTokenID ? index : nil
        }
        let output = inputsEmbeds
        for (featureIndex, position) in positions.enumerated() where featureIndex < imageFeatures.dim(0) {
            output[0, position] = imageFeatures[featureIndex]
        }
        return output
    }

    public static func computePositionData(
        inputIDs: MLXArray,
        config: FalconPerceptionModelConfig,
        imageGridHW: MLXArray
    ) -> (positionIDs: MLXArray, posHW: MLXArray, ropeDelta: Int, attentionMask: MLXArray) {
        let ids = inputIDs.asArray(Int32.self).map(Int.init)
        var positions: [Int32] = []
        positions.reserveCapacity(ids.count)
        var inImage = false
        var nextPosition = 0
        for token in ids {
            if token == config.imageCLSTokenID && !inImage {
                inImage = true
            }
            positions.append(Int32(nextPosition))
            if !inImage {
                nextPosition += 1
            }
            if token == config.imgEndID && inImage {
                inImage = false
                nextPosition += 1
            }
        }

        let maxPosition = positions.map(Int.init).max() ?? 0
        let ropeDelta = maxPosition + 1 - ids.count

        let gridValues = imageGridHW.asArray(Int32.self).map(Int.init)
        let grids: [(Int, Int)] = stride(from: 0, to: gridValues.count, by: 2).map { index in
            (gridValues[index], gridValues[min(index + 1, gridValues.count - 1)])
        }

        let posHW = computePosHW(
            tokenIDs: ids,
            imageTokenID: config.imgID,
            imageGridHWS: grids
        )
        let attentionMask = createAttentionMask(
            tokenIDs: ids,
            imageCLSId: config.imageCLSTokenID,
            imageEndId: config.imgEndID
        )
        return (MLXArray(positions, [positions.count]), posHW, ropeDelta, attentionMask)
    }

    public static func computePosHW(
        tokenIDs: [Int],
        imageTokenID: Int,
        imageGridHWS: [(Int, Int)]
    ) -> MLXArray {
        let imageIndices = tokenIDs.enumerated().compactMap { index, token in
            token == imageTokenID ? index : nil
        }

        var allCoordinates: [(Float, Float)] = []
        for (gridH, gridW) in imageGridHWS {
            for hi in 0..<gridH {
                for wi in 0..<gridW {
                    let hRatio = sqrt(Float(gridH) / Float(max(1, gridW)))
                    let wRatio = sqrt(Float(gridW) / Float(max(1, gridH)))
                    let hValue = -hRatio + 2.0 * hRatio * Float(hi) / Float(max(gridH - 1, 1))
                    let wValue = -wRatio + 2.0 * wRatio * Float(wi) / Float(max(gridW - 1, 1))
                    allCoordinates.append((hValue, wValue))
                }
            }
        }

        var values = [Float](repeating: 0, count: tokenIDs.count * 2)
        for (coordIndex, tokenIndex) in imageIndices.enumerated() where coordIndex < allCoordinates.count {
            values[(tokenIndex * 2)] = allCoordinates[coordIndex].0
            values[(tokenIndex * 2) + 1] = allCoordinates[coordIndex].1
        }
        return MLXArray(values, [1, tokenIDs.count, 2])
    }

    public static func createAttentionMask(
        tokenIDs: [Int],
        imageCLSId: Int,
        imageEndId: Int
    ) -> MLXArray {
        let count = tokenIDs.count
        var blockIDs = [Int](repeating: 0, count: count)
        var inImage = false
        var block = 0
        for (index, token) in tokenIDs.enumerated() {
            if token == imageCLSId && !inImage {
                inImage = true
                block += 1
            }

            // Mirror mlx-vlm: the image-start token participates in the image block,
            // but the closing <img_end> token does not.
            let isImageToken = inImage && token != imageEndId
            if isImageToken {
                blockIDs[index] = block
            }
            if token == imageEndId && inImage {
                inImage = false
            }
        }

        var values = [Bool](repeating: false, count: count * count)
        for q in 0..<count {
            for k in 0..<count {
                let causal = q >= k
                let sameImage = blockIDs[q] > 0 && blockIDs[q] == blockIDs[k]
                values[(q * count) + k] = causal || sameImage
            }
        }
        return MLXArray(values, [1, 1, count, count])
    }

    static func computeGoldenFrequencies(freqsGolden: MLXArray, posHW: MLXArray) -> (MLXArray, MLXArray) {
        let heads = freqsGolden.dim(0)
        let frequencyCount = freqsGolden.dim(1)
        let tokenCount = posHW.dim(1)
        let positions = posHW[0].asType(.float32).reshaped(tokenCount, 1, 1, 2)
        let frequencies = freqsGolden.asType(.float32).reshaped(1, heads, frequencyCount, 2)
        let angles = MLX.sum(positions * frequencies, axis: -1)
        return (
            MLX.cos(angles).reshaped(1, tokenCount, heads, frequencyCount),
            MLX.sin(angles).reshaped(1, tokenCount, heads, frequencyCount)
        )
    }

    static func selectCosSinTables(
        positionIDs: MLXArray?,
        cosTable: MLXArray,
        sinTable: MLXArray
    ) -> (MLXArray?, MLXArray?) {
        guard let positionIDs else { return (nil, nil) }
        let positions = MLX.maximum(positionIDs.asType(.int32), MLXArray(Int32(0)))
        return (MLX.take(cosTable, positions, axis: 0), MLX.take(sinTable, positions, axis: 0))
    }

    static func applyRotaryEmbeddings(
        queries: MLXArray,
        keys: MLXArray,
        cos1D: MLXArray,
        sin1D: MLXArray,
        cos2D: MLXArray?,
        sin2D: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let half = queries.dim(-1) / 2
        let qTemporal = queries[.ellipsis, 0..<half]
        let qSpatial = queries[.ellipsis, half...]
        let kTemporal = keys[.ellipsis, 0..<half]
        let kSpatial = keys[.ellipsis, half...]

        let rotatedTemporal = apply1DRotary(queries: qTemporal, keys: kTemporal, cos: cos1D, sin: sin1D)
        let rotatedSpatial: (MLXArray, MLXArray)
        if let cos2D, let sin2D {
            rotatedSpatial = (
                apply2DRotary(x: qSpatial, cos2D: cos2D, sin2D: sin2D),
                apply2DRotary(x: kSpatial, cos2D: cos2D, sin2D: sin2D)
            )
        } else {
            rotatedSpatial = (qSpatial, kSpatial)
        }

        return (
            concatenated([rotatedTemporal.0, rotatedSpatial.0], axis: -1).asType(queries.dtype),
            concatenated([rotatedTemporal.1, rotatedSpatial.1], axis: -1).asType(keys.dtype)
        )
    }

    private static func apply1DRotary(
        queries: MLXArray,
        keys: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let dtypeQ = queries.dtype
        let dtypeK = keys.dtype
        let qReshaped = queries.asType(.float32).reshaped(queries.dim(0), queries.dim(1), queries.dim(2), max(1, queries.dim(3) / 2), 2)
        let kReshaped = keys.asType(.float32).reshaped(keys.dim(0), keys.dim(1), keys.dim(2), max(1, keys.dim(3) / 2), 2)

        let qEven = qReshaped[.ellipsis, 0]
        let qOdd = qReshaped[.ellipsis, 1]
        let kEven = kReshaped[.ellipsis, 0]
        let kOdd = kReshaped[.ellipsis, 1]

        let cosExpanded = cos.reshaped(1, 1, cos.dim(0), cos.dim(1))
        let sinExpanded = sin.reshaped(1, 1, sin.dim(0), sin.dim(1))

        let qOutEven = qEven * cosExpanded - qOdd * sinExpanded
        let qOutOdd = qEven * sinExpanded + qOdd * cosExpanded
        let kOutEven = kEven * cosExpanded - kOdd * sinExpanded
        let kOutOdd = kEven * sinExpanded + kOdd * cosExpanded

        let qOut = stacked([qOutEven, qOutOdd], axis: -1).reshaped(queries.shape).asType(dtypeQ)
        let kOut = stacked([kOutEven, kOutOdd], axis: -1).reshaped(keys.shape).asType(dtypeK)
        return (qOut, kOut)
    }

    private static func apply2DRotary(
        x: MLXArray,
        cos2D: MLXArray,
        sin2D: MLXArray
    ) -> MLXArray {
        let dtype = x.dtype
        let cos = cos2D.transposed(0, 2, 1, 3)
        let sin = sin2D.transposed(0, 2, 1, 3)
        let reshaped = x.asType(.float32).reshaped(x.dim(0), x.dim(1), x.dim(2), max(1, x.dim(3) / 2), 2)
        let even = reshaped[.ellipsis, 0]
        let odd = reshaped[.ellipsis, 1]
        let outEven = even * cos - odd * sin
        let outOdd = even * sin + odd * cos
        return stacked([outEven, outOdd], axis: -1).reshaped(x.shape).asType(dtype)
    }

    static func bilinearUpsample(_ x: MLXArray, outputHeight: Int, outputWidth: Int) -> MLXArray {
        let inputHeight = x.dim(0)
        let inputWidth = x.dim(1)
        if inputHeight == outputHeight && inputWidth == outputWidth {
            return x
        }

        let inputValues = x.asType(.float32).asArray(Float.self)
        var outputValues = [Float](repeating: 0, count: outputHeight * outputWidth)
        for y in 0..<outputHeight {
            let sourceY = ((Float(y) + 0.5) * (Float(inputHeight) / Float(outputHeight))) - 0.5
            let clampedY = max(0, min(Float(inputHeight - 1), sourceY))
            let y0 = Int(floor(clampedY))
            let y1 = min(inputHeight - 1, y0 + 1)
            let wy = clampedY - Float(y0)
            for xIndex in 0..<outputWidth {
                let sourceX = ((Float(xIndex) + 0.5) * (Float(inputWidth) / Float(outputWidth))) - 0.5
                let clampedX = max(0, min(Float(inputWidth - 1), sourceX))
                let x0 = Int(floor(clampedX))
                let x1 = min(inputWidth - 1, x0 + 1)
                let wx = clampedX - Float(x0)

                let v00 = inputValues[(y0 * inputWidth) + x0]
                let v01 = inputValues[(y0 * inputWidth) + x1]
                let v10 = inputValues[(y1 * inputWidth) + x0]
                let v11 = inputValues[(y1 * inputWidth) + x1]

                let top = ((1 - wx) * v00) + (wx * v01)
                let bottom = ((1 - wx) * v10) + (wx * v11)
                outputValues[(y * outputWidth) + xIndex] = ((1 - wy) * top) + (wy * bottom)
            }
        }
        return MLXArray(outputValues, [outputHeight, outputWidth])
    }
}
