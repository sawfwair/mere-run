import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Cmlx

@inline(__always)
private func gemma4RMSNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let noWeight = MLXArray.mlxNone
    let stream = StreamOrDevice.default
    var result = mlx_array_new()
    _ = withExtendedLifetime((noWeight, stream)) {
        mlx_fast_rms_norm(&result, x.ctx, noWeight.ctx, eps, stream.ctx)
    }
    return MLXArray(result)
}

@inline(__always)
private func gemma4RMSNormNoScale(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: weight, eps: eps)
}

struct Gemma4SharedKVState {
    let keys: MLXArray
    let values: MLXArray
    let offset: Int
    let maxSize: Int?
}

struct Gemma4LanguageModelOutput {
    let hidden: MLXArray
    let preNormHidden: MLXArray
    let sharedKVStates: [String: Gemma4SharedKVState]
}

struct Gemma4ForwardOutput {
    let logits: MLXArray
    let hidden: MLXArray
    let sharedKVStates: [String: Gemma4SharedKVState]
}

protocol Gemma4CausalModel: AnyObject, Sendable {
    func forward(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray
    /// Forward for a prefill chunk: fills the KV cache but only returns logits
    /// for the chunk's final position.
    func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray
    func forwardForSpeculation(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> Gemma4ForwardOutput
    func inputEmbeddings(for inputIds: MLXArray) -> MLXArray
    func speculativeLogits(fromHidden hidden: MLXArray) -> MLXArray
    func speculativeDraftHidden(_ hidden: MLXArray) -> MLXArray
    func makeAttentionCache(quantization: Gemma4KVCacheQuantization?) -> [Gemma4AttentionCache]
}

/// Proportional RoPE for Gemma 4 full-attention layers.
///
/// Frequencies are computed relative to the **full** head dimension (not just the
/// rotated portion), and rotation is applied to the first `rotatedDims/2`
/// elements of each half of the head — matching HF's rotate_half convention.
final class Gemma4ProportionalRoPE: Module, OffsetLayer {
    private let dims: Int
    private let rotatedDims: Int
    private let traditional: Bool
    private let freqs: MLXArray?

    init(dims: Int, traditional: Bool = false, base: Float = 10_000, partialRotaryFactor: Float = 1.0, factor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        let ropeAngles = Int(partialRotaryFactor * Float(dims / 2))
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2))
                / MLXArray(Float(dims))
            self.freqs = MLXArray(factor) * pow(MLXArray(base), exponents)
        } else {
            self.freqs = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0 else { return x }

        let head = x[0..., 0..., 0..., ..<dims]
        let half = dims / 2
        let rotHalf = rotatedDims / 2

        let left = head[0..., 0..., 0..., ..<half]
        let right = head[0..., 0..., 0..., half...]

        let toRotate = concatenated(
            [left[0..., 0..., 0..., ..<rotHalf], right[0..., 0..., 0..., ..<rotHalf]],
            axis: -1
        )
        let rotated = MLXFast.RoPE(
            toRotate,
            dimensions: rotatedDims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )

        let newLeft = concatenated(
            [rotated[0..., 0..., 0..., ..<rotHalf], left[0..., 0..., 0..., rotHalf...]],
            axis: -1
        )
        let newRight = concatenated(
            [rotated[0..., 0..., 0..., rotHalf...], right[0..., 0..., 0..., rotHalf...]],
            axis: -1
        )
        let newHead = concatenated([newLeft, newRight], axis: -1)

        if x.dim(-1) > dims {
            return concatenated([newHead, x[0..., 0..., 0..., dims...]], axis: -1)
        }
        return newHead
    }
}

protocol Gemma4AttentionCache: AnyObject {
    var offset: Int { get }
    func currentState() -> (MLXArray, MLXArray)?
    /// State for single-token decode, where softmax attention is invariant to
    /// key/value ordering (positions are already baked in via RoPE at append
    /// time and the decode mask is `.none`). Ring-buffer caches may return
    /// storage order here to skip the temporal-order copy. Multi-token queries
    /// must keep using `currentState()` — their causal masks assume temporal
    /// order.
    func decodeState() -> (MLXArray, MLXArray)?
    func append(keys: MLXArray, values: MLXArray)
    /// Appends the new keys and values and returns the state that a query of
    /// the same length must attend over. Sliding caches override this for
    /// multi-token chunked prefill so early queries in the chunk can still see
    /// the preceding window even though the resident cache is trimmed for the
    /// next step.
    func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)?
    func fork() -> Gemma4AttentionCache
    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache?
    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]?
    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray?
    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache?
    func storageArraysForEvaluation() -> [MLXArray]
    func evaluateStorage()
}

extension Gemma4CausalModel {
    func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        forward(inputIds: inputIds, cache: cache)
    }
}

extension Gemma4AttentionCache {
    func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)? {
        append(keys: keys, values: values)
        return keys.dim(2) == 1 ? decodeState() : currentState()
    }

    func decodeState() -> (MLXArray, MLXArray)? {
        currentState()
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        nil
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        nil
    }

    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        nil
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        nil
    }

    func storageArraysForEvaluation() -> [MLXArray] {
        guard let state = currentState() else { return [] }
        return [state.0, state.1]
    }

    func evaluateStorage() {}
}

func evaluateGemma4CacheStorage(_ caches: [Gemma4AttentionCache]) {
    let arrays = caches.flatMap { $0.storageArraysForEvaluation() }
    guard !arrays.isEmpty else { return }
    MLX.eval(arrays)
}

final class Gemma4FullKVCache: Gemma4AttentionCache {
    private static let allocationStep = 256

    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset: Int = 0

    func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        guard offset < keys.dim(2) else {
            return (keys, values)
        }
        return (keys[0..., 0..., 0..<offset, 0...], values[0..., 0..., 0..<offset, 0...])
    }

    func append(keys: MLXArray, values: MLXArray) {
        let previousOffset = offset
        let newOffset = previousOffset + keys.dim(2)

        if self.keys == nil || newOffset > (self.keys?.dim(2) ?? 0) {
            let batch = keys.dim(0)
            let heads = keys.dim(1)
            let keyDim = keys.dim(3)
            let valueDim = values.dim(3)
            let missing = max(0, newOffset - (self.keys?.dim(2) ?? 0))
            let steps = max(1, (missing + Self.allocationStep - 1) / Self.allocationStep)
            let growth = steps * Self.allocationStep
            let newKeys = MLXArray.zeros([batch, heads, growth, keyDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros([batch, heads, growth, valueDim], dtype: values.dtype)
            if let existingKeys = self.keys, let existingValues = self.values {
                let trimmedKeys = previousOffset < existingKeys.dim(2)
                    ? existingKeys[0..., 0..., 0..<previousOffset, 0...]
                    : existingKeys
                let trimmedValues = previousOffset < existingValues.dim(2)
                    ? existingValues[0..., 0..., 0..<previousOffset, 0...]
                    : existingValues
                self.keys = concatenated([trimmedKeys, newKeys], axis: 2)
                self.values = concatenated([trimmedValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        guard let storedKeys = self.keys, let storedValues = self.values else {
            preconditionFailure("Gemma4FullKVCache should allocate before writing.")
        }
        storedKeys[0..., 0..., previousOffset..<newOffset, 0...] = keys
        storedValues[0..., 0..., previousOffset..<newOffset, 0...] = values
        self.offset = newOffset
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4FullKVCache()
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }

    static func reencoded(keys: MLXArray, values: MLXArray, offset: Int) -> Gemma4FullKVCache {
        let cache = Gemma4FullKVCache()
        cache.keys = keys
        cache.values = values
        cache.offset = offset
        return cache
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        guard let state = currentState() else { return nil }
        return makeGemma4AttentionCache(
            keys: state.0,
            values: state.1,
            offset: offset,
            maxSize: nil,
            quantization: quantization
        )
    }

    func evaluateStorage() {
        if let keys, let values {
            MLX.eval(keys, values)
        }
    }

    func storageArraysForEvaluation() -> [MLXArray] {
        guard let keys, let values else { return [] }
        return [keys, values]
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4FullKVCache],
              !typed.isEmpty,
              typed.allSatisfy({ $0.offset == offset }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4FullKVCache()
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0, let keys, let values, keys.dim(0) == count, values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4FullKVCache()
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            return copy
        }
    }
}

final class Gemma4SlidingKVCache: Gemma4AttentionCache {
    private static let allocationStep = 256

    private let maxSize: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset: Int = 0
    private var writeIndex: Int = 0

    init(maxSize: Int, initialOffset: Int = 0) {
        self.maxSize = max(1, maxSize)
        self.offset = max(0, initialOffset)
    }

    var configuredMaxSize: Int {
        maxSize
    }

    func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        return (temporalOrder(keys), temporalOrder(values))
    }

    func decodeState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        // Storage order is fine for q-len 1: every slot is a valid in-window
        // token once the ring is full, and attention over an unmasked key set
        // is permutation-invariant. Before the ring fills, the valid prefix is
        // already in temporal order.
        if offset < keys.dim(2) {
            return (
                keys[0..., 0..., 0..<offset, 0...],
                values[0..., 0..., 0..<offset, 0...]
            )
        }
        return (keys, values)
    }

    func append(keys: MLXArray, values: MLXArray) {
        if keys.dim(2) == 1 {
            updateInPlace(keys: keys, values: values)
        } else {
            updateByConcatenating(keys: keys, values: values)
        }
    }

    func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)? {
        guard keys.dim(2) > 1 else {
            append(keys: keys, values: values)
            return decodeState()
        }

        let previous = currentState()
        append(keys: keys, values: values)
        guard let previous else {
            return currentState()
        }
        return (
            concatenated([previous.0, keys], axis: 2),
            concatenated([previous.1, values], axis: 2)
        )
    }

    private func updateByConcatenating(keys: MLXArray, values: MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            let orderedKeys = temporalOrder(existingKeys)
            let orderedValues = temporalOrder(existingValues)
            writeIndex = orderedKeys.dim(2)
            let trimSize = writeIndex - maxSize + keys.dim(2)
            self.keys = trim(orderedKeys, trimSize: trimSize, appending: keys)
            self.values = trim(orderedValues, trimSize: trimSize, appending: values)
        } else {
            self.keys = keys
            self.values = values
        }
        offset += keys.dim(2)
        writeIndex = self.keys?.dim(2) ?? 0
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) {
        let previousOffset = offset
        if self.keys == nil || (previousOffset >= (self.keys?.dim(2) ?? 0) && (self.keys?.dim(2) ?? 0) < maxSize) {
            let batch = keys.dim(0)
            let heads = keys.dim(1)
            let keyDim = keys.dim(3)
            let valueDim = values.dim(3)
            let existingSize = self.keys?.dim(2) ?? 0
            let growth = min(Self.allocationStep, maxSize - existingSize)
            if growth > 0 {
                let newKeys = MLXArray.zeros([batch, heads, growth, keyDim], dtype: keys.dtype)
                let newValues = MLXArray.zeros([batch, heads, growth, valueDim], dtype: values.dtype)
                if let existingKeys = self.keys, let existingValues = self.values {
                    let trimmedKeys = previousOffset < existingKeys.dim(2)
                        ? existingKeys[0..., 0..., 0..<previousOffset, 0...]
                        : existingKeys
                    let trimmedValues = previousOffset < existingValues.dim(2)
                        ? existingValues[0..., 0..., 0..<previousOffset, 0...]
                        : existingValues
                    self.keys = concatenated([trimmedKeys, newKeys], axis: 2)
                    self.values = concatenated([trimmedValues, newValues], axis: 2)
                } else {
                    self.keys = newKeys
                    self.values = newValues
                }
                writeIndex = min(previousOffset, self.keys?.dim(2) ?? previousOffset)
            }
        }

        if let existingKeys = self.keys, existingKeys.dim(2) > maxSize {
            self.keys = trim(existingKeys, trimSize: existingKeys.dim(2) - maxSize)
            writeIndex = maxSize
        }
        if let existingValues = self.values, existingValues.dim(2) > maxSize {
            self.values = trim(existingValues, trimSize: existingValues.dim(2) - maxSize)
        }
        if writeIndex >= maxSize {
            writeIndex = 0
        }

        let newWriteIndex = writeIndex + keys.dim(2)
        guard let storedKeys = self.keys, let storedValues = self.values else {
            preconditionFailure("Gemma4SlidingKVCache should allocate before writing.")
        }
        storedKeys[0..., 0..., writeIndex..<newWriteIndex, 0...] = keys
        storedValues[0..., 0..., writeIndex..<newWriteIndex, 0...] = values
        offset += keys.dim(2)
        writeIndex = newWriteIndex
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        let length = array.dim(2)
        if offset < length {
            return array[0..., 0..., 0..<offset, 0...]
        }
        guard writeIndex < length, writeIndex < offset else {
            return array
        }
        return concatenated([
            array[0..., 0..., writeIndex..., 0...],
            array[0..., 0..., 0..<writeIndex, 0...],
        ], axis: 2)
    }

    private func trim(_ array: MLXArray, trimSize: Int, appending append: MLXArray? = nil) -> MLXArray {
        var parts: [MLXArray]
        if trimSize > 0 {
            parts = [array[0..., 0..., trimSize..., 0...]]
        } else {
            parts = [array]
        }
        if let append {
            parts.append(append)
        }
        return concatenated(parts, axis: 2)
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4SlidingKVCache(maxSize: maxSize)
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        copy.writeIndex = writeIndex
        return copy
    }

    static func reencoded(
        keys: MLXArray,
        values: MLXArray,
        offset: Int,
        maxSize: Int
    ) -> Gemma4SlidingKVCache {
        let cache = Gemma4SlidingKVCache(maxSize: maxSize)
        let totalLength = keys.dim(2)
        if totalLength > cache.maxSize {
            let start = totalLength - cache.maxSize
            cache.keys = keys[0..., 0..., start..., 0...]
            cache.values = values[0..., 0..., start..., 0...]
        } else {
            cache.keys = keys
            cache.values = values
        }
        cache.offset = offset
        cache.writeIndex = cache.keys?.dim(2) ?? 0
        return cache
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        guard let state = currentState() else { return nil }
        return makeGemma4AttentionCache(
            keys: state.0,
            values: state.1,
            offset: offset,
            maxSize: maxSize,
            quantization: quantization
        )
    }

    func evaluateStorage() {
        if let keys, let values {
            MLX.eval(keys, values)
        }
    }

    func storageArraysForEvaluation() -> [MLXArray] {
        guard let keys, let values else { return [] }
        return [keys, values]
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4SlidingKVCache],
              !typed.isEmpty,
              typed.allSatisfy({ $0.offset == offset && $0.maxSize == maxSize }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4SlidingKVCache(maxSize: maxSize)
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        copy.writeIndex = copy.keys?.dim(2) ?? 0
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0, let keys, let values, keys.dim(0) == count, values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4SlidingKVCache(maxSize: maxSize)
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            copy.writeIndex = writeIndex
            return copy
        }
    }
}

func makeGemma4AttentionCache(
    keys: MLXArray,
    values: MLXArray,
    offset: Int,
    maxSize: Int?,
    quantization: Gemma4KVCacheQuantization? = nil
) -> Gemma4AttentionCache {
    if let quantization, quantization.isEnabled {
        if quantization.scheme == .polar {
            return Gemma4PolarKVCache.reencoded(
                keys: keys,
                values: values,
                configuration: quantization,
                maxSize: maxSize,
                offset: offset
            )
        }
        return Gemma4QuantizedKVCache.reencoded(
            keys: keys,
            values: values,
            configuration: quantization,
            maxSize: maxSize,
            offset: offset
        )
    }

    if let maxSize {
        return Gemma4SlidingKVCache.reencoded(keys: keys, values: values, offset: offset, maxSize: maxSize)
    }
    return Gemma4FullKVCache.reencoded(keys: keys, values: values, offset: offset)
}

final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    private var fusedGateUp: Gemma4FusedQuantizedProjection?
    private var fusedGateUpAttempted = false

    init(config: Gemma4TextConfig, layerIndex: Int, forceKVShared: Bool = false) {
        let firstSharedIndex = config.numHiddenLayers - config.numKVSharedLayers
        let isSharedKVLayer = forceKVShared || (config.numKVSharedLayers > 0 && layerIndex >= firstSharedIndex)
        let widthMultiplier = (config.useDoubleWideMLP && isSharedKVLayer) ? 2 : 1
        let intermediate = config.intermediateSize * widthMultiplier

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let fused = resolvedFusedGateUp() {
            let parts = fused.callSplit(x)
            return downProj(geluApproximate(parts[0]) * parts[1])
        }
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }

    func resolvedFusedGateUp() -> Gemma4FusedQuantizedProjection? {
        guard Gemma4FusedProjectionPolicy.enabled else { return nil }
        let sources: [Linear?] = [gateProj, upProj]
        if let fused = fusedGateUp {
            if fused.matches(sources) { return fused }
            fusedGateUp = nil
            fusedGateUpAttempted = false
        }
        if !fusedGateUpAttempted {
            fusedGateUpAttempted = true
            fusedGateUp = Gemma4FusedQuantizedProjection.fuse(sources)
        }
        return fusedGateUp
    }
}

final class Gemma4SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode

    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        groupSize: Int = 16,
        bits: Int = 4,
        mode: QuantizationMode = .nvfp4,
        bias: Bool = false
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let scale = sqrt(1.0 / Float(max(1, inputDims)))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        self._biases.wrappedValue = nil
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)
        let batchTokens = batch * sequenceLength

        let flatX: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatX = x.reshaped([batchTokens * topK, 1, inputDim])
        } else {
            var expanded = x.reshaped([batchTokens, 1, inputDim])
            expanded = MLX.expandedDimensions(expanded, axis: 1)
            expanded = MLX.repeated(expanded, count: topK, axis: 1)
            flatX = expanded.reshaped([batchTokens * topK, 1, inputDim])
        }

        let flatIndices = indices.reshaped([batchTokens * topK])
        let output: MLXArray
        if let scales {
            output = portableGatherQuantizedMM(
                flatX,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: flatIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                sortedIndices: false
            )
        } else {
            output = gatherMM(
                flatX,
                weight.swappedAxes(-1, -2),
                rhsIndices: flatIndices,
                sortedIndices: false
            )
        }

        let outputDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outputDim])
        reshaped = reshaped.reshaped([batch, sequenceLength, topK, outputDim])

        if let bias {
            let selectedBias = take(bias, flatIndices, axis: 0)
                .reshaped([batch, sequenceLength, topK, outputDim])
            return reshaped + selectedBias
        }
        return reshaped
    }
}

final class Gemma4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4SwitchLinear

    init(config: Gemma4TextConfig) {
        self._gateProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: max(1, config.moeIntermediateSize),
            numExperts: max(1, config.numExperts)
        )
        self._upProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: max(1, config.moeIntermediateSize),
            numExperts: max(1, config.numExperts)
        )
        self._downProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: max(1, config.moeIntermediateSize),
            outputDims: config.hiddenSize,
            numExperts: max(1, config.numExperts)
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        return downProj(geluApproximate(gate) * up, indices: indices)
    }
}

final class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    private let topK: Int
    private let eps: Float
    private let rootSize: Float

    init(config: Gemma4TextConfig) {
        self.topK = max(1, config.topKExperts)
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(max(1, config.hiddenSize)), -0.5)
        self._proj.wrappedValue = Linear(config.hiddenSize, max(1, config.numExperts), bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([max(1, config.numExperts)])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let normWeight = scale * MLXArray(rootSize).asType(scale.dtype)
        let routedInput = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        let expertScores = proj(routedInput)
        let k = min(topK, expertScores.dim(-1))
        let indices = argPartition(-expertScores, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        var weights = takeAlong(expertScores, indices, axis: -1)
        weights = softmax(weights, axis: -1)
        weights = weights * take(perExpertScale, indices, axis: 0)
        return (indices, weights)
    }
}

final class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: Gemma4SwitchGLU

    init(config: Gemma4TextConfig) {
        self._switchGLU.wrappedValue = Gemma4SwitchGLU(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let routed = switchGLU(x, indices: indices)
        return (routed * MLX.expandedDimensions(weights, axis: weights.ndim)).sum(axis: -2)
    }
}

final class Gemma4Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let layerType: String
    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let windowSize: Int
    private let rope: any OffsetLayer
    private let rmsNormEps: Float
    private let isKVSharedLayer: Bool
    private let useKeyEqualsValue: Bool
    private var fusedQKV: Gemma4FusedQuantizedProjection?
    private var fusedQKVAttempted = false
    private var compiledQKVSegment: Gemma4CompiledSegment?
    private var compiledQKVAttempted = false

    init(config: Gemma4TextConfig, layerIndex: Int, forceKVShared: Bool = false) {
        self.layerType = config.layerTypes[layerIndex]
        self.numHeads = config.numAttentionHeads
        self.windowSize = config.slidingWindow
        self.rmsNormEps = config.rmsNormEps

        let isFullAttention = layerType == "full_attention"
        self.headDim = isFullAttention ? (config.globalHeadDim ?? config.headDim) : config.headDim
        self.numKVHeads = isFullAttention ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads) : config.numKeyValueHeads
        self.isKVSharedLayer = forceKVShared || (config.numKVSharedLayers > 0 && layerIndex >= (config.numHiddenLayers - config.numKVSharedLayers))
        self.useKeyEqualsValue = config.attentionKEqV && isFullAttention

        let ropeConfig = config.ropeParameters[layerType] ?? config.ropeParameters["sliding_attention"]
        let ropeBase = ropeConfig?.ropeTheta ?? 10_000
        let partial = max(0, min(1, ropeConfig?.partialRotaryFactor ?? 1))
        let ropeType = ropeConfig?.ropeType ?? "default"
        if ropeType == "proportional" {
            self.rope = Gemma4ProportionalRoPE(
                dims: headDim,
                traditional: false,
                base: ropeBase,
                partialRotaryFactor: partial
            )
        } else {
            let ropeDims = max(1, Int(Float(headDim) * partial))
            self.rope = RoPE(
                dimensions: ropeDims,
                traditional: false,
                base: ropeBase
            )
        }

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = useKeyEqualsValue
            ? nil
            : Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        visionBlockIDs: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let sequenceLength = x.dim(1)
        let offset = cache?.offset ?? 0

        func preRopeQueryHeads(_ rawQueries: MLXArray) -> MLXArray {
            let shaped = rawQueries.reshaped(batchSize, sequenceLength, numHeads, headDim)
            return qNorm(shaped).transposed(0, 2, 1, 3)
        }

        func preRopeKeyValueHeads(rawKeys: MLXArray, rawValues: MLXArray) -> (MLXArray, MLXArray) {
            let normedKeys = kNorm(rawKeys.reshaped(batchSize, sequenceLength, numKVHeads, headDim))
            let normedValues = gemma4RMSNormNoScale(
                rawValues.reshaped(batchSize, sequenceLength, numKVHeads, headDim),
                eps: rmsNormEps
            )
            return (normedKeys.transposed(0, 2, 1, 3), normedValues.transposed(0, 2, 1, 3))
        }

        func finishedQueries(_ rawQueries: MLXArray) -> MLXArray {
            rope(preRopeQueryHeads(rawQueries), offset: offset)
        }

        func finishedKeyValues(rawKeys: MLXArray, rawValues: MLXArray) -> (MLXArray, MLXArray) {
            let (preRopeKeys, computedValues) = preRopeKeyValueHeads(rawKeys: rawKeys, rawValues: rawValues)
            return (rope(preRopeKeys, offset: offset), computedValues)
        }

        let scale: Float = 1.0
        let repeats = max(1, numHeads / max(1, numKVHeads))

        let queries: MLXArray
        let keys: MLXArray
        let values: MLXArray
        if isKVSharedLayer, let cache {
            queries = finishedQueries(qProj(x))
            if sequenceLength == 1,
               let attended = cache.specializedAttention(queries: queries, repeats: repeats, scale: scale) {
                let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
                return oProj(reshaped)
            }
            if let shared = sequenceLength == 1 ? cache.decodeState() : cache.currentState() {
                keys = shared.0
                values = shared.1
            } else {
                let rawKeys = kProj(x)
                let rawValues = useKeyEqualsValue ? rawKeys : vProj!(x)
                let (computedKeys, computedValues) = finishedKeyValues(rawKeys: rawKeys, rawValues: rawValues)

                cache.append(keys: computedKeys, values: computedValues)
                let updated = cache.currentState()!
                keys = updated.0
                values = updated.1
            }
        } else {
            let computedKeys: MLXArray
            let computedValues: MLXArray
            if let fusedHeads = fusedDecodeQKV(x, sequenceLength: sequenceLength) {
                queries = rope(fusedHeads.0, offset: offset)
                computedKeys = rope(fusedHeads.1, offset: offset)
                computedValues = fusedHeads.2
            } else if let compiled = resolvedCompiledQKVSegment(sequenceLength: sequenceLength) {
                let parts = compiled.function([x])
                queries = rope(parts[0], offset: offset)
                computedKeys = rope(parts[1], offset: offset)
                computedValues = parts[2]
            } else {
                let rawQueries: MLXArray
                let rawKeys: MLXArray
                let rawValues: MLXArray
                if let fused = resolvedFusedQKV() {
                    let parts = fused.callSplit(x)
                    rawQueries = parts[0]
                    rawKeys = parts[1]
                    rawValues = useKeyEqualsValue ? parts[1] : parts[2]
                } else {
                    rawQueries = qProj(x)
                    rawKeys = kProj(x)
                    rawValues = useKeyEqualsValue ? rawKeys : vProj!(x)
                }
                queries = finishedQueries(rawQueries)
                let keyValues = finishedKeyValues(rawKeys: rawKeys, rawValues: rawValues)
                computedKeys = keyValues.0
                computedValues = keyValues.1
            }

            if let cache {
                cache.append(keys: computedKeys, values: computedValues)
                if sequenceLength == 1,
                   let attended = cache.specializedAttention(queries: queries, repeats: repeats, scale: scale) {
                    let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
                    return oProj(reshaped)
                }
                let updated = (sequenceLength == 1 ? cache.decodeState() : cache.currentState())!
                keys = updated.0
                values = updated.1
            } else {
                keys = computedKeys
                values = computedValues
            }
        }

        let mask = makeAttentionMask(
            queryLength: sequenceLength,
            queryOffset: offset,
            keyLength: keys.dim(2),
            windowSize: layerType == "sliding_attention" ? windowSize : nil,
            dtype: x.dtype,
            visionBlockIDs: visionBlockIDs
        )

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, numHeads * headDim)
        return oProj(reshaped)
    }

    /// Final-layer text-prefill path for callers that consume only the final
    /// hidden row. Every K/V row is normalized, rotated, and committed to the
    /// cache; only Q and the attention output are restricted to the final row.
    func callLastPrefillRow(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let sequenceLength = x.dim(1)
        precondition(batchSize == 1 && sequenceLength > 1)
        precondition(layerType == "full_attention" && !isKVSharedLayer)

        let offset = cache?.offset ?? 0
        let lastInput = x[0..., (sequenceLength - 1)..., 0...]
        let rawQueries = qProj(lastInput)
        let rawKeys = kProj(x)
        let rawValues = useKeyEqualsValue ? rawKeys : vProj!(x)

        var queries = qNorm(
            rawQueries.reshaped(batchSize, 1, numHeads, headDim)
        ).transposed(0, 2, 1, 3)
        var keys = kNorm(
            rawKeys.reshaped(batchSize, sequenceLength, numKVHeads, headDim)
        ).transposed(0, 2, 1, 3)
        var values = gemma4RMSNormNoScale(
            rawValues.reshaped(batchSize, sequenceLength, numKVHeads, headDim),
            eps: rmsNormEps
        ).transposed(0, 2, 1, 3)

        queries = rope(queries, offset: offset + sequenceLength - 1)
        keys = rope(keys, offset: offset)
        if let cache {
            guard let state = cache.attentionState(appending: keys, values: values) else {
                preconditionFailure("Gemma 4 terminal prefill cache must expose appended K/V state.")
            }
            keys = state.0
            values = state.1
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: .none
        )
        let reshaped = attended.transposed(0, 2, 1, 3).reshaped(
            batchSize,
            1,
            numHeads * headDim
        )
        return oProj(reshaped)
    }

    private func resolvedFusedQKV() -> Gemma4FusedQuantizedProjection? {
        guard Gemma4FusedProjectionPolicy.enabled else { return nil }
        let sources: [Linear?] = useKeyEqualsValue ? [qProj, kProj] : [qProj, kProj, vProj]
        if let fused = fusedQKV {
            if fused.matches(sources) { return fused }
            fusedQKV = nil
            fusedQKVAttempted = false
            compiledQKVSegment = nil
            compiledQKVAttempted = false
        }
        if !fusedQKVAttempted {
            fusedQKVAttempted = true
            fusedQKV = Gemma4FusedQuantizedProjection.fuse(sources)
        }
        return fusedQKV
    }

    /// Fused-kernel decode path: one quantized matmul over the concatenated
    /// QKV weights, then a single Metal kernel that splits heads and applies
    /// qNorm, kNorm, and the value no-scale norm in transposed layout.
    private func fusedDecodeQKV(
        _ x: MLXArray,
        sequenceLength: Int
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard Gemma4FusedProjectionPolicy.fusedDecodeKernelsEnabled,
              sequenceLength == 1,
              !useKeyEqualsValue else {
            return nil
        }
        guard let fused = resolvedFusedQKV() else { return nil }
        return Gemma4DecodeFusedKernels.qkvNorms(
            qkv: fused.callFused(x),
            qNormWeight: qNorm.weight,
            kNormWeight: kNorm.weight,
            eps: Gemma4DecodeScalarCache.epsilon(rmsNormEps),
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim
        )
    }

    /// Compiled decode segment: fused QKV matmul, head reshape, q/k norms, and
    /// the value no-scale norm — everything between the layer input and RoPE.
    /// RoPE stays outside because its integer offset changes every token, which
    /// would force a retrace per position.
    private func resolvedCompiledQKVSegment(sequenceLength: Int) -> Gemma4CompiledSegment? {
        guard Gemma4FusedProjectionPolicy.compiledSegmentsEnabled,
              sequenceLength == 1,
              !useKeyEqualsValue else {
            return nil
        }
        guard let fused = resolvedFusedQKV() else { return nil }
        let fingerprint = [
            ObjectIdentifier(fused),
            ObjectIdentifier(qNorm),
            ObjectIdentifier(kNorm),
        ]
        if let segment = compiledQKVSegment {
            if segment.matches(fingerprint) { return segment }
            compiledQKVSegment = nil
            compiledQKVAttempted = false
        }
        if !compiledQKVAttempted {
            compiledQKVAttempted = true
            let numHeads = self.numHeads
            let numKVHeads = self.numKVHeads
            let headDim = self.headDim
            let eps = self.rmsNormEps
            let qNorm = self.qNorm
            let kNorm = self.kNorm
            let function = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
                let x = inputs[0]
                let batch = x.dim(0)
                let sequence = x.dim(1)
                let parts = fused.callSplit(x)
                let queries = qNorm(parts[0].reshaped(batch, sequence, numHeads, headDim))
                    .transposed(0, 2, 1, 3)
                let keys = kNorm(parts[1].reshaped(batch, sequence, numKVHeads, headDim))
                    .transposed(0, 2, 1, 3)
                let values = gemma4RMSNormNoScale(
                    parts[2].reshaped(batch, sequence, numKVHeads, headDim),
                    eps: eps
                ).transposed(0, 2, 1, 3)
                return [queries, keys, values]
            }
            compiledQKVSegment = Gemma4CompiledSegment(function: function, fingerprint: fingerprint)
        }
        return compiledQKVSegment
    }

    private func makeAttentionMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        windowSize: Int?,
        dtype: DType,
        visionBlockIDs: MLXArray?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard queryLength > 1 else {
            return .none
        }

        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queryPositions = MLXArray(Int32(queryOffset)..<Int32(queryOffset + queryLength)).reshaped(queryLength, 1)
        let keyPositions = MLXArray(Int32(keyStart)..<Int32(keyStart + keyLength)).reshaped(1, keyLength)

        var allowed = keyPositions .<= queryPositions
        if let windowSize {
            allowed = allowed .&& (keyPositions .> (queryPositions - Int32(windowSize)))
        }
        if let visionBlockIDs,
           queryOffset == 0,
           keyStart == 0,
           visionBlockIDs.dim(0) == queryLength,
           keyLength == queryLength {
            let queryBlocks = visionBlockIDs.reshaped(queryLength, 1)
            let keyBlocks = visionBlockIDs.reshaped(1, keyLength)
            let sameVisionBlock = (queryBlocks .>= MLXArray(Int32(0))) .&& (queryBlocks .== keyBlocks)
            allowed = (allowed.asType(.int32) + sameVisionBlock.asType(.int32)) .> MLXArray(Int32(0))
        }

        let allowedTyped = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
        let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(allowedTyped .> MLXArray(0).asType(dtype), zeros, negative))
    }
}

final class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "router") var router: Gemma4Router?
    @ModuleInfo(key: "experts") var experts: Gemma4Experts?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    private let hasPerLayerInput: Bool
    private let rmsNormEps: Float
    private var compiledPostSegment: Gemma4CompiledSegment?
    private var compiledPostAttempted = false

    init(config: Gemma4TextConfig, layerIndex: Int, forceKVShared: Bool = false) {
        self._selfAttention.wrappedValue = Gemma4Attention(
            config: config,
            layerIndex: layerIndex,
            forceKVShared: forceKVShared
        )
        self._mlp.wrappedValue = Gemma4MLP(config: config, layerIndex: layerIndex, forceKVShared: forceKVShared)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4Router(config: config)
            self._experts.wrappedValue = Gemma4Experts(config: config)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._router.wrappedValue = nil
            self._experts.wrappedValue = nil
            self._preFeedforwardLayerNorm2.wrappedValue = nil
            self._postFeedforwardLayerNorm1.wrappedValue = nil
            self._postFeedforwardLayerNorm2.wrappedValue = nil
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0
        self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, max(1, config.hiddenSizePerLayerInput), bias: false)
        self._perLayerProjection.wrappedValue = Linear(max(1, config.hiddenSizePerLayerInput), config.hiddenSize, bias: false)
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        self.rmsNormEps = config.rmsNormEps
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        perLayerInput: MLXArray?,
        visionBlockIDs: MLXArray? = nil
    ) -> MLXArray {
        let attentionResidual = x
        var hidden = inputLayerNorm(x)
        hidden = selfAttention(hidden, cache: cache, visionBlockIDs: visionBlockIDs)

        let perLayerInputActive = hasPerLayerInput && perLayerInput != nil
        if let fusedOutput = fusedDecodeFeedForward(
            attentionOutput: hidden,
            attentionResidual: attentionResidual,
            sequenceLength: x.dim(1),
            perLayerInputActive: perLayerInputActive
        ) {
            return fusedOutput
        }
        if let compiled = resolvedCompiledPostSegment(
            sequenceLength: x.dim(1),
            perLayerInputActive: perLayerInputActive
        ) {
            return compiled.function([hidden, attentionResidual])[0]
        }

        hidden = postAttentionLayerNorm(hidden)
        hidden = attentionResidual + hidden

        let mlpResidual = hidden
        if let router, let experts, let preFeedforwardLayerNorm2, let postFeedforwardLayerNorm1, let postFeedforwardLayerNorm2 {
            var dense = preFeedforwardLayerNorm(hidden)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let route = router(hidden)
            var sparse = preFeedforwardLayerNorm2(hidden)
            sparse = experts(sparse, indices: route.indices, weights: route.weights)
            sparse = postFeedforwardLayerNorm2(sparse)

            hidden = dense + sparse
        } else {
            hidden = preFeedforwardLayerNorm(hidden)
            hidden = mlp(hidden)
        }
        hidden = postFeedforwardLayerNorm(hidden)
        hidden = mlpResidual + hidden

        if hasPerLayerInput, let perLayerInput {
            let gateResidual = hidden
            var gate = perLayerInputGate(hidden)
            gate = geluApproximate(gate)
            gate = gate * perLayerInput
            gate = perLayerProjection(gate)
            gate = postPerLayerInputNorm(gate)
            hidden = gateResidual + gate
        }

        return hidden * layerScalar
    }

    /// Preserve final-layer K/V cache semantics while avoiding prompt-row work
    /// whose hidden states are discarded immediately before the language head.
    func callLastPrefillRow(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?
    ) -> MLXArray {
        let sequenceLength = x.dim(1)
        precondition(x.dim(0) == 1 && sequenceLength > 1)
        precondition(!hasPerLayerInput)
        let attentionOutput = selfAttention.callLastPrefillRow(
            inputLayerNorm(x),
            cache: cache
        )

        let attentionResidual = x[0..., (sequenceLength - 1)..., 0...]
        var hidden = postAttentionLayerNorm(attentionOutput)
        hidden = attentionResidual + hidden

        let mlpResidual = hidden
        if let router,
           let experts,
           let preFeedforwardLayerNorm2,
           let postFeedforwardLayerNorm1,
           let postFeedforwardLayerNorm2 {
            var dense = preFeedforwardLayerNorm(hidden)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let route = router(hidden)
            var sparse = preFeedforwardLayerNorm2(hidden)
            sparse = experts(sparse, indices: route.indices, weights: route.weights)
            sparse = postFeedforwardLayerNorm2(sparse)
            hidden = dense + sparse
        } else {
            hidden = preFeedforwardLayerNorm(hidden)
            hidden = mlp(hidden)
        }
        hidden = postFeedforwardLayerNorm(hidden)
        hidden = mlpResidual + hidden
        return hidden * layerScalar
    }

    /// Fused-kernel decode path for everything after attention in the dense
    /// case: (post-attn norm + residual + pre-FFN norm) in one kernel, one
    /// fused gate/up matmul, (gelu·up) in one kernel, the down matmul, then
    /// (post-FFN norm + residual + layer scalar) in one kernel.
    private func fusedDecodeFeedForward(
        attentionOutput: MLXArray,
        attentionResidual: MLXArray,
        sequenceLength: Int,
        perLayerInputActive: Bool
    ) -> MLXArray? {
        guard Gemma4FusedProjectionPolicy.fusedDecodeKernelsEnabled,
              sequenceLength == 1,
              router == nil,
              experts == nil,
              !perLayerInputActive else {
            return nil
        }
        guard let gateUpFused = mlp.resolvedFusedGateUp() else { return nil }

        let hiddenSize = attentionOutput.dim(-1)
        let eps = Gemma4DecodeScalarCache.epsilon(rmsNormEps)
        let (mlpResidual, mlpInput) = Gemma4DecodeFusedKernels.residualDoubleNorm(
            attentionOutput: attentionOutput,
            residual: attentionResidual,
            postNormWeight: postAttentionLayerNorm.weight,
            preNormWeight: preFeedforwardLayerNorm.weight,
            eps: eps,
            hidden: hiddenSize
        )
        let gateUp = gateUpFused.callFused(mlpInput)
        let activated = Gemma4DecodeFusedKernels.geluMul(
            gateUp: gateUp,
            intermediate: gateUp.dim(-1) / 2
        )
        let downOutput = mlp.downProj(activated)
        return Gemma4DecodeFusedKernels.ffnResidualScale(
            downOutput: downOutput,
            mlpResidual: mlpResidual,
            postNormWeight: postFeedforwardLayerNorm.weight,
            layerScalar: layerScalar,
            eps: eps,
            hidden: hiddenSize
        )
    }

    /// Compiled decode segment covering everything after attention in the dense
    /// path: post-attention norm, residual, pre/post feed-forward norms, the MLP,
    /// the second residual, and the layer scalar. MoE and per-layer-input layers
    /// keep the interpreted path.
    private func resolvedCompiledPostSegment(
        sequenceLength: Int,
        perLayerInputActive: Bool
    ) -> Gemma4CompiledSegment? {
        guard Gemma4FusedProjectionPolicy.compiledSegmentsEnabled,
              sequenceLength == 1,
              router == nil,
              experts == nil,
              !perLayerInputActive else {
            return nil
        }
        let fingerprint = [
            ObjectIdentifier(postAttentionLayerNorm),
            ObjectIdentifier(preFeedforwardLayerNorm),
            ObjectIdentifier(postFeedforwardLayerNorm),
            ObjectIdentifier(mlp),
            ObjectIdentifier(mlp.gateProj),
            ObjectIdentifier(mlp.upProj),
            ObjectIdentifier(mlp.downProj),
            ObjectIdentifier(layerScalar),
        ]
        if let segment = compiledPostSegment {
            if segment.matches(fingerprint) { return segment }
            compiledPostSegment = nil
            compiledPostAttempted = false
        }
        if !compiledPostAttempted {
            compiledPostAttempted = true
            let postAttentionLayerNorm = self.postAttentionLayerNorm
            let preFeedforwardLayerNorm = self.preFeedforwardLayerNorm
            let postFeedforwardLayerNorm = self.postFeedforwardLayerNorm
            let mlp = self.mlp
            let layerScalar = self.layerScalar
            let function = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
                let attentionOutput = inputs[0]
                let attentionResidual = inputs[1]
                var hidden = postAttentionLayerNorm(attentionOutput)
                hidden = attentionResidual + hidden
                let mlpResidual = hidden
                hidden = preFeedforwardLayerNorm(hidden)
                hidden = mlp(hidden)
                hidden = postFeedforwardLayerNorm(hidden)
                hidden = mlpResidual + hidden
                return [hidden * layerScalar]
            }
            compiledPostSegment = Gemma4CompiledSegment(function: function, fingerprint: fingerprint)
        }
        return compiledPostSegment
    }
}

final class Gemma4LanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm

    let config: Gemma4TextConfig
    let firstKVSharedLayerIndex: Int
    let layerIndexToCacheIndex: [Int]
    private let embedScale: Float
    private let embedTokensPerLayerScale: Float
    private let perLayerInputScale: Float
    private let perLayerProjectionScale: Float

    init(config: Gemma4TextConfig) {
        self.config = config
        self.embedScale = sqrt(Float(config.hiddenSize))
        self.embedTokensPerLayerScale = sqrt(Float(max(1, config.hiddenSizePerLayerInput)))
        self.perLayerInputScale = pow(2, -0.5)
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
        self.firstKVSharedLayerIndex = max(0, config.numHiddenLayers - config.numKVSharedLayers)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: config.vocabSizePerLayerInput,
            dimensions: max(1, config.numHiddenLayers * max(1, config.hiddenSizePerLayerInput))
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            Gemma4DecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            max(1, config.numHiddenLayers * max(1, config.hiddenSizePerLayerInput)),
            bias: false
        )
        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: max(1, config.hiddenSizePerLayerInput),
            eps: config.rmsNormEps
        )

        var cacheMap: [Int] = Array(0..<firstKVSharedLayerIndex)
        if firstKVSharedLayerIndex < config.numHiddenLayers {
            let concreteLayerTypes = Array(config.layerTypes.prefix(firstKVSharedLayerIndex))
            let sharedFullIndex = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
            let sharedSlidingIndex = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
            for index in firstKVSharedLayerIndex..<config.numHiddenLayers {
                if config.layerTypes[index] == "full_attention" {
                    cacheMap.append(sharedFullIndex)
                } else {
                    cacheMap.append(sharedSlidingIndex)
                }
            }
        }
        self.layerIndexToCacheIndex = cacheMap
        super.init()
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        let hidden = embeddings(inputIds: tokenIds)
        return forward(
            embeddings: hidden,
            inputIds: tokenIds,
            cache: cache,
            mmTokenTypeIds: nil
        )
    }

    func embeddings(inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds) * MLXArray(embedScale).asType(embedTokens.weight.dtype)
    }

    func forward(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        forwardDetailed(
            embeddings: embeddings,
            inputIds: inputIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds,
            captureSharedKV: false
        ).hidden
    }

    func forwardDetailed(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil,
        captureSharedKV: Bool = true
    ) -> Gemma4LanguageModelOutput {
        var hidden = embeddings
        let perLayerInputs = inputIds.flatMap { tokenIds -> MLXArray? in
            guard config.hiddenSizePerLayerInput > 0 else { return nil }
            return projectPerLayerInputs(
                hiddenStates: hidden,
                inputIds: tokenIds
            )
        }
        let visionBlockIDs = Self.visionBlockIDs(from: mmTokenTypeIds, sequenceLength: hidden.dim(1))
        let caches = cache ?? makeCache()
        for (index, layer) in layers.enumerated() {
            let cacheIndex = index < layerIndexToCacheIndex.count ? layerIndexToCacheIndex[index] : index
            let perLayerInput = perLayerInputs?[0..., 0..., index, 0...]
            hidden = layer(
                hidden,
                cache: cacheIndex < caches.count ? caches[cacheIndex] : nil,
                perLayerInput: perLayerInput,
                visionBlockIDs: visionBlockIDs
            )
        }
        let preNormHidden = hidden
        let normalizedHidden = norm(hidden)
        let sharedKVStates = captureSharedKV ? collectSharedKVStates(from: caches) : [:]
        return Gemma4LanguageModelOutput(
            hidden: normalizedHidden,
            preNormHidden: preNormHidden,
            sharedKVStates: sharedKVStates
        )
    }

    func logits(_ inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        let hidden = self(inputIds, cache: cache)
        return embedTokens.asLinear(hidden)
    }

    /// Logits for the final position only. Prefill chunks never read the other
    /// positions' logits, and skipping them avoids a [seq, 262k] lm_head matmul
    /// plus its materialization per chunk.
    func lastPositionLogits(
        _ inputIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil,
        terminalPrefillRowEnabled: Bool? = nil
    ) -> MLXArray {
        let terminalEnabled = terminalPrefillRowEnabled
            ?? Gemma4FusedProjectionPolicy.terminalPrefillRowEnabled
        if terminalEnabled,
           inputIds.dim(0) == 1,
           inputIds.dim(1) > 1,
           config.hiddenSizePerLayerInput == 0,
           config.numKVSharedLayers == 0,
           config.layerTypes.last == "full_attention" {
            var tokenIds = inputIds
            if tokenIds.dtype != .int32 {
                tokenIds = tokenIds.asType(.int32)
            }
            var hidden = embeddings(inputIds: tokenIds)
            let caches = cache ?? makeCache()
            for index in layers.indices.dropLast() {
                let cacheIndex = layerIndexToCacheIndex[index]
                hidden = layers[index](
                    hidden,
                    cache: cacheIndex < caches.count ? caches[cacheIndex] : nil,
                    perLayerInput: nil
                )
            }
            if let lastIndex = layers.indices.last {
                let cacheIndex = layerIndexToCacheIndex[lastIndex]
                let last = layers[lastIndex].callLastPrefillRow(
                    hidden,
                    cache: cacheIndex < caches.count ? caches[cacheIndex] : nil
                )
                return embedTokens.asLinear(norm(last))
            }
        }

        var hidden = self(inputIds, cache: cache)
        let sequenceLength = hidden.dim(1)
        if sequenceLength > 1 {
            hidden = hidden[0..., (sequenceLength - 1)..., 0...]
        }
        return embedTokens.asLinear(hidden)
    }

    func logits(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        let hidden = forward(
            embeddings: embeddings,
            inputIds: inputIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        return embedTokens.asLinear(hidden)
    }

    func detailedLogits(
        embeddings: MLXArray,
        inputIds: MLXArray?,
        cache: [Gemma4AttentionCache]? = nil,
        mmTokenTypeIds: MLXArray? = nil
    ) -> Gemma4ForwardOutput {
        let output = forwardDetailed(
            embeddings: embeddings,
            inputIds: inputIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        return Gemma4ForwardOutput(
            logits: embedTokens.asLinear(output.hidden),
            hidden: output.preNormHidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [Gemma4AttentionCache] {
        config.layerTypes.prefix(firstKVSharedLayerIndex).map { layerType in
            let maxSize: Int? = layerType == "full_attention" ? nil : config.slidingWindow
            if let quantization, quantization.isEnabled {
                if quantization.scheme == .polar {
                    return Gemma4PolarKVCache(configuration: quantization, maxSize: maxSize)
                }
                return Gemma4QuantizedKVCache(configuration: quantization, maxSize: maxSize)
            }
            if let maxSize {
                return Gemma4SlidingKVCache(maxSize: maxSize)
            }
            return Gemma4FullKVCache()
        }
    }

    private func projectPerLayerInputs(
        hiddenStates: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let perLayerEmbedding = embedTokensPerLayer(inputIds)
            * MLXArray(embedTokensPerLayerScale).asType(hiddenStates.dtype)
        let reshapedEmbedding = perLayerEmbedding.reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            config.numHiddenLayers,
            max(1, config.hiddenSizePerLayerInput)
        )

        var projected = (perLayerModelProjection(hiddenStates) * MLXArray(perLayerProjectionScale).asType(hiddenStates.dtype)).reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            config.numHiddenLayers,
            max(1, config.hiddenSizePerLayerInput)
        )
        projected = perLayerProjectionNorm(projected)
        return (projected + reshapedEmbedding) * MLXArray(perLayerInputScale).asType(hiddenStates.dtype)
    }

    private func collectSharedKVStates(from caches: [Gemma4AttentionCache]) -> [String: Gemma4SharedKVState] {
        guard !caches.isEmpty else { return [:] }
        var states: [String: Gemma4SharedKVState] = [:]
        for index in 0..<min(layers.count, layerIndexToCacheIndex.count) {
            let cacheIndex = layerIndexToCacheIndex[index]
            guard cacheIndex < caches.count,
                  let state = caches[cacheIndex].currentState() else {
                continue
            }
            let maxSize = (caches[cacheIndex] as? Gemma4SlidingKVCache)?.configuredMaxSize
            states[layers[index].selfAttention.layerType] = Gemma4SharedKVState(
                keys: state.0,
                values: state.1,
                offset: caches[cacheIndex].offset,
                maxSize: maxSize
            )
        }
        return states
    }

    private static func visionBlockIDs(from mmTokenTypeIds: MLXArray?, sequenceLength: Int) -> MLXArray? {
        guard let mmTokenTypeIds, sequenceLength > 1 else { return nil }
        let typed = mmTokenTypeIds.asType(.int32)
        MLX.eval(typed)
        let values = typed.asArray(Int32.self)
        guard values.count >= sequenceLength else { return nil }

        var blockIDs = [Int32](repeating: -1, count: sequenceLength)
        var currentBlock: Int32 = -1
        var previousWasVision = false
        for index in 0..<sequenceLength {
            let value = values[index]
            let isVision = value == 1 || value == 2
            if isVision && !previousWasVision {
                currentBlock += 1
            }
            if isVision {
                blockIDs[index] = currentBlock
            }
            previousWasVision = isVision
        }
        return MLXArray(blockIDs)
    }
}

public final class Gemma4TextCausalLM: Module, Gemma4CausalModel, @unchecked Sendable {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel

    public let config: Gemma4TextConfig
    private let finalLogitSoftcapping: Float?

    public init(config: Gemma4TextConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._languageModel.wrappedValue = Gemma4LanguageModel(config: config)
        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [AnyObject]? = nil
    ) -> MLXArray {
        forward(inputIds: inputIds, cache: cache as? [Gemma4AttentionCache])
    }

    /// Logits for an explicit set of flattened (`[batch * seq]` row-major)
    /// positions only. SFT training reads logits solely at loss-masked target
    /// positions, and prompt/pad rows are the majority of a chat batch —
    /// projecting them through the 262k-vocab lm_head (plus the float32 loss
    /// chain) is pure waste. Hidden states still flow through every position,
    /// so gradients match the full-logits path exactly.
    func trainingLogits(inputIds: MLXArray, flatTargetPositions: MLXArray) -> MLXArray {
        let hidden = languageModel(inputIds)
        let flattened = hidden.reshaped([-1, hidden.dim(-1)])
        let selected = take(flattened, flatTargetPositions.asType(.int32), axis: 0)
        return applyFinalSoftcap(languageModel.embedTokens.asLinear(selected))
    }

    func forward(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        var logits = languageModel.logits(inputIds, cache: cache)
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        applyFinalSoftcap(languageModel.lastPositionLogits(inputIds, cache: cache))
    }

    func forwardForSpeculation(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> Gemma4ForwardOutput {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let output = languageModel.detailedLogits(
            embeddings: languageModel.embeddings(inputIds: tokenIds),
            inputIds: tokenIds,
            cache: cache
        )
        let logits = applyFinalSoftcap(output.logits)
        return Gemma4ForwardOutput(
            logits: logits,
            hidden: output.hidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    func inputEmbeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return languageModel.embeddings(inputIds: tokenIds)
    }

    func speculativeLogits(fromHidden hidden: MLXArray) -> MLXArray {
        applyFinalSoftcap(languageModel.embedTokens.asLinear(languageModel.norm(hidden)))
    }

    func speculativeDraftHidden(_ hidden: MLXArray) -> MLXArray {
        languageModel.norm(hidden)
    }

    func makeAttentionCache(quantization: Gemma4KVCacheQuantization? = nil) -> [Gemma4AttentionCache] {
        languageModel.makeCache(quantization: quantization)
    }

    public func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [AnyObject] {
        makeAttentionCache(quantization: quantization)
    }

    private func applyFinalSoftcap(_ logits: MLXArray) -> MLXArray {
        guard let finalLogitSoftcapping else { return logits }
        let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
        return tanh(logits / softcap) * softcap
    }
}

final class Gemma4UnifiedVisionEmbedder: Module {
    @ModuleInfo(key: "patch_ln1") var patchLN1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLN2: LayerNorm
    @ParameterInfo(key: "pos_embedding") var positionEmbedding: MLXArray
    @ModuleInfo(key: "pos_norm") var positionNorm: LayerNorm

    private let patchDim: Int

    init(config: Gemma4UnifiedVisionConfig) {
        self.patchDim = config.modelPatchSize * config.modelPatchSize * 3
        self._patchLN1.wrappedValue = LayerNorm(dimensions: patchDim, eps: config.rmsNormEps)
        self._patchDense.wrappedValue = Linear(patchDim, config.mmEmbedDim, bias: true)
        self._patchLN2.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim, eps: config.rmsNormEps)
        self._positionEmbedding.wrappedValue = MLXArray.zeros([config.mmPosembSize, 2, config.mmEmbedDim])
        self._positionNorm.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray?
    ) -> MLXArray {
        var hidden = pixelValues
        if hidden.ndim == 4 && hidden.dim(-1) == patchDim {
            hidden = hidden.reshaped(hidden.dim(0), -1, patchDim)
        }
        hidden = patchLN1(hidden)
        hidden = patchDense(hidden)
        hidden = patchLN2(hidden)

        if let imagePositionIds {
            let xIds = MLX.maximum(imagePositionIds[0..., 0..., 0], MLXArray(Int32(0))).asType(.int32)
            let yIds = MLX.maximum(imagePositionIds[0..., 0..., 1], MLXArray(Int32(0))).asType(.int32)
            let xValid = (imagePositionIds[0..., 0..., 0] .>= MLXArray(Int32(0)))
                .asType(hidden.dtype)
                .expandedDimensions(axis: -1)
            let yValid = (imagePositionIds[0..., 0..., 1] .>= MLXArray(Int32(0)))
                .asType(hidden.dtype)
                .expandedDimensions(axis: -1)
            let xTable = positionEmbedding[0..., 0, 0...]
            let yTable = positionEmbedding[0..., 1, 0...]
            let xPos = take(xTable, xIds, axis: 0)
            let yPos = take(yTable, yIds, axis: 0)
            hidden = hidden + (xPos * xValid + yPos * yValid).asType(hidden.dtype)
        }

        return positionNorm(hidden)
    }
}

final class Gemma4UnifiedMultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    private let eps: Float
    private let normWeight: MLXArray

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self.eps = eps
        self.normWeight = MLXArray.ones([embeddingDim])
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputsEmbeds: MLXArray) -> MLXArray {
        embeddingProjection(gemma4RMSNormNoScale(inputsEmbeds, weight: normWeight, eps: eps))
    }
}

public final class Gemma4UnifiedCausalLM: Module, Gemma4CausalModel, @unchecked Sendable {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "vision_embedder") var visionEmbedder: Gemma4UnifiedVisionEmbedder
    @ModuleInfo(key: "embed_vision") var embedVision: Gemma4UnifiedMultimodalEmbedder

    public let config: Gemma4Config
    private let finalLogitSoftcapping: Float?
    private let imageTokenId: Int

    public init(config: Gemma4Config) throws {
        guard let visionConfig = config.visionConfig else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified runtime requires vision_config.")
        }
        guard let imageTokenId = config.imageTokenId else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified runtime requires image_token_id.")
        }
        self.config = config
        self.finalLogitSoftcapping = config.textConfig.finalLogitSoftcapping
        self.imageTokenId = imageTokenId
        self._languageModel.wrappedValue = Gemma4LanguageModel(config: config.textConfig)
        self._visionEmbedder.wrappedValue = Gemma4UnifiedVisionEmbedder(config: visionConfig)
        self._embedVision.wrappedValue = Gemma4UnifiedMultimodalEmbedder(
            embeddingDim: visionConfig.outputProjDims,
            textHiddenSize: config.textConfig.hiddenSize,
            eps: visionConfig.rmsNormEps
        )
        super.init()
    }

    func forward(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        var logits = languageModel.logits(inputIds, cache: cache)
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        applyFinalSoftcap(languageModel.lastPositionLogits(
            inputIds,
            cache: cache,
            terminalPrefillRowEnabled: false
        ))
    }

    func forwardForSpeculation(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> Gemma4ForwardOutput {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let output = languageModel.detailedLogits(
            embeddings: languageModel.embeddings(inputIds: tokenIds),
            inputIds: tokenIds,
            cache: cache
        )
        let logits = applyFinalSoftcap(output.logits)
        return Gemma4ForwardOutput(
            logits: logits,
            hidden: output.hidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    func forward(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        mmTokenTypeIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) throws -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        var embeddings = languageModel.embeddings(inputIds: tokenIds)
        let imageFeatures = try compactImageFeatures(
            pixelValues: pixelValues,
            imagePositionIds: imagePositionIds,
            dtype: embeddings.dtype
        )
        embeddings = try replaceImageEmbeddings(
            embeddings,
            inputIds: tokenIds,
            imageFeatures: imageFeatures
        )
        var logits = languageModel.logits(
            embeddings: embeddings,
            inputIds: tokenIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    func forwardForSpeculation(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        mmTokenTypeIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) throws -> Gemma4ForwardOutput {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        var embeddings = languageModel.embeddings(inputIds: tokenIds)
        let imageFeatures = try compactImageFeatures(
            pixelValues: pixelValues,
            imagePositionIds: imagePositionIds,
            dtype: embeddings.dtype
        )
        embeddings = try replaceImageEmbeddings(
            embeddings,
            inputIds: tokenIds,
            imageFeatures: imageFeatures
        )
        let output = languageModel.detailedLogits(
            embeddings: embeddings,
            inputIds: tokenIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        return Gemma4ForwardOutput(
            logits: applyFinalSoftcap(output.logits),
            hidden: output.hidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    func inputEmbeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return languageModel.embeddings(inputIds: tokenIds)
    }

    func speculativeLogits(fromHidden hidden: MLXArray) -> MLXArray {
        applyFinalSoftcap(languageModel.embedTokens.asLinear(languageModel.norm(hidden)))
    }

    func speculativeDraftHidden(_ hidden: MLXArray) -> MLXArray {
        languageModel.norm(hidden)
    }

    func makeAttentionCache(quantization: Gemma4KVCacheQuantization? = nil) -> [Gemma4AttentionCache] {
        languageModel.makeCache(quantization: quantization)
    }

    public func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [AnyObject] {
        makeAttentionCache(quantization: quantization)
    }

    private func applyFinalSoftcap(_ logits: MLXArray) -> MLXArray {
        guard let finalLogitSoftcapping else { return logits }
        let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
        return tanh(logits / softcap) * softcap
    }

    private func compactImageFeatures(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        dtype: DType
    ) throws -> MLXArray {
        let embedded = visionEmbedder(pixelValues: pixelValues, imagePositionIds: imagePositionIds)
        let projected = embedVision(embedded).asType(dtype)
        let typedPositions = imagePositionIds.asType(.int32)
        MLX.eval(typedPositions)
        let positions = typedPositions.asArray(Int32.self)
        let batch = projected.dim(0)
        let tokenCount = projected.dim(1)
        let hiddenSize = projected.dim(2)
        guard positions.count >= batch * tokenCount * 2 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified image positions have an invalid shape.")
        }

        var rows: [MLXArray] = []
        rows.reserveCapacity(batch * tokenCount)
        for batchIndex in 0..<batch {
            for tokenIndex in 0..<tokenCount {
                let offset = ((batchIndex * tokenCount) + tokenIndex) * 2
                guard positions[offset] >= 0, positions[offset + 1] >= 0 else {
                    continue
                }
                rows.append(projected[batchIndex, tokenIndex, 0...].reshaped(1, hiddenSize))
            }
        }
        guard !rows.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified image preprocessing produced no visual tokens.")
        }
        return concatenated(rows, axis: 0)
    }

    private func replaceImageEmbeddings(
        _ embeddings: MLXArray,
        inputIds: MLXArray,
        imageFeatures: MLXArray
    ) throws -> MLXArray {
        let tokenIds = inputIds.asType(.int32)
        MLX.eval(tokenIds)
        let values = tokenIds.asArray(Int32.self)
        let sequenceLength = embeddings.dim(1)
        let imagePositions = values.prefix(sequenceLength).enumerated().compactMap { index, value in
            value == Int32(imageTokenId) ? index : nil
        }
        guard imagePositions.count == imageFeatures.dim(0) else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 unified prompt has \(imagePositions.count) image tokens but image preprocessing produced \(imageFeatures.dim(0)) visual features."
            )
        }

        let merged = embeddings
        for (featureIndex, position) in imagePositions.enumerated() {
            merged[0, position, 0...] = imageFeatures[featureIndex, 0...]
        }
        return merged
    }
}
