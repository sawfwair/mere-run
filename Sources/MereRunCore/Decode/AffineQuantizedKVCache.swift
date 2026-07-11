import MLX
import MLXFast

/// An explicitly selected, memory-oriented KV cache for model families that
/// do not have a model-specific packed-attention kernel.
///
/// Keys and values remain quantized while resident and are dequantized only
/// for the attention operation. This materially reduces persistent KV and
/// prefix-cache storage, but the dequantization work can reduce decode
/// throughput. It is therefore never selected implicitly by this type; the
/// caller must opt into `RuntimeKVCacheMode.affine8`.
public final class AffineQuantizedKVCache: KVCache {
    public static let defaultGroupSize = 64
    public static let defaultBits = 8

    private final class PackedStorage {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray?
        let sourceDType: DType
        let groupSize: Int
        let bits: Int
        var tokenCount: Int
        var tokenCapacity: Int

        init(
            weight: MLXArray,
            scales: MLXArray,
            biases: MLXArray?,
            sourceDType: DType,
            groupSize: Int,
            bits: Int,
            tokenCount: Int,
            tokenCapacity: Int
        ) {
            self.weight = weight
            self.scales = scales
            self.biases = biases
            self.sourceDType = sourceDType
            self.groupSize = groupSize
            self.bits = bits
            self.tokenCount = tokenCount
            self.tokenCapacity = tokenCapacity
        }

        convenience init(source: MLXArray, groupSize: Int, bits: Int, step: Int) {
            let quantized = MLX.quantized(
                source,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
            let count = source.dim(2)
            let capacity = Self.roundedCapacity(count, step: step)
            self.init(
                weight: Self.padded(quantized.wq, capacity: capacity),
                scales: Self.padded(quantized.scales, capacity: capacity),
                biases: quantized.biases.map { Self.padded($0, capacity: capacity) },
                sourceDType: source.dtype,
                groupSize: groupSize,
                bits: bits,
                tokenCount: count,
                tokenCapacity: capacity
            )
        }

        func append(_ source: MLXArray, step: Int) {
            precondition(source.dtype == sourceDType, "Affine KV cache dtype changed during generation.")
            let quantized = MLX.quantized(
                source,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
            let newCount = tokenCount + source.dim(2)
            ensureCapacity(newCount, step: step)

            weight[0..., 0..., tokenCount..<newCount, 0...] = quantized.wq
            scales[0..., 0..., tokenCount..<newCount, 0...] = quantized.scales
            if let nextBiases = quantized.biases {
                biases?[0..., 0..., tokenCount..<newCount, 0...] = nextBiases
            }
            tokenCount = newCount
        }

        func dequantized() -> MLXArray {
            MLX.dequantized(
                weight[0..., 0..., 0..<tokenCount, 0...],
                scales: scales[0..., 0..., 0..<tokenCount, 0...],
                biases: biases.map { $0[0..., 0..., 0..<tokenCount, 0...] },
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                dtype: sourceDType
            )
        }

        func fork() -> PackedStorage {
            PackedStorage(
                // New wrappers are required because subscript assignment
                // rebinds an MLXArray wrapper in place.
                weight: weight.asType(weight.dtype),
                scales: scales.asType(scales.dtype),
                biases: biases.map { $0.asType($0.dtype) },
                sourceDType: sourceDType,
                groupSize: groupSize,
                bits: bits,
                tokenCount: tokenCount,
                tokenCapacity: tokenCapacity
            )
        }

        func row(_ index: Int) -> PackedStorage {
            PackedStorage(
                weight: weight[index..<(index + 1), 0..., 0..., 0...],
                scales: scales[index..<(index + 1), 0..., 0..., 0...],
                biases: biases.map { $0[index..<(index + 1), 0..., 0..., 0...] },
                sourceDType: sourceDType,
                groupSize: groupSize,
                bits: bits,
                tokenCount: tokenCount,
                tokenCapacity: tokenCapacity
            )
        }

        static func batched(_ storages: [PackedStorage]) -> PackedStorage? {
            guard let first = storages.first,
                  storages.allSatisfy({
                      $0.groupSize == first.groupSize
                          && $0.bits == first.bits
                          && $0.sourceDType == first.sourceDType
                          && $0.tokenCount == first.tokenCount
                          && $0.tokenCapacity == first.tokenCapacity
                          && ($0.biases == nil) == (first.biases == nil)
                  }) else {
                return nil
            }
            let biases = first.biases == nil
                ? nil
                : concatenated(storages.compactMap(\.biases), axis: 0)
            return PackedStorage(
                weight: concatenated(storages.map(\.weight), axis: 0),
                scales: concatenated(storages.map(\.scales), axis: 0),
                biases: biases,
                sourceDType: first.sourceDType,
                groupSize: first.groupSize,
                bits: first.bits,
                tokenCount: first.tokenCount,
                tokenCapacity: first.tokenCapacity
            )
        }

        var storageBytes: Int {
            weight.size * weight.itemSize
                + scales.size * scales.itemSize
                + (biases.map { $0.size * $0.itemSize } ?? 0)
        }

        private func ensureCapacity(_ requested: Int, step: Int) {
            guard requested > tokenCapacity else { return }
            let newCapacity = Self.roundedCapacity(requested, step: step)
            weight = Self.extended(weight, from: tokenCapacity, to: newCapacity)
            scales = Self.extended(scales, from: tokenCapacity, to: newCapacity)
            biases = biases.map { Self.extended($0, from: tokenCapacity, to: newCapacity) }
            tokenCapacity = newCapacity
        }

        private static func roundedCapacity(_ count: Int, step: Int) -> Int {
            max(step, ((count + step - 1) / step) * step)
        }

        private static func padded(_ source: MLXArray, capacity: Int) -> MLXArray {
            extended(source, from: source.dim(2), to: capacity)
        }

        private static func extended(_ source: MLXArray, from count: Int, to capacity: Int) -> MLXArray {
            guard capacity > count else { return source }
            var paddingShape = source.shape
            paddingShape[2] = capacity - count
            return concatenated(
                [source, MLXArray.zeros(paddingShape, dtype: source.dtype)],
                axis: 2
            )
        }
    }

    private let groupSize: Int
    private let bits: Int
    private let step: Int
    private var keys: PackedStorage?
    private var values: PackedStorage?

    public private(set) var offset = 0

    public init(
        groupSize: Int = AffineQuantizedKVCache.defaultGroupSize,
        bits: Int = AffineQuantizedKVCache.defaultBits,
        step: Int = 256
    ) {
        precondition(groupSize > 0, "Affine KV group size must be positive.")
        precondition((2...8).contains(bits), "Affine KV bit width must be between 2 and 8.")
        precondition(step > 0, "Affine KV allocation step must be positive.")
        self.groupSize = groupSize
        self.bits = bits
        self.step = step
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys.ndim == 4 && values.ndim == 4, "Affine KV cache expects [B,H,T,D] tensors.")
        precondition(keys.dim(3) % groupSize == 0, "Key head width must divide the affine KV group size.")
        precondition(values.dim(3) % groupSize == 0, "Value head width must divide the affine KV group size.")
        precondition(keys.dim(2) == values.dim(2), "Key/value token counts must match.")

        if let storedKeys = self.keys, let storedValues = self.values {
            storedKeys.append(keys, step: step)
            storedValues.append(values, step: step)
        } else {
            self.keys = PackedStorage(source: keys, groupSize: groupSize, bits: bits, step: step)
            self.values = PackedStorage(source: values, groupSize: groupSize, bits: bits, step: step)
        }
        offset += keys.dim(2)
        return (self.keys!.dequantized(), self.values!.dequantized())
    }

    func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        return (keys.dequantized(), values.dequantized())
    }

    public func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n == 1 ? .none : .causal
    }

    public func fork() -> KVCache {
        let copy = AffineQuantizedKVCache(groupSize: groupSize, bits: bits, step: step)
        copy.keys = keys?.fork()
        copy.values = values?.fork()
        copy.offset = offset
        return copy
    }

    public func batched(with caches: [KVCache]) -> KVCache? {
        guard let typed = caches as? [AffineQuantizedKVCache],
              !typed.isEmpty,
              typed.allSatisfy({
                  $0.groupSize == groupSize
                      && $0.bits == bits
                      && $0.step == step
                      && $0.offset == offset
              }),
              let keyStorage = PackedStorage.batched(typed.compactMap(\.keys)),
              let valueStorage = PackedStorage.batched(typed.compactMap(\.values)),
              typed.allSatisfy({ $0.keys != nil && $0.values != nil }) else {
            return nil
        }
        let copy = AffineQuantizedKVCache(groupSize: groupSize, bits: bits, step: step)
        copy.keys = keyStorage
        copy.values = valueStorage
        copy.offset = offset
        return copy
    }

    public func unbatchedRows(count: Int) -> [KVCache]? {
        guard count > 0,
              let keys,
              let values,
              keys.weight.dim(0) == count,
              values.weight.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = AffineQuantizedKVCache(groupSize: groupSize, bits: bits, step: step)
            copy.keys = keys.row(index)
            copy.values = values.row(index)
            copy.offset = offset
            return copy
        }
    }

    /// Materialized bytes held by packed K/V buffers, including allocation
    /// headroom. Exposed for structural benchmarks and runtime diagnostics.
    public var storageBytes: Int {
        (keys?.storageBytes ?? 0) + (values?.storageBytes ?? 0)
    }
}
