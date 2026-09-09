import Foundation
import MLX
import MLXFast

final class Gemma4QuantizedKVCache: Gemma4AttentionCache {
    static let decodeChunkSize = 2_048

    let configuration: Gemma4KVCacheQuantization
    let maxSize: Int?

    var leadingKeys: MLXArray?
    var leadingValues: MLXArray?
    var quantizedKeys: Gemma4QuantizedTensorState?
    var quantizedValues: Gemma4QuantizedTensorState?

    var offset: Int = 0

    init(configuration: Gemma4KVCacheQuantization, maxSize: Int?) {
        self.configuration = configuration
        self.maxSize = maxSize
    }

    func currentState() -> (MLXArray, MLXArray)? {
        reconstructState()
    }

    func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)? {
        guard keys.dim(2) > 1, let maxSize else {
            append(keys: keys, values: values)
            return currentState()
        }
        let previous = currentState()
        let combinedKeys = previous.map { concatenated([$0.0, keys], axis: 2) } ?? keys
        let combinedValues = previous.map { concatenated([$0.1, values], axis: 2) } ?? values
        guard combinedKeys.dim(2) > maxSize else {
            append(keys: keys, values: values)
            return currentState()
        }

        // The chunk attends over the complete quantized context while resident
        // storage advances to the window needed by the next forward call.
        let context = Self.reencoded(
            keys: combinedKeys, values: combinedValues, configuration: configuration,
            maxSize: nil, offset: offset + keys.dim(2)
        )
        append(keys: keys, values: values)
        return context.currentState()
    }

    func append(keys: MLXArray, values: MLXArray) {
        guard maxSize != nil else {
            appendUnbounded(keys: keys, values: values)
            return
        }

        let existing = reconstructState()

        let combinedKeys: MLXArray
        let combinedValues: MLXArray
        if let existing {
            combinedKeys = concatenated([existing.0, keys], axis: 2)
            combinedValues = concatenated([existing.1, values], axis: 2)
        } else {
            combinedKeys = keys
            combinedValues = values
        }

        let newOffset = offset + keys.dim(2)
        var keptKeys = combinedKeys
        var keptValues = combinedValues
        if let maxSize {
            let totalLength = combinedKeys.dim(2)
            if totalLength > maxSize {
                let start = totalLength - maxSize
                keptKeys = combinedKeys[0..., 0..., start..., 0...]
                keptValues = combinedValues[0..., 0..., start..., 0...]
            }
        }

        repartition(keys: keptKeys, values: keptValues, newOffset: newOffset)
    }

    func appendUnbounded(keys: MLXArray, values: MLXArray) {
        let tokenCount = keys.dim(2)
        let newOffset = offset + tokenCount
        defer { offset = newOffset }

        guard let keyBits = configuration.keyBits,
              let valueBits = configuration.valueBits else {
            appendLeading(keys: keys, values: values)
            return
        }

        let plainCount = max(0, min(tokenCount, configuration.quantizedStart - offset))
        if plainCount > 0 {
            appendLeading(
                keys: keys[0..., 0..., ..<plainCount, 0...],
                values: values[0..., 0..., ..<plainCount, 0...]
            )
        }

        guard plainCount < tokenCount else {
            return
        }

        appendQuantized(
            keys: keys[0..., 0..., plainCount..., 0...],
            values: values[0..., 0..., plainCount..., 0...],
            keyBits: keyBits,
            valueBits: valueBits
        )
    }

    func appendLeading(keys: MLXArray, values: MLXArray) {
        if let existingKeys = leadingKeys, let existingValues = leadingValues {
            leadingKeys = concatenated([existingKeys, keys], axis: 2)
            leadingValues = concatenated([existingValues, values], axis: 2)
        } else {
            leadingKeys = keys
            leadingValues = values
        }
    }

    func appendQuantized(keys: MLXArray, values: MLXArray, keyBits: Int, valueBits: Int) {
        if let quantizedKeys {
            self.quantizedKeys = quantizedKeys.appending(keys)
        } else {
            quantizedKeys = Gemma4QuantizedTensorState(
                source: keys,
                groupSize: configuration.groupSize,
                bits: keyBits
            )
        }

        if let quantizedValues {
            self.quantizedValues = quantizedValues.appending(values)
        } else {
            quantizedValues = Gemma4QuantizedTensorState(
                source: values,
                groupSize: configuration.groupSize,
                bits: valueBits
            )
        }
    }

    func fork() -> Gemma4AttentionCache {
        let copy = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
        copy.leadingKeys = leadingKeys
        copy.leadingValues = leadingValues
        copy.quantizedKeys = quantizedKeys
        copy.quantizedValues = quantizedValues
        copy.offset = offset
        return copy
    }

    static func reencoded(
        keys: MLXArray,
        values: MLXArray,
        configuration: Gemma4KVCacheQuantization,
        maxSize: Int?,
        offset: Int
    ) -> Gemma4QuantizedKVCache {
        let cache = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
        cache.repartition(keys: keys, values: values, newOffset: offset)
        return cache
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        if quantization == configuration {
            return fork()
        }
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
        if let leadingKeys, let leadingValues {
            MLX.eval(leadingKeys, leadingValues)
        }
        if let quantizedKeys {
            if let biases = quantizedKeys.biases {
                MLX.eval(quantizedKeys.weight, quantizedKeys.scales, biases)
            } else {
                MLX.eval(quantizedKeys.weight, quantizedKeys.scales)
            }
        }
        if let quantizedValues {
            if let biases = quantizedValues.biases {
                MLX.eval(quantizedValues.weight, quantizedValues.scales, biases)
            } else {
                MLX.eval(quantizedValues.weight, quantizedValues.scales)
            }
        }
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4QuantizedKVCache],
              !typed.isEmpty,
              typed.allSatisfy({
                  $0.offset == offset
                      && $0.configuration == configuration
                      && $0.maxSize == maxSize
              }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
        copy.repartition(
            keys: concatenated(states.map(\.0), axis: 0),
            values: concatenated(states.map(\.1), axis: 0),
            newOffset: offset
        )
        return copy
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0,
              let state = currentState(),
              state.0.dim(0) == count,
              state.1.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4QuantizedKVCache(configuration: configuration, maxSize: maxSize)
            copy.repartition(
                keys: state.0[index..<(index + 1), 0..., 0..., 0...],
                values: state.1[index..<(index + 1), 0..., 0..., 0...],
                newOffset: offset
            )
            return copy
        }
    }

    func repartition(keys: MLXArray, values: MLXArray, newOffset: Int) {
        let keptLength = keys.dim(2)
        let keptStart = newOffset - keptLength
        let plainCount = max(0, min(keptLength, configuration.quantizedStart - keptStart))

        if plainCount > 0 {
            leadingKeys = keys[0..., 0..., ..<plainCount, 0...]
            leadingValues = values[0..., 0..., ..<plainCount, 0...]
        } else {
            leadingKeys = nil
            leadingValues = nil
        }

        let quantizedLength = keptLength - plainCount
        if quantizedLength > 0,
           let keyBits = configuration.keyBits,
           let valueBits = configuration.valueBits {
            quantizedKeys = Gemma4QuantizedTensorState(
                source: keys[0..., 0..., plainCount..., 0...],
                groupSize: configuration.groupSize,
                bits: keyBits
            )
            quantizedValues = Gemma4QuantizedTensorState(
                source: values[0..., 0..., plainCount..., 0...],
                groupSize: configuration.groupSize,
                bits: valueBits
            )
        } else {
            quantizedKeys = nil
            quantizedValues = nil
        }

        offset = newOffset
    }

    func reconstructState() -> (MLXArray, MLXArray)? {
        var keyParts: [MLXArray] = []
        var valueParts: [MLXArray] = []

        if let leadingKeys, let leadingValues {
            keyParts.append(leadingKeys)
            valueParts.append(leadingValues)
        }

        if let quantizedKeys, let quantizedValues {
            keyParts.append(quantizedKeys.dequantized())
            valueParts.append(quantizedValues.dequantized())
        }

        guard !keyParts.isEmpty else {
            return nil
        }

        if keyParts.count == 1 {
            return (keyParts[0], valueParts[0])
        }

        return (concatenated(keyParts, axis: 2), concatenated(valueParts, axis: 2))
    }

    static var supportsFastKernels: Bool {
        Device.defaultDevice().deviceType == .gpu
    }
}
