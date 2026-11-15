import MLX
import MereRunGemmaModel

/// SOV-only speculative storage. Trials write beyond the committed prefix;
/// rollback changes the valid length without restoring or copying tensor rows.
package final class SOVRollbackKVCache: Gemma4AttentionCache {
    private let window: Int?
    private var keys: MLXArray?
    private var values: MLXArray?
    private var baseOffset = 0
    private var stateCount = 0
    private var checkpoint: (offset: Int, count: Int)?
    package private(set) var offset = 0

    package init(window: Int?) { self.window = window }

    package func beginTrial(count: Int) {
        precondition(checkpoint == nil && count > 0)
        if let keys, let values { reserve(count, keySample: keys, valueSample: values) }
        checkpoint = (offset, stateCount)
    }

    package func commit(count: Int) {
        let saved = checkpoint!
        precondition(count >= 0 && saved.offset + count <= offset)
        offset = saved.offset + count
        stateCount = count == 0 ? saved.count : (window.map { min(saved.count + count, max($0, count)) } ?? offset)
        checkpoint = nil
    }

    package func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        let end = offset - baseOffset
        return (keys[0..., 0..., (end - stateCount)..<end, 0...],
                values[0..., 0..., (end - stateCount)..<end, 0...])
    }

    package func append(keys: MLXArray, values: MLXArray) {
        reserve(keys.dim(2), keySample: keys, valueSample: values)
        let start = offset - baseOffset, end = start + keys.dim(2)
        self.keys![0..., 0..., start..<end, 0...] = keys
        self.values![0..., 0..., start..<end, 0...] = values
        offset += keys.dim(2)
        stateCount = window.map { min(stateCount + keys.dim(2), max($0, keys.dim(2))) } ?? offset
    }

    package func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)? {
        let previousCount = stateCount
        append(keys: keys, values: values)
        let end = offset - baseOffset
        let start = keys.dim(2) == 1 ? end - stateCount : end - previousCount - keys.dim(2)
        return (self.keys![0..., 0..., start..<end, 0...], self.values![0..., 0..., start..<end, 0...])
    }

    private func reserve(_ count: Int, keySample: MLXArray, valueSample: MLXArray) {
        if let keys, offset - baseOffset + count <= keys.dim(2) { return }
        // Compaction happens before a trial starts. Every pre-trial row remains
        // available until its accepted length is known, including tiny windows.
        precondition(checkpoint == nil)
        let retained = currentState()
        let capacity = max(window.map { $0 + 256 } ?? 0, ((stateCount + count + 255) / 256) * 256)
        func storage(_ sample: MLXArray, previous: MLXArray?) -> MLXArray {
            let padding = MLXArray.zeros([sample.dim(0), sample.dim(1), capacity - stateCount, sample.dim(3)], dtype: sample.dtype)
            return previous.map { concatenated([$0, padding], axis: 2) } ?? padding
        }
        keys = storage(keySample, previous: retained?.0)
        values = storage(valueSample, previous: retained?.1)
        baseOffset = offset - stateCount
    }

    package func fork() -> Gemma4AttentionCache {
        precondition(checkpoint == nil)
        let copy = SOVRollbackKVCache(window: window)
        copy.keys = keys.map { $0.reshaped($0.shape) }; copy.values = values.map { $0.reshaped($0.shape) }
        copy.baseOffset = baseOffset; copy.offset = offset; copy.stateCount = stateCount
        return copy
    }

    package func storageArraysForEvaluation() -> [MLXArray] { [keys, values].compactMap { $0 } }
    package func evaluateStorage() { eval(storageArraysForEvaluation()) }
}
