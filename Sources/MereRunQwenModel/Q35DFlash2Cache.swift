import MLX

/// Only confirmed target context is stored. Draft attention reads this cache
/// without appending proposals, so a rejected block cannot evict valid history.
package final class Q35DFlash2Cache {
    package private(set) var keys: MLXArray?
    package private(set) var values: MLXArray?
    package private(set) var offset: Int
    package let window: Int

    package init(window: Int, offset: Int = 0) {
        self.window = window
        self.offset = offset
    }

    package func append(keys: MLXArray, values: MLXArray) {
        let allKeys = self.keys.map { concatenated([$0, keys], axis: 2) } ?? keys
        let allValues = self.values.map { concatenated([$0, values], axis: 2) } ?? values
        let start = max(0, allKeys.dim(2) - window)
        // A compact copy bounds storage instead of retaining a large prefill view.
        self.keys = contiguous(allKeys[0..., 0..., start..., 0...])
        self.values = contiguous(allValues[0..., 0..., start..., 0...])
        offset += keys.dim(2)
    }

    package var state: [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    package func fork() -> Q35DFlash2Cache {
        let copy = Q35DFlash2Cache(window: window, offset: offset)
        copy.keys = keys.map { $0.reshaped($0.shape) }
        copy.values = values.map { $0.reshaped($0.shape) }
        return copy
    }
}
