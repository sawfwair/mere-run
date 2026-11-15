import MLX
import MLXFast

/// One forward call can reuse masks across layers with identical positions.
/// Padding and vision masks stay on the uncached path.
final class Gemma4AttentionMaskCache {
    struct Key: Hashable {
        let queryLength: Int
        let queryOffset: Int
        let keyLength: Int
        let window: Int?
        let dtype: DType
    }

    private var masks: [Key: MLXFast.ScaledDotProductAttentionMaskMode] = [:]
    private var offsets: [Int: MLXArray] = [:]

    func positionOffset(_ offset: Int) -> MLXArray {
        if let value = offsets[offset] { return value }
        let value = MLXArray([Int32(offset)])
        offsets[offset] = value
        return value
    }

    func resolve(_ key: Key, build: () -> MLXFast.ScaledDotProductAttentionMaskMode) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let mask = masks[key] { return mask }
        let mask = build()
        masks[key] = mask
        return mask
    }
}
