import Foundation
import MLX

func mapLTX25DurationHeadWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("duration_head.") else { return [] }
    let mapped = String(key.dropFirst("duration_head.".count))
    let casted = value.dtype.isFloatingPoint && value.dtype != dtype
        ? value.asType(dtype)
        : value
    return [(mapped, casted)]
}
