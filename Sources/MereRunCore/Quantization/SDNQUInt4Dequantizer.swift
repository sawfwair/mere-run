import MLX

enum SDNQUInt4Dequantizer {
    static func unpack(_ weight: MLXArray, quantizedShape: [Int]) -> MLXArray {
        let packed = weight.asType(.uint32).reshaped([-1])
        let low = bitwiseAnd(packed, UInt32(0x0F))
        let high = bitwiseAnd(rightShift(packed, UInt32(4)), UInt32(0x0F))
        return MLX.stacked([low, high], axis: -1).reshaped(quantizedShape)
    }

    static func dequantize(
        weight: MLXArray,
        scales: MLXArray,
        zeroPoints: MLXArray,
        quantizedShape: [Int],
        resultShape: [Int]? = nil,
        dtype: DType
    ) -> MLXArray {
        let unpacked = unpack(weight, quantizedShape: quantizedShape).asType(dtype)
        var result = zeroPoints.asType(dtype) + unpacked * scales.asType(dtype)
        if let resultShape {
            result = result.reshaped(resultShape)
        }
        return result
    }

    static func squeezedGroupParameter(_ value: MLXArray) -> MLXArray {
        guard value.ndim > 2 else { return value }
        for axis in 2..<value.ndim where value.dim(axis) != 1 {
            return value
        }
        return value.reshaped([value.dim(0), value.dim(1)])
    }

    static func groupCount(_ value: MLXArray) -> Int {
        if value.ndim >= 2 {
            return value.dim(1)
        }
        return 1
    }
}
