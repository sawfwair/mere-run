import Foundation
import MLX

public enum ONNXMLXInitializerLoader {
    public static func load(
        archive: ONNXInitializerArchive,
        names: Set<String>? = nil
    ) throws -> [String: MLXArray] {
        var arrays: [String: MLXArray] = [:]
        for tensor in archive.tensors where names == nil || names!.contains(tensor.name) {
            arrays[tensor.name] = MLXArray(
                archive.rawData(for: tensor),
                tensor.shape,
                dtype: try mlxDataType(tensor.dataType)
            )
        }
        return arrays
    }

    private static func mlxDataType(_ type: ONNXTensorDataType) throws -> DType {
        switch type {
        case .float32: .float32
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .uint8: .uint8
        case .int8: .int8
        case .int16: .int16
        case .int32: .int32
        case .int64: .int64
        case .bool: .bool
        case .uint16, .uint32, .uint64, .float64:
            throw ONNXInitializerError.unsupportedTensorType(type.rawValue)
        }
    }
}
