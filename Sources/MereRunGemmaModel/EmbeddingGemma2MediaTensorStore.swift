import Foundation
import MLX

/// Original checkpoint tensors. Shapes are checked before any media computation.
struct EmbeddingGemma2MediaTensorStore {
    let values: [String: MLXArray]
    let dtype: DType

    init(tensors: [String: MLXArray], shapes: [String: [Int]], dtype: DType) throws {
        guard dtype == .bfloat16 || dtype == .float32 else {
            throw EmbeddingGemma2Error.invalidConfiguration("Media encoders require BF16 or FP32.")
        }
        for (key, shape) in shapes {
            guard let tensor = tensors[key], tensor.shape == shape, tensor.dtype.isFloatingPoint else {
                throw EmbeddingGemma2Error.invalidConfiguration("Missing or invalid media tensor: \(key), expected \(shape).")
            }
        }
        guard Set(tensors.keys) == Set(shapes.keys) else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unexpected media tensors: \(Set(tensors.keys).subtracting(shapes.keys).sorted()).")
        }
        self.dtype = dtype
        values = tensors.mapValues { $0.asType(dtype) }
    }

    subscript(_ key: String) -> MLXArray { values[key]! }

    func linear(_ x: MLXArray, _ key: String, bias: Bool = false) -> MLXArray {
        let y = matmul(x, self[key + ".weight"].T)
        return bias ? y + self[key + ".bias"] : y
    }

    func clippedLinear(_ x: MLXArray, _ key: String) -> MLXArray {
        let input = minimum(maximum(x, self[key + ".input_min"]), self[key + ".input_max"])
        return minimum(maximum(linear(input, key + ".linear"), self[key + ".output_min"]), self[key + ".output_max"])
    }

    func norm(_ x: MLXArray, _ key: String, eps: Float) -> MLXArray {
        (EmbeddingGemma2RMSNorm.normalize(x, eps: eps) * self[key + ".weight"].asType(.float32)).asType(x.dtype)
    }

    static func linearShapes(_ key: String, input: Int, output: Int, clipped: Bool = false) -> [String: [Int]] {
        var shapes = [key + (clipped ? ".linear.weight" : ".weight"): [output, input]]
        if clipped {
            for suffix in ["input_min", "input_max", "output_min", "output_max"] { shapes[key + "." + suffix] = [] }
        }
        return shapes
    }
}
