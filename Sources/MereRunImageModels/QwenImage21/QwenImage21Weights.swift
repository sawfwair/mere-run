import MLX
import MLXNN

/// Checkpoint tensors retain the official names. Shape admission happens before inference.
struct QwenImage21Weights {
    let arrays: [String: MLXArray]

    init(_ arrays: [String: MLXArray], shapes: [String: [Int]]) throws {
        for (key, shape) in shapes {
            guard let value = arrays[key], value.shape == shape else {
                throw QwenImage21Error.invalidWeights("\(key): expected \(shape), found \(arrays[key]?.shape ?? [])")
            }
        }
        let unexpected = Set(arrays.keys).subtracting(shapes.keys)
        guard unexpected.isEmpty else {
            throw QwenImage21Error.invalidWeights("Unexpected tensors: \(unexpected.sorted().joined(separator: ", "))")
        }
        self.arrays = arrays
    }

    subscript(_ key: String) -> MLXArray { arrays[key]! }

    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = matmul(x, self[name + ".weight"].T)
        return arrays[name + ".bias"].map { y + $0 } ?? y
    }

    func rms(_ x: MLXArray, _ name: String, epsilon: Float, zeroCentered: Bool = false) -> MLXArray {
        let value = x.asType(.float32)
        let normalized = value * rsqrt(mean(value * value, axis: -1, keepDims: true) + epsilon)
        let weight = self[name + ".weight"]
        if zeroCentered {
            return (normalized * (weight.asType(.float32) + 1)).asType(x.dtype)
        }
        // Diffusers RMSNorm rounds the normalized activation before the learned scale.
        return normalized.asType(weight.dtype) * weight
    }

    static func layerNorm(_ x: MLXArray, epsilon: Float) -> MLXArray {
        let value = x.asType(.float32)
        let centered = value - mean(value, axis: -1, keepDims: true)
        return (centered * rsqrt(mean(centered * centered, axis: -1, keepDims: true) + epsilon)).asType(x.dtype)
    }
}
