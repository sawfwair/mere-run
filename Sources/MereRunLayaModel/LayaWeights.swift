import MLX

/// Consumes the original PyTorch safetensors names without conversion or transposition.
struct LayaWeights {
    var arrays: [String: MLXArray]

    mutating func take(_ key: String, _ shape: [Int]) throws -> MLXArray {
        guard let value = arrays.removeValue(forKey: key), value.shape == shape,
              [.float16, .bfloat16, .float32].contains(value.dtype) else {
            throw LayaModelError.invalidWeights("Laya tensor \(key) must have shape \(shape) and floating-point weights.")
        }
        // The reference evaluates CPU/MPS in float32, including the action head.
        return value.asType(.float32)
    }

    mutating func linear(_ prefix: String, _ input: Int, _ output: Int, bias: Bool = true) throws -> LayaLinear {
        try LayaLinear(weight: take(prefix + ".weight", [output, input]),
                       bias: bias ? take(prefix + ".bias", [output]) : nil)
    }

    mutating func norm(_ prefix: String, _ size: Int, bias: Bool = true, epsilon: Float = 1e-5) throws -> LayaNorm {
        try LayaNorm(weight: take(prefix + ".weight", [size]),
                     bias: bias ? take(prefix + ".bias", [size]) : nil, epsilon: epsilon)
    }
}

struct LayaLinear {
    let weight: MLXArray
    let bias: MLXArray?

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let result = matmul(input, weight.T)
        return bias.map { result + $0 } ?? result
    }
}

struct LayaNorm {
    let weight: MLXArray
    let bias: MLXArray?
    let epsilon: Float

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.layerNorm(input, weight: weight, bias: bias, eps: epsilon)
    }
}
