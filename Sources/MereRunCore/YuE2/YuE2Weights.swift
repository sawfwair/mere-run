import MLX

/// Consumes every expected tensor exactly once; missing, extra, and mismatched weights are errors.
struct YuE2Weights {
    private var arrays: [String: MLXArray]

    init(_ arrays: [String: MLXArray]) { self.arrays = arrays }

    mutating func take(_ name: String, _ shape: [Int]) throws -> MLXArray {
        guard let value = arrays.removeValue(forKey: name) else {
            throw YuE2Error.invalidWeights("Missing tensor \(name).")
        }
        guard value.shape == shape, value.dtype == .bfloat16 || value.dtype == .float32 else {
            throw YuE2Error.invalidWeights("\(name): expected floating-point \(shape), got \(value.dtype) \(value.shape).")
        }
        return value
    }

    func finish() throws {
        guard arrays.isEmpty else {
            throw YuE2Error.invalidWeights("Unexpected tensors: \(arrays.keys.sorted().prefix(8).joined(separator: ", ")).")
        }
    }
}

struct YuE2Linear {
    let weight: MLXArray
    let bias: MLXArray?

    init(_ weights: inout YuE2Weights, _ prefix: String, input: Int, output: Int, bias: Bool = false) throws {
        weight = try weights.take(prefix + ".weight", [output, input])
        self.bias = try bias ? weights.take(prefix + ".bias", [output]) : nil
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let output = matmul(input, weight.T)
        return bias.map { output + $0 } ?? output
    }
}
