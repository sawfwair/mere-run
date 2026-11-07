import MLX
import MLXFast

struct SheetSage2Weights {
    var values: [String: MLXArray]

    func get(_ key: String) throws -> MLXArray {
        guard let value = values[key] else { throw SheetSage2Error.invalid("Missing tensor \(key).") }
        return value
    }

    func linear(_ x: MLXArray, _ prefix: String, bias: Bool = true) throws -> MLXArray {
        let y = try matmul(x, get(prefix + ".weight").T)
        return try bias ? y + get(prefix + ".bias") : y
    }

    func norm(_ x: MLXArray, _ prefix: String, epsilon: Float) throws -> MLXArray {
        try MLXFast.layerNorm(x, weight: get(prefix + ".weight"), bias: get(prefix + ".bias"), eps: epsilon)
    }

    func conv(_ x: MLXArray, _ prefix: String, stride: Int = 1, padding: Int = 0, groups: Int = 1,
              bias: Bool = true) throws -> MLXArray {
        let y = try MLX.conv1d(x, get(prefix + ".weight").transposed(0, 2, 1), stride: stride, padding: padding, groups: groups)
        return try bias ? y + get(prefix + ".bias") : y
    }

    mutating func require(_ key: String, _ shape: [Int], consumed: inout Set<String>) throws {
        let value = try get(key)
        guard value.shape == shape, value.dtype == .float32 || value.dtype == .bfloat16 else {
            throw SheetSage2Error.invalid("Tensor \(key) must have floating-point shape \(shape), got \(value.shape).")
        }
        consumed.insert(key)
    }

    mutating func requireLinear(_ prefix: String, input: Int, output: Int, bias: Bool = true, consumed: inout Set<String>) throws {
        try require(prefix + ".weight", [output, input], consumed: &consumed)
        if bias { try require(prefix + ".bias", [output], consumed: &consumed) }
    }

    mutating func requireNorm(_ prefix: String, size: Int, consumed: inout Set<String>) throws {
        try require(prefix + ".weight", [size], consumed: &consumed)
        try require(prefix + ".bias", [size], consumed: &consumed)
    }
}
