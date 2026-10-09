import MLX

/// Component-local affine weight packing. Layers without scales retain source precision.
public struct QwenImage21Quantization: Codable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    enum CodingKeys: String, CodingKey {
        case bits, groupSize = "group_size", mode
    }

    public init(bits: Int, groupSize: Int = 64, mode: String = "affine") {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
    }

    public func validate(arrays: [String: MLXArray], shapes: [String: [Int]]) throws {
        guard [4, 8].contains(bits), groupSize == 64, mode == "affine" else {
            throw QwenImage21Error.invalidConfiguration("Qwen Image 2.1 supports affine Q4/Q8 group-64 weights.")
        }
        var expected = Set(shapes.keys)
        for (key, shape) in shapes {
            guard let weight = arrays[key] else {
                throw QwenImage21Error.invalidWeights("Missing tensor: \(key)")
            }
            let prefix = String(key.dropLast(".weight".count))
            if key.hasSuffix(".weight"), let scales = arrays[prefix + ".scales"] {
                guard shape.count == 2, shape[1].isMultiple(of: groupSize), weight.dtype == .uint32,
                      weight.shape == [shape[0], shape[1] / (32 / bits)],
                      scales.shape == [shape[0], shape[1] / groupSize],
                      scales.dtype == .bfloat16 || scales.dtype == .float16 || scales.dtype == .float32,
                      let biases = arrays[prefix + ".biases"], biases.shape == scales.shape,
                      biases.dtype == scales.dtype else {
                    throw QwenImage21Error.invalidWeights("Invalid packed affine tensor: \(key)")
                }
                expected.insert(prefix + ".scales")
                expected.insert(prefix + ".biases")
            } else if weight.shape != shape || weight.dtype == .uint32 {
                throw QwenImage21Error.invalidWeights("Dense tensor mismatch: \(key)")
            }
        }
        guard Set(arrays.keys) == expected else {
            throw QwenImage21Error.invalidWeights("Unexpected or missing quantization tensors.")
        }
    }
}
