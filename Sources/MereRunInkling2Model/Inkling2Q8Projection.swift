import MLX

struct Inkling2Q8Projection {
    static let outputKey = "model.llm.unembed"
    static let draftKey = "mererun.dspark.markov"
    static let draftBiasKey = draftKey + ".output_bias"
    static let groupSize = 64
    static let bits = 8

    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?

    init(weight: MLXArray, scales: MLXArray, biases: MLXArray?) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }

    static func keys(_ key: String) -> [String] {
        [key + ".q8_weight", key + ".scales", key + ".biases"]
    }

    init(weights: Inkling2Weights, key: String, rows: Int, columns: Int) throws {
        guard columns.isMultiple(of: Self.groupSize) else {
            throw Inkling2Error.invalid("invalid Q8 projection group size: \(key)")
        }
        let keys = Self.keys(key)
        let groupShape = [rows, columns / Self.groupSize]
        try weights.require(keys[0], shape: [rows, columns / 4], dtype: "U32")
        try weights.require(keys[1], shape: groupShape, dtype: "BF16")
        try weights.require(keys[2], shape: groupShape, dtype: "BF16")
        self.init(weight: try weights.read(keys[0]), scales: try weights.read(keys[1]), biases: try weights.read(keys[2]))
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        quantizedMM(input, weight, scales: scales, biases: biases, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
    }
}
