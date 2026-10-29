import Foundation
import MLX

struct TranscribeTensorSpec: Decodable {
    let shape: [Int]
    let dtype: String
}

/// Original checkpoint names and layouts, including already-stacked [expert, input, output] matrices.
struct TranscribeWeights {
    let arrays: [String: MLXArray]
    let quantization: TranscribeExpertQuantization?

    init(arrays: [String: MLXArray], quantization: TranscribeExpertQuantization? = nil) {
        self.arrays = arrays
        self.quantization = quantization
    }

    subscript(_ key: String) -> MLXArray { arrays[key]! }

    func linear(_ x: MLXArray, _ prefix: String, bias: Bool = false) -> MLXArray {
        let output = x.matmul(self[prefix + ".weight"].T)
        return bias ? output + self[prefix + ".bias"] : output
    }

    func layerNorm(_ x: MLXArray, _ prefix: String) -> MLXArray {
        let values = x.asType(.float32)
        let centered = values - values.mean(axis: -1, keepDims: true)
        let normalized = centered * rsqrt(square(centered).mean(axis: -1, keepDims: true) + Float(1e-5))
        return (normalized * self[prefix + ".weight"].asType(.float32)
                + self[prefix + ".bias"].asType(.float32)).asType(x.dtype)
    }

    func rmsNorm(_ x: MLXArray, _ key: String, eps: Float) -> MLXArray {
        let values = x.asType(.float32)
        return (values * rsqrt(square(values).mean(axis: -1, keepDims: true) + eps)
                * self[key].asType(.float32)).asType(x.dtype)
    }

    static func inventory() throws -> [String: TranscribeTensorSpec] {
        try JSONDecoder().decode([String: TranscribeTensorSpec].self, from: Data(contentsOf:
            Bundle.module.url(forResource: "tensor-inventory", withExtension: "json")!))
    }

    static func load(at root: URL) throws -> Self {
        let quantization = try TranscribeExpertQuantization.load(at: root)
        let arrays: [String: MLXArray]
        if quantization != nil {
            arrays = try loadShards(at: root)
        } else {
            arrays = try MLX.loadArrays(url: root.appendingPathComponent("model.safetensors"))
        }
        return try validated(arrays: arrays, inventory: inventory(), quantization: quantization)
    }

    static func validated(arrays: [String: MLXArray], inventory: [String: TranscribeTensorSpec],
                          quantization: TranscribeExpertQuantization?) throws -> Self {
        var expected = inventory
        if let quantization {
            guard quantization == .supported else {
                throw NemotronTranscribeError.invalidCheckpoint("Unsupported routed-expert quantization format.")
            }
            for (key, spec) in inventory where TranscribeExpertQuantization.isExpert(key) {
                let shape = spec.shape
                expected.removeValue(forKey: key)
                expected[key + ".qweight"] = TranscribeTensorSpec(shape: [shape[0], shape[2], shape[1] / 8], dtype: "U32")
                for suffix in [".scales", ".biases"] {
                    expected[key + suffix] = TranscribeTensorSpec(shape: [shape[0], shape[2], shape[1] / quantization.groupSize], dtype: "BF16")
                }
            }
        }
        for (key, spec) in expected {
            guard let array = arrays[key], array.shape == spec.shape,
                  String(describing: array.dtype).lowercased() == dtypeName(spec.dtype) else {
                throw NemotronTranscribeError.invalidCheckpoint("Missing or incompatible tensor: \(key).")
            }
        }
        let unexpected = arrays.keys.filter {
            expected[$0] == nil && !$0.hasPrefix("ctc_timestamp.") && !$0.hasSuffix("._extra_state")
        }
        guard unexpected.isEmpty else {
            throw NemotronTranscribeError.invalidCheckpoint("Unexpected inference tensors: \(unexpected.sorted().joined(separator: ", ")).")
        }
        return Self(arrays: arrays.filter { expected[$0.key] != nil }, quantization: quantization)
    }

    private static func loadShards(at root: URL) throws -> [String: MLXArray] {
        struct Index: Decodable {
            let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
        }
        let index = try JSONDecoder().decode(Index.self,
            from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        var arrays: [String: MLXArray] = [:]
        for filename in Set(index.weightMap.values).sorted() {
            guard filename == URL(fileURLWithPath: filename).lastPathComponent, filename.hasSuffix(".safetensors") else {
                throw NemotronTranscribeError.invalidCheckpoint("Invalid checkpoint shard filename.")
            }
            let shard = try MLX.loadArrays(url: root.appendingPathComponent(filename))
            guard Set(shard.keys) == Set(index.weightMap.filter { $0.value == filename }.keys) else {
                throw NemotronTranscribeError.invalidCheckpoint("Shard tensors do not match the checkpoint index: \(filename).")
            }
            arrays.merge(shard) { _, value in value }
        }
        return arrays
    }

    private static func dtypeName(_ value: String) -> String {
        switch value { case "BF16": "bfloat16"; case "F32": "float32"; case "U32": "uint32"; default: value.lowercased() }
    }
}
