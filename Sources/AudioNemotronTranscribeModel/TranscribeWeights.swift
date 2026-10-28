import Foundation
import MLX

struct TranscribeTensorSpec: Decodable {
    let shape: [Int]
    let dtype: String
}

/// Original checkpoint names and layouts, including already-stacked [expert, input, output] matrices.
struct TranscribeWeights {
    let arrays: [String: MLXArray]

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
        let arrays = try MLX.loadArrays(url: root.appendingPathComponent("model.safetensors"))
        let expected = try inventory()
        for (key, spec) in expected {
            guard let array = arrays[key], array.shape == spec.shape,
                  String(describing: array.dtype).lowercased() == dtypeName(spec.dtype) else {
                throw NemotronTranscribeError.invalidCheckpoint("Missing or incompatible BF16 tensor: \(key).")
            }
        }
        let unexpected = arrays.keys.filter {
            expected[$0] == nil && !$0.hasPrefix("ctc_timestamp.") && !$0.hasSuffix("._extra_state")
        }
        guard unexpected.isEmpty else {
            throw NemotronTranscribeError.invalidCheckpoint("Unexpected inference tensors: \(unexpected.sorted().joined(separator: ", ")).")
        }
        return Self(arrays: arrays.filter { expected[$0.key] != nil })
    }

    private static func dtypeName(_ value: String) -> String {
        switch value { case "BF16": "bfloat16"; case "F32": "float32"; default: value.lowercased() }
    }
}
