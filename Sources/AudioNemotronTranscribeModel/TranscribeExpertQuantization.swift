import Foundation
import MLX

/// The local conversion changes only routed expert matrices. All other tensors keep their published dtype.
struct TranscribeExpertQuantization: Codable, Equatable {
    let version: Int
    let bits: Int
    let groupSize: Int
    let mode: String
    let scope: String
    let layout: String

    static let supported = Self(version: 1, bits: 4, groupSize: 64, mode: "affine",
                                scope: "routed-experts", layout: "expert-output-input")
    static let filename = "expert-quantization.json"

    static func load(at root: URL) throws -> Self? {
        let url = root.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        guard value == supported else {
            throw NemotronTranscribeError.invalidCheckpoint("Unsupported routed-expert quantization format.")
        }
        return value
    }

    static func isExpert(_ key: String) -> Bool {
        key.hasPrefix("llm.model.layers.") && key.contains(".mixer.experts.")
            && (key.hasSuffix(".gate_and_up_projs") || key.hasSuffix(".down_projs"))
    }
}

extension TranscribeWeights {
    /// Input has matrix shape [..., 1, input]; selected output has shape [..., 1, output].
    func routedLinear(_ x: MLXArray, indices: MLXArray, key: String) -> MLXArray {
        if let quantization {
            return gatherQuantizedMM(x, self[key + ".qweight"], scales: self[key + ".scales"],
                biases: self[key + ".biases"], rhsIndices: indices, transpose: true,
                groupSize: quantization.groupSize, bits: quantization.bits)
        }
        return matmul(x, self[key].take(indices, axis: 0))
    }
}
