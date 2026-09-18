import Foundation
import MLX
import MLXNN

/// The schema-2 pack records every packed module, including unrotated projections.
package struct Q35PrismConfiguration: Decodable {
    struct Record: Decodable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }
    let schemaVersion: Int
    let modelType: String
    let tensorNamespace: String
    let gdnActivationLayout: String
    let modules: [Record]
    let quantization: Q35QuantizationConfig
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelType = "model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case modules, quantization
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    package enum LoadError: LocalizedError {
        case unsupportedContract
        case invalidModule(String)

        package var errorDescription: String? {
            switch self {
            case .unsupportedContract:
                "Unsupported or incomplete Bonsai 2 pack; use the pinned schema-2 MLX snapshot."
            case .invalidModule(let path):
                "Invalid Bonsai 2 packed module \(path): check its shape, quantization, and sign vector."
            }
        }
    }

    /// The native hybrid layer owns both attention modules, but a checkpoint
    /// contains only the selected branch. The Q/K normalization buffer is generated.
    package func checkpointParameterNames(model: Q35Model) -> Set<String> {
        let inactive = model.config.textConfig.layerTypes.enumerated().map { index, type in
            let branch = Q35AttentionLayerType(type) == .linear ? "self_attn" : "linear_attn"
            return "model.layers.\(index).\(branch)."
        }
        return Set(model.parameters().flattened().map(\.0).filter { key in
            !key.hasSuffix(".linear_attn.qkNormWeightBF16")
                && !inactive.contains(where: key.hasPrefix)
        })
    }

    /// Validate against the original dense modules before installing packed replacements.
    package func replacements(model: Module, arrays: [String: MLXArray]) throws -> [(String, Module)] {
        guard schemaVersion == 2, modelType == "prism_hadamard_qwen35",
              tensorNamespace == "mlx-vlm-qwen3_5", gdnActivationLayout == "grouped",
              quantization.bits == 2, quantization.groupSize == 128, quantization.mode == "affine",
              !tieWordEmbeddings else {
            throw LoadError.unsupportedContract
        }
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var seen = Set<String>()
        var result: [(String, Module)] = []
        for record in modules {
            guard seen.insert(record.path).inserted, record.dtype == "float16",
                  let original = leaves[record.path],
                  record.embedding ? original is Embedding : original is Linear,
                  let originalWeight = original.parameters().flattened().first(where: { $0.0 == "weight" })?.1,
                  let weight = arrays[record.path + ".weight"],
                  let scales = arrays[record.path + ".scales"],
                  let biases = arrays[record.path + ".biases"] else {
                throw LoadError.invalidModule(record.path)
            }
            let width = originalWeight.dim(1)
            let rows = originalWeight.dim(0)
            let signs = arrays[record.path + ".signs"]
            guard width.isMultiple(of: 128), weight.dtype == .uint32,
                  weight.shape == [rows, width / 16], scales.shape == [rows, width / 128],
                  biases.shape == scales.shape,
                  [.float16, .float32, .bfloat16].contains(scales.dtype),
                  [.float16, .float32, .bfloat16].contains(biases.dtype),
                  MLX.all(MLX.isFinite(scales)).item(Bool.self),
                  MLX.all(MLX.isFinite(biases)).item(Bool.self) else {
                throw LoadError.invalidModule(record.path)
            }
            if record.block == 0 {
                guard signs == nil else { throw LoadError.invalidModule(record.path) }
            } else {
                guard [512, 1024, 2048, 4096].contains(record.block), width.isMultiple(of: record.block),
                      let signs, signs.shape == [width],
                      MLX.all((signs .== 1) .|| (signs .== -1)).item(Bool.self) else {
                    throw LoadError.invalidModule(record.path)
                }
            }
            let replacement: Module = record.embedding
                ? Q35PrismEmbedding(weight: weight, scales: scales, biases: biases, signs: signs, block: record.block)
                : Q35PrismLinear(weight: weight, scales: scales, biases: biases, signs: signs, block: record.block)
            result.append((record.path, replacement))
        }
        let packedPaths = Set(arrays.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        guard seen == packedPaths, !seen.isEmpty else { throw LoadError.unsupportedContract }
        return result
    }
}
