import Foundation
import MLX
import MLXNN

/// Imports the VisualInference layer layout without evaluating Python model code.
enum NemotronImageCommunityPack {
    enum Variant: Int {
        case mixed = 6
        case q8 = 8

        var repository: String { "VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-\(rawValue)bit" }
        var revision: String {
            self == .mixed ? "42b90afa978f4a4564b72b203e7e7669dcf3f36c" : "e9ea81ae28322e58d20fd94fb3292e1fc7f36ece"
        }
        var scheme: String { self == .mixed ? NemotronImageQuantization.mixedScheme : NemotronImageQuantization.scheme }
    }

    static let metadataFiles = ["config.json", "tokenizer.json", "tokenizer_config.json",
                                "chat_template.jinja", "unitron_extension_manifest.json"]
    static let files = metadataFiles + ["embed.npy", "head.safetensors", "vq/config.json", "vq/model.safetensors"]
        + (0..<52).map { String(format: "layers-%02d.safetensors", $0) }

    static func isPresent(at root: URL) -> Bool {
        FileManager.default.fileExists(atPath: root.appending(path: "embed.npy").path)
    }

    struct Config: Decodable {
        let recipe: [String: Recipe]
        enum CodingKeys: String, CodingKey { case recipe = "mlx_recipe" }

        struct Recipe: Decodable {
            let bits: Int
            let group: Int
            let mode: String

            init(from decoder: Decoder) throws {
                var values = try decoder.unkeyedContainer()
                bits = try values.decode(Int.self)
                group = try values.decode(Int.self)
                mode = try values.decode(String.self)
                guard values.isAtEnd else {
                    throw NemotronImageResources.invalid("Unexpected MLX recipe fields.")
                }
            }
        }
    }

    static func variant(at root: URL) throws -> Variant {
        let config = try NemotronImageResources.read(Config.self, at: root.appending(path: "config.json"))
        guard let bits = config.recipe["experts"]?.bits, let variant = Variant(rawValue: bits) else {
            throw NemotronImageResources.invalid("Unsupported routed-expert precision.")
        }
        let expected = variant == .mixed ? ["experts": 6, "dense": 8, "shared": 8, "lm_head": 8] : ["experts": 8]
        guard config.recipe.mapValues(\.bits) == expected,
              config.recipe.values.allSatisfy({ $0.group == 64 && $0.mode == "affine" }) else {
            throw NemotronImageResources.invalid("Unsupported Nemotron Image quantization recipe.")
        }
        return variant
    }

    static func weights(at root: URL, prefix: String, module: Module) throws -> [String: MLXArray] {
        if prefix == "backbone.embeddings." {
            let table = try MLX.loadArray(url: root.appending(path: "embed.npy"))
            guard table.dtype == .uint16 else { throw NemotronImageResources.invalid("Expected BF16 embedding bits.") }
            return ["weight": table.view(dtype: .bfloat16)]
        }
        if prefix == "lm_head." || prefix == "backbone.norm_f." {
            let arrays = try MLX.loadArrays(url: root.appending(path: "head.safetensors"))
            return try Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { key, _ in
                let name = prefix == "lm_head." ? prefix + key : "norm_f_weight"
                guard let value = arrays[name] else { throw NemotronImageResources.invalid("Missing \(name).") }
                return (key, value)
            })
        }
        guard let suffix = prefix.split(separator: ".").last, let index = Int(suffix) else {
            throw NemotronImageResources.invalid("Invalid layer prefix \(prefix).")
        }
        let file = String(format: "layers-%02d.safetensors", index)
        let arrays = try MLX.loadArrays(url: root.appending(path: file))
        let mapping = ["norm.weight": "norm_weight", "mixer.norm.weight": "mixer.norm_weight",
                       "mixer.conv1d.weight": "mixer.conv_weight", "mixer.conv1d.bias": "mixer.conv_bias",
                       "mixer.A_log": "mixer.A_neg", "mixer.gate.weight": "mixer.gate_weight",
                       "mixer.gate.e_score_correction_bias": "mixer.e_score_correction_bias"]
        var result: [String: MLXArray] = [:]
        for (key, _) in module.parameters().flattened() {
            let name = mapping[key] ?? key.replacingOccurrences(of: "mixer.experts.up_proj.", with: "mixer.fc1.")
                .replacingOccurrences(of: "mixer.experts.down_proj.", with: "mixer.fc2.")
            guard let value = arrays[name] else { throw NemotronImageResources.invalid("Missing \(file)/\(name).") }
            switch key {
            case "mixer.A_log": result[key] = log(-value).asType(.bfloat16)
            case "mixer.conv1d.weight": result[key] = value.expandedDimensions(axis: -1)
            case "mixer.gate.e_score_correction_bias": result[key] = value.asType(.bfloat16)
            default: result[key] = value
            }
        }
        return result
    }
}
