import Foundation
import MLX
import MLXNN

/// Imports the VisualInference layer layout without evaluating Python model code.
enum NemotronImageCommunityPack {
    static let repository = "VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-8bit"

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

    static func checkRecipe(at root: URL, bits: Int) throws {
        let config = try NemotronImageResources.read(Config.self, at: root.appending(path: "config.json"))
        guard bits == 8, config.recipe.count == 1, let expert = config.recipe["experts"],
              expert.bits == bits, expert.group == 64, expert.mode == "affine" else {
            throw NemotronImageResources.invalid("Import requires group-64 affine Q8 routed experts.")
        }
    }

    static func weights(at root: URL, prefix: String, module: Module) throws -> [String: MLXArray] {
        if prefix == "backbone.embeddings." {
            let table = try MLX.loadArray(url: root.appending(path: "embed.npy"))
            guard table.dtype == .uint16 else { throw NemotronImageResources.invalid("Expected BF16 embedding bits.") }
            return ["weight": table.view(dtype: .bfloat16)]
        }
        if prefix == "lm_head." || prefix == "backbone.norm_f." {
            let arrays = try MLX.loadArrays(url: root.appending(path: "head.safetensors"))
            let key = prefix == "lm_head." ? "lm_head.weight" : "norm_f_weight"
            guard let tensor = arrays[key] else { throw NemotronImageResources.invalid("Missing \(key).") }
            return ["weight": tensor]
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
