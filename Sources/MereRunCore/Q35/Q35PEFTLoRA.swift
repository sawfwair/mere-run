import Foundation
import MereRunQwenModel
import MLX
import MLXNN

enum Q35PEFTLoRA {
    private struct Configuration: Decodable {
        let base_model_name_or_path: String
        let peft_type: String
        let r: Int
        let lora_alpha: Float
        let bias: String
        let lora_bias: Bool?
        let fan_in_fan_out: Bool
        let use_dora: Bool?
        let use_rslora: Bool?
        let rank_pattern: [String: Int]?
        let alpha_pattern: [String: Float]?
    }

    static func apply(from url: URL, scale: Double, to model: Q35Model) throws {
        let configURL = url.deletingLastPathComponent().appendingPathComponent("adapter_config.json")
        let config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf: configURL))
        let multiplier = config.lora_alpha / Float(config.r) * Float(scale)
        guard config.base_model_name_or_path == Q35Resources.q38TwentySevenBUpstreamRepoId,
              config.peft_type == "LORA", config.r > 0, config.lora_alpha > 0, multiplier.isFinite,
              config.bias == "none", config.lora_bias != true, !config.fan_in_fan_out,
              config.use_dora != true, config.use_rslora != true,
              config.rank_pattern?.isEmpty != false, config.alpha_pattern?.isEmpty != false else {
            throw LoRAError.invalidFormat("Expected a Qwen3.8-27B PEFT LoRA with uniform rank and alpha and no bias.")
        }

        let weights = try MLX.loadArrays(url: url)
        let projections = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let downSuffix = ".lora_A.weight"
        let namespace = "base_model.model."
        let downKeys = weights.keys.filter { $0.hasSuffix(downSuffix) }.sorted()
        guard !downKeys.isEmpty else { throw LoRAError.noWeightPairs }

        var replacements: [(String, Module)] = []
        for key in downKeys {
            let moduleKey = String(key.dropLast(downSuffix.count))
            guard moduleKey.hasPrefix(namespace),
                  let path = Q35Generator.mapTextWeightKey(String(moduleKey.dropFirst(namespace.count))),
                  let base = projections[path] as? Linear,
                  let down = weights[key], let up = weights[moduleKey + ".lora_B.weight"] else {
                throw LoRAError.invalidFormat("Missing matching Qwen projection or LoRA pair for \(key).")
            }
            let (outDim, inDim) = base.shape
            guard down.shape == [config.r, inDim], up.shape == [outDim, config.r],
                  down.dtype.isFloatingPoint, up.dtype == down.dtype else {
                throw LoRAError.invalidFormat("Invalid LoRA tensor shape or dtype for \(path).")
            }
            replacements.append((path, Q35LoRALinear(base: base, down: down, up: up, scale: multiplier)))
        }
        guard weights.count == replacements.count * 2,
              Set(replacements.map(\.0)).count == replacements.count else {
            throw LoRAError.invalidFormat("Unexpected or duplicate tensors in the Qwen3.8 LoRA.")
        }
        model.update(modules: ModuleChildren.unflattened(replacements))
    }
}
