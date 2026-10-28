import Foundation
import MLX
import MLXNN
import MereRunFlintModel

enum FlintLoader {
    struct Index: Codable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    static func load(root: URL) throws -> FlintCausalLM {
        let config = try JSONDecoder().decode(FlintConfiguration.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        try config.validate()
        if config.quantization != nil { try FlintArtifactVerification.verify(root: root) }
        let model = try FlintCausalLM(config: config)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let optional = Set(expected.keys.filter { $0.hasSuffix(".gate.expert_bias") })
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        let index = try FileManager.default.fileExists(atPath: indexURL.path)
            ? JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL)) : nil
        if config.quantization == nil, index == nil { throw FlintModelError.invalidWeights("BF16 requires its source index.") }
        let names = index.map { Set($0.weightMap.values).sorted() }
            ?? FlintArtifactVerification.hashes.keys.filter { $0.hasSuffix(".safetensors") }.sorted()
        let actualShards = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil, options: .skipsHiddenFiles
        ).map(\.lastPathComponent).filter { $0.hasSuffix(".safetensors") }
        guard Set(names) == Set(actualShards) else { throw FlintModelError.invalidWeights("Shard set differs from the checkpoint index.") }
        var loaded = Set<String>()
        var sourceKeys = Set<String>()
        var pending: [String: [Int: MLXArray]] = [:]
        for name in names {
            try Task.checkCancellation()
            guard name == URL(fileURLWithPath: name).lastPathComponent, name.hasSuffix(".safetensors") else {
                throw FlintModelError.invalidWeights("Invalid shard name.")
            }
            let arrays = try MLX.loadArrays(url: root.appendingPathComponent(name))
            if let index, Set(arrays.keys) != Set(index.weightMap.filter { $0.value == name }.keys) {
                throw FlintModelError.invalidWeights("Shard ownership differs from index: \(name).")
            }
            var updates: [String: MLXArray] = [:]
            for (key, raw) in arrays {
                guard sourceKeys.insert(key).inserted else { throw FlintModelError.invalidWeights("Duplicate tensor: \(key).") }
                let mapping = expertMapping(key)
                let target = mapping?.target ?? key
                guard let parameter = expected[target] else { throw FlintModelError.invalidWeights("Unknown tensor: \(key).") }
                let array = raw.dtype == .uint8 && parameter.dtype == .uint32 ? raw.view(dtype: .uint32) : raw
                let shape = mapping == nil ? parameter.shape : Array(parameter.shape.dropFirst())
                let validDType = parameter.dtype == .uint32 ? array.dtype == .uint32 : [.float32, .float16, .bfloat16].contains(array.dtype)
                guard array.shape == shape, validDType else { throw FlintModelError.invalidWeights("Shape or dtype mismatch: \(key).") }
                if let mapping {
                    guard (0..<config.numExperts).contains(mapping.index), !loaded.contains(target) else {
                        throw FlintModelError.invalidWeights("Invalid expert index: \(key).")
                    }
                    pending[target, default: [:]][mapping.index] = array
                    if pending[target]?.count == config.numExperts, let group = pending.removeValue(forKey: target) {
                        let stackedWeights = stacked((0..<config.numExperts).map { group[$0]! })
                        eval(stackedWeights)
                        updates[target] = stackedWeights
                        loaded.insert(target)
                    }
                } else {
                    guard loaded.insert(target).inserted else { throw FlintModelError.invalidWeights("Duplicate target: \(target).") }
                    updates[target] = array
                }
            }
            try model.update(parameters: .unflattened(updates), verify: .shapeMismatch)
            eval(Array(updates.values))
        }
        guard pending.isEmpty, loaded.subtracting(optional) == Set(expected.keys).subtracting(optional) else {
            throw FlintModelError.invalidWeights("Incomplete checkpoint coverage.")
        }
        return model
    }

    static func expertMapping(_ key: String) -> (target: String, index: Int)? {
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count == 8, parts[0] == "model", parts[1] == "layers", parts[3] == "mlp",
              parts[4] == "experts", let expert = Int(parts[5]), parts[7] == "weight" else { return nil }
        return ("model.layers.\(parts[2]).mlp.switch_mlp.\(parts[6]).weight", expert)
    }
}
