import Foundation
import AudioSOVTranscribeModel
import MereRunCore
import MLX
import MLXNN

/// Original BF16 checkpoint layout. Router tensors retain the checkpoint's FP32 dtype.
enum SOVTranscribeWeights {
    static func mappedTarget(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard key.hasPrefix("model.language_model.") else { return [] }
        let name = String(key.dropFirst("model.".count))
        if name.hasSuffix(".experts.gate_up_proj") {
            let prefix = String(name.dropLast("gate_up_proj".count)) + "switch_glu."
            let width = value.dim(1) / 2
            return [(prefix + "gate_proj.weight", value[0..., ..<width, 0...]),
                    (prefix + "up_proj.weight", value[0..., width..., 0...])]
        }
        if name.hasSuffix(".experts.down_proj") {
            return [(String(name.dropLast("down_proj".count)) + "switch_glu.down_proj.weight", value)]
        }
        return [(name, value)]
    }

    static func load(root: URL, into model: Module, sourcePrefix: String = "", target: Bool = false) throws {
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        let shards: [String]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            shards = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL)).shardFilenames
        } else { shards = ["model.safetensors"] }
        var loaded: Set<String> = []
        let expected = model.parameters().flattened().map(\.0).filter { name in
            guard target else { return true }
            return !name.contains("per_layer") && !name.contains(".rope.")
        }
        for shard in shards {
            try Task.checkCancellation()
            guard URL(fileURLWithPath: shard).lastPathComponent == shard else {
                throw SOVTranscribeError.invalid("Invalid checkpoint shard filename: \(shard).")
            }
            let url = root.appendingPathComponent(shard)
            // Materialize file-backed MLX nodes directly. Foundation tensor reads
            // temporarily duplicate this 53 GB checkpoint in the process footprint.
            try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
                url: url, to: model, verify: [.noUnusedKeys, .shapeMismatch], include: {
                    target ? $0.hasPrefix("model.language_model.") : $0.hasPrefix(sourcePrefix)
                }, mapper: { key, value in
                let name = String(key.dropFirst(sourcePrefix.count))
                let tensor = sourcePrefix == "model.audio_tower." && ["conv1.weight", "conv2.weight"].contains(name)
                    ? value.transposed(0, 2, 1) : value
                let mapped = target ? mappedTarget(key, value) : [(name, tensor)]
                loaded.formUnion(mapped.map(\.0))
                return mapped
            }, batchSize: 24)
        }
        let missing = Set(expected).subtracting(loaded).sorted()
        guard missing.isEmpty else { throw SOVTranscribeError.invalid("Checkpoint is missing tensors: \(missing.joined(separator: ", ")).") }
    }
}
