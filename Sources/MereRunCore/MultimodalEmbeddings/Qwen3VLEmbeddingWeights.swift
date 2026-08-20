import Foundation
import MLX
import MLXNN

enum Qwen3VLEmbeddingWeights {
    static func load(
        resources: Qwen3VLEmbeddingResources,
        config: Qwen3VLEmbeddingRootConfig,
        into model: QwenVLEncoder,
        fileManager: FileManager = .default
    ) throws {
        let isQuantized = config.quantization != nil
            || indexContainsQuantizedWeights(resources.weightsIndexURL, fileManager: fileManager)
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        if isQuantized {
            if fileManager.fileExists(atPath: resources.weightsIndexURL.path) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.weightsIndexURL,
                    to: model,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: mapWeightKey
                )
                return
            }
            let arrays = try MLX.loadArrays(url: resources.weightsURL)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: mapWeightKey
            )
            return
        }

        if fileManager.fileExists(atPath: resources.weightsIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: resources.weightsIndexURL,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: mapWeight
            )
            return
        }

        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.weightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: mapWeight
        )
    }

    static func mapWeight(_ rawKey: String, _ value: MLXArray) -> [(String, MLXArray)] {
        let key = mapWeightKey(rawKey)
        guard key == "visionTower.patch_embed.proj.weight", value.ndim == 5 else {
            return [(key, value)]
        }

        // MLX Conv3d stores [out, temporal, height, width, channels]. Official
        // PyTorch checkpoints store [out, channels, temporal, height, width].
        if value.dim(4) == 3 {
            return [(key, value)]
        }
        return [(key, value.transposed(0, 2, 3, 4, 1))]
    }

    static func mapWeightKey(_ rawKey: String) -> String {
        var key = rawKey

        if key.hasPrefix("model.language_model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("model.language_model.".count))
        } else if key.hasPrefix("language_model.model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("language_model.model.".count))
        } else if key.hasPrefix("model.visual.") {
            key = "visionTower." + String(key.dropFirst("model.visual.".count))
        } else if key.hasPrefix("vision_tower.") {
            key = "visionTower." + String(key.dropFirst("vision_tower.".count))
        } else if key.hasPrefix("visual.") {
            key = "visionTower." + String(key.dropFirst("visual.".count))
        }

        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        key = key.replacingOccurrences(of: ".patch_merger.mlp.0.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.mlp.2.", with: ".patch_merger.mlp_2.")
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc1.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc2.", with: ".patch_merger.mlp_2.")

        if key.contains(".deepstack_merger_list.") {
            key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")
            key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
            key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        }

        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
        return key
    }

    private static func indexContainsQuantizedWeights(
        _ indexURL: URL,
        fileManager: FileManager
    ) -> Bool {
        guard fileManager.fileExists(atPath: indexURL.path),
              let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return false
        }
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }
}
