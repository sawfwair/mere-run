import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    private static func hasMTPWeights(resources: Q35Resources) -> Bool {
        if standaloneMTPWeightsURL(resources: resources) != nil {
            return true
        }
        guard FileManager.default.fileExists(atPath: resources.modelIndexURL.path),
              let data = try? Data(contentsOf: resources.modelIndexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return false
        }
        return index.weightMap.keys.contains { $0.hasPrefix("mtp.") }
    }

    static func mtpResources(
        primary resources: Q35Resources,
        companionRootURL: URL? = nil
    ) -> Q35Resources? {
        if hasMTPWeights(resources: resources) {
            return resources
        }
        let mounted = Q35Resources(
            rootURL: resources.rootURL.appendingPathComponent(
                Q35Resources.q38MTPComponentPath,
                isDirectory: true
            )
        )
        if hasMTPWeights(resources: mounted) {
            return mounted
        }
        if let companionRootURL {
            let companion = Q35Resources(rootURL: companionRootURL)
            if hasMTPWeights(resources: companion) {
                return companion
            }
        }
        return nil
    }

    private static func standaloneMTPWeightsURL(resources: Q35Resources) -> URL? {
        let explicit = resources.rootURL.appendingPathComponent("mtp.safetensors")
        if FileManager.default.fileExists(atPath: explicit.path) {
            return explicit
        }
        guard resources.rootURL.lastPathComponent == Q35Resources.q38MTPComponentPath,
              FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) else {
            return nil
        }
        return resources.modelWeightsURL
    }

    func loadQ38MTPWeights(
        into mtp: Q38MTPModel,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        guard FileManager.default.fileExists(atPath: resources.modelIndexURL.path) else {
            throw Q35Error.missingFiles([resources.modelIndexURL.lastPathComponent])
        }
        let data = try Data(contentsOf: resources.modelIndexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        let filenames = Self.embeddedMTPShardFilenames(weightMap: index.weightMap)
        guard !filenames.isEmpty else {
            throw Q35Error.missingFiles(["mtp.*"])
        }

        var arrays: [String: MLXArray] = [:]
        for filename in filenames {
            let shard = try SafetensorsStreamingLoader.loadArrays(
                url: resources.rootURL.appendingPathComponent(filename),
                where: { $0.hasPrefix("mtp.") }
            )
            arrays.merge(shard) { _, replacement in replacement }
        }

        let required = [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
        ]
        let missing = required.filter { arrays[$0] == nil }
        guard missing.isEmpty else {
            throw Q35Error.missingFiles(missing)
        }

        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            arrays,
            to: mtp,
            groupSize: groupSize,
            bits: bits,
            keyMapper: { key in
                Self.mapQ38MTPWeightKey(key) ?? "__unused__.\(key)"
            },
            mapper: { key, value in
                key.hasPrefix("__unused__.") ? [] : [(key, value)]
            }
        )
        Memory.clearCache()
    }

    static func mapQ38MTPWeightKey(_ key: String) -> String? {
        guard key.hasPrefix("mtp.") else { return nil }
        return String(key.dropFirst("mtp.".count))
    }

    func loadMTPWeights(
        into mtp: Q35MTPModel,
        baseModel: Q35Model,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        if let standalone = Self.standaloneMTPWeightsURL(resources: resources) {
            let metadata = try SafetensorsStreamingLoader.metadata(url: standalone)
            if metadata.keys.contains(where: { $0.hasSuffix(".scales") }) {
                let arrays = try MLX.loadArrays(url: standalone)
                if let weight = arrays["draft_lm_head.weight"],
                   let scales = arrays["draft_lm_head.scales"],
                   let biases = arrays["draft_lm_head.biases"] {
                    baseModel.installCoarseDraftHead(
                        weight: weight,
                        scales: scales,
                        biases: biases
                    )
                }
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    arrays,
                    to: mtp,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: { key in
                        Self.mapMTPWeightKey(key, standalone: true) ?? "__unused__.\(key)"
                    },
                    mapper: Self.mapMTPWeight
                )
            } else {
                try SafetensorsStreamingLoader.applyWeightsStreaming(
                    url: standalone,
                    to: mtp,
                    dtype: .bfloat16,
                    verify: .none,
                    include: { Self.mapMTPWeightKey($0, standalone: true) != nil },
                    mapper: { key, value in
                        guard let mapped = Self.mapMTPWeightKey(key, standalone: true) else { return [] }
                        return Self.mapMTPWeight(mapped, value)
                    },
                    batchSize: 32
                )
            }
            return
        }

        let data = try Data(contentsOf: resources.modelIndexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        let shardFilenames = Self.embeddedMTPShardFilenames(weightMap: index.weightMap)
        let checkpointUsesZeroCenteredNorms = try Self.checkpointUsesZeroCenteredRMSNorm(
            from: resources
        )
        for filename in shardFilenames {
            let arrays = try SafetensorsStreamingLoader.loadArrays(
                url: resources.rootURL.appendingPathComponent(filename),
                where: { Self.mapMTPWeightKey($0) != nil },
                dtype: .bfloat16
            )
            let updates = try Self.mappedMTPUpdates(
                arrays: arrays,
                expertCount: mtp.expertCount,
                checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
            )
            try mtp.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
            MLX.eval(updates.map(\.1))
            Memory.clearCache()
        }
    }
}
