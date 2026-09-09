import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    static func embeddedMTPShardFilenames(weightMap: [String: String]) -> [String] {
        Array(Set(weightMap.compactMap { key, filename in
            key.hasPrefix("mtp.") ? filename : nil
        })).sorted()
    }

    static func mapMTPWeightKey(_ key: String, standalone: Bool = false) -> String? {
        if key.hasPrefix("mtp.") {
            return String(key.dropFirst("mtp.".count))
        }
        guard standalone else { return nil }
        let barePrefixes = [
            "fc.",
            "layers.",
            "norm.",
            "pre_fc_norm_embedding.",
            "pre_fc_norm_hidden.",
        ]
        return barePrefixes.contains(where: { key.hasPrefix($0) }) ? key : nil
    }

    static func isMTPRMSNormWeight(_ key: String) -> Bool {
        key == "norm.weight"
            || key == "pre_fc_norm_embedding.weight"
            || key == "pre_fc_norm_hidden.weight"
            || key.hasSuffix(".input_layernorm.weight")
            || key.hasSuffix(".post_attention_layernorm.weight")
            || key.hasSuffix(".self_attn.q_norm.weight")
            || key.hasSuffix(".self_attn.k_norm.weight")
    }

    static func mapMTPWeight(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard !key.hasPrefix("__unused__.") else { return [] }
        if isMTPRMSNormWeight(key) {
            return [(key, value - MLXArray(1.0).asType(value.dtype))]
        }
        return [(key, value)]
    }

    static func mappedMTPUpdates(
        arrays: [String: MLXArray],
        expertCount: Int,
        checkpointUsesZeroCenteredNorms: Bool = false
    ) throws -> [(String, MLXArray)] {
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(arrays.count)
        let individualExpertMarker = ".mlp.experts."

        for (key, value) in arrays {
            guard let mapped = mapMTPWeightKey(key) else { continue }
            if expertCount > 0,
               mapped.contains(individualExpertMarker),
               mapped.contains(".weight") {
                continue
            }
            if isMTPRMSNormWeight(mapped) {
                updates.append((
                    mapped,
                    normalizedRMSNormWeight(
                        value,
                        checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
                    )
                ))
            } else {
                updates.append(contentsOf: mapMTPWeight(mapped, value))
            }
        }

        guard expertCount > 0 else { return updates }
        let prefix = "mtp.layers.0.mlp.experts"
        func required(_ expert: Int, _ projection: String) throws -> MLXArray {
            let key = "\(prefix).\(expert).\(projection).weight"
            guard let value = arrays[key] else {
                throw Q35Error.missingFiles([key])
            }
            return value
        }

        var gateUpExperts: [MLXArray] = []
        var downExperts: [MLXArray] = []
        gateUpExperts.reserveCapacity(expertCount)
        downExperts.reserveCapacity(expertCount)
        for expert in 0..<expertCount {
            let gate = try required(expert, "gate_proj")
            let up = try required(expert, "up_proj")
            gateUpExperts.append(MLX.concatenated([gate, up], axis: 0))
            downExperts.append(try required(expert, "down_proj"))
        }

        let gateUp = MLX.stacked(gateUpExperts, axis: 0)
        let down = MLX.stacked(downExperts, axis: 0)
        MLX.eval(gateUp, down)
        updates.append(("layers.0.mlp.experts.gate_up_proj", gateUp))
        updates.append(("layers.0.mlp.experts.down_proj", down))
        return updates
    }

    static func mapTextWeightKey(_ key: String) -> String? {
        // PLE tables are mapped separately and never installed as MLX weights.
        if key.contains(".ple.ple_embedding.ngram_embedding.") { return nil }
        if key.hasPrefix("lm_head.") {
            return key
        }
        if key.hasPrefix("model.language_model.") {
            return mapLanguageModelWeightSuffix(String(key.dropFirst("model.language_model.".count)))
        }
        if key.hasPrefix("language_model.") {
            return mapLanguageModelWeightSuffix(String(key.dropFirst("language_model.".count)))
        }
        return nil
    }

    private static func mapLanguageModelWeightSuffix(_ suffix: String) -> String {
        if suffix.hasPrefix("model.") || suffix.hasPrefix("lm_head.") {
            return suffix
        }
        return "model.\(suffix)"
    }

    static func normalizeMappedExpertWeightKey(_ key: String) -> String {
        let expertDownSuffix = ".mlp.experts.down_proj"
        if key.hasSuffix(expertDownSuffix) {
            return String(key.dropLast(expertDownSuffix.count)) + ".mlp.switch_mlp.down_proj.weight"
        }
        return key
    }

    static func normalizedLinearAttentionConv1DWeight(_ value: MLXArray) -> MLXArray {
        guard value.ndim == 3, value.dim(1) == 1, value.dim(2) > 1 else {
            return value
        }
        let transposed = value.transposed(0, 2, 1)
        return transposed.reshaped(-1).reshaped(transposed.shape)
    }

    static func isOffsetRMSNormWeight(_ key: String) -> Bool {
        key.hasSuffix(".input_layernorm.weight")
            || key.hasSuffix(".post_attention_layernorm.weight")
            || key.hasSuffix(".self_attn.q_norm.weight")
            || key.hasSuffix(".self_attn.k_norm.weight")
            || key == "model.norm.weight"
    }

    static func normalizedRMSNormWeight(
        _ value: MLXArray,
        checkpointUsesZeroCenteredNorms: Bool
    ) -> MLXArray {
        if checkpointUsesZeroCenteredNorms {
            return value
        }
        return value - MLXArray(1.0).asType(value.dtype)
    }

    static func checkpointUsesZeroCenteredRMSNorm(from resources: Q35Resources) throws -> Bool {
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            let data = try Data(contentsOf: resources.modelIndexURL)
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
            let weightKeys = Array(index.weightMap.keys)
            if checkpointUsesZeroCenteredRMSNorm(weightKeys: weightKeys, tensorShapes: [:]) {
                return true
            }
            guard let convEntry = index.weightMap.first(where: { key, _ in
                key.hasSuffix(".linear_attn.conv1d.weight")
            }) else {
                return false
            }
            let shardURL = resources.rootURL.appending(path: convEntry.value)
            let metadata = try SafetensorsStreamingLoader.metadata(url: shardURL)
            return checkpointUsesZeroCenteredRMSNorm(
                weightKeys: weightKeys,
                tensorShapes: metadata.mapValues(\.shape)
            )
        }

        let metadata = try SafetensorsStreamingLoader.metadata(url: resources.modelWeightsURL)
        return checkpointUsesZeroCenteredRMSNorm(
            weightKeys: Array(metadata.keys),
            tensorShapes: metadata.mapValues(\.shape)
        )
    }

    static func checkpointUsesZeroCenteredRMSNorm(
        weightKeys: [String],
        tensorShapes: [String: [Int]]
    ) -> Bool {
        if weightKeys.contains(where: { $0.hasPrefix("mtp.") || $0.contains(".mtp.") }) {
            return true
        }
        return tensorShapes.contains { key, shape in
            key.hasSuffix(".linear_attn.conv1d.weight")
                && shape.count == 3
                && shape.last != 1
        }
    }

    static func splitMappedExpertGateUpWeight(_ key: String, _ value: MLXArray) -> [(String, MLXArray)]? {
        let expertGateUpSuffix = ".mlp.experts.gate_up_proj"
        guard key.hasSuffix(expertGateUpSuffix), value.ndim == 3 else {
            return nil
        }

        let fusedDim = value.dim(1)
        guard fusedDim > 0, fusedDim.isMultiple(of: 2) else {
            return nil
        }

        let intermediate = fusedDim / 2
        let base = String(key.dropLast(expertGateUpSuffix.count)) + ".mlp.switch_mlp"
        return [
            ("\(base).gate_proj.weight", value[0..., 0..<intermediate, 0...]),
            ("\(base).up_proj.weight", value[0..., intermediate..., 0...]),
        ]
    }

    static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }
}
