import Foundation
import MLX

public enum QwenTextLoRAWeightLoader {
    private static let loraPatterns: [(down: String, up: String)] = [
        (".lora_down.", ".lora_up."),
        (".lora_A.", ".lora_B."),
        (".lora_a", ".lora_b"),
    ]

    public static func load(from lora: LoRA) async throws -> LoRAWeights {
        let url = try await LoRAWeightLoader.resolveURL(for: lora)
        return try load(from: url)
    }

    public static func load(from url: URL) throws -> LoRAWeights {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoRAError.fileNotFound(url.path)
        }

        let allWeights = try MLX.loadArrays(url: url)
        let keys = allWeights.keys.sorted()

        var loraWeights: [String: (down: MLXArray, up: MLXArray)] = [:]
        var processedKeys = Set<String>()

        for key in keys {
            if processedKeys.contains(key) { continue }
            guard let (downKey, upKey, baseKey) = resolveKeyPair(key) else { continue }
            guard let downWeight = allWeights[downKey],
                  let upWeight = allWeights[upKey] else {
                continue
            }

            let mappedKey = mapBaseKey(baseKey)
            loraWeights[mappedKey] = (down: downWeight, up: upWeight)

            processedKeys.insert(downKey)
            processedKeys.insert(upKey)
        }

        guard !loraWeights.isEmpty else {
            throw LoRAError.noWeightPairs
        }

        let rank = inferRank(from: loraWeights)
        let alpha = loadAlpha(from: url.deletingLastPathComponent(), fallback: Float(rank))

        return LoRAWeights(weights: loraWeights, rank: rank, alpha: alpha)
    }

    private static func resolveKeyPair(_ key: String) -> (downKey: String, upKey: String, baseKey: String)? {
        for (downPattern, upPattern) in loraPatterns {
            if key.contains(downPattern) {
                guard let base = extractBaseKey(key, pattern: downPattern) else { return nil }
                let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
                return (downKey: key, upKey: upKey, baseKey: base)
            }
            if key.contains(upPattern) {
                guard let base = extractBaseKey(key, pattern: upPattern) else { return nil }
                let downKey = key.replacingOccurrences(of: upPattern, with: downPattern)
                return (downKey: downKey, upKey: key, baseKey: base)
            }
        }

        return nil
    }

    private static func extractBaseKey(_ key: String, pattern: String) -> String? {
        if pattern.hasSuffix(".lora_a") || pattern.hasSuffix(".lora_b") {
            guard key.hasSuffix(pattern) else { return nil }
            return String(key.dropLast(pattern.count))
        }

        guard let range = key.range(of: pattern) else { return nil }
        return String(key[..<range.lowerBound])
    }

    private static func mapBaseKey(_ key: String) -> String {
        var mapped = key
        let prefixesToRemove = [
            "base_model.model.",
            "model.",
            "encoder.",
        ]

        for prefix in prefixesToRemove where mapped.hasPrefix(prefix) {
            mapped = String(mapped.dropFirst(prefix.count))
            break
        }

        if mapped.hasPrefix("encoder.") {
            return mapped
        }
        return "encoder." + mapped
    }

    private static func inferRank(from weights: [String: (down: MLXArray, up: MLXArray)]) -> Int {
        for (_, pair) in weights {
            let downShape = pair.down.shape
            if downShape.count == 2 {
                return min(downShape[0], downShape[1])
            }
        }
        return 16
    }

    private static func loadAlpha(from directory: URL, fallback: Float) -> Float {
        let configPath = directory.appendingPathComponent("adapter_config.json")

        guard FileManager.default.fileExists(atPath: configPath.path),
              let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return fallback
        }

        let alphaKeys = ["lora_alpha", "alpha", "network_alpha"]
        for key in alphaKeys {
            if let alpha = json[key] as? NSNumber {
                return alpha.floatValue
            }
        }

        if let params = json["lora_parameters"] as? [String: Any],
           let alpha = params["alpha"] as? NSNumber {
            return alpha.floatValue
        }

        return fallback
    }
}
