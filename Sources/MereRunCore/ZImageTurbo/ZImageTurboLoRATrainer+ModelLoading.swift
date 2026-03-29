import Foundation
import MLX
import MLXNN
import MLXRandom

extension ZImageTurboLoRATrainer {
    // MARK: - Assistant LoRA Resolution

    /// Resolves an assistant LoRA path.
    /// Supports local paths, `models/*` R2 keys, public URLs, and legacy adapter aliases.
    static func resolveAssistantLoRAPath(_ path: String) async throws -> URL {
        let localURL = URL(fileURLWithPath: path).standardizedFileURL
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        return try await LoRAWeightLoader.resolveRemoteReference(path)
    }

    // MARK: - Model Loading

    static func loadTextEncoderWeights(
        from resources: ZImageTurboResources,
        into model: QwenTextEncoder,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) throws {
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.textEncoderWeightsIndexURL,
            singleURL: resources.textEncoderWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                if key.hasPrefix("model.") {
                    let remainder = String(key.dropFirst("model.".count))
                    return [("encoder.\(remainder)", value)]
                }
                return [(key, value)]
            },
            keyMapper: { key in
                if key.hasPrefix("model.") {
                    return "encoder." + String(key.dropFirst("model.".count))
                }
                if key.hasPrefix("encoder.") {
                    return key
                }
                return "encoder.\(key)"
            },
            quantization: quantization
        )
    }

    static func loadTransformerWeights(
        from resources: ZImageTurboResources,
        into model: ZImageTransformer2DModel,
        quantization: ModelWeightsLoader.QuantizationParams?
    ) throws {
        let keyMapper: (String) -> String = { key in
            var mapped = key
            if key.contains("t_embedder.linear1") {
                mapped = key.replacingOccurrences(of: "t_embedder.linear1", with: "t_embedder.mlp.0")
            } else if key.contains("t_embedder.linear2") {
                mapped = key.replacingOccurrences(of: "t_embedder.linear2", with: "t_embedder.mlp.2")
            } else if key.contains("all_final_layer") && key.contains("adaLN_modulation.0.") {
                mapped = key.replacingOccurrences(of: "adaLN_modulation.0.", with: "adaLN_modulation.1.")
            }
            return mapped
        }
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.transformerWeightsIndexURL,
            singleURL: resources.transformerWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.noUnusedKeys, .shapeMismatch],
            keyMapper: keyMapper,
            quantization: quantization
        )
    }

    static func loadVAEWeights(
        from resources: ZImageTurboResources,
        into model: AutoencoderKL
    ) throws {
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.noUnusedKeys, .shapeMismatch],
            mapper: { key, value in
                let maybeConverted = value.ndim == 4 ? HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value) : value
                return [(key, maybeConverted)]
            }
        )
    }

    static func applyAssistantLoRA(
        from path: String,
        to transformer: ZImageTransformer2DModel
    ) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(
                domain: "ZImageTurboLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Assistant LoRA not found: \(url.path)"]
            )
        }

        let weights = try MLX.loadArrays(url: url)
        let leafModules = transformer.leafModules().flattened()
        var moduleByPath: [String: Module] = [:]
        moduleByPath.reserveCapacity(leafModules.count)
        for (path, module) in leafModules {
            moduleByPath[path] = module
        }

        func normalizePath(_ key: String) -> String {
            var path = key
            if path.hasPrefix("diffusion_model.") {
                path.removeFirst("diffusion_model.".count)
            }
            if path.hasPrefix("transformer.") {
                path.removeFirst("transformer.".count)
            }
            return path
        }

        let downSuffixes = [".lora_A.weight", ".lora_A.default.weight"]
        let upSuffixes = [".lora_B.weight", ".lora_B.default.weight"]

        var downKeys: [String: String] = [:]
        var upKeys: [String: String] = [:]

        for key in weights.keys {
            if let suffix = downSuffixes.first(where: { key.hasSuffix($0) }) {
                let base = String(key.dropLast(suffix.count))
                downKeys[base] = key
                continue
            }
            if let suffix = upSuffixes.first(where: { key.hasSuffix($0) }) {
                let base = String(key.dropLast(suffix.count))
                upKeys[base] = key
            }
        }

        var mergedCount = 0
        var replacements: [String: Module] = [:]
        for (base, downKey) in downKeys {
            guard let upKey = upKeys[base],
                  let down = weights[downKey],
                  let up = weights[upKey] else { continue }

            let rank = max(down.shape.first ?? 1, 1)
            let alphaKey = "\(base).alpha"
            let alpha: Float = {
                if let alphaTensor = weights[alphaKey] {
                    return alphaTensor.item(Float.self)
                }
                return Float(rank)
            }()
            let scale = alpha / Float(rank)

            let modulePath = normalizePath(base)
            guard let module = moduleByPath[modulePath] else { continue }

            if module is QuantizedLinear {
                throw NSError(
                    domain: "ZImageTurboLoRATrainer",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Assistant LoRA merge is not supported for quantized layers (\(modulePath)). Use a non-quantized model."]
                )
            }

            guard let linear = module as? Linear else { continue }

            let delta = MLX.matmul(up.asType(.float32), down.asType(.float32)) * MLXArray(scale)
            let newWeight = linear.weight + delta.asType(linear.weight.dtype)
            let updated = Linear(weight: newWeight, bias: linear.bias)
            replacements[modulePath] = updated
            mergedCount += 1
        }

        guard mergedCount > 0 else {
            throw NSError(
                domain: "ZImageTurboLoRATrainer",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Assistant LoRA merge found no matching layers."]
            )
        }

        if !replacements.isEmpty {
            applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        }

        print("[ZImageLoRATrainer] Merged assistant LoRA into \(mergedCount) layers")
    }
}
