import Foundation
import MereRunImageModels
import MLX
import MLXNN

enum NemotronImageLoader {
    static func generator(at root: URL, config: NemotronHConfig) throws -> NemotronOmniCausalLM {
        let manifest = try MereRunModelManifest.loadIfPresent(from: root)
        let bits = try NemotronImageQuantization.expertBits(manifest)
        let community = NemotronImageCommunityPack.isPresent(at: root)
        if community {
            try NemotronImageCommunityPack.checkRecipe(at: root, bits: bits ?? 0)
        }
        let model = NemotronOmniCausalLM(config: config, expertBits: bits)
        if manifest?.quantization?.scheme == NemotronImageQuantization.mixedScheme {
            // The dense recipe covers Mamba, attention, shared experts and the output head.
            // Embeddings, router parameters and the decoder retain their baseline precision.
            quantize(model: model, groupSize: 64, bits: 8, mode: .affine) { _, module in
                module is Linear
            }
        }
        let index = try bits == nil ? releasedIndex(at: root) : nil
        try visitModules(model) { module, prefix in
            let weights: [(String, MLXArray)]
            if let index {
                weights = try releasedWeights(at: root, config: config, index: index, module: module, prefix: prefix)
            } else if community {
                let arrays = try NemotronImageCommunityPack.weights(at: root, prefix: prefix, module: module)
                try NemotronImageQuantization.validate(arrays, for: module)
                weights = Array(arrays)
            } else {
                let file = NemotronImageQuantization.filename(prefix)
                let arrays = try MLX.loadArrays(url: root.appending(path: file))
                try NemotronImageQuantization.validate(arrays, for: module)
                weights = Array(arrays)
            }
            try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        }
        return model
    }

    static func visitModules(_ model: NemotronOmniCausalLM, _ visit: (Module, String) throws -> Void) throws {
        try visit(model.backbone.embeddings, "backbone.embeddings.")
        for (number, block) in model.backbone.layers.enumerated() {
            try Task.checkCancellation()
            try visit(block, "backbone.layers.\(number).")
            Memory.clearCache()
        }
        try visit(model.backbone.finalNorm, "backbone.norm_f.")
        try visit(model.lmHead, "lm_head.")
    }

    static func releasedIndex(at root: URL) throws -> HFSafetensorsIndex {
        let index = try NemotronImageResources.read(HFSafetensorsIndex.self,
            at: root.appending(path: "model.safetensors.index.json"))
        guard Set(index.shardFilenames) == Set(NemotronImageResources.shards) else {
            throw NemotronImageResources.invalid("Expected the 32 released Nemotron image generator shards.")
        }
        return index
    }

    static func releasedWeights(at root: URL, config: NemotronHConfig, index: HFSafetensorsIndex,
                                module: Module, prefix: String) throws -> [(String, MLXArray)] {
        var sources: [String: [String]] = [:]
        for (name, _) in module.parameters().flattened() {
            let fullName = prefix + name
            if let split = fullName.range(of: ".mixer.experts.") {
                sources[name] = (0..<config.nRoutedExperts).map {
                    String(fullName[..<split.upperBound]) + "\($0)." + fullName[split.upperBound...]
                }
            } else {
                sources[name] = [fullName]
            }
        }
        let byShard = try Dictionary(grouping: sources.values.joined()) { key in
            guard let filename = index.weightMap[key] else { throw missing(key) }
            return filename
        }
        var weights: [String: MLXArray] = [:]
        for (filename, keys) in byShard {
            let shard = try MLX.loadArrays(url: root.appending(path: filename))
            for key in keys {
                guard let tensor = shard[key], tensor.dtype.isFloatingPoint else { throw missing(key) }
                weights[key] = tensor.asType(.bfloat16)
            }
            // Release FP32 source buffers before opening the next shard.
            eval(keys.compactMap { weights[$0] })
        }
        var updates: [(String, MLXArray)] = []
        for (name, parameter) in module.parameters().flattened() {
            let tensors = try (sources[name] ?? []).map { key in
                guard let tensor = weights.removeValue(forKey: key) else { throw missing(key) }
                return tensor
            }
            let value: MLXArray
            if tensors.count == 1 {
                let tensor = tensors[0]
                if name.hasSuffix("conv1d.weight") {
                    guard tensor.ndim == 3 else { throw missing(prefix + name) }
                    value = tensor.transposed(0, 2, 1)
                } else {
                    value = tensor
                }
            } else {
                let shape = Array(parameter.shape.dropFirst())
                guard tensors.count == config.nRoutedExperts, tensors.allSatisfy({ $0.shape == shape }) else {
                    throw NemotronImageResources.invalid("Unexpected expert bank shape for \(prefix)\(name).")
                }
                value = stacked(tensors)
            }
            guard value.shape == parameter.shape else {
                throw NemotronImageResources.invalid("Unexpected shape for \(prefix)\(name).")
            }
            eval(value)
            updates.append((name, value))
        }
        return updates
    }

    static func decoder(at root: URL) throws -> L24SQDecoder {
        try NemotronImageResources.DecoderConfig.validate(at: root)
        let model = L24SQDecoder()
        let weights = try MLX.loadArrays(url: root.appending(path: "model.safetensors"))
        let updates = try model.parameters().flattened().map { key, _ in
            let source: String
            if key == "embedding.weight" { source = "quantize." + key }
            else if key.hasPrefix("post_quant_embed.") { source = key }
            else { source = "decoder." + key }
            guard let value = weights[source] else { throw missing(source) }
            return (key, value.asType(.bfloat16))
        }
        try model.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
        return model
    }

    private static func missing(_ key: String) -> ImageGenerationIssue {
        NemotronImageResources.invalid("Missing or invalid checkpoint tensor \(key).")
    }
}
