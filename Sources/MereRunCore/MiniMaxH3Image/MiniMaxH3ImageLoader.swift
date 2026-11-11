import Foundation
import MLX
import MLXNN

enum MiniMaxH3ImageLoader {
    static func transformer(_ resources: MiniMaxH3ImageResources) throws -> MiniMaxH3ImageTransformer {
        let configuration = try resources.configuration()
        let model = MiniMaxH3ImageTransformer(configuration: configuration.transformer)
        let native = try resources.mlxFormat()
        if let native { model.installModulationCache(try native.loadCache(resources, configuration: configuration.transformer)) }
        try loadShards(resources, component: "transformer", model: model,
            excludedParameters: ["final_layer.pdd_weight", "final_layer.pdd_bias"], nativeQ8: native != nil) { key, value in
            if key == "rope.inv_freq" || key.hasPrefix("final_layer.video_out.") { return [] }
            return MiniMaxH3ModelLoader.releasedBF16TransformerWeight(key: key, value: value, omitCachedAdaLNWeights: false)
        }
        let heads = try MLX.loadArrays(url: resources.rootURL.appending(path: "pdd_fused_heads.safetensors"))
        guard Set(heads.keys) == ["video_out.weight", "video_out.bias"],
              let weight = heads["video_out.weight"], let bias = heads["video_out.bias"],
              weight.dtype == .float32, bias.dtype == .float32 else {
            throw resources.invalid("Missing FP32 fused PDD output heads.")
        }
        try model.update(parameters: ModuleParameters.unflattened([
            ("final_layer.pdd_weight", weight), ("final_layer.pdd_bias", bias)
        ]), verify: [.shapeMismatch, .noUnusedKeys])
        return model
    }

    static func conditioner(_ resources: MiniMaxH3ImageResources) throws -> QwenVLEncoder {
        let encoder = MiniMaxH3ModelLoader.makeConditioner()
        // Layer-50 activation conditioning never evaluates the stack's final output norm.
        try loadShards(resources, component: "text_encoder", model: encoder,
                       excludedParameters: ["textEncoder.encoder.norm.weight"], nativeQ8: try resources.mlxFormat() != nil) { key, value in
            MiniMaxH3ModelLoader.conditionerWeight(key: MiniMaxH3ModelLoader.conditionerWeightKey(key), value: value)
        }
        return encoder
    }

    static func vae(_ resources: MiniMaxH3ImageResources) throws -> MiniMaxH3ImageVAE {
        let model = MiniMaxH3ImageVAE(configuration: try resources.vaeConfiguration())
        let raw = try MLX.loadArrays(url: resources.rootURL.appending(path: "vae/diffusion_pytorch_model.safetensors"))
        let mapped = raw.flatMap { vaeWeight(key: $0.key, value: $0.value) }
        try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        return model
    }

    static func vaeWeight(key rawKey: String, value: MLXArray) -> [(String, MLXArray)] {
        if rawKey == "decoder.mask_token" { return [] }
        let key = rawKey.replacingOccurrences(of: "decoder.x_embedder.", with: "decoder.proj_in.")
            .replacingOccurrences(of: ".ff.w1.", with: ".ff.linear_in.")
            .replacingOccurrences(of: ".ff.w2.", with: ".ff.linear_out.")
            .replacingOccurrences(of: "encoder.down.", with: "encoder.down_blocks.")
            .replacingOccurrences(of: ".block.", with: ".resnets.")
            .replacingOccurrences(of: ".nin_shortcut.", with: ".conv_shortcut.")
            .replacingOccurrences(of: ".downsample.conv.", with: ".downsamplers.0.conv.")
        if key.contains(".attn.to_qkv.") {
            let shape = Array(value.shape.dropFirst())
            let grouped = value.reshaped([16, 3, 64] + shape)
            let parts = MLX.split(grouped, parts: 3, axis: 1).map { $0.reshaped([1_024] + shape) }
            return [(key, MLX.concatenated(parts, axis: 0))]
        }
        if value.ndim == 5 { return [(key, value.transposed(0, 2, 3, 4, 1))] }
        if value.ndim == 4 { return [(key, value.transposed(0, 2, 3, 1))] }
        return [(key, value)]
    }

    private static func loadShards(_ resources: MiniMaxH3ImageResources, component: String, model: Module,
        excludedParameters: Set<String> = [], nativeQ8: Bool = false, mapper: (String, MLXArray) -> [(String, MLXArray)]) throws {
        let index = try resources.index(component)
        var loaded = Set<String>()
        for shard in index.shardFilenames {
            try Task.checkCancellation()
            let arrays = try MLX.loadArrays(url: resources.rootURL.appending(path: "\(component)/\(shard)"))
            guard Set(arrays.keys) == Set(index.weightMap.filter { $0.value == shard }.keys) else {
                throw resources.invalid("\(component)/\(shard) differs from its tensor index.")
            }
            let mapped = nativeQ8 ? arrays.map { ($0.key, $0.value) } : arrays.flatMap { mapper($0.key, $0.value) }
            for (key, _) in mapped {
                guard loaded.insert(key).inserted else { throw resources.invalid("Duplicate tensor: \(key)") }
            }
            if nativeQ8 {
                try MiniMaxH3ImageMLXFormat.apply(arrays, to: model)
            } else {
                try model.update(parameters: ModuleParameters.unflattened(mapped), verify: [.shapeMismatch, .noUnusedKeys])
            }
        }
        let required = Set(model.parameters().flattened().map(\.0).filter {
            !(nativeQ8 && component == "transformer" && MiniMaxH3ImageMLXFormat.isCacheWeight($0))
        }).subtracting(excludedParameters)
        guard loaded == required else { throw resources.invalid("Incomplete \(component) tensor coverage: \(required.subtracting(loaded).sorted()).") }
    }
}
