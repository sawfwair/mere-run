import Foundation
import MLX
import MLXNN

extension MiniMaxH3TurboAdapter {
    static func sourceFormat(at url: URL) throws -> SourceFormat {
        let format = try SafetensorsStreamingLoader.fileMetadata(url: url)["format"]
        if format == fastVideoFormat {
            return .fastVideo
        }
        if format == fastH3PremergedFormat {
            return .fastH3Premerged
        }
        let keys = try SafetensorsStreamingLoader.metadata(url: url).keys
        if keys.contains(where: { $0.hasSuffix(".lora_A.default.weight") }) {
            return .lightX2V
        }
        if keys.contains(where: { $0.hasSuffix(".lora_A.weight") }) {
            return .runtime
        }
        throw AdapterError.unrecognizedFormat(url)
    }

    static func target(
        for sourcePath: String,
        sourceFormat: SourceFormat
    ) throws -> LightX2VTarget {
        guard sourceFormat != .runtime else {
            return LightX2VTarget(modulePath: sourcePath, qkvBranch: nil)
        }
        precondition(sourceFormat != .fastH3Premerged)

        let runtimePrefix: String
        let remainder: Substring
        if sourcePath.hasPrefix("transformer_blocks.") {
            runtimePrefix = "blocks."
            remainder = sourcePath.dropFirst("transformer_blocks.".count)
        } else if sourcePath.hasPrefix("token_refiner.refiner_blocks.") {
            runtimePrefix = "token_refiner.blocks."
            remainder = sourcePath.dropFirst("token_refiner.refiner_blocks.".count)
        } else {
            throw AdapterError.unsupportedSourceModule(sourcePath)
        }
        let normalized = runtimePrefix + remainder

        for (suffix, branch) in [
            (".attn.to_q", QKVBranch.query),
            (".attn.to_k", QKVBranch.key),
            (".attn.to_v", QKVBranch.value),
        ] where normalized.hasSuffix(suffix) {
            return LightX2VTarget(
                modulePath: String(normalized.dropLast(suffix.count)) + ".attn.qkv_proj",
                qkvBranch: branch
            )
        }

        for (suffix, replacement) in [
            (".attn.to_out.0", ".attn.out_proj"),
            (".ff.net.0.proj", ".mlp.fc1"),
            (".ff.net.2", ".mlp.fc2"),
            (".adaln_proj.linear", ".adaln_proj.linear"),
        ] where normalized.hasSuffix(suffix) {
            return LightX2VTarget(
                modulePath: String(normalized.dropLast(suffix.count)) + replacement,
                qkvBranch: nil
            )
        }
        throw AdapterError.unsupportedSourceModule(sourcePath)
    }

    static func scaledUp(
        _ up: MLXArray,
        down: MLXArray,
        sourceFormat: SourceFormat,
        lightX2VAlpha: Float
    ) -> MLXArray {
        guard sourceFormat == .lightX2V else { return up }
        let scale = MLXArray(lightX2VAlpha / Float(down.dim(0))).asType(up.dtype)
        return up * scale
    }

    static func applyFastVideoDifferences(
        url: URL,
        to transformer: MiniMaxH3Transformer,
        strength: Float,
        omittingCacheCoveredParameters: Bool
    ) throws -> Int {
        var seenCount = 0
        var targetPaths: Set<String> = []
        let count = try SafetensorsStreamingLoader.forEachTensor(
            url: url,
            where: { $0.hasSuffix(".diff") || $0.hasSuffix(".diff_b") }
        ) { sourceKey, difference in
            seenCount += 1
            let targetPath = try fastVideoDifferenceTarget(sourceKey)
            guard targetPaths.insert(targetPath).inserted else {
                throw AdapterError.duplicateTarget(targetPath)
            }
            if omittingCacheCoveredParameters,
               (targetPath.hasPrefix("time_embedder.")
                || (targetPath.hasPrefix("blocks.") && targetPath.contains(".adaln_proj."))
                || targetPath.hasPrefix("final_layer.adaln_proj.")) {
                return
            }
            let parameters = transformer.parameters().flattened()
            guard let base = parameters.first(where: { $0.0 == targetPath })?.1 else {
                throw AdapterError.missingTargetParameter(targetPath)
            }
            guard base.shape == difference.shape else {
                throw AdapterError.targetShapeMismatch(
                    targetPath,
                    expected: base.shape,
                    actual: difference.shape
                )
            }
            let updated = base + difference.asType(base.dtype)
                * MLXArray(strength).asType(base.dtype)
            transformer.update(parameters: ModuleParameters.unflattened([(targetPath, updated)]))
            MLX.eval(updated)
            Memory.clearCache()
        }
        precondition(count == seenCount)
        return count
    }

    static func fastVideoDifferenceTarget(_ sourceKey: String) throws -> String {
        let suffix: String
        let parameter: String
        if sourceKey.hasSuffix(".diff_b") {
            suffix = ".diff_b"
            parameter = ".bias"
        } else if sourceKey.hasSuffix(".diff") {
            suffix = ".diff"
            parameter = ".weight"
        } else {
            throw AdapterError.unsupportedSourceModule(sourceKey)
        }
        let source = String(sourceKey.dropLast(suffix.count))
        let mapped: String
        switch source {
        case "proj_in": mapped = "video_patch_proj"
        case "audio_proj_in": mapped = "audio_patch_proj"
        case "context_embedder": mapped = "condition_proj"
        case "proj_out": mapped = "final_layer.video_out"
        case "audio_proj_out": mapped = "final_layer.audio_out"
        case "norm_out.norm": mapped = "final_layer.norm"
        case "norm_out.linear": mapped = "final_layer.adaln_proj.linear"
        case "time_embedder.linear_1": mapped = "time_embedder.proj_in"
        case "time_embedder.linear_2": mapped = "time_embedder.proj_out"
        default:
            if source.hasPrefix("transformer_blocks.") {
                mapped = "blocks." + String(source.dropFirst("transformer_blocks.".count))
            } else {
                throw AdapterError.unsupportedSourceModule(sourceKey)
            }
        }
        return mapped + parameter
    }

}
