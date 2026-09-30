import Foundation
import MLX
import MLXNN

/// Loads upstream safetensors directly, with explicit Conv3D layout transforms.
public enum FourDAnyoneModelLoader {
    public static let upstreamRevision = "8cd60c40d90882de07645cc435dcf24bc9b4fbd1"
    public static let modelRevision = "4c80e87b805a5f8461cf339cdbe2fb4249e585aa"
    public static let modelRepository = "AntResearch/4DAnyone"

    public static func loadTransformer(
        from url: URL,
        configuration: Wan2TransformerConfiguration = Wan2TransformerConfiguration(),
        dtype: DType = .bfloat16,
        computePrecision: FourDAnyoneComputePrecision = .float32
    ) throws -> FourDAnyoneTransformerModel {
        let model = FourDAnyoneTransformerModel(configuration: configuration, computePrecision: computePrecision)
        let pose = FourDAnyonePoseEncoder(outputChannels: configuration.hiddenSize)
        try validateCheckpoint(
            metadata: SafetensorsStreamingLoader.metadata(url: url),
            transformer: model, pose: pose, partition: .transformer
        )
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: url, to: model, dtype: dtype, verify: .noUnusedKeys,
            include: { !$0.hasPrefix("pose_encoder.") },
            mapper: { key, value in
                let mapped = isPatchWeight(key) ? value.reshaped(value.dim(0), -1) : value
                return [(nativeTransformerKey(key), mapped)]
            }, batchSize: 8
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadPoseEncoder(
        from url: URL,
        configuration: Wan2TransformerConfiguration = Wan2TransformerConfiguration(),
        dtype: DType = .bfloat16
    ) throws -> FourDAnyonePoseEncoder {
        let transformer = FourDAnyoneTransformerModel(configuration: configuration)
        let pose = FourDAnyonePoseEncoder(outputChannels: configuration.hiddenSize)
        try validateCheckpoint(
            metadata: SafetensorsStreamingLoader.metadata(url: url),
            transformer: transformer, pose: pose, partition: .pose
        )
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: url, to: pose, dtype: dtype, verify: .noUnusedKeys,
            include: { $0.hasPrefix("pose_encoder.") },
            mapper: { key, value in
                let name = String(key.dropFirst("pose_encoder.".count))
                if name.hasPrefix("conv_layers.") {
                    let components = name.split(separator: ".")
                    let index = Int(components[1])! / 2
                    let mapped = name.hasSuffix(".weight") ? value.transposed(0, 2, 3, 4, 1) : value
                    return [("convolutions.\(index).\(components[2])", mapped)]
                }
                return [(name, name == "final_proj.weight" ? value.reshaped(value.dim(0), -1) : value)]
            }, batchSize: 4
        )
        eval(pose.parameters().flattened().map(\.1))
        return pose
    }

    enum Partition { case transformer, pose }

    // Numeric Sequential children decode as arrays in MLX's parameter tree.
    // Give the native modules named children and map these exact upstream paths.
    static let sequentialKeys = [
        ("text_embedding.0.", "text_embedding.input."),
        ("text_embedding.2.", "text_embedding.output."),
        ("time_embedding.0.", "time_embedding.input."),
        ("time_embedding.2.", "time_embedding.output."),
        ("time_projection.1.", "time_projection.linear."),
        (".ffn.0.", ".ffn.input."),
        (".ffn.2.", ".ffn.output."),
    ]

    static func nativeTransformerKey(_ key: String) -> String {
        sequentialKeys.reduce(key) { $0.replacingOccurrences(of: $1.0, with: $1.1) }
    }

    static func upstreamTransformerKey(_ key: String) -> String {
        sequentialKeys.reduce(key) { $0.replacingOccurrences(of: $1.1, with: $1.0) }
    }

    static func isPatchWeight(_ key: String) -> Bool {
        key == "patch_embedding.weight"
            || key == "viewpack_embedding.proj_2x.weight"
            || key == "viewpack_embedding.proj_4x.weight"
    }

    static func transformerSchema(_ model: FourDAnyoneTransformerModel) -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
            let spatial: Int?
            switch key {
            case "patch_embedding.weight": spatial = 2
            case "viewpack_embedding.proj_2x.weight": spatial = 4
            case "viewpack_embedding.proj_4x.weight": spatial = 8
            default: spatial = nil
            }
            let shape = spatial.map {
                [model.configuration.hiddenSize, model.configuration.inputChannels, 1, $0, $0]
            } ?? value.shape
            return (upstreamTransformerKey(key), shape)
        })
    }

    static func poseSchema(_ model: FourDAnyonePoseEncoder) -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
            var name = key
            var shape = value.shape
            if key.hasPrefix("convolutions.") {
                let components = key.split(separator: ".")
                name = "conv_layers.\(Int(components[1])! * 2).\(components[2])"
                if key.hasSuffix(".weight") {
                    shape = [shape[0], shape[4], shape[1], shape[2], shape[3]]
                }
            } else if key == "final_proj.weight" {
                shape += [1, 1, 1]
            }
            return ("pose_encoder." + name, shape)
        })
    }

    /// Accepts a complete upstream checkpoint or the exact selected partition.
    static func validateCheckpoint(
        metadata: [String: SafetensorsStreamingLoader.TensorMetadata],
        transformer: FourDAnyoneTransformerModel,
        pose: FourDAnyonePoseEncoder,
        partition: Partition
    ) throws {
        let transformerKeys = transformerSchema(transformer)
        let poseKeys = poseSchema(pose)
        let selected = partition == .transformer ? transformerKeys : poseKeys
        let complete = transformerKeys.merging(poseKeys) { first, _ in first }
        let actualKeys = Set(metadata.keys)
        let expected: [String: [Int]]
        if actualKeys == Set(selected.keys) {
            expected = selected
        } else if actualKeys == Set(complete.keys) {
            expected = complete
        } else {
            let missing = Set(selected.keys).subtracting(actualKeys).sorted()
            let unexpected = actualKeys.subtracting(complete.keys).sorted()
            throw FourDAnyoneError.invalidCheckpoint(
                "Require a complete checkpoint or partition; missing \(missing.prefix(5)), unexpected \(unexpected.prefix(5))."
            )
        }
        for key in expected.keys.sorted() {
            guard let entry = metadata[key], entry.shape == expected[key],
                  [.bfloat16, .float16, .float32].contains(entry.dtype) else {
                throw FourDAnyoneError.invalidCheckpoint("Shape or floating-point dtype mismatch for \(key).")
            }
        }
    }

    public static func loadPromptContext(from url: URL) throws -> MLXArray {
        let tensors = try SafetensorsStreamingLoader.metadata(url: url)
        let metadata = try SafetensorsStreamingLoader.fileMetadata(url: url)
        let required = [
            "format": "fdanyone.prompt_context", "version": "2",
            "prompt": "视频中的人在做动作", "source_repo": "Wan-AI/Wan2.1-T2V-1.3B",
            "source_revision": "3f40b6dc4ca5c02dd23c9db74d9d2ccb82903b86",
        ]
        guard Set(tensors.keys) == ["context"],
              tensors["context"]?.shape == [1, 512, 4_096],
              tensors["context"]?.dtype == .bfloat16,
              required.allSatisfy({ metadata[$0.key] == $0.value }),
              ["text_encoder_sha256", "tokenizer_manifest_sha256"].allSatisfy({ key in
                  guard let hash = metadata[key] else { return false }
                  return hash.count == 64 && hash.allSatisfy { "0123456789abcdef".contains($0) }
              }) else {
            throw FourDAnyoneError.invalidCheckpoint("Prompt context shape, dtype, or provenance does not match the release.")
        }
        return try SafetensorsStreamingLoader.loadArrays(url: url)["context"]!
    }
}
