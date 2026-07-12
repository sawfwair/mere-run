import Foundation
@preconcurrency import MLX
import MLXNN

public enum DepthAnything3SmallCheckpoint {
    public static let repository = "depth-anything/DA3-SMALL"
    public static let revision = "e08cab65ca0ec38e7826075418411ab90cab4da3"
    public static let license = "Apache-2.0"
    public static let upstreamSourceRepository = "ByteDance-Seed/Depth-Anything-3"
    public static let upstreamSourceRevision = "41736238f5bced4debf3f2a12375d2466874866d"
    public static let artifact = ModelArtifactPin(
        filename: "model.safetensors",
        byteCount: 137_248_940,
        sha256: "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
    )
    public static let configurationArtifact = ModelArtifactPin(
        filename: "config.json",
        byteCount: 1_202,
        sha256: "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
    )
    public static let tensorCount = 437
    public static let scalarCount = 34_299_463
}

public enum DepthAnything3WeightError: Error, Equatable, LocalizedError, Sendable {
    case sizeMismatch(expected: Int64, actual: Int64)
    case checksumMismatch(expected: String, actual: String)
    case tensorCountMismatch(expected: Int, actual: Int)
    case scalarCountMismatch(expected: Int, actual: Int)
    case nonFloat32Tensor(String)
    case invalidPrefix(String)
    case targetTensorSetMismatch(missing: [String], unexpected: [String])
    case shapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .sizeMismatch(let expected, let actual):
            "DA3-Small checkpoint size mismatch: expected \(expected) bytes, found \(actual)."
        case .checksumMismatch(let expected, let actual):
            "DA3-Small checkpoint checksum mismatch: expected \(expected), found \(actual)."
        case .tensorCountMismatch(let expected, let actual):
            "DA3-Small tensor inventory mismatch: expected \(expected), found \(actual)."
        case .scalarCountMismatch(let expected, let actual):
            "DA3-Small scalar inventory mismatch: expected \(expected), found \(actual)."
        case .nonFloat32Tensor(let name):
            "DA3-Small tensor '\(name)' is not the pinned float32 dtype."
        case .invalidPrefix(let name):
            "DA3-Small tensor '\(name)' does not use the authoritative 'model.' prefix."
        case .targetTensorSetMismatch(let missing, let unexpected):
            "DA3-Small mapped tensor set differs from the native graph (missing: \(missing), unexpected: \(unexpected))."
        case .shapeMismatch(let name, let expected, let actual):
            "DA3-Small tensor '\(name)' shape mismatch: expected \(expected), found \(actual)."
        }
    }
}

public enum DepthAnything3Weights {
    public static func validate(
        model: DepthAnything3Model,
        safetensorsURL: URL,
        verifyChecksum: Bool = true
    ) throws {
        let byteCount = try ModelArtifactPin.fileByteCount(safetensorsURL)
        guard byteCount == DepthAnything3SmallCheckpoint.artifact.byteCount else {
            throw DepthAnything3WeightError.sizeMismatch(
                expected: DepthAnything3SmallCheckpoint.artifact.byteCount,
                actual: byteCount
            )
        }
        if verifyChecksum {
            let checksum = try ModelArtifactPin.fileSHA256(safetensorsURL.resolvingSymlinksInPath())
            guard checksum == DepthAnything3SmallCheckpoint.artifact.sha256 else {
                throw DepthAnything3WeightError.checksumMismatch(
                    expected: DepthAnything3SmallCheckpoint.artifact.sha256,
                    actual: checksum
                )
            }
        }

        let metadata = try SafetensorsStreamingLoader.metadata(url: safetensorsURL)
        guard metadata.count == DepthAnything3SmallCheckpoint.tensorCount else {
            throw DepthAnything3WeightError.tensorCountMismatch(
                expected: DepthAnything3SmallCheckpoint.tensorCount,
                actual: metadata.count
            )
        }
        let scalarCount = metadata.values.reduce(0) { partial, tensor in
            partial + tensor.shape.reduce(1, *)
        }
        guard scalarCount == DepthAnything3SmallCheckpoint.scalarCount else {
            throw DepthAnything3WeightError.scalarCountMismatch(
                expected: DepthAnything3SmallCheckpoint.scalarCount,
                actual: scalarCount
            )
        }
        if let nonFloat = metadata.first(where: { $0.value.dtype != .float32 }) {
            throw DepthAnything3WeightError.nonFloat32Tensor(nonFloat.key)
        }

        let target = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        var mappedShapes: [String: [Int]] = [:]
        mappedShapes.reserveCapacity(metadata.count)
        for (sourceName, tensor) in metadata {
            guard sourceName.hasPrefix("model.") else {
                throw DepthAnything3WeightError.invalidPrefix(sourceName)
            }
            let targetName = targetName(for: sourceName)
            mappedShapes[targetName] = mappedShape(sourceName: sourceName, sourceShape: tensor.shape)
        }
        let missing = Set(target.keys).subtracting(mappedShapes.keys).sorted()
        let unexpected = Set(mappedShapes.keys).subtracting(target.keys).sorted()
        guard missing.isEmpty && unexpected.isEmpty else {
            throw DepthAnything3WeightError.targetTensorSetMismatch(
                missing: missing,
                unexpected: unexpected
            )
        }
        for (name, expectedShape) in target {
            guard let actualShape = mappedShapes[name] else { continue }
            guard actualShape == expectedShape else {
                throw DepthAnything3WeightError.shapeMismatch(
                    name: name,
                    expected: expectedShape,
                    actual: actualShape
                )
            }
        }
    }

    public static func load(
        model: DepthAnything3Model,
        safetensorsURL: URL,
        dtype: DType = .float32,
        verifyChecksum: Bool = true
    ) throws {
        try validate(
            model: model,
            safetensorsURL: safetensorsURL,
            verifyChecksum: verifyChecksum
        )
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: safetensorsURL,
            to: model,
            dtype: dtype,
            // `validate` above proves an exact one-to-one name and shape set.
            // Streaming applies small batches, so `.all` cannot be used for an
            // individual partial update (it would demand every model key in
            // every batch).
            verify: .none,
            mapper: mapSourceTensor,
            batchSize: 32
        )
        MLX.eval(model)
    }

    public static func mapSourceTensor(
        key: String,
        value: MLXArray
    ) -> [(String, MLXArray)] {
        guard key.hasPrefix("model.") else { return [] }
        let targetName = targetName(for: key)
        if isTransposedConvolution(key), value.ndim == 4 {
            let transposed = value.transposed(1, 2, 3, 0)
            return [(targetName, contiguous(transposed))]
        }
        if value.ndim == 4 {
            let transposed = value.transposed(0, 2, 3, 1)
            return [(targetName, contiguous(transposed))]
        }
        return [(targetName, value)]
    }

    private static func mappedShape(sourceName: String, sourceShape: [Int]) -> [Int] {
        guard sourceShape.count == 4 else { return sourceShape }
        if isTransposedConvolution(sourceName) {
            return [sourceShape[1], sourceShape[2], sourceShape[3], sourceShape[0]]
        }
        return [sourceShape[0], sourceShape[2], sourceShape[3], sourceShape[1]]
    }

    private static func isTransposedConvolution(_ key: String) -> Bool {
        key == "model.head.resize_layers.0.weight"
            || key == "model.head.resize_layers.1.weight"
    }

    private static func targetName(for sourceName: String) -> String {
        var name = String(sourceName.dropFirst("model.".count))
        let exactReplacements: [(String, String)] = [
            ("cam_dec.backbone.0.", "cam_dec.backbone.first."),
            ("cam_dec.backbone.2.", "cam_dec.backbone.second."),
            ("cam_dec.fc_fov.0.", "cam_dec.fc_fov.projection."),
            ("head.scratch.output_conv2.0.", "head.scratch.output_conv2.first."),
            ("head.scratch.output_conv2.2.", "head.scratch.output_conv2.second."),
        ]
        for (source, target) in exactReplacements {
            name = name.replacingOccurrences(of: source, with: target)
        }
        let preHeadNames = ["first", "second", "third", "fourth", "fifth"]
        for level in 0..<4 {
            for (index, target) in preHeadNames.enumerated() {
                name = name.replacingOccurrences(
                    of: "head.scratch.output_conv1_aux.\(level).\(index).",
                    with: "head.scratch.output_conv1_aux.\(level).\(target)."
                )
            }
            name = name.replacingOccurrences(
                of: "head.scratch.output_conv2_aux.\(level).0.",
                with: "head.scratch.output_conv2_aux.\(level).first."
            )
            name = name.replacingOccurrences(
                of: "head.scratch.output_conv2_aux.\(level).5.",
                with: "head.scratch.output_conv2_aux.\(level).second."
            )
        }
        name = name.replacingOccurrences(
            of: "head.scratch.output_conv2_aux.0.2.",
            with: "head.scratch.output_conv2_aux.0.owned_shared_norm."
        )
        return name
    }

    private static func contiguous(_ value: MLXArray) -> MLXArray {
        value.reshaped(-1).reshaped(value.shape)
    }
}
