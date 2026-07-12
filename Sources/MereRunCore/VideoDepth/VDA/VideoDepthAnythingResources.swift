import Foundation
@preconcurrency import MLX

public enum VideoDepthAnythingVariant: String, Codable, CaseIterable, Hashable, Sendable {
    case relative
    case metric

    public var modelID: String {
        switch self {
        case .relative: ModelResolver.ModelID.visionDepthVDASmall.rawValue
        case .metric: ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue
        }
    }

    public var semantics: DepthSemantics {
        switch self {
        case .relative: .affineRelative
        case .metric: .metricMeters
        }
    }

    public var pin: GeometryModelPin {
        switch self {
        case .relative: GeometryModelPins.videoDepthAnythingSmall
        case .metric: GeometryModelPins.videoDepthAnythingSmallMetric
        }
    }

    public static func modelID(_ modelID: String) -> Self? {
        allCases.first { $0.modelID == modelID.lowercased() }
    }
}

public enum VideoDepthAnythingCheckpointFormat: String, Codable, Hashable, Sendable {
    case pinnedPyTorch = "pinned-pytorch-state-dict"
    case convertedSafetensors = "verified-converted-safetensors"
}

public struct VideoDepthAnythingCheckpoint: Codable, Equatable, Hashable, Sendable {
    public let variant: VideoDepthAnythingVariant
    public let format: VideoDepthAnythingCheckpointFormat
    public let weightsURL: URL
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let sourceSHA256: String

    public init(
        variant: VideoDepthAnythingVariant,
        format: VideoDepthAnythingCheckpointFormat,
        weightsURL: URL,
        weightsByteCount: Int64,
        weightsSHA256: String,
        sourceSHA256: String
    ) {
        self.variant = variant
        self.format = format
        self.weightsURL = weightsURL.standardizedFileURL
        self.weightsByteCount = weightsByteCount
        self.weightsSHA256 = weightsSHA256.lowercased()
        self.sourceSHA256 = sourceSHA256.lowercased()
    }
}

public enum VideoDepthAnythingResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case checkpointNotFound(String)
    case unsupportedCheckpointPath(String)
    case ambiguousCheckpointDirectory(String)
    case unrecognizedPinnedCheckpoint(String)
    case invalidConvertedPackage(String)
    case checkpointVariantMismatch(expected: String, actual: String)
    case tensorInventoryMismatch(tensors: Int, values: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported Video Depth Anything model '\(value)'."
        case .checkpointNotFound(let path):
            "Video Depth Anything checkpoint not found: \(path)"
        case .unsupportedCheckpointPath(let path):
            "Expected a pinned .pth file or converted model.safetensors directory at \(path)."
        case .ambiguousCheckpointDirectory(let path):
            "Checkpoint directory contains both relative and metric VDA weights: \(path)"
        case .unrecognizedPinnedCheckpoint(let path):
            "Checkpoint does not match either audited VDA-S artifact: \(path)"
        case .invalidConvertedPackage(let detail):
            "Invalid converted VDA package: \(detail)"
        case .checkpointVariantMismatch(let expected, let actual):
            "Checkpoint variant mismatch: expected \(expected), found \(actual)."
        case .tensorInventoryMismatch(let tensors, let values):
            "VDA checkpoint inventory mismatch: found \(tensors) tensors and \(values) values."
        }
    }
}

public enum VideoDepthAnythingResources {
    public static let defaultModelID = VideoDepthAnythingVariant.relative.modelID

    public static func resolve(requestedModel: String?) async throws -> VideoDepthAnythingCheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspectExplicit(explicit)
            }
            if looksLikePath(requested) {
                throw VideoDepthAnythingResourceError.checkpointNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard let expectedVariant = VideoDepthAnythingVariant.modelID(modelID) else {
            throw VideoDepthAnythingResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        return try inspectExplicit(resolution.url, expectedVariant: expectedVariant)
    }

    public static func inspectExplicit(_ url: URL) throws -> VideoDepthAnythingCheckpoint {
        try inspectExplicit(url.standardizedFileURL, expectedVariant: nil)
    }

    public static func loadModel(from checkpoint: VideoDepthAnythingCheckpoint) throws -> VideoDepthAnythingModel {
        let verified: VideoDepthAnythingCheckpoint
        switch checkpoint.format {
        case .pinnedPyTorch:
            verified = try inspectExplicit(checkpoint.weightsURL, expectedVariant: checkpoint.variant)
        case .convertedSafetensors:
            verified = try inspectExplicit(
                checkpoint.weightsURL.deletingLastPathComponent(),
                expectedVariant: checkpoint.variant
            )
        }
        guard verified == checkpoint else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "checkpoint identity changed after preflight"
            )
        }

        let model = VideoDepthAnythingModel()
        switch verified.format {
        case .pinnedPyTorch:
            try loadPyTorchCheckpoint(verified.weightsURL, into: model)
        case .convertedSafetensors:
            try VideoDepthAnythingWeights.load(
                model: model,
                safetensorsURL: verified.weightsURL,
                dtype: .float32
            )
        }
        return model
    }

    private static func inspectExplicit(
        _ url: URL,
        expectedVariant: VideoDepthAnythingVariant?
    ) throws -> VideoDepthAnythingCheckpoint {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw VideoDepthAnythingResourceError.checkpointNotFound(url.path)
        }
        if !isDirectory.boolValue {
            guard url.pathExtension.lowercased() == "pth" else {
                throw VideoDepthAnythingResourceError.unsupportedCheckpointPath(url.path)
            }
            return try inspectPyTorch(url, expectedVariant: expectedVariant)
        }

        let converted = url.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: converted.path) {
            return try inspectConvertedDirectory(url, expectedVariant: expectedVariant)
        }

        let candidates = VideoDepthAnythingVariant.allCases.compactMap { variant -> (VideoDepthAnythingVariant, URL)? in
            let candidate = url.appendingPathComponent(variant.pin.artifacts[0].filename)
            return FileManager.default.fileExists(atPath: candidate.path) ? (variant, candidate) : nil
        }
        if candidates.count > 1, expectedVariant == nil {
            throw VideoDepthAnythingResourceError.ambiguousCheckpointDirectory(url.path)
        }
        if let expectedVariant {
            let candidate = url.appendingPathComponent(expectedVariant.pin.artifacts[0].filename)
            guard FileManager.default.fileExists(atPath: candidate.path) else {
                throw VideoDepthAnythingResourceError.checkpointNotFound(candidate.path)
            }
            return try inspectPyTorch(candidate, expectedVariant: expectedVariant)
        }
        guard let candidate = candidates.first else {
            throw VideoDepthAnythingResourceError.unsupportedCheckpointPath(url.path)
        }
        return try inspectPyTorch(candidate.1, expectedVariant: candidate.0)
    }

    private static func inspectPyTorch(
        _ url: URL,
        expectedVariant: VideoDepthAnythingVariant?
    ) throws -> VideoDepthAnythingCheckpoint {
        let byteCount = try ModelArtifactPin.fileByteCount(url)
        let sha256 = try ModelArtifactPin.fileSHA256(url)
        let matches = VideoDepthAnythingVariant.allCases.filter {
            let artifact = $0.pin.artifacts[0]
            return artifact.byteCount == byteCount && artifact.sha256 == sha256
        }
        guard let variant = matches.first, matches.count == 1 else {
            throw VideoDepthAnythingResourceError.unrecognizedPinnedCheckpoint(url.path)
        }
        if let expectedVariant, expectedVariant != variant {
            throw VideoDepthAnythingResourceError.checkpointVariantMismatch(
                expected: expectedVariant.modelID,
                actual: variant.modelID
            )
        }
        return VideoDepthAnythingCheckpoint(
            variant: variant,
            format: .pinnedPyTorch,
            weightsURL: url,
            weightsByteCount: byteCount,
            weightsSHA256: sha256,
            sourceSHA256: sha256
        )
    }

    private static func inspectConvertedDirectory(
        _ directory: URL,
        expectedVariant: VideoDepthAnythingVariant?
    ) throws -> VideoDepthAnythingCheckpoint {
        let sourceURL = directory.appendingPathComponent("SOURCE.json")
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: sourceURL.path),
              FileManager.default.fileExists(atPath: configURL.path) else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "SOURCE.json and config.json are required beside model.safetensors"
            )
        }
        let source = try decode(ConvertedSource.self, at: sourceURL)
        guard let variant = VideoDepthAnythingVariant.modelID(source.modelID) else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "unsupported source model id '\(source.modelID)'"
            )
        }
        if let expectedVariant, expectedVariant != variant {
            throw VideoDepthAnythingResourceError.checkpointVariantMismatch(
                expected: expectedVariant.modelID,
                actual: variant.modelID
            )
        }
        try validate(source: source, variant: variant)
        try validate(config: decode(ConvertedConfig.self, at: configURL), variant: variant)

        let weightsURL = directory.appendingPathComponent(source.conversion.outputFile)
        let outputPin = ModelArtifactPin(
            filename: weightsURL.lastPathComponent,
            byteCount: source.conversion.outputByteCount,
            sha256: source.conversion.outputSHA256
        )
        _ = try outputPin.verify(in: directory)
        return VideoDepthAnythingCheckpoint(
            variant: variant,
            format: .convertedSafetensors,
            weightsURL: weightsURL,
            weightsByteCount: source.conversion.outputByteCount,
            weightsSHA256: source.conversion.outputSHA256,
            sourceSHA256: source.source.sha256
        )
    }

    private static func loadPyTorchCheckpoint(_ url: URL, into model: VideoDepthAnythingModel) throws {
        let archive = try PyTorchStateDictArchive(url: url)
        let valueCount = archive.tensors.reduce(0) { $0 + $1.elementCount }
        guard archive.tensors.count == VideoDepthAnythingWeights.sourceTensorCount,
              valueCount == VideoDepthAnythingWeights.sourceScalarCount,
              archive.tensors.allSatisfy({ $0.dataType == .float32 }) else {
            throw VideoDepthAnythingResourceError.tensorInventoryMismatch(
                tensors: archive.tensors.count,
                values: valueCount
            )
        }
        try VideoDepthAnythingWeights.load(model: model, archive: archive, dtype: .float32)
    }

    private static func validate(source: ConvertedSource, variant: VideoDepthAnythingVariant) throws {
        let pin = variant.pin
        let artifact = pin.artifacts[0]
        guard source.modelID == variant.modelID,
              source.license == "Apache-2.0",
              source.source.filename == artifact.filename,
              source.source.byteCount == artifact.byteCount,
              source.source.sha256.lowercased() == artifact.sha256,
              source.source.repository == pin.repository,
              source.source.revision == pin.revision,
              source.conversion.converter == "convert_vda_small.py",
              source.conversion.outputFile == "model.safetensors",
              source.conversion.outputByteCount > 0,
              isSHA256(source.conversion.outputSHA256),
              isSHA256(source.conversion.converterSHA256),
              source.conversion.tensorCount == VideoDepthAnythingWeights.sourceTensorCount,
              source.conversion.scalarCount == VideoDepthAnythingWeights.sourceScalarCount else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "SOURCE.json does not match the pinned \(variant.modelID) conversion contract"
            )
        }
    }

    private static func validate(config: ConvertedConfig, variant: VideoDepthAnythingVariant) throws {
        guard config.architecture == "video-depth-anything-small",
              config.backbone == "dinov2-vits14",
              config.depthSemantics == variant.semantics.rawValue,
              config.featureChannels == 64,
              config.intermediateLayers == [2, 5, 8, 11],
              config.projectedChannels == [48, 96, 192, 384],
              config.temporalAttentionBlocks == 2,
              config.temporalAttentionHeads == 8,
              config.temporalFrameCount == VideoDepthAnythingWindowing.windowLength,
              config.temporalOverlap == VideoDepthAnythingWindowing.overlap,
              config.temporalTransformerBlocks == 1 else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "config.json does not describe the production VDA-S graph"
            )
        }
    }

    private static func decode<Value: Decodable>(_ type: Value.Type, at url: URL) throws -> Value {
        do {
            return try JSONDecoder().decode(type, from: Data(contentsOf: url))
        } catch {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "could not decode \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    private static func looksLikePath(_ value: String) -> Bool {
        value.contains("/") || value.hasPrefix(".")
            || ["pth", "safetensors"].contains(URL(fileURLWithPath: value).pathExtension.lowercased())
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }
}

private struct ConvertedSource: Decodable {
    struct Conversion: Decodable {
        let converter: String
        let converterSHA256: String
        let outputByteCount: Int64
        let outputFile: String
        let outputSHA256: String
        let tensorCount: Int
        let scalarCount: Int
    }

    struct Source: Decodable {
        let byteCount: Int64
        let filename: String
        let repository: String
        let revision: String
        let sha256: String
    }

    let conversion: Conversion
    let license: String
    let modelID: String
    let source: Source
}

private struct ConvertedConfig: Decodable {
    let architecture: String
    let backbone: String
    let depthSemantics: String
    let featureChannels: Int
    let intermediateLayers: [Int]
    let projectedChannels: [Int]
    let temporalAttentionBlocks: Int
    let temporalAttentionHeads: Int
    let temporalFrameCount: Int
    let temporalOverlap: Int
    let temporalTransformerBlocks: Int
}
