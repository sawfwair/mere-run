import Crypto
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

public struct VideoDepthAnythingConvertedPackagePins: Equatable, Sendable {
    public let weights: ModelArtifactPin
    public let configuration: ModelArtifactPin
    public let sourceManifest: ModelArtifactPin
    public let license: ModelArtifactPin
}

public extension VideoDepthAnythingVariant {
    var convertedPackagePins: VideoDepthAnythingConvertedPackagePins {
        let license = ModelArtifactPin(
            filename: "LICENSE",
            byteCount: 11_356,
            sha256: "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
        )
        switch self {
        case .relative:
            return VideoDepthAnythingConvertedPackagePins(
                weights: ModelArtifactPin(
                    filename: "model.safetensors",
                    byteCount: 116_362_340,
                    sha256: "85c583474dcafda4d417776431343afcdfdfc97952d8ec00029d3452c55a05a2"
                ),
                configuration: ModelArtifactPin(
                    filename: "config.json",
                    byteCount: 418,
                    sha256: "5e9a1dc52e91799e2b91db2676bfc19a3832b6a29becba7f49473adfc29d7b62"
                ),
                sourceManifest: ModelArtifactPin(
                    filename: "SOURCE.json",
                    byteCount: 942,
                    sha256: "a9f1ed205023b00ec7263e42a6b8b7b8e02cf07ed0e2834121f0b57d4d47622e"
                ),
                license: license
            )
        case .metric:
            return VideoDepthAnythingConvertedPackagePins(
                weights: ModelArtifactPin(
                    filename: "model.safetensors",
                    byteCount: 116_362_340,
                    sha256: "0acf1e186750abddf5ae867a3a659ed67cd0c041e4e524e698a0dcb40195c779"
                ),
                configuration: ModelArtifactPin(
                    filename: "config.json",
                    byteCount: 416,
                    sha256: "e78fb3f37caa2fc93f25ddd3e0fefedb694fcda0eb7bbdd8d86c7e638003f7ec"
                ),
                sourceManifest: ModelArtifactPin(
                    filename: "SOURCE.json",
                    byteCount: 963,
                    sha256: "30a4cb7dffebbc2b51f153514a4b2365931f03b7177224019db35a11fcc3bcf5"
                ),
                license: license
            )
        }
    }
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
        let licenseURL = directory.appendingPathComponent("LICENSE")
        guard FileManager.default.fileExists(atPath: sourceURL.path),
              FileManager.default.fileExists(atPath: configURL.path),
              FileManager.default.fileExists(atPath: licenseURL.path) else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "SOURCE.json, config.json, and the pinned upstream LICENSE are required beside model.safetensors"
            )
        }
        let (variant, sourceData) = try verifiedSourceData(
            at: sourceURL,
            expectedVariant: expectedVariant
        )
        let source = try decode(ConvertedSource.self, from: sourceData, at: sourceURL)
        guard VideoDepthAnythingVariant.modelID(source.modelID) == variant else {
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
        let pins = variant.convertedPackagePins
        let configData = try verifiedData(for: pins.configuration, at: configURL)
        try validate(
            config: decode(ConvertedConfig.self, from: configData, at: configURL),
            variant: variant
        )
        _ = try pins.license.verify(in: directory)
        let weightsURL = try pins.weights.verify(in: directory)
        return VideoDepthAnythingCheckpoint(
            variant: variant,
            format: .convertedSafetensors,
            weightsURL: weightsURL,
            weightsByteCount: pins.weights.byteCount,
            weightsSHA256: pins.weights.sha256,
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
              source.source.sourceCodeRepository == pin.sourceCodeRepository,
              source.source.sourceCodeRevision == pin.sourceCodeRevision,
              source.conversion.converter == "convert_vda_small.py",
              source.conversion.converterVersion == 1,
              source.conversion.environment.python == "3.11.15",
              source.conversion.environment.numpy == "2.4.3",
              source.conversion.environment.torch == "2.13.0",
              source.conversion.environment.safetensors == "0.8.0",
              source.conversion.outputFile == "model.safetensors",
              source.conversion.outputByteCount == variant.convertedPackagePins.weights.byteCount,
              source.conversion.outputSHA256.lowercased() == variant.convertedPackagePins.weights.sha256,
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

    /// Reads only enough bytes to prove that `SOURCE.json` is one of the two
    /// fixed conversion manifests. No attacker-controlled JSON is decoded
    /// before its exact byte count and checksum match a known pin.
    private static func verifiedSourceData(
        at url: URL,
        expectedVariant: VideoDepthAnythingVariant?
    ) throws -> (variant: VideoDepthAnythingVariant, data: Data) {
        let variants = VideoDepthAnythingVariant.allCases
        let maximumByteCount = variants.map { $0.convertedPackagePins.sourceManifest.byteCount }.max() ?? 0
        let data = try boundedData(at: url, maximumByteCount: maximumByteCount)
        let digest = sha256(data)
        let matches = variants.filter {
            let pin = $0.convertedPackagePins.sourceManifest
            return Int64(data.count) == pin.byteCount && digest == pin.sha256
        }
        guard let variant = matches.first, matches.count == 1 else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "SOURCE.json does not match an exact pinned conversion manifest"
            )
        }
        if let expectedVariant, expectedVariant != variant {
            throw VideoDepthAnythingResourceError.checkpointVariantMismatch(
                expected: expectedVariant.modelID,
                actual: variant.modelID
            )
        }
        return (variant, data)
    }

    /// Returns the same bounded bytes whose length and checksum were verified,
    /// closing the verify-then-decode replacement window for small metadata.
    private static func verifiedData(for pin: ModelArtifactPin, at url: URL) throws -> Data {
        let data = try boundedData(at: url, maximumByteCount: pin.byteCount)
        guard Int64(data.count) == pin.byteCount else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "\(pin.filename) has \(data.count) bytes; expected exactly \(pin.byteCount)"
            )
        }
        let digest = sha256(data)
        guard digest == pin.sha256 else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "\(pin.filename) checksum mismatch: expected \(pin.sha256), found \(digest)"
            )
        }
        return data
    }

    /// Reads at most one byte beyond the fixed contract. This detects an
    /// oversized file without ever allocating from attacker-controlled size.
    private static func boundedData(at url: URL, maximumByteCount: Int64) throws -> Data {
        guard maximumByteCount >= 0, maximumByteCount < Int64(Int.max) else {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "invalid metadata byte limit for \(url.lastPathComponent)"
            )
        }
        do {
            let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
            defer { try? handle.close() }
            let limit = Int(maximumByteCount) + 1
            var data = Data()
            data.reserveCapacity(limit)
            while data.count < limit {
                let chunk = try handle.read(upToCount: min(4_096, limit - data.count)) ?? Data()
                if chunk.isEmpty { break }
                data.append(chunk)
            }
            return data
        } catch let error as VideoDepthAnythingResourceError {
            throw error
        } catch {
            throw VideoDepthAnythingResourceError.invalidConvertedPackage(
                "could not read \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func decode<Value: Decodable>(
        _ type: Value.Type,
        from data: Data,
        at url: URL
    ) throws -> Value {
        do {
            return try JSONDecoder().decode(type, from: data)
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

}

private struct ConvertedSource: Decodable {
    struct Conversion: Decodable {
        struct Environment: Decodable {
            let python: String
            let numpy: String
            let torch: String
            let safetensors: String
        }

        let converter: String
        let converterVersion: Int
        let environment: Environment
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
        let sourceCodeRepository: String
        let sourceCodeRevision: String
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
