import Crypto
import Foundation
@preconcurrency import MLX

public enum TripoSRCheckpointFormat: String, Codable, Equatable, Hashable, Sendable {
    case pinnedPyTorch = "pinned-pytorch-state-dict"
    case convertedSafetensors = "verified-converted-safetensors"
}

/// Verified runtime identity for the only TripoSR checkpoint accepted by
/// mere.run. `sourceSHA256` always identifies the pinned upstream CKPT even
/// when `weightsSHA256` identifies the deterministic converted safetensors.
public struct TripoSRCheckpoint: Codable, Equatable, Hashable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let license: String
    public let format: TripoSRCheckpointFormat
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let sourceSHA256: String
    public let configurationSHA256: String

    public init(
        modelID: String,
        repository: String,
        revision: String,
        sourceRepository: String,
        sourceRevision: String,
        license: String,
        format: TripoSRCheckpointFormat,
        rootURL: URL,
        weightsURL: URL,
        configurationURL: URL,
        weightsByteCount: Int64,
        weightsSHA256: String,
        sourceSHA256: String,
        configurationSHA256: String
    ) {
        self.modelID = modelID
        self.repository = repository
        self.revision = revision
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.license = license
        self.format = format
        self.rootURL = rootURL.standardizedFileURL
        self.weightsURL = weightsURL.standardizedFileURL
        self.configurationURL = configurationURL.standardizedFileURL
        self.weightsByteCount = weightsByteCount
        self.weightsSHA256 = weightsSHA256.lowercased()
        self.sourceSHA256 = sourceSHA256.lowercased()
        self.configurationSHA256 = configurationSHA256.lowercased()
    }
}

public enum TripoSRResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case checkpointNotFound(String)
    case unsupportedCheckpointPath(String)
    case missingCompanionConfiguration(String)
    case unrecognizedPinnedCheckpoint(String)
    case invalidConvertedPackage(String)
    case checkpointIdentityChanged
    case tensorInventoryMismatch(tensors: Int, values: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported TripoSR model '\(value)'. Only image-3d-triposr is permitted."
        case .checkpointNotFound(let path):
            "TripoSR checkpoint was not found: \(path)"
        case .unsupportedCheckpointPath(let path):
            "Expected the pinned model.ckpt or a converted safetensors package at \(path)."
        case .missingCompanionConfiguration(let path):
            "TripoSR checkpoint configuration is missing: \(path)"
        case .unrecognizedPinnedCheckpoint(let path):
            "TripoSR checkpoint does not match the audited upstream artifact: \(path)"
        case .invalidConvertedPackage(let detail):
            "Invalid converted TripoSR package: \(detail)"
        case .checkpointIdentityChanged:
            "TripoSR checkpoint identity changed after preflight verification."
        case .tensorInventoryMismatch(let tensors, let values):
            "TripoSR checkpoint has \(tensors) tensors/\(values) values; expected "
                + "\(TripoSRWeights.sourceTensorCount)/\(TripoSRWeights.sourceScalarCount)."
        }
    }
}

public enum TripoSRResources {
    public static let defaultModelID = ModelResolver.ModelID.image3DTripoSR.rawValue
    public static let convertedWeightsPin = ModelArtifactPin(
        filename: "model.safetensors",
        byteCount: TripoSRWeights.convertedSafetensorsByteCount,
        sha256: TripoSRWeights.convertedSafetensorsSHA256
    )
    public static let convertedConfigurationPin = ModelArtifactPin(
        filename: "config.json",
        byteCount: 378,
        sha256: "89bd2abd8024fba7474ca584b962aa1f50c67db2c6317cb86d04a3bfddd8f22c"
    )
    public static let convertedSourcePin = ModelArtifactPin(
        filename: "SOURCE.json",
        byteCount: 855,
        sha256: "5c12adbc30f80524007d946f78df11da077a0df6ba25b3409e566cda6afb902c"
    )
    public static let convertedLicensePin = ModelArtifactPin(
        filename: "LICENSE",
        byteCount: 1_080,
        sha256: "ade0a66629bdd7e01e46b3296b3851cff0fd27989bca53da470ad6e96ed620fb"
    )

    public static func resolve(requestedModel: String?) async throws -> TripoSRCheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspectExplicit(explicit)
            }
            if looksLikePath(requested) {
                throw TripoSRResourceError.checkpointNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard modelID == defaultModelID else {
            throw TripoSRResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        return try inspectExplicit(resolution.url)
    }

    public static func inspectExplicit(_ url: URL) throws -> TripoSRCheckpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw TripoSRResourceError.checkpointNotFound(standardized.path)
        }
        if isDirectory.boolValue {
            let convertedURL = standardized.appendingPathComponent(convertedWeightsPin.filename)
            if FileManager.default.fileExists(atPath: convertedURL.path) {
                return try inspectConvertedDirectory(standardized)
            }
            return try inspectPinnedCheckpoint(
                standardized.appendingPathComponent(sourceWeightsPin.filename)
            )
        }

        switch standardized.pathExtension.lowercased() {
        case "ckpt":
            return try inspectPinnedCheckpoint(standardized)
        case "safetensors":
            return try inspectConvertedDirectory(standardized.deletingLastPathComponent())
        default:
            throw TripoSRResourceError.unsupportedCheckpointPath(standardized.path)
        }
    }

    public static func loadModel(from checkpoint: TripoSRCheckpoint) throws -> TripoSRModel {
        let verified: TripoSRCheckpoint
        switch checkpoint.format {
        case .pinnedPyTorch:
            verified = try inspectExplicit(checkpoint.weightsURL)
        case .convertedSafetensors:
            verified = try inspectExplicit(checkpoint.rootURL)
        }
        guard verified == checkpoint else {
            throw TripoSRResourceError.checkpointIdentityChanged
        }

        let model = TripoSRModel()
        switch verified.format {
        case .pinnedPyTorch:
            // `inspectExplicit` immediately above verified the complete file
            // SHA-256. Re-reading 1.67 GB in pure-Swift ZIP CRC loops would add
            // minutes without increasing integrity beyond that exact digest.
            let archive = try PyTorchStateDictArchive(
                url: verified.weightsURL,
                verifyEntryChecksums: false
            )
            let valueCount = archive.tensors.reduce(0) { $0 + $1.elementCount }
            guard archive.tensors.count == TripoSRWeights.sourceTensorCount,
                  valueCount == TripoSRWeights.sourceScalarCount,
                  archive.tensors.allSatisfy({ $0.dataType == .float32 }) else {
                throw TripoSRResourceError.tensorInventoryMismatch(
                    tensors: archive.tensors.count,
                    values: valueCount
                )
            }
            try TripoSRWeights.load(model: model, archive: archive, dtype: .float32)
        case .convertedSafetensors:
            try TripoSRWeights.load(
                model: model,
                safetensorsURL: verified.weightsURL,
                dtype: .float32
            )
        }
        return model
    }

    private static var pin: GeometryModelPin { GeometryModelPins.tripoSR }

    private static var sourceWeightsPin: ModelArtifactPin {
        guard let artifact = pin.artifacts.first(where: { $0.filename == "model.ckpt" }) else {
            fatalError("TripoSR model.ckpt pin is missing")
        }
        return artifact
    }

    private static var sourceConfigurationPin: ModelArtifactPin {
        guard let artifact = pin.artifacts.first(where: { $0.filename == "config.yaml" }) else {
            fatalError("TripoSR config.yaml pin is missing")
        }
        return artifact
    }

    private static func inspectPinnedCheckpoint(_ weightsURL: URL) throws -> TripoSRCheckpoint {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw TripoSRResourceError.checkpointNotFound(weightsURL.path)
        }
        let configurationURL = weightsURL.deletingLastPathComponent()
            .appendingPathComponent(sourceConfigurationPin.filename)
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw TripoSRResourceError.missingCompanionConfiguration(configurationURL.path)
        }

        let byteCount = try ModelArtifactPin.fileByteCount(weightsURL)
        let sha256 = try ModelArtifactPin.fileSHA256(weightsURL.resolvingSymlinksInPath())
        guard byteCount == sourceWeightsPin.byteCount, sha256 == sourceWeightsPin.sha256 else {
            throw TripoSRResourceError.unrecognizedPinnedCheckpoint(weightsURL.path)
        }
        try verify(configurationURL, pin: sourceConfigurationPin)
        return checkpoint(
            format: .pinnedPyTorch,
            root: weightsURL.deletingLastPathComponent(),
            weights: weightsURL,
            configuration: configurationURL,
            weightsByteCount: byteCount,
            weightsSHA256: sha256,
            configurationSHA256: sourceConfigurationPin.sha256
        )
    }

    private static func inspectConvertedDirectory(_ root: URL) throws -> TripoSRCheckpoint {
        let weightsURL = root.appendingPathComponent(convertedWeightsPin.filename)
        let configurationURL = root.appendingPathComponent("config.json")
        let sourceURL = root.appendingPathComponent("SOURCE.json")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw TripoSRResourceError.checkpointNotFound(weightsURL.path)
        }
        guard FileManager.default.fileExists(atPath: configurationURL.path),
              FileManager.default.fileExists(atPath: sourceURL.path),
              FileManager.default.fileExists(
                atPath: root.appendingPathComponent(convertedLicensePin.filename).path
              ) else {
            throw TripoSRResourceError.invalidConvertedPackage(
                "config.json, SOURCE.json, and the pinned upstream LICENSE are required beside model.safetensors"
            )
        }

        let sourceData = try verifiedData(for: convertedSourcePin, at: sourceURL)
        let configurationData = try verifiedData(
            for: convertedConfigurationPin,
            at: configurationURL
        )
        let source = try decode(ConvertedSource.self, from: sourceData, at: sourceURL)
        let configuration = try decode(
            ConvertedConfiguration.self,
            from: configurationData,
            at: configurationURL
        )
        try validate(source: source)
        try validate(configuration: configuration)
        _ = try convertedLicensePin.verify(in: root)
        _ = try convertedWeightsPin.verify(in: root)
        return checkpoint(
            format: .convertedSafetensors,
            root: root,
            weights: weightsURL,
            configuration: configurationURL,
            weightsByteCount: convertedWeightsPin.byteCount,
            weightsSHA256: convertedWeightsPin.sha256,
            configurationSHA256: convertedConfigurationPin.sha256
        )
    }

    private static func checkpoint(
        format: TripoSRCheckpointFormat,
        root: URL,
        weights: URL,
        configuration: URL,
        weightsByteCount: Int64,
        weightsSHA256: String,
        configurationSHA256: String
    ) -> TripoSRCheckpoint {
        TripoSRCheckpoint(
            modelID: pin.modelID,
            repository: pin.repository,
            revision: pin.revision,
            sourceRepository: pin.sourceCodeRepository,
            sourceRevision: pin.sourceCodeRevision,
            license: pin.license,
            format: format,
            rootURL: root,
            weightsURL: weights,
            configurationURL: configuration,
            weightsByteCount: weightsByteCount,
            weightsSHA256: weightsSHA256,
            sourceSHA256: sourceWeightsPin.sha256,
            configurationSHA256: configurationSHA256
        )
    }

    private static func validate(source: ConvertedSource) throws {
        guard source.modelID == defaultModelID,
              source.license == pin.license,
              source.source.repository == pin.repository,
              source.source.revision == pin.revision,
              source.source.sourceCodeRepository == pin.sourceCodeRepository,
              source.source.sourceCodeRevision == pin.sourceCodeRevision,
              source.source.filename == sourceWeightsPin.filename,
              source.source.byteCount == sourceWeightsPin.byteCount,
              source.source.sha256.lowercased() == sourceWeightsPin.sha256,
              source.conversion.converter == "convert_triposr.py",
              source.conversion.converterVersion == 1,
              source.conversion.environment.python == "3.11.15",
              source.conversion.environment.torch == "2.13.0",
              source.conversion.environment.safetensors == "0.8.0",
              source.conversion.outputFile == convertedWeightsPin.filename,
              source.conversion.outputByteCount == convertedWeightsPin.byteCount,
              source.conversion.outputSHA256.lowercased() == convertedWeightsPin.sha256,
              source.conversion.tensorCount == TripoSRWeights.sourceTensorCount,
              source.conversion.scalarCount == TripoSRWeights.sourceScalarCount else {
            throw TripoSRResourceError.invalidConvertedPackage(
                "SOURCE.json does not match the pinned TripoSR conversion contract"
            )
        }
    }

    private static func validate(configuration: ConvertedConfiguration) throws {
        let production = TripoSRConfiguration.production
        guard configuration.architecture == "triposr",
              configuration.conditioningImageSize == production.conditioningImageSize,
              configuration.imageEncoder == "facebook/dino-vitb16",
              configuration.planeSize == production.scenePlaneSize,
              configuration.planeChannels == production.scenePlaneChannels,
              configuration.transformerLayers == production.transformerLayerCount,
              configuration.transformerAttentionHeads == production.transformerHeadCount,
              configuration.transformerTokenChannels == production.tokenChannels,
              configuration.decoderHiddenSize == production.decoderHiddenSize,
              configuration.decoderHiddenLayers == production.decoderHiddenLayerCount,
              configuration.rendererRadius == production.rendererRadius,
              configuration.densityBias == production.densityBias,
              configuration.densityThreshold == production.densityThreshold else {
            throw TripoSRResourceError.invalidConvertedPackage(
                "config.json does not describe the production TripoSR graph"
            )
        }
    }

    private static func verify(_ url: URL, pin: ModelArtifactPin) throws {
        let byteCount = try ModelArtifactPin.fileByteCount(url)
        guard byteCount == pin.byteCount else {
            throw ModelArtifactVerificationError.sizeMismatch(
                path: url.path,
                expected: pin.byteCount,
                actual: byteCount
            )
        }
        let sha256 = try ModelArtifactPin.fileSHA256(url.resolvingSymlinksInPath())
        guard sha256 == pin.sha256 else {
            throw ModelArtifactVerificationError.checksumMismatch(
                path: url.path,
                expected: pin.sha256,
                actual: sha256
            )
        }
    }

    /// Reads at most one byte beyond the fixed metadata contract, then hashes
    /// and returns those same bytes. This prevents both unbounded allocation
    /// and a verify-then-decode path replacement window.
    private static func verifiedData(for pin: ModelArtifactPin, at url: URL) throws -> Data {
        guard pin.byteCount >= 0, pin.byteCount < Int64(Int.max) else {
            throw TripoSRResourceError.invalidConvertedPackage(
                "invalid byte limit for \(pin.filename)"
            )
        }
        let limit = Int(pin.byteCount) + 1
        let data: Data
        do {
            let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
            defer { try? handle.close() }
            var captured = Data()
            captured.reserveCapacity(limit)
            while captured.count < limit {
                let chunk = try handle.read(upToCount: min(4_096, limit - captured.count)) ?? Data()
                if chunk.isEmpty { break }
                captured.append(chunk)
            }
            data = captured
        } catch {
            throw TripoSRResourceError.invalidConvertedPackage(
                "could not read \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
        guard Int64(data.count) == pin.byteCount else {
            throw TripoSRResourceError.invalidConvertedPackage(
                "\(pin.filename) has \(data.count) bytes; expected exactly \(pin.byteCount)"
            )
        }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == pin.sha256 else {
            throw TripoSRResourceError.invalidConvertedPackage(
                "\(pin.filename) checksum mismatch: expected \(pin.sha256), found \(digest)"
            )
        }
        return data
    }

    private static func decode<Value: Decodable>(
        _ type: Value.Type,
        from data: Data,
        at url: URL
    ) throws -> Value {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw TripoSRResourceError.invalidConvertedPackage(
                "could not decode \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    private static func looksLikePath(_ value: String) -> Bool {
        value.contains("/") || value.hasPrefix(".")
            || ["ckpt", "safetensors"].contains(
                URL(fileURLWithPath: value).pathExtension.lowercased()
            )
    }
}

private struct ConvertedSource: Decodable {
    struct Conversion: Decodable {
        struct Environment: Decodable {
            let python: String
            let torch: String
            let safetensors: String
        }

        let converter: String
        let converterVersion: Int
        let environment: Environment
        let outputByteCount: Int64
        let outputFile: String
        let outputSHA256: String
        let scalarCount: Int
        let tensorCount: Int
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

private struct ConvertedConfiguration: Decodable {
    let architecture: String
    let conditioningImageSize: Int
    let decoderHiddenLayers: Int
    let decoderHiddenSize: Int
    let densityBias: Float
    let densityThreshold: Float
    let imageEncoder: String
    let planeChannels: Int
    let planeSize: Int
    let rendererRadius: Float
    let transformerAttentionHeads: Int
    let transformerLayers: Int
    let transformerTokenChannels: Int
}
