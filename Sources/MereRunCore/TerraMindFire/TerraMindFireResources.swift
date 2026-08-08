import Foundation
@preconcurrency import MLX

public struct TerraMindFireConversionConfiguration: Codable, Equatable, Sendable {
    public let format: String
    public let modelID: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let sourceCheckpoint: String
    public let sourceCheckpointSHA256: String
    public let sourceConfigurationSHA256: String
    public let converter: String
    public let dtype: String
    public let tensorCount: Int
    public let scalarCount: Int
    public let tileSize: Int
    public let timestamps: Int

    private enum CodingKeys: String, CodingKey {
        case format
        case modelID = "model_id"
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case sourceCheckpoint = "source_checkpoint"
        case sourceCheckpointSHA256 = "source_checkpoint_sha256"
        case sourceConfigurationSHA256 = "source_configuration_sha256"
        case converter
        case dtype
        case tensorCount = "tensor_count"
        case scalarCount = "scalar_count"
        case tileSize = "tile_size"
        case timestamps
    }
}

public struct TerraMindFireCheckpoint: Equatable, Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let configuration: TerraMindFireConversionConfiguration
}

public enum TerraMindFireResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case modelNotFound(String)
    case missingArtifact(String)
    case invalidConfiguration(String)
    case unsupportedFormat(String)
    case unsupportedSource(String, String)
    case unsupportedPrecision(String)
    case unsupportedDimensions(tileSize: Int, timestamps: Int)
    case tensorInventory(expected: Int, actual: Int)
    case scalarInventory(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported TerraMind Fire model '\(value)'."
        case .modelNotFound(let path):
            "TerraMind Fire model was not found: \(path)"
        case .missingArtifact(let path):
            "TerraMind Fire model artifact is missing: \(path)"
        case .invalidConfiguration(let message):
            "TerraMind Fire config.json is invalid: \(message)"
        case .unsupportedFormat(let value):
            "Unsupported TerraMind Fire conversion format '\(value)'."
        case .unsupportedSource(let repository, let revision):
            "Unsupported TerraMind Fire source pin \(repository)@\(revision)."
        case .unsupportedPrecision(let value):
            "Unsupported TerraMind Fire conversion precision '\(value)'; reference parity requires float32."
        case .unsupportedDimensions(let tileSize, let timestamps):
            "TerraMind Fire conversion requires tile_size=256 and timestamps=4; found \(tileSize)/\(timestamps)."
        case .tensorInventory(let expected, let actual):
            "TerraMind Fire tensor inventory mismatch: expected \(expected), found \(actual)."
        case .scalarInventory(let expected, let actual):
            "TerraMind Fire scalar inventory mismatch: expected \(expected), found \(actual)."
        }
    }
}

public enum TerraMindFireResources {
    public static let defaultModelID = "vision-fire-terramind-base"
    public static let sourceRepository = "ibm-esa-geospatial/TerraMind-base-Fire"
    public static let sourceRevision = "6eb5178aac4f8a4191796258ae26e796195cc00d"
    public static let sourceCheckpointSHA256 =
        "c16c070d95e9944c4b1a14c56cdd16f7821b133c8f2521985b579d08c2d8c72e"
    public static let sourceConfigurationSHA256 =
        "0ccbd6b9464f5a95198204b31dda77a1e7637611bb69bff7d569315f8ccc863f"
    public static let sourceCheckpointFilename = "TerraMind_v1_base_ImpactMesh_fire.pt"
    public static let sourceConfigurationFilename = "terramind_v1_base_impactmesh_fire.yaml"
    public static let conversionFormat = "mere.run/terramind-fire-mlx-v1"
    public static let weightsFilename = "model.safetensors"
    public static let configurationFilename = "config.json"
    public static let tensorCount = 171
    public static let scalarCount = 168_416_386
    public static let weightsArtifact = ModelArtifactPin(
        filename: weightsFilename,
        byteCount: 673_684_984,
        sha256: "7ebd587e684285112554743a27d50de596e807746c4ea38cfab34999c0adf21a"
    )
    public static let configurationArtifact = ModelArtifactPin(
        filename: configurationFilename,
        byteCount: 657,
        sha256: "c616393fb539e96a75e1bb72d09fd588c4527eb0e2a17193a86a16f41a839d79"
    )

    public static func missingSourcePaths(
        in rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        [sourceCheckpointFilename, sourceConfigurationFilename]
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func resolve(requestedModel: String?) async throws -> TerraMindFireCheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspect(explicit)
            }
            if requested.contains("/") || requested.hasPrefix(".") {
                throw TerraMindFireResourceError.modelNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard modelID == defaultModelID else {
            throw TerraMindFireResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        return try inspect(resolution.url)
    }

    public static func inspect(_ url: URL) throws -> TerraMindFireCheckpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw TerraMindFireResourceError.modelNotFound(standardized.path)
        }
        let root = isDirectory.boolValue ? standardized : standardized.deletingLastPathComponent()
        let weightsURL = isDirectory.boolValue ? root.appendingPathComponent(weightsFilename) : standardized
        let configurationURL = root.appendingPathComponent(configurationFilename)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw TerraMindFireResourceError.missingArtifact(weightsURL.path)
        }
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw TerraMindFireResourceError.missingArtifact(configurationURL.path)
        }
        _ = try weightsArtifact.verify(in: root)
        _ = try configurationArtifact.verify(in: root)
        let configuration: TerraMindFireConversionConfiguration
        do {
            configuration = try JSONDecoder().decode(
                TerraMindFireConversionConfiguration.self,
                from: Data(contentsOf: configurationURL)
            )
        } catch {
            throw TerraMindFireResourceError.invalidConfiguration(error.localizedDescription)
        }
        try validateConfiguration(configuration)
        return TerraMindFireCheckpoint(
            rootURL: root,
            weightsURL: weightsURL,
            configurationURL: configurationURL,
            configuration: configuration
        )
    }

    public static func validateConfiguration(
        _ configuration: TerraMindFireConversionConfiguration
    ) throws {
        guard configuration.format == conversionFormat else {
            throw TerraMindFireResourceError.unsupportedFormat(configuration.format)
        }
        guard configuration.modelID == defaultModelID,
              configuration.sourceRepository == sourceRepository,
              configuration.sourceRevision == sourceRevision,
              configuration.sourceCheckpoint == sourceCheckpointFilename,
              configuration.sourceCheckpointSHA256 == sourceCheckpointSHA256,
              configuration.sourceConfigurationSHA256 == sourceConfigurationSHA256 else {
            throw TerraMindFireResourceError.unsupportedSource(
                configuration.sourceRepository,
                configuration.sourceRevision
            )
        }
        guard configuration.dtype == "float32" else {
            throw TerraMindFireResourceError.unsupportedPrecision(configuration.dtype)
        }
        guard configuration.tensorCount == tensorCount else {
            throw TerraMindFireResourceError.tensorInventory(
                expected: tensorCount,
                actual: configuration.tensorCount
            )
        }
        guard configuration.scalarCount == scalarCount else {
            throw TerraMindFireResourceError.scalarInventory(
                expected: scalarCount,
                actual: configuration.scalarCount
            )
        }
        guard configuration.tileSize == TerraMindFloodModel.tileSize,
              configuration.timestamps == TerraMindFloodModel.timestampCount else {
            throw TerraMindFireResourceError.unsupportedDimensions(
                tileSize: configuration.tileSize,
                timestamps: configuration.timestamps
            )
        }
    }

    public static func loadModel(from checkpoint: TerraMindFireCheckpoint) throws -> TerraMindFloodModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == tensorCount else {
            throw TerraMindFireResourceError.tensorInventory(expected: tensorCount, actual: arrays.count)
        }
        let actualScalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard actualScalarCount == scalarCount else {
            throw TerraMindFireResourceError.scalarInventory(
                expected: scalarCount,
                actual: actualScalarCount
            )
        }
        return try TerraMindFloodModel(weights: arrays)
    }
}
