import Foundation
@preconcurrency import MLX

public struct TerraMindFloodConversionConfiguration: Codable, Equatable, Sendable {
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

public struct TerraMindFloodCheckpoint: Equatable, Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let configuration: TerraMindFloodConversionConfiguration
}

public enum TerraMindFloodResourceError: Error, Equatable, LocalizedError, Sendable {
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
            "Unsupported TerraMind Flood model '\(value)'."
        case .modelNotFound(let path):
            "TerraMind Flood model was not found: \(path)"
        case .missingArtifact(let path):
            "TerraMind Flood model artifact is missing: \(path)"
        case .invalidConfiguration(let message):
            "TerraMind Flood config.json is invalid: \(message)"
        case .unsupportedFormat(let value):
            "Unsupported TerraMind Flood conversion format '\(value)'."
        case .unsupportedSource(let repository, let revision):
            "Unsupported TerraMind Flood source pin \(repository)@\(revision)."
        case .unsupportedPrecision(let value):
            "Unsupported TerraMind Flood conversion precision '\(value)'; exact flood parity requires float32."
        case .unsupportedDimensions(let tileSize, let timestamps):
            "TerraMind Flood conversion requires tile_size=256 and timestamps=4; found \(tileSize)/\(timestamps)."
        case .tensorInventory(let expected, let actual):
            "TerraMind Flood tensor inventory mismatch: expected \(expected), found \(actual)."
        case .scalarInventory(let expected, let actual):
            "TerraMind Flood scalar inventory mismatch: expected \(expected), found \(actual)."
        }
    }
}

public enum TerraMindFloodResources {
    public static let defaultModelID = ModelResolver.ModelID.visionFloodTerraMindBase.rawValue
    public static let sourceRepository = "ibm-esa-geospatial/TerraMind-base-Flood"
    public static let sourceRevision = "1e4b2429d17234922f8d92beb0d725af4db85c08"
    public static let sourceCheckpointSHA256 =
        "22627584c2db618c2f6ddb64b411a95762a893becb25104e3f66bfebecaa71e9"
    public static let sourceCheckpointFilename = "TerraMind_v1_base_ImpactMesh_flood.pt"
    public static let sourceConfigurationFilename = "terramind_v1_base_impactmesh_flood.yaml"
    public static let conversionFormat = "mere.run/terramind-flood-mlx-v1"
    public static let weightsFilename = "model.safetensors"
    public static let configurationFilename = "config.json"
    public static let tensorCount = 171
    public static let scalarCount = 168_416_386
    public static let weightsArtifact = ModelArtifactPin(
        filename: weightsFilename,
        byteCount: 673_684_984,
        sha256: "4940ad94df06a923e3a919f944a71ad01892872e89c428abe718eefc44d0f95a"
    )
    public static let configurationArtifact = ModelArtifactPin(
        filename: configurationFilename,
        byteCount: 662,
        sha256: "28f507dd0ad3f9825e64300b545bb889b125df4b53fc9c313f7ecd977b40c63d"
    )

    public static func missingSourcePaths(
        in rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        [sourceCheckpointFilename, sourceConfigurationFilename]
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func resolve(requestedModel: String?) async throws -> TerraMindFloodCheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspect(explicit)
            }
            if requested.contains("/") || requested.hasPrefix(".") {
                throw TerraMindFloodResourceError.modelNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard modelID == defaultModelID else {
            throw TerraMindFloodResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        return try inspect(resolution.url)
    }

    public static func inspect(_ url: URL) throws -> TerraMindFloodCheckpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw TerraMindFloodResourceError.modelNotFound(standardized.path)
        }
        let root = isDirectory.boolValue ? standardized : standardized.deletingLastPathComponent()
        let weightsURL = isDirectory.boolValue
            ? root.appendingPathComponent(weightsFilename)
            : standardized
        let configurationURL = root.appendingPathComponent(configurationFilename)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw TerraMindFloodResourceError.missingArtifact(weightsURL.path)
        }
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw TerraMindFloodResourceError.missingArtifact(configurationURL.path)
        }
        _ = try weightsArtifact.verify(in: root)
        _ = try configurationArtifact.verify(in: root)
        let configuration: TerraMindFloodConversionConfiguration
        do {
            configuration = try JSONDecoder().decode(
                TerraMindFloodConversionConfiguration.self,
                from: Data(contentsOf: configurationURL)
            )
        } catch {
            throw TerraMindFloodResourceError.invalidConfiguration(error.localizedDescription)
        }
        try validateConfiguration(configuration)
        return TerraMindFloodCheckpoint(
            rootURL: root,
            weightsURL: weightsURL,
            configurationURL: configurationURL,
            configuration: configuration
        )
    }

    static func validateConfiguration(_ configuration: TerraMindFloodConversionConfiguration) throws {
        guard configuration.format == conversionFormat else {
            throw TerraMindFloodResourceError.unsupportedFormat(configuration.format)
        }
        guard configuration.sourceRepository == sourceRepository,
              configuration.sourceRevision == sourceRevision,
              configuration.sourceCheckpointSHA256 == sourceCheckpointSHA256 else {
            throw TerraMindFloodResourceError.unsupportedSource(
                configuration.sourceRepository,
                configuration.sourceRevision
            )
        }
        guard configuration.dtype == "float32" else {
            throw TerraMindFloodResourceError.unsupportedPrecision(configuration.dtype)
        }
        guard configuration.tensorCount == tensorCount else {
            throw TerraMindFloodResourceError.tensorInventory(
                expected: tensorCount,
                actual: configuration.tensorCount
            )
        }
        guard configuration.scalarCount == scalarCount else {
            throw TerraMindFloodResourceError.scalarInventory(
                expected: scalarCount,
                actual: configuration.scalarCount
            )
        }
        guard configuration.tileSize == TerraMindFloodModel.tileSize,
              configuration.timestamps == TerraMindFloodModel.timestampCount else {
            throw TerraMindFloodResourceError.unsupportedDimensions(
                tileSize: configuration.tileSize,
                timestamps: configuration.timestamps
            )
        }
    }

    public static func loadModel(from checkpoint: TerraMindFloodCheckpoint) throws -> TerraMindFloodModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == checkpoint.configuration.tensorCount else {
            throw TerraMindFloodResourceError.tensorInventory(
                expected: checkpoint.configuration.tensorCount,
                actual: arrays.count
            )
        }
        let scalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard scalarCount == checkpoint.configuration.scalarCount else {
            throw TerraMindFloodResourceError.scalarInventory(
                expected: checkpoint.configuration.scalarCount,
                actual: scalarCount
            )
        }
        return try TerraMindFloodModel(weights: arrays)
    }
}
