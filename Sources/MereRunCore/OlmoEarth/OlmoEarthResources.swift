import Foundation
@preconcurrency import MLX

public enum OlmoEarthVariant: String, Codable, CaseIterable, Sendable {
    case nano
    case tiny
    case small
    case base
}

public struct OlmoEarthConversionConfiguration: Codable, Equatable, Sendable {
    public let format: String
    public let modelID: String
    public let variant: OlmoEarthVariant
    public let sourceRepository: String
    public let sourceRevision: String
    public let sourceWeights: String
    public let sourceWeightsSHA256: String
    public let sourceConfigurationSHA256: String
    public let converter: String
    public let dtype: String
    public let tensorCount: Int
    public let scalarCount: Int
    public let supportedModalities: [OlmoEarthModality]
    public let architecture: OlmoEarthArchitecture

    private enum CodingKeys: String, CodingKey {
        case format
        case modelID = "model_id"
        case variant
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case sourceWeights = "source_weights"
        case sourceWeightsSHA256 = "source_weights_sha256"
        case sourceConfigurationSHA256 = "source_configuration_sha256"
        case converter
        case dtype
        case tensorCount = "tensor_count"
        case scalarCount = "scalar_count"
        case supportedModalities = "supported_modalities"
        case architecture
    }
}

public struct OlmoEarthSourceSpec: Equatable, Sendable {
    public let variant: OlmoEarthVariant
    public let modelID: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let sourceWeightsSHA256: String
    public let sourceConfigurationSHA256: String
    public let sourceWeightsByteCount: Int64
    public let architecture: OlmoEarthArchitecture
    public let tensorCount: Int
    public let scalarCount: Int
    public let weightsArtifact: ModelArtifactPin
    public let configurationArtifact: ModelArtifactPin
}

public struct OlmoEarthCheckpoint: Equatable, Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let configuration: OlmoEarthConversionConfiguration
    public let source: OlmoEarthSourceSpec
}

public enum OlmoEarthResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case modelNotFound(String)
    case missingArtifact(String)
    case invalidConfiguration(String)
    case unsupportedFormat(String)
    case unsupportedSource(String, String)
    case unsupportedPrecision(String)
    case unsupportedModalities([OlmoEarthModality])
    case tensorInventory(expected: Int, actual: Int)
    case scalarInventory(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported OlmoEarth v1.2 model '\(value)'."
        case .modelNotFound(let path):
            "OlmoEarth v1.2 model was not found: \(path)"
        case .missingArtifact(let path):
            "OlmoEarth v1.2 model artifact is missing: \(path)"
        case .invalidConfiguration(let message):
            "OlmoEarth v1.2 config.json is invalid: \(message)"
        case .unsupportedFormat(let value):
            "Unsupported OlmoEarth v1.2 conversion format '\(value)'."
        case .unsupportedSource(let repository, let revision):
            "Unsupported OlmoEarth v1.2 source pin \(repository)@\(revision)."
        case .unsupportedPrecision(let value):
            "Unsupported OlmoEarth v1.2 conversion precision '\(value)'; native parity requires float32."
        case .unsupportedModalities(let values):
            "Unsupported OlmoEarth v1.2 modality inventory: \(values.map(\.rawValue).joined(separator: ", "))."
        case .tensorInventory(let expected, let actual):
            "OlmoEarth v1.2 tensor inventory mismatch: expected \(expected), found \(actual)."
        case .scalarInventory(let expected, let actual):
            "OlmoEarth v1.2 scalar inventory mismatch: expected \(expected), found \(actual)."
        }
    }
}

public enum OlmoEarthResources {
    public static let conversionFormat = "mere.run/olmoearth-v1.2-mlx-v1"
    public static let sourceWeightsFilename = "weights.pth"
    public static let sourceConfigurationFilename = "config.json"
    public static let weightsFilename = "model.safetensors"
    public static let configurationFilename = "config.json"
    public static let supportedModalities = OlmoEarthModality.allCases

    public static let allSpecs: [OlmoEarthSourceSpec] = [
        OlmoEarthSourceSpec(
            variant: .nano,
            modelID: "vision-embed-olmoearth-v12-nano",
            sourceRepository: "allenai/OlmoEarth-v1_2-Nano",
            sourceRevision: "e1f693ae2a7d5b57871a978e9d09e22d05206747",
            sourceWeightsSHA256: "2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378",
            sourceConfigurationSHA256: "4cd2888e79dc543f262cc3d86fcd30d667068fd53a728ca5bd306d5ddb509d1d",
            sourceWeightsByteCount: 17_038_723,
            architecture: OlmoEarthArchitecture(
                embeddingDimension: 128,
                headCount: 8,
                depth: 4,
                mlpRatio: 4,
                maximumSequenceLength: 12,
                maximumPatchSize: 8,
                patchHiddenDimension: 12,
                positionEncoding: "rope_3d_mixed",
                temporalCoordinateScale: 1 / 30
            ),
            tensorCount: 86,
            scalarCount: 1_090_224,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 4_369_648,
                sha256: "7f910264cf5f09f1853d9303c9f2c4b5e0f383d503b9d5800a9823ffa110dec1"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 992,
                sha256: "b70cf2285eb3276463c242bd37ea978259fc340747acc7e091d08c3775a25b5d"
            )
        ),
        OlmoEarthSourceSpec(
            variant: .tiny,
            modelID: "vision-embed-olmoearth-v12-tiny",
            sourceRepository: "allenai/OlmoEarth-v1_2-Tiny",
            sourceRevision: "12a9fdbfeff905d7e147e7497f9f7a95c518eefc",
            sourceWeightsSHA256: "835c0b21ab010c4c4515faafa44dc1a41c9bc512d3a30af184803c4f4257697d",
            sourceConfigurationSHA256: "bb11f91f5afbd6138f75feee3f66fc0e272da089d05a6e515713c799057155ac",
            sourceWeightsByteCount: 107_394_275,
            architecture: OlmoEarthArchitecture(
                embeddingDimension: 192,
                headCount: 3,
                depth: 12,
                mlpRatio: 4,
                maximumSequenceLength: 12,
                maximumPatchSize: 8,
                patchHiddenDimension: 64,
                positionEncoding: "rope_3d_mixed",
                temporalCoordinateScale: 1 / 30
            ),
            tensorCount: 222,
            scalarCount: 7_704_592,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 30_839_616,
                sha256: "308a98d482c5ddaec1c2552721a97802a9d2a144e84fcbb58b4dedbfa98f53b2"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 994,
                sha256: "438e1bac7f9396954a1b1c388152b9782a1eee8363750865332daeadeb8b5571"
            )
        ),
        OlmoEarthSourceSpec(
            variant: .small,
            modelID: "vision-embed-olmoearth-v12-small",
            sourceRepository: "allenai/OlmoEarth-v1_2-Small",
            sourceRevision: "a207c9a789483f95de1e9fb06acadb3da3775863",
            sourceWeightsSHA256: "459796ed5680bc85926f9a0e023476d14cb637bc19f826575c43836c909a5fa6",
            sourceConfigurationSHA256: "254703d9b5da4a6679003ff21f2da964a25d903fea70dc0b2cce5d0cd388f70b",
            sourceWeightsByteCount: 314_778_943,
            architecture: OlmoEarthArchitecture(
                embeddingDimension: 384,
                headCount: 6,
                depth: 12,
                mlpRatio: 4,
                maximumSequenceLength: 12,
                maximumPatchSize: 8,
                patchHiddenDimension: 64,
                positionEncoding: "rope_3d_mixed",
                temporalCoordinateScale: 1 / 30
            ),
            tensorCount: 222,
            scalarCount: 26_024_224,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 104_118_336,
                sha256: "a967dc27611e2effed4a7f74db82f6fcd1b561f58441430d5d7aed15dd2fdf6d"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 998,
                sha256: "d2f0bc9f8e95fe028c9b5ff5f139849935de1942b25e02a71c33755485c0de87"
            )
        ),
        OlmoEarthSourceSpec(
            variant: .base,
            modelID: "vision-embed-olmoearth-v12-base",
            sourceRepository: "allenai/OlmoEarth-v1_2-Base",
            sourceRevision: "581aa9baaa7aed4348c0903617eb92ee9f89e2ec",
            sourceWeightsSHA256: "57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec",
            sourceConfigurationSHA256: "0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a",
            sourceWeightsByteCount: 1_030_354_339,
            architecture: OlmoEarthArchitecture(
                embeddingDimension: 768,
                headCount: 12,
                depth: 12,
                mlpRatio: 4,
                maximumSequenceLength: 12,
                maximumPatchSize: 8,
                patchHiddenDimension: 64,
                positionEncoding: "rope_3d_mixed",
                temporalCoordinateScale: 0.0333
            ),
            tensorCount: 222,
            scalarCount: 94_513_984,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 378_077_744,
                sha256: "0f34899dc1b6e4ec9d436c2aa26f092dbd54dbb846098a3cc11661d5b00dcd29"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 983,
                sha256: "96b17e8882244b3812f5dbacf32093ce0f1efbbf15dbdc5c3b45b1ac3c55c621"
            )
        ),
    ]

    public static func spec(for modelID: String) -> OlmoEarthSourceSpec? {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allSpecs.first { $0.modelID == normalized }
    }

    public static func spec(for variant: OlmoEarthVariant) -> OlmoEarthSourceSpec {
        allSpecs.first { $0.variant == variant }!
    }

    public static func missingSourcePaths(
        in rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        [sourceWeightsFilename, sourceConfigurationFilename]
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func recommendedVariant(on machine: MereRunMachineProfile = .current) -> OlmoEarthVariant {
        switch machine.unifiedMemoryGB {
        case ..<8: .nano
        case 8..<12: .tiny
        case 12..<16: .small
        default: .base
        }
    }

    public static func defaultModelID(on machine: MereRunMachineProfile = .current) -> String {
        let recommended = recommendedVariant(on: machine)
        let recommendedIndex = allSpecs.firstIndex { $0.variant == recommended }!
        let installed = allSpecs[...recommendedIndex].reversed().first {
            ManagedModelResolver.resolveInstalledModel(id: $0.modelID) != nil
        }
        return installed?.modelID ?? spec(for: recommended).modelID
    }

    public static func resolve(
        requestedModel: String?,
        machine: MereRunMachineProfile = .current
    ) async throws -> OlmoEarthCheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspect(explicit)
            }
            if requested.contains("/") || requested.hasPrefix(".") {
                throw OlmoEarthResourceError.modelNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID(on: machine) : requested.lowercased()
        guard spec(for: modelID) != nil else {
            throw OlmoEarthResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: modelID,
            allowAutoDownload: false
        )
        return try inspect(resolution.url)
    }

    public static func inspect(_ url: URL) throws -> OlmoEarthCheckpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw OlmoEarthResourceError.modelNotFound(standardized.path)
        }
        let root = isDirectory.boolValue ? standardized : standardized.deletingLastPathComponent()
        let weightsURL = isDirectory.boolValue ? root.appendingPathComponent(weightsFilename) : standardized
        let configurationURL = root.appendingPathComponent(configurationFilename)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw OlmoEarthResourceError.missingArtifact(weightsURL.path)
        }
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw OlmoEarthResourceError.missingArtifact(configurationURL.path)
        }
        let configuration: OlmoEarthConversionConfiguration
        do {
            configuration = try JSONDecoder().decode(
                OlmoEarthConversionConfiguration.self,
                from: Data(contentsOf: configurationURL)
            )
        } catch {
            throw OlmoEarthResourceError.invalidConfiguration(error.localizedDescription)
        }
        guard let source = spec(for: configuration.modelID) else {
            throw OlmoEarthResourceError.unsupportedModel(configuration.modelID)
        }
        _ = try source.weightsArtifact.verify(in: root)
        _ = try source.configurationArtifact.verify(in: root)
        try validateConfiguration(configuration, source: source)
        return OlmoEarthCheckpoint(
            rootURL: root,
            weightsURL: weightsURL,
            configurationURL: configurationURL,
            configuration: configuration,
            source: source
        )
    }

    public static func validateConfiguration(
        _ configuration: OlmoEarthConversionConfiguration,
        source: OlmoEarthSourceSpec
    ) throws {
        guard configuration.format == conversionFormat else {
            throw OlmoEarthResourceError.unsupportedFormat(configuration.format)
        }
        guard configuration.modelID == source.modelID,
              configuration.variant == source.variant,
              configuration.sourceRepository == source.sourceRepository,
              configuration.sourceRevision == source.sourceRevision,
              configuration.sourceWeights == sourceWeightsFilename,
              configuration.sourceWeightsSHA256 == source.sourceWeightsSHA256,
              configuration.sourceConfigurationSHA256 == source.sourceConfigurationSHA256 else {
            throw OlmoEarthResourceError.unsupportedSource(
                configuration.sourceRepository,
                configuration.sourceRevision
            )
        }
        guard configuration.dtype == "float32" else {
            throw OlmoEarthResourceError.unsupportedPrecision(configuration.dtype)
        }
        guard configuration.supportedModalities == supportedModalities else {
            throw OlmoEarthResourceError.unsupportedModalities(configuration.supportedModalities)
        }
        guard configuration.architecture == source.architecture else {
            throw OlmoEarthResourceError.invalidConfiguration("architecture does not match the immutable source pin")
        }
        guard configuration.tensorCount == source.tensorCount else {
            throw OlmoEarthResourceError.tensorInventory(
                expected: source.tensorCount,
                actual: configuration.tensorCount
            )
        }
        guard configuration.scalarCount == source.scalarCount else {
            throw OlmoEarthResourceError.scalarInventory(
                expected: source.scalarCount,
                actual: configuration.scalarCount
            )
        }
    }

    public static func loadModel(from checkpoint: OlmoEarthCheckpoint) throws -> OlmoEarthModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == checkpoint.source.tensorCount else {
            throw OlmoEarthResourceError.tensorInventory(
                expected: checkpoint.source.tensorCount,
                actual: arrays.count
            )
        }
        let scalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard scalarCount == checkpoint.source.scalarCount else {
            throw OlmoEarthResourceError.scalarInventory(
                expected: checkpoint.source.scalarCount,
                actual: scalarCount
            )
        }
        return try OlmoEarthModel(weights: arrays, architecture: checkpoint.source.architecture)
    }
}
