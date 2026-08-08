import Foundation
@preconcurrency import MLX

public enum TESSERAVariant: String, Codable, CaseIterable, Sendable {
    case nano
    case small
    case medium
    case large
    case teacher
}

public struct TESSERAConversionConfiguration: Codable, Equatable, Sendable {
    public let format: String
    public let modelID: String
    public let variant: TESSERAVariant
    public let sourceRepository: String
    public let sourceRevision: String
    public let sourceCheckpoint: String
    public let sourceCheckpointSHA256: String
    public let converter: String
    public let dtype: String
    public let tensorCount: Int
    public let scalarCount: Int
    public let architecture: TESSERAArchitecture

    private enum CodingKeys: String, CodingKey {
        case format
        case modelID = "model_id"
        case variant
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case sourceCheckpoint = "source_checkpoint"
        case sourceCheckpointSHA256 = "source_checkpoint_sha256"
        case converter
        case dtype
        case tensorCount = "tensor_count"
        case scalarCount = "scalar_count"
        case architecture
    }
}

public struct TESSERASourceSpec: Equatable, Sendable {
    public let variant: TESSERAVariant
    public let modelID: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let sourceCheckpointFilename: String
    public let sourceCheckpointSHA256: String
    public let sourceCheckpointByteCount: Int64
    public let architecture: TESSERAArchitecture
    public let tensorCount: Int
    public let scalarCount: Int
    public let weightsArtifact: ModelArtifactPin
    public let configurationArtifact: ModelArtifactPin
}

public struct TESSERACheckpoint: Equatable, Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let configurationURL: URL
    public let configuration: TESSERAConversionConfiguration
    public let source: TESSERASourceSpec
}

public enum TESSERAResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case modelNotFound(String)
    case missingArtifact(String)
    case invalidConfiguration(String)
    case unsupportedFormat(String)
    case unsupportedSource(String, String)
    case unsupportedPrecision(String)
    case tensorInventory(expected: Int, actual: Int)
    case scalarInventory(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let value):
            "Unsupported TESSERA v2 model '\(value)'."
        case .modelNotFound(let path):
            "TESSERA v2 model was not found: \(path)"
        case .missingArtifact(let path):
            "TESSERA v2 model artifact is missing: \(path)"
        case .invalidConfiguration(let message):
            "TESSERA v2 config.json is invalid: \(message)"
        case .unsupportedFormat(let value):
            "Unsupported TESSERA v2 conversion format '\(value)'."
        case .unsupportedSource(let repository, let revision):
            "Unsupported TESSERA v2 source pin \(repository)@\(revision)."
        case .unsupportedPrecision(let value):
            "Unsupported TESSERA v2 conversion precision '\(value)'; native parity requires float32."
        case .tensorInventory(let expected, let actual):
            "TESSERA v2 tensor inventory mismatch: expected \(expected), found \(actual)."
        case .scalarInventory(let expected, let actual):
            "TESSERA v2 scalar inventory mismatch: expected \(expected), found \(actual)."
        }
    }
}

public enum TESSERAResources {
    public static let conversionFormat = "mere.run/tessera-v2-mlx-v1"
    public static let weightsFilename = "model.safetensors"
    public static let configurationFilename = "config.json"

    public static let allSpecs: [TESSERASourceSpec] = [
        TESSERASourceSpec(
            variant: .nano,
            modelID: "vision-embed-tessera-v2-nano",
            sourceRepository: "geotessera/TESSERA-V-2.0-2B-N",
            sourceRevision: "9645033fdcd5c0686bab00720e5553ce307629cf",
            sourceCheckpointFilename: "ckpt/student_nano.pt",
            sourceCheckpointSHA256: "bc9929e3643dfab2744c2ce14e7c14f698703a94e9562e4e0982f9d6540b0691",
            sourceCheckpointByteCount: 4_288_221,
            architecture: TESSERAArchitecture(
                representationDimension: 128,
                latentDimension: 36,
                layerCount: 2,
                headCount: 4,
                feedForwardDimension: 384,
                maximumSequenceLength: 256
            ),
            tensorCount: 66,
            scalarCount: 1_066_402,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 4_273_560,
                sha256: "a1125fbe82dd83e0377bd66dae8c67df1c2bd1d1873cb0c86d1befd22fbd4fa6"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 707,
                sha256: "0e14322efd86232635568c218e291d32c538971a1a15f450d2aef15fae5266fc"
            )
        ),
        TESSERASourceSpec(
            variant: .small,
            modelID: "vision-embed-tessera-v2-small",
            sourceRepository: "geotessera/TESSERA-V-2.0-2B-S",
            sourceRevision: "21760b27ff16ca7aab01986b7b3460e3027b19c6",
            sourceCheckpointFilename: "ckpt/student_small.pt",
            sourceCheckpointSHA256: "92619d4ffc2936895f145fdcf710140b58c23c860e02f01873828b10c6958d95",
            sourceCheckpointByteCount: 28_488_531,
            architecture: TESSERAArchitecture(
                representationDimension: 128,
                latentDimension: 64,
                layerCount: 4,
                headCount: 4,
                feedForwardDimension: 1_024,
                maximumSequenceLength: 256
            ),
            tensorCount: 114,
            scalarCount: 7_112_322,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 28_463_400,
                sha256: "f8be74a820e97791e3c27127e581c089b4300d886b6d03dcbe576a95227dda6c"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 712,
                sha256: "f0ad6aa9fbe315c54e39b8006d93571ad35e733c4159543c65e356ba5d3fd77e"
            )
        ),
        TESSERASourceSpec(
            variant: .medium,
            modelID: "vision-embed-tessera-v2-medium",
            sourceRepository: "geotessera/TESSERA-V-2.0-2B-M",
            sourceRevision: "41db8ee5ddfcf6867f965526c2097d70c3c55c31",
            sourceCheckpointFilename: "ckpt/student_medium.pt",
            sourceCheckpointSHA256: "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e",
            sourceCheckpointByteCount: 84_163_403,
            architecture: TESSERAArchitecture(
                representationDimension: 128,
                latentDimension: 110,
                layerCount: 4,
                headCount: 4,
                feedForwardDimension: 1_792,
                maximumSequenceLength: 256
            ),
            tensorCount: 114,
            scalarCount: 21_031_506,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 84_140_200,
                sha256: "535ab12bc548a281dfa5e64f0b7cc3dc60691df0b55d99525e6cd8f4d4420d34"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 717,
                sha256: "164acee518c1cca8bc0c3fb0a7fd8f5be062e02e9716ef3ed1ee2e94318e4edd"
            )
        ),
        TESSERASourceSpec(
            variant: .large,
            modelID: "vision-embed-tessera-v2-large",
            sourceRepository: "geotessera/TESSERA-V-2.0-2B-L",
            sourceRevision: "b45f24463acf3fcfe030f94735d3e817b24100d0",
            sourceCheckpointFilename: "ckpt/student_large.pt",
            sourceCheckpointSHA256: "b5f20239dbb1849c01a3e407b095aafe39b0bf764300206af78cb9b85f9ec1e1",
            sourceCheckpointByteCount: 175_363_923,
            architecture: TESSERAArchitecture(
                representationDimension: 128,
                latentDimension: 160,
                layerCount: 4,
                headCount: 4,
                feedForwardDimension: 2_560,
                maximumSequenceLength: 256
            ),
            tensorCount: 114,
            scalarCount: 43_831_170,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 175_338_976,
                sha256: "0fe8a10f0aca102e39f54206efa3f71992c6523a0d4113b23ba1552495b9c640"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 714,
                sha256: "710b541838988413a271533522fcd85bbd81bc0464298193e618612d2d9d1575"
            )
        ),
        TESSERASourceSpec(
            variant: .teacher,
            modelID: "vision-embed-tessera-v2-teacher",
            sourceRepository: "geotessera/TESSERA-V-2.0-2B-Teacher",
            sourceRevision: "262170691f167085a7f86750066066e3d6ab6e10",
            sourceCheckpointFilename: "ckpt/tessera_v2_2B_teacher.pt",
            sourceCheckpointSHA256: "bfca890a9956485edf9d6be61375fde4cd0b4cb4b47db79596f0523a289aa555",
            sourceCheckpointByteCount: 8_257_134_395,
            architecture: TESSERAArchitecture(
                representationDimension: 1_024,
                latentDimension: 1_024,
                layerCount: 4,
                headCount: 4,
                feedForwardDimension: 16_384,
                maximumSequenceLength: 256,
                qkNormalization: true,
                fusionLayerCount: 2
            ),
            tensorCount: 199,
            scalarCount: 2_064_266_242,
            weightsArtifact: ModelArtifactPin(
                filename: weightsFilename,
                byteCount: 8_257_089_768,
                sha256: "f89af4788204c7a5c8e15fbefcbc448425694914901ed970e6b7a04546e139d2"
            ),
            configurationArtifact: ModelArtifactPin(
                filename: configurationFilename,
                byteCount: 796,
                sha256: "fa6dfe45d5d654dc00ea5d98a2a751ebd9845e096d6259f300c6398390f5552a"
            )
        ),
    ]

    public static func spec(for modelID: String) -> TESSERASourceSpec? {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allSpecs.first { $0.modelID == normalized }
    }

    public static func spec(for variant: TESSERAVariant) -> TESSERASourceSpec {
        allSpecs.first { $0.variant == variant }!
    }

    public static func missingSourcePaths(
        for modelID: String,
        in rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        guard let source = spec(for: modelID) else { return [rootURL] }
        let checkpoint = rootURL.appendingPathComponent(source.sourceCheckpointFilename)
        return fileManager.fileExists(atPath: checkpoint.path) ? [] : [checkpoint]
    }

    public static func recommendedVariant(on machine: MereRunMachineProfile = .current) -> TESSERAVariant {
        switch machine.unifiedMemoryGB {
        case ..<6: .nano
        case 6..<8: .small
        case 8..<12: .medium
        case 12..<32: .large
        default: .teacher
        }
    }

    public static func defaultModelID(on machine: MereRunMachineProfile = .current) -> String {
        let recommended = recommendedVariant(on: machine)
        let supported = allSpecs.prefix { $0.variant != recommended }.map(\TESSERASourceSpec.modelID)
            + [spec(for: recommended).modelID]
        let installed = supported.reversed().first {
            ManagedModelResolver.resolveInstalledModel(id: $0) != nil
        }
        return installed ?? spec(for: recommended).modelID
    }

    public static func resolve(
        requestedModel: String?,
        machine: MereRunMachineProfile = .current
    ) async throws -> TESSERACheckpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try inspect(explicit)
            }
            if requested.contains("/") || requested.hasPrefix(".") {
                throw TESSERAResourceError.modelNotFound(explicit.path)
            }
        }

        let modelID = requested.isEmpty ? defaultModelID(on: machine) : requested.lowercased()
        guard spec(for: modelID) != nil else {
            throw TESSERAResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: modelID,
            allowAutoDownload: false
        )
        return try inspect(resolution.url)
    }

    public static func inspect(_ url: URL) throws -> TESSERACheckpoint {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw TESSERAResourceError.modelNotFound(standardized.path)
        }
        let root = isDirectory.boolValue ? standardized : standardized.deletingLastPathComponent()
        let weightsURL = isDirectory.boolValue ? root.appendingPathComponent(weightsFilename) : standardized
        let configurationURL = root.appendingPathComponent(configurationFilename)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw TESSERAResourceError.missingArtifact(weightsURL.path)
        }
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw TESSERAResourceError.missingArtifact(configurationURL.path)
        }
        let configuration: TESSERAConversionConfiguration
        do {
            configuration = try JSONDecoder().decode(
                TESSERAConversionConfiguration.self,
                from: Data(contentsOf: configurationURL)
            )
        } catch {
            throw TESSERAResourceError.invalidConfiguration(error.localizedDescription)
        }
        guard let source = spec(for: configuration.modelID) else {
            throw TESSERAResourceError.unsupportedModel(configuration.modelID)
        }
        _ = try source.weightsArtifact.verify(in: root)
        _ = try source.configurationArtifact.verify(in: root)
        try validateConfiguration(configuration, source: source)
        return TESSERACheckpoint(
            rootURL: root,
            weightsURL: weightsURL,
            configurationURL: configurationURL,
            configuration: configuration,
            source: source
        )
    }

    public static func validateConfiguration(
        _ configuration: TESSERAConversionConfiguration,
        source: TESSERASourceSpec
    ) throws {
        guard configuration.format == conversionFormat else {
            throw TESSERAResourceError.unsupportedFormat(configuration.format)
        }
        guard configuration.modelID == source.modelID,
              configuration.variant == source.variant,
              configuration.sourceRepository == source.sourceRepository,
              configuration.sourceRevision == source.sourceRevision,
              configuration.sourceCheckpoint == source.sourceCheckpointFilename,
              configuration.sourceCheckpointSHA256 == source.sourceCheckpointSHA256 else {
            throw TESSERAResourceError.unsupportedSource(
                configuration.sourceRepository,
                configuration.sourceRevision
            )
        }
        guard configuration.dtype == "float32" else {
            throw TESSERAResourceError.unsupportedPrecision(configuration.dtype)
        }
        guard configuration.architecture == source.architecture else {
            throw TESSERAResourceError.invalidConfiguration("architecture does not match the immutable source pin")
        }
        guard configuration.tensorCount == source.tensorCount else {
            throw TESSERAResourceError.tensorInventory(
                expected: source.tensorCount,
                actual: configuration.tensorCount
            )
        }
        guard configuration.scalarCount == source.scalarCount else {
            throw TESSERAResourceError.scalarInventory(
                expected: source.scalarCount,
                actual: configuration.scalarCount
            )
        }
    }

    public static func loadModel(from checkpoint: TESSERACheckpoint) throws -> TESSERAModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == checkpoint.source.tensorCount else {
            throw TESSERAResourceError.tensorInventory(
                expected: checkpoint.source.tensorCount,
                actual: arrays.count
            )
        }
        let scalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard scalarCount == checkpoint.source.scalarCount else {
            throw TESSERAResourceError.scalarInventory(
                expected: checkpoint.source.scalarCount,
                actual: scalarCount
            )
        }
        return try TESSERAModel(weights: arrays, architecture: checkpoint.source.architecture)
    }
}
