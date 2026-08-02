import Foundation

public struct RoFormerCheckpoint: Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let sourceConfigurationURL: URL
    public let licenseURL: URL
    public let readmeURL: URL
    public let configuration: RoFormerConfiguration
}

public struct RoFormerResources: Sendable {
    public static let modelID = "music-separate-bs-roformer-viperx-1297"
    public static let repository = "AEmotionStudio/roformer-models"
    public static let revision = "d323194290f8488ea51814143806609bfbd7a1e5"
    public static let sourceDirectory = "bs_roformer/vocals_viperx"
    public static let expectedTensorCount = 699
    public static let expectedScalarCount = 159_758_796

    public static let weightsPin = ModelArtifactPin(
        filename: "\(sourceDirectory)/model.safetensors",
        byteCount: 639_109_056,
        sha256: "fa296577206144929917601636b65ccdc407b6e6c2f209e4312d9d2b7b975a8a"
    )
    public static let sourceConfigurationPin = ModelArtifactPin(
        filename: "\(sourceDirectory)/config.yaml",
        byteCount: 1_881,
        sha256: "42e5635ceab7287b83d9591c9880966f0b61cff26a0a40a29d236bacb32e5f2c"
    )
    public static let licensePin = ModelArtifactPin(
        filename: "LICENSE",
        byteCount: 1_159,
        sha256: "899b277f35c41f0f5873e317319c84876f57a90d765b9e0d357035699cc4e4bc"
    )
    public static let readmePin = ModelArtifactPin(
        filename: "README.md",
        byteCount: 2_549,
        sha256: "d756be2a43d8373e650e16f86e4ee43aa90a56082c23182d074130080e3febaf"
    )
    public static let pins = [weightsPin, sourceConfigurationPin, licensePin, readmePin]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public static func loadBundledConfiguration() throws -> RoFormerConfiguration {
        let nested = Bundle.module.url(
            forResource: "viperx-1297",
            withExtension: "json",
            subdirectory: "RoFormer"
        )
        guard let url = nested ?? Bundle.module.url(
            forResource: "viperx-1297",
            withExtension: "json"
        ) else {
            throw RoFormerError.missingBundledConfiguration
        }
        let configuration = try JSONDecoder().decode(RoFormerConfiguration.self, from: Data(contentsOf: url))
        try configuration.validate()
        return configuration
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        Self.pins.compactMap { pin in
            do {
                _ = try pin.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return rootURL.appendingPathComponent(pin.filename)
            }
        }
    }

    public func validationMessages(fileManager: FileManager = .default) -> [String] {
        Self.pins.compactMap { pin in
            do {
                _ = try pin.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return error.localizedDescription
            }
        }
    }

    public func resolve() throws -> RoFormerCheckpoint {
        let weightsURL = try Self.weightsPin.verify(in: rootURL)
        let sourceConfigurationURL = try Self.sourceConfigurationPin.verify(in: rootURL)
        let licenseURL = try Self.licensePin.verify(in: rootURL)
        let readmeURL = try Self.readmePin.verify(in: rootURL)
        let configuration = try Self.loadBundledConfiguration()
        return RoFormerCheckpoint(
            rootURL: rootURL,
            weightsURL: weightsURL,
            sourceConfigurationURL: sourceConfigurationURL,
            licenseURL: licenseURL,
            readmeURL: readmeURL,
            configuration: configuration
        )
    }
}
