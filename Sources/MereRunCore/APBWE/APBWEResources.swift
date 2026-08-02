import Foundation

public struct APBWECheckpoint: Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let sourceConfigurationURL: URL
    public let codeLicenseURL: URL
    public let weightsLicenseURL: URL
    public let configuration: APBWEConfiguration
}

public struct APBWEResources: Sendable {
    public static let modelID = "audio-enhance-ap-bwe-16kto48k"
    public static let sourceRepository = "yxlu-0102/AP-BWE"
    public static let sourceRevision = "751710f22404c27e5bcc983248f8b856a04b8422"
    public static let artifactRepository = "rsxdalv/AP-BWE"
    public static let artifactRevision = "fa3f46d233cbc1d75cec9321188f86db627ba239"
    public static let officialDriveFolderID = "1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF"
    public static let officialWeightsFileID = "1HYkD_5ha9GMrjzbTiSFiO1uQQwxXET15"
    public static let expectedTensorCount = 162
    public static let expectedScalarCount = 29_760_515

    public static let weightsPin = ModelArtifactPin(
        filename: "weights/16kto48k/g_16kto48k.zip",
        byteCount: 119_097_961,
        sha256: "305a05dcab7dc29ffba09d32692d7a34550fc8fbdf338013641ff5d39a3cb285"
    )
    public static let sourceConfigurationPin = ModelArtifactPin(
        filename: "weights/16kto48k/config.json",
        byteCount: 515,
        sha256: "d722dcda233d6b00669cc37fa1286d9a49ad8d18082f28c20f895303e706852e"
    )
    public static let codeLicensePin = ModelArtifactPin(
        filename: "LICENSE.txt",
        byteCount: 1_066,
        sha256: "ebb9bf8bb881d60baf37b56772ad9e83813286ecc8b0853914de9d15fe3fec66"
    )
    public static let weightsLicensePin = ModelArtifactPin(
        filename: "weights_LICENSE.txt",
        byteCount: 175,
        sha256: "506d3a47a2c84814c8d6a84f0450a18fb9d67c03bb30d8b7e9d357d9540be2d3"
    )

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public static func loadBundledConfiguration() throws -> APBWEConfiguration {
        let nested = Bundle.module.url(
            forResource: "16kto48k",
            withExtension: "json",
            subdirectory: "APBWE"
        )
        guard let url = nested ?? Bundle.module.url(
            forResource: "16kto48k",
            withExtension: "json"
        ) else {
            throw APBWEError.missingBundledConfiguration
        }
        let configuration = try JSONDecoder().decode(
            APBWEConfiguration.self,
            from: Data(contentsOf: url)
        )
        try configuration.validate()
        return configuration
    }

    public static var pins: [ModelArtifactPin] {
        [weightsPin, sourceConfigurationPin, codeLicensePin, weightsLicensePin]
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

    public func resolve() throws -> APBWECheckpoint {
        APBWECheckpoint(
            rootURL: rootURL,
            weightsURL: try Self.weightsPin.verify(in: rootURL),
            sourceConfigurationURL: try Self.sourceConfigurationPin.verify(in: rootURL),
            codeLicenseURL: try Self.codeLicensePin.verify(in: rootURL),
            weightsLicenseURL: try Self.weightsLicensePin.verify(in: rootURL),
            configuration: try Self.loadBundledConfiguration()
        )
    }
}
