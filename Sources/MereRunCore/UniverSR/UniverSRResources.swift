import Foundation

public struct UniverSRCheckpoint: Sendable {
    public let rootURL: URL
    public let weightsURL: URL
    public let sourceConfigurationURL: URL
    public let modelCardURL: URL
    public let configuration: UniverSRConfiguration
}

public struct UniverSRResources: Sendable {
    public static let modelID = "audio-enhance-universr-audio"
    public static let sourceRepository = "woongzip1/UniverSR"
    public static let sourceRevision = "26dc21c44e11f9f19e823f02b0d4641dd5ea5af2"
    public static let artifactRepository = "woongzip1/universr-audio"
    public static let artifactRevision = "1c3294844285af851b6ffa56cbde4e43cd41fc2b"
    public static let expectedTensorCount = 394
    public static let expectedScalarCount = 57_231_302

    public static let weightsPin = ModelArtifactPin(
        filename: "pytorch_model.bin",
        byteCount: 229_072_395,
        sha256: "eb99f98943cc32fa82226c2da14b32b5d890416070af4946acbce442b30dc20b"
    )
    public static let sourceConfigurationPin = ModelArtifactPin(
        filename: "config.yaml",
        byteCount: 533,
        sha256: "2e035dacc833368319f9d5ba1a34e5efd343faf749008d433e309f3c65f4dac9"
    )
    public static let modelCardPin = ModelArtifactPin(
        filename: "README.md",
        byteCount: 1_406,
        sha256: "52aceb799f9fb0db692e59c53ea95ded528578664feb9b5726e8809bac8a44db"
    )

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public static var pins: [ModelArtifactPin] {
        [weightsPin, sourceConfigurationPin, modelCardPin]
    }

    public static func loadBundledConfiguration() throws -> UniverSRConfiguration {
        let nested = Bundle.module.url(
            forResource: "audio",
            withExtension: "json",
            subdirectory: "UniverSR"
        )
        guard let url = nested ?? Bundle.module.url(forResource: "audio", withExtension: "json") else {
            throw UniverSRError.missingBundledConfiguration
        }
        let configuration = try JSONDecoder().decode(
            UniverSRConfiguration.self,
            from: Data(contentsOf: url)
        )
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

    public func resolve() throws -> UniverSRCheckpoint {
        UniverSRCheckpoint(
            rootURL: rootURL,
            weightsURL: try Self.weightsPin.verify(in: rootURL),
            sourceConfigurationURL: try Self.sourceConfigurationPin.verify(in: rootURL),
            modelCardURL: try Self.modelCardPin.verify(in: rootURL),
            configuration: try Self.loadBundledConfiguration()
        )
    }
}
