import Foundation

public enum RoFormerModelProfile: String, CaseIterable, Sendable {
    case viperX1297
    case fourStem

    public static func resolve(modelID: String) throws -> Self {
        guard let profile = allCases.first(where: { $0.modelID == modelID }) else {
            throw RoFormerError.invalidConfiguration("unsupported BS-RoFormer model id \(modelID)")
        }
        return profile
    }

    public var modelID: String {
        switch self {
        case .viperX1297:
            ModelResolver.ModelID.roFormerViperX1297.rawValue
        case .fourStem:
            ModelResolver.ModelID.roFormerFourStem.rawValue
        }
    }

    var sourceDirectory: String {
        switch self {
        case .viperX1297: "bs_roformer/vocals_viperx"
        case .fourStem: "bs_roformer/multistem"
        }
    }

    var bundledConfigurationName: String {
        switch self {
        case .viperX1297: "viperx-1297"
        case .fourStem: "four-stem"
        }
    }

    var expectedTensorCount: Int {
        switch self {
        case .viperX1297: 699
        case .fourStem: 1_355
        }
    }

    var expectedScalarCount: Int {
        switch self {
        case .viperX1297: 159_758_796
        case .fourStem: 131_704_612
        }
    }

    public var weightsPin: ModelArtifactPin {
        switch self {
        case .viperX1297:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/model.safetensors",
                byteCount: 639_109_056,
                sha256: "fa296577206144929917601636b65ccdc407b6e6c2f209e4312d9d2b7b975a8a"
            )
        case .fourStem:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/model.safetensors",
                byteCount: 526_964_800,
                sha256: "14c9079fa428cb0a6d9b8a294a4c80a27276630a53772833c00c416f1f909098"
            )
        }
    }

    var sourceConfigurationPin: ModelArtifactPin {
        switch self {
        case .viperX1297:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/config.yaml",
                byteCount: 1_881,
                sha256: "42e5635ceab7287b83d9591c9880966f0b61cff26a0a40a29d236bacb32e5f2c"
            )
        case .fourStem:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/config.yaml",
                byteCount: 4_566,
                sha256: "d8afb980318d0c08b9c2e24a7adc00d4f3150320c127a7e4de861800d1321939"
            )
        }
    }

    var pins: [ModelArtifactPin] {
        [weightsPin, sourceConfigurationPin, RoFormerResources.licensePin, RoFormerResources.readmePin]
    }

    var derivesInstrumental: Bool { self == .viperX1297 }
}

public struct RoFormerCheckpoint: Sendable {
    public let profile: RoFormerModelProfile
    public let rootURL: URL
    public let weightsURL: URL
    public let sourceConfigurationURL: URL
    public let licenseURL: URL
    public let readmeURL: URL
    public let configuration: RoFormerConfiguration
}

public struct RoFormerResources: Sendable {
    public static let repository = "AEmotionStudio/roformer-models"
    public static let revision = "d323194290f8488ea51814143806609bfbd7a1e5"
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

    public let rootURL: URL
    public let profile: RoFormerModelProfile

    public init(rootURL: URL, profile: RoFormerModelProfile = .viperX1297) {
        self.rootURL = rootURL.standardizedFileURL
        self.profile = profile
    }

    public init(rootURL: URL, modelID: String) throws {
        self.init(rootURL: rootURL, profile: try RoFormerModelProfile.resolve(modelID: modelID))
    }

    public static func loadBundledConfiguration(
        profile: RoFormerModelProfile = .viperX1297
    ) throws -> RoFormerConfiguration {
        let name = profile.bundledConfigurationName
        let nested = Bundle.module.url(
            forResource: name,
            withExtension: "json",
            subdirectory: "RoFormer"
        )
        guard let url = nested ?? Bundle.module.url(
            forResource: name,
            withExtension: "json"
        ) else {
            throw RoFormerError.missingBundledConfiguration
        }
        let configuration = try JSONDecoder().decode(
            RoFormerConfiguration.self,
            from: Data(contentsOf: url)
        )
        try configuration.validate(profile: profile)
        return configuration
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        profile.pins.compactMap { pin in
            do {
                _ = try pin.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return rootURL.appendingPathComponent(pin.filename)
            }
        }
    }

    public func validationMessages(fileManager: FileManager = .default) -> [String] {
        profile.pins.compactMap { pin in
            do {
                _ = try pin.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return error.localizedDescription
            }
        }
    }

    public func resolve() throws -> RoFormerCheckpoint {
        let weightsURL = try profile.weightsPin.verify(in: rootURL)
        let sourceConfigurationURL = try profile.sourceConfigurationPin.verify(in: rootURL)
        let licenseURL = try Self.licensePin.verify(in: rootURL)
        let readmeURL = try Self.readmePin.verify(in: rootURL)
        let configuration = try Self.loadBundledConfiguration(profile: profile)
        return RoFormerCheckpoint(
            profile: profile,
            rootURL: rootURL,
            weightsURL: weightsURL,
            sourceConfigurationURL: sourceConfigurationURL,
            licenseURL: licenseURL,
            readmeURL: readmeURL,
            configuration: configuration
        )
    }
}
