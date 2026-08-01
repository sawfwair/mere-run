import Foundation

public enum MelBandRoFormerProfile: String, CaseIterable, Sendable {
    case dereverb
    case denoise

    public static func resolve(modelID: String) throws -> Self {
        guard let profile = allCases.first(where: { $0.modelID == modelID }) else {
            throw RoFormerError.invalidConfiguration("unsupported MelBand RoFormer model id \(modelID)")
        }
        return profile
    }

    public var modelID: String {
        switch self {
        case .dereverb: ModelResolver.ModelID.melRoFormerDereverb.rawValue
        case .denoise: ModelResolver.ModelID.melRoFormerDenoise.rawValue
        }
    }

    var sourceDirectory: String {
        switch self {
        case .dereverb: "mel_band_roformer/dereverb"
        case .denoise: "mel_band_roformer/denoise"
        }
    }

    var bundledConfigurationName: String {
        switch self {
        case .dereverb: "melband-dereverb"
        case .denoise: "melband-denoise"
        }
    }

    public var stemName: String {
        switch self {
        case .dereverb: "noreverb"
        case .denoise: "dry"
        }
    }

    public var weightsPin: ModelArtifactPin {
        switch self {
        case .dereverb:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/model.safetensors",
                byteCount: 912_885_624,
                sha256: "20af72ee52abdaf215dc5ece0c2b33a8304eff31c7b49a2d6730639f8a4aa75c"
            )
        case .denoise:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/model.safetensors",
                byteCount: 912_885_624,
                sha256: "721e2b13e30e968fcf23d5015f07a5d95dbf5cc5a66a4f8f8bc8f9fc11ffbb93"
            )
        }
    }

    var sourceConfigurationPin: ModelArtifactPin {
        switch self {
        case .dereverb:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/config.yaml",
                byteCount: 1_846,
                sha256: "66963a9d60756076506a230b4e503c553a3beb7b4e9a10e6bcc73dee9dbd4866"
            )
        case .denoise:
            ModelArtifactPin(
                filename: "\(sourceDirectory)/config.yaml",
                byteCount: 1_621,
                sha256: "4ed1e51a5af5013d705710b31d8d56a8c66d24f385a12155a1ab255c2d51cd5c"
            )
        }
    }

    var pins: [ModelArtifactPin] {
        [
            weightsPin,
            sourceConfigurationPin,
            RoFormerResources.licensePin,
            RoFormerResources.readmePin,
        ]
    }

    var expectedTensorCount: Int { 684 }
    var expectedScalarCount: Int { 228_203_172 }
}

public struct MelBandRoFormerCheckpoint: Sendable {
    public let profile: MelBandRoFormerProfile
    public let rootURL: URL
    public let weightsURL: URL
    public let sourceConfigurationURL: URL
    public let licenseURL: URL
    public let readmeURL: URL
    public let configuration: MelBandRoFormerConfiguration
}

public struct MelBandRoFormerResources: Sendable {
    public let rootURL: URL
    public let profile: MelBandRoFormerProfile

    public init(rootURL: URL, profile: MelBandRoFormerProfile) {
        self.rootURL = rootURL.standardizedFileURL
        self.profile = profile
    }

    public init(rootURL: URL, modelID: String) throws {
        self.init(rootURL: rootURL, profile: try MelBandRoFormerProfile.resolve(modelID: modelID))
    }

    public static func loadBundledConfiguration(
        profile: MelBandRoFormerProfile
    ) throws -> MelBandRoFormerConfiguration {
        let nested = Bundle.module.url(
            forResource: profile.bundledConfigurationName,
            withExtension: "json",
            subdirectory: "RoFormer"
        )
        guard let url = nested ?? Bundle.module.url(
            forResource: profile.bundledConfigurationName,
            withExtension: "json"
        ) else {
            throw RoFormerError.missingBundledConfiguration
        }
        let configuration = try JSONDecoder().decode(
            MelBandRoFormerConfiguration.self,
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

    public func resolve() throws -> MelBandRoFormerCheckpoint {
        let weightsURL = try profile.weightsPin.verify(in: rootURL)
        let sourceConfigurationURL = try profile.sourceConfigurationPin.verify(in: rootURL)
        let licenseURL = try RoFormerResources.licensePin.verify(in: rootURL)
        let readmeURL = try RoFormerResources.readmePin.verify(in: rootURL)
        let configuration = try Self.loadBundledConfiguration(profile: profile)
        return MelBandRoFormerCheckpoint(
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
