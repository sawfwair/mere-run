import Foundation

public struct MuScriptorConfiguration: Codable, Hashable, Sendable {
    public let dim: Int
    public let numHeads: Int
    public let numLayers: Int
    public let card: Int

    enum CodingKeys: String, CodingKey {
        case dim
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case card
    }

    public init(dim: Int, numHeads: Int, numLayers: Int, card: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.card = card
    }

    public static let small = MuScriptorConfiguration(
        dim: 768,
        numHeads: 12,
        numLayers: 14,
        card: 1_393
    )
    public static let medium = MuScriptorConfiguration(
        dim: 1_024,
        numHeads: 16,
        numLayers: 24,
        card: 1_395
    )
    public static let large = MuScriptorConfiguration(
        dim: 1_536,
        numHeads: 24,
        numLayers: 48,
        card: 1_395
    )

    public var headDimension: Int { dim / numHeads }
    public var hiddenDimension: Int { dim * 4 }

    public func validate() throws {
        guard dim > 0, numHeads > 0, numLayers > 0, card >= 1_393,
              dim.isMultiple(of: numHeads) else {
            throw MuScriptorError.invalidConfiguration
        }
    }
}

public enum MuScriptorVariant: String, CaseIterable, Sendable {
    case small
    case medium
    case large

    public var modelID: ModelResolver.ModelID {
        switch self {
        case .small: .muScriptorSmall
        case .medium: .muScriptorMedium
        case .large: .muScriptorLarge
        }
    }

    public var fallbackConfiguration: MuScriptorConfiguration {
        switch self {
        case .small: .small
        case .medium: .medium
        case .large: .large
        }
    }

    public static func resolve(modelID: String) -> MuScriptorVariant? {
        allCases.first { $0.modelID.rawValue == modelID || $0.rawValue == modelID }
    }
}

public enum MuScriptorError: LocalizedError {
    case invalidConfiguration
    case missingModelFile(URL)
    case unsupportedModel(String)
    case invalidInstrument(String)
    case invalidToken(Int)
    case generationLimit(chunk: Int, limit: Int)
    case malformedTokenStream(String)
    case invalidAudio(String)
    case invalidMIDI(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "Invalid MuScriptor model configuration."
        case .missingModelFile(let url):
            "Missing MuScriptor model file: \(url.path)"
        case .unsupportedModel(let model):
            "Unsupported MuScriptor model: \(model)"
        case .invalidInstrument(let instrument):
            "Unknown MuScriptor instrument: \(instrument)"
        case .invalidToken(let token):
            "MuScriptor emitted invalid token \(token)."
        case .generationLimit(let chunk, let limit):
            "MuScriptor chunk \(chunk) did not emit EOS within \(limit) tokens."
        case .malformedTokenStream(let message):
            "Malformed MuScriptor token stream: \(message)"
        case .invalidAudio(let message):
            "Could not analyze MuScriptor audio: \(message)"
        case .invalidMIDI(let message):
            "Could not encode MuScriptor MIDI: \(message)"
        }
    }
}
