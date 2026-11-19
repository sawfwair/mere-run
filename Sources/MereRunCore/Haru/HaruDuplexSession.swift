import AudioHaruModel
import Foundation
import Hub
import MLX
import Tokenizers

public enum HaruResources {
    public static let modelID = ManagedModelID.haru4B.rawValue
    static let repository = "higurashi-ai/Haru-4B"
    static let revision = "8d6e83bcd3701a76fc27a764965665da1df7da78"
    static let files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
}

public struct HaruAudioFrame: Sendable {
    public let audio: [Float]
    public let tokenID: Int
    public let text: String
}

/// A single conversation with a fixed character card and voice. Start once,
/// then submit successive mono PCM frames in capture order.
public actor HaruDuplexSession {
    public static let sampleRate = HaruConfiguration.sampleRate
    public static let frameSamples = HaruConfiguration.frameSamples

    private let session: HaruSession
    private let stream = MLX.Stream.Context()
    private let tokenizer: any Tokenizer

    public init(modelDirectory: URL, seed: UInt64? = nil, temperature: Float = 0.7) throws {
        guard temperature.isFinite, temperature >= 0 else {
            throw HaruSessionConfigurationError.temperature
        }
        let tokenizerConfig = try HubApi.shared.configuration(fileURL: modelDirectory.appending(path: "tokenizer_config.json"))
        let tokenizerData = try HubApi.shared.configuration(fileURL: modelDirectory.appending(path: "tokenizer.json"))
        tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        session = try HaruSession(
            model: HaruModel.load(from: modelDirectory),
            seed: seed ?? UInt64.random(in: 0...UInt64(UInt32.max)),
            temperature: temperature
        )
    }

    public func start(
        characterCard: String = "", referenceAudio: [Float] = [], speakFirst: Bool = false
    ) async throws -> HaruAudioFrame {
        let ids = characterCard.isEmpty ? [] : tokenizer.encode(
            text: "<|im_start|>system\n\(characterCard)<|im_end|>\n",
            addSpecialTokens: false
        )
        return try await MLX.Stream.withDefaultStream(stream) {
            let result = try session.start(card: ids, reference: referenceAudio, speakFirst: speakFirst, decode: decode)
            return HaruAudioFrame(audio: result.audio, tokenID: result.tokenID, text: result.text)
        }
    }

    public func step(_ userAudio: [Float]) async throws -> HaruAudioFrame {
        try await MLX.Stream.withDefaultStream(stream) {
            let result = try session.step(userAudio, decode: decode)
            return HaruAudioFrame(audio: result.audio, tokenID: result.tokenID, text: result.text)
        }
    }

    private func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
    }
}

private enum HaruSessionConfigurationError: LocalizedError {
    case temperature

    var errorDescription: String? { "Haru text temperature must be finite and nonnegative." }
}
