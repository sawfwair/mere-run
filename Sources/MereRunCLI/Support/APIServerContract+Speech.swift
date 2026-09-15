import Foundation
import MereRunCore
import AudioCore
import AudioTTS

extension APIServerContract {
    static let maxSpeechPromptUTF8Bytes = 32 * 1_024
    static let defaultSpeechModelID = Qwen3TTSResources.defaultModelId

    static func decodeSpeechRequest(from data: Data) throws -> OpenAIAudioSpeechRequest {
        try decodeJSONRequest(OpenAIAudioSpeechRequest.self, from: data)
    }

    struct SpeechPlan: Equatable, Sendable {
        let modelID: String
        let input: String
        let voiceDescription: String
        let responseFormat: String
        let speed: Float
        let temperature: Float

        func synthesisPlan(outputURL: URL) throws -> SpeechSynthesisPlan {
            try SpeechSynthesisPlan(request: TTSRequest(
                text: input, voiceDescription: voiceDescription, speed: speed,
                temperature: temperature, outputURL: outputURL
            ))
        }

        func modelSelection() throws -> SpeechSynthesisModelSelection {
            do {
                return try SpeechSynthesisModelSelection.resolve(modelID)
            } catch Qwen3TTSError.unsupportedModelId {
                throw APIRequestValidationError.invalidField(
                    "model", "use a mere.run TTS model id or a local Qwen3-TTS model path"
                )
            }
        }
    }

    static func speechPlan(from request: OpenAIAudioSpeechRequest) throws -> SpeechPlan {
        let input = request.input.trimmingCharacters(in: .whitespacesAndNewlines)
        let responseFormat = try speechResponseFormat(request.response_format)
        let speed = try speechSpeed(request.speed)
        let temperature = request.temperature ?? TTSRequest.defaultTemperature
        do {
            try SpeechSynthesisPlan.validateParameters(text: input, temperature: temperature, speed: speed)
        } catch SpeechSynthesisError.invalidInput(let field, let message) {
            throw APIRequestValidationError.invalidField(field == .text ? "input" : field.rawValue, message)
        }
        let voiceDescription = voiceDescription(for: request.voice, instructions: request.instructions)
        var promptUTF8Bytes = 0
        for component in [input, voiceDescription] {
            let componentBytes = component.utf8.count
            guard componentBytes <= maxSpeechPromptUTF8Bytes - promptUTF8Bytes else {
                throw APIRequestValidationError.invalidField(
                    "input",
                    "input and voice instructions must total at most \(maxSpeechPromptUTF8Bytes) UTF-8 bytes"
                )
            }
            promptUTF8Bytes += componentBytes
        }
        return SpeechPlan(
            modelID: normalizedSpeechModelID(request.model),
            input: input,
            voiceDescription: voiceDescription,
            responseFormat: responseFormat,
            speed: speed,
            temperature: temperature
        )
    }

    private static func speechResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "wav"
        guard ["wav", "mp3", "opus", "aac", "flac"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected wav, mp3, opus, aac, or flac"
            )
        }
        return value
    }

    static func speechContentType(for responseFormat: String) -> String {
        switch responseFormat {
        case "mp3":
            return "audio/mpeg"
        case "opus":
            return "audio/ogg"
        case "aac":
            return "audio/aac"
        case "flac":
            return "audio/flac"
        default:
            return "audio/wav"
        }
    }

    private static func speechSpeed(_ rawValue: Double?) throws -> Float {
        do {
            return try SpeechSynthesisPlan.validatedSpeed(rawValue ?? Double(TTSRequest.defaultSpeed))
        } catch SpeechSynthesisError.invalidInput(_, let message) {
            throw APIRequestValidationError.invalidField("speed", message)
        }
    }

    private static func normalizedSpeechModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultSpeechModelID)
        switch modelID.lowercased() {
        case "tts-1", "tts-1-hd", "gpt-4o-mini-tts":
            return defaultSpeechModelID
        default:
            return modelID
        }
    }

    private static func voiceDescription(for rawVoice: String?, instructions: String?) -> String {
        let instructionText = normalizedOptional(instructions)
        let voice = normalizedOptional(rawVoice)?.lowercased() ?? "nova"
        let base: String
        switch voice {
        case "alloy":
            base = "A balanced, natural voice with clear pronunciation"
        case "ash":
            base = "A calm, low voice with a steady delivery"
        case "ballad":
            base = "A warm, expressive voice with a storytelling cadence"
        case "coral":
            base = "A bright, friendly voice with gentle energy"
        case "echo":
            base = "A clear male voice with an even, conversational tone"
        case "fable":
            base = "A warm narrative voice with a measured pace"
        case "nova":
            base = TTSRequest.defaultVoiceDescription
        case "onyx":
            base = "A deep, confident voice with crisp articulation"
        case "sage":
            base = "A thoughtful, composed voice with soft emphasis"
        case "shimmer":
            base = "A bright, gentle voice with smooth pronunciation"
        default:
            base = rawVoice?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? TTSRequest.defaultVoiceDescription
        }
        if let instructionText {
            return "\(base). \(instructionText)"
        }
        return base
    }
}
