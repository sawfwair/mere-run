import Foundation

public enum SpeechSynthesisField: String, Sendable {
    case text, temperature, speed, cloneReference, output, streamChunkTokens
}

public enum SpeechSynthesisError: LocalizedError, Sendable {
    case invalidInput(SpeechSynthesisField, String)
    case streamingUnsupported
    case wrongExecutionMode
    case emptyAudio
    case invalidStream(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(_, let message), .invalidStream(let message): message
        case .streamingUnsupported: "The speech executor does not support streaming."
        case .wrongExecutionMode: "Use the synthesis operation that matches the plan's streaming mode."
        case .emptyAudio: "Speech synthesis produced no audio samples."
        }
    }
}

/// Retains the validated request and the entry point's explicit WAV policy.
/// Model selection and loading belong to AudioTTS; transport aliases belong to adapters.
public struct SpeechSynthesisPlan: Sendable, Hashable {
    public let request: TTSRequest
    public let streamingOptions: TTSStreamingOptions?
    public let exportPlan: AudioExportPlan

    public init(request: TTSRequest, streamingOptions: TTSStreamingOptions? = nil) throws {
        try Self.validateParameters(text: request.text, temperature: request.temperature, speed: request.speed)
        guard request.outputURL.isFileURL else {
            throw SpeechSynthesisError.invalidInput(.output, "Speech output must be a local file URL.")
        }
        if request.voiceMode == .clone {
            guard let reference = request.cloneReference else {
                throw SpeechSynthesisError.invalidInput(.cloneReference, "Clone mode requires a reference audio file and transcript.")
            }
            guard reference.audioURL.isFileURL else {
                throw SpeechSynthesisError.invalidInput(.cloneReference, "Reference audio must be a local file URL.")
            }
            guard !reference.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw SpeechSynthesisError.invalidInput(.cloneReference, "Reference transcript must not be empty.")
            }
        }
        try Self.validateStreamingOptions(streamingOptions)
        self.request = request
        self.streamingOptions = streamingOptions
        exportPlan = try AudioExportPlan(
            options: streamingOptions == nil ? .referencePCM16 : .speechStreaming,
            clipping: streamingOptions == nil ? .unitRange : .preserveFloatHeadroom
        )
    }

    /// Validates scalar input before profile lookup, reference transcription, or model loading.
    public static func validateParameters(text: String, temperature: Float, speed: Float) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SpeechSynthesisError.invalidInput(.text, "Speech text must not be empty.")
        }
        guard temperature.isFinite, (0...2).contains(temperature) else {
            throw SpeechSynthesisError.invalidInput(.temperature, "Temperature must be finite and between 0 and 2.")
        }
        _ = try validatedSpeed(Double(speed))
    }

    /// Checks the wire value before narrowing it to the native Float field.
    public static func validatedSpeed(_ speed: Double) throws -> Float {
        guard speed.isFinite, (0.25...4).contains(speed) else {
            throw SpeechSynthesisError.invalidInput(.speed, "Speed must be finite and between 0.25 and 4.0.")
        }
        return Float(speed)
    }

    public static func validateStreamingOptions(_ options: TTSStreamingOptions?) throws {
        if let options, options.chunkTokenInterval <= 0 {
            throw SpeechSynthesisError.invalidInput(.streamChunkTokens, "Streaming token interval must be greater than zero.")
        }
    }

    public func validateForExecution(fileManager: FileManager = .default) throws {
        try Task.checkCancellation()
        if request.voiceMode == .clone, let reference = request.cloneReference {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: reference.audioURL.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                throw SpeechSynthesisError.invalidInput(.cloneReference, "Reference audio not found: \(reference.audioURL.path)")
            }
        }
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: request.outputURL.path, isDirectory: &isDirectory), isDirectory.boolValue {
            throw SpeechSynthesisError.invalidInput(.output, "Speech output is a directory: \(request.outputURL.path)")
        }
    }
}
