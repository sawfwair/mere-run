import Foundation
import MLX

public struct ACEStepMusicUnderstandingMetadata: Sendable, Hashable, Codable {
    public var caption: String?
    public var lyrics: String?
    public var bpm: Int?
    public var durationSeconds: Float?
    public var keyscale: String?
    public var language: String?
    public var timesignature: String?

    public init(
        caption: String? = nil,
        lyrics: String? = nil,
        bpm: Int? = nil,
        durationSeconds: Float? = nil,
        keyscale: String? = nil,
        language: String? = nil,
        timesignature: String? = nil
    ) {
        self.caption = caption
        self.lyrics = lyrics
        self.bpm = bpm
        self.durationSeconds = durationSeconds
        self.keyscale = keyscale
        self.language = language
        self.timesignature = timesignature
    }

    public var understandingSummary: String {
        var fields: [String] = []
        if let bpm {
            fields.append("bpm=\(bpm)")
        }
        if let keyscale {
            fields.append("keyscale=\(keyscale)")
        }
        if let timesignature {
            fields.append("timesignature=\(timesignature)")
        }
        if let language {
            fields.append("language=\(language)")
        }
        if let durationSeconds {
            fields.append("duration=\(durationSeconds)s")
        }
        return fields.isEmpty ? "no metadata detected" : fields.joined(separator: ", ")
    }
}

public struct ACEStepMusicUnderstandingResult: Sendable, Hashable {
    public var audioCodes: String
    public var metadata: ACEStepMusicUnderstandingMetadata
    public var lmResult: ACEStep5HzLMResult

    public init(
        audioCodes: String,
        metadata: ACEStepMusicUnderstandingMetadata,
        lmResult: ACEStep5HzLMResult
    ) {
        self.audioCodes = audioCodes
        self.metadata = metadata
        self.lmResult = lmResult
    }
}

public struct ACEStepMusicPlan: Sendable, Hashable {
    public var metadata: ACEStepMusicUnderstandingMetadata
    public var lmResult: ACEStep5HzLMResult

    public init(
        metadata: ACEStepMusicUnderstandingMetadata,
        lmResult: ACEStep5HzLMResult
    ) {
        self.metadata = metadata
        self.lmResult = lmResult
    }
}

extension ACEStepPipeline {
    public func planMusic(
        caption: String,
        lyrics: String,
        instruction: String = ACEStepLMInstructions.defaultInstruction,
        userMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init(),
        lmConfig: ACEStep5HzLMGenerationConfig = .init(
            maxNewTokens: 1_024,
            temperature: 0.85,
            topP: 0.9
        )
    ) throws -> ACEStepMusicPlan {
        guard let lm else {
            throw PipelineError.lmNotConfigured
        }
        let sampler = lm.makeConstrainedSampler(
            enabled: true,
            skipCaption: false,
            skipLanguage: false,
            stopAtReasoning: true,
            generationPhase: .codes,
            userMetadata: userMetadata
        )
        let result = lm.generateConstrained(
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            systemInstruction: instruction,
            config: lmConfig,
            sampler: sampler
        )
        return ACEStepMusicPlan(
            metadata: Self.parseUnderstandingOutput(result.generatedText),
            lmResult: result
        )
    }

    public func audioCodeString(
        sourceAudio48kHz: MLXArray,
        durationSeconds: Float
    ) throws -> String {
        let targetFrames = max(1, Int((Double(durationSeconds) * 25.0).rounded()))
        let sourceLatents = try normalizeSourceLatents(
            nil,
            sourceAudio48kHz: sourceAudio48kHz,
            targetFrames: targetFrames
        )
        let attentionMask = MLXArray.ones([sourceLatents.dim(0), sourceLatents.dim(1)], dtype: .int32)
        let (_, indices, _) = tokenizeForLMHints(
            hiddenStates: sourceLatents,
            attentionMask: attentionMask,
            silenceLatent: silenceLatent
        )
        MLX.eval(indices)
        return Self.audioCodeString(fromIndices: indices)
    }

    public func understandSourceAudio(
        sourceAudio48kHz: MLXArray,
        durationSeconds: Float,
        lmConfig: ACEStep5HzLMGenerationConfig = .init(maxNewTokens: 2048, temperature: 0.3)
    ) throws -> ACEStepMusicUnderstandingResult {
        guard let lm else {
            throw PipelineError.lmNotConfigured
        }

        let audioCodes = try audioCodeString(
            sourceAudio48kHz: sourceAudio48kHz,
            durationSeconds: durationSeconds
        )
        let promptTokens = lm.buildPromptTokens(
            systemInstruction: ACEStepLMInstructions.understandInstruction,
            userContent: audioCodes
        )
        let sampler = lm.makeConstrainedSampler(
            enabled: true,
            skipCaption: false,
            skipLanguage: false,
            stopAtReasoning: false,
            generationPhase: .understand
        )
        let result = lm.generateConstrained(
            promptTokens: promptTokens,
            config: lmConfig,
            sampler: sampler
        )
        let metadata = Self.parseUnderstandingOutput(result.generatedText)
        return ACEStepMusicUnderstandingResult(audioCodes: audioCodes, metadata: metadata, lmResult: result)
    }

    public static func audioCodeString(fromIndices indices: MLXArray) -> String {
        indices
            .reshaped(-1)
            .asType(.int32)
            .asArray(Int32.self)
            .map { "<|audio_code_\(max(0, min(Int($0), 63_999)))|>" }
            .joined()
    }

    public static func parseUnderstandingOutput(_ output: String) -> ACEStepMusicUnderstandingMetadata {
        var metadata = ACEStepMusicUnderstandingMetadata()
        metadata.lyrics = parsedLyrics(from: output)

        let reasoningText = parsedReasoningText(from: output)
        var currentField: UnderstandingField?
        var currentLines: [String] = []

        func saveCurrentField() {
            guard let field = currentField else {
                currentLines = []
                return
            }
            let value = currentLines.joined(separator: "\n")
            switch field {
            case .bpm:
                metadata.bpm = parseInt(value)
            case .caption:
                metadata.caption = cleanMetadataString(value)
            case .duration:
                metadata.durationSeconds = parseFloat(value)
            case .keyscale:
                metadata.keyscale = cleanMetadataString(value)
            case .language:
                metadata.language = normalizedVocalLanguage(value)
            case .timesignature:
                metadata.timesignature = cleanMetadataString(value)
            case .genres:
                break
            }
            currentField = nil
            currentLines = []
        }

        for line in reasoningText.split(separator: "\n", omittingEmptySubsequences: false) {
            let rawLine = String(line)
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("<") {
                continue
            }
            if let separatorIndex = rawLine.firstIndex(of: ":"),
               rawLine.first?.isWhitespace != true {
                saveCurrentField()
                let key = rawLine[..<separatorIndex].trimmingCharacters(in: .whitespacesAndNewlines)
                currentField = UnderstandingField(rawValue: key.lowercased())
                let valueStart = rawLine.index(after: separatorIndex)
                let firstValue = rawLine[valueStart...]
                if !firstValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    currentLines.append(String(firstValue))
                }
            } else if rawLine.first?.isWhitespace == true, currentField != nil {
                currentLines.append(rawLine.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        saveCurrentField()

        return metadata
    }

    private enum UnderstandingField: String {
        case bpm
        case caption
        case duration
        case genres
        case keyscale
        case language
        case timesignature
    }

    private static func parsedReasoningText(from output: String) -> String {
        guard
            let start = output.range(of: "<think>")?.upperBound,
            let end = output.range(of: "</think>", range: start..<output.endIndex)?.lowerBound
        else {
            let firstAudioCode = output.range(of: "<|audio_code_")?.lowerBound ?? output.endIndex
            return String(output[..<firstAudioCode]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(output[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func parsedLyrics(from output: String) -> String? {
        guard let end = output.range(of: "</think>")?.upperBound else {
            return nil
        }
        return cleanMetadataString(String(output[end...]))
    }

    private static func cleanMetadataString(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != "N/A", trimmed.lowercased() != "unknown" else {
            return nil
        }
        return trimmed
    }

    private static func normalizedVocalLanguage(_ value: String) -> String? {
        guard let cleaned = cleanMetadataString(value) else {
            return nil
        }
        let normalized = cleaned.lowercased()
        return validVocalLanguages.contains(normalized) ? normalized : nil
    }

    private static let validVocalLanguages: Set<String> = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
        "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
        "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
        "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
        "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    ]

    private static func parseInt(_ value: String) -> Int? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if let direct = Int(trimmed) {
            return direct
        }
        let prefix = trimmed.prefix { $0.isNumber }
        return prefix.isEmpty ? nil : Int(prefix)
    }

    private static func parseFloat(_ value: String) -> Float? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if let direct = Float(trimmed) {
            return direct
        }
        let prefix = trimmed.prefix { $0.isNumber || $0 == "." }
        return prefix.isEmpty ? nil : Float(prefix)
    }
}
