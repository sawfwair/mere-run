import Foundation
import MLX
import AudioCore
import AudioCodecs
import AudioNemotronTranscribeModel
import MereRunCore
import MediaIO
@preconcurrency import Tokenizers

public actor NemotronTranscribeGenerator {
    private var model: NemotronTranscribeModel?
    private var tokenizer: (any Tokenizer)?
    private var loadedRoot: URL?

    public init() {}

    public func transcribe(
        _ request: ASRRequest, modelID: String = NemotronTranscribeResources.modelID, modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let language = try Self.validate(language: request.language)
        guard request.maxTokens > 0 else {
            throw SpeechTranscriptionIssue("invalid_max_tokens", "Maximum transcription tokens must be positive.")
        }
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelPath ?? modelID, defaultModelID: NemotronTranscribeResources.modelID)
        return try await MLXRequestStreams.withStream(isolation: self) {
            await Task.yield()
            try Task.checkCancellation()
            if loadedRoot != resolved.url {
                progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading native Nemotron Transcribe BF16 weights..."))
                let config = try NemotronTranscribeConfig.load(at: resolved.url)
                let loadedTokenizer = try await AutoTokenizer.from(modelFolder: resolved.url)
                guard loadedTokenizer.convertTokenToId("<|im_end|>") == config.eosTokenId.first,
                      loadedTokenizer.convertTokenToId("<|audio|>") == 131_072 else {
                    throw SpeechTranscriptionIssue("invalid_tokenizer", "Tokenizer does not match the released Nemotron Transcribe checkpoint.")
                }
                let loadedModel = try NemotronTranscribeModel(root: resolved.url)
                model = loadedModel
                tokenizer = loadedTokenizer
                loadedRoot = resolved.url
            }
            try Task.checkCancellation()
            progressHandler?(ASRProgress(stage: .loadingAudio, message: "Reading audio at 16 kHz..."))
            let audio = try MediaAudioIO.decode(request.audioURL, targetSampleRate: 16_000, channels: 1)
            return try generate(samples: audio.samples, request: request, language: language, progressHandler: progressHandler)
        }
    }

    private func generate(samples: [Float], request: ASRRequest, language: String?,
                          progressHandler: (@Sendable (ASRProgress) -> Void)?) throws -> ASRResult {
        guard let model, let tokenizer else { preconditionFailure("Model must be loaded before generation") }
        defer { Memory.clearCache() }
        progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Encoding audio and speaker features..."))
        let audio = try model.encodeAudio(samples) { completed, total in
            progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Encoded audio chunk \(completed)/\(total)."))
        }
        let prompt = Self.prompt(task: request.task, language: language)
        let prefix = tokenizer.encode(text: prompt.prefix, addSpecialTokens: false)
        let suffix = tokenizer.encode(text: prompt.suffix, addSpecialTokens: false)
        let inputCount = prefix.count + audio.dim(1) + suffix.count
        guard inputCount + request.maxTokens <= NemotronTranscribeConfig.maximumContext else {
            throw SpeechTranscriptionIssue("context_exceeded", "Audio, prompt, and requested output exceed the 65,536-token context.")
        }
        let embeddings = concatenated([model.inputEmbeddings(prefix), audio, model.inputEmbeddings(suffix)], axis: 1)
        let cache = model.makeCache()
        progressHandler?(ASRProgress(stage: .transcribing, message: "Decoding transcript..."))
        var logits = try model.prefill(embeddings: embeddings, cache: cache)
        var generated: [Int] = []
        for step in 0..<request.maxTokens {
            try Task.checkCancellation()
            let token = argMax(logits.reshaped(-1), axis: -1).item(Int.self)
            if model.config.eosTokenId.contains(token) { break }
            generated.append(token)
            if (step + 1).isMultiple(of: 10) {
                progressHandler?(ASRProgress(stage: .transcribing, tokensGenerated: generated.count))
            }
            if step + 1 < request.maxTokens { logits = try model.decode(token: token, cache: cache) }
        }
        try Task.checkCancellation()
        return ASRResult(text: tokenizer.decode(tokens: generated, skipSpecialTokens: false).trimmingCharacters(in: .whitespacesAndNewlines),
                         language: request.task == .translate ? "en" : language, duration: Double(samples.count) / 16_000)
    }

    public func unload() {
        model = nil
        tokenizer = nil
        loadedRoot = nil
        Memory.clearCache()
    }

    public static func validate(language: String?) throws -> String? {
        guard let raw = language?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty,
              raw.lowercased() != "auto" else { return nil }
        guard let normalized = ASRBackendRouting.normalizeLanguageHint(raw), languages.contains(normalized) else {
            throw SpeechTranscriptionIssue("unsupported_language", "Nemotron Transcribe supports 40 languages; unsupported hint: \(raw).")
        }
        return normalized
    }

    static func prompt(task: ASRTask, language: String?) -> (prefix: String, suffix: String) {
        let instruction = task == .translate ? "Please translate the given speech to English."
            : "Transcribe the following audio." + (language.map { " Language: \($0)." } ?? "")
        return ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" + instruction + "\n",
                "<|im_end|>\n<|im_start|>assistant\n<think></think>")
    }

    private static let languages: Set<String> = [
        "cs", "nl", "en", "fr", "de", "hi", "it", "ko", "pl", "pt", "ru", "sk", "es", "bg", "hr", "da", "et", "fi", "hu", "ja",
        "lv", "lt", "ro", "sl", "sv", "th", "tr", "uk", "ar", "bn", "zh", "el", "he", "id", "mt", "mr", "tl", "ta", "te", "vi"
    ]
}
