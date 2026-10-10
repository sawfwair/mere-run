import Foundation
import MLX
import AudioCore
import AudioCodecs
import AudioWhistleModel
import MereRunCore

/// Actor-confined resident inference over packed Cactus Quants or original FP32 weights.
public actor WhistleGenerator {
    private var model: WhistleModel?
    private var vocabulary: WhistleVocabulary?
    private var options = WhistleOptions()
    public static let modelID = "speech-asr-whistle"

    public init() {}

    public func transcribe(
        _ request: ASRRequest, modelID: String = WhistleGenerator.modelID, modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        _ = try Self.validate(task: request.task, language: request.language)
        guard request.maxTokens > 0 else { throw SpeechTranscriptionIssue("invalid_max_tokens", "Maximum tokens must be positive.") }
        let options = request.whistle ?? WhistleOptions()
        try await prepare(modelID: modelID, modelPath: modelPath, options: options, progressHandler: progressHandler)
        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Reading audio at 16 kHz..."))
        let samples = try AudioReader.readAudio(from: request.audioURL)
        return try await transcribePrepared(samples: samples, language: request.language, maxTokens: request.maxTokens,
                                      progressHandler: progressHandler)
    }

    public func prepare(
        modelID: String = WhistleGenerator.modelID, modelPath: String? = nil, options: WhistleOptions = WhistleOptions(),
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws {
        try options.validate()
        try Task.checkCancellation()
        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading native Whistle weights..."))
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelPath ?? modelID, defaultModelID: Self.modelID
        )
        try await MLXRequestStreams.withStream(isolation: self) {
            await Task.yield()
            try Task.checkCancellation()
            let root = resolved.url
            let cact = try Data(contentsOf: root.appendingPathComponent("whistle.cact"), options: .mappedIfSafe)
            let weights: WhistleWeights
            switch options.weights {
            case .cactus: weights = try WhistleWeights.load(cact: cact)
            case .fp32: weights = try WhistleWeights.load(from: root.appendingPathComponent("checkpoints/whistle.safetensors"))
            }
            vocabulary = try WhistleVocabulary(cact: cact)
            model = WhistleModel(weights: weights)
            self.options = options
            weights.evaluate()
        }
    }

    public func transcribePrepared(
        samples: [Float], language: String? = nil, maxTokens: Int = 448,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await MLXRequestStreams.withStream(isolation: self) {
            // Preserve task-local streams across optimized cooperative-executor hops.
            await Task.yield()
            try Task.checkCancellation()
            return try decodePrepared(samples: samples, language: language, maxTokens: maxTokens, progressHandler: progressHandler)
        }
    }

    private func decodePrepared(
        samples: [Float], language: String?, maxTokens: Int,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws -> ASRResult {
        defer { Memory.clearCache() }
        let language = try Self.validate(task: .transcribe, language: language)
        guard maxTokens > 0 else { throw SpeechTranscriptionIssue("invalid_max_tokens", "Maximum tokens must be positive.") }
        guard let model, let vocabulary else { throw SpeechTranscriptionIssue("model_not_prepared", "Prepare Whistle before decoding live audio.") }
        guard samples.allSatisfy(\.isFinite) else { throw SpeechTranscriptionIssue("invalid_audio", "Audio contains non-finite samples.") }
        let keywords = options.keywords.flatMap { keyword in
            let value = keyword.trimmingCharacters(in: .whitespacesAndNewlines)
            return [value, value.prefix(1).uppercased() + value.dropFirst()].map {
                vocabulary.encode(" " + $0).filter { (4..<8192).contains($0) }
            }
        }.filter { !$0.isEmpty }
        let duration = Double(samples.count) / 16000
        var transcript = ""
        var alignments: [ASRTokenAlignment] = []
        var detected = language
        var start = 0
        while start < samples.count {
            try Task.checkCancellation()
            let end = min(start + 480000, samples.count)
            let window = Array(samples[start..<end])
            if window.count >= 160, window.contains(where: { $0 != 0 }) {
                progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Encoding Whistle audio window..."))
                let mel = try WhistleFeatures(filterbank: vocabulary.filterbank).extract(window)
                let context = try model.encode(mel)
                var cache = WhistleDecoderCache()
                let languageLogits = try model.decode(token: 2, audio: context, cache: &cache, depth: options.decoderDepth)
                let languageIndex = language.flatMap { WhistleVocabulary.languages.firstIndex(of: $0) }
                    ?? argMax(languageLogits[8192..<8199]).item(Int.self)
                if detected == nil { detected = WhistleVocabulary.languages[languageIndex] }
                let generated = try model.search(audio: context, language: languageIndex, maxTokens: min(maxTokens, 318),
                                                 beams: options.beamSize, depth: options.decoderDepth, keywords: keywords) {
                    progressHandler?(ASRProgress(stage: .transcribing, tokensGenerated: $0))
                }
                let text = try vocabulary.decode(generated).trimmingCharacters(in: .whitespacesAndNewlines)
                let overlap = Self.overlap(transcript, text)
                transcript = Self.merge(transcript, text)
                if options.wordTimestamps, !generated.isEmpty {
                    let attention = try model.alignment(audio: context, language: languageIndex, tokens: generated, depth: options.decoderDepth)
                    let words = try WhistleAlignment.words(tokens: generated, attention: attention, frames: context.embedding.dim(0),
                                                           vocabulary: vocabulary, offset: Double(start) / 16000,
                                                           duration: Double(window.count) / 16000)
                    for word in words.dropFirst(overlap) {
                        let lower = max(alignments.last?.endSeconds ?? 0, word.startSeconds)
                        let upper = max(lower, word.endSeconds)
                        alignments.append(ASRTokenAlignment(text: word.text, startSeconds: lower, durationSeconds: upper - lower))
                    }
                }
            }
            if end == samples.count { break }
            start = end - 16000
        }
        return ASRResult(text: transcript, language: transcript.isEmpty ? nil : detected, duration: duration,
                         tokenAlignments: options.wordTimestamps ? alignments : nil)
    }

    public func unload() {
        model = nil
        vocabulary = nil
        Memory.clearCache()
    }

    public static func validate(task: ASRTask, language: String?) throws -> String? {
        guard task == .transcribe else {
            throw SpeechTranscriptionIssue("unsupported_task", "Whistle supports transcription; use Qwen for translation.")
        }
        guard let raw = language?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty,
              raw.lowercased() != "auto" else { return nil }
        guard let code = ASRBackendRouting.normalizeLanguageHint(raw), WhistleVocabulary.languages.contains(code) else {
            throw SpeechTranscriptionIssue("unsupported_language", "Whistle supports en, de, fr, es, it, nl, and pl.")
        }
        return code
    }

    /// Remove the common word overlap between consecutive 30-second windows.
    static func merge(_ previous: String, _ next: String) -> String {
        let left = previous.split(whereSeparator: \.isWhitespace).map(String.init)
        let right = next.split(whereSeparator: \.isWhitespace).map(String.init)
        return (left + right.dropFirst(Self.overlap(previous, next))).joined(separator: " ")
    }

    static func overlap(_ previous: String, _ next: String) -> Int {
        let left = previous.split(whereSeparator: \.isWhitespace).map(String.init)
        let right = next.split(whereSeparator: \.isWhitespace).map(String.init)
        for overlap in stride(from: min(32, min(left.count, right.count)), through: 1, by: -1) {
            if left.suffix(overlap).map({ $0.lowercased() }) == right.prefix(overlap).map({ $0.lowercased() }) {
                return overlap
            }
        }
        return 0
    }
}
