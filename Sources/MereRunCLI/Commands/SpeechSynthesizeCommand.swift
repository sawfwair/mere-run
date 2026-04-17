import ArgumentParser
import Foundation
import AudioCore
import AudioCodecs
import AudioSTT
import AudioTTS
import MereRunCore

// MARK: - Speech Synthesize Command

enum TalkModeOption: String, ExpressibleByArgument {
    case style
    case clone
}

struct SpeechSynthesize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "synthesize",
        abstract: "Generate speech from text using Qwen3-TTS.",
        discussion: """
        Converts text to speech using Qwen3-TTS-12Hz-1.7B-VoiceDesign.
        Model is downloaded from R2 on first use.

        Example:
          mere.run speech synthesize "Hello, world!" -o hello.wav
          mere.run speech synthesize "Welcome to mere.run." --voice "A calm British male voice" -o welcome.wav
        """
    )

    @Argument(help: "Text to convert to speech.")
    var text: String

    @Option(name: [.customShort("o"), .long], help: "Output WAV file path (required).")
    var output: String

    @Option(name: [.customShort("m"), .long], help: "Canonical model id (speech-tts-qwen3-nano) or a local model path.")
    var model: String = Qwen3TTSResources.defaultModelId

    @Option(name: [.customShort("v"), .long], help: "Voice description for speech style.")
    var voice: String = "A calm female voice with clear pronunciation"

    @Option(name: [.long], help: "Voice mode: style or clone.")
    var mode: TalkModeOption = .style

    @Option(name: [.long], help: "Saved profile id or name (clone mode).")
    var profile: String?

    @Option(name: [.long], help: "Reference audio file path (clone mode).")
    var refAudio: String?

    @Option(name: [.long], help: "Reference transcript override (clone mode).")
    var refText: String?

    @Option(name: [.long], help: "Language hint (default: auto).")
    var language: String = "auto"

    @Option(name: [.long], help: "Save this reference as a reusable profile name.")
    var saveProfile: String?

    @Option(name: [.long], help: "Sampling temperature (default: 0.6).")
    var temperature: Float = 0.6

    @Flag(name: [.long], help: "Enable streaming TTS mode.")
    var stream: Bool = false

    @Option(name: [.customLong("stream-chunk-tokens")], help: "Token interval between streaming audio chunk emits.")
    var streamChunkTokens: Int = 25

    @Flag(name: [.short, .long], help: "Quiet mode (suppress progress output).")
    var quiet: Bool = false

    func run() async throws {
        if stream {
            guard streamChunkTokens > 0 else {
                throw ValidationError("--stream-chunk-tokens must be > 0.")
            }
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL

        // Ensure output directory exists
        let outputDir = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let modelSelection = try resolveModelSelection()
        let generator = Qwen3TTSGenerator(modelId: modelSelection.modelId)
        let profileStore = VoiceProfileStore()

        let request = try await buildRequest(outputURL: outputURL, profileStore: profileStore)

        if stream {
            try await runStreaming(request: request, generator: generator, modelPath: modelSelection.modelPath)
            return
        }

        if !quiet {
            FileHandle.standardError.write(Data("Generating speech with native Qwen3-TTS...\n".utf8))
        }

        let progressHandler: (@Sendable (TTSProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else {
            progressHandler = { progress in
                var message = "[\(progress.stage.rawValue)]"
                if progress.tokensGenerated > 0 {
                    message += " \(progress.tokensGenerated) tokens"
                }
                if let msg = progress.message {
                    message += " \(msg)"
                }
                FileHandle.standardError.write(Data("\(message)\n".utf8))
            }
        }

        let result = try await generator.generate(
            request,
            modelPath: modelSelection.modelPath,
            progressHandler: progressHandler
        )

        if !quiet {
            let durationStr = String(format: "%.2f", result.duration)
            FileHandle.standardError.write(Data("Audio saved to: \(result.audioURL.path)\n".utf8))
            FileHandle.standardError.write(Data("Duration: \(durationStr)s @ \(result.sampleRate)Hz\n".utf8))
        }

        print(result.audioURL.path)
    }

    private func runStreaming(
        request: TTSRequest,
        generator: Qwen3TTSGenerator,
        modelPath: String?
    ) async throws {
        if !quiet {
            FileHandle.standardError.write(Data("Generating speech with native Qwen3-TTS (streaming)...\n".utf8))
        }

        let stream = generator.generateStream(
            request,
            options: TTSStreamingOptions(
                chunkTokenInterval: streamChunkTokens,
                emitTokenEvents: !quiet
            ),
            modelPath: modelPath
        )

        var writer: StreamingWAVWriter?
        var tokenCount = 0
        var result: TTSResult?

        for try await event in stream {
            switch event {
            case .token:
                tokenCount += 1
                if !quiet && tokenCount % 25 == 0 {
                    FileHandle.standardError.write(
                        Data("[generating] \(tokenCount) tokens\n".utf8)
                    )
                }

            case .audioChunk(let samples, let sampleRate):
                if writer == nil {
                    writer = try StreamingWAVWriter(outputURL: request.outputURL, sampleRate: sampleRate)
                }
                try writer?.append(samples: samples)

            case .completed(let completed):
                result = completed
            }
        }

        guard let result else {
            throw ValidationError("Streaming TTS completed without a final result.")
        }

        if !quiet {
            let durationStr = String(format: "%.2f", result.duration)
            FileHandle.standardError.write(Data("Audio saved to: \(result.audioURL.path)\n".utf8))
            FileHandle.standardError.write(Data("Duration: \(durationStr)s @ \(result.sampleRate)Hz\n".utf8))
        }

        print(result.audioURL.path)
    }

    private func buildRequest(
        outputURL: URL,
        profileStore: VoiceProfileStore
    ) async throws -> TTSRequest {
        switch mode {
        case .style:
            return TTSRequest(
                text: text,
                voiceDescription: voice,
                voiceMode: .style,
                cloneReference: nil,
                language: normalizedLanguageOrAuto(language),
                speed: 1.0,
                temperature: temperature,
                outputURL: outputURL
            )
        case .clone:
            let cloneReference = try await resolveCloneReference(profileStore: profileStore)
            return TTSRequest(
                text: text,
                voiceDescription: voice,
                voiceMode: .clone,
                cloneReference: cloneReference,
                language: normalizedLanguageOrAuto(language),
                speed: 1.0,
                temperature: temperature,
                outputURL: outputURL
            )
        }
    }

    private func resolveCloneReference(profileStore: VoiceProfileStore) async throws -> TTSCloneReference {
        if let profile = normalized(profile) {
            guard let saved = try await profileStore.profile(matching: profile) else {
                throw ValidationError("Profile not found: \(profile)")
            }

            let audioURL = await profileStore.referenceAudioURL(for: saved)
            let transcript = try await resolveTranscript(
                refText: refText,
                fallback: saved.transcript,
                audioURL: audioURL
            )

            return TTSCloneReference(
                audioURL: audioURL,
                transcript: transcript,
                language: saved.language ?? normalizedLanguage(language),
                profileID: saved.id
            )
        }

        guard let refAudio = normalized(refAudio) else {
            throw ValidationError("Clone mode requires --profile or --ref-audio.")
        }

        let audioURL = URL(fileURLWithPath: refAudio).standardizedFileURL
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("Reference audio not found: \(audioURL.path)")
        }

        let transcript = try await resolveTranscript(refText: refText, fallback: nil, audioURL: audioURL)

        var profileID: UUID?
        if let saveProfileName = normalized(saveProfile) {
            let created = try await profileStore.createProfile(
                name: saveProfileName,
                referenceAudioURL: audioURL,
                transcript: transcript,
                language: normalizedLanguage(language)
            )
            profileID = created.id
            if !quiet {
                FileHandle.standardError.write(Data("Saved profile '\(created.name)' (\(created.id.uuidString))\n".utf8))
            }
        }

        return TTSCloneReference(
            audioURL: audioURL,
            transcript: transcript,
            language: normalizedLanguage(language),
            profileID: profileID
        )
    }

    private func resolveTranscript(refText: String?, fallback: String?, audioURL: URL) async throws -> String {
        if let provided = normalized(refText) {
            return provided
        }

        if let fallback = normalized(fallback) {
            return fallback
        }

        if !quiet {
                    FileHandle.standardError.write(Data("Transcribing reference audio with the speech transcriber...\n".utf8))
                }
        let asrRequest = ASRRequest(
            audioURL: audioURL,
            language: normalizedLanguage(language),
            task: .transcribe,
            maxTokens: 448
        )
        let execution = try await CLIASRRouting.transcribe(
            request: asrRequest,
            preferredBackend: .auto,
            modelOverride: nil,
            progressHandler: nil
        )
        let result = execution.result
        let transcript = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else {
            throw ValidationError("Auto-transcription returned empty text. Use --ref-text.")
        }
        return transcript
    }

    private func resolveModelSelection() throws -> (modelId: String, modelPath: String?) {
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        let fm = FileManager.default
        let asPath = URL(fileURLWithPath: trimmed).standardizedFileURL
        if fm.fileExists(atPath: asPath.path) {
            return (Qwen3TTSResources.defaultModelId, asPath.path)
        }
        return (trimmed.isEmpty ? Qwen3TTSResources.defaultModelId : trimmed, nil)
    }

    private func normalized(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func normalizedLanguage(_ value: String?) -> String? {
        guard let normalized = normalized(value) else { return nil }
        if normalized.lowercased() == "auto" {
            return nil
        }
        return normalized
    }

    private func normalizedLanguageOrAuto(_ value: String?) -> String {
        normalized(value) ?? "auto"
    }
}
