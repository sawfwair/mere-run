import ArgumentParser
import Foundation
import AudioCore
import AudioCodecs
import AudioSTT
import MereRunCore

// MARK: - Speech Transcribe Command

enum SpeechBackendOption: String, ExpressibleByArgument {
    case auto
    case parakeet
    case qwen

    var backend: ASRBackend {
        switch self {
        case .auto: return .auto
        case .parakeet: return .parakeet
        case .qwen: return .qwen
        }
    }
}

enum SpeechTaskOption: String, ExpressibleByArgument {
    case transcribe
    case translate

    var task: ASRTask {
        switch self {
        case .transcribe: return .transcribe
        case .translate: return .translate
        }
    }
}

struct SpeechTranscribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe or translate speech to text using native ASR backends.",
        discussion: """
        Supports native Parakeet and Qwen backends with policy routing.
        In auto mode, transcription prefers Parakeet while translation routes to Qwen.

        Example:
          mere.run speech transcribe audio.wav
          mere.run speech transcribe - --stream --input-format pcm-s16le --sample-rate 16000 --jsonl
          mere.run speech transcribe audio.wav --backend parakeet
          mere.run speech transcribe audio.wav --task translate --backend auto
          mere.run speech transcribe audio.wav --backend qwen --model speech-asr-qwen3
        """
    )

    @Argument(help: "Input audio file, or '-' for raw streaming stdin.")
    var audio: String

    @Option(name: [.customShort("o"), .long], help: "Output file for transcript (optional, prints to stdout if omitted).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Model ID or local model path override for the selected backend.")
    var model: String?

    @Option(name: [.long], help: "ASR backend: auto, parakeet, or qwen.")
    var backend: SpeechBackendOption = .auto

    @Option(name: [.long], help: "Task: transcribe or translate.")
    var task: SpeechTaskOption = .transcribe

    @Option(name: [.long], help: "Language hint (optional, e.g. 'en', 'zh').")
    var language: String?

    @Option(name: [.long], help: "Maximum tokens to generate (default: 448).")
    var maxTokens: Int = 448

    @Flag(name: [.long], help: "Enable streaming ASR mode (forces Qwen backend).")
    var stream: Bool = false

    @Option(name: [.customLong("stream-chunk-ms")], help: "Audio feed chunk size in ms for streaming mode.")
    var streamChunkMs: Int = 200

    @Option(name: [.customLong("stream-decode-ms")], help: "Decode interval in ms for streaming mode.")
    var streamDecodeMs: Int = 2_000

    @Option(name: [.customLong("input-format")], help: "Raw stdin format. Protocol v1 accepts pcm-s16le.")
    var inputFormat: String?

    @Option(name: [.customLong("sample-rate")], help: "Raw stdin sample rate. Protocol v1 requires 16000.")
    var sampleRate: Int?

    @Flag(name: [.long], help: "Emit versioned live ASR events as JSON Lines on stdout.")
    var jsonl: Bool = false

    @Flag(
        name: [.customLong("timestamps")],
        inversion: .prefixedNo,
        help: "Include timestamped alignment lines in output when available (default: enabled)."
    )
    var timestamps: Bool = true

    @Flag(name: [.short, .long], help: "Quiet mode (suppress progress output).")
    var quiet: Bool = false

    func validate() throws {
        let readsStandardInput = audio == "-"
        if stream {
            guard streamChunkMs > 0 else {
                throw ValidationError("--stream-chunk-ms must be > 0.")
            }
            guard streamDecodeMs > 0 else {
                throw ValidationError("--stream-decode-ms must be > 0.")
            }
        }

        if readsStandardInput {
            guard stream else {
                throw ValidationError("Raw stdin requires --stream.")
            }
            guard inputFormat == "pcm-s16le" else {
                throw ValidationError("Raw stdin requires --input-format pcm-s16le.")
            }
            guard sampleRate == 16_000 else {
                throw ValidationError("Raw stdin requires --sample-rate 16000.")
            }
            guard output == nil else {
                throw ValidationError("Raw streaming stdin cannot be combined with --output.")
            }
        } else if inputFormat != nil || sampleRate != nil {
            throw ValidationError("--input-format and --sample-rate are only valid when reading raw stdin ('-').")
        }
        if jsonl && !stream {
            throw ValidationError("--jsonl requires --stream.")
        }
        if jsonl && !readsStandardInput {
            throw ValidationError("--jsonl is only valid with raw streaming stdin ('-').")
        }
    }

    func run() async throws {
        let readsStandardInput = audio == "-"

        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        if readsStandardInput {
            try await runStandardInput()
            return
        }
        let audioURL = URL(fileURLWithPath: audio).standardizedFileURL

        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("Audio file not found: \(audioURL.path)")
        }

        if stream && backend == .parakeet {
            CLIStderr.write("Streaming mode forces Qwen backend; ignoring --backend parakeet.\n")
        }

        if !quiet {
            let backendLabel = stream ? SpeechBackendOption.qwen.rawValue : backend.rawValue
            CLIStderr.write("Task=\(task.rawValue) backend=\(backendLabel)\n")
        }

        let progressHandler: (@Sendable (ASRProgress) -> Void)?
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
                CLIStderr.write("\(message)\n")
            }
        }

        let request = ASRRequest(
            audioURL: audioURL,
            language: language,
            task: task.task,
            maxTokens: maxTokens
        )

        if stream {
            try await runStreaming(
                request: request,
                audioURL: audioURL,
                progressHandler: progressHandler
            )
            return
        }

        let execution = try await CLIASRRouting.transcribe(
            request: request,
            preferredBackend: backend.backend,
            modelOverride: model,
            progressHandler: progressHandler
        )
        let result = execution.result
        let outputText = renderOutput(result: result, includeTimestamps: timestamps)

        if !quiet {
            let durationStr = String(format: "%.2f", result.duration)
            CLIStderr.write(
                "Resolved backend: \(execution.backend.rawValue) (\(execution.decision.reason))\n"
            )
            CLIStderr.write("Audio duration: \(durationStr)s\n")
            if let lang = result.language {
                CLIStderr.write("Language: \(lang)\n")
            }
        }

        if let outputPath = output {
            let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
            let outputDir = outputURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
            try outputText.write(to: outputURL, atomically: true, encoding: .utf8)
            if !quiet {
                CLIStderr.write("Transcript saved to: \(outputURL.path)\n")
            }
        }

        print(outputText)
    }

    private func runStandardInput() async throws {
        let progressHandler: (@Sendable (ASRProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else {
            progressHandler = { progress in
                if let message = progress.message {
                    CLIStderr.write("[\(progress.stage.rawValue)] \(message)\n")
                }
            }
        }
        let generator = try await CLIQwenASRLoader.prepare(model: model, progressHandler: progressHandler)
        let request = ASRStreamingRequest(
            language: language,
            task: task.task,
            maxTokens: maxTokens,
            sampleRate: 16_000,
            decodeIntervalMs: streamDecodeMs,
            minDecodeAudioMs: 1_600
        )
        let live = Qwen3ASRLiveSession(
            generator: generator,
            request: request,
            configuration: Qwen3ASRLiveConfiguration(decodeIntervalMs: streamDecodeMs)
        )
        if jsonl {
            try LiveASRCLIWriter.write(.ready())
        } else if !quiet {
            CLIStderr.write("Ready: qwen live ASR (pcm-s16le/16000/mono).\n")
        }
        let eventTask = Task {
            try await LiveASRCLIWriter.consume(live.events, jsonl: jsonl, quiet: quiet)
        }

        var decoder = PCM16LittleEndianDecoder()
        do {
            while let data = try FileHandle.standardInput.read(upToCount: 32_000), !data.isEmpty {
                let samples = decoder.decode(data)
                if !samples.isEmpty {
                    try await live.feed(samples: samples)
                }
            }
            try decoder.validateEOF()
            try await live.finish(reason: .eof)
            try await eventTask.value
        } catch {
            eventTask.cancel()
            await live.cancel()
            if jsonl {
                try? LiveASRCLIWriter.write(.error(
                    code: LiveASRCLIWriter.errorCode(for: error),
                    message: error.localizedDescription
                ))
            }
            throw error
        }
    }

    private func runStreaming(
        request: ASRRequest,
        audioURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws {
        let normalizedOverride = normalized(model)
        let overridePath = existingPath(from: normalizedOverride)
        let qwenModelId: String
        if let normalizedOverride, overridePath == nil {
            qwenModelId = normalizedOverride
        } else {
            qwenModelId = Qwen3ASRResources.defaultModelId
        }

        let generator = Qwen3ASRGenerator(modelId: qwenModelId)
        if let modelPath = overridePath {
            try await generator.prepare(modelPath: modelPath.path, progressHandler: progressHandler)
        } else if let localModelRoot = localQwenModelRootIfAvailable() {
            try await generator.prepare(modelPath: localModelRoot.path, progressHandler: progressHandler)
        }

        let streamRequest = ASRStreamingRequest(
            language: request.language,
            task: request.task,
            maxTokens: request.maxTokens,
            sampleRate: 16_000,
            decodeIntervalMs: streamDecodeMs
        )
        let session = try await generator.makeStreamingSession(streamRequest)
        let chunkSamples = max(1, (streamRequest.sampleRate * streamChunkMs) / 1_000)
        let audioSamples = try AudioReader.readAudio(from: audioURL)

        let eventTask = Task { () throws -> ASRResult in
            var finalResult: ASRResult?
            for try await event in session.events {
                switch event {
                case .partial(let text):
                    if !quiet {
                        CLIStderr.write("[partial] \(text)\n")
                    }
                case .stats:
                    break
                case .final(let result):
                    finalResult = result
                }
            }
            if let finalResult {
                return finalResult
            }
            throw ASRStreamingError.invalidState("ASR stream ended without final result.")
        }

        do {
            var start = 0
            while start < audioSamples.count {
                let end = min(audioSamples.count, start + chunkSamples)
                try await session.feed(samples: Array(audioSamples[start..<end]))
                start = end
            }
            try await session.finish()

            let result = try await eventTask.value
            let outputText = renderOutput(result: result, includeTimestamps: timestamps)

            if !quiet {
                let durationStr = String(format: "%.2f", result.duration)
                CLIStderr.write("Resolved backend: qwen (streaming_forced_qwen_v1)\n")
                CLIStderr.write("Audio duration: \(durationStr)s\n")
                if let lang = result.language {
                    CLIStderr.write("Language: \(lang)\n")
                }
            }

            if let outputPath = output {
                let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
                let outputDir = outputURL.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
                try outputText.write(to: outputURL, atomically: true, encoding: .utf8)
                if !quiet {
                    CLIStderr.write("Transcript saved to: \(outputURL.path)\n")
                }
            }

            print(outputText)
        } catch {
            eventTask.cancel()
            await session.cancel()
            throw error
        }
    }

    private func localQwenModelRootIfAvailable() -> URL? {
        let fm = FileManager.default
        let base = MereRunModelPaths.resolveModelDir(Qwen3ASRResources.defaultModelId) { root in
            fm.fileExists(atPath: root.appendingPathComponent("config.json").path)
                || fm.fileExists(atPath: root.appendingPathComponent("\(Qwen3ASRResources.defaultModelId)/config.json").path)
        }
        let nested = base.appendingPathComponent(Qwen3ASRResources.defaultModelId, isDirectory: true)
        if fm.fileExists(atPath: nested.appendingPathComponent("config.json").path) {
            return nested
        }
        if fm.fileExists(atPath: base.appendingPathComponent("config.json").path) {
            return base
        }
        return nil
    }

    private func existingPath(from value: String?) -> URL? {
        guard let value else { return nil }
        let url = URL(fileURLWithPath: value).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        return nil
    }

    private func normalized(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func renderOutput(result: ASRResult, includeTimestamps: Bool) -> String {
        guard includeTimestamps else { return result.text }

        if let sentences = result.sentenceAlignments, !sentences.isEmpty {
            let lines = sentences.map { sentence in
                let text = sentence.text.trimmingCharacters(in: .whitespacesAndNewlines)
                return "[\(formatTimestamp(sentence.startSeconds)) --> \(formatTimestamp(sentence.endSeconds))] \(text)"
            }
            return result.text + "\n\n" + lines.joined(separator: "\n")
        }

        if let tokens = result.tokenAlignments, !tokens.isEmpty {
            let lines = tokens.map { token in
                "[\(formatTimestamp(token.startSeconds)) --> \(formatTimestamp(token.endSeconds))] \(token.text)"
            }
            return result.text + "\n\n" + lines.joined(separator: "\n")
        }

        return result.text
    }

    private func formatTimestamp(_ seconds: TimeInterval) -> String {
        let clamped = max(0, seconds)
        let totalMilliseconds = Int((clamped * 1000).rounded())

        let milliseconds = totalMilliseconds % 1000
        let totalSeconds = totalMilliseconds / 1000
        let secs = totalSeconds % 60
        let totalMinutes = totalSeconds / 60
        let minutes = totalMinutes % 60
        let hours = totalMinutes / 60

        if hours > 0 {
            return String(format: "%02d:%02d:%02d.%03d", hours, minutes, secs, milliseconds)
        }
        return String(format: "%02d:%02d.%03d", minutes, secs, milliseconds)
    }
}
