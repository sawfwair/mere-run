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

    @Flag(name: [.long], help: "Enable streaming ASR mode using the selected backend.")
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

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt: Bool = false

    func validate() throws {
        let readsStandardInput = audio == "-"
        if receipt && readsStandardInput {
            throw ValidationError("--receipt is not available for raw streaming stdin ('-'); use --jsonl.")
        }
        if stream {
            guard streamChunkMs > 0 else {
                throw ValidationError("--stream-chunk-ms must be > 0.")
            }
            guard streamDecodeMs > 0 else {
                throw ValidationError("--stream-decode-ms must be > 0.")
            }
            if task == .translate, backend == .parakeet {
                throw ValidationError("Parakeet does not support translation; use --backend qwen or --backend auto.")
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

        if !quiet {
            CLIStderr.write("Task=\(task.rawValue) backend=\(backend.rawValue)\n")
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

        let transcriptURL = try writeTranscriptIfRequested(outputText)
        print(outputText)
        try emitReceipt(transcriptURL: transcriptURL)
    }

    /// Writes the transcript to `--output` when set and returns its URL. The
    /// receipt lists no outputs when the transcript only went to stdout.
    private func writeTranscriptIfRequested(_ outputText: String) throws -> URL? {
        guard let outputPath = output else { return nil }
        let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        let outputDir = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        try outputText.write(to: outputURL, atomically: true, encoding: .utf8)
        if !quiet {
            CLIStderr.write("Transcript saved to: \(outputURL.path)\n")
        }
        return outputURL
    }

    private func emitReceipt(transcriptURL: URL?) throws {
        try RunReceipt.emit(RunReceipt.transcriptOutputs(transcriptURL), enabled: receipt)
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
        let live = try await makeLiveSession(progressHandler: progressHandler)
        if jsonl {
            try LiveASRCLIWriter.write(.ready())
        } else if !quiet {
            CLIStderr.write("Ready: \(live.backend.rawValue) live ASR (pcm-s16le/16000/mono).\n")
        }
        let eventTask = Task {
            try await LiveASRCLIWriter.consume(live.events, jsonl: jsonl, quiet: quiet)
        }

        var decoder = PCM16LittleEndianDecoder()
        do {
            while let data = try FileHandle.standardInput.read(upToCount: 32_000), !data.isEmpty {
                let samples = decoder.decode(data)
                if !samples.isEmpty {
                    try await live.feed(samples)
                }
            }
            try decoder.validateEOF()
            try await live.finish(.eof)
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
        let live = try await makeLiveSession(progressHandler: progressHandler)
        let chunkSamples = max(1, (16_000 * streamChunkMs) / 1_000)
        let audioSamples = try AudioReader.readAudio(from: audioURL)

        let eventTask = Task { () throws -> [String] in
            var commits: [String] = []
            for try await event in live.events {
                switch event {
                case .partial(let transcript):
                    if !quiet {
                        CLIStderr.write("[partial] \(transcript.text)\n")
                    }
                case .commit(let transcript):
                    commits.append(transcript.text)
                case .stats:
                    break
                case .final:
                    break
                }
            }
            return commits
        }

        do {
            var start = 0
            while start < audioSamples.count {
                let end = min(audioSamples.count, start + chunkSamples)
                try await live.feed(Array(audioSamples[start..<end]))
                start = end
                try await Task.sleep(for: .milliseconds(streamChunkMs))
            }
            try await live.finish(.eof)

            let text = try await eventTask.value.joined(separator: "\n")
            let result = ASRResult(
                text: text,
                language: request.language,
                duration: Double(audioSamples.count) / 16_000
            )
            let outputText = renderOutput(result: result, includeTimestamps: timestamps)

            if !quiet {
                let durationStr = String(format: "%.2f", result.duration)
                CLIStderr.write("Resolved backend: \(live.backend.rawValue) (streaming_policy_v1)\n")
                CLIStderr.write("Audio duration: \(durationStr)s\n")
                if let lang = result.language {
                    CLIStderr.write("Language: \(lang)\n")
                }
            }

            let transcriptURL = try writeTranscriptIfRequested(outputText)
            print(outputText)
            try emitReceipt(transcriptURL: transcriptURL)
        } catch {
            eventTask.cancel()
            await live.cancel()
            throw error
        }
    }

    private func makeLiveSession(
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> CLILiveASRSession {
        let selected = try resolvedStreamingBackend()
        let request = ASRStreamingRequest(
            language: language,
            task: task.task,
            maxTokens: maxTokens,
            sampleRate: 16_000,
            decodeIntervalMs: streamDecodeMs,
            minDecodeAudioMs: 1_600
        )
        let configuration = Qwen3ASRLiveConfiguration(decodeIntervalMs: streamDecodeMs)

        switch selected {
        case .parakeet:
            var parakeetConfiguration = configuration
            parakeetConfiguration.silenceMs = min(configuration.silenceMs, 600)
            let generator = try await CLIParakeetASRLoader.prepare(
                model: model,
                progressHandler: progressHandler
            )
            let live = ParakeetASRLiveSession(
                generator: generator,
                request: request,
                configuration: parakeetConfiguration
            )
            return CLILiveASRSession(
                backend: .parakeet,
                events: live.events,
                feed: { try await live.feed(samples: $0) },
                finish: { try await live.finish(reason: $0) },
                cancel: { await live.cancel() }
            )
        case .qwen:
            let generator = try await CLIQwenASRLoader.prepare(
                model: model,
                progressHandler: progressHandler
            )
            let live = Qwen3ASRLiveSession(
                generator: generator,
                request: request,
                configuration: configuration
            )
            return CLILiveASRSession(
                backend: .qwen,
                events: live.events,
                feed: { try await live.feed(samples: $0) },
                finish: { try await live.finish(reason: $0) },
                cancel: { await live.cancel() }
            )
        case .auto:
            preconditionFailure("Streaming backend must resolve before session creation.")
        }
    }

    private func resolvedStreamingBackend() throws -> SpeechBackendOption {
        if task == .translate {
            guard backend != .parakeet else {
                throw ValidationError("Parakeet does not support translation; use --backend qwen or --backend auto.")
            }
            return .qwen
        }
        return backend == .auto ? .parakeet : backend
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
