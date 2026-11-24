import ArgumentParser
import AudioCore
import Foundation
import MediaIO
import MereRunContract
import MereRunCore

#if os(macOS)
@preconcurrency import AVFoundation
#endif

struct SpeechConverse: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "converse",
        abstract: "Run a full-duplex spoken conversation with Haru-4B."
    )

    @Option(name: [.customShort("m"), .long], help: "Installed Haru model ID or checkpoint directory.")
    var model = HaruResources.modelID

    @Option(help: "User audio file to replay as successive conversation frames.")
    var input: String?

    @Flag(help: "Use the macOS microphone and default audio output.")
    var live = false

    @Option(name: .shortAndLong, help: "Output WAV path. Required for file replay.")
    var output: String?

    @Option(help: "UTF-8 file containing a Haru character card.")
    var systemPromptFile: String?

    @Option(help: "Reference audio for Haru's voice.")
    var referenceAudio: String?

    @Flag(help: "Prompt Haru to open the conversation.")
    var speakFirst = false

    @Option(help: "Seed for the audio and text sampling streams.")
    var seed: UInt64?

    @Option(help: "Text sampling temperature. Zero selects the highest-scoring token.")
    var temperature: Float = 0.7

    @Option(help: "Maximum number of output frames, including the initial frame.")
    var maxFrames = 27_000

    func validate() throws {
        guard live != (input != nil) else { throw ValidationError("Choose exactly one of --input and --live.") }
        guard live || output != nil else { throw ValidationError("--input requires --output.") }
        guard temperature.isFinite, temperature >= 0 else {
            throw ValidationError("--temperature must be finite and nonnegative.")
        }
        guard maxFrames > 0 else { throw ValidationError("--max-frames must be positive.") }
        #if !os(macOS)
        if live { throw ValidationError("--live requires macOS.") }
        #endif
    }

    func run() async throws {
        #if os(macOS)
        if live, !(await MicrophoneCapture.requestPermission()) {
            throw ValidationError("Enable microphone access in System Settings > Privacy & Security > Microphone.")
        }
        #endif
        let directory: URL
        if ManagedModelCatalog.spec(for: model)?.id == HaruResources.modelID {
            directory = try ModelResolver().resolve(.haru4B).rootURL
        } else {
            directory = url(model)
        }
        let card = try systemPromptFile.map {
            try String(contentsOf: url($0), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        } ?? ""
        let reference = try referenceAudio.map(decode) ?? []
        let recording = try input.map(decode)
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let session = try HaruDuplexSession(modelDirectory: directory, seed: seed, temperature: temperature)
        let writer = try output.map {
            try AudioExportStream(
                plan: AudioExportPlan(options: .speechStreaming), to: url($0), sampleRate: HaruDuplexSession.sampleRate
            )
        }
        let first = try await session.start(characterCard: card, referenceAudio: reference, speakFirst: speakFirst)
        let result = HaruConversationOutput(writer: writer)
        if let recording {
            try result.append(first)
            let frameSamples = HaruDuplexSession.frameSamples
            for offset in stride(from: 0, to: recording.count, by: frameSamples) {
                if result.frames == maxFrames { break }
                var chunk = Array(recording[offset..<min(offset + frameSamples, recording.count)])
                chunk += Array(repeating: 0, count: frameSamples - chunk.count)
                try await result.append(session.step(chunk))
            }
        } else {
            #if os(macOS)
            try await runLive(session: session, first: first, result: result)
            #endif
        }
        let file = try writer?.finish()
        let summary = HaruConversationSummary(
            output: file?.url.path, frames: result.frames, sampleRate: HaruDuplexSession.sampleRate,
            transcript: result.transcript.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: result.segments, inputTranscriptionError: result.inputTranscriptionError
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        print(String(decoding: try encoder.encode(summary), as: UTF8.self))
    }

    private func url(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    private func decode(_ path: String) throws -> [Float] {
        try MediaAudioIO.decode(url(path), targetSampleRate: HaruDuplexSession.sampleRate, channels: 1).samples
    }

    #if os(macOS)
    private func runLive(session: HaruDuplexSession, first: HaruAudioFrame, result: HaruConversationOutput) async throws {
        let capture = try MicrophoneCapture(
            deviceUID: nil, sampleRate: Double(HaruDuplexSession.sampleRate), bufferingPolicy: .bufferingOldest(8)
        )
        let player = try HaruPlayback(sampleRate: HaruDuplexSession.sampleRate)
        let stop = InterruptSignal()
        defer { capture.stop() }
        try capture.start()
        let interrupt = Task {
            let interrupted = await stop.wait()
            if interrupted { capture.stop() }
            return interrupted
        }
        defer { interrupt.cancel() }
        CLIStderr.write("Conversation started. Press Ctrl-C to stop.\n")
        var events = HaruLiveOutput()
        events.begin()
        let input = events.transcribesInput ? HaruInputTranscription() : nil
        defer { input?.cancel() }
        try await play(first, into: result, using: player, events: &events)
        if result.frames == maxFrames { capture.stop() }
        var pending: [Float] = []
        let frameSamples = HaruDuplexSession.frameSamples
        for await samples in capture.samples {
            try Task.checkCancellation()
            if result.frames == maxFrames { break }
            input?.feed(samples)
            pending += samples
            var consumed = 0
            while !capture.isStopped, pending.count - consumed >= frameSamples, result.frames < maxFrames {
                let end = consumed + frameSamples
                let frame = try await session.step(Array(pending[consumed..<end]))
                try await play(frame, into: result, using: player, events: &events)
                consumed = end
            }
            pending.removeFirst(consumed)
            if capture.isStopped || result.frames == maxFrames { break }
        }
        if result.frames < maxFrames {
            interrupt.cancel()
            guard await interrupt.value else {
                throw ValidationError("Microphone capture ended or inference fell behind the input.")
            }
        }
        // Close capture before waiting for Parakeet to commit its last utterance.
        capture.stop()
        if let input {
            let final = await input.finish()
            result.segments = HaruTranscriptSegment.chronological(final.segments + events.reply.segments)
            result.inputTranscriptionError = final.error
        }
        CLIStderr.write("\n")
    }

    private func play(
        _ frame: HaruAudioFrame, into result: HaruConversationOutput, using player: HaruPlayback, events: inout HaruLiveOutput
    ) async throws {
        try result.append(frame)
        let startMs = (result.frames - 1) * HaruDuplexSession.frameSamples * 1_000 / HaruDuplexSession.sampleRate
        events.report(frame, at: startMs)
        try await player.play(frame.audio, drain: result.frames == maxFrames)
    }
    #endif
}

private final class HaruConversationOutput {
    let writer: AudioExportStream?
    private(set) var frames = 0
    private(set) var transcript = ""
    var segments: [HaruTranscriptSegment]?
    var inputTranscriptionError: String?

    init(writer: AudioExportStream?) {
        self.writer = writer
    }

    func append(_ frame: HaruAudioFrame) throws {
        try writer?.append(samples: frame.audio)
        frames += 1
        transcript += frame.text
    }
}

#if os(macOS)
private struct HaruLiveOutput {
    private static let releaseFrames = 3
    private static let speechFloorMeanSquare: Float = 0.000_032
    private let streamsEvents = ProcessInfo.processInfo.environment[HaruConversationEvent.environmentKey] == "1"
    private let inputRequested = ProcessInfo.processInfo.environment[HaruConversationEvent.inputTranscriptionEnvironmentKey] == "1"
    private var quietFrames = Self.releaseFrames
    private(set) var reply = HaruReplyTranscript()

    var transcribesInput: Bool { streamsEvents && inputRequested }

    func begin() {
        if streamsEvents { HaruEventWriter.write(.init(.capturing)) }
    }

    mutating func report(_ frame: HaruAudioFrame, at startMs: Int) {
        if !frame.text.isEmpty { CLIStderr.write(frame.text) }
        guard streamsEvents else { return }
        // PCM includes background noise even between words. A short release keeps brief gaps
        // between syllables from flickering the playback indicator.
        let wasSpeaking = quietFrames < Self.releaseFrames
        let meanSquare = frame.audio.reduce(Float.zero) { $0 + $1 * $1 } / Float(frame.audio.count)
        quietFrames = meanSquare >= Self.speechFloorMeanSquare ? 0 : min(quietFrames + 1, Self.releaseFrames)
        let speaking = quietFrames < Self.releaseFrames
        if transcribesInput { reply.append(frame.text, at: startMs, speaking: speaking) }
        if speaking != wasSpeaking || !frame.text.isEmpty {
            HaruEventWriter.write(.init(.frame, speaking: speaking, text: frame.text, startMs: transcribesInput ? startMs : nil))
        }
    }
}

private final class HaruPlayback {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat

    init(sampleRate: Int) throws {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1) else {
            throw ValidationError("Could not configure Haru playback.")
        }
        self.format = format
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try engine.start()
        node.play()
    }

    deinit {
        node.stop()
        engine.stop()
    }

    func play(_ samples: [Float], drain: Bool) async throws {
        guard engine.isRunning else { throw ValidationError("The Haru playback device stopped.") }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else {
            throw ValidationError("Could not allocate a Haru playback buffer.")
        }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress { channel.update(from: base, count: source.count) }
        }
        if drain {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in continuation.resume() }
            }
        } else {
            enqueue(buffer)
        }
    }

    private func enqueue(_ buffer: AVAudioPCMBuffer) {
        node.scheduleBuffer(buffer)
    }
}
#endif
