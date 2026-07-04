import ArgumentParser
import AudioCodecs
#if canImport(AVFoundation)
import AVFoundation
#endif
import Foundation
import MereRunCore

extension MagentaRT2StyleConditioning: ExpressibleByArgument {}

struct MusicRealtime: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "realtime",
        abstract: "Run Magenta RealTime 2 music generation.",
        discussion: """
        Plays generated audio to the default output device by default.
        Pass --output to capture the stream to a WAV file at the same time.

        Example:
          mere.run music realtime "ambient pads with sub bass" \
            --model music-magenta-rt2-small \
            --duration 4 \
            --output live.wav
        """
    )

    @Argument(help: "Text prompt describing the target music.")
    var prompt: String?

    @Option(name: [.customShort("m"), .long], help: "Managed Magenta RT2 model id or local Magenta RT2 root.")
    var model: String = MagentaRT2Resources.smallModelId

    @Option(name: [.customLong("duration")], help: "Realtime run duration in seconds.")
    var durationSeconds: Float = 30.0

    @Option(name: [.customShort("o"), .long], help: "Optional output WAV path for stream capture.")
    var output: String?

    @Flag(name: [.long], inversion: .prefixedNo, help: "Play realtime audio through the default output device.")
    var play: Bool = true

    @Option(name: [.customLong("style-conditioning")], help: "Magenta RT2 style conditioning detail: streaming or full.")
    var styleConditioning: MagentaRT2StyleConditioning = .streaming

    @Option(name: [.customLong("temperature")], help: "Magenta RT2 sampling temperature.")
    var temperature: Float = 1.0

    @Option(name: [.customLong("top-k")], help: "Magenta RT2 top-k sampling.")
    var topK: Int = 100

    @Option(name: [.customLong("cfg-musiccoca")], help: "Magenta RT2 MusicCoCa guidance scale.")
    var cfgMusicCoCa: Float = 3.0

    @Option(name: [.customLong("cfg-notes")], help: "Magenta RT2 MIDI-notes guidance scale.")
    var cfgNotes: Float = 5.0

    @Option(name: [.customLong("cfg-drums")], help: "Magenta RT2 drums guidance scale.")
    var cfgDrums: Float = 1.0

    @Flag(name: [.customLong("drumless")], help: "Enable Magenta RT2 drumless mode.")
    var drumless: Bool = false

    @Option(name: [.customLong("unmask-width")], help: "Magenta RT2 token unmask width.")
    var unmaskWidth: Int = 0

    @Option(name: [.customLong("seed-rotation")], help: "Magenta RT2 seed rotation.")
    var seedRotation: Int = 0

    @Flag(name: [.customLong("prefill-silence")], help: "Prefill Magenta RT2 state with silence before generation.")
    var prefillSilence: Bool = false

    @Option(name: [.customLong("prefill-duration")], help: "Magenta RT2 silent prefill duration in seconds.")
    var prefillDuration: Float = 1.64

    @Flag(name: [.customLong("interactive")], help: "Read live steering commands from stdin while generation runs.")
    var interactive: Bool = false

    @Flag(name: [.customLong("list-midi-inputs")], help: "List available CoreMIDI input sources and exit.")
    var listMIDIInputs: Bool = false

    @Option(name: [.customLong("midi-input")], help: "CoreMIDI input source name or unique ID to steer realtime notes and controls.")
    var midiInput: String?

    @Option(name: [.customLong("midi-channel")], help: "MIDI channel to listen to, 1-16 or all.")
    var midiChannel: String = "all"

    @Option(name: [.customLong("midi-note-offset")], help: "Transpose incoming MIDI notes before sending them to Magenta RT2.")
    var midiNoteOffset: Int = 0

    @Option(name: [.customLong("midi-cc")], help: "Map a MIDI CC as cc=target:min:max, for example 1=temp:0.2:1.4.")
    var midiCCMappings: [String] = []

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        if listMIDIInputs {
            try printMIDIInputs()
            return
        }
        guard let prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Provide a prompt or use --list-midi-inputs.")
        }
        let midiConfiguration = try resolvedMIDIConfiguration()
        guard MagentaRT2Resources.isMagentaRT2Model(model)
            || MagentaRT2Resources.looksLikeMagentaRT2Root(URL(fileURLWithPath: model).standardizedFileURL) else {
            throw ValidationError("music realtime requires a Magenta RT2 model id or local Magenta RT2 root.")
        }
        guard durationSeconds > 0 else {
            throw ValidationError("--duration must be > 0")
        }
        guard play || output != nil else {
            throw ValidationError("Use --play or provide --output when --no-play is set.")
        }

        let resources = try await MagentaRT2Resources.resolve(
            requestedModel: model,
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .downloading(let percent):
                    CLIStderr.write("Downloading Magenta RT2 assets... \(percent)%\n")
                case .extracting:
                    CLIStderr.write("Extracting Magenta RT2 assets...\n")
                }
            }
        )
        let outputURL = output.map { URL(fileURLWithPath: $0).standardizedFileURL }
        if let outputURL {
            try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        }
        let initialControls = try controls()
        let liveControls = interactive || midiConfiguration != nil
            ? MagentaRT2LiveControlQueue(initialControls: initialControls)
            : nil
        let sink = try MagentaRT2RealtimeOutputSink(outputURL: outputURL, play: play)
        defer { try? sink.close() }
        let midiInputConnection = try liveControls.flatMap { liveControls in
            try midiConfiguration.map { configuration in
                try MagentaRT2MIDIInput(configuration: configuration, controls: liveControls)
            }
        }
        defer { midiInputConnection?.stop() }

        if !quiet {
            CLIStderr.write("Starting Magenta RT2 realtime model \(resources.modelID)\n")
        }
        if let liveControls, interactive, !quiet {
            CLIStderr.write(liveControls.helpText)
        }
        if let midiInputConnection, !quiet {
            CLIStderr.write("Connected MIDI input \(midiInputConnection.sourceDisplayName)\n")
        }
        let inputTask: Task<Void, Never>? = interactive
            ? liveControls.map { controls in
                Task.detached {
                    while let line = readLine(strippingNewline: true) {
                        let response = controls.applyCommand(line)
                        if !response.isEmpty {
                            CLIStderr.write(response + "\n")
                        }
                        if controls.shouldStop {
                            break
                        }
                    }
                }
            }
            : nil
        defer { inputTask?.cancel() }

        let startedAt = Date()
        do {
            try await MagentaRT2RealtimeSession.run(
                MagentaRT2RealtimeRequest(
                    prompt: prompt,
                    resources: resources,
                    durationSeconds: durationSeconds,
                    controls: initialControls
                ),
                liveControls: { _ in
                    liveControls?.snapshot()
                }
            ) { frameIndex, frame in
                try sink.consume(frameIndex: frameIndex, frame: frame)
                if !quiet, frameIndex % 25 == 0 {
                    CLIStderr.write("Realtime frame \(frameIndex + 1)\n")
                }
                paceFrameIfNeeded(frameIndex: frameIndex, startedAt: startedAt)
            }
        } catch is CancellationError {
            if liveControls?.shouldStop != true {
                throw CancellationError()
            }
        }

        if let outputURL {
            print(outputURL.path)
        }
    }

    private func controls() throws -> MagentaRT2Controls {
        guard topK >= 0 else {
            throw ValidationError("--top-k must be >= 0")
        }
        guard unmaskWidth >= 0 else {
            throw ValidationError("--unmask-width must be >= 0")
        }
        guard prefillDuration > 0 else {
            throw ValidationError("--prefill-duration must be > 0")
        }
        return MagentaRT2Controls(
            styleConditioning: styleConditioning,
            temperature: temperature,
            topK: Int32(topK),
            cfgMusicCoCa: cfgMusicCoCa,
            cfgNotes: cfgNotes,
            cfgDrums: cfgDrums,
            drumless: drumless,
            unmaskWidth: Int32(unmaskWidth),
            seedRotation: Int32(seedRotation),
            prefillSilence: prefillSilence,
            prefillDurationSeconds: prefillDuration
        )
    }

    private func resolvedMIDIConfiguration() throws -> MagentaRT2MIDIInputConfiguration? {
        let trimmedInput = midiInput?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmedInput, !trimmedInput.isEmpty else {
            if midiChannel.lowercased() != "all" || midiNoteOffset != 0 || !midiCCMappings.isEmpty {
                throw ValidationError("Use --midi-input when setting MIDI channel, note offset, or CC mappings.")
            }
            return nil
        }
        return MagentaRT2MIDIInputConfiguration(
            source: trimmedInput,
            channel: try MagentaRT2MIDIInputConfiguration.parseChannel(midiChannel),
            noteOffset: midiNoteOffset,
            ccMappings: try midiCCMappings.map(MagentaRT2MIDICCMapping.parse)
        )
    }

    private func printMIDIInputs() throws {
        let sources = try MagentaRT2MIDIInput.availableSources()
        guard !sources.isEmpty else {
            print("No MIDI input sources found.")
            return
        }
        for source in sources {
            print(source.displayName)
        }
    }

    private func paceFrameIfNeeded(frameIndex: Int, startedAt: Date) {
        guard play || interactive || midiInput != nil else { return }
        let targetElapsed = Double(frameIndex + 1) / Double(MagentaRT2Resources.frameRate)
        let actualElapsed = Date().timeIntervalSince(startedAt)
        let sleepSeconds = targetElapsed - actualElapsed
        if sleepSeconds > 0 {
            Thread.sleep(forTimeInterval: sleepSeconds)
        }
    }
}

private final class MagentaRT2RealtimeOutputSink: @unchecked Sendable {
    private let lock = NSLock()
    private let writer: StreamingWAVWriter?
    private let player: MagentaRT2AudioPlayer?

    init(outputURL: URL?, play: Bool) throws {
        if let outputURL {
            writer = try StreamingWAVWriter(
                outputURL: outputURL,
                sampleRate: MagentaRT2Resources.sampleRate,
                channels: MagentaRT2Resources.channels
            )
        } else {
            writer = nil
        }
        player = play ? try MagentaRT2AudioPlayer() : nil
    }

    func consume(frameIndex: Int, frame: MagentaRT2Frame) throws {
        _ = frameIndex
        lock.lock()
        defer { lock.unlock() }
        let samples = frame.interleavedStereo
        try writer?.append(samples: samples)
        try player?.enqueue(frame)
    }

    func close() throws {
        lock.lock()
        defer { lock.unlock() }
        player?.stop()
        try writer?.close()
    }
}

#if canImport(AVFoundation)
private final class MagentaRT2AudioPlayer: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat

    init() throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(MagentaRT2Resources.sampleRate),
            channels: AVAudioChannelCount(MagentaRT2Resources.channels),
            interleaved: false
        ) else {
            throw ValidationError("Could not create audio output format.")
        }
        self.format = format
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        player.play()
    }

    func enqueue(_ frame: MagentaRT2Frame) throws {
        let count = min(frame.left.count, frame.right.count)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(count)
        ) else {
            throw ValidationError("Could not allocate audio output buffer.")
        }
        buffer.frameLength = AVAudioFrameCount(count)
        guard let channels = buffer.floatChannelData else {
            throw ValidationError("Audio output buffer has no float channels.")
        }
        for index in 0..<count {
            channels[0][index] = frame.left[index]
            channels[1][index] = frame.right[index]
        }
        player.scheduleBuffer(buffer, completionHandler: nil)
    }

    func stop() {
        player.stop()
        engine.stop()
    }
}
#else
private final class MagentaRT2AudioPlayer: @unchecked Sendable {
    init() throws {
        throw ValidationError("Audio playback requires AVFoundation on macOS.")
    }

    func enqueue(_ frame: MagentaRT2Frame) throws {
        _ = frame
    }

    func stop() {}
}
#endif
