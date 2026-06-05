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
    var prompt: String

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

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
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
        let liveControls = interactive ? MagentaRT2InteractiveControls(initialControls: initialControls) : nil
        let sink = try MagentaRT2RealtimeOutputSink(outputURL: outputURL, play: play)
        defer { try? sink.close() }

        if !quiet {
            CLIStderr.write("Starting Magenta RT2 realtime model \(resources.modelID)\n")
        }
        if let liveControls, !quiet {
            CLIStderr.write(liveControls.helpText)
        }
        let inputTask = liveControls.map { controls in
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
                liveControls: { frameIndex in
                    liveControls?.snapshot(frameIndex: frameIndex)
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

    private func paceFrameIfNeeded(frameIndex: Int, startedAt: Date) {
        guard play || interactive else { return }
        let targetElapsed = Double(frameIndex + 1) / Double(MagentaRT2Resources.frameRate)
        let actualElapsed = Date().timeIntervalSince(startedAt)
        let sleepSeconds = targetElapsed - actualElapsed
        if sleepSeconds > 0 {
            Thread.sleep(forTimeInterval: sleepSeconds)
        }
    }
}

private final class MagentaRT2InteractiveControls: @unchecked Sendable {
    private let lock = NSLock()
    private var controls: MagentaRT2Controls
    private var pendingPrompt: String?
    private var pendingNoteOn: [Int32] = []
    private var pendingNoteOff: [Int32] = []
    private var pendingOnsetMode: Int32?
    private var pendingControls: Bool = false
    private var pendingReset: Bool = false
    private(set) var shouldStop: Bool = false

    let helpText = """
    Interactive steering enabled. Commands:
      prompt <text>
      style streaming|full
      temp <value> | topk <value> | mc <value> | notes <value> | drums <value>
      noteon <0-131> | noteoff <0-131> | onset 0|1
      drumless on|off | unmask <value> | seed <value> | reset | quit | help

    """

    init(initialControls: MagentaRT2Controls) {
        controls = initialControls
    }

    func applyCommand(_ rawLine: String) -> String {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return "" }
        let parts = line.split(maxSplits: 1, whereSeparator: \.isWhitespace).map(String.init)
        let command = parts[0].lowercased()
        let value = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespacesAndNewlines) : ""

        lock.lock()
        defer { lock.unlock() }

        switch command {
        case "prompt":
            guard !value.isEmpty else { return "usage: prompt <text>" }
            pendingPrompt = value
            return "queued prompt"
        case "style", "style-conditioning":
            guard let parsed = MagentaRT2StyleConditioning(rawValue: value.lowercased()) else {
                return "usage: style streaming|full"
            }
            controls.styleConditioning = parsed
            pendingControls = true
            return "style-conditioning \(parsed.rawValue)"
        case "temp", "temperature":
            guard let parsed = Float(value) else { return "usage: temp <value>" }
            controls.temperature = parsed
            pendingControls = true
            return "temperature \(parsed)"
        case "topk", "top-k":
            guard let parsed = Int32(value), parsed >= 0 else { return "usage: topk <value>" }
            controls.topK = parsed
            pendingControls = true
            return "top-k \(parsed)"
        case "mc", "musiccoca", "cfg-musiccoca":
            guard let parsed = Float(value) else { return "usage: mc <value>" }
            controls.cfgMusicCoCa = parsed
            pendingControls = true
            return "cfg-musiccoca \(parsed)"
        case "notes", "cfg-notes":
            guard let parsed = Float(value) else { return "usage: notes <value>" }
            controls.cfgNotes = parsed
            pendingControls = true
            return "cfg-notes \(parsed)"
        case "drums", "cfg-drums":
            guard let parsed = Float(value) else { return "usage: drums <value>" }
            controls.cfgDrums = parsed
            pendingControls = true
            return "cfg-drums \(parsed)"
        case "drumless":
            guard let parsed = parseBool(value) else { return "usage: drumless on|off" }
            controls.drumless = parsed
            pendingControls = true
            return "drumless \(parsed ? "on" : "off")"
        case "unmask", "unmask-width":
            guard let parsed = Int32(value), parsed >= 0 else { return "usage: unmask <value>" }
            controls.unmaskWidth = parsed
            pendingControls = true
            return "unmask-width \(parsed)"
        case "seed", "seed-rotation":
            guard let parsed = Int32(value) else { return "usage: seed <value>" }
            controls.seedRotation = parsed
            pendingControls = true
            return "seed-rotation \(parsed)"
        case "noteon", "note-on":
            guard let parsed = parseMIDINote(value) else { return "usage: noteon <0-131>" }
            pendingNoteOn.append(parsed)
            return "note on \(parsed)"
        case "noteoff", "note-off":
            guard let parsed = parseMIDINote(value) else { return "usage: noteoff <0-131>" }
            pendingNoteOff.append(parsed)
            return "note off \(parsed)"
        case "onset", "onset-mode":
            guard let parsed = Int32(value), parsed == 0 || parsed == 1 else { return "usage: onset 0|1" }
            pendingOnsetMode = parsed
            return "onset \(parsed)"
        case "reset":
            pendingReset = true
            return "queued reset"
        case "quit", "exit", "stop":
            shouldStop = true
            return "stopping"
        case "help", "?":
            return helpText
        default:
            return "unknown command: \(command)"
        }
    }

    func snapshot(frameIndex: Int) -> MagentaRT2LiveControlSnapshot? {
        _ = frameIndex
        lock.lock()
        defer { lock.unlock() }
        if shouldStop {
            return MagentaRT2LiveControlSnapshot(shouldStop: true)
        }
        guard pendingPrompt != nil
            || pendingControls
            || !pendingNoteOn.isEmpty
            || !pendingNoteOff.isEmpty
            || pendingOnsetMode != nil
            || pendingReset else {
            return nil
        }
        let snapshot = MagentaRT2LiveControlSnapshot(
            prompt: pendingPrompt,
            controls: pendingControls ? controls : nil,
            noteOn: pendingNoteOn,
            noteOff: pendingNoteOff,
            onsetMode: pendingOnsetMode,
            resetState: pendingReset
        )
        pendingPrompt = nil
        pendingNoteOn = []
        pendingNoteOff = []
        pendingOnsetMode = nil
        pendingControls = false
        pendingReset = false
        return snapshot
    }

    private func parseBool(_ value: String) -> Bool? {
        switch value.lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return nil
        }
    }

    private func parseMIDINote(_ value: String) -> Int32? {
        guard let parsed = Int32(value), parsed >= 0, parsed < 132 else {
            return nil
        }
        return parsed
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
