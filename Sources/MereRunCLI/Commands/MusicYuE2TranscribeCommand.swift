import ArgumentParser
import AudioCore
import Foundation
import MediaIO
import MereRunCore

struct YuE2Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "transcribe", abstract: "Transcribe a recording into a YuE2-compatible score with native SheetSage2.")
    @Argument(help: "Input recording.") var input: String
    @Option(name: [.short, .long], help: "Managed SheetSage2 ID or local adapter directory with mert2/.") var model = "music-sheetsage2"
    @Option(name: [.short, .long], help: "New output directory for score.abc and transcription.json.") var output: String
    @Flag(name: .customLong("melody-only"), help: "Omit chord conditioning for a freely arranged cover.") var melodyOnly = false
    @Flag(name: .customLong("allow-notation-failure"), help: "Succeed with raw JSON, MIDI and LAB analysis when ABC export fails.") var allowNotationFailure = false
    @Option(name: .long, help: "Window overlap in seconds (default: 200).") var overlap: Double = 200
    @Option(name: .long, help: "Right-hand lookahead in seconds (default: 100).") var lookahead: Double = 100
    func run() async throws {
        let destination = Self.url(output)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ValidationError("Output directory already exists.") }
        let result = try await Self.transcribe(input: input, model: model, melodyOnly: melodyOnly, overlap: overlap, lookahead: lookahead)
        try Self.save(result, to: destination)
        guard result.abc != nil || allowNotationFailure else {
            throw ValidationError("Raw transcription saved to \(destination.path); score export failed: \(result.notationError ?? "unknown notation error").")
        }
        print(destination.appendingPathComponent(result.abc == nil ? "transcription.json" : "score.abc").path)
    }

    static func transcribe(input: String, model: String, melodyOnly: Bool, overlap: Double = 200,
                           lookahead: Double = 100) async throws -> SheetSage2Transcription {
        _ = try SheetSage2Window.plan(duration: 1, overlap: overlap, lookahead: lookahead)
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let root = model == SheetSage2Resources.modelID ? try ModelResolver().resolve(.sheetSage2).rootURL : url(model)
        let audio = try MediaAudioIO.decode(url(input), targetSampleRate: 24000, channels: 1)
        let transcriber = SheetSage2Transcriber(resources: SheetSage2Resources(rootURL: root))
        return try await transcriber.transcribe(samples: audio.samples, melodyOnly: melodyOnly,
            overlap: overlap, lookahead: lookahead) { stage, completed, total in
                if completed == total || completed % 64 == 0 { CLIStderr.write("SheetSage2 \(stage): \(completed)/\(total)\n") }
            }
    }

    static func save(_ value: SheetSage2Transcription, to destination: URL) throws {
        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(".sheetsage2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try value.abc?.write(to: temporary.appendingPathComponent("score.abc"), atomically: true, encoding: .utf8)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: temporary.appendingPathComponent("transcription.json"))
        for (name, data) in try value.symbolicArtifacts() { try data.write(to: temporary.appendingPathComponent(name)) }
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    static func url(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL }
}

struct YuE2Cover: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "cover", abstract: "Transcribe a recording and render a new YuE2 interpretation.")
    @Argument(help: "Input recording.") var input: String
    @Option(name: .long, help: "YuE2 JSON request containing the target style and optional new lyrics.") var request: String
    @Option(name: .customLong("transcriber-model"), help: "Managed SheetSage2 ID or local adapter directory.") var transcriberModel = "music-sheetsage2"
    @Option(name: [.short, .long], help: "Managed YuE2 ID or local model directory.") var model = "music-yue2"
    @Option(name: [.short, .long], help: "New output directory for transcription and saved generation stages.") var output: String
    @Flag(name: .customLong("full-score"), help: "Keep source harmony; the default preserves melody and lets the arrangement change.") var fullScore = false
    func run() async throws {
        let destination = YuE2Transcribe.url(output)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ValidationError("Output directory already exists.") }
        let original = try JSONDecoder().decode(YuE2GenerationPlan.self, from: Data(contentsOf: YuE2Transcribe.url(request)))
        guard original.abc == nil else { throw ValidationError("A recording cover supplies its score; remove abc from the request.") }
        let transcription = try await YuE2Transcribe.transcribe(input: input, model: transcriberModel, melodyOnly: !fullScore)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        try YuE2Transcribe.save(transcription, to: destination.appendingPathComponent("transcription"))
        guard let abc = transcription.abc else {
            throw ValidationError("Raw transcription saved; cover requires a valid score: \(transcription.notationError ?? "unknown notation error").")
        }
        let plan = try YuE2GenerationPlan(style: original.style, lyrics: original.lyrics, planning: fullScore ? .full : .melody,
            abc: abc, seed: original.seed, guidanceScale: original.guidanceScale,
            scoreSampling: original.scoreSampling, semanticSampling: original.semanticSampling, steps: original.steps,
            instrumental: original.instrumental == true)
        let root = model == "music-yue2" ? try ModelResolver().resolve(.yue2).rootURL : YuE2Transcribe.url(model)
        let operation = YuE2GenerationOperation(resources: YuE2Resources(rootURL: root))
        let score = destination.appendingPathComponent("score")
        let semantic = destination.appendingPathComponent("semantic")
        let acoustic = destination.appendingPathComponent("acoustic")
        _ = try await operation.saveStage(.score, plan: plan, to: score, progress: YuE2StageOptions.progress)
        _ = try await operation.saveStage(.semantic, resuming: score, to: semantic, progress: YuE2StageOptions.progress)
        _ = try await operation.saveStage(.acoustic, resuming: semantic, to: acoustic, progress: YuE2StageOptions.progress)
        let waveform = try await operation.decodeLatents(at: acoustic.appendingPathComponent("latent.npy"), progress: YuE2StageOptions.progress)
        _ = try AudioExportService.write(waveform, plan: AudioExportPlan(options: .music), to: destination.appendingPathComponent("audio.wav"))
        print(destination.appendingPathComponent("audio.wav").path)
    }
}
