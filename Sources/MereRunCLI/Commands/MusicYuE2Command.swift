import ArgumentParser
import AudioCore
import Foundation
import MLX
import MereRunCore

struct MusicYuE2: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "yue2", abstract: "Plan, reuse, and inspect native YuE2 generation stages.",
        subcommands: [YuE2Plan.self, YuE2Semantic.self, YuE2Synthesize.self, YuE2Decode.self,
                      YuE2Inspect.self, YuE2Validate.self, YuE2Instrumentalize.self, YuE2Transcribe.self, YuE2Cover.self,
                      YuE2Export.self, YuE2Batch.self, YuE2Serve.self])
}

struct YuE2StageOptions: ParsableArguments {
    @Option(name: [.short, .long], help: "Managed YuE2 model ID or local model directory.")
    var model = "music-yue2"
    @Option(name: [.short, .long], help: "New output directory; existing directories are refused.")
    var output: String
    @Option(name: .customLong("memory-mode"), help: "Weight loading: staged, resident, or low-memory.")
    var memoryMode: YuE2MemoryMode = .staged

    func operation() throws -> YuE2GenerationOperation {
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let root = model == "music-yue2" ? try ModelResolver().resolve(.yue2).rootURL : resolveUserPath(model)
        return YuE2GenerationOperation(resources: YuE2Resources(rootURL: root), memoryMode: memoryMode)
    }

    static func progress(_ event: YuE2Progress) {
        if event.completed == 1 || event.completed == event.total || event.completed % 50 == 0 {
            CLIStderr.write("YuE2 \(event.stage.rawValue): \(event.completed)/\(event.total)\n")
        }
    }
}

struct YuE2Plan: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "plan", abstract: "Save an exact YuE2 score stage from a JSON request.")
    @Option(name: .long, help: "JSON request; style is required, other native plan fields have defaults.") var request: String
    @OptionGroup var options: YuE2StageOptions
    func run() async throws {
        let plan = try JSONDecoder().decode(YuE2GenerationPlan.self, from: Data(contentsOf: resolveUserPath(request)))
        let result = try await options.operation().saveStage(.score, plan: plan, to: resolveUserPath(options.output),
                                                            progress: YuE2StageOptions.progress)
        try printYuE2JSON(result)
    }
}

struct YuE2Semantic: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "semantic", abstract: "Generate music tokens from a saved score stage.")
    @Argument(help: "Saved score-stage directory.") var input: String
    @OptionGroup var options: YuE2StageOptions
    func run() async throws {
        try printYuE2JSON(await options.operation().saveStage(.semantic, resuming: resolveUserPath(input),
            to: resolveUserPath(options.output), progress: YuE2StageOptions.progress))
    }
}

struct YuE2Synthesize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "synthesize", abstract: "Generate acoustic latents from saved music tokens.")
    @Argument(help: "Saved semantic-stage directory.") var input: String
    @OptionGroup var options: YuE2StageOptions
    func run() async throws {
        try printYuE2JSON(await options.operation().saveStage(.acoustic, resuming: resolveUserPath(input),
            to: resolveUserPath(options.output), progress: YuE2StageOptions.progress))
    }
}

struct YuE2Decode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "decode", abstract: "Decode saved NumPy latents with only the YuE2 VAE.")
    @Argument(help: "Saved acoustic-stage directory or latent.npy file.") var input: String
    @Option(name: .long, help: "Local VAE directory; defaults to the installed YuE2 VAE.") var vae: String?
    @Option(name: [.short, .long], help: "New output WAV file.") var output: String
    @Option(name: .customLong("core-frames"), help: "VAE tile size in frames (default: 1024).") var coreFrames: Int?
    @Flag(name: .long, help: "Decode the complete latent sequence in one pass.") var full = false
    func run() async throws {
        let destination = resolveUserPath(output)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ValidationError("Output already exists.") }
        var latents = resolveUserPath(input)
        if latents.pathExtension != "npy" {
            let saved = try YuE2SavedStage.load(from: latents)
            guard saved.stage == .acoustic else { throw ValidationError("Decode requires an acoustic stage.") }
            latents.appendPathComponent("latent.npy")
        }
        try MLXBundleSupport.ensureAvailable(quiet: false)
        // Reject malformed imports before resolving even an uninstalled managed decoder.
        _ = try YuE2SavedStage.loadLatents(from: latents)
        let root: URL
        if vae == YuE2Resources.legacyVAEModelID { root = try ModelResolver().resolve(.yue2VAELegacy).rootURL }
        else if vae == YuE2Resources.modelID { root = try ModelResolver().resolve(.yue2).rootURL.appendingPathComponent("vae") }
        else { root = try vae.map(resolveUserPath) ?? ModelResolver().resolve(.yue2).rootURL.appendingPathComponent("vae") }
        let operation = YuE2GenerationOperation(resources: YuE2Resources(rootURL: root.deletingLastPathComponent()))
        let waveform = try await operation.decodeLatents(at: latents, vaeRoot: root, coreFrames: coreFrames,
                                                          full: full, progress: YuE2StageOptions.progress)
        _ = try AudioExportService.write(waveform, plan: AudioExportPlan(options: .speechStreaming), to: destination)
        print(destination.path)
    }
}

struct YuE2Inspect: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "inspect", abstract: "Verify a saved YuE2 stage without loading models.")
    @Argument(help: "Saved stage directory.") var input: String
    func run() throws { try printYuE2JSON(YuE2SavedStage.load(from: resolveUserPath(input))) }
}

struct YuE2Validate: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "validate", abstract: "Validate a YuE2 JSON request without loading models.")
    @Argument(help: "JSON request file.") var request: String
    @Option(name: .long, help: "Optional qwen.tiktoken file for exact prompt and context preflight without model weights.") var tokenizer: String?
    func run() throws {
        let plan = try JSONDecoder().decode(YuE2GenerationPlan.self, from: Data(contentsOf: resolveUserPath(request)))
        if plan.instrumental == true, let abc = plan.abc { _ = try YuE2Score(abc: abc).instrumentalized() }
        if let tokenizer { try printYuE2JSON(plan.preflight(tokenizerURL: resolveUserPath(tokenizer))) }
        else { try printYuE2JSON(plan) }
    }
}

extension YuE2Score.OverlapPolicy: ExpressibleByArgument {}

struct YuE2Instrumentalize: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "instrumentalize", abstract: "Move a score's vocal melody into the instrumental voice.")
    @Argument(help: "Native YuE2 or SheetSage2 ABC file.") var input: String
    @Option(name: [.short, .long], help: "New ABC output file.") var output: String
    @Option(name: .long, help: "Overlapping accompaniment: vocal priority (default) or error.") var overlap: YuE2Score.OverlapPolicy = .vocal
    @Flag(name: .customLong("drop-chords"), help: "Remove chord labels for a melody-only cover.") var dropChords = false
    func run() throws {
        let destination = resolveUserPath(output)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ValidationError("Output already exists.") }
        let source = try String(contentsOf: resolveUserPath(input), encoding: .utf8)
        let result = try YuE2Score(abc: source).instrumentalized(overlap: overlap, keepChords: !dropChords)
        try result.abc.write(to: destination, atomically: true, encoding: .utf8)
        try printYuE2JSON(result)
    }
}

private func printYuE2JSON(_ value: some Encodable) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    print(String(decoding: try encoder.encode(value), as: UTF8.self))
}

private func resolveUserPath(_ path: String) -> URL {
    URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
}
