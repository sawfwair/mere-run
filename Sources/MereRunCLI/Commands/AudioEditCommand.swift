import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct AudioEdit: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "edit",
        abstract: "Generate or edit speech with native AuK instruction conditioning.",
        discussion: "AuK requires a base or Flash checkpoint and the separate Qwen2.5-Omni-3B encoder. "
            + "Use --audio for editing or voice conditioning; text-only generation requires --duration. "
            + "Flash always uses its fixed four-step schedule with guidance disabled. "
            + "Experimental native FP32 runtime; see the model guide for qualification results."
    )
    @Argument(help: "Natural-language speech generation or editing instruction.")
    var instruction: String
    @Option(name: .long, help: "Reference audio path for editing or voice conditioning.")
    var audio: String?
    @Option(name: .long, help: "Managed model: audio-auk-base or audio-auk-flash.")
    var model = "audio-auk-base"
    @Option(name: .customLong("model-path"), help: "Local original AuK checkpoint directory.")
    var modelPath: String?
    @Option(name: .customLong("thinker-path"), help: "Local Qwen2.5-Omni-3B checkpoint directory.")
    var thinkerPath: String?
    @Option(name: [.short, .long], help: "Output WAV path.")
    var output = "auk.wav"
    @Option(name: .long, help: "Output duration in seconds; otherwise follows the reference latent length.")
    var duration: Double?
    @Option(name: .long, help: "Base model Euler steps (1...1000); Flash uses four.")
    var steps = 32
    @Option(name: .long, help: "Base model classifier-free guidance; Flash disables guidance.")
    var guidance: Float = 2
    @Option(name: .long, help: "Random seed.")
    var seed: UInt64 = 42
    @Flag(name: [.short, .long], help: "Suppress progress diagnostics.")
    var quiet = false

    var options: AuKGenerationOptions {
        var result = AuKGenerationOptions()
        result.variant = model == "audio-auk-flash" ? .flash : .base
        result.duration = duration
        result.steps = steps
        result.guidance = guidance
        result.seed = seed
        return result
    }

    func validate() throws {
        guard ["audio-auk-base", "audio-auk-flash"].contains(model) else {
            throw ValidationError("--model must be audio-auk-base or audio-auk-flash")
        }
        guard !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Instruction cannot be empty")
        }
        try options.validate(hasReference: audio != nil)
        if let audio, Self.url(audio) == Self.url(output) {
            throw ValidationError("Output must differ from reference audio")
        }
    }

    func run() throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let modelID: ModelResolver.ModelID = model == "audio-auk-flash" ? .aukFlash : .aukBase
        let root = try modelPath.map(Self.url) ?? ModelResolver().resolve(modelID).rootURL
        let thinker = try thinkerPath.map(Self.url) ?? ModelResolver().resolve(.aukThinker).rootURL
        let reference24k = try audio.map { try MediaAudioIO.decode(Self.url($0), targetSampleRate: 24000, channels: 1).samples }
        let reference16k = try audio.map { try MediaAudioIO.decode(Self.url($0), targetSampleRate: 16000, channels: 1).samples }
        let samples = try AuKGenerator.generate(instruction: instruction, modelRoot: root, thinkerRoot: thinker,
            reference24k: reference24k, reference16k: reference16k, options: options) { message in
                if !quiet { CLIStderr.write(message + "\n") }
            }
        let destination = Self.url(output)
        try MediaAudioIO.writeFloatWAV(samples: samples, sampleRate: AuKGenerator.sampleRate, channels: 1, to: destination)
        struct Result: Encodable {
            let model: String
            let output: String
            let sampleRate: Int
            let frames: Int
            let seed: UInt64
            let steps: Int
            let guidance: Float
            let sourceRevision: String
            let validation: String
        }
        let result = Result(model: model, output: destination.path, sampleRate: AuKGenerator.sampleRate,
                            frames: samples.count, seed: seed, steps: options.variant == .flash ? 4 : steps,
                            guidance: options.variant == .flash ? 0 : guidance,
                            sourceRevision: AuKGenerator.sourceRevision, validation: "experimental")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(result)
        data.append(10)
        try FileHandle.standardOutput.write(contentsOf: data)
    }

    private static func url(_ path: String) -> URL {
        URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL
    }
}
