import ArgumentParser
import Foundation
import MLX
import MediaIO
import MereRunCore

enum MuScriptorOutputFormat: String, CaseIterable, ExpressibleByArgument {
    case midi
    case json
    case jsonl
}

struct MusicTranscribe: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe a full music mix into instrument-separated MIDI with MuScriptor.",
        discussion: """
        MuScriptor runs locally with native MLX inference. Published weights are gated
        and licensed CC BY-NC 4.0; accept the terms on Hugging Face before pulling.

        Examples:
          mere.run model pull music-muscriptor-medium
          mere.run music transcribe ./song.mp3 --output ./song.mid
          mere.run music transcribe ./song.wav --instruments voice,drums,bass
          mere.run music transcribe ./song.wav --format jsonl --output -
        """
    )

    @Argument(help: "Input audio file (WAV, MP3, M4A, FLAC, or another supported format).")
    var audio: String?

    @Option(name: [.customShort("m"), .long], help: "Managed MuScriptor model id or local model directory.")
    var model: String = ModelResolver.ModelID.muScriptorMedium.rawValue

    @Option(name: [.customLong("model-path")], help: "Explicit local MuScriptor model directory.")
    var modelPath: String?

    @Option(name: [.long], help: "Checkpoint architecture fallback: small, medium, or large.")
    var variant: String?

    @Option(name: [.customShort("o"), .long], help: "Output path. Use '-' for stdout. Defaults beside the input.")
    var output: String?

    @Option(name: [.customShort("f"), .long], help: "Output format: midi, json, or jsonl.")
    var format: MuScriptorOutputFormat = .midi

    @Option(name: [.long], help: "Comma-separated expected instrument groups; abbreviations are accepted.")
    var instruments: String?

    @Flag(name: [.long], help: "List valid instrument group names and exit.")
    var listInstruments: Bool = false

    @Flag(name: [.long], help: "Sample tokens instead of greedy decoding.")
    var sampling: Bool = false

    @Option(name: [.customShort("t"), .long], help: "Sampling temperature (only used with --sampling).")
    var temperature: Float = 1

    @Option(name: [.customLong("max-tokens-per-chunk")], help: "Maximum tokens generated for each five-second chunk.")
    var maxTokensPerChunk: Int = 2_000

    @Flag(name: [.customLong("strict-eos")], help: "Fail if any chunk does not emit EOS before the token limit.")
    var strictEOS: Bool = false

    @Option(name: [.customLong("beam-size")], help: "Beam-search width (1 disables beam search).")
    var beamSize: Int = 1

    @Option(name: [.customLong("chunk-batch-size")], help: "Independent five-second chunks decoded together, including beam search.")
    var chunkBatchSize: Int = 4

    @Option(name: [.long], help: "Model compute type: bfloat16, float16, or float32.")
    var dtype: String = "bfloat16"

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics on stderr.")
    var quiet: Bool = false

    func validate() throws {
        if listInstruments { return }
        guard audio != nil else { throw ValidationError("Missing input audio file.") }
        guard maxTokensPerChunk > 0 else {
            throw ValidationError("--max-tokens-per-chunk must be positive")
        }
        guard !sampling || temperature > 0 else {
            throw ValidationError("--temperature must be positive with --sampling")
        }
        guard beamSize > 0 else { throw ValidationError("--beam-size must be positive") }
        guard chunkBatchSize > 0 else { throw ValidationError("--chunk-batch-size must be positive") }
        guard beamSize == 1 || !sampling else {
            throw ValidationError("--beam-size greater than 1 cannot be combined with --sampling")
        }
        if let variant, MuScriptorVariant(rawValue: variant.lowercased()) == nil {
            throw ValidationError("--variant must be small, medium, or large")
        }
        guard ["bfloat16", "float16", "float32"].contains(dtype.lowercased()) else {
            throw ValidationError("--dtype must be bfloat16, float16, or float32")
        }
    }

    func run() throws {
        if listInstruments {
            print(MuScriptorInstruments.names.joined(separator: "\n"))
            return
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        guard let audio else { throw ValidationError("Missing input audio file.") }
        let inputURL = Self.userURL(audio)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input audio file not found: \(inputURL.path)")
        }

        let resolvedVariant = try resolveVariant()
        let modelRoot = try resolveModelRoot(variant: resolvedVariant)
        let resolvedInstruments = try instruments.map { value in
            try MuScriptorInstruments.resolve(value.split(separator: ",").map(String.init))
        }
        let computeType: DType = switch dtype.lowercased() {
        case "float16": .float16
        case "float32": .float32
        default: .bfloat16
        }

        if !quiet {
            CLIStderr.write("Loading MuScriptor \(resolvedVariant.rawValue) from \(modelRoot.path)\n")
            CLIStderr.write("Decoding \(inputURL.path) at 16 kHz mono\n")
        }
        let decoded = try MediaAudioIO.decode(inputURL, targetSampleRate: 16_000, channels: 1)
        let transcriber = try MuScriptorTranscriber.load(
            rootURL: modelRoot,
            variant: resolvedVariant,
            dtype: computeType
        )
        let transcriptionOptions = MuScriptorTranscriptionOptions(
            useSampling: sampling,
            temperature: temperature,
            maxTokensPerChunk: maxTokensPerChunk,
            strictEOS: strictEOS,
            beamSize: beamSize,
            chunkBatchSize: chunkBatchSize,
            instruments: resolvedInstruments
        )
        let progress: MuScriptorTranscriber.ProgressHandler?
        if quiet {
            progress = nil
        } else {
            progress = { completed, total in
                CLIStderr.write("MuScriptor chunks: \(completed)/\(total)\n")
            }
        }
        let transcription = try transcriber.transcribe(
            samples: decoded.samples,
            options: transcriptionOptions,
            progress: progress
        )

        let destination = output ?? Self.defaultOutput(for: inputURL, format: format)
        switch format {
        case .midi:
            try write(MuScriptorMIDI.encode(notes: transcription.notes), to: destination)
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try write(encoder.encode(transcription.events), to: destination)
        case .jsonl:
            let encoder = JSONEncoder()
            let lines = try transcription.events.map { event -> Data in
                var data = try encoder.encode(event)
                data.append(0x0A)
                return data
            }
            try write(lines.reduce(into: Data(), { $0.append($1) }), to: destination)
        }
        if !quiet, destination != "-" {
            CLIStderr.write("Saved \(transcription.notes.count) notes to \(Self.userURL(destination).path)\n")
        }
    }

    private func resolveVariant() throws -> MuScriptorVariant {
        if let variant, let value = MuScriptorVariant(rawValue: variant.lowercased()) {
            return value
        }
        if let value = MuScriptorVariant.resolve(modelID: model) { return value }
        return .medium
    }

    private func resolveModelRoot(variant: MuScriptorVariant) throws -> URL {
        if let modelPath { return Self.userURL(modelPath) }
        let direct = Self.userURL(model)
        if FileManager.default.fileExists(atPath: direct.path) { return direct }
        guard let modelID = ModelResolver.ModelID(rawValue: model),
              MuScriptorVariant.resolve(modelID: modelID.rawValue) != nil else {
            throw MuScriptorError.unsupportedModel(model)
        }
        return try ModelResolver().resolve(modelID).rootURL
    }

    private func write(_ data: Data, to destination: String) throws {
        if destination == "-" {
            try FileHandle.standardOutput.write(contentsOf: data)
        } else {
            try data.write(to: Self.userURL(destination), options: .atomic)
        }
    }

    private static func defaultOutput(for input: URL, format: MuScriptorOutputFormat) -> String {
        input.deletingPathExtension().appendingPathExtension(format == .midi ? "mid" : format.rawValue).path
    }

    private static func userURL(_ path: String) -> URL {
        URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL
    }
}
