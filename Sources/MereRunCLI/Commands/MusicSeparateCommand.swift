import ArgumentParser
import Foundation
import MLX
import MediaIO
import MereRunCore

private struct MusicSeparationManifest: Codable {
    struct Source: Codable {
        let path: String
        let sha256: String
        let sampleRate: Int
        let channels: Int
        let frames: Int
    }

    struct Model: Codable {
        let id: String
        let repository: String
        let revision: String
        let license: String
        let weightsSHA256: String
        let computeType: String
    }

    struct Stem: Codable {
        let name: String
        let path: String
        let sha256: String
    }

    let schemaVersion: Int
    let createdAt: String
    let source: Source
    let model: Model
    let chunkSize: Int
    let overlap: Int
    let chunks: Int
    let elapsedSeconds: Double
    let stems: [Stem]
    let manifestPath: String
}

struct MusicSeparate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "separate",
        abstract: "Separate vocals and instrumental audio with ViperX BS-RoFormer.",
        discussion: """
        Runs the pinned MIT-licensed ViperX 1297 BS-RoFormer checkpoint entirely
        locally with native Swift/MLX inference. Output WAVs use 44.1 kHz stereo
        float samples. A provenance manifest is saved and emitted on stdout.

        Examples:
          mere.run model pull music-separate-bs-roformer-viperx-1297
          mere.run music separate ./song.mp3
          mere.run music separate ./song.wav --output-dir ./song-stems --overlap 4
        """
    )

    @Argument(help: "Input audio file (WAV, MP3, M4A, FLAC, or another supported format).")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed ViperX model id or local model directory.")
    var model: String = ModelResolver.ModelID.roFormerViperX1297.rawValue

    @Option(name: [.customLong("model-path")], help: "Explicit local root of the pinned AEmotion model snapshot.")
    var modelPath: String?

    @Option(name: [.customShort("o"), .customLong("output-dir")], help: "Stem output directory. Defaults beside the input.")
    var outputDirectory: String?

    @Option(name: [.long], help: "Chunk overlap count. Must divide 352800; 2 matches the published inference config.")
    var overlap: Int = 2

    @Option(name: [.long], help: "Model compute type: float16 or float32.")
    var dtype: String = "float16"

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics on stderr.")
    var quiet: Bool = false

    func validate() throws {
        guard overlap > 0, 352_800.isMultiple(of: overlap) else {
            throw ValidationError("--overlap must be a positive divisor of 352800")
        }
        guard ["float16", "float32"].contains(dtype.lowercased()) else {
            throw ValidationError("--dtype must be float16 or float32")
        }
    }

    func run() throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let inputURL = Self.userURL(audio)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input audio file not found: \(inputURL.path)")
        }
        let modelRoot = try resolveModelRoot()
        let outputRoot = outputDirectory.map(Self.userURL)
            ?? Self.defaultOutputDirectory(for: inputURL)
        try FileManager.default.createDirectory(
            at: outputRoot,
            withIntermediateDirectories: true
        )

        if !quiet {
            CLIStderr.write("Decoding \(inputURL.path) at 44.1 kHz stereo\n")
        }
        let decoded = try MediaAudioIO.decode(
            inputURL,
            targetSampleRate: 44_100,
            channels: 2
        )
        let computeType: DType = dtype.lowercased() == "float32" ? .float32 : .float16
        if !quiet {
            CLIStderr.write("Loading ViperX BS-RoFormer from \(modelRoot.path)\n")
        }
        let separator = try RoFormerSeparator.load(
            resources: RoFormerResources(rootURL: modelRoot),
            dtype: computeType
        )
        let progress: RoFormerSeparator.ProgressHandler?
        if quiet {
            progress = nil
        } else {
            progress = { completed, total in
                CLIStderr.write("RoFormer chunks: \(completed)/\(total)\n")
            }
        }
        let result = try separator.separate(
            interleavedSamples: decoded.samples,
            sampleRate: 44_100,
            channels: 2,
            overlap: overlap,
            progress: progress
        )

        let vocalsURL = outputRoot.appendingPathComponent("vocals.wav")
        let instrumentalURL = outputRoot.appendingPathComponent("instrumental.wav")
        try MediaAudioIO.writeFloatWAV(
            samples: result.vocals,
            sampleRate: result.sampleRate,
            channels: result.channels,
            to: vocalsURL
        )
        try MediaAudioIO.writeFloatWAV(
            samples: result.instrumental,
            sampleRate: result.sampleRate,
            channels: result.channels,
            to: instrumentalURL
        )

        let manifestURL = outputRoot.appendingPathComponent("separation.json")
        let manifest = MusicSeparationManifest(
            schemaVersion: 1,
            createdAt: ISO8601DateFormatter().string(from: Date()),
            source: .init(
                path: inputURL.path,
                sha256: try ModelArtifactPin.fileSHA256(inputURL),
                sampleRate: result.sampleRate,
                channels: result.channels,
                frames: result.frameCount
            ),
            model: .init(
                id: RoFormerResources.modelID,
                repository: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                license: "MIT",
                weightsSHA256: RoFormerResources.weightsPin.sha256,
                computeType: dtype.lowercased()
            ),
            chunkSize: separator.checkpoint.configuration.chunkSize,
            overlap: overlap,
            chunks: result.chunkCount,
            elapsedSeconds: result.elapsedSeconds,
            stems: [
                .init(
                    name: "vocals",
                    path: vocalsURL.path,
                    sha256: try ModelArtifactPin.fileSHA256(vocalsURL)
                ),
                .init(
                    name: "instrumental",
                    path: instrumentalURL.path,
                    sha256: try ModelArtifactPin.fileSHA256(instrumentalURL)
                ),
            ],
            manifestPath: manifestURL.path
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var manifestData = try encoder.encode(manifest)
        manifestData.append(0x0A)
        try manifestData.write(to: manifestURL, options: .atomic)
        try FileHandle.standardOutput.write(contentsOf: manifestData)

        if !quiet {
            CLIStderr.write("Saved vocals, instrumental, and manifest to \(outputRoot.path)\n")
        }
    }

    private func resolveModelRoot() throws -> URL {
        if let modelPath { return Self.userURL(modelPath) }
        let direct = Self.userURL(model)
        if FileManager.default.fileExists(atPath: direct.path) { return direct }
        guard model == ModelResolver.ModelID.roFormerViperX1297.rawValue else {
            throw ValidationError(
                "Unsupported separation model '\(model)'; expected \(RoFormerResources.modelID)"
            )
        }
        return try ModelResolver().resolve(.roFormerViperX1297).rootURL
    }

    private static func defaultOutputDirectory(for input: URL) -> URL {
        input.deletingLastPathComponent().appendingPathComponent(
            input.deletingPathExtension().lastPathComponent + "-stems",
            isDirectory: true
        )
    }

    private static func userURL(_ value: String) -> URL {
        URL(fileURLWithPath: NSString(string: value).expandingTildeInPath).standardizedFileURL
    }
}
