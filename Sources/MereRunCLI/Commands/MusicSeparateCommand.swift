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
        abstract: "Separate music into stems with native BS-RoFormer models.",
        discussion: """
        Runs pinned MIT-licensed AEmotion BS-RoFormer checkpoints entirely
        locally with native Swift/MLX inference. ViperX produces vocals and
        instrumental; the four-stem model produces drums, bass, other, and
        vocals. Output WAVs use 44.1 kHz stereo float samples. A provenance
        manifest is saved and emitted on stdout.

        Examples:
          mere.run model pull music-separate-bs-roformer-viperx-1297
          mere.run music separate ./song.mp3
          mere.run model pull music-separate-bs-roformer-4stem
          mere.run music separate ./song.wav --model music-separate-bs-roformer-4stem
          mere.run music separate ./song.wav --output-dir ./song-stems --overlap 4
        """
    )

    @Argument(help: "Input audio file (WAV, MP3, M4A, FLAC, or another supported format).")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed BS-RoFormer model id.")
    var model: String = ModelResolver.ModelID.roFormerViperX1297.rawValue

    @Option(name: [.customLong("model-path")], help: "Explicit local root of the selected pinned model snapshot.")
    var modelPath: String?

    @Option(name: [.customShort("o"), .customLong("output-dir")], help: "Stem output directory. Defaults beside the input.")
    var outputDirectory: String?

    @Option(name: [.long], help: "Chunk overlap count. Must divide the selected model's chunk size.")
    var overlap: Int = 2

    @Option(name: [.long], help: "Model compute type: float16 or float32.")
    var dtype: String = "float16"

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics on stderr.")
    var quiet: Bool = false

    func validate() throws {
        let profile: RoFormerModelProfile
        do {
            profile = try RoFormerModelProfile.resolve(modelID: model)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let configuration = try RoFormerResources.loadBundledConfiguration(profile: profile)
        guard overlap > 0, configuration.chunkSize.isMultiple(of: overlap) else {
            throw ValidationError(
                "--overlap must be a positive divisor of \(configuration.chunkSize)"
            )
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
        let profile = try RoFormerModelProfile.resolve(modelID: model)
        let modelRoot = try resolveModelRoot(profile: profile)
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
            CLIStderr.write("Loading \(profile.modelID) from \(modelRoot.path)\n")
        }
        let separator = try RoFormerSeparator.load(
            resources: RoFormerResources(rootURL: modelRoot, profile: profile),
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

        let stemArtifacts = try result.stems.map { stem in
            let outputURL = outputRoot.appendingPathComponent("\(stem.name).wav")
            try MediaAudioIO.writeFloatWAV(
                samples: stem.samples,
                sampleRate: result.sampleRate,
                channels: result.channels,
                to: outputURL
            )
            return MusicSeparationManifest.Stem(
                name: stem.name,
                path: outputURL.path,
                sha256: try ModelArtifactPin.fileSHA256(outputURL)
            )
        }

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
                id: profile.modelID,
                repository: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                license: "MIT",
                weightsSHA256: profile.weightsPin.sha256,
                computeType: dtype.lowercased()
            ),
            chunkSize: separator.checkpoint.configuration.chunkSize,
            overlap: overlap,
            chunks: result.chunkCount,
            elapsedSeconds: result.elapsedSeconds,
            stems: stemArtifacts,
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
            CLIStderr.write("Saved \(result.stems.count) stems and manifest to \(outputRoot.path)\n")
        }
    }

    private func resolveModelRoot(profile: RoFormerModelProfile) throws -> URL {
        if let modelPath { return Self.userURL(modelPath) }
        let modelID: ModelResolver.ModelID = switch profile {
        case .viperX1297: .roFormerViperX1297
        case .fourStem: .roFormerFourStem
        }
        return try ModelResolver().resolve(modelID).rootURL
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
