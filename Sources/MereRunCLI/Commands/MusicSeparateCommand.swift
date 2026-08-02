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

private enum MusicRoFormerSelection {
    case bandSplit(RoFormerModelProfile)
    case melBand(MelBandRoFormerProfile)

    static func resolve(modelID: String) throws -> Self {
        if let profile = RoFormerModelProfile.allCases.first(where: { $0.modelID == modelID }) {
            return .bandSplit(profile)
        }
        if let profile = MelBandRoFormerProfile.allCases.first(where: { $0.modelID == modelID }) {
            return .melBand(profile)
        }
        throw RoFormerError.invalidConfiguration("unsupported RoFormer model id \(modelID)")
    }

    var modelID: String {
        switch self {
        case .bandSplit(let profile): profile.modelID
        case .melBand(let profile): profile.modelID
        }
    }

    var resolverID: ModelResolver.ModelID {
        switch self {
        case .bandSplit(.viperX1297): .roFormerViperX1297
        case .bandSplit(.fourStem): .roFormerFourStem
        case .melBand(.dereverb): .melRoFormerDereverb
        case .melBand(.denoise): .melRoFormerDenoise
        }
    }

    var weightsPin: ModelArtifactPin {
        switch self {
        case .bandSplit(let profile): profile.weightsPin
        case .melBand(let profile): profile.weightsPin
        }
    }

    func chunkSize() throws -> Int {
        switch self {
        case .bandSplit(let profile):
            try RoFormerResources.loadBundledConfiguration(profile: profile).chunkSize
        case .melBand(let profile):
            try MelBandRoFormerResources.loadBundledConfiguration(profile: profile).chunkSize
        }
    }

    func defaultOverlap() throws -> Int {
        switch self {
        case .bandSplit(let profile):
            try RoFormerResources.loadBundledConfiguration(profile: profile).overlap
        case .melBand(let profile):
            try MelBandRoFormerResources.loadBundledConfiguration(profile: profile).overlap
        }
    }
}

struct MusicSeparate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "separate",
        abstract: "Separate or restore audio with native RoFormer models.",
        discussion: """
        Runs pinned MIT-licensed AEmotion RoFormer checkpoints entirely locally
        with native Swift/MLX inference. ViperX produces vocals and instrumental;
        the four-stem model produces drums, bass, other, and vocals; MelBand
        models produce dereverberated or denoised audio. Output WAVs use 44.1 kHz
        stereo float samples. A provenance manifest is saved and emitted on stdout.

        Examples:
          mere.run model pull music-separate-bs-roformer-viperx-1297
          mere.run music separate ./song.mp3
          mere.run model pull music-separate-bs-roformer-4stem
          mere.run music separate ./song.wav --model music-separate-bs-roformer-4stem
          mere.run model pull music-separate-mel-roformer-dereverb
          mere.run music separate ./room.wav --model music-separate-mel-roformer-dereverb
          mere.run music separate ./song.wav --output-dir ./song-stems --overlap 4
        """
    )

    @Argument(help: "Input audio file (WAV, MP3, M4A, FLAC, or another supported format).")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed RoFormer model id.")
    var model: String = ModelResolver.ModelID.roFormerViperX1297.rawValue

    @Option(name: [.customLong("model-path")], help: "Explicit local root of the selected pinned model snapshot.")
    var modelPath: String?

    @Option(name: [.customShort("o"), .customLong("output-dir")], help: "Stem output directory. Defaults beside the input.")
    var outputDirectory: String?

    @Option(name: [.long], help: "Chunk overlap count. Defaults to the selected model's published value.")
    var overlap: Int?

    @Option(name: [.long], help: "Model compute type: float16 or float32.")
    var dtype: String = "float16"

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics on stderr.")
    var quiet: Bool = false

    func validate() throws {
        let selection: MusicRoFormerSelection
        do {
            selection = try MusicRoFormerSelection.resolve(modelID: model)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let chunkSize = try selection.chunkSize()
        let resolvedOverlap = try overlap ?? selection.defaultOverlap()
        guard resolvedOverlap > 0, chunkSize.isMultiple(of: resolvedOverlap) else {
            throw ValidationError(
                "--overlap must be a positive divisor of \(chunkSize)"
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
        let selection = try MusicRoFormerSelection.resolve(modelID: model)
        let resolvedOverlap = try overlap ?? selection.defaultOverlap()
        let modelRoot = try resolveModelRoot(selection: selection)
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
            CLIStderr.write("Loading \(selection.modelID) from \(modelRoot.path)\n")
        }
        let progress: @Sendable (Int, Int) -> Void = { completed, total in
            if !quiet {
                CLIStderr.write("RoFormer chunks: \(completed)/\(total)\n")
            }
        }
        let result: RoFormerSeparationResult
        switch selection {
        case .bandSplit(let profile):
            let separator = try RoFormerSeparator.load(
                resources: RoFormerResources(rootURL: modelRoot, profile: profile),
                dtype: computeType
            )
            result = try separator.separate(
                interleavedSamples: decoded.samples,
                sampleRate: 44_100,
                channels: 2,
                overlap: resolvedOverlap,
                progress: progress
            )
        case .melBand(let profile):
            let separator = try MelBandRoFormerSeparator.load(
                resources: MelBandRoFormerResources(rootURL: modelRoot, profile: profile),
                dtype: computeType
            )
            result = try separator.separate(
                interleavedSamples: decoded.samples,
                sampleRate: 44_100,
                channels: 2,
                overlap: resolvedOverlap,
                progress: progress
            )
        }

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
                id: selection.modelID,
                repository: RoFormerResources.repository,
                revision: RoFormerResources.revision,
                license: "MIT",
                weightsSHA256: selection.weightsPin.sha256,
                computeType: dtype.lowercased()
            ),
            chunkSize: try selection.chunkSize(),
            overlap: resolvedOverlap,
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

    private func resolveModelRoot(selection: MusicRoFormerSelection) throws -> URL {
        if let modelPath { return Self.userURL(modelPath) }
        return try ModelResolver().resolve(selection.resolverID).rootURL
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
