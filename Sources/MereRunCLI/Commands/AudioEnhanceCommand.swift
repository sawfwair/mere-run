import ArgumentParser
import Foundation
import MLX
import MediaIO
import MereRunCore

private struct AudioEnhancementManifest: Codable {
    struct Source: Codable {
        let path: String
        let sha256: String
        let sampleRate: Int
        let channels: Int
        let frames: Int64
    }

    struct Preprocessing: Codable {
        let decodedSampleRate: Int
        let decodedChannels: Int
        let decodedFrames: Int
        let modelInputSampleRate: Int
        let modelInputFrames: Int
        let resampler: String
    }

    struct Model: Codable {
        let id: String
        let sourceRepository: String
        let sourceRevision: String
        let artifactRepository: String
        let artifactRevision: String
        let license: String
        let weightsSHA256: String
        let sourceConfigurationSHA256: String
        let computeType: String
    }

    struct Output: Codable {
        let path: String
        let sha256: String
        let sampleRate: Int
        let channels: Int
        let frames: Int
    }

    let schemaVersion: Int
    let createdAt: String
    let source: Source
    let preprocessing: Preprocessing
    let model: Model
    let chunkSize: Int
    let overlap: Int
    let chunks: Int
    let elapsedSeconds: Double
    let output: Output
    let manifestPath: String
}

struct AudioEnhance: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "enhance",
        abstract: "Extend narrowband speech from 16 kHz to 48 kHz with AP-BWE.",
        discussion: """
        Decodes the source to 16 kHz mono, performs band-limited 3x
        interpolation, and reconstructs missing speech bandwidth with the
        pinned MIT-licensed AP-BWE checkpoint using native Swift/MLX inference.
        The output is a 48 kHz mono float WAV. A provenance manifest is saved
        beside the output and emitted as JSON on stdout.

        Examples:
          mere.run model pull audio-enhance-ap-bwe-16kto48k
          mere.run audio enhance ./speech.wav
          mere.run audio enhance ./speech.mp3 --output ./speech-wideband.wav
          mere.run audio enhance ./speech.wav --dtype float16 --overlap 4
        """
    )

    @Argument(help: "Input speech file (WAV, MP3, M4A, FLAC, or another supported format).")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed AP-BWE model id.")
    var model: String = ModelResolver.ModelID.apBWE16kTo48k.rawValue

    @Option(name: [.customLong("model-path")], help: "Explicit local root of the pinned AP-BWE snapshot.")
    var modelPath: String?

    @Option(name: [.customShort("o"), .long], help: "Output 48 kHz mono float WAV path.")
    var output: String?

    @Option(name: [.long], help: "Chunk overlap count. Defaults to the published profile value (2).")
    var overlap: Int?

    @Option(name: [.long], help: "Model compute type: float16 or float32.")
    var dtype: String = "float32"

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics on stderr.")
    var quiet: Bool = false

    func validate() throws {
        guard model == ModelResolver.ModelID.apBWE16kTo48k.rawValue else {
            throw ValidationError("Unsupported audio enhancement model id: \(model)")
        }
        let configuration = try APBWEResources.loadBundledConfiguration()
        let resolvedOverlap = overlap ?? configuration.overlap
        guard resolvedOverlap > 0, configuration.chunkSize.isMultiple(of: resolvedOverlap) else {
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
        let configuration = try APBWEResources.loadBundledConfiguration()
        let resolvedOverlap = overlap ?? configuration.overlap
        let modelRoot = try resolveModelRoot()
        let outputURL = output.map(Self.userURL) ?? Self.defaultOutputURL(for: inputURL)
        let manifestURL = outputURL.deletingPathExtension().appendingPathExtension("json")
        let sourceMetadata = try MediaAudioIO.probe(inputURL)

        if !quiet {
            CLIStderr.write("Decoding \(inputURL.path) at 16 kHz mono\n")
        }
        let decoded = try MediaAudioIO.decode(
            inputURL,
            targetSampleRate: configuration.lowSampleRate,
            channels: 1
        )
        guard !decoded.samples.isEmpty else {
            throw ValidationError("Input audio decoded to an empty buffer.")
        }
        let modelInput = APBWEAudio.upsample16kTo48k(decoded.samples)
        let computeType: DType = dtype.lowercased() == "float16" ? .float16 : .float32
        if !quiet {
            CLIStderr.write("Loading \(model) from \(modelRoot.path)\n")
        }
        let enhancer = try APBWEEnhancer.load(
            resources: APBWEResources(rootURL: modelRoot),
            dtype: computeType
        )
        let result = try enhancer.enhance(
            narrowband48kSamples: modelInput,
            sampleRate: configuration.highSampleRate,
            channels: 1,
            overlap: resolvedOverlap
        ) { completed, total in
            if !quiet {
                CLIStderr.write("AP-BWE chunks: \(completed)/\(total)\n")
            }
        }
        try MediaAudioIO.writeFloatWAV(
            samples: result.samples,
            sampleRate: result.sampleRate,
            channels: result.channels,
            to: outputURL
        )

        let manifest = AudioEnhancementManifest(
            schemaVersion: 1,
            createdAt: ISO8601DateFormatter().string(from: Date()),
            source: .init(
                path: inputURL.path,
                sha256: try ModelArtifactPin.fileSHA256(inputURL),
                sampleRate: sourceMetadata.sampleRate,
                channels: sourceMetadata.channelCount,
                frames: sourceMetadata.frameCount
            ),
            preprocessing: .init(
                decodedSampleRate: decoded.sampleRate,
                decodedChannels: decoded.channelCount,
                decodedFrames: decoded.samples.count / decoded.channelCount,
                modelInputSampleRate: configuration.highSampleRate,
                modelInputFrames: modelInput.count,
                resampler: "windowed-sinc-lanczos-radius-32"
            ),
            model: .init(
                id: model,
                sourceRepository: APBWEResources.sourceRepository,
                sourceRevision: APBWEResources.sourceRevision,
                artifactRepository: APBWEResources.artifactRepository,
                artifactRevision: APBWEResources.artifactRevision,
                license: "MIT",
                weightsSHA256: APBWEResources.weightsPin.sha256,
                sourceConfigurationSHA256: APBWEResources.sourceConfigurationPin.sha256,
                computeType: dtype.lowercased()
            ),
            chunkSize: configuration.chunkSize,
            overlap: resolvedOverlap,
            chunks: result.chunkCount,
            elapsedSeconds: result.elapsedSeconds,
            output: .init(
                path: outputURL.path,
                sha256: try ModelArtifactPin.fileSHA256(outputURL),
                sampleRate: result.sampleRate,
                channels: result.channels,
                frames: result.frameCount
            ),
            manifestPath: manifestURL.path
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(manifest)
        data.append(0x0A)
        try FileManager.default.createDirectory(
            at: manifestURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: manifestURL, options: .atomic)
        try FileHandle.standardOutput.write(contentsOf: data)

        if !quiet {
            CLIStderr.write("Saved enhanced audio to \(outputURL.path)\n")
            CLIStderr.write("Saved provenance manifest to \(manifestURL.path)\n")
        }
    }

    private func resolveModelRoot() throws -> URL {
        if let modelPath { return Self.userURL(modelPath) }
        return try ModelResolver().resolve(.apBWE16kTo48k).rootURL
    }

    private static func defaultOutputURL(for input: URL) -> URL {
        input.deletingLastPathComponent().appendingPathComponent(
            input.deletingPathExtension().lastPathComponent + "-enhanced.wav"
        )
    }

    private static func userURL(_ value: String) -> URL {
        URL(fileURLWithPath: NSString(string: value).expandingTildeInPath).standardizedFileURL
    }
}
