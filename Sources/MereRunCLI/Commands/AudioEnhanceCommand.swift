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

private struct UniverSREnhancementManifest: Codable {
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
        let effectiveInputRate: Int
        let modelInputSampleRate: Int
        let modelInputFrames: Int
        let resampler: String
    }

    struct Model: Codable {
        let id: String
        let sourceRepository: String
        let sourceRevision: String
        let codeLicense: String
        let artifactRepository: String
        let artifactRevision: String
        let checkpointLicense: String
        let weightsSHA256: String
        let sourceConfigurationSHA256: String
        let modelCardSHA256: String
        let computeType: String
    }

    struct Inference: Codable {
        let odeMethod: String
        let odeSteps: Int
        let guidanceScale: Float
        let seed: UInt64
        let chunkSeconds: Int
        let chunks: Int
        let elapsedSeconds: Double
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
    let inference: Inference
    let output: Output
    let manifestPath: String
}

struct AudioEnhance: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "enhance",
        abstract: "Extend speech or general-audio bandwidth to 48 kHz.",
        discussion: """
        AP-BWE is the default for deterministic 16 kHz speech bandwidth
        extension. Select audio-enhance-universr-audio for flow-matching
        super-resolution of speech, music, or sound effects from an effective
        8, 12, 16, or 24 kHz bandwidth. Both runtimes use native Swift/MLX.
        The output is a 48 kHz mono float WAV. A provenance manifest is saved
        beside the output and emitted as JSON on stdout.

        Examples:
          mere.run model pull audio-enhance-ap-bwe-16kto48k
          mere.run audio enhance ./speech.wav
          mere.run audio enhance ./speech.mp3 --output ./speech-wideband.wav
          mere.run audio enhance ./speech.wav --dtype float16 --overlap 4
          mere.run model pull audio-enhance-universr-audio
          mere.run audio enhance ./music-12k.wav --model audio-enhance-universr-audio
          mere.run audio enhance ./limited-48k.wav --model audio-enhance-universr-audio --input-rate 16000
        """
    )

    @Argument(help: "Input audio file (WAV, MP3, M4A, FLAC, or another supported format).")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed audio enhancement model id.")
    var model: String = ModelResolver.ModelID.apBWE16kTo48k.rawValue

    @Option(name: [.customLong("model-path")], help: "Explicit local root of the pinned model snapshot.")
    var modelPath: String?

    @Option(name: [.customShort("o"), .long], help: "Output 48 kHz mono float WAV path.")
    var output: String?

    @Option(name: [.long], help: "Chunk overlap count. Defaults to the published profile value (2).")
    var overlap: Int?

    @Option(name: [.customLong("input-rate")], help: "UniverSR effective input bandwidth rate: 8000, 12000, 16000, or 24000 Hz.")
    var inputRate: Int?

    @Option(name: [.customLong("ode-method")], help: "UniverSR ODE method: euler, midpoint, or rk4.")
    var odeMethod: String = "midpoint"

    @Option(name: [.customLong("ode-steps")], help: "UniverSR ODE integration steps.")
    var odeSteps: Int = 4

    @Option(name: [.customLong("guidance-scale")], help: "UniverSR classifier-free guidance scale.")
    var guidanceScale: Float = 1.5

    @Option(name: [.long], help: "UniverSR noise seed.")
    var seed: UInt64 = 42

    @Option(name: [.customLong("chunk-seconds")], help: "UniverSR sequential chunk length in seconds (minimum 3).")
    var chunkSeconds: Int = 10

    @Option(name: [.long], help: "Model compute type: float16 or float32.")
    var dtype: String = "float32"

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics on stderr.")
    var quiet: Bool = false

    func validate() throws {
        let supported = [
            ModelResolver.ModelID.apBWE16kTo48k.rawValue,
            ModelResolver.ModelID.univerSRAudio.rawValue,
        ]
        guard supported.contains(model) else {
            throw ValidationError("Unsupported audio enhancement model id: \(model)")
        }
        if model == ModelResolver.ModelID.apBWE16kTo48k.rawValue {
            let configuration = try APBWEResources.loadBundledConfiguration()
            let resolvedOverlap = overlap ?? configuration.overlap
            guard resolvedOverlap > 0, configuration.chunkSize.isMultiple(of: resolvedOverlap) else {
                throw ValidationError(
                    "--overlap must be a positive divisor of \(configuration.chunkSize)"
                )
            }
            if let inputRate, inputRate != configuration.lowSampleRate {
                throw ValidationError("AP-BWE requires --input-rate 16000 when the option is supplied")
            }
        } else {
            guard overlap == nil else {
                throw ValidationError("--overlap applies only to AP-BWE")
            }
            if let inputRate, ![8_000, 12_000, 16_000, 24_000].contains(inputRate) {
                throw ValidationError("--input-rate must be 8000, 12000, 16000, or 24000")
            }
            guard UniverSRODEMethod(rawValue: odeMethod.lowercased()) != nil else {
                throw ValidationError("--ode-method must be euler, midpoint, or rk4")
            }
            guard odeSteps > 0 else { throw ValidationError("--ode-steps must be positive") }
            guard guidanceScale >= 0, guidanceScale.isFinite else {
                throw ValidationError("--guidance-scale must be finite and non-negative")
            }
            guard chunkSeconds >= 3 else {
                throw ValidationError("--chunk-seconds must be at least 3")
            }
        }
        guard ["float16", "float32"].contains(dtype.lowercased()) else {
            throw ValidationError("--dtype must be float16 or float32")
        }
    }

    func run() throws {
        if model == ModelResolver.ModelID.univerSRAudio.rawValue {
            try runUniverSR()
        } else {
            try runAPBWE()
        }
    }

    private func runAPBWE() throws {
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

    private func runUniverSR() throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let inputURL = Self.userURL(audio)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input audio file not found: \(inputURL.path)")
        }
        let configuration = try UniverSRResources.loadBundledConfiguration()
        let sourceMetadata = try MediaAudioIO.probe(inputURL)
        let effectiveInputRate = inputRate ?? sourceMetadata.sampleRate
        guard configuration.supportedInputRates.contains(effectiveInputRate) else {
            if sourceMetadata.sampleRate == configuration.transform.samplingRate, inputRate == nil {
                throw ValidationError(
                    "48 kHz input requires --input-rate to declare its effective bandwidth "
                        + "(8000, 12000, 16000, or 24000)"
                )
            }
            throw ValidationError(
                "UniverSR effective input rate must be 8000, 12000, 16000, or 24000 Hz"
            )
        }
        guard let method = UniverSRODEMethod(rawValue: odeMethod.lowercased()) else {
            throw ValidationError("--ode-method must be euler, midpoint, or rk4")
        }
        let modelRoot = try resolveModelRoot()
        let outputURL = output.map(Self.userURL) ?? Self.defaultOutputURL(for: inputURL)
        let manifestURL = outputURL.deletingPathExtension().appendingPathExtension("json")

        if !quiet {
            CLIStderr.write("Decoding \(inputURL.path) at \(effectiveInputRate) Hz mono\n")
        }
        let decoded = try MediaAudioIO.decode(
            inputURL,
            targetSampleRate: effectiveInputRate,
            channels: 1
        )
        guard !decoded.samples.isEmpty else {
            throw ValidationError("Input audio decoded to an empty buffer.")
        }
        let modelInput = try UniverSRAudio.upsampleTo48k(
            decoded.samples,
            inputRate: effectiveInputRate
        )
        let computeType: DType = dtype.lowercased() == "float16" ? .float16 : .float32
        if !quiet {
            CLIStderr.write("Loading \(model) from \(modelRoot.path)\n")
        }
        let enhancer = try UniverSREnhancer.load(
            resources: UniverSRResources(rootURL: modelRoot),
            dtype: computeType
        )
        let result = try enhancer.enhance(
            lowResolution48kSamples: modelInput,
            options: UniverSREnhancementOptions(
                inputRateHz: effectiveInputRate,
                odeMethod: method,
                odeSteps: odeSteps,
                guidanceScale: guidanceScale,
                seed: seed,
                chunkSeconds: chunkSeconds
            )
        ) { completed, total in
            if !quiet {
                CLIStderr.write("UniverSR chunks: \(completed)/\(total)\n")
            }
        }
        try MediaAudioIO.writeFloatWAV(
            samples: result.samples,
            sampleRate: result.sampleRate,
            channels: result.channels,
            to: outputURL
        )

        let manifest = UniverSREnhancementManifest(
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
                effectiveInputRate: effectiveInputRate,
                modelInputSampleRate: configuration.transform.samplingRate,
                modelInputFrames: modelInput.count,
                resampler: "windowed-sinc-lanczos-radius-32"
            ),
            model: .init(
                id: model,
                sourceRepository: UniverSRResources.sourceRepository,
                sourceRevision: UniverSRResources.sourceRevision,
                codeLicense: "MIT",
                artifactRepository: UniverSRResources.artifactRepository,
                artifactRevision: UniverSRResources.artifactRevision,
                checkpointLicense: "CC BY 4.0",
                weightsSHA256: UniverSRResources.weightsPin.sha256,
                sourceConfigurationSHA256: UniverSRResources.sourceConfigurationPin.sha256,
                modelCardSHA256: UniverSRResources.modelCardPin.sha256,
                computeType: dtype.lowercased()
            ),
            inference: .init(
                odeMethod: method.rawValue,
                odeSteps: odeSteps,
                guidanceScale: guidanceScale,
                seed: seed,
                chunkSeconds: chunkSeconds,
                chunks: result.chunkCount,
                elapsedSeconds: result.elapsedSeconds
            ),
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
        let modelID: ModelResolver.ModelID = model == ModelResolver.ModelID.univerSRAudio.rawValue
            ? .univerSRAudio
            : .apBWE16kTo48k
        return try ModelResolver().resolve(modelID).rootURL
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
