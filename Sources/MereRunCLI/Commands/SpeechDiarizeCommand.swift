import ArgumentParser
import AudioCodecs
import AudioSTT
import Foundation
import MereRunCore

enum SpeechDiarizationOutputFormat: String, CaseIterable, ExpressibleByArgument {
    case json
    case rttm
}

enum Nemotron3DiarizationLatency: String, CaseIterable, ExpressibleByArgument {
    case offline
    case standard = "1.04"
    case low = "0.64"
    case ultraLow = "0.32"

    var configuration: (chunk: Int, right: Int, fifo: Int, update: Int) {
        switch self {
        case .offline: (340, 40, 40, 300)
        case .standard: (9, 4, 264, 222)
        case .low: (6, 2, 264, 222)
        case .ultraLow: (3, 1, 264, 222)
        }
    }
}

struct SpeechDiarizationSegmentPayload: Codable, Equatable {
    let speaker: String
    let speakerIndex: Int
    let startSeconds: Double
    let endSeconds: Double
    let durationSeconds: Double

    enum CodingKeys: String, CodingKey {
        case speaker
        case speakerIndex = "speaker_index"
        case startSeconds = "start_seconds"
        case endSeconds = "end_seconds"
        case durationSeconds = "duration_seconds"
    }
}

struct SpeechDiarizationPayload: Codable, Equatable {
    let schemaVersion: Int
    let model: String
    let source: String
    let runtime: String
    let device: String
    let durationSeconds: Double
    let speakerCount: Int
    let processingSeconds: Double
    let segments: [SpeechDiarizationSegmentPayload]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case model
        case source
        case runtime
        case device
        case durationSeconds = "duration_seconds"
        case speakerCount = "speaker_count"
        case processingSeconds = "processing_seconds"
        case segments
    }

    static func make(
        result: DiarizationOutput,
        model: String,
        source: String,
        durationSeconds: Double
    ) -> Self {
        Self(
            schemaVersion: 1,
            model: model,
            source: source,
            runtime: NativeMLXRuntime.backendDescription,
            device: NativeMLXRuntime.defaultDeviceType,
            durationSeconds: durationSeconds,
            speakerCount: result.numSpeakers,
            processingSeconds: result.totalTime,
            segments: result.segments.map { segment in
                SpeechDiarizationSegmentPayload(
                    speaker: "speaker_\(segment.speaker)",
                    speakerIndex: segment.speaker,
                    startSeconds: Double(segment.start),
                    endSeconds: Double(segment.end),
                    durationSeconds: Double(segment.end - segment.start)
                )
            }
        )
    }
}

struct SpeechDiarize: AsyncParsableCommand {
    static let defaultManagedModelID = ModelResolver.ModelID.sortformerDiarization

    static let configuration = CommandConfiguration(
        commandName: "diarize",
        abstract: "Identify who spoke when in an audio file with native MLX diarization."
    )

    @Argument(help: "Audio file to diarize.")
    var audio: String

    @Option(
        name: [.customShort("m"), .long],
        help: "Canonical Sortformer or Nemotron 3 model id, or a local model directory."
    )
    var model: String = Self.defaultManagedModelID.rawValue

    @Option(name: [.customShort("f"), .long], help: "Output format: json or rttm.")
    var format: SpeechDiarizationOutputFormat = .json

    @Option(name: [.customShort("o"), .long], help: "Optional output file path.")
    var output: String?

    @Option(name: [.long], help: "Speaker activity threshold from 0 through 1.")
    var threshold: Float = 0.5

    @Option(name: [.customLong("min-duration")], help: "Discard speaker segments shorter than this many seconds.")
    var minDuration: Float = 0.25

    @Option(name: [.customLong("merge-gap")], help: "Merge same-speaker segments separated by at most this many seconds.")
    var mergeGap: Float = 0.25

    @Option(name: [.long], help: "Nemotron 3 input-buffer latency: offline, 1.04, 0.64, or 0.32 seconds.")
    var latency: Nemotron3DiarizationLatency = .offline

    @Flag(name: [.short, .long], help: "Suppress diagnostic progress output.")
    var quiet = false

    func validate() throws {
        guard (0...1).contains(threshold) else {
            throw ValidationError("--threshold must be between 0 and 1.")
        }
        guard minDuration.isFinite, minDuration >= 0 else {
            throw ValidationError("--min-duration must be greater than or equal to 0.")
        }
        guard mergeGap.isFinite, mergeGap >= 0 else {
            throw ValidationError("--merge-gap must be greater than or equal to 0.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let audioURL = URL(fileURLWithPath: audio).standardizedFileURL
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("Audio file not found: \(audioURL.path)")
        }

        // The capability gate refused a Sortformer run with a streaming `--latency` before this.
        let modelRoot = try Self.resolveModelRoot(model)
        let useNemotron = Nemotron3DiarizationResources.isNemotron3(model: model, root: modelRoot)
        if !quiet {
            CLIStderr.write("Loading \(useNemotron ? "Nemotron 3 Diarization" : "Sortformer") from \(modelRoot.path)\n")
            CLIStderr.write("[runtime] diarization backend: \(NativeMLXRuntime.backendDescription)\n")
        }
        let audioBuffer = try AudioReader.readAudioBuffer(
            from: audioURL,
            sampleRate: 16_000,
            channels: 1
        )
        let result: DiarizationOutput
        if useNemotron {
            let diarizer = try Nemotron3Diarizer(modelDirectory: modelRoot)
            let selected = latency.configuration
            result = try diarizer.diarize(
                samples: audioBuffer.samples,
                sampleRate: audioBuffer.sampleRate,
                threshold: threshold,
                minDuration: minDuration,
                mergeGap: mergeGap,
                chunkLength: selected.chunk,
                rightContext: selected.right,
                fifoLength: selected.fifo,
                cacheUpdatePeriod: selected.update
            )
        } else {
            let diarizer = try SortformerDiarizer(modelDirectory: modelRoot)
            result = try diarizer.diarize(
                samples: audioBuffer.samples,
                sampleRate: audioBuffer.sampleRate,
                threshold: threshold,
                minDuration: minDuration,
                mergeGap: mergeGap
            )
        }
        let durationSeconds = Double(audioBuffer.samples.count) / Double(audioBuffer.sampleRate)

        if let output {
            let rendered = try render(result, sourceURL: audioURL, durationSeconds: durationSeconds, warnings: [])
            let outputURL = URL(fileURLWithPath: output).standardizedFileURL
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try rendered.write(to: outputURL, atomically: true, encoding: .utf8)
            if !quiet {
                CLIStderr.write("Diarization saved to: \(outputURL.path)\n")
            }
        }

        if !quiet {
            CLIStderr.write(
                "Detected \(result.numSpeakers) speaker(s) across \(result.segments.count) segment(s).\n"
            )
        }
        // Only stdout carries the gate's warnings; the file keeps the diarization alone.
        print(try render(result, sourceURL: audioURL, durationSeconds: durationSeconds, warnings: CLIGateWarnings.current))
    }

    static func resolveModelRoot(
        _ rawModel: String,
        fileManager: FileManager = .default
    ) throws -> URL {
        let localURL = URL(fileURLWithPath: rawModel).standardizedFileURL
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: localURL.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return localURL
        }

        guard let modelID = ModelResolver.ModelID(rawValue: rawModel),
              modelID == defaultManagedModelID || modelID == .nemotron3Diarization else {
            throw ValidationError(
                "Unsupported diarization model '\(rawModel)'. Use a managed diarization model or a local model directory."
            )
        }
        return try ModelResolver(fileManager: fileManager).resolve(modelID).rootURL
    }

    private func render(
        _ result: DiarizationOutput,
        sourceURL: URL,
        durationSeconds: Double,
        warnings: [String]
    ) throws -> String {
        switch format {
        case .rttm:
            let fileID = sourceURL.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: " ", with: "_")
            return result.rttm(fileID: fileID)
        case .json:
            let payload = SpeechDiarizationPayload.make(
                result: result,
                model: model,
                source: sourceURL.lastPathComponent,
                durationSeconds: durationSeconds
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            return String(decoding: try encoder.encode(GateWarned(payload, warnings: warnings)), as: UTF8.self)
        }
    }
}
