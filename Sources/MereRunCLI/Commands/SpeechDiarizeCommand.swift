import ArgumentParser
import AudioCodecs
import AudioSTT
import Foundation
import MereRunCore

enum SpeechDiarizationOutputFormat: String, CaseIterable, ExpressibleByArgument {
    case json
    case rttm
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
}

struct SpeechDiarize: AsyncParsableCommand {
    static let defaultManagedModelID = ModelResolver.ModelID.sortformerDiarization

    static let configuration = CommandConfiguration(
        commandName: "diarize",
        abstract: "Identify who spoke when in an audio file with native MLX Sortformer."
    )

    @Argument(help: "Audio file to diarize.")
    var audio: String

    @Option(
        name: [.customShort("m"), .long],
        help: "Canonical model id (speech-diarization-sortformer) or a local model directory."
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

    @Flag(name: [.short, .long], help: "Suppress diagnostic progress output.")
    var quiet = false

    func validate() throws {
        guard (0...1).contains(threshold) else {
            throw ValidationError("--threshold must be between 0 and 1.")
        }
        guard minDuration >= 0 else {
            throw ValidationError("--min-duration must be greater than or equal to 0.")
        }
        guard mergeGap >= 0 else {
            throw ValidationError("--merge-gap must be greater than or equal to 0.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let audioURL = URL(fileURLWithPath: audio).standardizedFileURL
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("Audio file not found: \(audioURL.path)")
        }

        let modelRoot = try Self.resolveModelRoot(model)
        if !quiet {
            CLIStderr.write("Loading Sortformer from \(modelRoot.path)\n")
            CLIStderr.write("[runtime] diarization backend: \(NativeMLXRuntime.backendDescription)\n")
        }
        let audioBuffer = try AudioReader.readAudioBuffer(
            from: audioURL,
            sampleRate: 16_000,
            channels: 1
        )
        let diarizer = try SortformerDiarizer(modelDirectory: modelRoot)
        let result = try diarizer.diarize(
            samples: audioBuffer.samples,
            sampleRate: audioBuffer.sampleRate,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap
        )
        let rendered = try render(
            result,
            sourceURL: audioURL,
            durationSeconds: Double(audioBuffer.samples.count) / Double(audioBuffer.sampleRate)
        )

        if let output {
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
        print(rendered)
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

        guard let modelID = ModelResolver.ModelID(rawValue: rawModel), modelID == defaultManagedModelID else {
            throw ValidationError(
                "Unsupported diarization model '\(rawModel)'. Use \(defaultManagedModelID.rawValue) or a local model directory."
            )
        }
        return try ModelResolver(fileManager: fileManager).resolve(modelID).rootURL
    }

    private func render(
        _ result: DiarizationOutput,
        sourceURL: URL,
        durationSeconds: Double
    ) throws -> String {
        switch format {
        case .rttm:
            let fileID = sourceURL.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: " ", with: "_")
            return result.rttm(fileID: fileID)
        case .json:
            let payload = SpeechDiarizationPayload(
                schemaVersion: 1,
                model: model,
                source: sourceURL.lastPathComponent,
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
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            return String(decoding: try encoder.encode(payload), as: UTF8.self)
        }
    }
}
