import ArgumentParser
import AudioCodecs
import AudioSTT
import Foundation
import MereRunCore

struct ModelBenchmarkParakeetCoreML: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "parakeet-coreml",
        abstract: "Benchmark the prepared Parakeet Core ML pipeline in one resident process.",
        discussion: """
        Loads and verifies one Mere-built Parakeet artifact, decodes the audio once,
        performs unmeasured warmups, then records stage-level timings without
        restarting the executable or reloading the model.

        Run this benchmark from an optimized build:
          swift run -c release mere.run model benchmark parakeet-coreml audio.wav --artifact /path/to/parakeet-coreml
        """
    )

    @Argument(help: "Input audio file.")
    var audio: String

    @Option(
        name: [.long],
        help: "Mere-built Parakeet Core ML artifact directory."
    )
    var artifact: String

    @Option(name: [.long], help: "Unmeasured resident warmup repetitions.")
    var warmups: Int = 2

    @Option(name: [.long], help: "Measured resident repetitions.")
    var repetitions: Int = 5

    @Option(name: [.long], help: "Optional language label copied to the transcript result.")
    var language: String?

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        guard warmups >= 0 else {
            throw ValidationError("--warmups must be zero or greater.")
        }
        guard repetitions > 0 else {
            throw ValidationError("--repetitions must be greater than zero.")
        }
    }

    func run() async throws {
        #if DEBUG
        throw ValidationError(
            "Parakeet performance measurements require an optimized executable; "
                + "run `swift run -c release mere.run model benchmark parakeet-coreml ...`."
        )
        #else
        try MLXBundleSupport.ensureAvailable(quiet: json)

        let audioURL = try existingFile(audio, label: "Audio file")
        let artifactURL = try existingDirectory(artifact, label: "Core ML artifact directory")

        let audioReadStarted = ProcessInfo.processInfo.systemUptime
        let samples = try AudioReader.readAudio(from: audioURL)
        let audioReadSeconds = ProcessInfo.processInfo.systemUptime - audioReadStarted
        let audioDurationSeconds = TimeInterval(samples.count) / 16_000

        let generator = ParakeetGenerator(
            executionProvider: .coreML(artifactURL: artifactURL)
        )
        let modelLoadStarted = ProcessInfo.processInfo.systemUptime
        try await generator.prepare(modelPath: artifactURL.path)
        let modelLoadSeconds = ProcessInfo.processInfo.systemUptime - modelLoadStarted

        for index in 0..<warmups {
            if !json {
                CLIStderr.write("Warmup \(index + 1)/\(warmups)\n")
            }
            _ = try await generator.transcribePreparedMeasured(
                samples: samples,
                language: language
            )
        }

        var measured: [ParakeetCoreMLBenchmarkSample] = []
        measured.reserveCapacity(repetitions)
        var referenceTranscript: String?
        for index in 0..<repetitions {
            if !json {
                CLIStderr.write("Measured run \(index + 1)/\(repetitions)\n")
            }
            let transcription = try await generator.transcribePreparedMeasured(
                samples: samples,
                language: language
            )
            if referenceTranscript == nil {
                referenceTranscript = transcription.result.text
            }
            measured.append(
                ParakeetCoreMLBenchmarkSample(
                    repetition: index + 1,
                    timings: transcription.timings,
                    realtimeFactor: audioDurationSeconds / transcription.timings.totalSeconds,
                    transcriptMatchesReference: transcription.result.text == referenceTranscript
                )
            )
        }

        let report = ParakeetCoreMLBenchmarkReport(
            schemaVersion: 1,
            provider: "coreml",
            artifact: artifactURL.path,
            audio: audioURL.path,
            audioDurationSeconds: audioDurationSeconds,
            audioReadSeconds: audioReadSeconds,
            modelLoadSeconds: modelLoadSeconds,
            warmups: warmups,
            repetitions: repetitions,
            hardware: .current,
            transcript: referenceTranscript ?? "",
            transcriptConsistent: measured.allSatisfy(\.transcriptMatchesReference),
            samples: measured,
            summary: ParakeetCoreMLBenchmarkSummary(samples: measured)
        )

        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
        #endif
    }

    private func existingFile(_ path: String, label: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw ValidationError("\(label) not found: \(url.path)")
        }
        return url
    }

    private func existingDirectory(_ path: String, label: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ValidationError("\(label) not found: \(url.path)")
        }
        return url
    }
}

private struct ParakeetCoreMLBenchmarkReport: Encodable {
    let schemaVersion: Int
    let provider: String
    let artifact: String
    let audio: String
    let audioDurationSeconds: TimeInterval
    let audioReadSeconds: TimeInterval
    let modelLoadSeconds: TimeInterval
    let warmups: Int
    let repetitions: Int
    let hardware: ParakeetCoreMLBenchmarkHardware
    let transcript: String
    let transcriptConsistent: Bool
    let samples: [ParakeetCoreMLBenchmarkSample]
    let summary: ParakeetCoreMLBenchmarkSummary

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case provider
        case artifact
        case audio
        case audioDurationSeconds = "audio_duration_seconds"
        case audioReadSeconds = "audio_read_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case warmups
        case repetitions
        case hardware
        case transcript
        case transcriptConsistent = "transcript_consistent"
        case samples
        case summary
    }

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Parakeet Core ML resident benchmark",
            "provider: \(provider)",
            "hardware: \(hardware.processor) (\(hardware.unifiedMemoryGB) GiB)",
            "audio_seconds: \(Self.format(audioDurationSeconds))",
            "audio_read_seconds: \(Self.format(audioReadSeconds))",
            "model_load_seconds: \(Self.format(modelLoadSeconds))",
            "warmups: \(warmups)",
            "repetitions: \(repetitions)",
        ]
        for sample in samples {
            lines.append(
                "run_\(sample.repetition): total=\(Self.format(sample.timings.totalSeconds))s "
                    + "frontend=\(Self.format(sample.timings.featureExtractionSeconds))s "
                    + "encoder=\(Self.format(sample.timings.encoderSeconds))s "
                    + "decoder=\(Self.format(sample.timings.decoderSeconds))s "
                    + "alignment=\(Self.format(sample.timings.alignmentSeconds))s "
                    + "merge=\(Self.format(sample.timings.windowMergeSeconds))s "
                    + "rtf=\(Self.format(sample.realtimeFactor))x"
            )
        }
        lines += [
            "median_total_seconds: \(Self.format(summary.medianTotalSeconds))",
            "median_realtime_factor: \(Self.format(summary.medianRealtimeFactor))x",
            "transcript_consistent: \(transcriptConsistent)",
            "transcript: \(transcript)",
        ]
        return lines.joined(separator: "\n")
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.6f", value)
    }
}

private struct ParakeetCoreMLBenchmarkSample: Encodable {
    let repetition: Int
    let timings: ParakeetPipelineTimings
    let realtimeFactor: Double
    let transcriptMatchesReference: Bool

    enum CodingKeys: String, CodingKey {
        case repetition
        case timings
        case realtimeFactor = "realtime_factor"
        case transcriptMatchesReference = "transcript_matches_reference"
    }
}

private struct ParakeetCoreMLBenchmarkSummary: Encodable {
    let medianTotalSeconds: TimeInterval
    let medianRealtimeFactor: Double
    let medianFeatureExtractionSeconds: TimeInterval
    let medianEncoderSeconds: TimeInterval
    let medianDecoderSeconds: TimeInterval
    let medianAlignmentSeconds: TimeInterval
    let medianWindowMergeSeconds: TimeInterval

    init(samples: [ParakeetCoreMLBenchmarkSample]) {
        self.medianTotalSeconds = Self.median(samples.map(\.timings.totalSeconds))
        self.medianRealtimeFactor = Self.median(samples.map(\.realtimeFactor))
        self.medianFeatureExtractionSeconds = Self.median(samples.map(\.timings.featureExtractionSeconds))
        self.medianEncoderSeconds = Self.median(samples.map(\.timings.encoderSeconds))
        self.medianDecoderSeconds = Self.median(samples.map(\.timings.decoderSeconds))
        self.medianAlignmentSeconds = Self.median(samples.map(\.timings.alignmentSeconds))
        self.medianWindowMergeSeconds = Self.median(samples.map(\.timings.windowMergeSeconds))
    }

    enum CodingKeys: String, CodingKey {
        case medianTotalSeconds = "median_total_seconds"
        case medianRealtimeFactor = "median_realtime_factor"
        case medianFeatureExtractionSeconds = "median_feature_extraction_seconds"
        case medianEncoderSeconds = "median_encoder_seconds"
        case medianDecoderSeconds = "median_decoder_seconds"
        case medianAlignmentSeconds = "median_alignment_seconds"
        case medianWindowMergeSeconds = "median_window_merge_seconds"
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }
}

private struct ParakeetCoreMLBenchmarkHardware: Encodable {
    let processor: String
    let unifiedMemoryGB: Int

    static var current: ParakeetCoreMLBenchmarkHardware {
        let profile = MereRunMachineProfile.current
        return ParakeetCoreMLBenchmarkHardware(
            processor: profile.processorName,
            unifiedMemoryGB: profile.unifiedMemoryGB
        )
    }

    enum CodingKeys: String, CodingKey {
        case processor
        case unifiedMemoryGB = "unified_memory_gb"
    }
}
