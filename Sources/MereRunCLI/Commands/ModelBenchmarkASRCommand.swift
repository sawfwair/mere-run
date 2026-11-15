import ArgumentParser
import AudioCore
import AudioCodecs
import AudioSTT
import Foundation
import MereRunCore
import MLX
#if canImport(Darwin)
import Darwin
#endif

struct ModelBenchmarkASR: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "asr", abstract: "Compare prepared ASR latency on local recordings without reloading between runs.")

    enum Backend: String, ExpressibleByArgument, Sendable { case parakeet, qwen, whistle, nemotron, sov }
    @Argument(help: "One or more input recordings, processed in order.") var audio: [String]
    @Option(name: .long, help: "ASR family: parakeet, qwen, whistle, nemotron, or sov.") var backend: Backend
    @Option(name: .long, help: "Installed checkpoint directory; benchmarking never downloads models.") var modelRoot: String
    @Option(name: .long, help: "Core ML artifact for Parakeet or SOV.") var artifact: String?
    @Option(name: .long, help: "Optional SOV DFlash checkpoint.") var draftModel: String?
    @Flag(name: .long, help: "Experimental M5 dynamic-int8 SOV FFNs.") var experimentalDynamicInt8 = false
    @Option(name: .long, help: "SOV expert policy: bf16, bf16-gemv, q8, q8-g32, or q8-metal. Q8 changes target weights.")
    var experimentalDecoder: ModelBenchmarkSOV.DecoderExperts = .bf16
    @Option(name: .long, help: "Resident warmups per recording.") var warmups: Int = 2
    @Option(name: .long, help: "Measured repetitions per recording.") var repetitions: Int = 5
    @Option(name: .long, help: "Autoregressive output token budget.") var maxTokens: Int = 448
    @Option(name: .long, help: "Optional language hint, shared by all recordings.") var language: String?
    @Option(name: .long, help: "Whistle beam size, 1...8.") var beamSize: Int = 5
    @Option(name: .long, help: "Whistle decoder depth, 2...8.") var decoderDepth: Int = 8
    @Flag(name: .long, help: "Include Whistle word alignment work.") var wordTimestamps = false
    @Flag(name: .long, help: "Emit JSON with every measured transcript and latency.") var json = false

    func validate() throws {
        guard !audio.isEmpty, warmups >= 0, repetitions > 0, maxTokens > 0 else {
            throw ValidationError("Provide audio; warmups must be nonnegative and repetitions/max-tokens positive.")
        }
        guard artifact == nil || [.parakeet, .sov].contains(backend) else {
            throw ValidationError("--artifact requires Parakeet or SOV.")
        }
        guard backend == .sov || (draftModel == nil && !experimentalDynamicInt8 && experimentalDecoder == .bf16) else {
            throw ValidationError("DFlash and experimental encoder/decoder policies require SOV.")
        }
        guard !experimentalDynamicInt8 || artifact == nil else {
            throw ValidationError("Experimental int8 requires the MLX provider.")
        }
        guard backend != .sov || language == nil else {
            throw ValidationError("The SOV measured API uses automatic language detection; omit --language.")
        }
        guard backend == .whistle || (beamSize == 5 && decoderDepth == 8 && !wordTimestamps) else {
            throw ValidationError("Beam size, decoder depth, and word alignment overrides require Whistle.")
        }
        try WhistleOptions(beamSize: beamSize, decoderDepth: decoderDepth).validate()
    }

    func run() async throws {
        #if DEBUG
        throw ValidationError("ASR measurements require an optimized release executable.")
        #else
        for path in [modelRoot, artifact, draftModel].compactMap({ $0 }) {
            var directory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &directory), directory.boolValue else {
                throw ValidationError("Installed checkpoint/artifact directory not found: \(path)")
            }
        }
        for path in audio {
            var directory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &directory), !directory.boolValue else {
                throw ValidationError("Audio file not found: \(path)")
            }
        }
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let provider = artifact.map { ParakeetExecutionProvider.coreML(artifactURL: URL(fileURLWithPath: $0)) } ?? .mlx
        let session: PreparedASRBenchmark
        switch backend {
        case .parakeet: session = .parakeet(ParakeetGenerator(executionProvider: provider))
        case .qwen: session = .qwen(Qwen3ASRGenerator())
        case .whistle: session = .whistle(WhistleGenerator(), WhistleOptions(
            beamSize: beamSize, decoderDepth: decoderDepth, wordTimestamps: wordTimestamps))
        case .nemotron: session = .nemotron(NemotronTranscribeGenerator())
        case .sov: session = .sov(SOVTranscribeGenerator(executionProvider: provider,
            experimentalDynamicInt8: experimentalDynamicInt8, decoderExperts: experimentalDecoder.policy))
        }
        let loadStarted = ProcessInfo.processInfo.systemUptime
        try await session.prepare(root: modelRoot, draft: draftModel)
        let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStarted
        let preparedActiveBytes = Memory.activeMemory
        var recordings: [ASRBenchmarkRecording] = []
        for path in audio {
            let readStarted = ProcessInfo.processInfo.systemUptime
            let samples = try AudioReader.readAudio(from: URL(fileURLWithPath: path))
            let readSeconds = ProcessInfo.processInfo.systemUptime - readStarted
            var first: ASRBenchmarkRun?
            var measurements: [ASRBenchmarkRun] = []
            for index in 0..<(warmups + repetitions) {
                CLIStderr.write("ASR \(backend.rawValue) \(URL(fileURLWithPath: path).lastPathComponent) \(index + 1)/\(warmups + repetitions)\n")
                Memory.peakMemory = 0
                let started = ProcessInfo.processInfo.systemUptime
                let (result, parakeet, sov) = try await session.transcribe(samples, language: language, maxTokens: maxTokens)
                let measured = ASRBenchmarkRun(seconds: ProcessInfo.processInfo.systemUptime - started,
                    text: result.text, mlxPeakActiveBytes: Memory.peakMemory, parakeet: parakeet, sov: sov)
                if first == nil { first = measured }
                if index >= warmups { measurements.append(measured) }
            }
            recordings.append(ASRBenchmarkRecording(audio: path, audioSeconds: Double(samples.count) / 16_000,
                audioReadSeconds: readSeconds, firstInference: first!, measurements: measurements))
        }
        await session.unload()
        let report = ASRBenchmarkReport(schemaVersion: 1, backend: backend.rawValue, provider: artifact == nil ? "mlx" : "coreml",
            modelRoot: modelRoot, artifact: artifact, draftModel: draftModel, experimentalDynamicInt8: experimentalDynamicInt8,
            decoderExperts: backend == .sov ? experimentalDecoder.rawValue : nil, preparedActiveBytes: preparedActiveBytes,
            warmups: warmups, repetitions: repetitions, maxTokens: maxTokens, language: language,
            beamSize: backend == .whistle ? beamSize : nil, decoderDepth: backend == .whistle ? decoderDepth : nil,
            wordTimestamps: wordTimestamps, modelLoadSeconds: loadSeconds, processor: MereRunMachineProfile.current.processorName,
            mlxDevice: Device.defaultDevice().deviceType == .gpu ? "gpu" : "cpu",
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            peakProcessResidentBytes: Self.peakResidentBytes(), recordings: recordings)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if json { print(String(decoding: try encoder.encode(report), as: UTF8.self)) }
        else {
            for recording in recordings {
                print("\(backend.rawValue) \(recording.audio): \(recording.measurements.map(\.seconds)) seconds")
                print(recording.measurements.first?.text ?? "")
            }
        }
        #endif
    }

    private static func peakResidentBytes() -> UInt64? {
        #if canImport(Darwin)
        var usage = rusage()
        return getrusage(RUSAGE_SELF, &usage) == 0 ? UInt64(usage.ru_maxrss) : nil
        #else
        return nil
        #endif
    }
}

private struct ASRBenchmarkRun: Encodable {
    let seconds: Double
    let text: String
    let mlxPeakActiveBytes: Int
    let parakeet: ParakeetPipelineTimings?
    let sov: SOVMeasuredTranscription?
}

private struct ASRBenchmarkRecording: Encodable {
    let audio: String
    let audioSeconds: Double
    let audioReadSeconds: Double
    let firstInference: ASRBenchmarkRun
    let measurements: [ASRBenchmarkRun]
}

private struct ASRBenchmarkReport: Encodable {
    let schemaVersion: Int
    let backend: String
    let provider: String
    let modelRoot: String
    let artifact: String?
    let draftModel: String?
    let experimentalDynamicInt8: Bool
    let decoderExperts: String?
    let preparedActiveBytes: Int
    let warmups: Int
    let repetitions: Int
    let maxTokens: Int
    let language: String?
    let beamSize: Int?
    let decoderDepth: Int?
    let wordTimestamps: Bool
    let modelLoadSeconds: Double
    let processor: String
    let mlxDevice: String
    let physicalMemoryBytes: UInt64
    let operatingSystem: String
    let peakProcessResidentBytes: UInt64?
    let recordings: [ASRBenchmarkRecording]
}

private enum PreparedASRBenchmark: Sendable {
    case parakeet(ParakeetGenerator)
    case qwen(Qwen3ASRGenerator)
    case whistle(WhistleGenerator, WhistleOptions)
    case nemotron(NemotronTranscribeGenerator)
    case sov(SOVTranscribeGenerator)

    func prepare(root: String, draft: String?) async throws {
        switch self {
        case .parakeet(let generator): try await generator.prepare(modelPath: root)
        case .qwen(let generator): try await generator.prepare(modelPath: root)
        case .whistle(let generator, let options): try await generator.prepare(modelPath: root, options: options)
        case .nemotron(let generator): try await generator.prepare(modelPath: root)
        case .sov(let generator): try await generator.prepare(modelPath: root, draftModel: draft)
        }
    }

    func transcribe(_ samples: [Float], language: String?, maxTokens: Int) async throws
        -> (ASRResult, ParakeetPipelineTimings?, SOVMeasuredTranscription?) {
        switch self {
        case .parakeet(let generator):
            let measured = try await generator.transcribePreparedMeasured(samples: samples, language: language)
            return (measured.result, measured.timings, nil)
        case .qwen(let generator):
            return (try await generator.transcribePrepared(samples: samples, language: language, maxTokens: maxTokens), nil, nil)
        case .whistle(let generator, _):
            return (try await generator.transcribePrepared(samples: samples, language: language, maxTokens: maxTokens), nil, nil)
        case .nemotron(let generator):
            return (try await generator.transcribePrepared(samples: samples, language: language, maxTokens: maxTokens), nil, nil)
        case .sov(let generator):
            let measured = try await generator.transcribePreparedMeasured(samples: samples, maxTokens: maxTokens)
            return (ASRResult(text: measured.text, duration: Double(samples.count) / 16_000), nil, measured)
        }
    }

    func unload() async {
        switch self {
        case .parakeet(let generator): await generator.unload()
        case .qwen(let generator): await generator.unload()
        case .whistle(let generator, _): await generator.unload()
        case .nemotron(let generator): await generator.unload()
        case .sov(let generator): await generator.unload()
        }
    }
}
