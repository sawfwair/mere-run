import ArgumentParser
import AudioCodecs
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import AudioSTT
import Foundation
import MereRunCore
import MLX

struct ModelBenchmarkSOV: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sov", abstract: "Profile resident SOV encoder/projector and DFlash transcription.")

    @Argument(help: "Input audio file.") var audio: String
    @Option(name: .long, help: "Original SOV checkpoint directory.") var modelRoot: String
    @Option(name: .long, help: "Optional DFlash checkpoint directory.") var draftModel: String?
    @Option(name: .long, help: "Optional compiled Core ML encoder/projector artifact.") var artifact: String?
    @Option(name: .long, help: "Write projected embeddings as safetensors for parity analysis.") var projectedOutput: String?
    @Option(name: .long, help: "Unmeasured resident warmups.") var warmups: Int = 1
    @Option(name: .long, help: "Measured resident runs.") var repetitions: Int = 3
    @Option(name: .long, help: "Maximum output tokens.") var maxTokens: Int = 448
    @Option(name: .long, help: "Target prefill tokens per chunk.") var prefillChunkSize: Int = 1024
    @Flag(name: .long, help: "Experimental M5 GPU dynamic-int8 FFNs; transcript parity must be qualified.") var experimentalDynamicInt8 = false
    @Option(name: .long, help: "Experimental decoder experts: bf16, bf16-gemv, q4, q6, q6-matrix, mixed, q8, q8-g32, q8-metal. Quantization changes target weights.")
    var experimentalDecoder: DecoderExperts = .bf16
    @Option(name: .long, help: "Calibrated per-layer Q4/Q6/Q8 recipe for the mixed decoder.") var expertRecipe: String?
    @Option(name: .long, help: "Native packed target directory; stream and save once, then reuse. Requires quantized experts.") var packedWeights: String?
    @Flag(name: .long, help: "Pair Q4/Q6/mixed experts with a separately resident Q8 Metal baseline.") var interleaveExperts = false
    @Flag(name: .long, help: "Start expert comparisons with the candidate; reverse order on alternate cycles.") var expertsFirst = false
    @Flag(name: .long, help: "Request bounded Metal residency for all prepared MLX models during sampling.") var wireModels = false
    @Flag(name: .long, help: "Pair default and explicitly wired residency on one model.") var interleaveWired = false
    @Flag(name: .long, help: "Start residency comparisons wired; reverse order on alternate cycles.") var wiredFirst = false
    @Flag(name: .long, help: "Interleave stock and Metal on one resident Q8 target; warmups/repetitions apply per mode.")
    var interleaveMetal = false
    @Option(name: .long, help: "Paired Metal baseline: mlx, reuse, byte, or layout with the original SIMD reduction.")
    var metalBaseline: MetalBaseline = .mlx
    @Flag(name: .long, help: "Experimental exact-BF16 expert projection and activation fusion. Requires Q8 Metal or a paired Metal comparison.")
    var experimentalExpertFusion = false
    @Flag(name: .long, help: "Force distributed expert reduction (automatic on qualified M5 Max GPUs). Requires Q8 Metal or a paired Metal comparison.")
    var experimentalExpertReduction = false
    @Flag(name: .long, help: "Start interleaved cycles with Metal; reverse order on alternate cycles.") var metalFirst = false
    @Flag(name: .long, help: "Experimental shared attention masks and direct partial RoPE; parameters unchanged.")
    var experimentalAttention = false
    @Flag(name: .long, help: "Interleave reference and optimized attention on one resident model; warmups/repetitions apply per mode.")
    var interleaveAttention = false
    @Flag(name: .long, help: "Start interleaved attention cycles with the optimized path.") var attentionFirst = false
    @Flag(name: .long, help: "Experimental fused BF16 head normalization and RoPE on the MLX GPU path.") var experimentalHeadNorm = false
    @Flag(name: .long, help: "Interleave reference and fused head normalization on one resident model.") var interleaveHeadNorm = false
    @Flag(name: .long, help: "Start interleaved head-normalization cycles with the fused path.") var headNormFirst = false
    @Option(name: .long, help: "Generation policy: reference, pipeline, draft-q8, shared-cache, retained-allocator, combined, rollback, compiled, block-4, block-8, block-12, adaptive, adaptive-after-first, vocabulary-32k.")
    var generationExperiment: GenerationExperiment = .reference
    @Option(name: .long, help: "Control policy for paired generation experiments: reference or pipeline.")
    var generationBaseline: GenerationExperiment = .reference
    @Option(name: .long, help: "JSON array of global token IDs for vocabulary-32k drafting; calibration occurs outside timing.")
    var draftVocabulary: String?
    @Flag(name: .long, help: "Pair the selected generation experiment with reference scheduling on the same model.") var interleaveGeneration = false
    @Flag(name: .long, help: "Start paired generation cycles with the experiment.") var generationFirst = false
    @Option(name: .long, help: "Experimental final-window audio token margin, 0...375; omitted retains upstream padding. Can change recognition.")
    var audioTokenMargin: Int?
    @Flag(name: .long, help: "Pair full padding and the selected audio margin on one resident model.") var interleaveAudioTokens = false
    @Flag(name: .long, help: "Start audio-token comparisons with trimming; reverse order on alternate cycles.") var audioTokensFirst = false
    @Flag(name: .long, help: "Record per-round host timings and signposts; diagnostic runs are separate from latency benchmarks.")
    var profileDecode = false
    @Option(name: .long, help: "Capture one measured decode round as a .gputrace; launch with MTL_CAPTURE_ENABLED=1.")
    var metalCapture: String?
    @Option(name: .long, help: "One-based decode round to capture after warmup.") var captureRound: Int = 3
    @Flag(name: .long, help: "Emit JSON.") var json = false

    func validate() throws {
        guard captureRound > 0 else { throw ValidationError("--capture-round must be positive.") }
        if profileDecode || metalCapture != nil {
            guard draftModel != nil, projectedOutput == nil, !interleaveMetal, !interleaveAttention, !interleaveHeadNorm,
                  !interleaveGeneration, !interleaveExperts, !interleaveWired, !interleaveAudioTokens else {
                throw ValidationError("Decode profiling requires --draft-model, no interleaving, and no projected output.")
            }
        }
        if let metalCapture {
            guard URL(fileURLWithPath: metalCapture).pathExtension == "gputrace",
                  ProcessInfo.processInfo.environment["MTL_CAPTURE_ENABLED"] == "1" else {
                throw ValidationError("--metal-capture requires a .gputrace path and MTL_CAPTURE_ENABLED=1 at launch.")
            }
        }
        guard audioTokenMargin.map({ (0...375).contains($0) }) ?? true else {
            throw ValidationError("--audio-token-margin must be between 0 and 375.")
        }
        guard !interleaveAudioTokens || (audioTokenMargin != nil && projectedOutput == nil && !interleaveMetal
            && !interleaveAttention && !interleaveHeadNorm && !interleaveGeneration && !interleaveExperts && !interleaveWired) else {
            throw ValidationError("--interleave-audio-tokens requires --audio-token-margin, no other interleaving, and no projected output.")
        }
        guard !audioTokensFirst || interleaveAudioTokens else { throw ValidationError("--audio-tokens-first requires --interleave-audio-tokens.") }
        guard packedWeights == nil || ![.bf16, .bf16GEMV].contains(experimentalDecoder) else {
            throw ValidationError("--packed-weights requires quantized --experimental-decoder experts.")
        }
        guard (experimentalDecoder == .mixed) == (expertRecipe != nil) else {
            throw ValidationError("--experimental-decoder mixed requires --expert-recipe; other decoder policies do not accept it.")
        }
        guard !interleaveExperts || ([.q4, .q6, .q6Matrix, .mixed].contains(experimentalDecoder) && artifact == nil && projectedOutput == nil
            && !interleaveMetal && !interleaveAttention && !interleaveHeadNorm && !interleaveGeneration && !generationExperiment.policy.retainsAllocator) else {
            throw ValidationError("--interleave-experts requires Q4/Q6/mixed, the MLX provider, no other interleaving or projected output, and no allocator retention.")
        }
        guard !expertsFirst || interleaveExperts else { throw ValidationError("--experts-first requires --interleave-experts.") }
        guard !wireModels || artifact == nil else { throw ValidationError("--wire-models requires the MLX provider.") }
        guard !interleaveWired || (artifact == nil && !wireModels && !interleaveMetal && !interleaveAttention
            && !interleaveHeadNorm && !interleaveGeneration && !interleaveExperts && projectedOutput == nil && !generationExperiment.policy.retainsAllocator) else {
            throw ValidationError("--interleave-wired requires the MLX provider, no other interleaving, projected output, wired override, or allocator retention.")
        }
        guard !wiredFirst || interleaveWired else { throw ValidationError("--wired-first requires --interleave-wired.") }
        guard draftVocabulary == nil || generationExperiment == .vocabulary32k else {
            throw ValidationError("--draft-vocabulary requires --generation-experiment vocabulary-32k.")
        }
        guard generationBaseline == .reference || generationBaseline == .pipeline else {
            throw ValidationError("--generation-baseline must be reference or pipeline.")
        }
        guard !interleaveGeneration || generationBaseline != generationExperiment else {
            throw ValidationError("The generation experiment and baseline must differ.")
        }
        guard !interleaveGeneration || (generationExperiment != .reference && !generationExperiment.policy.retainsAllocator
            && !interleaveMetal && !interleaveAttention && !interleaveHeadNorm && projectedOutput == nil) else {
            throw ValidationError("--interleave-generation requires an experiment without allocator retention, no other interleaving, and no --projected-output.")
        }
        guard !generationFirst || interleaveGeneration else { throw ValidationError("--generation-first requires --interleave-generation.") }
        guard !generationExperiment.policy.quantizedDraftHead || draftModel != nil else {
            throw ValidationError("The Q8 draft-head experiment requires --draft-model.")
        }
        guard ![.rollback, .compiled, .block4, .block8, .block12, .adaptive, .adaptiveAfterFirst, .vocabulary32k].contains(generationExperiment) || draftModel != nil else {
            throw ValidationError("This generation experiment requires --draft-model.")
        }
        guard !interleaveHeadNorm || (!interleaveMetal && !interleaveAttention && !experimentalHeadNorm && projectedOutput == nil) else {
            throw ValidationError("--interleave-head-norm requires no other interleaving mode, --experimental-head-norm or --projected-output.")
        }
        guard !headNormFirst || interleaveHeadNorm else { throw ValidationError("--head-norm-first requires --interleave-head-norm.") }
        guard !interleaveAttention || (!interleaveMetal && !experimentalAttention && projectedOutput == nil) else {
            throw ValidationError("--interleave-attention requires no --interleave-metal, --experimental-attention or --projected-output.")
        }
        guard !attentionFirst || interleaveAttention else { throw ValidationError("--attention-first requires --interleave-attention.") }
        guard !interleaveMetal || (experimentalDecoder == .q8 && projectedOutput == nil) else {
            throw ValidationError("--interleave-metal requires --experimental-decoder q8 and no --projected-output.")
        }
        guard !metalFirst || interleaveMetal else { throw ValidationError("--metal-first requires --interleave-metal.") }
        guard metalBaseline == .mlx || interleaveMetal else { throw ValidationError("--metal-baseline reuse, byte or layout requires --interleave-metal.") }
        guard metalBaseline != .layout || experimentalExpertFusion || experimentalExpertReduction else {
            throw ValidationError("--metal-baseline layout requires --experimental-expert-fusion or --experimental-expert-reduction.")
        }
        guard !experimentalExpertFusion || (experimentalDecoder == .q8Metal || (experimentalDecoder == .q8 && interleaveMetal)) else {
            throw ValidationError("--experimental-expert-fusion requires Q8 Metal or --experimental-decoder q8 --interleave-metal.")
        }
        guard !experimentalExpertReduction || (experimentalDecoder == .q8Metal || (experimentalDecoder == .q8 && interleaveMetal)) else {
            throw ValidationError("--experimental-expert-reduction requires Q8 Metal or --experimental-decoder q8 --interleave-metal.")
        }
        guard !experimentalDynamicInt8 || artifact == nil else {
            throw ValidationError("--experimental-dynamic-int8 requires the MLX provider; omit --artifact.")
        }
        guard warmups >= 0, repetitions > 0, maxTokens > 0, prefillChunkSize > 0 else {
            throw ValidationError("Warmups must be nonnegative; repetitions, max-tokens, and prefill-chunk-size must be positive.")
        }
    }

    func run() async throws {
        #if DEBUG
        throw ValidationError("SOV measurements require `swift run -c release mere.run model benchmark sov ...`.")
        #else
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let readStarted = ProcessInfo.processInfo.systemUptime
        let samples = try AudioReader.readAudio(from: URL(fileURLWithPath: audio))
        let readSeconds = ProcessInfo.processInfo.systemUptime - readStarted
        let recipeData = try expertRecipe.map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
        let recipe = try recipeData.map { try JSONDecoder().decode(SOVExpertRecipe.self, from: $0) }
        let vocabularyData = try draftVocabulary.map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
        let vocabulary = try vocabularyData.map { try JSONDecoder().decode([Int].self, from: $0) }
        var expertBaseline: SOVTranscribeGenerator?
        var expertBaselineLoadSeconds: Double?
        if interleaveExperts {
            let baseline = SOVTranscribeGenerator(experimentalDynamicInt8: experimentalDynamicInt8, decoderExperts: .q8Metal,
                experimentalAttention: experimentalAttention, experimentalHeadNorm: experimentalHeadNorm,
                generationExperiment: generationExperiment.policy, draftVocabulary: vocabulary)
            let baselineStarted = ProcessInfo.processInfo.systemUptime
            try await baseline.prepare(modelPath: modelRoot, draftModel: draftModel)
            try await baseline.setAudioTokenMarginForBenchmark(audioTokenMargin)
            expertBaselineLoadSeconds = ProcessInfo.processInfo.systemUptime - baselineStarted
            expertBaseline = baseline
        }
        let baselineActiveBytes = interleaveExperts ? Memory.activeMemory : 0
        let generator = SOVTranscribeGenerator(executionProvider: artifact.map { .coreML(artifactURL: URL(fileURLWithPath: $0)) } ?? .mlx,
            experimentalDynamicInt8: experimentalDynamicInt8, decoderExperts: experimentalDecoder.policy,
            experimentalAttention: experimentalAttention, experimentalHeadNorm: experimentalHeadNorm,
            generationExperiment: generationExperiment.policy, expertRecipe: recipe, draftVocabulary: vocabulary,
            packedWeights: packedWeights.map { URL(fileURLWithPath: $0) },
            experimentalExpertFusion: experimentalExpertFusion && !interleaveMetal,
            experimentalExpertReduction: experimentalExpertReduction && !interleaveMetal)
        let loadStarted = ProcessInfo.processInfo.systemUptime
        try await generator.prepare(modelPath: modelRoot, draftModel: draftModel)
        let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStarted
        if !interleaveAudioTokens { try await generator.setAudioTokenMarginForBenchmark(audioTokenMargin) }
        let loadPeakMemoryBytes = Memory.peakMemory
        let loadProcessMemory = SOVBenchmarkProcessMemory.snapshot()
        let packedMeasurement = await generator.packedWeightsMeasurement
        let preparedActiveBytes = Memory.activeMemory - baselineActiveBytes
        let dualResidentActiveBytes = interleaveExperts ? Memory.activeMemory : nil
        var measurements: [SOVMeasuredTranscription] = []
        var firstInference: SOVMeasuredTranscription?
        var measurementKernels: [String] = []
        var residentSeconds: [Double] = []
        var cacheBytes: [Int] = []
        let wiredMemoryBytes = wireModels || interleaveWired
            ? SOVBenchmarkResidency.budget(activeBytes: Memory.activeMemory, recommended: SOVBenchmarkResidency.recommendedBytes) : 0
        let outerTicket = SOVBenchmarkWiredPolicy().ticket(size: wireModels ? wiredMemoryBytes : 0)
        let comparisonTicket = SOVBenchmarkWiredPolicy().ticket(size: wiredMemoryBytes)
        func sampleRuns() async throws {
            for index in 0..<(warmups + repetitions) {
                CLIStderr.write("SOV run \(index + 1)/\(warmups + repetitions)\n")
                let interleaved = interleaveMetal || interleaveAttention || interleaveHeadNorm || interleaveGeneration || interleaveExperts || interleaveWired || interleaveAudioTokens
                let optimizedFirst = interleaveAudioTokens ? audioTokensFirst : interleaveWired ? wiredFirst : (interleaveExperts ? expertsFirst : (interleaveGeneration ? generationFirst : (interleaveHeadNorm ? headNormFirst : (interleaveAttention ? attentionFirst : metalFirst))))
                let modes = interleaved ? ((index.isMultiple(of: 2) != optimizedFirst) ? [false, true] : [true, false]) : [false]
                for metal in modes {
                    if interleaveAudioTokens { try await generator.setAudioTokenMarginForBenchmark(metal ? audioTokenMargin : nil) }
                    if interleaveMetal {
                        try await generator.setMetalVerificationForBenchmark(metal || metalBaseline != .mlx,
                            byteLoads: metal || metalBaseline == .byte || metalBaseline == .layout,
                            optimizedLayout: metal || metalBaseline == .layout,
                            fusedActivation: experimentalExpertFusion && metal,
                            distributedReduction: metal ? (experimentalExpertReduction ? true : nil) : false)
                    }
                    if interleaveAttention { try await generator.setAttentionForBenchmark(metal) }
                    if interleaveHeadNorm { try await generator.setHeadNormForBenchmark(metal) }
                    if interleaveGeneration { try await generator.setGenerationForBenchmark(metal ? generationExperiment.policy : generationBaseline.policy) }
                    try await generator.setDecodeProfileForBenchmark(index >= warmups && (profileDecode || metalCapture != nil),
                        captureURL: index == warmups ? metalCapture.map { URL(fileURLWithPath: $0) } : nil, captureRound: captureRound)
                    let selectedGenerator = interleaveExperts && !metal ? expertBaseline! : generator
                    let reductionEnabled = await selectedGenerator.usesDistributedExpertReduction
                    func measure() async throws -> (SOVMeasuredTranscription, Double) {
                        let started = ProcessInfo.processInfo.systemUptime
                        let result = try await selectedGenerator.transcribePreparedMeasured(samples: samples, maxTokens: maxTokens,
                            projectedOutput: index == warmups ? projectedOutput.map { URL(fileURLWithPath: $0) } : nil, prefillChunkSize: prefillChunkSize)
                        return (result, ProcessInfo.processInfo.systemUptime - started)
                    }
                    let (measured, elapsed) = interleaveWired && metal
                        ? try await comparisonTicket.withWiredLimit { try await measure() }
                        : try await measure()
                    if firstInference == nil { firstInference = measured }
                    if index >= warmups {
                        measurements.append(measured)
                        let attentionEnabled = interleaveAttention ? metal : experimentalAttention
                        let referenceLabel = metalBaseline == .mlx ? "q8" : "q8-metal-" + metalBaseline.rawValue
                        let expertLabel = interleaveExperts && !metal ? "q8-metal" : (interleaveMetal ? (metal ? "q8-metal" : referenceLabel) : experimentalDecoder.rawValue)
                        let headNormEnabled = interleaveHeadNorm ? metal : experimentalHeadNorm
                        let generation = interleaveGeneration && !metal ? generationBaseline.rawValue : generationExperiment.rawValue
                        let margin = interleaveAudioTokens && !metal ? nil : audioTokenMargin
                        measurementKernels.append(expertLabel + (attentionEnabled ? "+attention" : "") + (headNormEnabled ? "+heads" : "")
                            + (experimentalExpertFusion && (!interleaveMetal || metal) ? "+expert-fusion" : "")
                            + (reductionEnabled ? "+expert-reduction" : "")
                            + (generation == "reference" ? "" : "+" + generation) + ((wireModels || (interleaveWired && metal)) ? "+wired" : "") + (margin.map { "+audio-margin-\($0)" } ?? ""))
                        residentSeconds.append(elapsed)
                        cacheBytes.append(Memory.cacheMemory)
                    }
                }
            }
        }
        if wireModels { try await outerTicket.withWiredLimit { try await sampleRuns() } }
        else { try await sampleRuns() }
        await generator.unload()
        await expertBaseline?.unload()
        let inputDigest = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: audio), options: .mappedIfSafe))
            .map { String(format: "%02x", $0) }.joined()
        let report = Report(schemaVersion: 1, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory, audioSHA256: inputDigest,
                            warmups: warmups, profileDecode: profileDecode || metalCapture != nil, metalCapture: metalCapture, captureRound: captureRound, maxTokens: maxTokens, prefillChunkSize: prefillChunkSize,
                            audioTokenMargin: audioTokenMargin, interleavedAudioTokens: interleaveAudioTokens, audioTokensFirst: audioTokensFirst,
                            provider: artifact == nil ? "mlx" : "coreml", experimentalDynamicInt8: experimentalDynamicInt8,
                            decoderExperts: experimentalDecoder.rawValue, experimentalExpertFusion: experimentalExpertFusion, experimentalExpertReduction: experimentalExpertReduction,
                            interleavedExperts: interleaveExperts, expertsFirst: expertsFirst,
                            wireModels: wireModels, interleavedWired: interleaveWired, wiredFirst: wiredFirst, wiredMemoryBytes: wiredMemoryBytes,
                            expertBaselineLoadSeconds: expertBaselineLoadSeconds, dualResidentActiveBytes: dualResidentActiveBytes,
                            expertRecipeSHA256: recipeData.map { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }, preparedActiveBytes: preparedActiveBytes, audioReadSeconds: readSeconds,
                            modelLoadSeconds: loadSeconds, modelLoadPeakMemoryBytes: loadPeakMemoryBytes,
                            modelLoadProcessMemory: loadProcessMemory, packedWeights: packedMeasurement,
                            audioSeconds: Double(samples.count) / 16_000,
                            dflash: draftModel != nil, firstInference: firstInference, measurements: measurements,
                            interleaved: interleaveMetal, metalFirst: metalFirst, metalBaseline: metalBaseline.rawValue,
                            experimentalAttention: experimentalAttention, interleavedAttention: interleaveAttention, attentionFirst: attentionFirst,
                            experimentalHeadNorm: experimentalHeadNorm, interleavedHeadNorm: interleaveHeadNorm, headNormFirst: headNormFirst,
                            generationExperiment: generationExperiment.rawValue, generationBaseline: generationBaseline.rawValue,
                            draftVocabularySHA256: vocabularyData.map { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() },
                            interleavedGeneration: interleaveGeneration,
                            generationFirst: generationFirst, allocatorCacheLimitBytes: Memory.cacheLimit, peakMemoryBytes: Memory.peakMemory,
                            measurementKernels: measurementKernels, residentSeconds: residentSeconds, allocatorCacheBytes: cacheBytes)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if json { print(String(decoding: try encoder.encode(report), as: UTF8.self)) }
        else {
            print("SOV \(report.provider), DFlash=\(report.dflash), load=\(loadSeconds)s")
            for (kernel, measured) in zip(measurementKernels, measurements) {
                print("kernel=\(kernel) total=\(measured.totalSeconds)s encoder/projector=\(measured.encoderProjectorSeconds)s prefill=\(measured.prefillSeconds)s generation=\(measured.generationSeconds)s tokens=\(measured.tokens.count)")
            }
            print(measurements.first?.text ?? "")
        }
        #endif
    }

    private struct Report: Encodable {
        let schemaVersion: Int
        let operatingSystem: String
        let physicalMemoryBytes: UInt64
        let audioSHA256: String
        let warmups: Int
        let profileDecode: Bool
        let metalCapture: String?
        let captureRound: Int
        let maxTokens: Int
        let prefillChunkSize: Int
        let audioTokenMargin: Int?
        let interleavedAudioTokens: Bool
        let audioTokensFirst: Bool
        let provider: String
        let experimentalDynamicInt8: Bool
        let decoderExperts: String
        let experimentalExpertFusion: Bool
        let experimentalExpertReduction: Bool
        let interleavedExperts: Bool
        let expertsFirst: Bool
        let wireModels: Bool
        let interleavedWired: Bool
        let wiredFirst: Bool
        let wiredMemoryBytes: Int
        let expertBaselineLoadSeconds: Double?
        let dualResidentActiveBytes: Int?
        let expertRecipeSHA256: String?
        let preparedActiveBytes: Int
        let audioReadSeconds: Double
        let modelLoadSeconds: Double
        let modelLoadPeakMemoryBytes: Int
        let modelLoadProcessMemory: SOVBenchmarkProcessMemory?
        let packedWeights: SOVPackedWeightsMeasurement?
        let audioSeconds: Double
        let dflash: Bool
        let firstInference: SOVMeasuredTranscription?
        let measurements: [SOVMeasuredTranscription]
        let interleaved: Bool
        let metalFirst: Bool
        let metalBaseline: String
        let experimentalAttention: Bool
        let interleavedAttention: Bool
        let attentionFirst: Bool
        let experimentalHeadNorm: Bool
        let interleavedHeadNorm: Bool
        let headNormFirst: Bool
        let generationExperiment: String
        let generationBaseline: String
        let draftVocabularySHA256: String?
        let interleavedGeneration: Bool
        let generationFirst: Bool
        let allocatorCacheLimitBytes: Int
        let peakMemoryBytes: Int
        let measurementKernels: [String]
        let residentSeconds: [Double]
        let allocatorCacheBytes: [Int]
    }

    enum MetalBaseline: String, ExpressibleByArgument {
        case mlx
        case reuse
        case byte
        case layout
    }

    enum DecoderExperts: String, ExpressibleByArgument {
        case bf16
        case bf16GEMV = "bf16-gemv"
        case q4
        case q6
        case q6Matrix = "q6-matrix"
        case mixed
        case q8
        case q8Metal = "q8-metal"
        case q8Group32 = "q8-g32"

        var policy: SOVDecoderExperts { SOVDecoderExperts(rawValue: rawValue)! }
    }

    enum GenerationExperiment: String, ExpressibleByArgument {
        case reference, pipeline
        case draftQ8 = "draft-q8"
        case retainedAllocator = "retained-allocator"
        case sharedCache = "shared-cache"
        case combined
        case rollback
        case compiled
        case block4 = "block-4"
        case block8 = "block-8"
        case block12 = "block-12"
        case adaptive
        case adaptiveAfterFirst = "adaptive-after-first"
        case vocabulary32k = "vocabulary-32k"
        var policy: SOVGenerationExperiment { SOVGenerationExperiment(rawValue: rawValue)! }
    }
}
