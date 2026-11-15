import Foundation
import AudioCore
import AudioCodecs
import AudioSOVTranscribeModel
import MereRunCore
import MLX
import MLXNN
import MediaIO
@preconcurrency import Tokenizers

public actor SOVTranscribeGenerator {
    private let experimentalDynamicInt8: Bool
    private let decoderExperts: SOVDecoderExperts
    private let expertRecipe: SOVExpertRecipe?
    private let packedWeights: URL?
    public private(set) var packedWeightsMeasurement: SOVPackedWeightsMeasurement?
    private let experimentalAttention: Bool
    private let experimentalHeadNorm: Bool
    private let experimentalExpertFusion: Bool
    private let experimentalExpertReduction: Bool
    private let executionProvider: ParakeetExecutionProvider
    private var generationExperiment: SOVGenerationExperiment
    private var audioTokenMargin: Int?
    private var profileDecode = false
    private var captureURL: URL?
    private var captureRound = 3
    private var draftHead: Linear?
    private var restrictedDraftHead: Linear?
    private let draftVocabulary: [Int]?
    private var draftTokenMapping: MLXArray?
    private var coreMLEncoder: SOVCoreMLEncoder?
    private var lastMeasurement: SOVMeasuredTranscription?
    private var target: Gemma4TextCausalLM?
    private var encoder: SOVAudioEncoder?
    private var projector: SOVAudioProjector?
    private var draft: SOVDFlashModel?
    private var config: SOVTranscribeConfig?
    private var tokenizer: (any Tokenizer)?
    private var loadedRoot: URL?
    private var loadedDraft: String?

    public init(executionProvider: ParakeetExecutionProvider = .mlx, experimentalDynamicInt8: Bool = false,
                decoderExperts: SOVDecoderExperts = .bf16, experimentalAttention: Bool = false, experimentalHeadNorm: Bool = false,
                generationExperiment: SOVGenerationExperiment = .pipeline, expertRecipe: SOVExpertRecipe? = nil, draftVocabulary: [Int]? = nil,
                packedWeights: URL? = nil, experimentalExpertFusion: Bool = false, experimentalExpertReduction: Bool = false) {
        self.executionProvider = executionProvider
        self.experimentalDynamicInt8 = experimentalDynamicInt8
        self.decoderExperts = decoderExperts
        self.expertRecipe = expertRecipe
        self.packedWeights = packedWeights
        self.experimentalAttention = experimentalAttention
        self.experimentalHeadNorm = experimentalHeadNorm
        self.experimentalExpertFusion = experimentalExpertFusion
        self.experimentalExpertReduction = experimentalExpertReduction
        self.generationExperiment = generationExperiment
        self.draftVocabulary = draftVocabulary
    }

    public func transcribe(
        _ request: ASRRequest, modelID: String = SOVTranscribeResources.modelID, modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try Task.checkCancellation()
        let language = try Self.validate(task: request.task, language: request.language)
        guard request.maxTokens > 0 else {
            throw SpeechTranscriptionIssue("invalid_max_tokens", "Maximum transcription tokens must be positive.")
        }
        try await prepare(modelID: modelID, modelPath: modelPath, draftModel: request.sov?.draftModel, progressHandler: progressHandler)
        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Reading audio at 16 kHz..."))
        let audio = try MediaAudioIO.decode(request.audioURL, targetSampleRate: 16_000, channels: 1)
        return try await MLXRequestStreams.withStream(isolation: self) {
            try Task.checkCancellation()
            return try generate(samples: audio.samples, language: language, options: request.sov ?? SOVTranscribeOptions(),
                                maxTokens: request.maxTokens, progressHandler: progressHandler)
        }
    }

    public func prepare(
        modelID: String = SOVTranscribeResources.modelID, modelPath: String? = nil, draftModel: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws {
        guard !experimentalExpertFusion || decoderExperts == .q8Metal else {
            throw SpeechTranscriptionIssue("invalid_expert_fusion", "Expert fusion requires Q8 Metal experts.")
        }
        guard !experimentalExpertReduction || decoderExperts == .q8Metal else {
            throw SpeechTranscriptionIssue("invalid_expert_reduction", "Distributed expert reduction requires Q8 Metal experts.")
        }
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelPath ?? modelID, defaultModelID: SOVTranscribeResources.modelID)
        let draftRoot: URL?
        if let draftModel {
            draftRoot = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: draftModel, defaultModelID: SOVTranscribeResources.draftModelID).url
        } else { draftRoot = nil }
        try await MLXRequestStreams.withStream(isolation: self) {
            try Task.checkCancellation()
            guard loadedRoot != resolved.url || loadedDraft != draftRoot?.path else { return }
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading native SOV-Transcribe-1..."))
            let configuration = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: resolved.url.appendingPathComponent("config.json")))
            try configuration.validate()
            guard (decoderExperts == .mixed) == (expertRecipe != nil) else {
                throw SpeechTranscriptionIssue("invalid_expert_recipe", "A mixed decoder requires an expert recipe; other decoder policies do not accept one.")
            }
            try expertRecipe?.validate(layerCount: configuration.textConfig.numHiddenLayers)
            guard packedWeights == nil || decoderExperts.groupSize != nil else {
                throw SpeechTranscriptionIssue("invalid_packed_weights", "Packed SOV weights require quantized decoder experts.")
            }
            if let draftVocabulary {
                guard !draftVocabulary.isEmpty, draftVocabulary.count <= 32768,
                      Set(draftVocabulary).count == draftVocabulary.count,
                      draftVocabulary.allSatisfy({ (0..<configuration.textConfig.vocabSize).contains($0) }) else {
                    throw SpeechTranscriptionIssue("invalid_draft_vocabulary", "Draft vocabulary requires at most 32768 distinct valid token IDs.")
                }
            }
            guard (!decoderExperts.requiresMetal && !experimentalHeadNorm) || Gemma4TextCausalLM.supportsMetalExpertVerification else {
                throw SpeechTranscriptionIssue("invalid_decoder_kernel", "Experimental Metal decoder kernels require an Apple GPU.")
            }
            if let groupSize = expertRecipe?.groupSize ?? decoderExperts.groupSize {
                guard configuration.textConfig.hiddenSize.isMultiple(of: groupSize),
                      configuration.textConfig.moeIntermediateSize.isMultiple(of: groupSize) else {
                    throw SpeechTranscriptionIssue("invalid_decoder_kernel", "Quantized expert dimensions must be divisible by the quantization group size.")
                }
            }
            try SOVTranscribePreprocessing.validate(root: resolved.url, downsampleFactor: configuration.downsampleFactor)
            let loadedTokenizer = try await AutoTokenizer.from(modelFolder: resolved.url)
            guard loadedTokenizer.convertTokenToId("<|audio|>") == configuration.audioTokenID else {
                throw SpeechTranscriptionIssue("invalid_tokenizer", "Tokenizer audio marker does not match SOV-Transcribe-1.")
            }
            try Task.checkCancellation()
            let target = Gemma4TextCausalLM(config: configuration.textConfig)
            if packedWeights == nil { target.prepareForFullPrecisionExpertWeights() }
            let encoder: SOVAudioEncoder?
            let projector: SOVAudioProjector?
            let coreMLEncoder: SOVCoreMLEncoder?
            switch executionProvider {
            case .mlx:
                encoder = SOVAudioEncoder(config: configuration.audioConfig)
                projector = SOVAudioProjector(config: configuration)
                coreMLEncoder = nil
            case .coreML(let artifactURL):
                guard !experimentalDynamicInt8 else {
                    throw SpeechTranscriptionIssue("invalid_encoder_kernel", "Dynamic int8 is available only with the MLX provider.")
                }
                guard configuration.audioConfig.dModel == 1280, configuration.audioConfig.encoderLayers == 32,
                      configuration.audioConfig.encoderAttentionHeads == 20, configuration.audioConfig.encoderFFNDim == 5120,
                      configuration.audioConfig.maxSourcePositions == 1500, configuration.audioConfig.numMelBins == 128,
                      configuration.downsampleFactor == 4, configuration.textConfig.hiddenSize == 2816 else {
                    throw SpeechTranscriptionIssue("invalid_coreml_artifact", "SOV Core ML requires the original encoder/projector dimensions.")
                }
                encoder = nil; projector = nil
                coreMLEncoder = try SOVCoreMLEncoder(root: artifactURL)
            }
            var packedMeasurement: SOVPackedWeightsMeasurement?
            if let packedWeights {
                let groupSize = expertRecipe?.groupSize ?? decoderExperts.groupSize!
                let layerBits = expertRecipe?.layerBits ?? Array(repeating: decoderExperts.bits, count: configuration.textConfig.numHiddenLayers)
                try target.preparePackedExpertWeights(groupSize: groupSize, layerBits: layerBits,
                    expandedPrefill: decoderExperts == .q6Matrix, metalVerification: decoderExperts.usesMetal)
                packedMeasurement = try SOVPackedWeights.prepare(root: resolved.url, destination: packedWeights,
                    model: target, groupSize: groupSize, layerBits: layerBits)
            } else {
                try SOVTranscribeWeights.load(root: resolved.url, into: target, target: true)
            }
            if packedWeights == nil && decoderExperts != .bf16 {
                try target.prepareExperimentalExperts(groupSize: expertRecipe?.groupSize ?? decoderExperts.groupSize, bits: decoderExperts.bits, expandedPrefill: decoderExperts == .q6Matrix,
                    layerBits: expertRecipe?.layerBits, verificationGEMV: decoderExperts == .bf16GEMV,
                    metalVerification: decoderExperts.usesMetal)
            }
            target.setExperimentalAttention(experimentalAttention)
            if experimentalExpertFusion || experimentalExpertReduction {
                try target.setMetalExpertVerification(true, fusedActivation: experimentalExpertFusion,
                                                      distributedReduction: experimentalExpertReduction ? true : nil)
            }
            try target.setExperimentalHeadNorm(experimentalHeadNorm)
            target.setExperimentalCompiledPost(generationExperiment == .compiled)
            if let encoder, let projector {
                try SOVTranscribeWeights.load(root: resolved.url, into: encoder, sourcePrefix: "model.audio_tower.")
                try SOVTranscribeWeights.load(root: resolved.url, into: projector, sourcePrefix: "model.embed_audio.")
                if experimentalDynamicInt8 { try encoder.prepareDynamicInt8() }
            }
            let draft: SOVDFlashModel?
            if let draftRoot {
                let draftConfig = try JSONDecoder().decode(SOVDFlashConfig.self, from: Data(contentsOf: draftRoot.appendingPathComponent("config.json")))
                try draftConfig.validate(target: configuration.textConfig)
                let model = SOVDFlashModel(config: draftConfig)
                try SOVTranscribeWeights.load(root: draftRoot, into: model)
                draft = model
            } else { draft = nil }
            let preparedDraftHead = generationExperiment.quantizedDraftHead ? try Self.makeDraftHead(target: target, draft: draft) : nil
            try Task.checkCancellation()
            self.coreMLEncoder = coreMLEncoder
            self.target = target
            self.encoder = encoder
            self.projector = projector
            self.draft = draft
            self.draftHead = preparedDraftHead
            self.restrictedDraftHead = nil
            self.draftTokenMapping = nil
            if generationExperiment == .vocabulary32k { prepareRestrictedHead(target: target) }
            self.config = configuration
            self.packedWeightsMeasurement = packedMeasurement
            tokenizer = loadedTokenizer
            loadedRoot = resolved.url
            loadedDraft = draftRoot?.path
        }
    }

    private func generate(
        samples: [Float], language: String?, options: SOVTranscribeOptions, maxTokens: Int,
        progressHandler: (@Sendable (ASRProgress) -> Void)?, projectedOutput: URL? = nil,
        prefillChunkSize: Int = 1024
    ) throws -> ASRResult {
        guard let target, let config, let tokenizer else {
            throw SpeechTranscriptionIssue("model_not_prepared", "Prepare SOV-Transcribe-1 before decoding audio.")
        }
        let windows = try Self.windowCount(sampleCount: samples.count)
        guard samples.allSatisfy(\.isFinite), maxTokens > 0 else {
            throw SpeechTranscriptionIssue("invalid_audio", "SOV-Transcribe requires finite audio and a positive token budget.")
        }
        defer { if !generationExperiment.retainsAllocator { Memory.clearCache() } }
        progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Extracting Whisper features..."))
        let started = ProcessInfo.processInfo.systemUptime
        let mel = try Self.audioFeatures(samples: samples)
        eval(mel)
        let featuresFinished = ProcessInfo.processInfo.systemUptime
        var projected: [MLXArray] = []
        let batchSize = coreMLEncoder == nil ? 8 : 1
        for start in stride(from: 0, to: windows, by: batchSize) {
            try Task.checkCancellation()
            let window = mel[start..<min(start + batchSize, windows), 0..., 0...]
            let output: MLXArray
            if let coreMLEncoder { output = try coreMLEncoder.encode(window) }
            else if let encoder, let projector { output = projector(try encoder(window)) }
            else { throw SpeechTranscriptionIssue("model_not_prepared", "SOV audio provider is not prepared.") }
            eval(output)
            projected.append(output.reshaped(1, -1, config.textConfig.hiddenSize))
        }
        let fullAudio = concatenated(projected, axis: 1)
        let audioTokens = try SOVAudioTokenTrimming.retainedCount(sampleCount: samples.count,
            tokensPerWindow: config.audioConfig.maxSourcePositions / config.downsampleFactor, margin: audioTokenMargin)
        let audio = audioTokens == fullAudio.dim(1) ? fullAudio : fullAudio[0..., ..<audioTokens, 0...]
        eval(audio)
        let encoderFinished = ProcessInfo.processInfo.systemUptime
        let text = Self.prompt(instruction: options.instruction, language: language, audioTokens: audio.dim(1))
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let positions = ids.indices.filter { ids[$0] == config.audioTokenID }
        guard positions.count == audio.dim(1), let first = positions.first,
              positions == Array(first..<(first + positions.count)) else {
            throw SpeechTranscriptionIssue("invalid_tokenizer", "Tokenizer must encode one contiguous audio marker per projected frame.")
        }
        let inputIDs = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
        let embeddings = target.inputEmbeddings(for: inputIDs)
        let fused = concatenated([embeddings[0..., ..<first, 0...], audio.asType(embeddings.dtype),
                                  embeddings[0..., (first + positions.count)..., 0...]], axis: 1)
        progressHandler?(ASRProgress(stage: .transcribing, message: "Decoding transcript..."))
        var prefillFinished = encoderFinished
        var decodeRounds: [SOVDecodeRoundTiming] = []
        let diagnostics = profileDecode || captureURL != nil ? SOVDecodeDiagnostics(captureURL: captureURL, captureRound: captureRound) : nil
        let collectRound: ((SOVDecodeRoundTiming) -> Void)? = diagnostics == nil ? nil : { decodeRounds.append($0) }
        let result = try SOVTranscribeDecoder.decode(
            target: target, embeddings: fused, inputIDs: inputIDs, draft: draft,
            maxTokens: maxTokens, eosTokens: [1, 106, 50], prefillChunkSize: prefillChunkSize, prefillCompleted: {
                prefillFinished = ProcessInfo.processInfo.systemUptime
            }, progress: { count in
                if count.isMultiple(of: 10) { progressHandler?(ASRProgress(stage: .transcribing, tokensGenerated: count)) }
            }, pipelined: generationExperiment.pipelined,
            draftHead: generationExperiment == .vocabulary32k ? restrictedDraftHead : (generationExperiment.quantizedDraftHead ? draftHead : nil),
            reuseAttentionStorage: generationExperiment.reusesAttentionStorage, rollbackCache: generationExperiment == .rollback,
            draftTokenLimit: generationExperiment.draftTokenLimit,
            adaptiveConfidence: [.adaptive, .adaptiveAfterFirst].contains(generationExperiment) ? 0.5 : nil,
            draftTokenMapping: generationExperiment == .vocabulary32k ? draftTokenMapping : nil,
            adaptiveAfterFirst: generationExperiment == .adaptiveAfterFirst,
            roundTiming: collectRound, diagnostics: diagnostics
        )
        let finished = ProcessInfo.processInfo.systemUptime
        let transcription = ASRResult(text: tokenizer.decode(tokens: result.tokens, skipSpecialTokens: true).trimmingCharacters(in: .whitespacesAndNewlines),
                                      language: language, duration: Double(samples.count) / 16_000)
        if !generationExperiment.retainsAllocator { Memory.clearCache() }
        let cleanupFinished = ProcessInfo.processInfo.systemUptime
        lastMeasurement = SOVMeasuredTranscription(
            text: transcription.text, tokens: result.tokens, featureSeconds: featuresFinished - started,
            encoderProjectorSeconds: encoderFinished - featuresFinished, decodeSeconds: finished - encoderFinished,
            prefillSeconds: prefillFinished - encoderFinished, generationSeconds: finished - prefillFinished,
            cleanupSeconds: cleanupFinished - finished, totalSeconds: cleanupFinished - started, verificationRounds: result.verificationRounds,
            acceptedDraftTokens: result.acceptedDraftTokens, rejectedRounds: result.rejectedRounds,
            reusedVerificationRounds: result.reusedVerificationRounds, acceptedLengths: result.acceptedLengths,
            emittedDraftTokens: result.emittedDraftTokens, proposedLengths: result.proposedLengths,
            fullAudioTokens: fullAudio.dim(1), audioTokens: audio.dim(1), promptTokens: ids.count, audioTokenMargin: audioTokenMargin, decodeRounds: diagnostics == nil ? nil : decodeRounds)
        if let projectedOutput {
            try save(arrays: ["mel": mel.asType(.float32), "audio_embeddings": fullAudio.asType(.float32),
                              "decoder_audio_embeddings": audio.asType(.float32)], url: projectedOutput)
        }
        return transcription
    }

    public func transcribePreparedMeasured(
        samples: [Float], maxTokens: Int = 448, projectedOutput: URL? = nil, prefillChunkSize: Int = 1024
    ) async throws -> SOVMeasuredTranscription {
        guard prefillChunkSize > 0 else {
            throw SpeechTranscriptionIssue("invalid_prefill_chunk_size", "Prefill chunk size must be positive.")
        }
        return try await MLXRequestStreams.withStream(isolation: self) {
            _ = try generate(samples: samples, language: nil, options: SOVTranscribeOptions(), maxTokens: maxTokens,
                             progressHandler: nil, projectedOutput: projectedOutput, prefillChunkSize: prefillChunkSize)
            return lastMeasurement!
        }
    }

    public func unload() {
        target = nil; encoder = nil; projector = nil; draft = nil; draftHead = nil; restrictedDraftHead = nil; coreMLEncoder = nil; lastMeasurement = nil
        config = nil; tokenizer = nil; loadedRoot = nil; loadedDraft = nil; draftTokenMapping = nil
        Memory.clearCache()
    }

    /// Opt-in host-stage/signpost diagnostics, independent of performance sampling.
    package func setDecodeProfileForBenchmark(_ enabled: Bool, captureURL: URL? = nil, captureRound: Int = 3) throws {
        guard captureRound > 0 else { throw SpeechTranscriptionIssue("invalid_capture_round", "Capture round must be positive.") }
        profileDecode = enabled
        self.captureURL = captureURL
        self.captureRound = captureRound
    }

    /// Benchmark-only input policy. Ordinary transcription retains the upstream padded windows.
    package func setAudioTokenMarginForBenchmark(_ margin: Int?) throws {
        try SOVAudioTokenTrimming.validate(margin: margin)
        audioTokenMargin = margin
    }

    /// Prepare outside the timed interval; switching never repacks an existing head.
    package func setGenerationForBenchmark(_ experiment: SOVGenerationExperiment) async throws {
        guard let target else { throw SpeechTranscriptionIssue("unprepared_model", "Prepare SOV before changing generation scheduling.") }
        try await MLXRequestStreams.withStream(isolation: self) {
            if experiment.quantizedDraftHead, draftHead == nil { draftHead = try Self.makeDraftHead(target: target, draft: draft) }
            if experiment == .vocabulary32k { prepareRestrictedHead(target: target) }
            target.setExperimentalCompiledPost(experiment == .compiled)
            generationExperiment = experiment
        }
    }

    private func prepareRestrictedHead(target: Gemma4TextCausalLM) {
        if restrictedDraftHead == nil {
            if let draftVocabulary {
                let mapping = MLXArray(draftVocabulary.map(Int32.init))
                restrictedDraftHead = target.makeRestrictedDraftHead(tokenIDs: mapping)
                draftTokenMapping = mapping
            } else { restrictedDraftHead = target.makeRestrictedDraftHead(count: 32768) }
            eval(restrictedDraftHead!.parameters())
        }
    }

    private static func makeDraftHead(target: Gemma4TextCausalLM, draft: SOVDFlashModel?) throws -> QuantizedLinear {
        guard draft != nil, target.config.hiddenSize.isMultiple(of: 64) else {
            throw SpeechTranscriptionIssue("invalid_draft_head", "The Q8 draft head requires DFlash and a hidden width divisible by 64.")
        }
        let head = target.makeQuantizedDraftHead()
        eval(head.parameters())
        return head
    }

    /// Paired benchmark control: changes no weights and performs no quantization.
    package func setHeadNormForBenchmark(_ enabled: Bool) async throws {
        guard let target else { throw SpeechTranscriptionIssue("unprepared_model", "Prepare SOV before changing head normalization dispatch.") }
        try await MLXRequestStreams.withStream(isolation: self) { try target.setExperimentalHeadNorm(enabled) }
    }

    /// Paired benchmark control: changes no weights and performs no quantization.
    package func setAttentionForBenchmark(_ enabled: Bool) async throws {
        guard let target else { throw SpeechTranscriptionIssue("unprepared_model", "Prepare SOV before changing attention dispatch.") }
        await MLXRequestStreams.withStream(isolation: self) { target.setExperimentalAttention(enabled) }
    }

    /// Selected reduction policy; individual projections retain shape fallbacks.
    package var usesDistributedExpertReduction: Bool { target?.usesDistributedMetalExpertReduction ?? false }

    /// A nil reduction override selects the qualified device default.
    package func setMetalVerificationForBenchmark(_ enabled: Bool, byteLoads: Bool = true, optimizedLayout: Bool = true,
                                                 fusedActivation: Bool = false, distributedReduction: Bool? = nil) async throws {
        guard [.q8, .q8Metal].contains(decoderExperts), let target, Gemma4TextCausalLM.supportsMetalExpertVerification else {
            throw SpeechTranscriptionIssue("invalid_decoder_kernel", "Paired Metal comparison requires a prepared Q8 group-64 target on an Apple GPU.")
        }
        try await MLXRequestStreams.withStream(isolation: self) {
            try target.setMetalExpertVerification(enabled, byteLoads: byteLoads, optimizedLayout: optimizedLayout,
                                                  fusedActivation: fusedActivation, distributedReduction: distributedReduction)
        }
    }

    public static func validate(task: ASRTask, language: String?) throws -> String? {
        guard task == .transcribe else {
            throw SpeechTranscriptionIssue("unsupported_task", "SOV-Transcribe-1 supports transcription; use Qwen for translation.")
        }
        guard let raw = language?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty, raw.lowercased() != "auto" else { return nil }
        guard let code = ASRBackendRouting.normalizeLanguageHint(raw), languages[code] != nil else {
            throw SpeechTranscriptionIssue("unsupported_language", "SOV-Transcribe-1 supports English, French, German, Spanish, Italian, Portuguese, and Dutch.")
        }
        return code
    }

    static func audioFeatures(samples: [Float]) throws -> MLXArray {
        let windows = try windowCount(sampleCount: samples.count)
        guard samples.allSatisfy(\.isFinite) else {
            throw SpeechTranscriptionIssue("invalid_audio", "SOV-Transcribe-1 requires finite audio samples.")
        }
        try Task.checkCancellation()
        let outputSamples = windows * 480_000
        // Keep a full zero FFT window after the last real sample. Its final
        // mel column is the globally normalized silence value for the tail.
        let transformSamples = min(outputSamples, ((samples.count + 400 + 159) / 160) * 160)
        var padded = samples
        padded += Array(repeating: Float(0), count: transformSamples - samples.count)
        // Global normalization and STFT precede window splitting.
        var mel = MelSpectrogram().extract(from: padded)
        if transformSamples < outputSamples {
            let silence = mel[0..., 0..., (-1)...]
            mel = concatenated([mel, broadcast(silence, to: [1, 128, outputSamples / 160 - mel.dim(2)])], axis: 2)
        }
        try Task.checkCancellation()
        return mel.reshaped(128, windows, 3000).transposed(1, 0, 2)
    }

    static func windowCount(sampleCount: Int) throws -> Int {
        guard sampleCount > 0, sampleCount <= 16_000 * 1800 else {
            throw SpeechTranscriptionIssue("invalid_audio", "SOV-Transcribe-1 expects audio longer than zero and at most 30 minutes.")
        }
        return (sampleCount + 479_999) / 480_000
    }

    static func prompt(instruction: String?, language: String?, audioTokens: Int) -> String {
        let instruction = instruction?.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = instruction.flatMap { $0.isEmpty ? nil : $0 }
            ?? language.flatMap { languages[$0].map { "Transcribe the \($0) audio into text, ensuring all punctuation marks are included." } }
            ?? "Transcribe the audio verbatim, outputting only spoken words in sequence."
        return "<bos><|turn>user\n" + text + "<|audio>" + String(repeating: "<|audio|>", count: audioTokens)
            + "<audio|><turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    }

    private static let languages = ["en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch"]
}
