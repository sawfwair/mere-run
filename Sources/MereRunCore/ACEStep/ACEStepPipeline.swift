import Foundation
import MLX
import MLXNN
import MLXRandom

// Owns the public ACEStep pipeline entrypoints and top-level orchestration.
// Prompt construction and generation helpers live in companion files so the
// main file reads in the same order callers use the pipeline.

public struct ACEStepConditionInputs {
    public var task: ACEStepTask
    public var textHiddenStates: MLXArray
    public var textAttentionMask: MLXArray
    public var lyricHiddenStates: MLXArray
    public var lyricAttentionMask: MLXArray
    public var referAudioAcousticHiddenStatesPacked: MLXArray
    public var referAudioOrderMask: MLXArray
    public var srcLatents: MLXArray
    public var chunkMasks: MLXArray
    public var isCovers: MLXArray
    public var hiddenStates: MLXArray?
    public var attentionMask: MLXArray?
    public var silenceLatent: MLXArray?
    public var nonCoverTextHiddenStates: MLXArray?
    public var nonCoverTextAttentionMask: MLXArray?
    public var repaintMask: MLXArray?
    public var cleanSourceLatents: MLXArray?
    public var repaintConfiguration: ACEStepRepaintConfiguration?
    public var sourceAudio48kHz: MLXArray?

    public init(
        task: ACEStepTask = .textToMusic,
        textHiddenStates: MLXArray,
        textAttentionMask: MLXArray,
        lyricHiddenStates: MLXArray,
        lyricAttentionMask: MLXArray,
        referAudioAcousticHiddenStatesPacked: MLXArray,
        referAudioOrderMask: MLXArray,
        srcLatents: MLXArray,
        chunkMasks: MLXArray,
        isCovers: MLXArray,
        hiddenStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        silenceLatent: MLXArray? = nil,
        nonCoverTextHiddenStates: MLXArray? = nil,
        nonCoverTextAttentionMask: MLXArray? = nil,
        repaintMask: MLXArray? = nil,
        cleanSourceLatents: MLXArray? = nil,
        repaintConfiguration: ACEStepRepaintConfiguration? = nil,
        sourceAudio48kHz: MLXArray? = nil
    ) {
        self.task = task
        self.textHiddenStates = textHiddenStates
        self.textAttentionMask = textAttentionMask
        self.lyricHiddenStates = lyricHiddenStates
        self.lyricAttentionMask = lyricAttentionMask
        self.referAudioAcousticHiddenStatesPacked = referAudioAcousticHiddenStatesPacked
        self.referAudioOrderMask = referAudioOrderMask
        self.srcLatents = srcLatents
        self.chunkMasks = chunkMasks
        self.isCovers = isCovers
        self.hiddenStates = hiddenStates
        self.attentionMask = attentionMask
        self.silenceLatent = silenceLatent
        self.nonCoverTextHiddenStates = nonCoverTextHiddenStates
        self.nonCoverTextAttentionMask = nonCoverTextAttentionMask
        self.repaintMask = repaintMask
        self.cleanSourceLatents = cleanSourceLatents
        self.repaintConfiguration = repaintConfiguration
        self.sourceAudio48kHz = sourceAudio48kHz
    }
}

public final class ACEStepPipeline {
    public enum PipelineError: LocalizedError {
        case missingDecoderFiles([URL])
        case missingVAEFiles([URL])
        case missingTextEncoderFiles([URL])
        case lmNotConfigured
        case lmReturnedNoAudioCodes
        case lmReturnedTooFewCodes(expected: Int, actual: Int)
        case invalidConditioningInput(String)

        public var errorDescription: String? {
            switch self {
            case .missingDecoderFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing ACE-Step decoder resources:\n\(list)"
            case .missingVAEFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing Oobleck VAE resources:\n\(list)"
            case .missingTextEncoderFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing ACE-Step text encoder resources:\n\(list)"
            case .lmNotConfigured:
                return "ACEStepPipeline was initialized without 5Hz LM resources."
            case .lmReturnedNoAudioCodes:
                return "5Hz LM completed without producing any audio codes."
            case .lmReturnedTooFewCodes(let expected, let actual):
                return "5Hz LM produced too few audio codes (\(actual)); expected at least \(expected)."
            case .invalidConditioningInput(let reason):
                return "Invalid ACE-Step conditioning input: \(reason)"
            }
        }
    }

    let decoderConfig: ACEStepConfig
    public let checkpointVariant: ACEStepCheckpointVariant
    let decoder: ACEStepDiT
    let encoder: ACEStepConditionEncoder
    let tokenizer: ACEStepAudioTokenizer
    let detokenizer: ACEStepAudioTokenDetokenizer
    let nullConditionEmbedding: MLXArray
    let silenceLatent: MLXArray?

    let vaeConfig: OobleckVAEConfig
    let vae: OobleckVAE
    let lm: ACEStep5HzLM?
    let conditionTextTokenizer: ACEStep5HzLMTokenizer?
    let conditionTextEncoder: QwenEncoder?

    static let defaultVocalLanguage = "en"
    static let defaultDiTInstruction = "Fill the audio semantic mask based on the given conditions:"

    public init(
        decoderResources: ACEStepResources,
        vaeResources: OobleckVAEResources,
        lmResources: ACEStep5HzLMResources? = nil,
        textEncoderResources: ACEStep5HzLMResources? = nil,
        dtype: DType? = .float32,
        verify: Module.VerifyUpdate = .noUnusedKeys,
        fileManager: FileManager = .default
    ) throws {
        let decoderMissing = decoderResources.validate(fileManager: fileManager)
        if !decoderMissing.isEmpty {
            throw PipelineError.missingDecoderFiles(decoderMissing)
        }

        let vaeMissing = vaeResources.validate(fileManager: fileManager)
        if !vaeMissing.isEmpty {
            throw PipelineError.missingVAEFiles(vaeMissing)
        }
        if let textEncoderResources {
            let textMissing = textEncoderResources.validate(fileManager: fileManager)
            if !textMissing.isEmpty {
                throw PipelineError.missingTextEncoderFiles(textMissing)
            }
        }

        let decoderConfig = try ACEStepCheckpointLoader.loadConfig(
            resources: decoderResources,
            fileManager: fileManager
        )
        self.decoderConfig = decoderConfig
        self.checkpointVariant = ACEStepCheckpointVariant.detect(
            modelRootURL: decoderResources.modelRootURL,
            config: decoderConfig
        )
        let bundle = try ACEStepCheckpointLoader.loadTurboBundle(
            resources: decoderResources,
            dtype: dtype,
            verify: verify,
            fileManager: fileManager
        )
        self.decoder = bundle.decoder
        self.encoder = bundle.encoder
        self.tokenizer = bundle.tokenizer
        self.detokenizer = bundle.detokenizer
        self.nullConditionEmbedding = bundle.nullConditionEmbedding
        self.silenceLatent = try ACEStepCheckpointLoader.loadSilenceLatentIfPresent(
            resources: decoderResources,
            latentDim: decoderConfig.audioAcousticHiddenDim,
            dtype: dtype,
            fileManager: fileManager
        )

        self.vaeConfig = try OobleckVAECheckpointLoader.loadConfig(resources: vaeResources, fileManager: fileManager)
        self.vae = try OobleckVAECheckpointLoader.loadVAE(
            resources: vaeResources,
            dtype: .float32,
            verify: verify,
            fileManager: fileManager
        )

        if let lmResources {
            self.lm = try ACEStep5HzLM(
                resources: lmResources,
                dtype: dtype,
                verify: verify,
                fileManager: fileManager
            )
        } else {
            self.lm = nil
        }

        if let textEncoderResources {
            let data = try Data(contentsOf: textEncoderResources.configURL)
            let textConfig = try JSONDecoder().decode(ACEStep5HzLMConfig.self, from: data)
            let qwenConfig = QwenTextEncoderConfiguration(
                vocabSize: textConfig.vocabSize,
                hiddenSize: textConfig.hiddenSize,
                numHiddenLayers: textConfig.numHiddenLayers,
                numAttentionHeads: textConfig.numAttentionHeads,
                numKeyValueHeads: textConfig.numKeyValueHeads,
                intermediateSize: textConfig.intermediateSize,
                ropeTheta: textConfig.ropeTheta,
                maxPositionEmbeddings: textConfig.maxPositionEmbeddings,
                rmsNormEps: textConfig.rmsNormEps,
                promptDropIndex: 0,
                headDim: textConfig.headDim,
                mropeSection: nil,
                mropeInterleaved: false,
                useFloat32Activations: true
            )

            let promptTextEncoder = QwenEncoder(configuration: qwenConfig)
            try ModelWeightsLoader.applyHFSafetensors(
                indexURL: textEncoderResources.weightsIndexURL,
                singleURL: textEncoderResources.weightsURL,
                to: promptTextEncoder,
                dtype: dtype,
                verify: verify,
                mapper: QwenEncoder.mapHFSafetensorWeight,
                fileManager: fileManager
            )

            self.conditionTextTokenizer = try ACEStep5HzLMTokenizer.load(from: textEncoderResources.modelRootURL)
            self.conditionTextEncoder = promptTextEncoder
        } else {
            self.conditionTextTokenizer = nil
            self.conditionTextEncoder = nil
        }
    }

    /// Minimal end-to-end generation that uses the checkpoint's unconditional embedding and a basic context latent.
    ///
    /// This is intended as a bring-up path while the condition encoders are being ported.
    public func generateUnconditional(_ config: ACEStepInferenceConfig = .init()) -> MLXArray {
        let B = 1
        let T = max(1, Int((Double(config.durationSeconds) * 25.0).rounded()))

        let noise = prepareInitialNoise(
            shape: [B, T, decoderConfig.audioAcousticHiddenDim],
            config: config
        )

        let encoderHiddenStates = MLX.broadcast(
            nullConditionEmbedding,
            to: [B, nullConditionEmbedding.dim(1), nullConditionEmbedding.dim(2)]
        ).asType(noise.dtype)
        let encoderAttentionMask = MLXArray.ones([B, encoderHiddenStates.dim(1)], dtype: .float32)

        let contextChannels = decoderConfig.inChannels - decoderConfig.audioAcousticHiddenDim
        precondition(contextChannels > 0, "Expected context channels > 0 (inChannels=\(decoderConfig.inChannels), acoustic=\(decoderConfig.audioAcousticHiddenDim))")

        // Heuristic: model expects `context_latents = concat([src_latents, chunk_masks], -1)`.
        let srcChannels = contextChannels / 2
        let chunkChannels = contextChannels - srcChannels
        let srcLatents = defaultSourceLatents(targetFrames: T).asType(noise.dtype)
        precondition(srcLatents.dim(2) == srcChannels, "Expected source latent channels to match context layout.")
        let chunkMask = MLXArray.ones([B, T, chunkChannels], dtype: noise.dtype)
        let contextLatents = MLX.concatenated([srcLatents, chunkMask], axis: -1)

        let timesteps = inferenceTimesteps(config)
        let latents = denoiseTurbo(
            noise: noise,
            timesteps: timesteps,
            inferMethod: config.inferMethod,
            samplerMode: config.samplerMode,
            encoderHiddenStates: encoderHiddenStates,
            encoderAttentionMask: encoderAttentionMask,
            contextLatents: contextLatents,
            dcwEnabled: config.dcwEnabled,
            dcwMode: config.dcwMode,
            dcwScaler: config.dcwScaler,
            dcwHighScaler: config.dcwHighScaler,
            velocityNormThreshold: config.velocityNormThreshold,
            velocityEMAFactor: config.velocityEMAFactor
        )

        if config.useTiledVaeDecode {
            return vae.tiledDecode(latents, chunkSize: config.vaeChunkSize, overlap: config.vaeOverlap)
        }
        return vae.decode(latents)
    }

    public func generateWithLM(
        caption: String,
        lyrics: String,
        conditionInputs: ACEStepConditionInputs,
        config: ACEStepInferenceConfig = .init(),
        lmConfig: ACEStep5HzLMGenerationConfig = .init(),
        instruction: String = ACEStepLMInstructions.defaultInstruction,
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init(),
        lmSystemInstruction: String = ACEStepLMInstructions.defaultInstruction,
        audioCoverStrength: Float = 1.0
    ) throws -> (audio: MLXArray, lmResult: ACEStep5HzLMResult) {
        guard let lm else {
            throw PipelineError.lmNotConfigured
        }
        try checkpointVariant.validate(conditionInputs.task)

        let B = conditionInputs.srcLatents.dim(0)
        precondition(B == 1, "generateWithLM currently supports batch size 1.")

        let T = conditionInputs.srcLatents.dim(1)
        let targetCodes = Int(ceil(Double(T) / Double(decoderConfig.poolWindowSize)))
        let targetDurationSeconds = Float(targetCodes) / 5.0

        var effectiveLMConfig = lmConfig
        effectiveLMConfig.seed = effectiveLMConfig.seed ?? config.seed
        let lmResult = try generateAudioCodesWithFallback(
            lm: lm,
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            lmSystemInstruction: lmSystemInstruction,
            targetCodes: targetCodes,
            targetDurationSeconds: targetDurationSeconds,
            lmConfig: effectiveLMConfig,
            lmUserMetadata: lmUserMetadata
        )

        guard !lmResult.audioCodeValues.isEmpty else {
            throw PipelineError.lmReturnedNoAudioCodes
        }

        let effectiveCodes = lmResult.audioCodeValues
        guard effectiveCodes.count >= targetCodes else {
            throw PipelineError.lmReturnedTooFewCodes(expected: targetCodes, actual: effectiveCodes.count)
        }

        let trimmedCodes = Array(effectiveCodes.prefix(targetCodes))
        let audioCodes = MLXArray(trimmedCodes, [B, targetCodes, 1]).asType(.int32)

        let hiddenStates = conditionInputs.hiddenStates ?? conditionInputs.srcLatents
        let attentionMask = conditionInputs.attentionMask
            ?? MLXArray.ones([B, T], dtype: .int32)

        let prepared = prepareCondition(
            textHiddenStates: conditionInputs.textHiddenStates,
            textAttentionMask: conditionInputs.textAttentionMask,
            lyricHiddenStates: conditionInputs.lyricHiddenStates,
            lyricAttentionMask: conditionInputs.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked: conditionInputs.referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: conditionInputs.referAudioOrderMask,
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            silenceLatent: conditionInputs.silenceLatent,
            srcLatents: conditionInputs.srcLatents,
            chunkMasks: conditionInputs.chunkMasks,
            // Upstream promotes any request carrying semantic audio-code hints
            // to cover conditioning. Otherwise the decoded 5 Hz LM plan is
            // computed and then discarded for ordinary text-to-music requests.
            isCovers: MLXArray.ones(
                conditionInputs.isCovers.shape,
                dtype: conditionInputs.isCovers.dtype
            ),
            audioCodes: audioCodes
        )
        let nonCoverPrepared = prepareNonCoverConditionIfNeeded(conditionInputs: conditionInputs)

        let noise = prepareInitialNoise(
            shape: [B, T, decoderConfig.audioAcousticHiddenDim],
            config: config
        )

        let timesteps = inferenceTimesteps(config)
        let latents = denoiseTurbo(
            noise: noise,
            timesteps: timesteps,
            inferMethod: config.inferMethod,
            samplerMode: config.samplerMode,
            encoderHiddenStates: prepared.encoderHiddenStates.asType(noise.dtype),
            encoderAttentionMask: prepared.encoderAttentionMask,
            contextLatents: prepared.contextLatents.asType(noise.dtype),
            sourceLatentsForCoverNoise: conditionInputs.srcLatents.asType(noise.dtype),
            nonCoverEncoderHiddenStates: nonCoverPrepared?.encoderHiddenStates.asType(noise.dtype),
            nonCoverEncoderAttentionMask: nonCoverPrepared?.encoderAttentionMask,
            nonCoverContextLatents: nonCoverPrepared?.contextLatents.asType(noise.dtype),
            audioCoverStrength: audioCoverStrength,
            coverNoiseStrength: config.coverNoiseStrength,
            dcwEnabled: config.dcwEnabled,
            dcwMode: config.dcwMode,
            dcwScaler: config.dcwScaler,
            dcwHighScaler: config.dcwHighScaler,
            guidanceScale: checkpointVariant.isTurbo ? 1 : config.guidanceScale,
            guidanceMode: config.guidanceMode,
            nullConditionEmbedding: nullConditionEmbedding,
            cfgIntervalStart: config.cfgIntervalStart,
            cfgIntervalEnd: config.cfgIntervalEnd,
            velocityNormThreshold: config.velocityNormThreshold,
            velocityEMAFactor: config.velocityEMAFactor,
            repaintMask: conditionInputs.repaintMask,
            cleanSourceLatents: conditionInputs.cleanSourceLatents,
            repaintInjectionRatio: conditionInputs.repaintConfiguration?.injectionRatio ?? 0,
            repaintCrossfadeFrames: conditionInputs.repaintConfiguration?.latentCrossfadeFrames ?? 0
        )

        let audio = try decodeAndApplyRepaintSplice(
            latents: latents,
            conditionInputs: conditionInputs,
            config: config
        )

        return (audio, lmResult)
    }


    public func generatePromptToAudio(
        caption: String,
        lyrics: String,
        config: ACEStepInferenceConfig = .init(),
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init(),
        sourceLatents25Hz: MLXArray? = nil,
        sourceAudio48kHz: MLXArray? = nil,
        referenceTimbreLatents25Hz: [MLXArray]? = nil,
        referenceTimbreAudio48kHz: [MLXArray]? = nil,
        audioCoverStrength: Float = 1.0,
        vocalLanguage: String = "en",
        instruction: String? = nil,
        task: ACEStepTask = .textToMusic,
        repaintConfiguration: ACEStepRepaintConfiguration? = nil
    ) throws -> MLXArray {
        try checkpointVariant.validate(task)
        if task.requiresSourceAudio,
           sourceLatents25Hz == nil,
           sourceAudio48kHz == nil
        {
            throw PipelineError.invalidConditioningInput(
                "task '\(task.rawValue)' requires sourceLatents25Hz or sourceAudio48kHz."
            )
        }
        let effectiveRepaintConfiguration = task == .repaint || task == .lego
            ? repaintConfiguration ?? .init()
            : nil
        let effectiveInstruction = instruction ?? task.instruction()
        let T = max(1, Int((Double(config.durationSeconds) * 25.0).rounded()))
        let srcLatents = try normalizeSourceLatents(
            sourceLatents25Hz,
            sourceAudio48kHz: sourceAudio48kHz,
            targetFrames: T
        )
        let chunkChannels = chunkChannelsForPromptConditioning()

        let conditionInputs = try preparePromptConditionInputs(
            caption: caption,
            lyrics: lyrics,
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: lmUserMetadata,
            referenceTimbreLatents25Hz: referenceTimbreLatents25Hz,
            referenceTimbreAudio48kHz: referenceTimbreAudio48kHz,
            sourceAudio48kHz: sourceAudio48kHz,
            audioCoverStrength: audioCoverStrength,
            vocalLanguage: vocalLanguage,
            instruction: effectiveInstruction,
            task: task,
            repaintConfiguration: effectiveRepaintConfiguration
        )

        let B = conditionInputs.srcLatents.dim(0)
        precondition(B == 1, "generatePromptToAudio currently supports batch size 1.")

        let hiddenStates = conditionInputs.hiddenStates ?? conditionInputs.srcLatents
        let attentionMask = conditionInputs.attentionMask
            ?? MLXArray.ones([B, T], dtype: .int32)

        let prepared = prepareCondition(
            textHiddenStates: conditionInputs.textHiddenStates,
            textAttentionMask: conditionInputs.textAttentionMask,
            lyricHiddenStates: conditionInputs.lyricHiddenStates,
            lyricAttentionMask: conditionInputs.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked: conditionInputs.referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: conditionInputs.referAudioOrderMask,
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            silenceLatent: conditionInputs.silenceLatent,
            srcLatents: conditionInputs.srcLatents,
            chunkMasks: conditionInputs.chunkMasks,
            isCovers: conditionInputs.isCovers
        )
        let nonCoverPrepared = prepareNonCoverConditionIfNeeded(conditionInputs: conditionInputs)

        let noise = prepareInitialNoise(
            shape: [B, T, decoderConfig.audioAcousticHiddenDim],
            config: config
        )

        let timesteps = inferenceTimesteps(config)
        let latents = denoiseTurbo(
            noise: noise,
            timesteps: timesteps,
            inferMethod: config.inferMethod,
            samplerMode: config.samplerMode,
            encoderHiddenStates: prepared.encoderHiddenStates.asType(noise.dtype),
            encoderAttentionMask: prepared.encoderAttentionMask,
            contextLatents: prepared.contextLatents.asType(noise.dtype),
            sourceLatentsForCoverNoise: conditionInputs.srcLatents.asType(noise.dtype),
            nonCoverEncoderHiddenStates: nonCoverPrepared?.encoderHiddenStates.asType(noise.dtype),
            nonCoverEncoderAttentionMask: nonCoverPrepared?.encoderAttentionMask,
            nonCoverContextLatents: nonCoverPrepared?.contextLatents.asType(noise.dtype),
            audioCoverStrength: audioCoverStrength,
            coverNoiseStrength: config.coverNoiseStrength,
            dcwEnabled: config.dcwEnabled,
            dcwMode: config.dcwMode,
            dcwScaler: config.dcwScaler,
            dcwHighScaler: config.dcwHighScaler,
            guidanceScale: checkpointVariant.isTurbo ? 1 : config.guidanceScale,
            guidanceMode: config.guidanceMode,
            nullConditionEmbedding: nullConditionEmbedding,
            cfgIntervalStart: config.cfgIntervalStart,
            cfgIntervalEnd: config.cfgIntervalEnd,
            velocityNormThreshold: config.velocityNormThreshold,
            velocityEMAFactor: config.velocityEMAFactor,
            repaintMask: conditionInputs.repaintMask,
            cleanSourceLatents: conditionInputs.cleanSourceLatents,
            repaintInjectionRatio: conditionInputs.repaintConfiguration?.injectionRatio ?? 0,
            repaintCrossfadeFrames: conditionInputs.repaintConfiguration?.latentCrossfadeFrames ?? 0
        )

        return try decodeAndApplyRepaintSplice(
            latents: latents,
            conditionInputs: conditionInputs,
            config: config
        )
    }

    public func generatePromptToAudioWithLM(
        caption: String,
        lyrics: String,
        config: ACEStepInferenceConfig = .init(),
        lmConfig: ACEStep5HzLMGenerationConfig = .init(),
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init(),
        sourceLatents25Hz: MLXArray? = nil,
        sourceAudio48kHz: MLXArray? = nil,
        referenceTimbreLatents25Hz: [MLXArray]? = nil,
        referenceTimbreAudio48kHz: [MLXArray]? = nil,
        audioCoverStrength: Float = 1.0,
        vocalLanguage: String = "en",
        instruction: String? = nil,
        lmSystemInstruction: String = ACEStepLMInstructions.defaultInstruction,
        task: ACEStepTask = .textToMusic,
        repaintConfiguration: ACEStepRepaintConfiguration? = nil
    ) throws -> (audio: MLXArray, lmResult: ACEStep5HzLMResult) {
        try checkpointVariant.validate(task)
        if task.requiresSourceAudio,
           sourceLatents25Hz == nil,
           sourceAudio48kHz == nil
        {
            throw PipelineError.invalidConditioningInput(
                "task '\(task.rawValue)' requires sourceLatents25Hz or sourceAudio48kHz."
            )
        }
        let effectiveRepaintConfiguration = task == .repaint || task == .lego
            ? repaintConfiguration ?? .init()
            : nil
        let effectiveInstruction = instruction ?? task.instruction()
        let T = max(1, Int((Double(config.durationSeconds) * 25.0).rounded()))
        let srcLatents = try normalizeSourceLatents(
            sourceLatents25Hz,
            sourceAudio48kHz: sourceAudio48kHz,
            targetFrames: T
        )
        let chunkChannels = chunkChannelsForPromptConditioning()


        let conditionInputs = try preparePromptConditionInputs(
            caption: caption,
            lyrics: lyrics,
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: lmUserMetadata,
            referenceTimbreLatents25Hz: referenceTimbreLatents25Hz,
            referenceTimbreAudio48kHz: referenceTimbreAudio48kHz,
            sourceAudio48kHz: sourceAudio48kHz,
            audioCoverStrength: audioCoverStrength,
            vocalLanguage: vocalLanguage,
            instruction: effectiveInstruction,
            task: task,
            repaintConfiguration: effectiveRepaintConfiguration
        )

        return try generateWithLM(
            caption: caption,
            lyrics: lyrics,
            conditionInputs: conditionInputs,
            config: config,
            lmConfig: lmConfig,
            instruction: effectiveInstruction,
            lmUserMetadata: lmUserMetadata,
            lmSystemInstruction: lmSystemInstruction,
            audioCoverStrength: audioCoverStrength
        )
    }

    func inferenceTimesteps(_ config: ACEStepInferenceConfig) -> [Float] {
        if checkpointVariant.isTurbo {
            return ACEStepTurboScheduler(
                fixNFE: config.fixNFE,
                shift: config.shift,
                timesteps: config.timesteps
            ).timesteps
        }
        return ACEStepContinuousScheduler(
            inferenceSteps: config.fixNFE,
            shift: config.shift,
            timesteps: config.timesteps
        ).timesteps
    }

}
