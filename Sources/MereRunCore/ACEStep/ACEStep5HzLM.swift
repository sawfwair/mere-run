import Foundation
import MLX
import MLXNN
import MLXRandom

public struct ACEStep5HzLMGenerationConfig: Sendable, Hashable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    /// Classifier-free guidance applied during audio-code generation.
    /// Upstream keeps metadata planning at 1.0 and uses 2.0 for codes.
    public var cfgScale: Float
    public var negativePrompt: String
    public var stopTokenIds: Set<Int>
    public var seed: UInt64?

    public init(
        maxNewTokens: Int = 4096,
        temperature: Float = 0.85,
        topK: Int = 0,
        topP: Float = 0.9,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 40_960,
        cfgScale: Float = 1.0,
        negativePrompt: String = "NO USER INPUT",
        stopTokenIds: Set<Int> = [],
        seed: UInt64? = nil
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.cfgScale = cfgScale
        self.negativePrompt = negativePrompt
        self.stopTokenIds = stopTokenIds
        self.seed = seed
    }
}

public struct ACEStep5HzLMResult: Sendable, Hashable {
    public let generatedText: String
    public let generatedTokens: [Int]
    public let audioCodeValues: [Int]
}

public final class ACEStep5HzLM {
    public enum LMError: LocalizedError {
        case missingFiles([URL])

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing ACE-Step 5Hz LM resources:\n\(list)"
            }
        }
    }

    public let config: ACEStep5HzLMConfig
    public let tokenizer: ACEStep5HzLMTokenizer
    private let model: QwenEncoder

    public init(
        resources: ACEStep5HzLMResources,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .noUnusedKeys,
        fileManager: FileManager = .default
    ) throws {
        let missing = resources.validate(fileManager: fileManager)
        if !missing.isEmpty {
            throw LMError.missingFiles(missing)
        }

        let data = try Data(contentsOf: resources.configURL)
        self.config = try JSONDecoder().decode(ACEStep5HzLMConfig.self, from: data)
        self.tokenizer = try ACEStep5HzLMTokenizer.load(from: resources.modelRootURL)

        let qwenConfig = QwenTextEncoderConfiguration(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: config.numHiddenLayers,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            intermediateSize: config.intermediateSize,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            rmsNormEps: config.rmsNormEps,
            promptDropIndex: 0,
            headDim: config.headDim,
            mropeSection: nil,
            mropeInterleaved: false,
            useFloat32Activations: true
        )

        let encoder = QwenEncoder(configuration: qwenConfig)
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: encoder,
            dtype: dtype,
            verify: verify,
            mapper: QwenEncoder.mapHFSafetensorWeight,
            fileManager: fileManager
        )
        self.model = encoder
    }

    public func buildPromptTokens(
        caption: String,
        lyrics: String,
        instruction: String = ACEStepLMInstructions.defaultInstruction,
        systemInstruction: String? = nil
    ) -> [Int] {
        let user = "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n"
        return buildPromptTokens(
            systemInstruction: systemInstruction ?? instruction,
            userContent: user
        )
    }

    public func buildPromptTokens(
        systemInstruction: String,
        userContent: String,
        assistantContent: String? = nil,
        addGenerationPrompt: Bool = true
    ) -> [Int] {
        let system = "# Instruction\n\(systemInstruction)\n\n"
        let prompt = Self.formatChat(
            system: system,
            user: userContent,
            assistant: assistantContent,
            addGenerationPrompt: addGenerationPrompt
        )
        return tokenizer.encode(prompt, addSpecialTokens: false)
    }

    /// Builds the open assistant-turn prefix used by upstream's second LM
    /// phase. Audio codes continue immediately after the pre-generated CoT.
    public func buildAudioCodePromptTokens(
        caption: String,
        lyrics: String,
        reasoning: String,
        isUnconditional: Bool = false,
        negativePrompt: String = "NO USER INPUT"
    ) -> [Int] {
        let user: String
        let effectiveReasoning: String
        if isUnconditional {
            let trimmedNegative = negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            user = trimmedNegative.isEmpty || trimmedNegative == "NO USER INPUT"
                ? "NO USER INPUT"
                : negativePrompt
            effectiveReasoning = "<think>\n\n</think>"
        } else {
            user = "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n"
            effectiveReasoning = reasoning
        }
        let prompt = Self.audioCodePromptText(
            user: user,
            reasoning: effectiveReasoning
        )
        return tokenizer.encode(prompt, addSpecialTokens: false)
    }

    static func audioCodePromptText(user: String, reasoning: String) -> String {
        let system = "# Instruction\n\(ACEStepLMInstructions.defaultInstruction)\n\n"
        return Self.formatChat(
            system: system,
            user: user,
            assistant: nil,
            addGenerationPrompt: true
        ) + reasoning + "\n\n"
    }

    static func classifierFreeGuidanceLogits(
        conditional: MLXArray,
        unconditional: MLXArray,
        scale: Float
    ) -> MLXArray {
        let conditional = conditional.asType(.float32)
        let unconditional = unconditional.asType(.float32)
        return unconditional + MLXArray(scale) * (conditional - unconditional)
    }

    public func generate(
        caption: String,
        lyrics: String,
        instruction: String = ACEStepLMInstructions.defaultInstruction,
        systemInstruction: String = ACEStepLMInstructions.defaultInstruction,
        config: ACEStep5HzLMGenerationConfig = .init(),
        logitsProcessor: ((MLXArray, [Int]) -> MLXArray)? = nil,
        didSampleToken: ((Int) -> Void)? = nil
    ) -> ACEStep5HzLMResult {
        var effectiveConfig = config
        if effectiveConfig.stopTokenIds.isEmpty, let eos = tokenizer.eosTokenId {
            effectiveConfig.stopTokenIds = [eos]
        }

        let promptTokens = buildPromptTokens(
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            systemInstruction: systemInstruction
        )
        return generateFromPromptTokens(
            promptTokens,
            config: effectiveConfig,
            logitsProcessor: logitsProcessor,
            didSampleToken: didSampleToken
        )
    }

    public func generateFromPromptTokens(
        _ promptTokens: [Int],
        config: ACEStep5HzLMGenerationConfig = .init(),
        logitsProcessor: ((MLXArray, [Int]) -> MLXArray)? = nil,
        didSampleToken: ((Int) -> Void)? = nil
    ) -> ACEStep5HzLMResult {
        var effectiveConfig = config
        if effectiveConfig.stopTokenIds.isEmpty, let eos = tokenizer.eosTokenId {
            effectiveConfig.stopTokenIds = [eos]
        }

        let generated = generateTokens(
            promptTokens: promptTokens,
            config: effectiveConfig,
            logitsProcessor: logitsProcessor,
            didSampleToken: didSampleToken
        )

        let text = tokenizer.decode(tokens: generated)
        let audioCodeValues = generated.compactMap { tokenizer.audioCodeTokenIdToValue[$0] }
        return ACEStep5HzLMResult(generatedText: text, generatedTokens: generated, audioCodeValues: audioCodeValues)
    }

    public func makeConstrainedSampler(
        enabled: Bool = true,
        debug: Bool = false,
        skipCaption: Bool = false,
        skipLanguage: Bool = false,
        stopAtReasoning: Bool = false,
        generationPhase: ACEStep5HzLMConstrainedSampler.GenerationPhase = .codes,
        targetDurationSeconds: Float? = nil,
        userMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init()
    ) -> ACEStep5HzLMConstrainedSampler {
        ACEStep5HzLMConstrainedSampler(
            tokenizer: tokenizer,
            vocabSize: config.vocabSize,
            enabled: enabled,
            debug: debug,
            skipCaption: skipCaption,
            skipLanguage: skipLanguage,
            stopAtReasoning: stopAtReasoning,
            generationPhase: generationPhase,
            targetDurationSeconds: targetDurationSeconds,
            userMetadata: userMetadata
        )
    }

    public func generateConstrained(
        caption: String,
        lyrics: String,
        instruction: String = ACEStepLMInstructions.defaultInstruction,
        systemInstruction: String = ACEStepLMInstructions.defaultInstruction,
        config: ACEStep5HzLMGenerationConfig = .init(),
        sampler: ACEStep5HzLMConstrainedSampler
    ) -> ACEStep5HzLMResult {
        generate(
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            systemInstruction: systemInstruction,
            config: config,
            logitsProcessor: { logits, tokens in sampler.processLogits(logits, tokens: tokens) },
            didSampleToken: { token in sampler.update(with: token) }
        )
    }

    /// Generates only semantic audio codes from a pre-generated reasoning
    /// prefix. This mirrors ACE-Step's two-phase generation path and applies
    /// LM CFG only to the second phase.
    public func generateAudioCodes(
        caption: String,
        lyrics: String,
        reasoning: String,
        config: ACEStep5HzLMGenerationConfig = .init(),
        sampler: ACEStep5HzLMConstrainedSampler
    ) -> ACEStep5HzLMResult {
        var effectiveConfig = config
        if effectiveConfig.stopTokenIds.isEmpty, let eos = tokenizer.eosTokenId {
            effectiveConfig.stopTokenIds = [eos]
        }

        let conditionalPrompt = buildAudioCodePromptTokens(
            caption: caption,
            lyrics: lyrics,
            reasoning: reasoning
        )
        sampler.beginAudioCodeGeneration()

        let generated: [Int]
        if effectiveConfig.cfgScale > 1 {
            let unconditionalPrompt = buildAudioCodePromptTokens(
                caption: caption,
                lyrics: lyrics,
                reasoning: reasoning,
                isUnconditional: true,
                negativePrompt: effectiveConfig.negativePrompt
            )
            generated = generateGuidedAudioCodeTokens(
                conditionalPromptTokens: conditionalPrompt,
                unconditionalPromptTokens: unconditionalPrompt,
                config: effectiveConfig,
                sampler: sampler
            )
        } else {
            generated = generateTokens(
                promptTokens: conditionalPrompt,
                config: effectiveConfig,
                logitsProcessor: { logits, tokens in
                    sampler.processLogits(logits, tokens: tokens)
                },
                didSampleToken: { token in sampler.update(with: token) }
            )
        }

        let text = tokenizer.decode(tokens: generated)
        let values = generated.compactMap { tokenizer.audioCodeTokenIdToValue[$0] }
        return ACEStep5HzLMResult(
            generatedText: text,
            generatedTokens: generated,
            audioCodeValues: values
        )
    }

    public func generateConstrained(
        promptTokens: [Int],
        config: ACEStep5HzLMGenerationConfig = .init(),
        sampler: ACEStep5HzLMConstrainedSampler
    ) -> ACEStep5HzLMResult {
        generateFromPromptTokens(
            promptTokens,
            config: config,
            logitsProcessor: { logits, tokens in sampler.processLogits(logits, tokens: tokens) },
            didSampleToken: { token in sampler.update(with: token) }
        )
    }

    private func generateTokens(
        promptTokens: [Int],
        config: ACEStep5HzLMGenerationConfig,
        logitsProcessor: ((MLXArray, [Int]) -> MLXArray)?,
        didSampleToken: ((Int) -> Void)?
    ) -> [Int] {
        let cache: [KVCache] = (0..<configNumLayers).map { _ in KVCacheSimple(step: 256) }

        let inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)
        MLX.eval(inputIds)

        let logits = model.forwardCausal(inputIds: inputIds, cache: cache, lastPositionOnly: true)
        MLX.eval(logits)

        let processLogits: (MLXArray, [Int]) -> MLXArray = { logits, tokens in
            logitsProcessor?(logits, tokens) ?? logits
        }

        let topP = (0.0..<1.0).contains(config.topP) ? config.topP : 1.0
        let generationConfig = GenerationConfig(
            maxTokens: config.maxNewTokens,
            temperature: config.temperature,
            topK: max(config.topK, 0),
            topP: topP,
            repetitionPenalty: config.repetitionPenalty,
            repetitionContextSize: config.repetitionContextSize
        )
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }
        let result = AutoregressiveDecodeEngine.decodeStateful(
            AutoregressiveDecodeRequest(
                initialLogits: logits,
                generationConfig: generationConfig,
                eosTokens: config.stopTokenIds,
                tokenBudget: config.maxNewTokens,
                historySeedTokens: promptTokens
            ),
            processLogits: processLogits,
            stepForward: { nextInput in
                self.model.forwardCausal(inputIds: nextInput, cache: cache)
            },
            didSampleToken: didSampleToken
        )
        return result.generatedTokens
    }

    private func generateGuidedAudioCodeTokens(
        conditionalPromptTokens: [Int],
        unconditionalPromptTokens: [Int],
        config: ACEStep5HzLMGenerationConfig,
        sampler: ACEStep5HzLMConstrainedSampler
    ) -> [Int] {
        let conditionalCache: [KVCache] = (0..<configNumLayers).map { _ in KVCacheSimple(step: 256) }
        let unconditionalCache: [KVCache] = (0..<configNumLayers).map { _ in KVCacheSimple(step: 256) }

        let conditionalInput = MLXArray(conditionalPromptTokens.map(Int32.init))
            .reshaped(1, conditionalPromptTokens.count)
        let unconditionalInput = MLXArray(unconditionalPromptTokens.map(Int32.init))
            .reshaped(1, unconditionalPromptTokens.count)
        var conditionalLogits = model.forwardCausal(
            inputIds: conditionalInput,
            cache: conditionalCache,
            lastPositionOnly: true
        )
        var unconditionalLogits = model.forwardCausal(
            inputIds: unconditionalInput,
            cache: unconditionalCache,
            lastPositionOnly: true
        )
        MLX.eval(conditionalLogits, unconditionalLogits)

        let topP = (0.0..<1.0).contains(config.topP) ? config.topP : 1.0
        let generationConfig = GenerationConfig(
            maxTokens: config.maxNewTokens,
            temperature: config.temperature,
            topK: max(config.topK, 0),
            topP: topP,
            repetitionPenalty: config.repetitionPenalty,
            repetitionContextSize: config.repetitionContextSize
        )
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        var generated: [Int] = []
        generated.reserveCapacity(config.maxNewTokens)
        var repetitionHistory = repetitionHistoryArray(
            promptTokens: conditionalPromptTokens,
            config: generationConfig
        )

        while generated.count < config.maxNewTokens {
            let conditional = conditionalLogits[0, -1, 0...]
            let unconditional = unconditionalLogits[0, -1, 0...]
            var guided = Self.classifierFreeGuidanceLogits(
                conditional: conditional,
                unconditional: unconditional,
                scale: config.cfgScale
            )
            guided = sampler.processLogits(guided, tokens: generated)

            let tokenArray = sampledTokenArray(
                logits: guided,
                config: generationConfig,
                previousTokenIndices: repetitionHistory,
                banMask: nil
            )
            MLX.eval(tokenArray)
            let token = tokenArray.item(Int.self)
            sampler.update(with: token)
            if config.stopTokenIds.contains(token) {
                break
            }
            generated.append(token)
            repetitionHistory = appendingRepetitionHistory(
                repetitionHistory,
                token: tokenArray,
                config: generationConfig
            )

            let nextInput = tokenArray.asType(.int32).reshaped(1, 1)
            conditionalLogits = model.forwardCausal(inputIds: nextInput, cache: conditionalCache)
            unconditionalLogits = model.forwardCausal(inputIds: nextInput, cache: unconditionalCache)
        }
        return generated
    }

    private var configNumLayers: Int { config.numHiddenLayers }

    private static func formatChat(system: String, user: String, assistant: String?, addGenerationPrompt: Bool) -> String {
        var s = ""
        s += "<|im_start|>system\n"
        s += system
        s += "<|im_end|>\n"

        s += "<|im_start|>user\n"
        s += user
        s += "<|im_end|>\n"

        if let assistant {
            s += "<|im_start|>assistant\n"
            s += assistant
            s += "<|im_end|>\n"
        }

        if addGenerationPrompt {
            s += "<|im_start|>assistant\n"
        }

        return s
    }
}

public enum ACEStepLMInstructions {
    public static let defaultInstruction = "Generate audio semantic tokens based on the given conditions:"
    public static let understandInstruction = "Understand the given musical conditions and describe the audio semantics accordingly:"
    public static let inspiredInstruction = "Expand the user's input into a more detailed and specific musical description:"
    public static let rewriteInstruction = "Format the user's input into a more detailed and specific musical description:"
}
