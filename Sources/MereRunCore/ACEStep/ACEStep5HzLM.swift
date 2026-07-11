import Foundation
import MLX
import MLXNN

public struct ACEStep5HzLMGenerationConfig: Sendable, Hashable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    public var stopTokenIds: Set<Int>

    public init(
        maxNewTokens: Int = 4096,
        temperature: Float = 0.85,
        topK: Int = 0,
        topP: Float = 0.9,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 64,
        stopTokenIds: Set<Int> = []
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.stopTokenIds = stopTokenIds
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
            var nextLogits = logitsProcessor?(logits, tokens) ?? logits
            let topK = max(config.topK, 0)
            if topK > 0 && topK < nextLogits.dim(-1) {
                // k-th largest value via argPartition — identical threshold
                // to the full argSort without sorting the whole vocabulary
                // every token.
                let kth = nextLogits.dim(-1) - topK
                let partition = argPartition(nextLogits, kth: kth, axis: -1)
                let threshold = nextLogits.take(partition[kth..<(kth + 1)], axis: -1)
                nextLogits = MLX.where(nextLogits .< threshold, MLXArray(-Float.infinity), nextLogits)
            }
            return nextLogits
        }

        let topP = (0.0..<1.0).contains(config.topP) ? config.topP : 1.0
        let generationConfig = GenerationConfig(
            maxTokens: config.maxNewTokens,
            temperature: config.temperature,
            topP: topP,
            repetitionPenalty: config.repetitionPenalty,
            repetitionContextSize: config.repetitionContextSize
        )
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
