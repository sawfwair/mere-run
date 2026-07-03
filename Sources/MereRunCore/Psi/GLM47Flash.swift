import Foundation
import MLX
import MLXNN

public final class GLM47Flash: @unchecked Sendable {
    public enum Error: Swift.Error {
        case invalidConfig(URL)
        case modelNotLoaded
        case generationNotImplemented
    }

    public let config: GLM47FlashConfig
    public let tokenizer: GLM47Tokenizer
    public let model: GLM47FlashModel

    public init(modelRoot: URL) throws {
        let configURL = modelRoot.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        self.config = try JSONDecoder().decode(GLM47FlashConfig.self, from: configData)
        self.tokenizer = try GLM47Tokenizer.load(from: modelRoot)
        self.model = GLM47FlashModel(config: config)

        let indexURL = modelRoot.appendingPathComponent("model.safetensors.index.json")
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8
        try HFSafetensorsWeightsLoader.applyQuantizedWeights(
            indexURL: indexURL,
            to: model,
            groupSize: groupSize,
            bits: bits
        )
    }

    public func generate(
        messages: [ChatMessage],
        maxNewTokens: Int = 512,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        tools: [ToolDefinition]? = nil
    ) throws -> String {
        try generateWithStats(
            messages: messages,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topP: topP,
            tools: tools
        ).response
    }

    public func generateWithStats(
        messages: [ChatMessage],
        maxNewTokens: Int = 512,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        tools: [ToolDefinition]? = nil
    ) throws -> (response: String, tokensGenerated: Int) {
        let tokens = tokenizer.encodeChat(messages: messages, tools: tools, addGenerationPrompt: true)
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

        let cache: [KVCache] = (0..<config.numHiddenLayers).map { _ in
            KVCacheSimple(step: 256)
        }

        var logits = model(inputIds, cache: cache)
        MLX.eval(logits)

        let eosTokens = Set(config.eosTokenId ?? []).union([tokenizer.eosTokenId ?? -1])
        var generatedTokens: [Int] = []

        let genConfig = GenerationConfig(
            maxTokens: maxNewTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: 1.05,
            repetitionContextSize: 32
        )

        // Depth-1 pipelined decode; see generateStream for the shape. The
        // legacy loop synchronized twice per token.
        var repetitionHistory = repetitionHistoryArray(promptTokens: tokens, config: genConfig)
        var pendingToken: MLXArray?
        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1, 0...]
            let tokenArray = sampledTokenArray(
                logits: lastLogits,
                config: genConfig,
                previousTokenIndices: repetitionHistory,
                banMask: nil
            )
            repetitionHistory = appendingRepetitionHistory(repetitionHistory, token: tokenArray, config: genConfig)
            logits = model(tokenArray.asType(.int32).reshaped(1, 1), cache: cache)
            asyncEval([logits, tokenArray])

            if let previous = pendingToken {
                pendingToken = nil
                let value = previous.item(Int.self)
                if eosTokens.contains(value) {
                    let decoded = tokenizer.decode(tokens: generatedTokens)
                    return (decoded.trimmingCharacters(in: .whitespacesAndNewlines), generatedTokens.count)
                }
                generatedTokens.append(value)
            }
            pendingToken = tokenArray
        }
        if let previous = pendingToken {
            let value = previous.item(Int.self)
            if !eosTokens.contains(value) {
                generatedTokens.append(value)
            }
        }

        let decoded = tokenizer.decode(tokens: generatedTokens)
        return (decoded.trimmingCharacters(in: .whitespacesAndNewlines), generatedTokens.count)
    }

    public func generateStream(
        messages: [ChatMessage],
        maxNewTokens: Int = 512,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        tools: [ToolDefinition]? = nil,
        onUpdate: (@Sendable (String) -> Void)? = nil
    ) throws -> String {
        let tokens = tokenizer.encodeChat(messages: messages, tools: tools, addGenerationPrompt: true)
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

        let cache: [KVCache] = (0..<config.numHiddenLayers).map { _ in
            KVCacheSimple(step: 256)
        }

        var logits = model(inputIds, cache: cache)
        MLX.eval(logits)

        let eosTokens = Set(config.eosTokenId ?? []).union([tokenizer.eosTokenId ?? -1])
        var generatedTokens: [Int] = []

        let genConfig = GenerationConfig(
            maxTokens: maxNewTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: 1.05,
            repetitionContextSize: 32
        )

        // Depth-1 pipelined decode: GPU-side sampling with an on-GPU
        // repetition window, the sampled token feeding the next forward
        // directly, and the previous step's token read back while the
        // current step executes. The legacy loop synchronized twice per
        // token (sample readback plus a blocking logits eval).
        var repetitionHistory = repetitionHistoryArray(promptTokens: tokens, config: genConfig)
        var pendingToken: MLXArray?
        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1, 0...]
            let tokenArray = sampledTokenArray(
                logits: lastLogits,
                config: genConfig,
                previousTokenIndices: repetitionHistory,
                banMask: nil
            )
            repetitionHistory = appendingRepetitionHistory(repetitionHistory, token: tokenArray, config: genConfig)
            logits = model(tokenArray.asType(.int32).reshaped(1, 1), cache: cache)
            asyncEval([logits, tokenArray])

            if let previous = pendingToken {
                pendingToken = nil
                let value = previous.item(Int.self)
                if eosTokens.contains(value) {
                    let decoded = tokenizer.decode(tokens: generatedTokens)
                    return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
                }
                generatedTokens.append(value)
                if let onUpdate {
                    onUpdate(tokenizer.decode(tokens: generatedTokens))
                }
            }
            pendingToken = tokenArray
        }
        if let previous = pendingToken {
            let value = previous.item(Int.self)
            if !eosTokens.contains(value) {
                generatedTokens.append(value)
                if let onUpdate {
                    onUpdate(tokenizer.decode(tokens: generatedTokens))
                }
            }
        }

        let decoded = tokenizer.decode(tokens: generatedTokens)
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
