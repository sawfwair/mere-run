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

        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1, 0...]
            let nextToken = sampleToken(
                logits: lastLogits,
                config: genConfig,
                previousTokens: tokens + generatedTokens
            )

            if eosTokens.contains(nextToken) {
                break
            }

            generatedTokens.append(nextToken)
            let nextInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            logits = model(nextInput, cache: cache)
            MLX.eval(logits)
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

        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1, 0...]
            let nextToken = sampleToken(
                logits: lastLogits,
                config: genConfig,
                previousTokens: tokens + generatedTokens
            )

            if eosTokens.contains(nextToken) {
                break
            }

            generatedTokens.append(nextToken)
            let nextInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            logits = model(nextInput, cache: cache)
            MLX.eval(logits)

            if let onUpdate {
                let partial = tokenizer.decode(tokens: generatedTokens)
                onUpdate(partial)
            }
        }

        let decoded = tokenizer.decode(tokens: generatedTokens)
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
