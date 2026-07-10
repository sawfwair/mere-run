import Foundation
import MLX
import MLXNN

final class LingBotVideoPromptEncoder {
    static let maxTokenLength = 37_698

    static let promptPrefix = """
    <|im_start|>system
    Given a user input that may include a text prompt alone, a text prompt with an image reference, or a text prompt with a video reference or a video reference alone, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user's input: if it is simple, enrich it by adding specifics about colors, shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal progression, and spatial relationships to create vivid, concrete, and temporally coherent scenes to create vivid and concrete scenes. Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:<|im_end|>
    <|im_start|>user
    """ + "\n"

    static let promptSuffix = "<|im_end|>\n<|im_start|>assistant\n"

    enum EncoderError: LocalizedError {
        case emptyEncoding

        var errorDescription: String? {
            switch self {
            case .emptyEncoding:
                return "LingBot-Video prompt tokenization produced no conditioning tokens."
            }
        }
    }

    private let tokenizer: QwenTokenizer
    private let model: QwenTextEncoder

    init(resources: LingBotVideoResources) throws {
        self.tokenizer = try QwenTokenizer.load(
            from: resources.processorURL,
            maxLengthOverride: Self.maxTokenLength
        )

        let config = resources.textConfig.textConfig
        let rope = config.ropeScaling
        self.model = QwenTextEncoder(configuration: QwenTextEncoderConfiguration(
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
            mropeSection: rope?.mropeSection ?? [24, 20, 20],
            mropeInterleaved: rope?.mropeInterleaved ?? true
        ))

        try Self.loadLanguageWeights(from: resources.textEncoderURL, into: model)
        MLX.eval(model)
        Memory.clearCache()
    }

    func encode(_ prompt: String) throws -> MLXArray {
        let tokenization = Self.tokenize(prompt, using: tokenizer)
        let tokenIDs = tokenization.ids
        let cropStart = tokenization.cropStart
        guard tokenIDs.count > cropStart else {
            throw EncoderError.emptyEncoding
        }

        let inputIDs = MLXArray(tokenIDs.map(Int32.init)).reshaped(1, tokenIDs.count)
        let attentionMask = MLX.ones([1, tokenIDs.count], dtype: .int32)
        let (hiddenStates, _) = model.encode(
            inputIds: inputIDs,
            attentionMask: attentionMask,
            keepFullSequence: true
        )
        let cropped = hiddenStates[0..., cropStart..., 0...].asType(.bfloat16)
        MLX.eval(cropped)
        return cropped
    }

    static func tokenize(_ prompt: String, using tokenizer: QwenTokenizer) -> (ids: [Int], cropStart: Int) {
        let fullPrompt = promptPrefix + prompt + promptSuffix
        return (
            ids: Array(tokenizer.encodeText(fullPrompt).prefix(maxTokenLength)),
            cropStart: tokenizer.encodeText(promptPrefix).count
        )
    }

    private static func loadLanguageWeights(from directory: URL, into model: QwenTextEncoder) throws {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let indexData = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: indexData)

        for shard in index.shardFilenames {
            let shardURL = directory.appendingPathComponent(shard)
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: shardURL,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                include: { Self.mapLanguageWeightKey($0) != nil },
                mapper: { key, value in
                    guard let mapped = Self.mapLanguageWeightKey(key) else { return [] }
                    return [(mapped, value)]
                },
                batchSize: 8
            )
        }
    }

    static func mapLanguageWeightKey(_ key: String) -> String? {
        let prefix = "model.language_model."
        guard key.hasPrefix(prefix) else { return nil }
        return "encoder." + String(key.dropFirst(prefix.count))
    }
}
