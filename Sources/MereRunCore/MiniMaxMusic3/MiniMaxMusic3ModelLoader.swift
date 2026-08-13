import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3Models {
    public let languageModel: MiniMaxMusic3LanguageModel
    public let depthDecoder: MiniMaxMusic3DepthDecoder
    public let conditionEncoder: MiniMaxMusic3ConditionEncoder
    public let transformer: MiniMaxMusic3Transformer
    public let vocoder: MiniMaxMusic3Vocoder
    public let tokenizer: ACEStep5HzLMTokenizer
}

public enum MiniMaxMusic3ModelLoader {
    public static func load(from resources: MiniMaxMusic3Resources) throws -> MiniMaxMusic3Models {
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw MiniMaxMusic3Error.missingResources(missing)
        }

        let languageModel = MiniMaxMusic3LanguageModel(
            configuration: try resources.loadLanguageConfiguration()
        )
        try loadWeights(
            directory: resources.languageModelURL,
            indexName: "model.safetensors.index.json",
            singleName: "model.safetensors",
            into: languageModel
        )

        let depthDecoder = MiniMaxMusic3DepthDecoder(
            configuration: try resources.loadDepthConfiguration()
        )
        try loadWeights(
            directory: resources.depthDecoderURL,
            into: depthDecoder
        )

        let conditionEncoder = MiniMaxMusic3ConditionEncoder(
            configuration: try resources.loadConditionConfiguration()
        )
        try loadWeights(
            directory: resources.conditionEncoderURL,
            into: conditionEncoder,
            mapper: MiniMaxMusic3ConditionEncoder.mapWeight
        )

        let transformer = MiniMaxMusic3Transformer(
            configuration: try resources.loadTransformerConfiguration()
        )
        try loadWeights(
            directory: resources.transformerURL,
            into: transformer,
            mapper: MiniMaxMusic3Transformer.mapWeight
        )

        let vocoder = MiniMaxMusic3Vocoder(
            configuration: try resources.loadVocoderConfiguration()
        )
        try loadWeights(
            directory: resources.vocoderURL,
            into: vocoder,
            mapper: MiniMaxMusic3Vocoder.mapWeight
        )

        let tokenizer = try ACEStep5HzLMTokenizer.load(
            from: resources.tokenizerURL,
            requireAudioCodeTokens: false
        )
        MLX.eval(
            languageModel.parameters(),
            depthDecoder.parameters(),
            conditionEncoder.parameters(),
            transformer.parameters(),
            vocoder.parameters()
        )
        return MiniMaxMusic3Models(
            languageModel: languageModel,
            depthDecoder: depthDecoder,
            conditionEncoder: conditionEncoder,
            transformer: transformer,
            vocoder: vocoder,
            tokenizer: tokenizer
        )
    }

    private static func loadWeights(
        directory: URL,
        indexName: String = "diffusion_pytorch_model.safetensors.index.json",
        singleName: String = "diffusion_pytorch_model.safetensors",
        into model: Module,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { [($0, $1)] }
    ) throws {
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: directory.appendingPathComponent(indexName),
            singleURL: directory.appendingPathComponent(singleName),
            to: model,
            dtype: .bfloat16,
            verify: .noUnusedKeys,
            mapper: mapper
        )
    }
}

public enum MiniMaxMusic3Error: LocalizedError {
    case missingResources([URL])
    case invalidPrompt(String)
    case generatedNoFrames

    public var errorDescription: String? {
        switch self {
        case .missingResources(let urls):
            return "MiniMax Music 3 is missing required files: \(urls.map(\.path).joined(separator: ", "))"
        case .invalidPrompt(let reason):
            return "Invalid MiniMax Music 3 prompt: \(reason)"
        case .generatedNoFrames:
            return "MiniMax Music 3 ended before generating an audio frame."
        }
    }
}
