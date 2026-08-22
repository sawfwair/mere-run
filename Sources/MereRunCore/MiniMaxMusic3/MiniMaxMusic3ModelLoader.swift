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

public struct MiniMaxMusic3AutoregressiveModels {
    public let languageModel: MiniMaxMusic3LanguageModel
    public let depthDecoder: MiniMaxMusic3DepthDecoder
    public let tokenizer: ACEStep5HzLMTokenizer
}

public struct MiniMaxMusic3FlowModels {
    public let conditionEncoder: MiniMaxMusic3ConditionEncoder
    public let transformer: MiniMaxMusic3Transformer
}

public enum MiniMaxMusic3LoadingStrategy: String, CaseIterable, Codable, Sendable {
    case staged
    case resident
}

public enum MiniMaxMusic3PerformanceMode: String, CaseIterable, Codable, Sendable {
    /// Original upstream-equivalent graph, retained as an A/B and recovery path.
    case reference
    /// BF16 graph optimizations that preserve the model's sampling distribution.
    case optimized
    /// Optimized graph with affine 8-bit autoregressive weights.
    case q8
    /// Optimized graph with affine 4-bit autoregressive weights.
    case q4

    var quantizationBits: Int? {
        switch self {
        case .reference, .optimized:
            nil
        case .q8:
            8
        case .q4:
            4
        }
    }

    var usesOptimizedGraph: Bool {
        self != .reference
    }
}

public enum MiniMaxMusic3ModelLoader {
    public static func load(
        from resources: MiniMaxMusic3Resources,
        performanceMode: MiniMaxMusic3PerformanceMode = .optimized
    ) throws -> MiniMaxMusic3Models {
        try validate(resources)
        let autoregressive = try loadAutoregressive(
            from: resources,
            performanceMode: performanceMode
        )
        let flow = try loadFlow(from: resources, performanceMode: performanceMode)
        let vocoder = try loadVocoder(from: resources)
        MLX.eval(
            autoregressive.languageModel.parameters(),
            autoregressive.depthDecoder.parameters(),
            flow.conditionEncoder.parameters(),
            flow.transformer.parameters(),
            vocoder.parameters()
        )
        return MiniMaxMusic3Models(
            languageModel: autoregressive.languageModel,
            depthDecoder: autoregressive.depthDecoder,
            conditionEncoder: flow.conditionEncoder,
            transformer: flow.transformer,
            vocoder: vocoder,
            tokenizer: autoregressive.tokenizer
        )
    }

    public static func loadAutoregressive(
        from resources: MiniMaxMusic3Resources,
        performanceMode: MiniMaxMusic3PerformanceMode = .optimized
    ) throws -> MiniMaxMusic3AutoregressiveModels {
        try validate(resources)
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
        if performanceMode.usesOptimizedGraph {
            languageModel.prepareCompactSemanticHead()
            languageModel.prepareFusedProjections()
        }
        if let bits = performanceMode.quantizationBits {
            quantizeAutoregressive(
                languageModel: languageModel,
                depthDecoder: depthDecoder,
                bits: bits
            )
        }
        let tokenizer = try ACEStep5HzLMTokenizer.load(
            from: resources.tokenizerURL,
            requireAudioCodeTokens: false
        )
        MLX.eval(languageModel.parameters(), depthDecoder.parameters())
        return MiniMaxMusic3AutoregressiveModels(
            languageModel: languageModel,
            depthDecoder: depthDecoder,
            tokenizer: tokenizer
        )
    }

    public static func loadFlow(
        from resources: MiniMaxMusic3Resources,
        performanceMode: MiniMaxMusic3PerformanceMode = .optimized
    ) throws -> MiniMaxMusic3FlowModels {
        try validate(resources)
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
        if performanceMode.usesOptimizedGraph {
            transformer.prepareFusedProjections()
        }
        MLX.eval(conditionEncoder.parameters(), transformer.parameters())
        return MiniMaxMusic3FlowModels(
            conditionEncoder: conditionEncoder,
            transformer: transformer
        )
    }

    private static func quantizeAutoregressive(
        languageModel: MiniMaxMusic3LanguageModel,
        depthDecoder: MiniMaxMusic3DepthDecoder,
        bits: Int
    ) {
        for model in [languageModel as Module, depthDecoder as Module] {
            quantize(model: model, bits: bits)
        }
        MLX.eval(languageModel.parameters(), depthDecoder.parameters())
        MLX.Memory.clearCache()
    }

    private static func quantize(model: Module, bits: Int) {
        MLXNN.quantize(model: model, groupSize: 64, bits: bits) { _, module in
            if let linear = module as? Linear {
                return linear.shape.1 % 64 == 0
            }
            if let embedding = module as? Embedding {
                return embedding.shape.1 % 64 == 0
            }
            return false
        }
    }

    public static func loadVocoder(
        from resources: MiniMaxMusic3Resources
    ) throws -> MiniMaxMusic3Vocoder {
        try validate(resources)
        let vocoder = MiniMaxMusic3Vocoder(
            configuration: try resources.loadVocoderConfiguration()
        )
        try loadWeights(
            directory: resources.vocoderURL,
            into: vocoder,
            mapper: MiniMaxMusic3Vocoder.mapWeight
        )
        MLX.eval(vocoder.parameters())
        return vocoder
    }

    private static func validate(_ resources: MiniMaxMusic3Resources) throws {
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw MiniMaxMusic3Error.missingResources(missing)
        }
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
    case invalidAudio(String)
    case generatedNoFrames

    public var errorDescription: String? {
        switch self {
        case .missingResources(let urls):
            return "MiniMax Music 3 is missing required files: \(urls.map(\.path).joined(separator: ", "))"
        case .invalidPrompt(let reason):
            return "Invalid MiniMax Music 3 prompt: \(reason)"
        case .invalidAudio(let reason):
            return "Invalid MiniMax Music 3 audio: \(reason)"
        case .generatedNoFrames:
            return "MiniMax Music 3 ended before generating an audio frame."
        }
    }
}
