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

private final class MiniMaxMusic3PrepackedModelContainer: Module {
    @ModuleInfo(key: "language_model") var languageModel: MiniMaxMusic3LanguageModel
    @ModuleInfo(key: "rvq_depth_decoder") var depthDecoder: MiniMaxMusic3DepthDecoder
    @ModuleInfo(key: "condition_encoder") var conditionEncoder: MiniMaxMusic3ConditionEncoder
    @ModuleInfo(key: "transformer") var transformer: MiniMaxMusic3Transformer
    @ModuleInfo(key: "vocoder") var vocoder: MiniMaxMusic3Vocoder

    init(resources: MiniMaxMusic3Resources) throws {
        self._languageModel.wrappedValue = MiniMaxMusic3LanguageModel(
            configuration: try resources.loadLanguageConfiguration()
        )
        self._depthDecoder.wrappedValue = MiniMaxMusic3DepthDecoder(
            configuration: try resources.loadDepthConfiguration()
        )
        self._conditionEncoder.wrappedValue = MiniMaxMusic3ConditionEncoder(
            configuration: try resources.loadConditionConfiguration()
        )
        self._transformer.wrappedValue = MiniMaxMusic3Transformer(
            configuration: try resources.loadTransformerConfiguration()
        )
        self._vocoder.wrappedValue = MiniMaxMusic3Vocoder(
            configuration: try resources.loadVocoderConfiguration()
        )
    }
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
    /// Experimental optimized graph with MXFP8 transformer weights.
    case mxfp8

    var affineQuantizationBits: Int? {
        switch self {
        case .reference, .optimized, .mxfp8:
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

    var usesMXFP8: Bool {
        self == .mxfp8
    }
}

public enum MiniMaxMusic3ModelLoader {
    public static func load(
        from resources: MiniMaxMusic3Resources,
        performanceMode: MiniMaxMusic3PerformanceMode = .optimized
    ) throws -> MiniMaxMusic3Models {
        try validate(resources)
        if performanceMode.usesMXFP8 {
            return try loadPrepackedMXFP8(from: resources)
        }
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
        guard !performanceMode.usesMXFP8 else {
            throw MiniMaxMusic3Error.prepackedMXFP8RequiresResident
        }
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
            depthDecoder.prepareFusedProjections()
        }
        if let bits = performanceMode.affineQuantizationBits {
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
        guard !performanceMode.usesMXFP8 else {
            throw MiniMaxMusic3Error.prepackedMXFP8RequiresResident
        }
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

    private static func loadPrepackedMXFP8(
        from resources: MiniMaxMusic3Resources
    ) throws -> MiniMaxMusic3Models {
        let rootURL = resources.prepackedMXFP8RootURL()
        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let missing = resources.validatePrepackedMXFP8(at: rootURL)
        guard missing.isEmpty else {
            throw MiniMaxMusic3Error.missingResources(missing)
        }

        let container = try MiniMaxMusic3PrepackedModelContainer(resources: resources)
        try HFSafetensorsWeightsLoader.applyQuantizedWeights(
            indexURL: indexURL,
            to: container,
            groupSize: 32,
            bits: 8,
            applySVDResiduals: false,
            quantizedModuleResolver: { _, _, _, _, _, _, _ in
                (groupSize: 32, bits: 8, mode: .mxfp8)
            },
            mapper: mapPrepackedMXFP8Weight
        )
        container.languageModel.prepareCompactSemanticHead()
        container.languageModel.prepareFusedProjections()
        container.depthDecoder.prepareFusedProjections()
        container.transformer.prepareFusedBlockProjections()
        MLX.eval(container.parameters())

        let tokenizer = try ACEStep5HzLMTokenizer.load(
            from: rootURL.appendingPathComponent("tokenizer", isDirectory: true),
            requireAudioCodeTokens: false
        )
        return MiniMaxMusic3Models(
            languageModel: container.languageModel,
            depthDecoder: container.depthDecoder,
            conditionEncoder: container.conditionEncoder,
            transformer: container.transformer,
            vocoder: container.vocoder,
            tokenizer: tokenizer
        )
    }

    static func mapPrepackedMXFP8Weight(
        key: String,
        value: MLXArray
    ) -> [(String, MLXArray)] {
        guard key.hasPrefix("vocoder.") else {
            return [(key, value)]
        }
        if key.hasSuffix(".alpha") {
            return [(key, value.transposed(0, 2, 1))]
        }
        guard key.hasSuffix(".weight"), key != "vocoder.dec_in_proj.weight" else {
            return [(key, value)]
        }

        let base = String(key.dropLast("weight".count))
        let floatValue = value.asType(.float32)
        let norm: MLXArray
        if key.contains(".conv_t1.") {
            norm = MLX.sqrt(
                MLX.sum(floatValue * floatValue, axes: [0, 1], keepDims: true) + 1e-12
            ).reshaped(-1, 1, 1)
        } else {
            norm = MLX.sqrt(
                MLX.sum(floatValue * floatValue, axes: [1, 2], keepDims: true) + 1e-12
            )
        }
        return [
            ("\(base)weight_v", value),
            // Keeping the matching norm in float32 makes weight_g / norm
            // exactly one when the WN layer reconstructs this fused weight.
            ("\(base)weight_g", norm),
        ]
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
    case prepackedMXFP8RequiresResident

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
        case .prepackedMXFP8RequiresResident:
            return "MiniMax Music 3 MXFP8 uses a consolidated checkpoint and requires resident memory mode."
        }
    }
}
