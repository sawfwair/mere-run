import Foundation
import MLX
import MLXFast
import MLXNN
@preconcurrency import Hub
@preconcurrency import Tokenizers

public struct WooshTextConditioning {
    public let embeddings: MLXArray
    public let attentionMask: MLXArray

    public init(embeddings: MLXArray, attentionMask: MLXArray) {
        self.embeddings = embeddings
        self.attentionMask = attentionMask
    }
}

public final class WooshRobertaTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    private let padTokenId: Int
    private let maxLength: Int

    public init(tokenizer: Tokenizer, padTokenId: Int, maxLength: Int) {
        self.tokenizer = tokenizer
        self.padTokenId = padTokenId
        self.maxLength = maxLength
    }

    public static func load(from directory: URL, maxLength: Int = 77, hubApi: HubApi = .shared) throws -> WooshRobertaTokenizer {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw WooshError.invalidTokenizer(directory)
        }

        let tokenizerConfig = try normalizedTokenizerConfig(url: configURL)
        let tokenizer: Tokenizer
        if FileManager.default.fileExists(atPath: tokenizerURL.path) {
            tokenizer = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: hubApi.configuration(fileURL: tokenizerURL)
            )
        } else {
            guard FileManager.default.fileExists(atPath: vocabURL.path),
                  FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw WooshError.invalidTokenizer(directory)
            }
            tokenizer = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: try makeRobertaTokenizerData(vocabURL: vocabURL, mergesURL: mergesURL)
            )
        }

        let padID = tokenizer.convertTokenToId("<pad>") ?? 1
        return WooshRobertaTokenizer(tokenizer: tokenizer, padTokenId: padID, maxLength: maxLength)
    }

    public func encode(_ prompts: [String]) -> (inputIds: MLXArray, attentionMask: MLXArray) {
        var encodedRows: [[Int32]] = []
        var maskRows: [[Int32]] = []
        encodedRows.reserveCapacity(prompts.count)
        maskRows.reserveCapacity(prompts.count)

        for prompt in prompts {
            var ids = tokenizer.encode(text: prompt)
            if ids.count > maxLength {
                ids = Array(ids.prefix(maxLength))
            }
            var mask = Array(repeating: Int32(1), count: ids.count)
            if ids.count < maxLength {
                ids.append(contentsOf: Array(repeating: padTokenId, count: maxLength - ids.count))
                mask.append(contentsOf: Array(repeating: Int32(0), count: maxLength - mask.count))
            }
            encodedRows.append(ids.map(Int32.init))
            maskRows.append(mask)
        }

        return (
            MLXArray(encodedRows.flatMap { $0 }).reshaped(prompts.count, maxLength),
            MLXArray(maskRows.flatMap { $0 }).reshaped(prompts.count, maxLength)
        )
    }

    private static func normalizedTokenizerConfig(url: URL) throws -> Config {
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode(WooshTokenizerConfigFile.self, from: data)
        return Config([
            "model_max_length": Config(decoded.modelMaxLength),
            "tokenizer_class": Config("RobertaTokenizer")
        ])
    }

    private static func makeRobertaTokenizerData(vocabURL: URL, mergesURL: URL) throws -> Config {
        let vocabData = try Data(contentsOf: vocabURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        let vocabConfig = Config(vocab.mapValues { Config($0) })
        let mergesConfig = Config(merges.map { Config($0) })
        let modelConfig = Config([
            "type": Config("BPE"),
            "vocab": vocabConfig,
            "merges": mergesConfig,
            "unk_token": Config("<unk>")
        ])
        let preTokenizerConfig = Config([
            "type": Config("ByteLevel"),
            "addPrefixSpace": Config(false),
            "trimOffsets": Config(true),
            "useRegex": Config(true)
        ])
        let decoderConfig = Config(["type": Config("ByteLevel")])
        let postProcessorConfig = Config([
            "type": Config("RobertaProcessing"),
            "sep": Config([Config("</s>"), Config(2)]),
            "cls": Config([Config("<s>"), Config(0)]),
            "trimOffset": Config(true),
            "addPrefixSpace": Config(false)
        ])

        return Config([
            "model": modelConfig,
            "preTokenizer": preTokenizerConfig,
            "decoder": decoderConfig,
            "postProcessor": postProcessorConfig
        ])
    }
}

private struct WooshTokenizerConfigFile: Decodable {
    let modelMaxLength: Int

    enum CodingKeys: String, CodingKey {
        case modelMaxLength = "model_max_length"
    }
}

final class WooshRobertaEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    private let padTokenId: Int

    init(config: WooshRobertaConfig) {
        self.padTokenId = config.padTokenId
        self._wordEmbeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._positionEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )
        self._tokenTypeEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.typeVocabSize,
            dimensions: config.hiddenSize
        )
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let batch = inputIds.dim(0)
        let sequence = inputIds.dim(1)
        let tokenTypes = MLXArray.zeros([batch, sequence], dtype: .int32)
        let positionIds = makePositionIds(inputIds: inputIds, attentionMask: attentionMask)
        return layerNorm(
            wordEmbeddings(inputIds)
                + positionEmbeddings(positionIds)
                + tokenTypeEmbeddings(tokenTypes)
        )
    }

    private func makePositionIds(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let batch = inputIds.dim(0)
        let sequence = inputIds.dim(1)
        let rows = inputIds.asArray(Int32.self)
        var positions = [Int32](repeating: Int32(padTokenId), count: rows.count)
        for batchIndex in 0..<batch {
            var current = Int32(padTokenId)
            for tokenIndex in 0..<sequence {
                let flat = batchIndex * sequence + tokenIndex
                if rows[flat] != Int32(padTokenId) {
                    current += 1
                    positions[flat] = current
                }
            }
        }
        return MLXArray(positions).reshaped(batch, sequence)
    }
}

final class WooshRobertaSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    init(config: WooshRobertaConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let sequence = hidden.dim(1)
        var q = query(hidden)
        var k = key(hidden)
        var v = value(hidden)
        q = q.reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
        let mask = WooshTensorOps.boolMaskToAttentionBias(attentionMask, dtype: hidden.dtype, queryLength: sequence)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, numHeads * headDim)
    }
}

final class WooshRobertaSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(config: WooshRobertaConfig) {
        self._dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ hidden: MLXArray, residual: MLXArray) -> MLXArray {
        layerNorm(dense(hidden) + residual)
    }
}

final class WooshRobertaAttention: Module {
    @ModuleInfo(key: "self") var selfAttention: WooshRobertaSelfAttention
    @ModuleInfo(key: "output") var output: WooshRobertaSelfOutput

    init(config: WooshRobertaConfig) {
        self._selfAttention.wrappedValue = WooshRobertaSelfAttention(config: config)
        self._output.wrappedValue = WooshRobertaSelfOutput(config: config)
    }

    func callAsFunction(_ hidden: MLXArray, attentionMask: MLXArray) -> MLXArray {
        output(selfAttention(hidden, attentionMask: attentionMask), residual: hidden)
    }
}

final class WooshRobertaIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(config: WooshRobertaConfig) {
        self._dense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        WooshTensorOps.geluTanh(dense(hidden))
    }
}

final class WooshRobertaOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(config: WooshRobertaConfig) {
        self._dense.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ hidden: MLXArray, residual: MLXArray) -> MLXArray {
        layerNorm(dense(hidden) + residual)
    }
}

final class WooshRobertaLayer: Module {
    @ModuleInfo(key: "attention") var attention: WooshRobertaAttention
    @ModuleInfo(key: "intermediate") var intermediate: WooshRobertaIntermediate
    @ModuleInfo(key: "output") var output: WooshRobertaOutput

    init(config: WooshRobertaConfig) {
        self._attention.wrappedValue = WooshRobertaAttention(config: config)
        self._intermediate.wrappedValue = WooshRobertaIntermediate(config: config)
        self._output.wrappedValue = WooshRobertaOutput(config: config)
    }

    func callAsFunction(_ hidden: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let attended = attention(hidden, attentionMask: attentionMask)
        return output(intermediate(attended), residual: attended)
    }
}

final class WooshRobertaEncoder: Module {
    @ModuleInfo(key: "layer") var layers: [WooshRobertaLayer]

    init(config: WooshRobertaConfig) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            WooshRobertaLayer(config: config)
        }
    }

    func callAsFunction(
        _ hidden: MLXArray,
        attentionMask: MLXArray,
        captureLayerIndex: Int
    ) -> MLXArray {
        var current = hidden
        var captured = hidden
        for (index, layer) in layers.enumerated() {
            current = layer(current, attentionMask: attentionMask)
            if index == captureLayerIndex {
                captured = current
            }
        }
        return captured
    }
}

final class WooshRobertaModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: WooshRobertaEmbeddings
    @ModuleInfo(key: "encoder") var encoder: WooshRobertaEncoder

    private let config: WooshRobertaConfig

    init(config: WooshRobertaConfig) {
        self.config = config
        self._embeddings.wrappedValue = WooshRobertaEmbeddings(config: config)
        self._encoder.wrappedValue = WooshRobertaEncoder(config: config)
    }

    func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let hidden = embeddings(inputIds: inputIds, attentionMask: attentionMask)
        let capture = resolvedCaptureLayerIndex()
        return encoder(hidden, attentionMask: attentionMask, captureLayerIndex: capture)
    }

    private func resolvedCaptureLayerIndex() -> Int {
        if config.lhsIndex >= 0 {
            return max(0, min(config.lhsIndex, config.numHiddenLayers - 1))
        }
        let hiddenStateCount = config.numHiddenLayers + 1
        let resolvedHiddenStateIndex = hiddenStateCount + config.lhsIndex
        return max(0, min(resolvedHiddenStateIndex - 1, config.numHiddenLayers - 1))
    }
}

public final class WooshTextConditioner: Module {
    @ModuleInfo(key: "sentence_frontend") var sentenceFrontend: WooshRobertaModel

    private let config: WooshRobertaConfig
    private let tokenizer: WooshRobertaTokenizer

    public init(config: WooshRobertaConfig = WooshRobertaConfig(), tokenizer: WooshRobertaTokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self._sentenceFrontend.wrappedValue = WooshRobertaModel(config: config)
    }

    public static func load(resources: WooshModelResources, config: WooshRobertaConfig = WooshRobertaConfig()) throws -> WooshTextConditioner {
        let tokenizer = try WooshRobertaTokenizer.load(from: resources.tokenizerRootURL, maxLength: config.maxSentenceTokens)
        let conditioner = WooshTextConditioner(config: config, tokenizer: tokenizer)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.textConditionerWeightsURL,
            to: conditioner,
            dtype: .float32,
            verify: .none,
            mapper: { key, value in
                [(key, value)]
            }
        )
        return conditioner
    }

    public func encode(prompts: [String]) -> WooshTextConditioning {
        let batch = tokenizer.encode(prompts)
        let embeddings = sentenceFrontend(inputIds: batch.inputIds, attentionMask: batch.attentionMask)
        return WooshTextConditioning(embeddings: embeddings, attentionMask: batch.attentionMask)
    }
}
