import Foundation
@preconcurrency import Hub
import MLX
import MLXFast
import MLXNN
@preconcurrency import Tokenizers

final class Flux1Tokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer
    let maximumLength: Int
    let padTokenID: Int
    let eosTokenID: Int?

    private init(tokenizer: any Tokenizer, maximumLength: Int, padTokenID: Int) {
        self.tokenizer = tokenizer
        self.maximumLength = maximumLength
        self.padTokenID = padTokenID
        self.eosTokenID = tokenizer.eosTokenId
    }

    static func load(
        from directory: URL,
        maximumLength: Int,
        fallbackPadTokenID: Int,
        hubAPI: HubApi = .shared
    ) throws -> Flux1Tokenizer {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = try hubAPI.configuration(fileURL: configURL)
        let tokenizerData: Config
        let effectiveTokenizerConfig: Config
        if FileManager.default.fileExists(atPath: dataURL.path) {
            tokenizerData = try hubAPI.configuration(fileURL: dataURL)
            effectiveTokenizerConfig = tokenizerConfig
        } else {
            tokenizerData = try makeBPETokenizerData(
                vocabURL: directory.appendingPathComponent("vocab.json"),
                mergesURL: directory.appendingPathComponent("merges.txt")
            )
            effectiveTokenizerConfig = Config([
                "tokenizer_class": Config("GPT2Tokenizer"),
                "bos_token": Config("<|startoftext|>"),
                "eos_token": Config("<|endoftext|>"),
                "unk_token": Config("<|endoftext|>"),
                "pad_token": Config("<|endoftext|>"),
                "model_max_length": Config(maximumLength),
                "clean_up_tokenization_spaces": Config(true),
            ])
        }
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: effectiveTokenizerConfig,
            tokenizerData: tokenizerData,
            strict: false
        )
        return Flux1Tokenizer(
            tokenizer: tokenizer,
            maximumLength: maximumLength,
            padTokenID: tokenizer.convertTokenToId("<pad>") ?? tokenizer.eosTokenId ?? fallbackPadTokenID
        )
    }

    private static func makeBPETokenizerData(vocabURL: URL, mergesURL: URL) throws -> Config {
        let vocabData = try Data(contentsOf: vocabURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        let data = try JSONEncoder().encode(Flux1BPETokenizerData(vocab: vocab, merges: merges))
        return try JSONDecoder().decode(Config.self, from: data)
    }

    func encode(_ text: String) -> (ids: [Int], mask: [Int], pooledIndex: Int) {
        var ids = tokenizer.encode(text: text, addSpecialTokens: true)
        if ids.count > maximumLength {
            ids = Array(ids.prefix(maximumLength))
            if let eosTokenID {
                ids[maximumLength - 1] = eosTokenID
            }
        }
        let pooledIndex = eosTokenID.flatMap { ids.firstIndex(of: $0) } ?? max(ids.count - 1, 0)
        let activeCount = ids.count
        ids.append(contentsOf: repeatElement(padTokenID, count: maximumLength - activeCount))
        return (
            ids,
            Array(repeating: 1, count: activeCount) + Array(repeating: 0, count: maximumLength - activeCount),
            pooledIndex
        )
    }
}

private struct Flux1BPETokenizerData: Encodable {
    let model: Model
    let addedTokens = [
        AddedToken(id: 49_406, content: "<|startoftext|>"),
        AddedToken(id: 49_407, content: "<|endoftext|>"),
    ]
    let normalizer = LowercaseNormalizer()
    let preTokenizer = ByteLevelPreTokenizer()
    let postProcessor = TemplatePostProcessor()
    let decoder = ByteLevelDecoder()

    init(vocab: [String: Int], merges: [String]) {
        self.model = Model(vocab: vocab, merges: merges)
    }

    struct Model: Encodable {
        let vocab: [String: Int]
        let merges: [String]
    }

    struct AddedToken: Encodable {
        let id: Int
        let content: String
        let lstrip = false
        let rstrip = false
        let normalized = false
        let singleWord = false
        let special = true
    }

    struct LowercaseNormalizer: Encodable {
        let type = "Lowercase"
    }

    struct ByteLevelPreTokenizer: Encodable {
        let type = "ByteLevel"
        let addPrefixSpace = false
        let trimOffsets = true
        let useRegex = true
    }

    struct ByteLevelDecoder: Encodable {
        let type = "ByteLevel"
    }

    struct TemplatePostProcessor: Encodable {
        let type = "TemplateProcessing"
        let single = [
            Item(specialToken: "<|startoftext|>"),
            Item(sequence: "A"),
            Item(specialToken: "<|endoftext|>"),
        ]
        let pair = [
            Item(specialToken: "<|startoftext|>"),
            Item(sequence: "A"),
            Item(specialToken: "<|endoftext|>"),
            Item(sequence: "B", typeID: 1),
            Item(specialToken: "<|endoftext|>", typeID: 1),
        ]

        struct Item: Encodable {
            let specialToken: Payload?
            let sequence: Payload?

            init(specialToken: String, typeID: Int = 0) {
                self.specialToken = Payload(id: specialToken, typeID: typeID)
                self.sequence = nil
            }

            init(sequence: String, typeID: Int = 0) {
                self.specialToken = nil
                self.sequence = Payload(id: sequence, typeID: typeID)
            }

            enum CodingKeys: String, CodingKey {
                case specialToken = "SpecialToken"
                case sequence = "Sequence"
            }
        }

        struct Payload: Encodable {
            let id: String
            let typeID: Int
        }
    }
}

final class Flux1CLIPAttention: Module {
    let headCount: Int
    let headSize: Int
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear

    init(configuration: Flux1CLIPConfiguration) {
        self.headCount = configuration.numAttentionHeads
        self.headSize = configuration.hiddenSize / configuration.numAttentionHeads
        self._query.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._key.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._value.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        self._output.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        func heads(_ value: MLXArray) -> MLXArray {
            value.reshaped(batch, length, headCount, headSize).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: heads(query(input)),
            keys: heads(key(input)),
            values: heads(value(input)),
            scale: 1 / sqrt(Float(headSize)),
            mask: mask
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, headCount * headSize))
    }
}

final class Flux1CLIPMLP: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

    init(configuration: Flux1CLIPConfiguration) {
        self._input.wrappedValue = Linear(configuration.hiddenSize, configuration.intermediateSize, bias: true)
        self._output.wrappedValue = Linear(configuration.intermediateSize, configuration.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let projected = input(value)
        return output(projected * MLX.sigmoid(projected * 1.702))
    }
}

final class Flux1CLIPLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Flux1CLIPAttention
    @ModuleInfo(key: "layer_norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Flux1CLIPMLP
    @ModuleInfo(key: "layer_norm2") var mlpNorm: LayerNorm

    init(configuration: Flux1CLIPConfiguration) {
        self._attention.wrappedValue = Flux1CLIPAttention(configuration: configuration)
        self._attentionNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEps
        )
        self._mlp.wrappedValue = Flux1CLIPMLP(configuration: configuration)
        self._mlpNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEps
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let attended = input + attention(attentionNorm(input), mask: mask)
        return attended + mlp(mlpNorm(attended))
    }
}

final class Flux1CLIPEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(configuration: Flux1CLIPConfiguration) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.maxPositionEmbeddings,
            dimensions: configuration.hiddenSize
        )
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        let positionIDs = MLXArray(0..<inputIDs.dim(1)).asType(.int32)
        return tokenEmbedding(inputIDs) + positionEmbedding(positionIDs)
    }
}

final class Flux1CLIPEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Flux1CLIPLayer]

    init(configuration: Flux1CLIPConfiguration) {
        self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
            Flux1CLIPLayer(configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return hidden
    }
}

final class Flux1CLIPTextModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Flux1CLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: Flux1CLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(configuration: Flux1CLIPConfiguration) {
        self._embeddings.wrappedValue = Flux1CLIPEmbeddings(configuration: configuration)
        self._encoder.wrappedValue = Flux1CLIPEncoder(configuration: configuration)
        self._finalNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEps
        )
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        let length = inputIDs.dim(1)
        let hidden = embeddings(inputIDs)
        var values = [Float](repeating: 0, count: length * length)
        for row in 0..<length {
            for column in (row + 1)..<length {
                values[row * length + column] = -1e9
            }
        }
        let mask = MLXArray(values, [1, 1, length, length]).asType(hidden.dtype)
        return finalNorm(encoder(hidden, mask: mask))
    }
}

final class Flux1CLIPModel: Module {
    @ModuleInfo(key: "text_model") var textModel: Flux1CLIPTextModel

    init(configuration: Flux1CLIPConfiguration) {
        self._textModel.wrappedValue = Flux1CLIPTextModel(configuration: configuration)
        super.init()
    }

    func pooled(inputIDs: MLXArray, index: Int) -> MLXArray {
        textModel(inputIDs)[0..., index, 0...]
    }
}

enum Flux1TextEncoderLoader {
    static func loadCLIP(
        resources: Flux1Resources,
        configuration: Flux1CLIPConfiguration
    ) throws -> Flux1CLIPModel {
        let model = Flux1CLIPModel(configuration: configuration)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.clipWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.shapeMismatch]
        )
        return model
    }

    static func loadT5(
        resources: Flux1Resources,
        configuration: Flux1T5Configuration,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> Wan2TextEncoderModel {
        let model = Wan2TextEncoderModel(configuration: configuration.wanConfiguration)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.t5WeightsIndexURL,
            to: model,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: { key, value in
                mapT5Weight(
                    key: key,
                    value: value,
                    layerCount: configuration.numLayers
                )
            },
            progressHandler: progressHandler
        )
        return model
    }

    static func mapT5Weight(key: String, value: MLXArray, layerCount: Int) -> [(String, MLXArray)] {
        if key == "shared.weight" {
            return [("token_embedding.weight", value)]
        }
        if key == "encoder.final_layer_norm.weight" {
            return [("norm.weight", value)]
        }
        if key == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" {
            return (0..<layerCount).map { ("blocks.\($0).pos_embedding.embedding.weight", value) }
        }

        let components = key.split(separator: ".").map(String.init)
        guard components.count >= 6,
              components[0] == "encoder",
              components[1] == "block",
              let layerIndex = Int(components[2]),
              components[3] == "layer" else {
            return []
        }
        let remainder = components.dropFirst(5).joined(separator: ".")
        let prefix = "blocks.\(layerIndex)"
        switch (components[4], remainder) {
        case ("0", "SelfAttention.q.weight"):
            return [("\(prefix).attn.q.weight", value)]
        case ("0", "SelfAttention.k.weight"):
            return [("\(prefix).attn.k.weight", value)]
        case ("0", "SelfAttention.v.weight"):
            return [("\(prefix).attn.v.weight", value)]
        case ("0", "SelfAttention.o.weight"):
            return [("\(prefix).attn.o.weight", value)]
        case ("0", "layer_norm.weight"):
            return [("\(prefix).norm1.weight", value)]
        case ("1", "DenseReluDense.wi_0.weight"):
            return [("\(prefix).ffn.gate_proj.weight", value)]
        case ("1", "DenseReluDense.wi_1.weight"):
            return [("\(prefix).ffn.fc1.weight", value)]
        case ("1", "DenseReluDense.wo.weight"):
            return [("\(prefix).ffn.fc2.weight", value)]
        case ("1", "layer_norm.weight"):
            return [("\(prefix).norm2.weight", value)]
        default:
            return []
        }
    }
}
