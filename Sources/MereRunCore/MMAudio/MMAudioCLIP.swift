import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

private enum MMAudioCLIPConfig {
    static let outputDimensions = 1_024
    static let imageSize = 384
    static let patchSize = 14
    static let visionWidth = 1_280
    static let visionLayers = 32
    static let visionHeads = 16
    static let textWidth = 1_024
    static let textLayers = 24
    static let textHeads = 16
    static let vocabularySize = 49_408
    static let contextLength = 77
}

private struct MMAudioCLIPTokenizerDocument: Decodable {
    struct Model: Decodable {
        let vocab: [String: Int]
        let merges: [String]
    }

    let model: Model
}

private struct MMAudioCLIPBytePair: Hashable {
    let left: String
    let right: String
}

private enum MMAudioCLIPTokenizerError: LocalizedError {
    case malformedMerge(String)

    var errorDescription: String? {
        switch self {
        case let .malformedMerge(merge):
            "Malformed CLIP BPE merge: \(merge)"
        }
    }
}

private final class MMAudioCLIPTokenizer {
    private let vocabulary: [String: Int]
    private let mergeRanks: [MMAudioCLIPBytePair: Int]
    private let byteEncoder: [UInt8: String]
    private let tokenRegex: NSRegularExpression
    private let whitespaceRegex: NSRegularExpression

    init(tokenizerURL: URL) throws {
        let document = try JSONDecoder().decode(
            MMAudioCLIPTokenizerDocument.self,
            from: Data(contentsOf: tokenizerURL)
        )
        self.vocabulary = document.model.vocab
        var mergeRanks: [MMAudioCLIPBytePair: Int] = [:]
        for (index, merge) in document.model.merges.enumerated() {
            let pieces = merge.split(separator: " ", omittingEmptySubsequences: false)
            guard pieces.count == 2 else {
                throw MMAudioCLIPTokenizerError.malformedMerge(merge)
            }
            mergeRanks[MMAudioCLIPBytePair(left: String(pieces[0]), right: String(pieces[1]))] = index
        }
        self.mergeRanks = mergeRanks
        self.byteEncoder = Self.makeByteEncoder()
        self.tokenRegex = try NSRegularExpression(
            pattern: #"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
        )
        self.whitespaceRegex = try NSRegularExpression(pattern: #"\s+"#)
    }

    func encode(_ text: String) -> [Int] {
        let normalized = normalize(text)
        let nsText = normalized as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        var ids = [49_406]
        tokenRegex.enumerateMatches(in: normalized, range: fullRange) { match, _, _ in
            guard let match else { return }
            let token = nsText.substring(with: match.range)
            let encoded = token.utf8.map { byteEncoder[$0]! }.joined()
            ids.append(contentsOf: bytePairEncode(encoded).map { vocabulary[$0] ?? 49_407 })
        }
        ids.append(49_407)
        return ids
    }

    private func normalize(_ text: String) -> String {
        let canonical = text.precomposedStringWithCanonicalMapping.lowercased()
        let range = NSRange(location: 0, length: (canonical as NSString).length)
        return whitespaceRegex.stringByReplacingMatches(
            in: canonical,
            range: range,
            withTemplate: " "
        )
    }

    private func bytePairEncode(_ token: String) -> [String] {
        var symbols = token.unicodeScalars.map(String.init)
        guard !symbols.isEmpty else { return [] }
        symbols[symbols.count - 1] += "</w>"

        while symbols.count > 1 {
            var selectedPair: MMAudioCLIPBytePair?
            var selectedRank = Int.max
            for index in 0..<(symbols.count - 1) {
                let pair = MMAudioCLIPBytePair(left: symbols[index], right: symbols[index + 1])
                if let rank = mergeRanks[pair], rank < selectedRank {
                    selectedPair = pair
                    selectedRank = rank
                }
            }
            guard let selectedPair else { break }

            var merged: [String] = []
            var index = 0
            while index < symbols.count {
                if index + 1 < symbols.count,
                   symbols[index] == selectedPair.left,
                   symbols[index + 1] == selectedPair.right {
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                } else {
                    merged.append(symbols[index])
                    index += 1
                }
            }
            symbols = merged
        }
        return symbols
    }

    private static func makeByteEncoder() -> [UInt8: String] {
        var bytes = Array(UInt8(ascii: "!")...UInt8(ascii: "~"))
        bytes.append(contentsOf: UInt8(0xA1)...UInt8(0xAC))
        bytes.append(contentsOf: UInt8(0xAE)...UInt8(0xFF))
        var codePoints = bytes.map(Int.init)
        var extra = 0
        for byte in UInt8.min...UInt8.max where !bytes.contains(byte) {
            bytes.append(byte)
            codePoints.append(256 + extra)
            extra += 1
        }
        return Dictionary(uniqueKeysWithValues: zip(bytes, codePoints).map { byte, codePoint in
            (byte, String(UnicodeScalar(codePoint)!))
        })
    }
}

private final class MMAudioCLIPAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inputProjectionWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inputProjectionBias: MLXArray
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    private let dimensions: Int
    private let heads: Int
    private let headDimensions: Int

    init(dimensions: Int, heads: Int) {
        self.dimensions = dimensions
        self.heads = heads
        self.headDimensions = dimensions / heads
        self._inputProjectionWeight.wrappedValue = MLXRandom.normal([dimensions * 3, dimensions]) * 0.02
        self._inputProjectionBias.wrappedValue = MLXArray.zeros([dimensions * 3])
        self._outputProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let qkv = (matmul(x, inputProjectionWeight.transposed()) + inputProjectionBias)
            .reshaped(batch, sequence, 3, heads, headDimensions)
        let query = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        let key = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let value = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / sqrt(Float(headDimensions)),
            mask: mask.map { .array($0) } ?? .none
        )
        return outputProjection(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, dimensions))
    }
}

private final class MMAudioCLIPMLP: Module {
    @ModuleInfo(key: "c_fc") var input: Linear
    @ModuleInfo(key: "c_proj") var output: Linear

    init(dimensions: Int) {
        self._input.wrappedValue = Linear(dimensions, dimensions * 4, bias: true)
        self._output.wrappedValue = Linear(dimensions * 4, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = input(x)
        return output(projected * MLX.sigmoid(1.702 * projected))
    }
}

private final class MMAudioCLIPResidualBlock: Module {
    @ModuleInfo(key: "ln_1") var firstNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: MMAudioCLIPAttention
    @ModuleInfo(key: "ln_2") var secondNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MMAudioCLIPMLP

    init(dimensions: Int, heads: Int) {
        self._firstNorm.wrappedValue = LayerNorm(dimensions: dimensions)
        self._attention.wrappedValue = MMAudioCLIPAttention(dimensions: dimensions, heads: heads)
        self._secondNorm.wrappedValue = LayerNorm(dimensions: dimensions)
        self._mlp.wrappedValue = MMAudioCLIPMLP(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let attended = x + attention(firstNorm(x), mask: mask)
        return attended + mlp(secondNorm(attended))
    }
}

private final class MMAudioCLIPTransformer: Module {
    @ModuleInfo(key: "resblocks") var blocks: [MMAudioCLIPResidualBlock]

    init(dimensions: Int, heads: Int, layers: Int) {
        self._blocks.wrappedValue = (0..<layers).map { _ in
            MMAudioCLIPResidualBlock(dimensions: dimensions, heads: heads)
        }
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        blocks.reduce(input) { hidden, block in
            block(hidden, mask: mask)
        }
    }
}

private final class MMAudioCLIPVisionTower: Module {
    @ModuleInfo(key: "conv1") var patchEmbedding: Conv2d
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "ln_pre") var preNorm: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: MMAudioCLIPTransformer
    @ModuleInfo(key: "ln_post") var postNorm: LayerNorm
    @ParameterInfo(key: "proj") var projection: MLXArray

    override init() {
        let grid = MMAudioCLIPConfig.imageSize / MMAudioCLIPConfig.patchSize
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: MMAudioCLIPConfig.visionWidth,
            kernelSize: IntOrPair(MMAudioCLIPConfig.patchSize),
            stride: IntOrPair(MMAudioCLIPConfig.patchSize),
            bias: false
        )
        self._classEmbedding.wrappedValue = MLXRandom.normal([MMAudioCLIPConfig.visionWidth]) * 0.02
        self._positionalEmbedding.wrappedValue = MLXRandom.normal([
            grid * grid + 1, MMAudioCLIPConfig.visionWidth,
        ]) * 0.02
        self._preNorm.wrappedValue = LayerNorm(dimensions: MMAudioCLIPConfig.visionWidth)
        self._transformer.wrappedValue = MMAudioCLIPTransformer(
            dimensions: MMAudioCLIPConfig.visionWidth,
            heads: MMAudioCLIPConfig.visionHeads,
            layers: MMAudioCLIPConfig.visionLayers
        )
        self._postNorm.wrappedValue = LayerNorm(dimensions: MMAudioCLIPConfig.visionWidth)
        self._projection.wrappedValue = MLXRandom.normal([
            MMAudioCLIPConfig.visionWidth, MMAudioCLIPConfig.outputDimensions,
        ]) * 0.02
    }

    func callAsFunction(_ imagesNHWC: MLXArray) -> MLXArray {
        var hidden = patchEmbedding(imagesNHWC)
        let batch = hidden.dim(0)
        hidden = hidden.reshaped(batch, -1, MMAudioCLIPConfig.visionWidth)
        let classTokens = MLX.broadcast(
            classEmbedding.reshaped(1, 1, MMAudioCLIPConfig.visionWidth),
            to: [batch, 1, MMAudioCLIPConfig.visionWidth]
        )
        hidden = MLX.concatenated([classTokens, hidden], axis: 1)
        hidden = preNorm(hidden + positionalEmbedding.expandedDimensions(axis: 0).asType(hidden.dtype))
        hidden = transformer(hidden)
        let pooled = postNorm(hidden[0..., 0, 0...])
        return matmul(pooled, projection)
    }
}

private final class MMAudioCLIPModel: Module {
    @ModuleInfo(key: "visual") var visual: MMAudioCLIPVisionTower
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ParameterInfo(key: "positional_embedding") var textPositionEmbedding: MLXArray
    @ModuleInfo(key: "transformer") var textTransformer: MMAudioCLIPTransformer
    @ModuleInfo(key: "ln_final") var textFinalNorm: LayerNorm
    @ParameterInfo(key: "text_projection") var textProjection: MLXArray
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    override init() {
        self._visual.wrappedValue = MMAudioCLIPVisionTower()
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: MMAudioCLIPConfig.vocabularySize,
            dimensions: MMAudioCLIPConfig.textWidth
        )
        self._textPositionEmbedding.wrappedValue = MLXRandom.normal([
            MMAudioCLIPConfig.contextLength, MMAudioCLIPConfig.textWidth,
        ]) * 0.01
        self._textTransformer.wrappedValue = MMAudioCLIPTransformer(
            dimensions: MMAudioCLIPConfig.textWidth,
            heads: MMAudioCLIPConfig.textHeads,
            layers: MMAudioCLIPConfig.textLayers
        )
        self._textFinalNorm.wrappedValue = LayerNorm(dimensions: MMAudioCLIPConfig.textWidth)
        self._textProjection.wrappedValue = MLXRandom.normal([
            MMAudioCLIPConfig.textWidth, MMAudioCLIPConfig.outputDimensions,
        ]) * 0.02
        self._logitScale.wrappedValue = MLXArray([Float(log(1 / 0.07))])
    }

    func encodeText(_ tokenIDs: MLXArray) -> MLXArray {
        let length = tokenIDs.dim(1)
        var hidden = tokenEmbedding(tokenIDs)
        hidden = hidden + textPositionEmbedding[0..<length, 0...].expandedDimensions(axis: 0).asType(hidden.dtype)
        hidden = textTransformer(hidden, mask: causalMask(length: length, dtype: hidden.dtype))
        return normalize(textFinalNorm(hidden))
    }

    func encodeImages(_ imagesNHWC: MLXArray) -> MLXArray {
        normalize(visual(imagesNHWC))
    }

    private func causalMask(length: Int, dtype: DType) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        for row in 0..<length {
            for column in (row + 1)..<length {
                values[row * length + column] = -.infinity
            }
        }
        return MLXArray(values).reshaped(length, length).asType(dtype)
    }

    private func normalize(_ features: MLXArray) -> MLXArray {
        let norm = MLX.sqrt((features * features).sum(axis: -1, keepDims: true))
        return features / MLX.maximum(norm, MLXArray(1e-12).asType(features.dtype))
    }
}

public final class MMAudioCLIPConditioner {
    private let model: MMAudioCLIPModel
    private let tokenizer: MMAudioCLIPTokenizer

    private init(model: MMAudioCLIPModel, tokenizer: MMAudioCLIPTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public static func load(resources: MMAudioModelResources) async throws -> MMAudioCLIPConditioner {
        let tokenizer = try MMAudioCLIPTokenizer(
            tokenizerURL: resources.clipTokenizerURL.appendingPathComponent("tokenizer.json")
        )
        let model = MMAudioCLIPModel()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.clipWeightsURL,
            to: model,
            dtype: .float16,
            verify: .none,
            mapper: mapWeights
        )
        return MMAudioCLIPConditioner(model: model, tokenizer: tokenizer)
    }

    public func encodeText(_ prompts: [String]) -> MLXArray {
        let rows = tokenRows(for: prompts)
        return model.encodeText(
            MLXArray(rows.flatMap { $0 }).reshaped(rows.count, MMAudioCLIPConfig.contextLength)
        )
    }

    func tokenIDs(for prompts: [String]) -> [Int32] {
        tokenRows(for: prompts).flatMap { $0 }
    }

    private func tokenRows(for prompts: [String]) -> [[Int32]] {
        prompts.map { prompt -> [Int32] in
            var ids = tokenizer.encode(prompt).map(Int32.init)
            if ids.count > MMAudioCLIPConfig.contextLength {
                ids = Array(ids.prefix(MMAudioCLIPConfig.contextLength))
                ids[ids.count - 1] = 49_407
            }
            ids.append(contentsOf: repeatElement(0, count: MMAudioCLIPConfig.contextLength - ids.count))
            return ids
        }
    }

    public func encodeImages(_ normalizedImagesNHWC: MLXArray) -> MLXArray {
        model.encodeImages(normalizedImagesNHWC.asType(.float16))
    }

    private static func mapWeights(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "logit_bias" {
            return []
        }
        if key == "visual.conv1.weight" {
            let transposed = value.transposed(0, 2, 3, 1)
            return [(key, transposed.reshaped(-1).reshaped(transposed.shape))]
        }
        return [(key, value)]
    }
}
