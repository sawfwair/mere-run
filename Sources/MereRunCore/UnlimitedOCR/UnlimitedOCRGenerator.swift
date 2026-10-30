#if !os(iOS)
import Foundation
import MLX
import MLXRandom
import MereRunUnlimitedOCRModel
import MereRunDecode
@preconcurrency import Tokenizers

public actor UnlimitedOCRGenerator {
    public struct Result: Sendable {
        public let text: String
        public let tokensGenerated: Int
        public let pagesGenerated: Int
        public let reachedTokenLimit: Bool
        public let loadSeconds: Double
        public let visionSeconds: Double
        public let prefillSeconds: Double
        public let decodeSeconds: Double
    }

    private var decoder: UnlimitedOCRDecoder?
    private var vision: UnlimitedOCRVision?
    private var tokenizer: (any Tokenizer)?
    private var loadedPath: String?

    public init() {}

    /// All images belong to one document, in the supplied order.
    public func ocr(
        imageURLs: [URL], modelPath: String, maxTokens: Int = 16384,
        temperature: Float = 0, crop: Bool? = nil, noRepeatNgramSize: Int = 35, ngramWindow: Int? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> Result {
        try await MLXRequestStreams.withStream(isolation: self) {
            let window = ngramWindow ?? (imageURLs.count > 1 ? 1024 : 128)
            guard !imageURLs.isEmpty, maxTokens > 0, temperature.isFinite, temperature >= 0,
                  noRepeatNgramSize >= 0, window > 0 else {
                throw UnlimitedOCRError.invalidInput("Provide images, a positive token budget, nonnegative temperature and n-gram size, and a positive n-gram window.")
            }
            let loadStart = Date()
            progress?("Loading checkpoint")
            try await load(modelPath)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            guard let decoder, let vision, let tokenizer else {
                throw UnlimitedOCRError.checkpoint("runtime did not load")
            }
            var ids = [decoder.config.bosTokenId]
            var pieces: [MLXArray] = [decoder.embeddings(ids)]
            let visionStart = Date()
            for (index, url) in imageURLs.enumerated() {
                try Task.checkCancellation()
                let page = try UnlimitedOCRImage.prepare(url, crop: crop ?? (imageURLs.count == 1))
                progress?("Encoding page \(index + 1)/\(imageURLs.count): \(page.tiles.count) tiles, \(page.tokenCount) visual tokens")
                let marker = decoder.config.sparseAttentionConfig.pageTokenId
                ids.append(marker)
                pieces.append(decoder.embeddings([marker]))
                let features = vision.page(global: page.global, tiles: page.tiles, columns: page.columns, rows: page.rows)
                guard features.dim(0) == page.tokenCount else { throw UnlimitedOCRError.invalidInput("Image token count mismatch.") }
                pieces.append(features)
                ids += [Int](repeating: decoder.config.sparseAttentionConfig.imageTokenId, count: page.tokenCount)
            }
            let visionSeconds = Date().timeIntervalSince(visionStart)
            let instruction = tokenizer.encode(text: "Multi page parsing.", addSpecialTokens: false)
            ids += instruction
            pieces.append(decoder.embeddings(instruction))
            guard ids.count + maxTokens <= 131072 else {
                throw UnlimitedOCRError.invalidInput("Document prompt and output budget exceed 131072 tokens. Reduce pages or --max-tokens.")
            }
            var state = try UnlimitedOCRPageState(tokens: ids)
            let cache = (0..<12).map { _ in UnlimitedOCRCache() }
            progress?("Prefilling \(ids.count) tokens")
            let prefillStart = Date()
            var logits = decoder.forward(concatenated(pieces, axis: 0), state: state, cache: cache)
            eval(logits)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            progress?("Decoding")
            let decodeStart = Date()
            var output: [Int] = []
            var pages = 0
            var pageLength = 0
            var stopped = false
            for step in 0..<maxTokens {
                try Task.checkCancellation()
                let pageID = decoder.config.sparseAttentionConfig.pageTokenId
                let eos = decoder.config.eosTokenId
                let banned = NoRepeatNGram.bannedTokens(in: output, size: noRepeatNgramSize, window: window)
                if !banned.isEmpty { logits[MLXArray(banned)] = MLXArray(-Float.infinity) }
                // Match upstream's page-count and 20000-token page guard.
                if pages > imageURLs.count || pageLength >= 20000 {
                    let forced = pages < imageURLs.count ? pageID : eos
                    logits = which(MLXArray(0..<decoder.config.vocabSize) .== forced, logits, MLXArray(-Float.infinity))
                } else if pages < imageURLs.count {
                    logits[eos] = MLXArray(-Float.infinity)
                }
                let token = temperature == 0 ? argMax(logits).item(Int.self)
                    : MLXRandom.categorical(logits / temperature).item(Int.self)
                if token == eos { stopped = true; break }
                output.append(token)
                if output.count.isMultiple(of: 256) { progress?("Decoded \(output.count) tokens") }
                if token == pageID { pages += 1; pageLength = 0 } else { pageLength += 1 }
                if step + 1 < maxTokens {
                    state.append(token)
                    logits = decoder.forward(decoder.embeddings([token]), state: state, cache: cache)
                }
                if Memory.cacheMemory >= 1_073_741_824 { Memory.clearCache() }
            }
            return Result(text: tokenizer.decode(tokens: output, skipSpecialTokens: false),
                          tokensGenerated: output.count, pagesGenerated: pages, reachedTokenLimit: !stopped,
                          loadSeconds: loadSeconds, visionSeconds: visionSeconds, prefillSeconds: prefillSeconds,
                          decodeSeconds: Date().timeIntervalSince(decodeStart))
        }
    }

    private func load(_ path: String) async throws {
        let root = URL(fileURLWithPath: path).standardizedFileURL
        if loadedPath == root.path { return }
        decoder = nil
        vision = nil
        tokenizer = nil
        loadedPath = nil
        let config = try UnlimitedOCRResources.configuration(root: root)
        struct Index: Decodable { let weightMap: [String: String] }
        let parser = JSONDecoder()
        parser.keyDecodingStrategy = .convertFromSnakeCase
        let index = try parser.decode(Index.self, from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        let allowedShards: Set<String> = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        guard Set(index.weightMap.values) == allowedShards else {
            throw UnlimitedOCRError.checkpoint("expected the two original BF16 checkpoint shards")
        }
        var tensors: [String: MLXArray] = [:]
        for shard in allowedShards.sorted() {
            try Task.checkCancellation()
            let values = try MLX.loadArrays(url: root.appendingPathComponent(shard))
            guard Set(values.keys) == Set(index.weightMap.filter { $0.value == shard }.keys) else {
                throw UnlimitedOCRError.checkpoint("tensor index disagrees with \(shard)")
            }
            tensors.merge(values) { first, _ in first }
        }
        try UnlimitedOCRCheckpoint.validate(tensors)
        let loadedTokenizer = try await AutoTokenizer.from(modelFolder: root)
        guard loadedTokenizer.encode(text: "<PAGE>", addSpecialTokens: false) == [128827],
              loadedTokenizer.encode(text: "<image>", addSpecialTokens: false) == [128815] else {
            throw UnlimitedOCRError.checkpoint("tokenizer page/image marker IDs disagree with config")
        }
        decoder = UnlimitedOCRDecoder(config: config, tensors: tensors)
        vision = UnlimitedOCRVision(weights: UnlimitedOCRWeights(tensors: tensors.filter {
            $0.key.hasPrefix("model.sam_model.") || $0.key.hasPrefix("model.vision_model.")
                || $0.key.hasPrefix("model.projector.") || $0.key == "model.image_newline" || $0.key == "model.view_seperator"
        }))
        tokenizer = loadedTokenizer
        loadedPath = root.path
    }
}
#endif
