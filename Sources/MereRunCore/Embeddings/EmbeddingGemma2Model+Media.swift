import Foundation
import MediaIO
import MereRunTensor
import MLX

extension EmbeddingGemma2Model {
    /// One pooled embedding per ordered record. Only requested media towers are loaded.
    public func embed(inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task = .raw,
                      title: String? = nil, dimensions: Int = 768, maxTokens: Int? = nil) throws -> (embeddings: [[Float]], tokenCounts: [Int]) {
        guard !inputs.isEmpty, EmbeddingGemma2Config.outputDimensions.contains(dimensions),
              title == nil || task == .document else {
            throw EmbeddingGemma2Error.invalidInput("Invalid embedding inputs, dimensions, or document title.")
        }
        let limit = min(maxTokens ?? EmbeddingGemma2Config.contextLength, EmbeddingGemma2Config.contextLength, config.textConfig.maxPositionEmbeddings)
        guard limit >= 2 else { throw EmbeddingGemma2Error.invalidInput("maxTokens must include BOS and EOS.") }
        var vectors: [[Float]] = [], counts: [Int] = []
        for input in inputs {
            guard !input.content.isEmpty else { throw EmbeddingGemma2Error.invalidInput("Input content is empty.") }
            if input.content.allSatisfy({ if case .text = $0 { return true }; return false }) {
                let text = input.content.map { if case .text(let text) = $0 { return text }; return "" }.joined()
                let result = try embed(texts: [text], task: task, title: title, dimensions: dimensions, maxTokens: limit)
                vectors += result.embeddings
                counts += result.tokenCounts
                continue
            }
            if let title { try validateMediaText(title) }
            let prepared = try prepareMedia(input)
            let formatted = task.format(prepared.text, title: title)
            let body = tokenizer.encode(text: formatted, addSpecialTokens: false)
            guard body.count + 2 <= limit else {
                throw EmbeddingGemma2Error.invalidInput("Media record needs \(body.count + 2) tokens, exceeding the \(limit)-token budget. Reduce media or text; media blocks cannot be truncated.")
            }
            let ids = Self.tokenIDs(body: body, config: config.textConfig, limit: limit)
            let replacements = prepared.blocks
            var safe = ids
            for block in replacements { for index in safe.indices where safe[index] == Int32(block.tokenID) { safe[index] = Int32(config.textConfig.padTokenID) } }
            let embeddings = encoder.inputEmbeddings(MLXArray(safe, [1, ids.count]))
            for token in Set(replacements.map(\.tokenID)) {
                let blocks = replacements.filter { $0.tokenID == token }
                let features = concatenated(blocks.map(\.features), axis: 0)
                let locations = ids.indices.filter { ids[$0] == Int32(token) }
                guard locations.count == features.dim(0) else { throw EmbeddingGemma2Error.invalidInput("Reserved media token count does not match supplied media.") }
                embeddings[0, MLXArray(locations.map(Int32.init)), 0...] = features
            }
            // Media activations are no longer needed once the text input is materialized.
            // Release cached buffers before the larger bidirectional text pass.
            MLX.eval(embeddings)
            MLX.Memory.clearCache()
            let mask = MLXArray.ones([1, ids.count], dtype: .int32)
            let output = EmbeddingGemma2TextModel.pool(tokenEmbeddings: encoder(embeddings: embeddings, attentionMask: mask), attentionMask: mask, dimensions: dimensions)
            MLX.eval(output)
            vectors.append(output.asArray(Float.self))
            counts.append(ids.count)
        }
        return (vectors, counts)
    }

    private struct MediaBlock { let tokenID: Int; let features: MLXArray }
    private struct PreparedMedia { let text: String; let blocks: [MediaBlock] }

    private func prepareMedia(_ input: EmbeddingGemma2Input) throws -> PreparedMedia {
        let url = resources.rootURL.appending(path: "processor_config.json")
        guard FileManager.default.fileExists(atPath: url.path), let visionConfig = config.visionConfig else {
            throw EmbeddingGemma2Error.invalidConfiguration("Media requires vision_config and processor_config.json. Refresh the model with model pull --force.")
        }
        try visionConfig.validate()
        let processor = try JSONDecoder().decode(EmbeddingGemma2ProcessorConfig.self, from: Data(contentsOf: url))
        try processor.validate(vision: visionConfig)
        let mediaURLs = input.content.flatMap { part -> [URL] in
            switch part {
            case .text: return []
            case .image(let url), .audio(let url), .video(let url): return [url]
            case .videoFrames(let urls): return urls
            }
        }
        guard mediaURLs.allSatisfy(\.isFileURL) else {
            throw EmbeddingGemma2Error.invalidInput("Media inputs require local file URLs.")
        }
        var text = "", blocks: [MediaBlock] = []
        func token(_ id: Int?) throws -> (Int, String) {
            guard let id, (0..<config.textConfig.vocabSize).contains(id), let value = tokenizer.convertIdToToken(id) else {
                throw EmbeddingGemma2Error.invalidConfiguration("Missing or invalid media token ID.")
            }
            return (id, value)
        }
        func image(_ url: URL, video: Bool) throws {
            let pixels = try EmbeddingGemma2MediaProcessor.image(url, config: video ? processor.videoProcessor : processor.imageProcessor)
            if vision == nil {
                vision = try EmbeddingGemma2VisionModel(config: visionConfig, textHiddenSize: config.textConfig.hiddenSize,
                    tensors: mediaWeights(prefixes: ["vision_tower.", "embed_vision."]), dtype: dtype)
            }
            let (id, marker) = try token(video ? config.videoTokenID : config.imageTokenID)
            let start = try token(config.boiTokenID).1, end = try token(config.eoiTokenID).1
            let features = try vision!(pixels: pixels.pixels, positions: pixels.positions)
            guard features.dim(0) == pixels.softTokens else { throw EmbeddingGemma2Error.invalidConfiguration("Vision soft-token count mismatch.") }
            text += start + String(repeating: marker, count: pixels.softTokens) + end
            MLX.eval(features)
            blocks.append(MediaBlock(tokenID: id, features: features))
        }
        for content in input.content {
            switch content {
            case .text(let value):
                try validateMediaText(value)
                text += value
            case .image(let url): try image(url, video: false)
            case .videoFrames(let frames):
                guard !frames.isEmpty, frames.count <= 32 else { throw EmbeddingGemma2Error.invalidInput("Provide between 1 and 32 video frames.") }
                for frame in frames { try image(frame, video: true) }
            case .video(let url):
                let temporary = FileManager.default.temporaryDirectory.appending(path: "embeddinggemma2-" + UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: temporary) }
                let frames = try MediaVideoIO.sampleFrames(from: url, into: temporary, framesPerSecond: 1, maximumFrames: 32, strategy: .frameRate)
                for frame in frames.frameURLs { try image(frame, video: true) }
            case .audio(let url):
                guard let audioConfig = config.audioConfig else { throw EmbeddingGemma2Error.invalidConfiguration("Missing audio_config.") }
                let prepared = try EmbeddingGemma2MediaProcessor.audio(url, config: processor.featureExtractor)
                if audio == nil {
                    audio = try EmbeddingGemma2AudioModel(config: audioConfig, textHiddenSize: config.textConfig.hiddenSize,
                        tensors: mediaWeights(prefixes: ["audio_tower.", "embed_audio."]), dtype: dtype)
                }
                let features = try audio!(features: prepared.features, validFrames: prepared.validFrames)
                let (id, marker) = try token(config.audioTokenID)
                text += try token(config.boaTokenID).1 + String(repeating: marker, count: prepared.softTokens) + token(config.eoaTokenID).1
                guard features.dim(0) == prepared.softTokens else { throw EmbeddingGemma2Error.invalidConfiguration("Audio soft-token count mismatch.") }
                MLX.eval(features)
                blocks.append(MediaBlock(tokenID: id, features: features))
            }
        }
        return PreparedMedia(text: text, blocks: blocks)
    }

    private func validateMediaText(_ text: String) throws {
        let reserved = [config.imageTokenID, config.videoTokenID, config.audioTokenID,
                        config.boiTokenID, config.eoiTokenID, config.boaTokenID, config.eoaTokenID]
        for id in reserved.compactMap({ $0 }) {
            if let marker = tokenizer.convertIdToToken(id), text.contains(marker) {
                throw EmbeddingGemma2Error.invalidInput("Text content and titles cannot contain reserved media markers.")
            }
        }
    }

    private func mediaWeights(prefixes: [String]) throws -> [String: MLXArray] {
        let arrays = FileManager.default.fileExists(atPath: resources.indexURL.path)
            ? try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: resources.indexURL)
            : try MLX.loadArrays(url: resources.weightsURL)
        return arrays.filter { key, _ in prefixes.contains(where: key.hasPrefix) }
    }
}
