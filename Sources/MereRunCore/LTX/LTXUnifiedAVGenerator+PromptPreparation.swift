import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
    func loadTextEncoderIfNeeded() async throws {
        guard textEncoder == nil else { return }
        guard let loadedRoot else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let text = LTXGemmaTextEncoder()
        try await text.load(
            modelRoot: loadedRoot,
            textEncoderRoot: loadedForLTX25 ? nil : try resolveLTX23TextEncoderRoot(modelRoot: loadedRoot),
            dtype: loadedDType,
            loadConnectorWeights: true
        )
        textEncoder = text
    }

    func cachedPromptEmbeddings(
        prompt: String,
        maxLength: Int
    ) async throws -> (embeddings: LTXCachedPromptEmbeddings, cacheHit: Bool) {
        let key = LTXPromptEmbeddingCacheKey(prompt: prompt, maxLength: maxLength)
        if let cached = promptEmbeddingCache.value(for: key) {
            return (cached, true)
        }
        guard let textEncoder else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let encoding = try await textEncoder.encode(prompt: prompt, maxLength: maxLength)
        let cached = LTXCachedPromptEmbeddings(
            video: encoding.videoEmbeddings,
            audio: encoding.audioEmbeddings
        )
        if let audio = cached.audio {
            MLX.eval(cached.video, audio)
        } else {
            MLX.eval(cached.video)
        }
        promptEmbeddingCache.insert(cached, for: key)
        return (cached, false)
    }
}
