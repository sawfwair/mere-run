import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func prefillQuantization(for quantization: Gemma4KVCacheQuantization) -> Gemma4KVCacheQuantization {
        guard shouldDeferQuantizationUntilDecode(quantization) else {
            return quantization
        }

        if Gemma4Resources.usesTurboDefaults(modelSpec: modelId)
            && Gemma4Resources.supportsDefaultTurboKVQuantization {
            return Gemma4KVCacheQuantization(
                bits: Gemma4Resources.defaultTurboKVBits,
                scheme: Gemma4Resources.defaultTurboKVQuantizationScheme,
                groupSize: quantization.groupSize,
                quantizedStart: Gemma4Resources.defaultTurboQuantizedKVStart
            )
        }

        return Gemma4KVCacheQuantization()
    }

    nonisolated static func multimodalDecodeBannedTokens(
        imageTokenId: Int?,
        audioTokenId: Int?,
        videoTokenId: Int?,
        boiTokenId: Int?,
        boaTokenId: Int?,
        eoiTokenId: Int?,
        eoaTokenId: Int?,
        excluding excludedTokens: Set<Int>
    ) -> [Int] {
        var seen = Set<Int>()
        var result: [Int] = []
        for token in [
            imageTokenId,
            audioTokenId,
            videoTokenId,
            boiTokenId,
            boaTokenId,
            eoiTokenId,
            eoaTokenId,
        ].compactMap({ $0 }) {
            guard !excludedTokens.contains(token), seen.insert(token).inserted else { continue }
            result.append(token)
        }
        return result
    }

    nonisolated static func noRepeatNgramBannedTokens(
        history: [Int],
        size: Int
    ) -> Set<Int> {
        precondition(size > 0, "size must be positive")
        if size == 1 {
            return Set(history)
        }
        guard history.count >= size - 1 else { return [] }

        let prefix = Array(history.suffix(size - 1))
        guard history.count >= size else { return [] }
        var banned = Set<Int>()
        for start in 0...(history.count - size) {
            let candidatePrefix = Array(history[start..<(start + size - 1)])
            if candidatePrefix == prefix {
                banned.insert(history[start + size - 1])
            }
        }
        return banned
    }

    func shouldDeferQuantizationUntilDecode(_ quantization: Gemma4KVCacheQuantization) -> Bool {
        quantization.isEnabled && quantization.scheme == .polar
    }

    static func cleanedResponse(_ response: String, showThinking: Bool) -> String {
        guard !showThinking else { return response }

        if let finalRange = response.range(
            of: #"(?is)<\|channel>final\s*"#,
            options: .regularExpression
        ) {
            return String(response[finalRange.upperBound...])
                .replacingOccurrences(
                    of: #"(?is)<\|channel>[a-z_]+\s*"#,
                    with: "",
                    options: .regularExpression
                )
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let thoughtCloseRange = response.range(
            of: #"(?is)<\|channel>thought\b.*?<channel\|>\s*"#,
            options: .regularExpression
        ) {
            return String(response[thoughtCloseRange.upperBound...])
                .replacingOccurrences(
                    of: #"(?is)<\|channel>[a-z_]+\s*|<channel\|>"#,
                    with: "",
                    options: .regularExpression
                )
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var cleaned = response.replacingOccurrences(
            of: #"(?is)<\|channel>thought\b.*\z"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*?</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*\\z",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?i)</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"(?is)<\|channel>[a-z_]+\s*|<channel\|>"#,
            with: "",
            options: .regularExpression
        )
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func prepareLayerCachesForDecode(
        _ layerCaches: [Gemma4AttentionCache],
        quantization: Gemma4KVCacheQuantization,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> (caches: [Gemma4AttentionCache], conversionSeconds: Double?) {
        guard shouldDeferQuantizationUntilDecode(quantization) else {
            return (layerCaches, nil)
        }

        progressHandler?(ChatProgress(stage: .encoding, message: "Packing KV cache for decode"))
        let start = Date()
        let converted = try layerCaches.map { cache -> Gemma4AttentionCache in
            guard let reencoded = cache.reencoded(quantization: quantization) else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 could not reencode the prefill KV cache for decode.")
            }
            reencoded.evaluateStorage()
            return reencoded
        }
        return (converted, Date().timeIntervalSince(start))
    }
}
