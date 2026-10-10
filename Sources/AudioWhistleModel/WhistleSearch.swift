import Foundation
import MLX

extension WhistleModel {
    private struct Hypothesis {
        var tokens: [Int]
        var score: Float
        var cache: WhistleDecoderCache
        var ended: Bool
        var rank: Float { score / pow(Float(max(1, tokens.count)), 0.6) }
    }

    /// Deterministic beam search. Each branch owns its value-semantic KV/conv history.
    package func search(
        audio: WhistleAudioContext, language: Int, maxTokens: Int, beams: Int, depth: Int,
        keywords: [[Int]] = [], progress: ((Int) -> Void)? = nil
    ) throws -> [Int] {
        guard (1...8).contains(beams), (1...318).contains(maxTokens), (0..<7).contains(language) else {
            throw WhistleError.invalid("invalid decoder search controls")
        }
        var initial = WhistleDecoderCache()
        _ = try decode(token: 2, audio: audio, cache: &initial, depth: depth)
        var hypotheses = [Hypothesis(tokens: [], score: 0, cache: initial, ended: false)]
        for _ in 0..<maxTokens {
            try Task.checkCancellation()
            var candidates: [Hypothesis] = []
            for hypothesis in hypotheses {
                if hypothesis.ended { candidates.append(hypothesis); continue }
                var cache = hypothesis.cache
                let token = hypothesis.tokens.last ?? (8192 + language)
                let logits = try decode(token: token, audio: audio, cache: &cache, depth: depth)[0..<8192]
                var bias = Self.keywordBias(history: hypothesis.tokens, sequences: keywords)
                bias[0] = -1e9
                bias[2] = -1e9
                let adjusted = logits + MLXArray(bias)
                let probabilities = adjusted - logSumExp(adjusted)
                let choices = argSort(-probabilities)[0..<beams].asArray(Int32.self)
                let scores = take(probabilities, MLXArray(choices)).asArray(Float.self)
                for (index, choice) in choices.enumerated() {
                    let token = Int(choice)
                    candidates.append(Hypothesis(tokens: hypothesis.tokens + [token], score: hypothesis.score + scores[index],
                                                 cache: cache, ended: token == 1))
                }
            }
            candidates.sort {
                if $0.rank == $1.rank { return $0.tokens.lexicographicallyPrecedes($1.tokens) }
                return $0.rank > $1.rank
            }
            hypotheses = Array(candidates.prefix(beams))
            progress?(hypotheses[0].tokens.count)
            if hypotheses.allSatisfy(\.ended) { break }
        }
        return hypotheses[0].tokens.filter { $0 != 1 }
    }

    package static func keywordBias(history: [Int], sequences: [[Int]]) -> [Float] {
        var bias = Array(repeating: Float(0), count: 8192)
        for sequence in sequences where !sequence.isEmpty {
            bias[sequence[0]] = max(bias[sequence[0]], 2)
            for length in 1..<sequence.count where history.count >= length {
                if history.suffix(length).elementsEqual(sequence.prefix(length)) {
                    bias[sequence[length]] = max(bias[sequence[length]], 5)
                }
            }
        }
        return bias
    }

    /// Replay only the winning beam for alignment, avoiding a cross-attention history per beam.
    package func alignment(audio: WhistleAudioContext, language: Int, tokens: [Int], depth: Int) throws -> [[Float]] {
        var cache = WhistleDecoderCache()
        _ = try decode(token: 2, audio: audio, cache: &cache, depth: depth)
        var rows: [[Float]] = []
        for token in [8192 + language] + Array(tokens.prefix(317)) {
            _ = try decode(token: token, audio: audio, cache: &cache, depth: depth, alignment: true)
            rows.append(cache.alignment!.asArray(Float.self))
        }
        return rows
    }
}
