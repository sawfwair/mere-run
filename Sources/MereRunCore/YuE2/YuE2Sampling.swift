import MLX
import MLXRandom

enum YuE2TokenPhase: String, Sendable {
    case score, semantic

    var range: Range<Int> {
        self == .score ? 0..<YuE2Protocol.endOfText
            : YuE2Protocol.codecOffset..<(YuE2Protocol.codecOffset + YuE2Protocol.codecSize)
    }
    var end: Int { self == .score ? YuE2Protocol.abcEnd : YuE2Protocol.musicEnd }
}

enum YuE2Sampler {
    /// The reachable vocabulary plus EOS. Frequency penalties count repeated
    /// occurrences in the rolling window, as in the checkpoint's sampler.
    static func scores(
        logits: MLXArray, sampling: YuE2Sampling, history: [Int], step: Int,
        phase: YuE2TokenPhase, legacyOff: Bool
    ) -> MLXArray {
        let range = phase.range
        let logits = legacyOff ? logits : logits.asType(.float32)
        var selected = concatenated([logits[range], logits[phase.end..<(phase.end + 1)]])
        if step < sampling.minimumTokens { selected[range.count] = MLXArray(-Float.infinity).asType(selected.dtype) }
        if sampling.repetitionPenalty != 1, !history.isEmpty {
            var frequencies = [Float](repeating: 0, count: range.count + 1)
            for token in history.suffix(sampling.penaltyWindow) where range.contains(token) {
                frequencies[token - range.lowerBound] += 1
            }
            let alpha = pow(sampling.repetitionPenalty, MLXArray(frequencies).asType(selected.dtype))
            selected = which(selected .< 0, selected * alpha, selected / alpha)
        }
        if sampling.temperature == 0 { return selected }
        if sampling.temperature != 1 { selected = selected / sampling.temperature }
        let indices = argSort(-selected)
        var sorted = selected[indices]
        let threshold = sorted[min(sampling.topK, sorted.size) - 1]
        sorted = which(sorted .< threshold, MLXArray(-Float.infinity).asType(sorted.dtype), sorted)
        if sampling.topP < 1 {
            let probabilities = softmax(sorted)
            let removed = (cumsum(probabilities) - probabilities) .> sampling.topP
            let preserve = MLXArray(0..<sorted.size) .< (legacyOff ? 3 : 1)
            sorted = which(logicalAnd(removed, logicalNot(preserve)), MLXArray(-Float.infinity).asType(sorted.dtype), sorted)
        }
        let result = MLXArray.full([selected.size], values: MLXArray(-Float.infinity), dtype: selected.dtype)
        result[indices] = sorted
        return result
    }

    static func generate(
        model: YuE2Model, prefix: [Int], negative: [Int]?, sampling: YuE2Sampling,
        seed: UInt64, phase: YuE2TokenPhase, guidance: Float = 1, legacyOff: Bool = false,
        progress: (Int) -> Void
    ) throws -> (tokens: [Int], truncated: Bool) {
        guard prefix.count + sampling.maximumTokens <= model.configuration.maxPositionEmbeddings,
              negative.map({ $0.count + sampling.maximumTokens <= model.configuration.maxPositionEmbeddings }) ?? true,
              guidance == 1 || negative != nil else {
            throw YuE2Error.invalidRequest("Prefix plus generation budget exceeds context, or CFG lacks a negative prefix.")
        }
        let positiveCache = model.makeCache()
        let negativeCache = model.makeCache()
        let random = MLXRandom.RandomState(seed: seed)
        var conditional = try prefill(model: model, tokens: prefix, cache: positiveCache)
        var unconditional = try negative.map { try prefill(model: model, tokens: $0, cache: negativeCache) }
        var history: [Int] = []
        for step in 0..<sampling.maximumTokens {
            try Task.checkCancellation()
            let logits = unconditional.map { $0 + guidance * (conditional - $0) } ?? conditional
            let scores = scores(logits: logits, sampling: sampling, history: history,
                                step: step, phase: phase, legacyOff: legacyOff)
            guard any(isFinite(scores)).item(Bool.self), !any(isNaN(scores)).item(Bool.self) else {
                throw YuE2Error.invalidAudio("The \(phase.rawValue) sampler has no finite distribution.")
            }
            let index = sampling.temperature == 0
                ? argMax(scores).item(Int.self) : MLXRandom.categorical(scores, key: random).item(Int.self)
            progress(step + 1)
            if index == phase.range.count { return (history, false) }
            let token = index + phase.range.lowerBound
            history.append(token)
            if step + 1 < sampling.maximumTokens {
                conditional = try model.logits([token], cache: positiveCache)
                if unconditional != nil { unconditional = try model.logits([token], cache: negativeCache) }
                MLX.eval(conditional)
                if let unconditional { MLX.eval(unconditional) }
            }
            if step % 64 == 63 { MLX.Memory.clearCache() }
        }
        return (history, true)
    }

    private static func prefill(model: YuE2Model, tokens: [Int], cache: [KVCacheSimple]) throws -> MLXArray {
        guard !tokens.isEmpty else { throw YuE2Error.invalidRequest("Empty prompt prefix.") }
        var logits = MLXArray.zeros([model.configuration.vocabSize])
        for start in stride(from: 0, to: tokens.count, by: 256) {
            try Task.checkCancellation()
            logits = try model.logits(Array(tokens[start..<min(start + 256, tokens.count)]), cache: cache)
            MLX.eval(logits)
        }
        return logits
    }
}
