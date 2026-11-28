import MereRunCore
#if !os(iOS)
import Foundation
import MLX
import MereRunQwenModel

final class ClairicSlotGrammar {
    struct Candidates {
        let indices: MLXArray
        let nullOffset: Int?
        let ids: [Int]
    }

    let grammar: ClairicGrammar
    private var candidates: [ClairicGrammar.State: Candidates] = [:]

    init(_ spec: ClairicField) throws {
        grammar = try ClairicGrammar(spec.grammar())
    }

    func allowedTokens(at state: ClairicGrammar.State, using tokenizer: ClairicTokenizer) throws -> Candidates {
        if let cached = candidates[state] { return cached }
        try Task.checkCancellation()
        var ids = grammar.leadingBytes(for: state).flatMap { byte in
            tokenizer.buckets[Int(byte)].filter { grammar.advance(state, bytes: tokenizer.bytes($0)) != nil }
        }
        if grammar.accepts(state) { ids.append(tokenizer.eos) }
        ids.sort()
        guard !ids.isEmpty else { throw ClairicError.inference("Clairic field grammar has no admissible tokens.") }
        let result = Candidates(indices: MLXArray(ids.map(Int32.init)), nullOffset: ids.firstIndex(of: tokenizer.null), ids: ids)
        candidates[state] = result
        return result
    }

}

final class ClairicDecodeRow {
    let slot: ClairicPrompt.Slot
    var caches: [Q35LayerCache?]
    var logits: MLXArray
    var nextPosition: Int
    let usesMRoPE: Bool

    private let slotGrammar: ClairicSlotGrammar
    private var state: ClairicGrammar.State
    private var output: [UInt8] = []
    private var totalLogProbability: Float = 0
    private var pNull: Float = 0
    private var complete = false
    private(set) var tokens = 0

    init(slot: ClairicPrompt.Slot, caches: [Q35LayerCache?], logits: MLXArray, grammar: ClairicSlotGrammar,
         nextPosition: Int, usesMRoPE: Bool) {
        self.slot = slot
        self.caches = caches
        self.logits = logits
        self.nextPosition = nextPosition
        self.usesMRoPE = usesMRoPE
        slotGrammar = grammar
        state = grammar.grammar.start
    }

    func step(tokenizer: ClairicTokenizer, isFirst: Bool) throws -> Int? {
        let candidates = try slotGrammar.allowedTokens(at: state, using: tokenizer)
        let scores = logits.take(candidates.indices).asType(.float32)
        let normalizer = scores.logSumExp()
        let index = scores.argMax()
        let logProbability = scores.max() - normalizer
        let nullLogProbability = isFirst ? candidates.nullOffset.map { scores[$0] - normalizer } : nil
        if let nullLogProbability { MLX.eval(index, logProbability, nullLogProbability) }
        else { MLX.eval(index, logProbability) }
        let score = logProbability.item(Float.self)
        guard score.isFinite else { throw ClairicError.inference("Clairic generated a nonfinite field probability.") }
        let token = candidates.ids[index.item(Int.self)]
        if let nullLogProbability {
            pNull = min(1, exp(nullLogProbability.item(Float.self)))
        }
        tokens += 1
        totalLogProbability += score
        if token == tokenizer.eos {
            complete = true
            return nil
        }
        let bytes = tokenizer.bytes(token)
        guard let next = slotGrammar.grammar.advance(state, bytes: bytes) else {
            throw ClairicError.inference("Clairic decoder emitted a token outside the field grammar.")
        }
        state = next
        output += bytes
        return token
    }

    func answer(source: String, hasMedia: Bool) -> ClairicResponse.Answer {
        let decoded = complete ? try? JSONDecoder().decode(ChatJSONValue.self, from: Data(output)) : nil
        let value = decoded.flatMap(slot.spec.normalized).flatMap { ClairicPrintedTemporal.normalize($0, as: slot.temporal) }
        let canGround = !hasMedia && slot.spec.knowledge != true
        let span = canGround ? decoded.flatMap { Self.uniqueSpan(of: $0, in: source) } : nil
        let requiresSource = canGround && (slot.spec.source_only == true || slot.spec.type == .text)
        let valid = value != nil && (!requiresSource || value == .null || span != nil || Self.occurs(decoded, in: source))
        return .init(value: valid ? value ?? .null : .null, valid: valid, complete: complete,
                     confidence: valid ? min(1, exp(totalLogProbability)) : 0, p_null: pNull,
                     span: valid ? span ?? .null : .null)
    }

    private static func occurs(_ value: ChatJSONValue?, in source: String) -> Bool {
        switch value {
        case .string(let text): !text.isEmpty && source.range(of: text, options: .literal) != nil
        case .number(let number): !numericMatches(number, in: source).isEmpty
        default: false
        }
    }

    static func uniqueSpan(of value: ChatJSONValue, in source: String) -> ChatJSONValue? {
        let matches: [Range<String.Index>]
        switch value {
        case .string(let text) where !text.isEmpty:
            guard let first = source.range(of: text, options: .literal) else { return nil }
            let nextStart = source.unicodeScalars.index(after: first.lowerBound)
            guard source.range(of: text, options: .literal, range: nextStart..<source.endIndex) == nil else { return nil }
            matches = [first]
        case .number(let number):
            matches = numericMatches(number, in: source)
        default: return nil
        }
        guard matches.count == 1, let match = matches.first else { return nil }
        let scalars = source.unicodeScalars
        return .array([match.lowerBound, match.upperBound].map { .number(Double(scalars.distance(from: scalars.startIndex, to: $0))) })
    }

    private static func numericMatches(_ number: Double, in source: String) -> [Range<String.Index>] {
        guard let pattern = try? NSRegularExpression(pattern: "(?<![\\p{L}\\d.])[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\\.[0-9]+)?(?![\\p{L}\\d])") else {
            return []
        }
        return pattern.matches(in: source, range: NSRange(source.startIndex..., in: source)).compactMap { match in
            guard let range = Range(match.range, in: source),
                  Double(source[range].replacingOccurrences(of: ",", with: "")) == number else { return nil }
            return range
        }
    }
}
#endif
