import Foundation
import MLX

/// Incremental validity filter for JSON prefixes — deliberately permissive.
///
/// This is NOT a grammar: it never rejects a valid JSON prefix, but it does allow
/// some invalid ones (e.g. stray punctuation between values). Its job is to keep
/// constrained decoding cheap while making degenerate output — special-token salad,
/// prose, markdown fences, non-JSON characters outside strings — unsampleable.
public struct JSONPrefixScanner: Sendable {
    private enum Container { case object, array }
    private static let literals = ["true", "false", "null"]

    private var stack: [Container] = []
    private var inString = false
    private var escaped = false
    private var started = false
    private var literal = ""
    public private(set) var isComplete = false

    public init() {}

    /// Feeds text into the scanner. Returns false at the first character that cannot
    /// extend a valid JSON prefix; state is undefined after a rejection, so callers
    /// must probe on a copy.
    @discardableResult
    public mutating func accept(_ text: String) -> Bool {
        for character in text where !acceptCharacter(character) {
            return false
        }
        return true
    }

    private mutating func acceptCharacter(_ character: Character) -> Bool {
        if isComplete {
            return character.isWhitespace
        }
        if inString {
            if escaped {
                escaped = false
                return true
            }
            if character == "\\" {
                escaped = true
                return true
            }
            if character == "\"" {
                inString = false
                return true
            }
            // Everything is legal string content — including U+FFFD, which is how a
            // single byte-fallback BPE token (one piece of a multibyte character)
            // decodes in isolation. Rejecting it walls off any string containing
            // é, em-dashes, curly quotes, CJK, …
            return true
        }
        if !started {
            if character.isWhitespace { return true }
            if character == "{" {
                stack.append(.object)
                started = true
                return true
            }
            if character == "[" {
                stack.append(.array)
                started = true
                return true
            }
            return false
        }
        if character.isLetter {
            // A lone e/E is a number exponent (e.g. -2.5e3); any other letters
            // outside strings may only spell `true`, `false`, or `null`.
            if literal.isEmpty, character == "e" || character == "E" { return true }
            literal.append(character)
            return Self.literals.contains { $0.hasPrefix(literal) }
        }
        literal = ""
        if character.isWhitespace { return true }
        switch character {
        case "\"":
            inString = true
            return true
        case "{":
            stack.append(.object)
            return true
        case "[":
            stack.append(.array)
            return true
        case "}":
            guard stack.last == .object else { return false }
            stack.removeLast()
            if stack.isEmpty { isComplete = true }
            return true
        case "]":
            guard stack.last == .array else { return false }
            stack.removeLast()
            if stack.isEmpty { isComplete = true }
            return true
        case ",", ":", "-", "+", ".":
            return true
        case "0"..."9":
            return true
        default:
            return false
        }
    }
}

/// Sample-then-validate constrained decoding: keep the sampled token when it extends
/// a valid JSON prefix, otherwise walk the remaining candidates by descending LOGIT
/// (not post-top-p probability: outside the nucleus the probabilities are all zero
/// and their sort order is arbitrary — resampling from it injects random tokens that
/// poison the context and trigger degeneration). EOS is only valid once the JSON
/// value has closed. Returns nil when no candidate extends the prefix — the caller
/// must stop generation rather than emit garbage.
func jsonConstrainedToken(
    initial: Int,
    logits: MLXArray,
    config: GenerationConfig,
    previousTokens: [Int],
    eosSet: Set<Int>,
    scanner: inout JSONPrefixScanner,
    decode: (Int) -> String
) -> Int? {
    func admit(_ token: Int, into scanner: inout JSONPrefixScanner) -> Bool {
        if eosSet.contains(token) { return scanner.isComplete }
        let piece = decode(token)
        if piece.isEmpty { return true }
        var probe = scanner
        guard probe.accept(piece) else { return false }
        scanner = probe
        return true
    }

    if admit(initial, into: &scanner) { return initial }

    let masked = applyTokenBan(logits: logits, tokens: config.bannedTokens)
    let ascending = argSort(masked, axis: -1)
    let vocabularySize = ascending.dim(-1)
    let maxCandidates = min(64, vocabularySize)
    for offset in 1...maxCandidates {
        let candidate = ascending[vocabularySize - offset].item(Int.self)
        if candidate == initial { continue }
        if admit(candidate, into: &scanner) { return candidate }
    }

    // Nothing in the head of the distribution extends the prefix. Terminating with
    // truncated-but-clean JSON beats emitting salad; the caller's validation and
    // retry loop handle the rest.
    return nil
}
