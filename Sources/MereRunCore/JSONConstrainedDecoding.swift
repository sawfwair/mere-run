import Foundation
import MLX

/// An incremental grammar for an RFC 8259 JSON object.
///
/// Every accepted string is a prefix of a JSON document whose root is an object.
/// The grammar tracks object and array members, strings and escapes, numbers, and
/// literals. It becomes complete only after the root object's closing brace and
/// accepts only JSON whitespace after that point.
public struct JSONObjectPrefixGrammar: Sendable {
    private enum ObjectExpectation: Sendable {
        case keyOrEnd
        case key
        case colon
        case value
        case commaOrEnd
    }

    private enum ArrayExpectation: Sendable {
        case valueOrEnd
        case value
        case commaOrEnd
    }

    private enum Container: Sendable {
        case object(ObjectExpectation)
        case array(ArrayExpectation)
    }

    private enum StringRole: Sendable {
        case key
        case value
    }

    private enum NumberState: Sendable {
        case afterMinus
        case zero
        case integer
        case fractionStart
        case fraction
        case exponentStart
        case exponentSign
        case exponent

        var canTerminate: Bool {
            switch self {
            case .zero, .integer, .fraction, .exponent:
                return true
            case .afterMinus, .fractionStart, .exponentStart, .exponentSign:
                return false
            }
        }
    }

    private enum Literal: Sendable {
        case trueValue
        case falseValue
        case nullValue

        var text: String {
            switch self {
            case .trueValue: "true"
            case .falseValue: "false"
            case .nullValue: "null"
            }
        }
    }

    private enum State: Sendable {
        case structural
        case string(StringRole)
        case stringEscape(StringRole)
        case unicodeEscape(StringRole, remainingDigits: Int)
        case number(NumberState)
        case literal(Literal, matchedScalars: Int)
    }

    private var stack: [Container] = []
    private var state: State = .structural
    private var started = false
    public private(set) var isComplete = false

    public init() {}

    /// Feeds decoded token text into the grammar. Returns false at the first
    /// scalar that cannot extend a valid JSON-object prefix. State is undefined
    /// after rejection, so constrained decoders must probe on a copy.
    @discardableResult
    public mutating func accept(_ text: String) -> Bool {
        for scalar in text.unicodeScalars where !accept(scalar) {
            return false
        }
        return true
    }

    private mutating func accept(_ scalar: Unicode.Scalar) -> Bool {
        if isComplete {
            return Self.isJSONWhitespace(scalar)
        }

        switch state {
        case .structural:
            return acceptStructural(scalar)
        case .string(let role):
            return acceptString(scalar, role: role)
        case .stringEscape(let role):
            return acceptStringEscape(scalar, role: role)
        case .unicodeEscape(let role, let remainingDigits):
            guard Self.isHexDigit(scalar) else { return false }
            if remainingDigits == 1 {
                state = .string(role)
            } else {
                state = .unicodeEscape(role, remainingDigits: remainingDigits - 1)
            }
            return true
        case .number(let numberState):
            return acceptNumber(scalar, numberState: numberState)
        case .literal(let literal, let matchedScalars):
            let target = Array(literal.text.unicodeScalars)
            guard matchedScalars < target.count, scalar == target[matchedScalars] else {
                return false
            }
            let nextMatch = matchedScalars + 1
            if nextMatch == target.count {
                state = .structural
                return completeValue()
            }
            state = .literal(literal, matchedScalars: nextMatch)
            return true
        }
    }

    private mutating func acceptStructural(_ scalar: Unicode.Scalar) -> Bool {
        if !started {
            if Self.isJSONWhitespace(scalar) { return true }
            guard scalar == "{" else { return false }
            started = true
            stack.append(.object(.keyOrEnd))
            return true
        }

        guard let container = stack.last else { return false }
        switch container {
        case .object(let expectation):
            return acceptObjectStructural(scalar, expectation: expectation)
        case .array(let expectation):
            return acceptArrayStructural(scalar, expectation: expectation)
        }
    }

    private mutating func acceptObjectStructural(
        _ scalar: Unicode.Scalar,
        expectation: ObjectExpectation
    ) -> Bool {
        if Self.isJSONWhitespace(scalar) { return true }

        switch expectation {
        case .keyOrEnd:
            if scalar == "}" { return closeObject() }
            guard scalar == "\"" else { return false }
            state = .string(.key)
            return true
        case .key:
            guard scalar == "\"" else { return false }
            state = .string(.key)
            return true
        case .colon:
            guard scalar == ":" else { return false }
            stack[stack.count - 1] = .object(.value)
            return true
        case .value:
            return beginValue(with: scalar)
        case .commaOrEnd:
            if scalar == "," {
                stack[stack.count - 1] = .object(.key)
                return true
            }
            if scalar == "}" { return closeObject() }
            return false
        }
    }

    private mutating func acceptArrayStructural(
        _ scalar: Unicode.Scalar,
        expectation: ArrayExpectation
    ) -> Bool {
        if Self.isJSONWhitespace(scalar) { return true }

        switch expectation {
        case .valueOrEnd:
            if scalar == "]" { return closeArray() }
            return beginValue(with: scalar)
        case .value:
            return beginValue(with: scalar)
        case .commaOrEnd:
            if scalar == "," {
                stack[stack.count - 1] = .array(.value)
                return true
            }
            if scalar == "]" { return closeArray() }
            return false
        }
    }

    private mutating func beginValue(with scalar: Unicode.Scalar) -> Bool {
        switch scalar {
        case "\"":
            state = .string(.value)
        case "{":
            stack.append(.object(.keyOrEnd))
        case "[":
            stack.append(.array(.valueOrEnd))
        case "t":
            state = .literal(.trueValue, matchedScalars: 1)
        case "f":
            state = .literal(.falseValue, matchedScalars: 1)
        case "n":
            state = .literal(.nullValue, matchedScalars: 1)
        case "-":
            state = .number(.afterMinus)
        case "0":
            state = .number(.zero)
        case "1"..."9":
            state = .number(.integer)
        default:
            return false
        }
        return true
    }

    private mutating func acceptString(_ scalar: Unicode.Scalar, role: StringRole) -> Bool {
        if scalar == "\"" {
            state = .structural
            switch role {
            case .key:
                guard case .object = stack.last else { return false }
                stack[stack.count - 1] = .object(.colon)
                return true
            case .value:
                return completeValue()
            }
        }
        if scalar == "\\" {
            state = .stringEscape(role)
            return true
        }
        return scalar.value >= 0x20
    }

    private mutating func acceptStringEscape(_ scalar: Unicode.Scalar, role: StringRole) -> Bool {
        switch scalar {
        case "\"", "\\", "/", "b", "f", "n", "r", "t":
            state = .string(role)
            return true
        case "u":
            state = .unicodeEscape(role, remainingDigits: 4)
            return true
        default:
            return false
        }
    }

    private mutating func acceptNumber(
        _ scalar: Unicode.Scalar,
        numberState: NumberState
    ) -> Bool {
        switch numberState {
        case .afterMinus:
            if scalar == "0" {
                state = .number(.zero)
                return true
            }
            guard Self.isOneToNine(scalar) else { return false }
            state = .number(.integer)
            return true
        case .zero:
            if scalar == "." {
                state = .number(.fractionStart)
                return true
            }
            if scalar == "e" || scalar == "E" {
                state = .number(.exponentStart)
                return true
            }
        case .integer:
            if Self.isDigit(scalar) { return true }
            if scalar == "." {
                state = .number(.fractionStart)
                return true
            }
            if scalar == "e" || scalar == "E" {
                state = .number(.exponentStart)
                return true
            }
        case .fractionStart:
            guard Self.isDigit(scalar) else { return false }
            state = .number(.fraction)
            return true
        case .fraction:
            if Self.isDigit(scalar) { return true }
            if scalar == "e" || scalar == "E" {
                state = .number(.exponentStart)
                return true
            }
        case .exponentStart:
            if scalar == "+" || scalar == "-" {
                state = .number(.exponentSign)
                return true
            }
            guard Self.isDigit(scalar) else { return false }
            state = .number(.exponent)
            return true
        case .exponentSign:
            guard Self.isDigit(scalar) else { return false }
            state = .number(.exponent)
            return true
        case .exponent:
            if Self.isDigit(scalar) { return true }
        }

        guard numberState.canTerminate else { return false }
        state = .structural
        guard completeValue() else { return false }
        return acceptStructural(scalar)
    }

    private mutating func completeValue() -> Bool {
        guard let container = stack.last else { return false }
        switch container {
        case .object(.value):
            stack[stack.count - 1] = .object(.commaOrEnd)
        case .array(.valueOrEnd), .array(.value):
            stack[stack.count - 1] = .array(.commaOrEnd)
        default:
            return false
        }
        return true
    }

    private mutating func closeObject() -> Bool {
        guard case .object = stack.last else { return false }
        stack.removeLast()
        return finishContainerValue()
    }

    private mutating func closeArray() -> Bool {
        guard case .array = stack.last else { return false }
        stack.removeLast()
        return finishContainerValue()
    }

    private mutating func finishContainerValue() -> Bool {
        if stack.isEmpty {
            isComplete = true
            return true
        }
        return completeValue()
    }

    private static func isJSONWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        scalar == " " || scalar == "\t" || scalar == "\n" || scalar == "\r"
    }

    private static func isDigit(_ scalar: Unicode.Scalar) -> Bool {
        ("0"..."9").contains(scalar)
    }

    private static func isOneToNine(_ scalar: Unicode.Scalar) -> Bool {
        ("1"..."9").contains(scalar)
    }

    private static func isHexDigit(_ scalar: Unicode.Scalar) -> Bool {
        ("0"..."9").contains(scalar)
            || ("a"..."f").contains(scalar)
            || ("A"..."F").contains(scalar)
    }
}

@available(*, deprecated, renamed: "JSONObjectPrefixGrammar")
public typealias JSONPrefixScanner = JSONObjectPrefixGrammar

/// Sample-then-validate constrained decoding. The sampled token is kept when it
/// extends the JSON-object prefix. Otherwise candidates are checked in descending
/// raw-logit order. EOS is admitted only after the root object has closed.
func jsonConstrainedToken(
    initial: Int,
    logits: MLXArray,
    config: GenerationConfig,
    eosSet: Set<Int>,
    grammar: inout JSONObjectPrefixGrammar,
    decode: (Int) -> String
) -> Int? {
    func admit(_ token: Int, into grammar: inout JSONObjectPrefixGrammar) -> Bool {
        if eosSet.contains(token) { return grammar.isComplete }
        let piece = decode(token)
        guard !piece.isEmpty else { return false }
        var probe = grammar
        guard probe.accept(piece) else { return false }
        grammar = probe
        return true
    }

    if admit(initial, into: &grammar) { return initial }

    let bannedTokens = Set(config.bannedTokens)
    let masked = applyTokenBan(logits: logits, tokens: config.bannedTokens)
    let ascending = argSort(masked, axis: -1).asArray(Int32.self)
    for rawCandidate in ascending.reversed() {
        let candidate = Int(rawCandidate)
        if candidate == initial || bannedTokens.contains(candidate) { continue }
        if admit(candidate, into: &grammar) { return candidate }
    }

    return nil
}
