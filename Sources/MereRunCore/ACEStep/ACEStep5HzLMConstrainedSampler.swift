import Foundation
import MLX

public final class ACEStep5HzLMConstrainedSampler {
    public enum State: Sendable, Hashable {
        case thinkTag
        case newlineAfterThink
        case bpmName
        case bpmValue
        case captionName
        case captionValue
        case durationName
        case durationValue
        case keyscaleName
        case keyscaleValue
        case languageName
        case languageValue
        case timesigName
        case timesigValue
        case thinkEndTag
        case codesGeneration
        case freeTextGeneration
        case completed
    }

    public enum GenerationPhase: Sendable, Hashable {
        case codes
        case understand
    }

    public struct UserMetadata: Sendable, Hashable {
        public var bpm: String?
        public var caption: String?
        public var duration: String?
        public var keyscale: String?
        public var language: String?
        public var timesignature: String?

        public init(
            bpm: String? = nil,
            caption: String? = nil,
            duration: String? = nil,
            keyscale: String? = nil,
            language: String? = nil,
            timesignature: String? = nil
        ) {
            self.bpm = bpm
            self.caption = caption
            self.duration = duration
            self.keyscale = keyscale
            self.language = language
            self.timesignature = timesignature
        }
    }

    public var enabled: Bool
    public var debug: Bool

    public var skipCaption: Bool
    public var skipLanguage: Bool
    public var stopAtReasoning: Bool

    public var generationPhase: GenerationPhase {
        didSet { rebuildTransitions() }
    }

    public var targetDurationSeconds: Float? {
        didSet { targetCodes = targetDurationSeconds.map { max(0, Int(($0 * 5.0).rounded())) } }
    }

    public var userMetadata: UserMetadata {
        didSet { rebuildTransitions() }
    }

    public private(set) var state: State = .thinkTag

    private let tokenizer: ACEStep5HzLMTokenizer
    private let vocabSize: Int

    private let newlineToken: Int32?
    private let periodToken: Int32?
    private let backtickToken: Int32?
    private let eosTokenId: Int32?

    private let audioCodeTokenIds: [Int32]
    private let audioCodeMask: MLXArray
    private let nonAudioCodeMask: MLXArray

    private var bpmPrefixTree: [[Int32]: Set<Int32>] = [:]
    private var durationPrefixTree: [[Int32]: Set<Int32>] = [:]
    private var timesigPrefixTree: [[Int32]: Set<Int32>] = [:]
    private var keyscalePrefixTree: [[Int32]: Set<Int32>] = [:]
    private var languagePrefixTree: [[Int32]: Set<Int32>] = [:]

    private var fixedStrings: [State: String] = [:]
    private var nextState: [State: State] = [:]

    private var positionInFixedString: Int = 0
    private var accumulatedTokenIds: [Int32] = []

    private var userFieldTokenQueue: [Int32] = []

    private var captionTokenCount: Int = 0
    private let maxCaptionTokens: Int = 512

    private var targetCodes: Int?
    private var codesCount: Int = 0

    public init(
        tokenizer: ACEStep5HzLMTokenizer,
        vocabSize: Int,
        enabled: Bool = true,
        debug: Bool = false,
        skipCaption: Bool = false,
        skipLanguage: Bool = false,
        stopAtReasoning: Bool = false,
        generationPhase: GenerationPhase = .codes,
        targetDurationSeconds: Float? = nil,
        userMetadata: UserMetadata = .init()
    ) {
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.enabled = enabled
        self.debug = debug
        self.skipCaption = skipCaption
        self.skipLanguage = skipLanguage
        self.stopAtReasoning = stopAtReasoning
        self.generationPhase = generationPhase
        self.userMetadata = userMetadata
        self.targetCodes = targetDurationSeconds.map { max(0, Int(($0 * 5.0).rounded())) }

        self.newlineToken = tokenizer.encode("\n", addSpecialTokens: false).last.map(Int32.init)
        self.periodToken = tokenizer.encode(".", addSpecialTokens: false).last.map(Int32.init)
        self.backtickToken = tokenizer.encode("`", addSpecialTokens: false).last.map(Int32.init)
        self.eosTokenId = tokenizer.eosTokenId.map(Int32.init)

        self.audioCodeTokenIds = tokenizer.audioCodeTokenIds.map(Int32.init).sorted()

        self.audioCodeMask = Self.makeAudioCodeMask(vocabSize: vocabSize, audioCodeTokenIds: self.audioCodeTokenIds)
        self.nonAudioCodeMask = Self.makeNonAudioCodeMask(
            vocabSize: vocabSize,
            audioCodeTokenIds: self.audioCodeTokenIds,
            eosTokenId: self.eosTokenId
        )

        buildPrefixTrees()
        buildFixedStrings()
        rebuildTransitions()
        reset()
    }

    public func reset() {
        state = .thinkTag
        positionInFixedString = 0
        accumulatedTokenIds = []
        userFieldTokenQueue = []
        captionTokenCount = 0
        codesCount = 0
    }

    public func processLogits(_ logits: MLXArray, tokens: [Int]) -> MLXArray {
        guard enabled else { return logits }
        if state == .completed { return logits }

        // User field injection overrides everything else.
        if let next = userFieldTokenQueue.first {
            return whitelist(logits, allowed: [next])
        }

        switch state {
        case .codesGeneration:
            var out = logits + nonAudioCodeMask.asType(logits.dtype)
            if let targetCodes, let eos = eosTokenId {
                if codesCount < targetCodes {
                    out[MLXArray([eos])] = MLXArray([-Float.infinity]).asType(out.dtype)
                } else {
                    out = whitelist(out, allowed: [eos])
                }
            }
            return out
        case .freeTextGeneration:
            return logits + audioCodeMask.asType(logits.dtype)

        default:
            break
        }

        if let fixed = fixedStrings[state] {
            if state == .thinkEndTag, stopAtReasoning, let eos = eosTokenId {
                return whitelist(logits, allowed: [eos])
            }

            let allowed = allowedTokensForFixedString(fixed)
            return whitelist(logits, allowed: allowed)
        }

        switch state {
        case .bpmValue:
            if let v = userMetadata.bpm {
                maybeBeginUserFieldInjection(value: v)
                if let next = userFieldTokenQueue.first {
                    return whitelist(logits, allowed: [next])
                }
            }
            return whitelist(logits, allowed: allowedFromPrefixTree(bpmPrefixTree))

        case .durationValue:
            if let v = userMetadata.duration {
                maybeBeginUserFieldInjection(value: v)
                if let next = userFieldTokenQueue.first {
                    return whitelist(logits, allowed: [next])
                }
            }
            return whitelist(logits, allowed: allowedFromPrefixTree(durationPrefixTree))

        case .timesigValue:
            if let v = userMetadata.timesignature {
                maybeBeginUserFieldInjection(value: v)
                if let next = userFieldTokenQueue.first {
                    return whitelist(logits, allowed: [next])
                }
            }
            return whitelist(logits, allowed: allowedFromPrefixTree(timesigPrefixTree))

        case .keyscaleValue:
            if let v = userMetadata.keyscale {
                maybeBeginUserFieldInjection(value: v)
                if let next = userFieldTokenQueue.first {
                    return whitelist(logits, allowed: [next])
                }
            }
            return whitelist(logits, allowed: allowedFromPrefixTree(keyscalePrefixTree))

        case .languageValue:
            if let v = userMetadata.language {
                maybeBeginUserFieldInjection(value: v)
                if let next = userFieldTokenQueue.first {
                    return whitelist(logits, allowed: [next])
                }
            }
            return whitelist(logits, allowed: allowedFromPrefixTree(languagePrefixTree))

        case .captionValue:
            if let v = userMetadata.caption {
                maybeBeginUserFieldInjection(value: v)
                if let next = userFieldTokenQueue.first {
                    return whitelist(logits, allowed: [next])
                }
            }

            let out = logits + audioCodeMask.asType(logits.dtype)
            if let backtick = backtickToken {
                out[MLXArray([backtick])] = MLXArray([-Float.infinity]).asType(out.dtype)
            }

            if captionTokenCount >= maxCaptionTokens, let nl = newlineToken {
                return whitelist(out, allowed: [nl])
            }

            if let nl = newlineToken {
                let allowNewline: Bool = {
                    guard let last = tokens.last else { return false }
                    if let periodToken, Int32(last) == periodToken { return true }
                    return false
                }()
                if !allowNewline {
                    out[MLXArray([nl])] = MLXArray([-Float.infinity]).asType(out.dtype)
                }
            }

            return out

        default:
            return logits
        }
    }

    public func update(with token: Int) {
        guard enabled else { return }
        if state == .completed { return }

        if !userFieldTokenQueue.isEmpty {
            userFieldTokenQueue.removeFirst()
        }

        if state == .codesGeneration {
            if tokenizer.audioCodeTokenIdToValue[token] != nil {
                codesCount += 1
            }
            if token == eosTokenId.map(Int.init) {
                state = .completed
            }
            return
        }

        if state == .freeTextGeneration {
            if token == eosTokenId.map(Int.init) {
                state = .completed
            }
            return
        }

        if let fixed = fixedStrings[state] {
            let tokenText = tokenizer.decode(tokens: [token])
            positionInFixedString += tokenText.count
            if positionInFixedString >= fixed.count {
                positionInFixedString = 0
                transitionToNextState()
            }
            return
        }

        if let nl = newlineToken, Int32(token) == nl {
            accumulatedTokenIds = []
            if state == .captionValue {
                captionTokenCount = 0
            }
            transitionToNextState()
            return
        }

        switch state {
        case .captionValue:
            captionTokenCount += 1
        case .bpmValue, .durationValue, .timesigValue, .keyscaleValue, .languageValue:
            accumulatedTokenIds.append(Int32(token))
        default:
            break
        }
    }

    // MARK: - Setup

    private func buildPrefixTrees() {
        bpmPrefixTree = Self.buildNumericPrefixTree(
            tokenizer: tokenizer,
            validValues: (30...300).map(String.init),
            contextPrefixForMatching: "bpm:",
            contextPrefixForTokenization: "bpm: ",
            newlineToken: newlineToken
        )
        durationPrefixTree = Self.buildNumericPrefixTree(
            tokenizer: tokenizer,
            validValues: (10...600).map(String.init),
            contextPrefixForMatching: "duration:",
            contextPrefixForTokenization: "duration: ",
            newlineToken: newlineToken
        )
        timesigPrefixTree = Self.buildNumericPrefixTree(
            tokenizer: tokenizer,
            validValues: ["2", "3", "4", "6"],
            contextPrefixForMatching: "timesignature:",
            contextPrefixForTokenization: "timesignature: ",
            newlineToken: newlineToken
        )
        keyscalePrefixTree = Self.buildKeyscalePrefixTree(tokenizer: tokenizer, newlineToken: newlineToken)
        languagePrefixTree = Self.buildLanguagePrefixTree(tokenizer: tokenizer, newlineToken: newlineToken)
    }

    private func buildFixedStrings() {
        fixedStrings = [
            .thinkTag: "<think>",
            .newlineAfterThink: "\n",
            .bpmName: "bpm:",
            .captionName: "caption:",
            .durationName: "duration:",
            .keyscaleName: "keyscale:",
            .languageName: "language:",
            .timesigName: "timesignature:",
            .thinkEndTag: "</think>",
        ]
    }

    private func rebuildTransitions() {
        nextState = [
            .thinkTag: .newlineAfterThink,
            .newlineAfterThink: .bpmName,
            .bpmName: .bpmValue,
            .captionName: .captionValue,
            .durationName: .durationValue,
            .keyscaleName: .keyscaleValue,
            .languageName: .languageValue,
            .timesigName: .timesigValue,
            .thinkEndTag: generationPhase == .codes ? .codesGeneration : .freeTextGeneration,
            .codesGeneration: .completed,
            .freeTextGeneration: .completed,
        ]

        nextState[.bpmValue] = nextFieldName(after: .bpmValue)
        nextState[.captionValue] = nextFieldName(after: .captionValue)
        nextState[.durationValue] = nextFieldName(after: .durationValue)
        nextState[.keyscaleValue] = nextFieldName(after: .keyscaleValue)
        nextState[.languageValue] = nextFieldName(after: .languageValue)
        nextState[.timesigValue] = .thinkEndTag
    }

    private func nextFieldName(after current: State) -> State {
        let order: [State] = [.bpmValue, .captionValue, .durationValue, .keyscaleValue, .languageValue, .timesigValue]
        guard let idx = order.firstIndex(of: current) else { return .thinkEndTag }

        for i in (idx + 1)..<order.count {
            switch order[i] {
            case .captionValue where skipCaption:
                continue
            case .languageValue where skipLanguage:
                continue
            default:
                return nameState(forValueState: order[i])
            }
        }

        return .thinkEndTag
    }

    private func nameState(forValueState state: State) -> State {
        switch state {
        case .captionValue: return .captionName
        case .durationValue: return .durationName
        case .keyscaleValue: return .keyscaleName
        case .languageValue: return .languageName
        case .timesigValue: return .timesigName
        default: return .thinkEndTag
        }
    }

    // MARK: - Fixed strings

    private func allowedTokensForFixedString(_ fixed: String) -> [Int32] {
        let remaining = String(fixed.dropFirst(positionInFixedString))
        guard !remaining.isEmpty else { return [] }

        // Prefer the longest prefix that encodes to a single token.
        for end in stride(from: remaining.count, through: 1, by: -1) {
            let prefix = String(remaining.prefix(end))
            let ids = tokenizer.encode(prefix, addSpecialTokens: false)
            if ids.count == 1, let id = ids.first {
                return [Int32(id)]
            }
        }

        // Fallback: allow first tokens for a few short prefixes.
        var best: [Int32: Int] = [:]
        for end in 1...min(remaining.count, 20) {
            let prefix = String(remaining.prefix(end))
            let ids = tokenizer.encode(prefix, addSpecialTokens: false)
            guard let first = ids.first else { continue }
            best[Int32(first)] = max(best[Int32(first)] ?? 0, end)
        }

        return best.sorted { $0.value > $1.value }.map(\.key)
    }

    // MARK: - Value constraints

    private func allowedFromPrefixTree(_ tree: [[Int32]: Set<Int32>]) -> [Int32] {
        if let allowed = tree[accumulatedTokenIds], !allowed.isEmpty {
            return Array(allowed)
        }
        if let nl = newlineToken {
            return [nl]
        }
        return []
    }

    private func maybeBeginUserFieldInjection(value: String) {
        guard userFieldTokenQueue.isEmpty, accumulatedTokenIds.isEmpty else { return }
        let text = " \(value)\n"
        let ids = tokenizer.encode(text, addSpecialTokens: false).map(Int32.init)
        if !ids.isEmpty {
            userFieldTokenQueue = ids
        }
    }

    // MARK: - Transitions

    private func transitionToNextState() {
        if let next = nextState[state] {
            state = next
        } else {
            state = .completed
        }
    }

    // MARK: - Logits helpers

    private func whitelist(_ logits: MLXArray, allowed: [Int32]) -> MLXArray {
        let negInf = MLXArray(-Float.infinity, dtype: logits.dtype)

        guard !allowed.isEmpty else {
            return full(logits.shape, values: negInf, dtype: logits.dtype)
        }

        let idx = MLXArray(allowed.map { Int($0) })
        let selected = logits[idx]
        let out = full(logits.shape, values: negInf, dtype: logits.dtype)
        out[idx] = selected
        return out
    }

    private static func makeAudioCodeMask(vocabSize: Int, audioCodeTokenIds: [Int32]) -> MLXArray {
        var values = [Float32](repeating: 0, count: vocabSize)
        for id in audioCodeTokenIds where id >= 0 && Int(id) < vocabSize {
            values[Int(id)] = -Float.infinity
        }
        return MLXArray(values, [vocabSize]).asType(.float32)
    }

    private static func makeNonAudioCodeMask(vocabSize: Int, audioCodeTokenIds: [Int32], eosTokenId: Int32?) -> MLXArray {
        var values = [Float32](repeating: -Float.infinity, count: vocabSize)
        for id in audioCodeTokenIds where id >= 0 && Int(id) < vocabSize {
            values[Int(id)] = 0
        }
        if let eosTokenId, eosTokenId >= 0 && Int(eosTokenId) < vocabSize {
            values[Int(eosTokenId)] = 0
        }
        return MLXArray(values, [vocabSize]).asType(.float32)
    }

    // MARK: - Prefix tree builders

    private static func buildNumericPrefixTree(
        tokenizer: ACEStep5HzLMTokenizer,
        validValues: [String],
        contextPrefixForMatching: String,
        contextPrefixForTokenization: String,
        newlineToken: Int32?
    ) -> [[Int32]: Set<Int32>] {
        var tree: [[Int32]: Set<Int32>] = [:]
        let ctx = tokenizer.encode(contextPrefixForMatching, addSpecialTokens: false).map(Int32.init)

        for value in validValues {
            let fullText = contextPrefixForTokenization + value
            let ids = tokenizer.encode(fullText, addSpecialTokens: false).map(Int32.init)
            guard ids.starts(with: ctx) else { continue }
            let valueTokens = Array(ids.dropFirst(ctx.count))
            addSequence(valueTokens, to: &tree, newlineToken: newlineToken)
        }

        return tree
    }

    private static func buildKeyscalePrefixTree(tokenizer: ACEStep5HzLMTokenizer, newlineToken: Int32?) -> [[Int32]: Set<Int32>] {
        let notes = ["A", "B", "C", "D", "E", "F", "G"]
        let acc = ["", "#", "b", "♯", "♭"]
        let modes = ["major", "minor"]

        var values: [String] = []
        values.reserveCapacity(notes.count * acc.count * modes.count)
        for n in notes {
            for a in acc {
                for m in modes {
                    values.append("\(n)\(a) \(m)")
                }
            }
        }

        return buildNumericPrefixTree(
            tokenizer: tokenizer,
            validValues: values,
            contextPrefixForMatching: "keyscale:",
            contextPrefixForTokenization: "keyscale: ",
            newlineToken: newlineToken
        )
    }

    private static func buildLanguagePrefixTree(tokenizer: ACEStep5HzLMTokenizer, newlineToken: Int32?) -> [[Int32]: Set<Int32>] {
        return buildNumericPrefixTree(
            tokenizer: tokenizer,
            validValues: Self.validLanguages,
            contextPrefixForMatching: "language:",
            contextPrefixForTokenization: "language: ",
            newlineToken: newlineToken
        )
    }

    private static func addSequence(_ seq: [Int32], to tree: inout [[Int32]: Set<Int32>], newlineToken: Int32?) {
        for i in 0...seq.count {
            let prefix = Array(seq.prefix(i))
            if tree[prefix] == nil { tree[prefix] = [] }

            if i < seq.count {
                tree[prefix, default: []].insert(seq[i])
            } else if let newlineToken {
                tree[prefix, default: []].insert(newlineToken)
            }
        }
    }

    private static let validLanguages: [String] = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
        "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
        "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
        "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
        "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
        "unknown",
    ]
}
