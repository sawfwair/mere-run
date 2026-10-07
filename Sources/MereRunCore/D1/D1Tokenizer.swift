// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import Hub
import Tokenizers
import MereRunD1Model

struct D1TokenSequence {
    let ids: [Int]
    let markers: [Int]
    let readout: [[Int]]
    let dropped: Int
}

struct D1Tokenizer {
    let encode: (String) -> [Int]
    static func load(root: URL) throws -> D1Tokenizer {
        let config = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer_config.json"))
        let data = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer.json"))
        guard data.model.type.string() == "BPE" else { throw D1Error.invalid("D1 requires its checkpoint-native BPE tokenizer.") }
        guard let stages = data.preTokenizer.pretokenizers.array(), stages.count == 2,
              stages[0].type.string() == "Split", stages[0].behavior.string() == "Isolated",
              stages[0].invert.boolean() == false, let pattern = stages[0].pattern.Regex.string(),
              stages[1].type.string() == "ByteLevel", stages[1].useRegex.boolean() == false,
              stages[1].addPrefixSpace.boolean() == false else {
            throw D1Error.invalid("Unsupported D1 pretokenizer configuration.")
        }
        let added = data.addedTokens.array(or: [])
        guard added.allSatisfy({ !$0.lstrip.boolean(or: false) && !$0.rstrip.boolean(or: false) }) else {
            throw D1Error.invalid("Unsupported D1 added-token whitespace policy.")
        }
        let special = added.compactMap { $0.content.string() }.sorted { $0.utf16.count > $1.utf16.count }
        guard !special.isEmpty else { throw D1Error.invalid("D1 tokenizer requires its added tokens.") }
        let specialRegex = try NSRegularExpression(pattern: special.map(NSRegularExpression.escapedPattern).joined(separator: "|"))
        let ordinaryRegex = try NSRegularExpression(pattern: pattern)
        var native = config.dictionary(or: [:]); native["tokenizer_class"] = Config("Qwen2Tokenizer")
        var nativeData = data.dictionary(or: [:]); nativeData["pre_tokenizer"] = stages[1]
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(native), tokenizerData: Config(nativeData))
        // Foundation String.range(regex:) changes the greedy newline split used by D1-3B.
        // Match the complete ordinary section once with NSRegularExpression instead,
        // then let the native ByteLevel/BPE encode each reference pretoken separately.
        return D1Tokenizer(encode: { text in
            let string = text as NSString
            func ordinary(_ section: String) -> [Int] {
                let input = section as NSString
                let matches = ordinaryRegex.matches(in: section, range: NSRange(location: 0, length: input.length))
                var ids: [Int] = [], cursor = 0
                for match in matches {
                    if match.range.location > cursor {
                        ids += tokenizer.encode(text: input.substring(with: NSRange(location: cursor, length: match.range.location - cursor)), addSpecialTokens: false)
                    }
                    ids += tokenizer.encode(text: input.substring(with: match.range), addSpecialTokens: false)
                    cursor = NSMaxRange(match.range)
                }
                if cursor < input.length { ids += tokenizer.encode(text: input.substring(from: cursor), addSpecialTokens: false) }
                return ids
            }
            var ids: [Int] = [], cursor = 0
            for match in specialRegex.matches(in: text, range: NSRange(location: 0, length: string.length)) {
                ids += ordinary(string.substring(with: NSRange(location: cursor, length: match.range.location - cursor)))
                ids += tokenizer.encode(text: string.substring(with: match.range), addSpecialTokens: false)
                cursor = NSMaxRange(match.range)
            }
            ids += ordinary(string.substring(from: cursor))
            return ids
        })
    }
    func token(_ text: String) throws -> Int {
        let ids = encode(text)
        guard ids.count == 1 else { throw D1Error.invalid("D1 tokenizer requires a single token for \(text).") }
        return ids[0]
    }
    func aliases(_ labels: [String]) throws -> [(String, Int)] {
        let labels = labels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let letters = labels.allSatisfy { $0.unicodeScalars.count == 1 && $0.unicodeScalars.allSatisfy(CharacterSet.letters.contains) }
        let codes = letters ? labels : labels.count <= 26 ? (0..<labels.count).map { String(UnicodeScalar(65 + $0) ?? "A") }
            : (0..<labels.count).map { String(format: "%02d", $0) }
        let capitals = (65...90).map { String(UnicodeScalar($0) ?? "A") }
        let pool = capitals + (0..<100).map { String(format: "%02d", $0) } + (97...122).map { String(UnicodeScalar($0) ?? "a") }
            + (0..<200).map { "#\($0)" } + capitals.flatMap { a in capitals.map { a + $0 } }
        var used = Set<Int>()
        return try codes.map { code in
            for raw in [code] + pool {
                let ids = encode(raw)
                if ids.count == 1, used.insert(ids[0]).inserted { return (raw, ids[0]) }
            }
            throw D1Error.invalid("D1 tokenizer has no distinct single-token aliases for these choices.")
        }
    }
    func causal(_ request: D1DecisionRequest, question: D1Question, mediaText: String, maxLength: Int) throws -> D1TokenSequence {
        let body: String
        if case .null = request.state { body = "" }
        else { body = try D1JSON.render(request.state, indent: 2, stringAsText: true) + "\n\n\nQUESTION:\n" }
        let prefix = "<|startoftext|><|im_start|>user\n" + mediaText + body
        var groups: [[Int]] = []
        let questionText: String
        func ids(_ forms: [String]) -> [Int] { Array(Set(forms.map(encode).filter { $0.count == 1 }.map { $0[0] })).sorted() }
        switch question.type {
        case .choice:
            let codes = try aliases(question.options.map(\.id))
            let lines = try zip(question.options, codes).map { option, code -> String in
                let desc = option.description.string
                if case .null = option.description {} else if desc == nil { throw D1Error.invalid("D1-3B choice descriptions must be strings or null.") }
                return code.0 + " " + ((desc?.isEmpty == false ? desc : nil) ?? option.id.replacingOccurrences(of: "_", with: " "))
            }
            groups = codes.map { code in [code.1] + ids([" " + code.0]).filter { $0 != code.1 } }
            questionText = question.instructions + "\n\nOptions:\n" + lines.joined(separator: "\n") + "\n\nReply with the option code only."
        case .noul:
            groups = [ids(["yes", "Yes", "YES"]), ids(["no", "No", "NO"])]
            var extra = ""
            if let criteria = question.criteria, criteria.isTruthy {
                extra = "\nYes: " + (try D1JSON.pythonText(criteria["true"] ?? .null)) + "\nNo: " + (try D1JSON.pythonText(criteria["false"] ?? .null))
            }
            questionText = question.instructions + extra + "\n\nReply with yes or no only."
        case .score:
            groups = question.options.indices.map { ids([String($0)]) }
            let legend = try question.options.enumerated().map { "\($0.offset) " + (try D1JSON.pythonText($0.element.description)) }.joined(separator: "\n")
            questionText = question.instructions + "\n\n" + legend + "\n\nReply with a single digit 0-\(question.options.count - 1) only."
        }
        guard groups.allSatisfy({ !$0.isEmpty }) else { throw D1Error.invalid("D1 tokenizer cannot score these answer tokens.") }
        let sequence = encode(prefix + questionText + "<|im_end|>\n<|im_start|>assistant\n")
        guard sequence.count <= maxLength else { throw D1Error.invalid("D1-3B prompt requires \(sequence.count) tokens; maximum is \(maxLength).") }
        return D1TokenSequence(ids: sequence, markers: [], readout: groups, dropped: 0)
    }
    func omni(_ request: D1DecisionRequest, question: D1Question, config: D1Configuration, mediaTokens: Int) throws -> D1TokenSequence {
        let audio = request.audio != nil, media = mediaTokens > 0
        let limit = media ? audio ? config.audio_text_length ?? 15_360 : config.image_text_length ?? 896 : config.maxLength
        let context = min(request.maxTokens ?? config.maxLength, config.maxLength)
        let maxLength = min(limit, context - mediaTokens)
        guard maxLength >= 64 else { throw D1Error.invalid("D1 media leave fewer than 64 text positions.") }
        func enc(_ text: String) -> [Int] {
            let escaped = text.replacingOccurrences(of: "<\\|([A-Za-z0-9_]+)\\|>", with: "<¦$1¦>", options: .regularExpression)
            return encode(escaped)
        }
        let options: [String]
        switch question.type {
        case .choice:
            options = try question.options.enumerated().map { index, option in
                let desc = try D1JSON.render(option.description, stringAsText: true)
                let empty = D1JSON.emptyCriterion(option.description)
                if audio { return String(format: "option_%03d: ", index) + (empty ? option.id : desc) }
                return empty ? option.id : option.id + ": " + desc
            }
        case .score:
            options = try question.options.enumerated().map { "level \($0.offset): " + (try D1JSON.render($0.element.description, stringAsText: true)) }
        case .noul:
            options = try question.options.reversed().map { option in
                let defaults = option.id == "false" ? "no, the statement does not hold" : "yes, the statement holds"
                let mediaDefault = option.id == "false" ? "no" : "yes"
                let desc = try D1JSON.render(option.description, stringAsText: true)
                let useMediaDefault = media && !(question.criteria?.isTruthy ?? false)
                return option.id + ": " + (audio ? mediaDefault : !D1JSON.emptyCriterion(option.description) ? desc : useMediaDefault ? mediaDefault : defaults)
            }
        }
        let budget = max(96, min(options.count * 24 + 32, maxLength / 2))
        let per = max(2, (budget - 3 * options.count) / options.count)
        var questionIDs = Array(([try token("<|reserved_8|>")] + enc(question.instructions)).prefix(max(16, budget)))
        var markers: [Int] = []
        for option in options {
            markers.append(questionIDs.count + 1)
            questionIDs += [try token("<|reserved_9|>"), try token("<|mask|>")] + Array(enc(" " + option).prefix(per)) + [try token("<|reserved_10|>")]
        }
        questionIDs += [try token("<|reserved_11|>")]
        let state: ClefJSON
        if case .null = request.state { state = audio ? .object([]) : .string("") } else { state = request.state }
        let stateIDs = enc(try D1JSON.render(state, stringAsText: true))
        let room = max(0, maxLength - questionIDs.count - 2)
        let kept = Array(stateIDs.prefix(room))
        let sequence = Array(([config.bos_token_id, try token("<|reserved_7|>")] + kept + questionIDs).prefix(maxLength))
        markers = markers.map { $0 + kept.count + 2 }
        guard markers.allSatisfy({ $0 < sequence.count }) else { throw D1Error.invalid("D1 options do not fit in the context.") }
        return D1TokenSequence(ids: sequence, markers: markers, readout: [], dropped: stateIDs.count - kept.count)
    }
}

/// Ordered Python-compatible JSON spacing. Option and state key order are part of the checkpoint contract.
enum D1JSON {
    static func render(_ value: ClefJSON, indent: Int? = nil, depth: Int = 0, stringAsText: Bool = false) throws -> String {
        if stringAsText, let text = value.string { return text }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.withoutEscapingSlashes]
        func quote(_ text: String) throws -> String { String(decoding: try encoder.encode(text), as: UTF8.self) }
        let lead = indent.map { String(repeating: " ", count: $0 * (depth + 1)) } ?? ""
        let close = indent.map { String(repeating: " ", count: $0 * depth) } ?? ""
        func collection(_ parts: [String], open: String, end: String) -> String {
            guard !parts.isEmpty else { return open + end }
            if indent != nil { return open + "\n" + lead + parts.joined(separator: ",\n" + lead) + "\n" + close + end }
            return open + parts.joined(separator: ", ") + end
        }
        switch value {
        case .string(let text): return try quote(text)
        case .number, .bool, .null: return try value.canonical()
        case .array(let values): return collection(try values.map { try render($0, indent: indent, depth: depth + 1) }, open: "[", end: "]")
        case .object(let fields):
            return collection(try fields.map { try quote($0.key) + ": " + render($0.value, indent: indent, depth: depth + 1) }, open: "{", end: "}")
        }
    }
    static func emptyCriterion(_ value: ClefJSON) -> Bool {
        if case .null = value { return true }
        return value.string == ""
    }
    static func pythonText(_ value: ClefJSON) throws -> String {
        switch value {
        case .null: return "None"
        case .bool(let value): return value ? "True" : "False"
        default: return try render(value, stringAsText: true)
        }
    }
}
