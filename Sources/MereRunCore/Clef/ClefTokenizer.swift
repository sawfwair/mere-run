#if !os(iOS)
import Foundation
import Hub
import Tokenizers
import MereRunQwenModel

struct ClefTokenSequence {
    let ids: [Int]
    let fields: [ClefHeadField]
    let plan: ClefDecisionPlan
}

struct ClefTokenizer {
    static let systemPrompt = "Read the complete state and schema. Decide every field jointly. Each answer "
        + "must be exactly one of that field's allowed options."
    static let prefixText = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n"
    let encode: (String) -> [Int]

    static func load(root: URL) throws -> ClefTokenizer {
        let config = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer_config.json"))
        let data = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer.json"))
        guard data.model.type.string() == "BPE" else {
            throw ClefError.invalidConfiguration("Clef requires its checkpoint-native BPE tokenizer.")
        }
        // Qwen3_5Tokenizer uses the same BPE implementation. Naming the registered
        // factory prevents Swift Transformers' fallback warning from polluting stdout.
        var nativeConfig = config.dictionary(or: [:])
        nativeConfig["tokenizer_class"] = Config("Qwen2Tokenizer")
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(nativeConfig), tokenizerData: data)
        return ClefTokenizer(encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
    }

    func sequence(_ request: ClefDecisionRequest, modelID: String, mediaText: String = "", mediaIDs: [Int]? = nil) throws -> ClefTokenSequence {
        var schema = encode("\n\nSCHEMA FIELDS:\n")
        var spans: [ClefHeadField] = []
        for (index, question) in request.questions.enumerated() {
            schema += encode("\nFIELD \(index + 1)\nID: \(question.id)\nTYPE: \(question.type.rawValue)\nINSTRUCTION: ")
            let start = schema.count
            schema += encode(try question.instructions.rendered())
            let questionSpan = start..<schema.count
            schema += encode("\nALLOWED OPTIONS:\n")
            var optionSpans: [Range<Int>] = []
            for (optionIndex, option) in question.options.enumerated() {
                schema += encode("OPTION \(optionIndex + 1): ")
                let start = schema.count
                var semantics = [ClefJSON.Field(key: "option_id", value: .string(option.id))]
                if case .null = option.description {} else { semantics.append(.init(key: "description", value: option.description)) }
                schema += encode(try ClefJSON.object(semantics).canonical())
                optionSpans.append(start..<schema.count)
                schema += encode("\n")
            }
            schema += encode("END FIELD\n")
            guard !questionSpan.isEmpty, optionSpans.allSatisfy({ !$0.isEmpty }) else {
                throw ClefError.invalidInput("Clef instructions and options must tokenize to nonempty spans.")
            }
            let type = question.type == .noul ? 0 : question.type == .choice ? 1 : 2
            spans.append(ClefHeadField(type: type, questionSpan: questionSpan, optionSpans: optionSpans))
        }
        let prefix = encode(Self.prefixText)
            + (mediaIDs ?? encode(mediaText))
        let suffix = encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
        let state = encode(try request.state.rendered())
        let fixed = prefix.count + schema.count + suffix.count
        guard fixed <= request.maxTokens else {
            throw ClefError.invalidInput("Clef schema and media require \(fixed) tokens before state; maximum is \(request.maxTokens).")
        }
        let kept = Array(state.prefix(min(request.maxStateTokens ?? state.count, request.maxTokens - fixed)))
        let offset = prefix.count + kept.count
        let shifted = spans.map { field in
            ClefHeadField(type: field.type, questionSpan: (field.questionSpan.lowerBound + offset)..<(field.questionSpan.upperBound + offset),
                          optionSpans: field.optionSpans.map { ($0.lowerBound + offset)..<($0.upperBound + offset) })
        }
        let ids = prefix + kept + schema + suffix
        let details = zip(request.questions, shifted).map { question, field in
            ClefDecisionPlan.Field(id: question.id, optionIDs: question.options.map(\.id),
                                   questionSpan: [field.questionSpan.lowerBound, field.questionSpan.upperBound],
                                   optionSpans: field.optionSpans.map { [$0.lowerBound, $0.upperBound] })
        }
        let plan = ClefDecisionPlan(model: modelID, inputTokens: ids.count, stateTokens: kept.count,
                                    stateTokensDropped: state.count - kept.count, questions: details)
        return ClefTokenSequence(ids: ids, fields: shifted, plan: plan)
    }
}
#endif
