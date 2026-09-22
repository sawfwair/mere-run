import Foundation
import MereRunLayaModel
@preconcurrency import Hub
@preconcurrency import Tokenizers

public struct LayaTokenSequence: Sendable {
    public let ids: [Int]
    public let markers: [Int]
    public let details: LayaPreparedQuestion
}

public final class LayaTokenizer {
    private let encodeText: (String) -> [Int]
    public let padTokenID: Int
    private let clsTokenID: Int
    private let sepTokenID: Int
    private let maskTokenID: Int
    private let maskToken: String

    init(padTokenID: Int, clsTokenID: Int, sepTokenID: Int, maskTokenID: Int,
         maskToken: String, encode: @escaping (String) -> [Int]) {
        self.padTokenID = padTokenID
        self.clsTokenID = clsTokenID
        self.sepTokenID = sepTokenID
        self.maskTokenID = maskTokenID
        self.maskToken = maskToken
        self.encodeText = encode
    }

    private struct SpecialTokens: Decodable {
        let clsToken: String
        let sepToken: String
        let maskToken: String
        let padToken: String
        enum CodingKeys: String, CodingKey {
            case clsToken = "cls_token", sepToken = "sep_token", maskToken = "mask_token", padToken = "pad_token"
        }
    }

    public static func load(root: URL, vocabularySize: Int) throws -> LayaTokenizer {
        let configURL = root.appending(path: "tokenizer/tokenizer_config.json")
        let special = try JSONDecoder().decode(SpecialTokens.self, from: Data(contentsOf: configURL))
        let config = try HubApi.shared.configuration(fileURL: configURL)
        let data = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer/tokenizer.json"))
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data, strict: false)
        func tokenID(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token), (0..<vocabularySize).contains(id) else {
                throw LayaModelError.invalidConfiguration("Tokenizer is missing the special token \(token).")
            }
            return id
        }
        return try LayaTokenizer(
            padTokenID: tokenID(special.padToken), clsTokenID: tokenID(special.clsToken),
            sepTokenID: tokenID(special.sepToken), maskTokenID: tokenID(special.maskToken), maskToken: special.maskToken,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
    }

    public func sequence(state: String, question: LayaQuestion, maxLength: Int, headMaxLength: Int) throws -> LayaTokenSequence {
        try sequence(stateIDs: encodeState(state), question: question, maxLength: maxLength, headMaxLength: headMaxLength)
    }

    func encodeState(_ state: String) -> [Int] {
        encodeText(state.replacingOccurrences(of: maskToken, with: " "))
    }

    func sequence(stateIDs: [Int], question: LayaQuestion, maxLength: Int, headMaxLength: Int) throws -> LayaTokenSequence {
        try question.validate()
        guard maxLength >= 16, maxLength <= 8_192, headMaxLength >= 8, headMaxLength < maxLength else {
            throw LayaModelError.invalidInput("The head token budget must be at least 8 and smaller than the context budget.")
        }
        func sanitized(_ text: String) -> String { text.replacingOccurrences(of: maskToken, with: " ") }
        let instruction = encodeText("\(question.type.rawValue) question: \(sanitized(question.instructions))")
        let fullOptions = question.renderedOptions.map { encodeText(" " + sanitized($0)) }
        var options = fullOptions.map { [maskTokenID] + Array($0.prefix(48)) }
        var budget = headMaxLength - options.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let perOption = max(4, (headMaxLength - 16) / max(1, options.count))
            options = options.map { Array($0.prefix(perOption)) }
            budget = headMaxLength - options.reduce(0) { $0 + $1.count }
        }
        let head = Array(instruction.prefix(max(8, budget)))
        var ids = [clsTokenID] + head + [sepTokenID]
        var markers: [Int] = []
        for option in options {
            markers.append(ids.count)
            ids += option
        }
        ids.append(sepTokenID)
        // Preserve every option's text and terminal separator; never score a partial label.
        guard ids.count < maxLength else {
            throw LayaModelError.invalidInput("Question \(question.id) options exceed the context budget; increase max_tokens or reduce criteria.")
        }
        let kept = Array(stateIDs.prefix(maxLength - ids.count - 1))
        ids += kept + [sepTokenID]
        let details = LayaPreparedQuestion(
            id: question.id, inputTokens: ids.count, stateTokens: kept.count, stateTokensDropped: stateIDs.count - kept.count,
            instructionTokensDropped: instruction.count - head.count,
            optionTokensDropped: zip(fullOptions, options).map { $0.count - ($1.count - 1) }, optionCount: markers.count)
        return LayaTokenSequence(ids: ids, markers: markers, details: details)
    }
}
