import Foundation
import MLX
import MLXNN

public struct OpenAIPrivacyFilterSpan: Codable, Hashable, Sendable {
    public let label: String
    public let text: String
    public let startToken: Int
    public let endToken: Int
}

public struct OpenAIPrivacyFilterAnonymizationResult: Codable, Hashable, Sendable {
    public let text: String
    public let anonymizedText: String
    public let tokenCount: Int
    public let spans: [OpenAIPrivacyFilterSpan]

    enum CodingKeys: String, CodingKey {
        case text
        case anonymizedText = "anonymized_text"
        case tokenCount = "token_count"
        case spans
    }
}

public final class OpenAIPrivacyFilterAnonymizer {
    public enum AnonymizerError: LocalizedError, Sendable {
        case missingFiles([URL])
        case noInputTexts
        case invalidMaxTokens(Int)
        case invalidLabelSpace(String)
        case invalidTokenID(Int)

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map(\.path).joined(separator: "\n")
                return "Missing OpenAI privacy filter resources:\n\(list)"
            case .noInputTexts:
                return "At least one input text is required."
            case .invalidMaxTokens(let value):
                return "maxTokens must be positive (received \(value))."
            case .invalidLabelSpace(let message):
                return message
            case .invalidTokenID(let tokenID):
                return "Privacy filter tokenizer could not decode token id \(tokenID)."
            }
        }
    }

    public let resources: OpenAIPrivacyFilterResources
    public let config: OpenAIPrivacyFilterConfig

    private let tokenizer: OpenAIPrivacyFilterTokenizer
    private let model: OpenAIPrivacyFilterModel
    private let labelInfo: OpenAIPrivacyFilterLabelInfo
    private let decoder: OpenAIPrivacyFilterViterbiDecoder

    public init(
        resources: OpenAIPrivacyFilterResources,
        dtype: DType? = .bfloat16,
        fileManager: FileManager = .default
    ) throws {
        let missing = resources.validate(fileManager: fileManager)
        if !missing.isEmpty {
            throw AnonymizerError.missingFiles(missing)
        }

        self.resources = resources
        self.config = try JSONDecoder().decode(
            OpenAIPrivacyFilterConfig.self,
            from: Data(contentsOf: resources.configURL)
        )

        let model = OpenAIPrivacyFilterModel(config: config)
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: model,
            dtype: dtype,
            mapper: OpenAIPrivacyFilterModel.sanitizeWeight
        )

        self.tokenizer = try OpenAIPrivacyFilterTokenizer.load(from: resources.rootURL, config: config)
        self.model = model
        do {
            self.labelInfo = try OpenAIPrivacyFilterLabelInfo(classNames: config.orderedLabels)
        } catch let error as OpenAIPrivacyFilterDecodingError {
            throw AnonymizerError.invalidLabelSpace(error.localizedDescription)
        }
        let biases = try OpenAIPrivacyFilterTransitionBiases.loadIfPresent(
            from: resources.viterbiCalibrationURL,
            fileManager: fileManager
        )
        self.decoder = OpenAIPrivacyFilterViterbiDecoder(labelInfo: labelInfo, biases: biases)
    }

    public func anonymize(
        texts: [String],
        maxTokens: Int? = nil,
        replacementTemplate: String = "[{label}]"
    ) throws -> [OpenAIPrivacyFilterAnonymizationResult] {
        guard !texts.isEmpty else {
            throw AnonymizerError.noInputTexts
        }

        let modelMax = min(config.maxPositionEmbeddings, tokenizer.maxLength)
        let effectiveMaxTokens = min(maxTokens ?? modelMax, modelMax)
        guard effectiveMaxTokens > 0 else {
            throw AnonymizerError.invalidMaxTokens(effectiveMaxTokens)
        }

        return try texts.map { text in
            try anonymizeOne(text: text, maxTokens: effectiveMaxTokens, replacementTemplate: replacementTemplate)
        }
    }

    private func anonymizeOne(
        text: String,
        maxTokens: Int,
        replacementTemplate: String
    ) throws -> OpenAIPrivacyFilterAnonymizationResult {
        let fullTokenIDs = tokenizer.encode(text, addSpecialTokens: false)
        if fullTokenIDs.isEmpty {
            return OpenAIPrivacyFilterAnonymizationResult(
                text: text,
                anonymizedText: text,
                tokenCount: 0,
                spans: []
            )
        }

        let tokenIDs = Array(fullTokenIDs.prefix(maxTokens))

        let shape = [1, tokenIDs.count]
        let inputIDs = MLXArray(tokenIDs.map { Int32($0) }, shape)
        let attentionMask = MLX.ones(shape, dtype: .int32)

        let outputs = model(inputIds: inputIDs, attentionMask: attentionMask)
        let logProbabilities = logSoftmax(outputs.logits.asType(.float32), axis: -1)
        MLX.eval(logProbabilities)

        let shapeInfo = logProbabilities.shape
        precondition(shapeInfo.count == 3, "privacy filter logits must have shape [batch, sequence, labels]")
        let sequenceLength = shapeInfo[1]
        let classCount = shapeInfo[2]
        let flattenedLogProbs = logProbabilities.asArray(Float.self)
        let decodedLabels = decoder.decode(
            logProbs: flattenedLogProbs,
            sequenceLength: sequenceLength,
            classCount: classCount
        )

        return try aggregate(
            originalText: text,
            tokenIDs: tokenIDs,
            predictedIDs: decodedLabels,
            replacementTemplate: replacementTemplate
        )
    }

    private func aggregate(
        originalText: String,
        tokenIDs: [Int],
        predictedIDs: [Int],
        replacementTemplate: String
    ) throws -> OpenAIPrivacyFilterAnonymizationResult {
        let tokenStrings = try tokenIDs.map { tokenID in
            guard let token = tokenizer.tokenString(for: tokenID) else {
                throw AnonymizerError.invalidTokenID(tokenID)
            }
            return token
        }

        let labelsByIndex = Dictionary(uniqueKeysWithValues: predictedIDs.enumerated().map { ($0.offset, $0.element) })
        let tokenSpans = OpenAIPrivacyFilterSpanUtilities.labelsToTokenSpans(
            labelsByIndex: labelsByIndex,
            labelInfo: labelInfo
        )

        let sourceText: String
        let charStarts: [Int]
        let charEnds: [Int]
        if let ranges = OpenAIPrivacyFilterByteLevelCodec.charRanges(for: tokenStrings, in: originalText) {
            sourceText = originalText
            charStarts = ranges.charStarts
            charEnds = ranges.charEnds
        } else {
            let decoded = OpenAIPrivacyFilterByteLevelCodec.decodeTextWithOffsets(tokenStrings: tokenStrings)
            sourceText = decoded.text
            charStarts = decoded.charStarts
            charEnds = decoded.charEnds
        }

        let charSpans = OpenAIPrivacyFilterSpanUtilities.trimWhitespace(
            OpenAIPrivacyFilterSpanUtilities.tokenSpansToCharSpans(
                tokenSpans,
                charStarts: charStarts,
                charEnds: charEnds
            ),
            in: sourceText
        )

        let detectedSpans = charSpans.compactMap { span -> (label: String, text: String, start: Int, end: Int, startToken: Int, endToken: Int)? in
            guard span.label >= 0, span.label < labelInfo.spanClassNames.count else {
                return nil
            }
            guard let textRange = characterRange(in: sourceText, start: span.start, end: span.end) else {
                return nil
            }
            return (
                label: labelInfo.spanClassNames[span.label],
                text: String(sourceText[textRange]),
                start: span.start,
                end: span.end,
                startToken: span.startToken,
                endToken: span.endToken
            )
        }

        let selectedSpans = OpenAIPrivacyFilterSpanUtilities.selectNonOverlapping(
            detectedSpans,
            start: { $0.start },
            end: { $0.end },
            label: { $0.label }
        )

        let spans = selectedSpans.map {
            OpenAIPrivacyFilterSpan(
                label: $0.label,
                text: $0.text,
                startToken: $0.startToken,
                endToken: $0.endToken
            )
        }

        return OpenAIPrivacyFilterAnonymizationResult(
            text: sourceText,
            anonymizedText: applyRedactions(
                to: sourceText,
                spans: selectedSpans,
                replacementTemplate: replacementTemplate
            ),
            tokenCount: tokenIDs.count,
            spans: spans
        )
    }

    private func applyRedactions(
        to text: String,
        spans: [(label: String, text: String, start: Int, end: Int, startToken: Int, endToken: Int)],
        replacementTemplate: String
    ) -> String {
        guard !spans.isEmpty else {
            return text
        }

        var result = ""
        var cursor = 0

        for (index, span) in spans.enumerated() {
            guard let gapRange = characterRange(in: text, start: cursor, end: span.start) else {
                continue
            }
            result += text[gapRange]
            result += replacementTemplate
                .replacingOccurrences(of: "{label}", with: span.label)
                .replacingOccurrences(of: "{index}", with: String(index + 1))
            cursor = span.end
        }

        if let tailRange = characterRange(in: text, start: cursor, end: text.count) {
            result += text[tailRange]
        }

        return result
    }

    private func characterRange(in text: String, start: Int, end: Int) -> Range<String.Index>? {
        guard start >= 0, end >= start, end <= text.count else {
            return nil
        }
        let startIndex = text.index(text.startIndex, offsetBy: start)
        let endIndex = text.index(text.startIndex, offsetBy: end)
        return startIndex..<endIndex
    }
}
