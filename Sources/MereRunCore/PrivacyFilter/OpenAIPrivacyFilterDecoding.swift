import Foundation

enum OpenAIPrivacyFilterDecodingError: LocalizedError {
    case invalidLabelSpace(String)

    var errorDescription: String? {
        switch self {
        case .invalidLabelSpace(let message):
            return message
        }
    }
}

struct OpenAIPrivacyFilterLabelInfo {
    let tokenToSpanLabel: [Int: Int]
    let tokenBoundaryTags: [Int: String?]
    let spanClassNames: [String]
    let backgroundTokenLabel: Int
    let backgroundSpanLabel: Int

    init(classNames: [String]) throws {
        var spanClassNames = ["O"]
        var spanLabelLookup = ["O": 0]
        var tokenToSpanLabel: [Int: Int] = [:]
        var tokenBoundaryTags: [Int: String?] = [:]
        var boundaryLabelLookup: [String: Set<String>] = [:]
        var backgroundTokenLabel: Int?

        for (index, name) in classNames.enumerated() {
            if name == "O" {
                backgroundTokenLabel = index
                tokenToSpanLabel[index] = 0
                tokenBoundaryTags[index] = nil
                continue
            }

            let components = name.split(separator: "-", maxSplits: 1).map(String.init)
            guard components.count == 2 else {
                throw OpenAIPrivacyFilterDecodingError.invalidLabelSpace(
                    "Privacy filter label '\(name)' must use '<B|I|E|S>-<label>' format."
                )
            }

            let boundary = components[0]
            let baseLabel = components[1]
            guard ["B", "I", "E", "S"].contains(boundary), !baseLabel.isEmpty else {
                throw OpenAIPrivacyFilterDecodingError.invalidLabelSpace(
                    "Privacy filter label '\(name)' must use '<B|I|E|S>-<label>' format."
                )
            }

            let spanIndex: Int
            if let existing = spanLabelLookup[baseLabel] {
                spanIndex = existing
            } else {
                spanIndex = spanClassNames.count
                spanClassNames.append(baseLabel)
                spanLabelLookup[baseLabel] = spanIndex
            }

            tokenToSpanLabel[index] = spanIndex
            tokenBoundaryTags[index] = boundary
            boundaryLabelLookup[baseLabel, default: []].insert(boundary)
        }

        guard let backgroundTokenLabel else {
            throw OpenAIPrivacyFilterDecodingError.invalidLabelSpace(
                "Privacy filter labels must include background label 'O'."
            )
        }

        for (baseLabel, boundaries) in boundaryLabelLookup {
            let missing = Set(["B", "I", "E", "S"]).subtracting(boundaries)
            if !missing.isEmpty {
                throw OpenAIPrivacyFilterDecodingError.invalidLabelSpace(
                    "Privacy filter label space is missing \(missing.sorted()) for '\(baseLabel)'."
                )
            }
        }

        self.tokenToSpanLabel = tokenToSpanLabel
        self.tokenBoundaryTags = tokenBoundaryTags
        self.spanClassNames = spanClassNames
        self.backgroundTokenLabel = backgroundTokenLabel
        self.backgroundSpanLabel = 0
    }
}

struct OpenAIPrivacyFilterTransitionBiases: Equatable {
    var backgroundStay: Float = 0
    var backgroundToStart: Float = 0
    var insideToContinue: Float = 0
    var insideToEnd: Float = 0
    var endToBackground: Float = 0
    var endToStart: Float = 0

    static let zero = OpenAIPrivacyFilterTransitionBiases()

    private struct CalibrationArtifact: Decodable {
        struct OperatingPoints: Decodable {
            struct Entry: Decodable {
                struct Biases: Decodable {
                    let transitionBiasBackgroundStay: Float
                    let transitionBiasBackgroundToStart: Float
                    let transitionBiasInsideToContinue: Float
                    let transitionBiasInsideToEnd: Float
                    let transitionBiasEndToBackground: Float
                    let transitionBiasEndToStart: Float

                    enum CodingKeys: String, CodingKey {
                        case transitionBiasBackgroundStay = "transition_bias_background_stay"
                        case transitionBiasBackgroundToStart = "transition_bias_background_to_start"
                        case transitionBiasInsideToContinue = "transition_bias_inside_to_continue"
                        case transitionBiasInsideToEnd = "transition_bias_inside_to_end"
                        case transitionBiasEndToBackground = "transition_bias_end_to_background"
                        case transitionBiasEndToStart = "transition_bias_end_to_start"
                    }
                }

                let biases: Biases
            }

            let `default`: Entry
        }

        let operatingPoints: OperatingPoints

        enum CodingKeys: String, CodingKey {
            case operatingPoints = "operating_points"
        }
    }

    static func loadIfPresent(from url: URL, fileManager: FileManager = .default) throws -> OpenAIPrivacyFilterTransitionBiases {
        guard fileManager.fileExists(atPath: url.path) else {
            return .zero
        }

        let artifact = try JSONDecoder().decode(CalibrationArtifact.self, from: Data(contentsOf: url))
        let biases = artifact.operatingPoints.default.biases
        return OpenAIPrivacyFilterTransitionBiases(
            backgroundStay: biases.transitionBiasBackgroundStay,
            backgroundToStart: biases.transitionBiasBackgroundToStart,
            insideToContinue: biases.transitionBiasInsideToContinue,
            insideToEnd: biases.transitionBiasInsideToEnd,
            endToBackground: biases.transitionBiasEndToBackground,
            endToStart: biases.transitionBiasEndToStart
        )
    }
}

struct OpenAIPrivacyFilterViterbiDecoder {
    private let labelInfo: OpenAIPrivacyFilterLabelInfo
    private let startScores: [Float]
    private let endScores: [Float]
    private let transitionScores: [Float]

    init(labelInfo: OpenAIPrivacyFilterLabelInfo, biases: OpenAIPrivacyFilterTransitionBiases) {
        self.labelInfo = labelInfo

        let numClasses = labelInfo.tokenToSpanLabel.keys.max().map { $0 + 1 } ?? 0
        var startScores = Array(repeating: -Float.infinity, count: numClasses)
        var endScores = Array(repeating: -Float.infinity, count: numClasses)
        var transitionScores = Array(repeating: -Float.infinity, count: numClasses * numClasses)

        for index in 0..<numClasses {
            let tag = labelInfo.tokenBoundaryTags[index] ?? nil
            let span = labelInfo.tokenToSpanLabel[index]

            if tag == "B" || tag == "S" || index == labelInfo.backgroundTokenLabel {
                startScores[index] = 0
            }
            if tag == "E" || tag == "S" || index == labelInfo.backgroundTokenLabel {
                endScores[index] = 0
            }

            for nextIndex in 0..<numClasses {
                let nextTag = labelInfo.tokenBoundaryTags[nextIndex] ?? nil
                let nextSpan = labelInfo.tokenToSpanLabel[nextIndex]
                guard Self.isValidTransition(
                    prevTag: tag,
                    prevSpan: span,
                    nextTag: nextTag,
                    nextSpan: nextSpan,
                    backgroundTokenLabel: labelInfo.backgroundTokenLabel,
                    backgroundSpanLabel: labelInfo.backgroundSpanLabel,
                    nextIndex: nextIndex
                ) else {
                    continue
                }

                transitionScores[index * numClasses + nextIndex] = Self.transitionBias(
                    prevTag: tag,
                    prevSpan: span,
                    nextTag: nextTag,
                    nextSpan: nextSpan,
                    backgroundTokenLabel: labelInfo.backgroundTokenLabel,
                    backgroundSpanLabel: labelInfo.backgroundSpanLabel,
                    prevIndex: index,
                    nextIndex: nextIndex,
                    biases: biases
                )
            }
        }

        self.startScores = startScores
        self.endScores = endScores
        self.transitionScores = transitionScores
    }

    func decode(logProbs: [Float], sequenceLength: Int, classCount: Int) -> [Int] {
        guard sequenceLength > 0, classCount > 0 else {
            return []
        }
        guard logProbs.count == sequenceLength * classCount else {
            return []
        }

        var scores = Array(repeating: -Float.infinity, count: classCount)
        for label in 0..<classCount {
            scores[label] = logProbs[label] + startScores[label]
        }

        var backpointers = Array(repeating: 0, count: max(0, (sequenceLength - 1) * classCount))

        if sequenceLength > 1 {
            for position in 1..<sequenceLength {
                var nextScores = Array(repeating: -Float.infinity, count: classCount)
                for nextLabel in 0..<classCount {
                    var bestScore = -Float.infinity
                    var bestPath = 0
                    for prevLabel in 0..<classCount {
                        let transition = transitionScores[prevLabel * classCount + nextLabel]
                        if !transition.isFinite || !scores[prevLabel].isFinite {
                            continue
                        }
                        let candidate = scores[prevLabel] + transition
                        if candidate > bestScore {
                            bestScore = candidate
                            bestPath = prevLabel
                        }
                    }

                    nextScores[nextLabel] = bestScore + logProbs[position * classCount + nextLabel]
                    backpointers[(position - 1) * classCount + nextLabel] = bestPath
                }
                scores = nextScores
            }
        }

        if !scores.contains(where: { $0.isFinite }) {
            return fallbackArgmax(logProbs: logProbs, sequenceLength: sequenceLength, classCount: classCount)
        }

        var lastLabel = 0
        var bestFinalScore = -Float.infinity
        for label in 0..<classCount {
            let score = scores[label] + endScores[label]
            if score > bestFinalScore {
                bestFinalScore = score
                lastLabel = label
            }
        }

        var path = Array(repeating: 0, count: sequenceLength)
        path[sequenceLength - 1] = lastLabel
        if sequenceLength > 1 {
            for position in stride(from: sequenceLength - 2, through: 0, by: -1) {
                let nextLabel = path[position + 1]
                path[position] = backpointers[position * classCount + nextLabel]
            }
        }
        return path
    }

    private func fallbackArgmax(logProbs: [Float], sequenceLength: Int, classCount: Int) -> [Int] {
        (0..<sequenceLength).map { position in
            var bestLabel = 0
            var bestScore = -Float.infinity
            for label in 0..<classCount {
                let score = logProbs[position * classCount + label]
                if score > bestScore {
                    bestScore = score
                    bestLabel = label
                }
            }
            return bestLabel
        }
    }

    private static func transitionBias(
        prevTag: String?,
        prevSpan: Int?,
        nextTag: String?,
        nextSpan: Int?,
        backgroundTokenLabel: Int,
        backgroundSpanLabel: Int,
        prevIndex: Int,
        nextIndex: Int,
        biases: OpenAIPrivacyFilterTransitionBiases
    ) -> Float {
        let prevIsBackground = prevSpan == backgroundSpanLabel || prevIndex == backgroundTokenLabel
        let nextIsBackground = nextSpan == backgroundSpanLabel || nextIndex == backgroundTokenLabel

        if prevIsBackground {
            if nextIsBackground {
                return biases.backgroundStay
            }
            if nextTag == "B" || nextTag == "S" {
                return biases.backgroundToStart
            }
            return 0
        }

        if prevTag == "B" || prevTag == "I" {
            if nextTag == "I", prevSpan == nextSpan {
                return biases.insideToContinue
            }
            if nextTag == "E", prevSpan == nextSpan {
                return biases.insideToEnd
            }
            return 0
        }

        if prevTag == "E" || prevTag == "S" {
            if nextIsBackground {
                return biases.endToBackground
            }
            if nextTag == "B" || nextTag == "S" {
                return biases.endToStart
            }
            return 0
        }

        return 0
    }

    private static func isValidTransition(
        prevTag: String?,
        prevSpan: Int?,
        nextTag: String?,
        nextSpan: Int?,
        backgroundTokenLabel: Int,
        backgroundSpanLabel: Int,
        nextIndex: Int
    ) -> Bool {
        let nextIsBackground = nextSpan == backgroundSpanLabel || nextIndex == backgroundTokenLabel
        if (nextSpan == nil || nextTag == nil) && !nextIsBackground {
            return false
        }

        if prevSpan == nil || prevTag == nil {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }

        let prevIsBackground = prevSpan == backgroundSpanLabel
        if prevIsBackground {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }

        if prevTag == "E" || prevTag == "S" {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }

        if prevTag == "B" || prevTag == "I" {
            let sameSpan = prevSpan == nextSpan
            return sameSpan && (nextTag == "I" || nextTag == "E")
        }

        return false
    }
}

enum OpenAIPrivacyFilterSpanUtilities {
    static func labelsToTokenSpans(
        labelsByIndex: [Int: Int],
        labelInfo: OpenAIPrivacyFilterLabelInfo
    ) -> [(label: Int, startToken: Int, endToken: Int)] {
        guard !labelsByIndex.isEmpty else {
            return []
        }

        var spans: [(label: Int, startToken: Int, endToken: Int)] = []
        var currentLabel: Int?
        var currentStartToken: Int?
        var previousIndex: Int?

        for tokenIndex in labelsByIndex.keys.sorted() {
            let labelID = labelsByIndex[tokenIndex] ?? labelInfo.backgroundTokenLabel
            let spanLabel = labelInfo.tokenToSpanLabel[labelID]
            let boundary = labelInfo.tokenBoundaryTags[labelID] ?? nil

            if let previousIndex, tokenIndex != previousIndex + 1 {
                if let currentLabel, let currentStartToken {
                    spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
                }
                currentLabel = nil
                currentStartToken = nil
            }

            guard let spanLabel else {
                previousIndex = tokenIndex
                continue
            }

            let isBackground = spanLabel == labelInfo.backgroundSpanLabel
            if isBackground {
                if let currentLabel, let currentStartToken {
                    spans.append((label: currentLabel, startToken: currentStartToken, endToken: tokenIndex))
                }
                currentLabel = nil
                currentStartToken = nil
                previousIndex = tokenIndex
                continue
            }

            switch boundary {
            case "S":
                if let currentLabel, let currentStartToken, let previousIndex {
                    spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
                }
                spans.append((label: spanLabel, startToken: tokenIndex, endToken: tokenIndex + 1))
                currentLabel = nil
                currentStartToken = nil
            case "B":
                if let currentLabel, let currentStartToken, let previousIndex {
                    spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
                }
                currentLabel = spanLabel
                currentStartToken = tokenIndex
            case "I":
                if currentLabel == nil || currentLabel != spanLabel {
                    if let currentLabel, let currentStartToken, let previousIndex {
                        spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
                    }
                    currentLabel = spanLabel
                    currentStartToken = tokenIndex
                }
            case "E":
                if currentLabel == nil || currentLabel != spanLabel || currentStartToken == nil {
                    if let currentLabel, let currentStartToken, let previousIndex {
                        spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
                    }
                    spans.append((label: spanLabel, startToken: tokenIndex, endToken: tokenIndex + 1))
                    currentLabel = nil
                    currentStartToken = nil
                } else if let activeStartToken = currentStartToken {
                    spans.append((label: spanLabel, startToken: activeStartToken, endToken: tokenIndex + 1))
                    currentLabel = nil
                    currentStartToken = nil
                }
            default:
                if let currentLabel, let currentStartToken, let previousIndex {
                    spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
                }
                currentLabel = nil
                currentStartToken = nil
            }

            previousIndex = tokenIndex
        }

        if let currentLabel, let currentStartToken, let previousIndex {
            spans.append((label: currentLabel, startToken: currentStartToken, endToken: previousIndex + 1))
        }

        return spans
    }

    static func tokenSpansToCharSpans(
        _ spans: [(label: Int, startToken: Int, endToken: Int)],
        charStarts: [Int],
        charEnds: [Int]
    ) -> [(label: Int, start: Int, end: Int, startToken: Int, endToken: Int)] {
        spans.compactMap { span in
            guard span.startToken >= 0, span.endToken > span.startToken, span.endToken <= charStarts.count else {
                return nil
            }
            let start = charStarts[span.startToken]
            let end = charEnds[span.endToken - 1]
            guard end > start else {
                return nil
            }
            return (label: span.label, start: start, end: end, startToken: span.startToken, endToken: span.endToken)
        }
    }

    static func trimWhitespace(
        _ spans: [(label: Int, start: Int, end: Int, startToken: Int, endToken: Int)],
        in text: String
    ) -> [(label: Int, start: Int, end: Int, startToken: Int, endToken: Int)] {
        let characters = Array(text)
        return spans.compactMap { span in
            guard span.start >= 0, span.end <= characters.count, span.end > span.start else {
                return nil
            }

            var start = span.start
            var end = span.end
            while start < end && isWhitespace(characters[start]) {
                start += 1
            }
            while end > start && isWhitespace(characters[end - 1]) {
                end -= 1
            }

            guard end > start else {
                return nil
            }
            return (label: span.label, start: start, end: end, startToken: span.startToken, endToken: span.endToken)
        }
    }

    static func selectNonOverlapping<T>(
        _ spans: [T],
        start: (T) -> Int,
        end: (T) -> Int,
        label: (T) -> String
    ) -> [T] {
        let ordered = spans.sorted {
            let leftStart = start($0)
            let rightStart = start($1)
            if leftStart != rightStart {
                return leftStart < rightStart
            }

            let leftLength = end($0) - leftStart
            let rightLength = end($1) - rightStart
            if leftLength != rightLength {
                return leftLength > rightLength
            }

            return label($0) < label($1)
        }

        var kept: [T] = []
        var cursor = 0
        for span in ordered {
            guard end(span) > start(span), start(span) >= cursor else {
                continue
            }
            kept.append(span)
            cursor = end(span)
        }
        return kept
    }

    private static func isWhitespace(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { $0.properties.isWhitespace }
    }
}

enum OpenAIPrivacyFilterByteLevelCodec {
    private static let inverseByteDecoder: [UnicodeScalar: UInt8] = {
        let preserved = Array(33...126) + Array(161...172) + Array(174...255)
        let preservedSet = Set(preserved)

        var byteValues = preserved
        var scalarValues = preserved
        var nextScalar = 256

        for byte in 0...255 where !preservedSet.contains(byte) {
            byteValues.append(byte)
            scalarValues.append(nextScalar)
            nextScalar += 1
        }

        var mapping: [UnicodeScalar: UInt8] = [:]
        for (byte, scalar) in zip(byteValues, scalarValues) {
            mapping[UnicodeScalar(scalar)!] = UInt8(byte)
        }
        return mapping
    }()

    static func decodeTextWithOffsets(tokenStrings: [String]) -> (text: String, charStarts: [Int], charEnds: [Int]) {
        let tokenBytes = tokenStrings.map(bytes(for:))
        let decodedText = String(decoding: tokenBytes.flatMap { $0 }, as: UTF8.self)
        let (charStarts, charEnds) = charRanges(for: tokenBytes, in: decodedText)
        return (text: decodedText, charStarts: charStarts, charEnds: charEnds)
    }

    static func charRanges(for tokenStrings: [String], in sourceText: String) -> (charStarts: [Int], charEnds: [Int])? {
        let tokenBytes = tokenStrings.map(bytes(for:))
        let prefixBytes = tokenBytes.flatMap { $0 }
        guard Array(sourceText.utf8).starts(with: prefixBytes) else {
            return nil
        }
        return charRanges(for: tokenBytes, in: sourceText)
    }

    private static func bytes(for token: String) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(token.utf8.count)

        for scalar in token.unicodeScalars {
            if let byte = inverseByteDecoder[scalar] {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: String(scalar).utf8)
            }
        }

        return bytes
    }

    private static func charRanges(for tokenBytes: [[UInt8]], in text: String) -> (charStarts: [Int], charEnds: [Int]) {
        var charByteStarts: [Int] = []
        var charByteEnds: [Int] = []
        charByteStarts.reserveCapacity(text.count)
        charByteEnds.reserveCapacity(text.count)

        var byteCursor = 0
        for character in text {
            charByteStarts.append(byteCursor)
            byteCursor += String(character).lengthOfBytes(using: .utf8)
            charByteEnds.append(byteCursor)
        }

        var tokenCharStarts: [Int] = []
        var tokenCharEnds: [Int] = []
        tokenCharStarts.reserveCapacity(tokenBytes.count)
        tokenCharEnds.reserveCapacity(tokenBytes.count)

        var tokenByteCursor = 0
        for bytes in tokenBytes {
            let tokenByteStart = tokenByteCursor
            let tokenByteEnd = tokenByteStart + bytes.count
            tokenByteCursor = tokenByteEnd

            let startIndex = upperBound(in: charByteEnds, value: tokenByteStart)
            var endIndex = lowerBound(in: charByteStarts, value: tokenByteEnd)
            if endIndex < startIndex {
                endIndex = startIndex
            }
            tokenCharStarts.append(startIndex)
            tokenCharEnds.append(endIndex)
        }

        return (charStarts: tokenCharStarts, charEnds: tokenCharEnds)
    }

    private static func lowerBound(in values: [Int], value: Int) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] < value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }

    private static func upperBound(in values: [Int], value: Int) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] <= value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }
}
