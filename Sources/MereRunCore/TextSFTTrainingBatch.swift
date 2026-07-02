import Foundation
import MLX
import MLXNN

public struct TextSFTTokenizedExample: Sendable, Hashable {
    public let inputTokenIds: [Int]
    public let labelTokenIds: [Int]
    public let lossMask: [Float]

    public init(inputTokenIds: [Int], labelTokenIds: [Int], lossMask: [Float]) {
        self.inputTokenIds = inputTokenIds
        self.labelTokenIds = labelTokenIds
        self.lossMask = lossMask
    }
}

public enum TextSFTTrainingBatchBuilder {
    public static func shiftedTargetExample(
        prefixTokenIds: [Int],
        targetTokenIds: [Int],
        maxSequenceLength: Int
    ) throws -> TextSFTTokenizedExample {
        guard maxSequenceLength >= 2 else {
            throw TextSFTTrainingBatchError.maxSequenceLengthTooSmall(maxSequenceLength)
        }
        guard !targetTokenIds.isEmpty else {
            throw TextSFTTrainingBatchError.noAssistantTargets
        }

        var tokenIds = prefixTokenIds + targetTokenIds
        var targetMask = Array(repeating: Float(0), count: max(prefixTokenIds.count - 1, 0))
            + Array(repeating: Float(1), count: targetTokenIds.count)

        if tokenIds.count > maxSequenceLength {
            let trimCount = tokenIds.count - maxSequenceLength
            tokenIds = Array(tokenIds.dropFirst(trimCount))
            targetMask = Array(targetMask.dropFirst(min(trimCount, targetMask.count)))
        }

        guard tokenIds.count >= 2 else {
            throw TextSFTTrainingBatchError.notEnoughTokens
        }

        let inputs = Array(tokenIds.dropLast())
        let labels = Array(tokenIds.dropFirst())
        if targetMask.count > labels.count {
            targetMask = Array(targetMask.suffix(labels.count))
        }
        guard targetMask.count == labels.count else {
            throw TextSFTTrainingBatchError.shiftedLengthMismatch
        }
        guard targetMask.contains(where: { $0 > 0 }) else {
            throw TextSFTTrainingBatchError.noAssistantTargets
        }

        return TextSFTTokenizedExample(
            inputTokenIds: inputs,
            labelTokenIds: labels,
            lossMask: targetMask
        )
    }

    public static func shiftedExample(
        messageTokenIds: [[Int]],
        messageRoles: [ChatMessage.Role],
        maxSequenceLength: Int
    ) throws -> TextSFTTokenizedExample {
        guard maxSequenceLength >= 2 else {
            throw TextSFTTrainingBatchError.maxSequenceLengthTooSmall(maxSequenceLength)
        }
        guard messageTokenIds.count == messageRoles.count else {
            throw TextSFTTrainingBatchError.messageTokenRoleMismatch
        }

        var tokenIds: [Int] = []
        var assistantTargets: [Float] = []
        for (index, tokens) in messageTokenIds.enumerated() {
            guard !tokens.isEmpty else { continue }
            let isAssistant = messageRoles[index] == .assistant
            tokenIds.append(contentsOf: tokens)
            assistantTargets.append(contentsOf: Array(repeating: isAssistant ? 1 : 0, count: tokens.count))
        }

        if tokenIds.count > maxSequenceLength {
            tokenIds = Array(tokenIds.suffix(maxSequenceLength))
            assistantTargets = Array(assistantTargets.suffix(maxSequenceLength))
        }

        guard tokenIds.count >= 2 else {
            throw TextSFTTrainingBatchError.notEnoughTokens
        }

        let inputs = Array(tokenIds.dropLast())
        let labels = Array(tokenIds.dropFirst())
        let shiftedMask = Array(assistantTargets.dropFirst())
        guard shiftedMask.contains(where: { $0 > 0 }) else {
            throw TextSFTTrainingBatchError.noAssistantTargets
        }
        return TextSFTTokenizedExample(
            inputTokenIds: inputs,
            labelTokenIds: labels,
            lossMask: shiftedMask
        )
    }

    public static func makeBatch(
        _ examples: [TextSFTTokenizedExample],
        padTokenId: Int = 0
    ) throws -> (inputIds: MLXArray, labels: MLXArray, lossMask: MLXArray) {
        guard !examples.isEmpty else {
            throw TextSFTTrainingBatchError.emptyBatch
        }
        let maxLength = examples.map(\.inputTokenIds.count).max() ?? 0
        guard maxLength > 0 else {
            throw TextSFTTrainingBatchError.notEnoughTokens
        }

        var inputValues: [Int32] = []
        var labelValues: [Int32] = []
        var maskValues: [Float] = []
        inputValues.reserveCapacity(examples.count * maxLength)
        labelValues.reserveCapacity(examples.count * maxLength)
        maskValues.reserveCapacity(examples.count * maxLength)

        for example in examples {
            guard example.inputTokenIds.count == example.labelTokenIds.count,
                  example.inputTokenIds.count == example.lossMask.count else {
                throw TextSFTTrainingBatchError.shiftedLengthMismatch
            }
            let padCount = maxLength - example.inputTokenIds.count
            inputValues.append(contentsOf: example.inputTokenIds.map(Int32.init))
            labelValues.append(contentsOf: example.labelTokenIds.map(Int32.init))
            maskValues.append(contentsOf: example.lossMask)
            if padCount > 0 {
                inputValues.append(contentsOf: Array(repeating: Int32(padTokenId), count: padCount))
                labelValues.append(contentsOf: Array(repeating: Int32(padTokenId), count: padCount))
                maskValues.append(contentsOf: Array(repeating: Float(0), count: padCount))
            }
        }

        return (
            inputIds: MLXArray(inputValues, [examples.count, maxLength]),
            labels: MLXArray(labelValues, [examples.count, maxLength]),
            lossMask: MLXArray(maskValues, [examples.count, maxLength])
        )
    }
}

public enum TextSFTTrainingLoss {
    public static func maskedNextTokenCrossEntropy(
        logits: MLXArray,
        labels: MLXArray,
        lossMask: MLXArray
    ) -> MLXArray {
        let logProbabilities = logSoftmax(logits.asType(.float32), axis: -1)
        let selected = takeAlong(
            logProbabilities,
            labels.asType(.int32).expandedDimensions(axis: -1),
            axis: -1
        ).squeezed(axis: -1)
        let mask = lossMask.asType(.float32)
        let denominator = mask.sum() + MLXArray(Float(1e-8))
        return -(selected * mask).sum() / denominator
    }
}

public enum TextSFTTrainingBatchError: Error, LocalizedError, Sendable {
    case emptyBatch
    case maxSequenceLengthTooSmall(Int)
    case messageTokenRoleMismatch
    case notEnoughTokens
    case noAssistantTargets
    case shiftedLengthMismatch

    public var errorDescription: String? {
        switch self {
        case .emptyBatch:
            return "Text SFT training batch is empty."
        case .maxSequenceLengthTooSmall(let value):
            return "Text SFT max sequence length must be >= 2 (got \(value))."
        case .messageTokenRoleMismatch:
            return "Text SFT tokenized messages and roles have different counts."
        case .notEnoughTokens:
            return "Text SFT example must contain at least two tokens after truncation."
        case .noAssistantTargets:
            return "Text SFT example contains no assistant target tokens."
        case .shiftedLengthMismatch:
            return "Text SFT shifted input, label, and mask lengths must match."
        }
    }
}
