import Foundation
import MLX

public struct LoRAWeights: @unchecked Sendable {
    public let weights: [String: (down: MLXArray, up: MLXArray)]
    public let rank: Int
    public let alpha: Float
    public let targetRanks: [String: Int]

    public init(
        weights: [String: (down: MLXArray, up: MLXArray)],
        rank: Int,
        alpha: Float? = nil,
        targetRanks: [String: Int] = [:]
    ) {
        self.weights = weights
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
        self.targetRanks = targetRanks
    }

    public var effectiveScale: Float {
        guard rank > 0 else { return 1.0 }
        return alpha / Float(rank)
    }
}

public enum LoRAError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)
    case noWeightPairs

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "LoRA file not found: \(path)"
        case .invalidFormat(let message):
            return "Invalid LoRA format: \(message)"
        case .noWeightPairs:
            return "No valid LoRA weight pairs found."
        }
    }
}
