import Foundation

/// Architecture of the publisher's original BF16 FLUX 3 Klein checkpoint.
public struct Flux3Configuration: Sendable {
    public var hiddenSize: Int
    public var heads: Int
    public var earlyDepth: Int
    public var singleDepth: Int
    public var imageChannels: Int
    public var textChannels: Int
    public var mlpRatio: Int
    public var axesDimensions: [Int]

    public init(hiddenSize: Int = 3072, heads: Int = 24, earlyDepth: Int = 5,
                singleDepth: Int = 28, imageChannels: Int = 128, textChannels: Int = 20480,
                mlpRatio: Int = 3, axesDimensions: [Int] = [32, 32, 32, 32]) {
        precondition(hiddenSize > 0 && heads > 0 && hiddenSize.isMultiple(of: heads))
        precondition(axesDimensions.count == 4 && axesDimensions.allSatisfy { $0 > 0 && $0.isMultiple(of: 2) })
        precondition(axesDimensions.reduce(0, +) == hiddenSize / heads)
        precondition(earlyDepth >= 0 && singleDepth >= 0 && imageChannels > 0 && textChannels > 0 && mlpRatio > 0)
        self.hiddenSize = hiddenSize
        self.heads = heads
        self.earlyDepth = earlyDepth
        self.singleDepth = singleDepth
        self.imageChannels = imageChannels
        self.textChannels = textChannels
        self.mlpRatio = mlpRatio
        self.axesDimensions = axesDimensions
    }
}

public enum Flux3Error: LocalizedError {
    case invalidConfiguration(String)
    case invalidWeights(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail): "Invalid FLUX 3 configuration: \(detail)"
        case .invalidWeights(let detail): "Invalid FLUX 3 weights: \(detail)"
        }
    }
}

public enum Flux3Sampling {
    public static let defaultSteps = 16
    public static let defaultAlpha: Float = 6.93

    /// The distilled checkpoint predicts a flow between consecutive shifted times.
    public static func schedule(steps: Int, alpha: Float = defaultAlpha) throws -> [Float] {
        guard (1...1000).contains(steps), alpha.isFinite, alpha > 0 else {
            throw Flux3Error.invalidConfiguration("Use 1–1000 steps and a positive finite alpha.")
        }
        return (0...steps).map {
            let time = 1 - Float($0) / Float(steps)
            return alpha * time / (1 + (alpha - 1) * time)
        }
    }
}

public enum Flux3PromptPlan {
    public static let outputLayers = [4, 8, 12, 16, 20, 24, 28, 32]
    public static let maximumLength = 4096

    public static func presentation(_ prompt: String) -> String {
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    }

    public static func paddedLength(tokenCount: Int) -> Int {
        precondition(tokenCount > 0)
        let count = min(tokenCount, maximumLength)
        return min((count + 79) / 80 * 80, maximumLength)
    }
}
