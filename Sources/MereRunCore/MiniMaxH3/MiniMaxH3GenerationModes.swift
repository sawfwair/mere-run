import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

public struct MiniMaxH3ReferenceInput: Sendable, Hashable {
    public let kind: MiniMaxH3ReferenceKind
    public let url: URL

    public init(kind: MiniMaxH3ReferenceKind, url: URL) {
        self.kind = kind
        self.url = url.standardizedFileURL
    }
}

public struct MiniMaxH3FrameInput: Sendable, Hashable {
    public let frameIndex: Int
    public let url: URL

    public init(frameIndex: Int, url: URL) {
        self.frameIndex = frameIndex
        self.url = url.standardizedFileURL
    }
}

public enum MiniMaxH3TransformerWeightMode: String, Sendable, Hashable {
    case automatic = "auto"
    case quantized
    case residentBF16 = "resident-bf16"
}

public enum MiniMaxH3AccelerationMode: String, Sendable, Hashable {
    case quality
    case balanced
    case maximum
    case layers45 = "layers-45"
    case layers40 = "layers-40"
    case velocityReuse2 = "velocity-reuse-2"
    case tokenReduction = "token-reduction"

    var adaptiveFirstBlockCachePolicy: MiniMaxH3AdaptiveFirstBlockCachePolicy? {
        switch self {
        case .quality, .layers45, .layers40, .velocityReuse2, .tokenReduction: nil
        case .balanced:
            MiniMaxH3AdaptiveFirstBlockCachePolicy(
                globalThreshold: 0.08,
                temporalThreshold: 0.12,
                window: 0.1...0.9,
                maximumConsecutiveCachedSteps: 2,
                requiredFinalFullSteps: 2
            )
        case .maximum:
            MiniMaxH3AdaptiveFirstBlockCachePolicy(
                globalThreshold: 0.30,
                temporalThreshold: 0.40,
                window: 0.1...0.95,
                maximumConsecutiveCachedSteps: 4,
                requiredFinalFullSteps: 1
            )
        }
    }

    var dynamicSparseAttentionPolicy: DynamicSparseAttentionPolicy? {
        switch self {
        case .quality, .layers45, .layers40, .velocityReuse2, .tokenReduction: nil
        case .balanced:
            DynamicSparseAttentionPolicy(
                thresholdStandardDeviations: 0.75
            )
        case .maximum:
            DynamicSparseAttentionPolicy(
                thresholdStandardDeviations: 1
            )
        }
    }

    // Retained as an internal benchmark baseline. Production acceleration
    // selects the adaptive first-block cache unless explicitly overridden.
    var blockReusePolicy: MiniMaxH3BlockReusePolicy? {
        switch self {
        case .quality, .layers45, .layers40, .velocityReuse2, .tokenReduction: nil
        case .balanced:
            MiniMaxH3BlockReusePolicy(
                cacheDepth: 0.5,
                window: 0.1...0.9,
                maximumConsecutiveCachedSteps: 2
            )
        case .maximum:
            MiniMaxH3BlockReusePolicy(
                cacheDepth: 0.82,
                window: 0.1...0.9,
                maximumConsecutiveCachedSteps: 4
            )
        }
    }

    var velocityReusePolicy: MiniMaxH3VelocityReusePolicy? {
        switch self {
        case .quality, .balanced, .maximum, .layers45, .layers40, .tokenReduction: nil
        case .velocityReuse2: MiniMaxH3VelocityReusePolicy(interval: 2)
        }
    }

    var layerThinningPolicy: MiniMaxH3LayerThinningPolicy? {
        switch self {
        case .quality, .balanced, .maximum, .velocityReuse2, .tokenReduction: nil
        case .layers45: MiniMaxH3LayerThinningPolicy(activeBlockCount: 45)
        case .layers40: MiniMaxH3LayerThinningPolicy(activeBlockCount: 40)
        }
    }

    var tokenReductionPolicy: MiniMaxH3TokenReductionPolicy? {
        switch self {
        case .quality, .balanced, .maximum, .layers45, .layers40, .velocityReuse2: nil
        case .tokenReduction: MiniMaxH3TokenReductionPolicy()
        }
    }
}
