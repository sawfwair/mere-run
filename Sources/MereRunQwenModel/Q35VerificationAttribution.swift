import MLX

/// Synchronized debug attribution; release builds have no observer.
package enum Q35VerificationAttribution {
    package enum Stage: String, Sendable {
        case attentionNorm, attentionResidual, feedForwardNorm, feedForwardResidual
        case linearInput, linearPrework, linearRecurrent, linearNormGate, linearOutput
        case fullInput, fullRotaryCache, fullAttention, fullGate, fullOutput
        case feedForwardGateUp, feedForwardDown, head
    }

    package typealias Observer = @Sendable (Stage, [MLXArray]) -> Void

    #if DEBUG
    @TaskLocal package static var observer: Observer?
    #else
    @inline(__always) package static var observer: Observer? { nil }
    #endif
}
