import Foundation
import MereRunEnvironment

public enum FusedAttentionPolicy {
    /// Emergency fallback for MLX's fused scaled-dot-product attention paths.
    /// Supported attention shapes use the fused implementation by default.
    public static let enabled: Bool = MereRunEnvironmentSwitch.isEnabled("MERERUN_FUSED_SDPA", default: true)
}
