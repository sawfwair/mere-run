import Foundation

enum FusedAttentionPolicy {
    /// Emergency fallback for MLX's fused scaled-dot-product attention paths.
    /// Supported attention shapes use the fused implementation by default.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_FUSED_SDPA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
}
