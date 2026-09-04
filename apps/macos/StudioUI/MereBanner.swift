import StudioKit
import SwiftUI

/// The single inline banner used across Studio, Advanced, and Settings. Severity is the source of
/// truth for color + icon so failures read red, warnings yellow, and info accent — consistently.
struct MereBanner: View {
    enum Severity {
        case info
        case warning
        case error

        var tint: Color {
            switch self {
            case .info: return MereRunTheme.accent
            case .warning: return MereRunTheme.yellow
            case .error: return MereRunTheme.red
            }
        }

        var systemImage: String {
            switch self {
            case .info: return "info.circle"
            case .warning: return "exclamationmark.triangle.fill"
            case .error: return "exclamationmark.octagon.fill"
            }
        }

        var voiceOverPrefix: String {
            switch self {
            case .info: return ""
            case .warning: return "Warning: "
            case .error: return "Error: "
            }
        }
    }

    let severity: Severity
    let text: String
    var systemImage: String?
    var onDismiss: (() -> Void)?

    var body: some View {
        HStack(spacing: MereRunTheme.Spacing.xs) {
            Image(systemName: systemImage ?? severity.systemImage)
                .foregroundStyle(severity.tint)
            Text(text)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(3)
            Spacer(minLength: 0)
            if let onDismiss {
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(MereRunTheme.textMuted)
                .accessibilityLabel("Dismiss")
            }
        }
        .padding(MereRunTheme.Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(severity.tint.opacity(0.12))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                        .strokeBorder(severity.tint.opacity(0.35), lineWidth: 1)
                }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(severity.voiceOverPrefix + text)
    }
}
