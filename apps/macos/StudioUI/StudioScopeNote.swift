import StudioKit
import SwiftUI

/// The note a surface shows about its model's options (`StudioScopeNotice`): the values the
/// model does not use, kept for when the user switches back, and the values its family runs in
/// place of the draft's; or, while the CLI identifies a local folder, that every option shows
/// until it knows. One copy shows at a time: at the top of the inspector, beside the controls it
/// explains; as the Command view's "Not sent" line, beside the argv; and under the composer's
/// chips only while neither column is open. One quiet style reads as information, not an error.
struct StudioScopeNote: View {
    let notice: StudioScopeNotice
    /// The eyebrow above the note: the Command view says "Not sent".
    var eyebrow: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let eyebrow { MereEyebrow(eyebrow) }
            HStack(alignment: .firstTextBaseline, spacing: MereRunTheme.Spacing.xs) {
                glyph
                VStack(alignment: .leading, spacing: 3) {
                    Text(notice.title)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    ForEach(notice.details, id: \.self) { line in
                        Text(line)
                            .font(.caption)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, MereRunTheme.Spacing.sm)
            .padding(.vertical, MereRunTheme.Spacing.xs)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                    .fill(MereRunTheme.surfaceRaised.opacity(0.7))
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                            .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                    }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(eyebrow.map { "\($0): \(notice.accessibilityLabel)" } ?? notice.accessibilityLabel)
    }

    @ViewBuilder
    private var glyph: some View {
        if let systemImage = Self.systemImage(for: notice) {
            Image(systemName: systemImage)
                .font(.caption.weight(.semibold))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 14)
        } else {
            ProgressView()
                .controlSize(.mini)
                .frame(width: 14)
        }
    }

    /// The note's glyph; nil while the CLI identifies the model, which shows a spinner. Only a
    /// folder the CLI could not identify gets the folder glyph: a model id or the default gets
    /// a plain question mark.
    static func systemImage(for notice: StudioScopeNotice) -> String? {
        switch notice.kind {
        case .identifying: return nil
        case .unidentified: return notice.isFolder ? "questionmark.folder" : "questionmark.circle"
        case .unused: return "eye.slash"
        }
    }
}

private struct StudioScopeSourceKey: EnvironmentKey {
    static let defaultValue = StudioScopeSource.contract
}

extension EnvironmentValues {
    /// Where the surfaces below read their `StudioOptionScope`s: the shipped contract and the
    /// controller's `catalog resolve` answers (`MereRunController.scopeSource`), which the app
    /// sets at each window's root. Unset, it is the contract alone. Offscreen renders set a
    /// fixed source, so a routed command's scoping renders without a CLI.
    package var studioScopeSource: StudioScopeSource {
        get { self[StudioScopeSourceKey.self] }
        set { self[StudioScopeSourceKey.self] = newValue }
    }
}
