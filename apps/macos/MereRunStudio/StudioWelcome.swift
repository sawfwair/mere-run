import SwiftUI

/// First run: the promise in one serif line, what lives here in three, and the single step
/// that unlocks everything (pulling a first model) as the primary action.
struct StudioWelcomeSheet: View {
    let resolvedCLI: String
    let onBrowseModels: () -> Void
    let onDone: () -> Void

    private static let heroModes: [String] = [
        "photo", "film", "music.note", "waveform", "bubble.left.and.bubble.right", "scope"
    ]

    private let highlights: [(String, String, String)] = [
        ("photo", "Create locally", "Images, video, music, sound effects, and speech — all on-device."),
        ("bubble.left.and.bubble.right", "Chat & code", "Run local language models for chat, code, OCR, and vision."),
        ("lock.shield", "Private by default", "Nothing leaves your Mac. The app drives the public mere.run CLI underneath.")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.xl) {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                Text("mere.run")
                    .font(.system(size: 13, weight: .semibold, design: .serif))
                    .foregroundStyle(MereRunTheme.textMuted)

                Text("Create anything.\nLocally.")
                    .font(.system(.largeTitle, design: .serif, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineSpacing(3)

                HStack(spacing: MereRunTheme.Spacing.xs) {
                    ForEach(Self.heroModes, id: \.self) { symbol in
                        Image(systemName: symbol)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(MereRunTheme.accent)
                            .frame(width: 32, height: 32)
                            .background {
                                Circle().fill(MereRunTheme.accentSoft.opacity(0.8))
                            }
                    }
                }
                .accessibilityHidden(true)
            }

            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                ForEach(Array(highlights.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .top, spacing: MereRunTheme.Spacing.sm) {
                        Image(systemName: item.0)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(MereRunTheme.accent)
                            .frame(width: 24)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.1)
                                .font(.system(size: 13.5, weight: .semibold))
                            Text(item.2)
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.xs) {
                Text("Start by pulling a model for the mode you want — the catalog handles the rest.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: MereRunTheme.Spacing.sm) {
                    Button {
                        onBrowseModels()
                    } label: {
                        Label("Browse models", systemImage: "shippingbox")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.merePrimary)

                    Button("Start creating") {
                        onDone()
                    }
                    .buttonStyle(.bordered)
                    .keyboardShortcut(.defaultAction)
                }
            }

            HStack(spacing: 7) {
                Image(systemName: "terminal")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                Text(resolvedCLI)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(MereRunTheme.Spacing.xxl)
        .frame(width: 460)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }
}
