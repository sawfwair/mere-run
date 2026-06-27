import SwiftUI

/// The canonical primary action button: accent-tinted, prominent, theme-aware in light and dark.
struct MerePrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MereRunTheme.bodyFont.weight(.semibold))
            .padding(.horizontal, MereRunTheme.Spacing.md)
            .padding(.vertical, MereRunTheme.Spacing.xs)
            .frame(minHeight: 28)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                    .fill(MereRunTheme.accent.opacity(configuration.isPressed ? 0.78 : 1))
            }
            .foregroundStyle(MereRunTheme.background)
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.95 : 1)
    }
}

extension ButtonStyle where Self == MerePrimaryButtonStyle {
    static var merePrimary: MerePrimaryButtonStyle { MerePrimaryButtonStyle() }
}

extension View {
    /// The canonical text-field chrome (plain field over a themed panel), standardizing the
    /// repeated `.textFieldStyle(.plain).padding().merePanel()` pattern and replacing `.roundedBorder`.
    func mereField(cornerRadius: CGFloat = MereRunTheme.cornerRadius) -> some View {
        textFieldStyle(.plain)
            .padding(MereRunTheme.Spacing.sm)
            .merePanel(cornerRadius: cornerRadius)
    }
}
