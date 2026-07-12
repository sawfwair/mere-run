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

/// A quiet secondary action: themed surface + border with hover feedback, replacing the
/// native `.bordered` look so custom panels stay visually consistent in both appearances.
struct MereSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        MereSecondaryButtonBody(configuration: configuration)
    }

    private struct MereSecondaryButtonBody: View {
        let configuration: Configuration
        @State private var hovering = false

        var body: some View {
            configuration.label
                .font(MereRunTheme.captionFont)
                .foregroundStyle(hovering ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                .padding(.horizontal, MereRunTheme.Spacing.sm)
                .frame(minHeight: 26)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(hovering ? MereRunTheme.surfaceRaised : MereRunTheme.surface.opacity(0.5))
                        .overlay {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                                .strokeBorder(MereRunTheme.border.opacity(0.7), lineWidth: 1)
                        }
                }
                .contentShape(Rectangle())
                .scaleEffect(configuration.isPressed ? 0.98 : 1)
                .onHover { hovering = $0 }
                .animation(MereRunTheme.Motion.quick, value: hovering)
                .animation(MereRunTheme.Motion.quick, value: configuration.isPressed)
        }
    }
}

extension ButtonStyle where Self == MereSecondaryButtonStyle {
    static var mereSecondary: MereSecondaryButtonStyle { MereSecondaryButtonStyle() }
}

/// Icon-only buttons that acknowledge the pointer: a soft fill on hover, a slight press dip.
/// Hover feedback is what separates a native-feeling control from a static glyph.
struct MereIconButtonStyle: ButtonStyle {
    var tint: Color = MereRunTheme.textSecondary

    func makeBody(configuration: Configuration) -> some View {
        MereIconButtonBody(configuration: configuration, tint: tint)
    }

    private struct MereIconButtonBody: View {
        let configuration: Configuration
        let tint: Color
        @State private var hovering = false

        var body: some View {
            configuration.label
                .foregroundStyle(hovering ? MereRunTheme.textPrimary : tint)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(hovering ? MereRunTheme.hoverFill : Color.clear)
                }
                .scaleEffect(configuration.isPressed ? 0.94 : 1)
                .onHover { hovering = $0 }
                .animation(MereRunTheme.Motion.quick, value: hovering)
                .animation(MereRunTheme.Motion.quick, value: configuration.isPressed)
        }
    }
}

extension ButtonStyle where Self == MereIconButtonStyle {
    static var mereIcon: MereIconButtonStyle { MereIconButtonStyle() }

    static func mereIcon(tint: Color) -> MereIconButtonStyle { MereIconButtonStyle(tint: tint) }
}

extension View {
    /// The canonical text-field chrome (plain field over a themed panel), standardizing the
    /// repeated `.textFieldStyle(.plain).padding().merePanel()` pattern and replacing `.roundedBorder`.
    func mereField(cornerRadius: CGFloat = MereRunTheme.cornerRadius) -> some View {
        textFieldStyle(.plain)
            .padding(MereRunTheme.Spacing.sm)
            .merePanel(cornerRadius: cornerRadius)
    }

    /// Row hover treatment for list-like custom rows (library, sidebar, model list).
    func mereHoverRow(_ hovering: Bool, cornerRadius: CGFloat = MereRunTheme.Radius.md) -> some View {
        background {
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(hovering ? MereRunTheme.hoverFill : Color.clear)
        }
        .animation(MereRunTheme.Motion.quick, value: hovering)
    }
}
