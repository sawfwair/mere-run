import StudioKit
import SwiftUI

/// The canonical primary action button: accent-tinted, prominent, theme-aware in light and dark.
package struct MerePrimaryButtonStyle: ButtonStyle {
    package func makeBody(configuration: Configuration) -> some View {
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
    package static var merePrimary: MerePrimaryButtonStyle { MerePrimaryButtonStyle() }
}

/// A quiet secondary action, drawn as the design boards draw it: a 26pt `surfaceRaised` pill
/// with a `border` hairline at 60% and its label in 11.5pt medium `textPrimary`, so Replace,
/// Segment these, Save JSON, Re-track, Reveal, Cancel and Log read as filled controls rather
/// than the near-white cards a translucent `surface` fill produced.
package struct MereSecondaryButtonStyle: ButtonStyle {
    package func makeBody(configuration: Configuration) -> some View {
        MereSecondaryButtonBody(configuration: configuration)
    }

    private struct MereSecondaryButtonBody: View {
        let configuration: Configuration
        @State private var hovering = false

        var body: some View {
            configuration.label
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .padding(.horizontal, MereRunTheme.Spacing.sm)
                .frame(minHeight: 26)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(MereRunTheme.surfaceRaised)
                        .overlay {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                                .fill(hovering ? MereRunTheme.hoverFill : .clear)
                        }
                        .overlay {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                                .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
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
    package static var mereSecondary: MereSecondaryButtonStyle { MereSecondaryButtonStyle() }
}

/// Icon-only buttons that acknowledge the pointer: a soft fill on hover, a slight press dip.
/// Hover feedback is what separates a native-feeling control from a static glyph.
package struct MereIconButtonStyle: ButtonStyle {
    package var tint: Color = MereRunTheme.textSecondary

    package func makeBody(configuration: Configuration) -> some View {
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
    package static var mereIcon: MereIconButtonStyle { MereIconButtonStyle() }

    package static func mereIcon(tint: Color) -> MereIconButtonStyle { MereIconButtonStyle(tint: tint) }
}

extension View {
    /// The canonical text-field chrome (plain field over a themed panel), standardizing the
    /// repeated `.textFieldStyle(.plain).padding().merePanel()` pattern and replacing `.roundedBorder`.
    package func mereField(cornerRadius: CGFloat = MereRunTheme.cornerRadius) -> some View {
        textFieldStyle(.plain)
            .padding(MereRunTheme.Spacing.sm)
            .merePanel(cornerRadius: cornerRadius)
    }

    /// Row hover treatment for list-like custom rows (library, sidebar, model list).
    package func mereHoverRow(_ hovering: Bool, cornerRadius: CGFloat = MereRunTheme.Radius.md) -> some View {
        background {
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(hovering ? MereRunTheme.hoverFill : Color.clear)
        }
        .animation(MereRunTheme.Motion.quick, value: hovering)
    }
}

/// The Studio's segmented control: a 2pt-padded `surfaceRaised` pill whose selected segment is a
/// raised `segmentedSelection` tile. Every segment is a real button (VoiceOver reads it as a
/// selected/unselected button; Tab and Space work), so it keeps the semantics of a native picker
/// while drawing the way the design specifies.
package struct MereSegmentedControl<Item: Hashable>: View {
    package let items: [Item]
    @Binding package var selection: Item
    package let title: (Item) -> String
    /// Names the whole control for VoiceOver ("Image task", "Library scope").
    package var accessibilityLabelText: String?

    package init(
        _ items: [Item],
        selection: Binding<Item>,
        accessibilityLabel: String? = nil,
        title: @escaping (Item) -> String
    ) {
        self.items = items
        _selection = selection
        self.title = title
        accessibilityLabelText = accessibilityLabel
    }

    package var body: some View {
        HStack(spacing: 2) {
            ForEach(items, id: \.self) { item in
                MereSegment(
                    title: title(item),
                    isSelected: item == selection,
                    action: { selection = item }
                )
            }
        }
        .padding(2)
        .background {
            RoundedRectangle(cornerRadius: 7)
                .fill(MereRunTheme.surfaceRaised)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityLabelText ?? "Options")
    }
}

/// One segment of `MereSegmentedControl` (also its overflow "More" segment): 24pt tall, 12pt
/// horizontal padding, radius 5.5; selected = `segmentedSelection` fill, a 1pt shadow, semibold.
package struct MereSegment<Label: View>: View {
    package let isSelected: Bool
    package let action: () -> Void
    @ViewBuilder package let label: () -> Label

    @State private var hovering = false

    package init(isSelected: Bool, action: @escaping () -> Void, @ViewBuilder label: @escaping () -> Label) {
        self.isSelected = isSelected
        self.action = action
        self.label = label
    }

    package var body: some View {
        Button(action: action) {
            label()
                .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                .foregroundStyle(isSelected ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                .lineLimit(1)
                .padding(.horizontal, 12)
                .frame(height: 24)
                .background {
                    RoundedRectangle(cornerRadius: 5.5)
                        .fill(fill)
                        .shadow(
                            color: isSelected ? MereRunTheme.shadowColor : .clear,
                            radius: 1,
                            y: 1
                        )
                }
                .contentShape(RoundedRectangle(cornerRadius: 5.5))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var fill: Color {
        if isSelected { return MereRunTheme.segmentedSelection }
        if hovering { return MereRunTheme.hoverFill }
        return .clear
    }
}

extension MereSegment where Label == Text {
    package init(title: String, isSelected: Bool, action: @escaping () -> Void) {
        self.init(isSelected: isSelected, action: action) { Text(title) }
    }
}

/// A 28pt toolbar toggle: `accentSoft` tile and accent glyph while its panel is shown, quiet
/// `textSecondary` glyph otherwise, hover fill in between.
package struct MereToolbarIconButton: View {
    package let systemImage: String
    package let isActive: Bool
    package let action: () -> Void

    @State private var hovering = false

    package var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(isActive ? MereRunTheme.accent : MereRunTheme.textSecondary)
                .frame(width: 28, height: 28)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(fill)
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }

    private var fill: Color {
        if isActive { return MereRunTheme.accentSoft }
        if hovering { return MereRunTheme.hoverFill }
        return .clear
    }
}

/// The 10.5pt uppercase caption that labels sidebar sections, Library days, and panel groups.
package struct MereEyebrow: View {
    package let text: String

    package init(_ text: String) {
        self.text = text
    }

    package var body: some View {
        Text(text)
            .font(.system(size: 10.5, weight: .semibold))
            .kerning(0.63)
            .textCase(.uppercase)
            .foregroundStyle(MereRunTheme.textMuted)
            .lineLimit(1)
            .accessibilityAddTraits(.isHeader)
    }
}
