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

/// The Studio's segmented control: a 2pt-padded `surfaceRaised` pill whose selected segment is a
/// raised `segmentedSelection` tile. Every segment is a real button (VoiceOver reads it as a
/// selected/unselected button; Tab and Space work), so it keeps the semantics of a native picker
/// while drawing the way the design specifies.
struct MereSegmentedControl<Item: Hashable>: View {
    let items: [Item]
    @Binding var selection: Item
    let title: (Item) -> String
    /// Names the whole control for VoiceOver ("Image task", "Library scope").
    var accessibilityLabelText: String?

    init(
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

    var body: some View {
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
struct MereSegment<Label: View>: View {
    let isSelected: Bool
    let action: () -> Void
    @ViewBuilder let label: () -> Label

    @State private var hovering = false

    init(isSelected: Bool, action: @escaping () -> Void, @ViewBuilder label: @escaping () -> Label) {
        self.isSelected = isSelected
        self.action = action
        self.label = label
    }

    var body: some View {
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
    init(title: String, isSelected: Bool, action: @escaping () -> Void) {
        self.init(isSelected: isSelected, action: action) { Text(title) }
    }
}

/// A 28pt toolbar toggle: `accentSoft` tile and accent glyph while its panel is shown, quiet
/// `textSecondary` glyph otherwise, hover fill in between.
struct MereToolbarIconButton: View {
    let systemImage: String
    let isActive: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
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
struct MereEyebrow: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.system(size: 10.5, weight: .semibold))
            .kerning(0.63)
            .textCase(.uppercase)
            .foregroundStyle(MereRunTheme.textMuted)
            .lineLimit(1)
            .accessibilityAddTraits(.isHeader)
    }
}
