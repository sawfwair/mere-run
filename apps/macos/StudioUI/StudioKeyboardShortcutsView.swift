import StudioKit
import SwiftUI

// The menu bar's side of `StudioKeyboardShortcuts`: each combo as a SwiftUI `KeyboardShortcut`,
// the Help ▸ Keyboard Shortcuts window, and the two things a shortcut reaches into a view for —
// focusing a list's search field, and the Library row ⌘⌫ deletes.

extension StudioKeyCombo {
    /// The combo as a menu item binds it.
    package var keyboardShortcut: KeyboardShortcut {
        KeyboardShortcut(keyEquivalent, modifiers: eventModifiers)
    }

    private var keyEquivalent: KeyEquivalent {
        switch key {
        case .character(let character): return KeyEquivalent(character)
        case .returnKey: return .return
        case .delete: return .delete
        case .space: return .space
        case .upArrow: return .upArrow
        case .downArrow: return .downArrow
        case .escape: return .escape
        }
    }

    private var eventModifiers: EventModifiers {
        var result: EventModifiers = []
        if modifiers.contains(.control) { result.insert(.control) }
        if modifiers.contains(.option) { result.insert(.option) }
        if modifiers.contains(.shift) { result.insert(.shift) }
        if modifiers.contains(.command) { result.insert(.command) }
        return result
    }
}

extension View {
    /// Binds the table's key for `id` to this menu item.
    package func studioShortcut(_ id: StudioShortcutID) -> some View {
        keyboardShortcut(StudioKeyboardShortcuts.combo(id).keyboardShortcut)
    }
}

// MARK: - Help ▸ Keyboard Shortcuts

/// The Keyboard Shortcuts window's identity, shared by the scene and the Help menu.
package enum StudioKeyboardShortcutsWindow {
    package static let id = "keyboard-shortcuts"
    package static let title = "Keyboard Shortcuts"
}

/// Every Studio shortcut, grouped by the menu it is in, read from `StudioKeyboardShortcuts` so
/// the window and the menus cannot disagree.
package struct StudioKeyboardShortcutsView: View {
    package init() {}

    package var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Keyboard shortcuts")
                        .font(MereRunTheme.titleFont)
                        .foregroundStyle(MereRunTheme.textPrimary)
                    Text("Menu shortcuts work anywhere in the Studio window. Library and feed keys work while that list has focus; prompt keys while you type a prompt.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(StudioKeyboardShortcuts.helpSections, id: \.title) { section in
                    VStack(alignment: .leading, spacing: 6) {
                        MereEyebrow(section.title)
                        VStack(spacing: 0) {
                            ForEach(section.rows) { row in
                                StudioShortcutRow(row: row)
                                if row != section.rows.last {
                                    Divider().overlay(MereRunTheme.border.opacity(0.4))
                                }
                            }
                        }
                        .background {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                                .fill(MereRunTheme.surface)
                                .overlay {
                                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                                        .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                                }
                        }
                    }
                }
            }
            .padding(MereRunTheme.Spacing.xl)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }
}

private struct StudioShortcutRow: View {
    let row: StudioKeyboardShortcuts.HelpRow

    var body: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(row.title)
                .font(.callout)
                .foregroundStyle(MereRunTheme.textPrimary)
            Spacer(minLength: MereRunTheme.Spacing.sm)
            Text(row.keys)
                .font(.callout.weight(.medium).monospaced())
                .foregroundStyle(MereRunTheme.textSecondary)
                .padding(.horizontal, 7)
                .frame(minHeight: 22)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(MereRunTheme.surfaceRaised)
                }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(row.title): \(row.keys)")
    }
}

// MARK: - Search focus

/// A request from Edit ▸ Find in List (⌘F) or Search Library (⌘L) for a list's search field to
/// take focus. It carries when it was made so a field that appears a moment later (the Library
/// column the same command just showed) still takes it, and one that appears long after does not.
package struct StudioSearchFocusRequest: Equatable {
    package let field: StudioSearchField
    package let at: Date

    package init(field: StudioSearchField, at: Date = Date()) {
        self.field = field
        self.at = at
    }

    /// How long a request stays good for a field that was not on screen when it was made.
    static let lifetime: TimeInterval = 1
}

private struct StudioSearchFocusRequestKey: EnvironmentKey {
    static let defaultValue: StudioSearchFocusRequest? = nil
}

extension EnvironmentValues {
    /// The latest search-focus request; the Studio window sets it from `NavigationModel`.
    package var studioSearchFocusRequest: StudioSearchFocusRequest? {
        get { self[StudioSearchFocusRequestKey.self] }
        set { self[StudioSearchFocusRequestKey.self] = newValue }
    }
}

extension View {
    /// Focuses `focused` when a search-focus request names `field`; a nil field answers none.
    func studioSearchFocus(_ field: StudioSearchField?, focused: FocusState<Bool>.Binding) -> some View {
        modifier(StudioSearchFocusModifier(field: field, focused: focused))
    }
}

private struct StudioSearchFocusModifier: ViewModifier {
    let field: StudioSearchField?
    let focused: FocusState<Bool>.Binding
    @Environment(\.studioSearchFocusRequest) private var request

    func body(content: Content) -> some View {
        content.onChange(of: request, initial: true) { _, request in
            guard let request, request.field == field,
                  Date().timeIntervalSince(request.at) < StudioSearchFocusRequest.lifetime else { return }
            focused.wrappedValue = true
        }
    }
}

// MARK: - Library deletion

/// What File ▸ Delete from Library… (⌘⌫) deletes: the Library list publishes it only while the
/// list itself has focus, so the key never reaches past a text field's own ⌘⌫.
package struct StudioLibraryDeletion {
    package let count: Int
    /// Asks the Library's own confirmation, which offers to keep or trash the files and deletes
    /// as one undo step.
    package let confirm: () -> Void

    package var title: String {
        count > 1 ? "Delete \(count) Runs from Library…" : "Delete from Library…"
    }
}

private struct StudioLibraryDeletionKey: FocusedValueKey { typealias Value = StudioLibraryDeletion }

extension FocusedValues {
    package var studioLibraryDeletion: StudioLibraryDeletion? {
        get { self[StudioLibraryDeletionKey.self] }
        set { self[StudioLibraryDeletionKey.self] = newValue }
    }
}
