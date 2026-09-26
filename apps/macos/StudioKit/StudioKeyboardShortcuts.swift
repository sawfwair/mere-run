import Foundation

// Studio's keyboard shortcuts as one table. The menu bar binds its items from it, the Help ▸
// Keyboard Shortcuts window lists it, and the tests hold it unique and clear of the keys macOS
// and text editing already own. StudioKit stays SwiftUI-free: `StudioUI` turns a `StudioKeyCombo`
// into a `KeyboardShortcut`.

/// A key with its modifiers, as a menu shows it.
package struct StudioKeyCombo: Hashable, Sendable {
    package enum Key: Hashable, Sendable {
        /// A printable key, lowercased: "l", "1", ".", "?".
        case character(Character)
        case returnKey
        case delete
        case space
        case upArrow
        case downArrow
        case escape
    }

    package struct Modifiers: OptionSet, Hashable, Sendable {
        package let rawValue: Int
        package init(rawValue: Int) { self.rawValue = rawValue }

        package static let control = Modifiers(rawValue: 1 << 0)
        package static let option = Modifiers(rawValue: 1 << 1)
        package static let shift = Modifiers(rawValue: 1 << 2)
        package static let command = Modifiers(rawValue: 1 << 3)
    }

    package let key: Key
    package let modifiers: Modifiers

    package init(_ key: Key, _ modifiers: Modifiers = []) {
        self.key = key
        self.modifiers = modifiers
    }

    package init(_ character: Character, _ modifiers: Modifiers = []) {
        self.init(.character(character), modifiers)
    }

    /// The combo the way macOS menus print it: modifiers in ⌃⌥⇧⌘ order, then the key.
    package var symbols: String {
        var text = ""
        if modifiers.contains(.control) { text += "⌃" }
        if modifiers.contains(.option) { text += "⌥" }
        if modifiers.contains(.shift) { text += "⇧" }
        if modifiers.contains(.command) { text += "⌘" }
        switch key {
        case .character(let character): text += String(character).uppercased()
        case .returnKey: text += "↩"
        case .delete: text += "⌫"
        case .space: text += "Space"
        case .upArrow: text += "↑"
        case .downArrow: text += "↓"
        case .escape: text += "⎋"
        }
        return text
    }
}

/// Where a shortcut is live. A window shortcut is a menu key equivalent, which AppKit checks
/// before the focused view sees the key, so it must never take a key a text field uses. The
/// others are handled by the view that has focus and only while it has it.
package enum StudioShortcutContext: String, CaseIterable, Sendable {
    /// A menu item: live anywhere in the window.
    case window
    /// While the Library list or the feed has focus, never a text field.
    case list
    /// While the composer's prompt has focus.
    case prompt
    /// While Compare shows sounds or videos (it takes focus when it opens).
    case compare
}

/// Every action Studio binds a key to.
package enum StudioShortcutID: Hashable, Sendable {
    case newChat
    case quickLook
    case deleteFromLibrary
    case find
    case searchLibrary
    case showSidebar
    case showLibrary
    case showInspector
    case showCommandView
    case domain(StudioDomain)
    case run
    case stop
    case openLastOutput
    case revealLastOutput
    case commandConsole
    case guide
    case quickLookSelection
    case olderPrompt
    case newerPrompt
    case comparePlayPause
    case compareSide
}

/// One row of the table: the action, the menu it is in, what it says there, and its key.
package struct StudioShortcut: Identifiable, Hashable, Sendable {
    package let id: StudioShortcutID
    /// The menu the item is in, or where it works for a shortcut no menu shows.
    package let menu: String
    package let title: String
    package let combo: StudioKeyCombo
    package let context: StudioShortcutContext
}

/// The list a search shortcut focuses: the Library column (or Chat's thread list in its place),
/// or the list a page is made of (Models, Adapters, Plugins).
package enum StudioSearchField: String, Hashable, Sendable {
    case library
    case page
}

package enum StudioKeyboardShortcuts {
    /// The table, in menu order.
    package static let all: [StudioShortcut] = {
        let fixed: [StudioShortcut] = [
            StudioShortcut(id: .newChat, menu: "File", title: "New Chat", combo: .init("n", .command), context: .window),
            StudioShortcut(id: .quickLook, menu: "File", title: "Quick Look", combo: .init("y", .command), context: .window),
            StudioShortcut(
                id: .deleteFromLibrary, menu: "File", title: "Delete from Library…",
                combo: .init(.delete, .command), context: .list
            ),
            StudioShortcut(id: .find, menu: "Edit", title: "Find in List", combo: .init("f", .command), context: .window),
            StudioShortcut(
                id: .searchLibrary, menu: "Edit", title: "Search Library", combo: .init("l", .command), context: .window
            ),
            StudioShortcut(
                id: .showSidebar, menu: "View", title: "Show Sidebar", combo: .init("s", [.control, .command]), context: .window
            ),
            StudioShortcut(
                id: .showLibrary, menu: "View", title: "Show Library", combo: .init("l", [.shift, .command]), context: .window
            ),
            StudioShortcut(
                id: .showInspector, menu: "View", title: "Show Inspector", combo: .init("e", .command), context: .window
            ),
            StudioShortcut(
                id: .showCommandView, menu: "View", title: "Show Command View",
                combo: .init("c", [.option, .command]), context: .window
            ),
        ]
        let domains = StudioDomain.allCases.compactMap { domain -> StudioShortcut? in
            guard let combo = domainCombo(domain) else { return nil }
            return StudioShortcut(id: .domain(domain), menu: "Go", title: domain.title, combo: combo, context: .window)
        }
        let rest: [StudioShortcut] = [
            StudioShortcut(id: .run, menu: "Run", title: "Run", combo: .init(.returnKey, .command), context: .window),
            StudioShortcut(id: .stop, menu: "Run", title: "Stop", combo: .init(".", .command), context: .window),
            StudioShortcut(
                id: .openLastOutput, menu: "Run", title: "Open Last Output", combo: .init("o", [.shift, .command]),
                context: .window
            ),
            StudioShortcut(
                id: .revealLastOutput, menu: "Run", title: "Reveal Last Output in Finder",
                combo: .init("r", [.shift, .command]), context: .window
            ),
            StudioShortcut(
                id: .commandConsole, menu: "Window", title: "Command Console", combo: .init("c", [.shift, .command]),
                context: .window
            ),
            StudioShortcut(id: .guide, menu: "Help", title: "mere.run Guide", combo: .init("?", .command), context: .window),
            StudioShortcut(
                id: .quickLookSelection, menu: "Library and feed", title: "Quick Look the selected run",
                combo: .init(.space), context: .list
            ),
            StudioShortcut(
                id: .olderPrompt, menu: "Prompt", title: "Previous prompt on this page",
                combo: .init(.upArrow), context: .prompt
            ),
            StudioShortcut(
                id: .newerPrompt, menu: "Prompt", title: "Next prompt, then what you were typing",
                combo: .init(.downArrow), context: .prompt
            ),
            StudioShortcut(
                id: .comparePlayPause, menu: "Compare", title: "Play or pause", combo: .init(.space), context: .compare
            ),
            // 1 stands for the row: 2…8 hear the second to eighth pane the same way.
            StudioShortcut(
                id: .compareSide, menu: "Compare", title: "Hear another pane at the same moment", combo: .init("1"),
                context: .compare
            ),
        ]
        return fixed + domains + rest
    }()

    package static func shortcut(_ id: StudioShortcutID) -> StudioShortcut {
        guard let shortcut = all.first(where: { $0.id == id }) else {
            preconditionFailure("No keyboard shortcut is declared for \(id)")
        }
        return shortcut
    }

    package static func combo(_ id: StudioShortcutID) -> StudioKeyCombo {
        shortcut(id).combo
    }

    /// "Show Library (⇧⌘L)": a tooltip naming the item's shortcut.
    package static func help(_ title: String, _ id: StudioShortcutID) -> String {
        "\(title) (\(combo(id).symbols))"
    }

    /// ⌘1…⌘9 for the first nine sidebar sections, ⌥⌘1… for the rest, in sidebar order.
    package static func domainCombo(_ domain: StudioDomain) -> StudioKeyCombo? {
        guard let index = StudioDomain.allCases.firstIndex(of: domain) else { return nil }
        let (digit, modifiers): (Int, StudioKeyCombo.Modifiers) = index < 9
            ? (index + 1, .command) : (index - 8, [.option, .command])
        guard digit <= 9, let key = String(digit).first else { return nil }
        return StudioKeyCombo(key, modifiers)
    }

    /// The search ⌘F focuses on `task`: its page's own list, or the Library column beside a
    /// Generate, Converse, or Analyze task. nil where there is no list to search.
    package static func findField(for task: StudioTask) -> StudioSearchField? {
        switch task {
        case .modelsInstalled, .modelsAdapters, .pluginsCatalog: return .page
        default: return task.showsPromptChrome ? .library : nil
        }
    }

    // MARK: Keys Studio must leave alone

    /// Keys macOS or every Mac app's standard menus own. No Studio shortcut may take one.
    package static let reservedBySystem: [StudioKeyCombo: String] = [
        .init("q", .command): "Quit",
        .init("h", .command): "Hide",
        .init("h", [.option, .command]): "Hide Others",
        .init("m", .command): "Minimize",
        .init("w", .command): "Close Window",
        .init(",", .command): "Settings",
        .init("`", .command): "Cycle Windows",
        .init("f", [.control, .command]): "Full Screen",
        .init("z", .command): "Undo",
        .init("z", [.shift, .command]): "Redo",
        .init("x", .command): "Cut",
        .init("c", .command): "Copy",
        .init("v", .command): "Paste",
        .init("v", [.option, .shift, .command]): "Paste and Match Style",
        .init("a", .command): "Select All",
        .init(.space, .command): "Spotlight",
        .init(.space, .control): "Input Sources",
        .init("q", [.control, .command]): "Lock Screen",
        .init("q", [.shift, .command]): "Log Out",
        .init("d", [.option, .command]): "Dock Hiding",
        .init(.escape, [.option, .command]): "Force Quit",
        .init("3", [.shift, .command]): "Screenshot",
        .init("4", [.shift, .command]): "Screenshot",
        .init("5", [.shift, .command]): "Screenshot",
    ]

    /// Keys a focused text field acts on. A menu key equivalent reaches AppKit before the field
    /// does, so a window shortcut must not take one; a list shortcut may, because it is live only
    /// while no text field has focus.
    package static let textEditing: [StudioKeyCombo: String] = [
        .init(.delete, .command): "Delete to the start of the line",
        .init(.delete, .option): "Delete the word before",
        .init(.upArrow): "Move up",
        .init(.downArrow): "Move down",
        .init(.upArrow, .command): "Move to the start",
        .init(.downArrow, .command): "Move to the end",
        .init(.upArrow, .option): "Move to the paragraph start",
        .init(.downArrow, .option): "Move to the paragraph end",
        .init(.space): "Type a space",
        .init(.returnKey): "New line or submit",
        .init("b", .command): "Bold",
        .init("i", .command): "Italic",
        .init("u", .command): "Underline",
        .init("t", .command): "Show Fonts",
        .init(";", .command): "Check Spelling",
        .init(":", .command): "Spelling and Grammar",
    ]

    /// Every clash the table has, described; empty when it is clean. Two shortcuts clash when
    /// they share a combo and either is a window shortcut or both live in the same context; a
    /// shortcut clashes with the system's keys everywhere, and a window shortcut with text
    /// editing's.
    package static func conflicts(in shortcuts: [StudioShortcut] = all) -> [String] {
        var found: [String] = []
        for (index, shortcut) in shortcuts.enumerated() {
            for other in shortcuts[(index + 1)...] where other.combo == shortcut.combo
                && (shortcut.context == .window || other.context == .window || shortcut.context == other.context) {
                found.append("\(shortcut.title) and \(other.title) both use \(shortcut.combo.symbols)")
            }
            if let owner = reservedBySystem[shortcut.combo] {
                found.append("\(shortcut.title) takes \(shortcut.combo.symbols) from \(owner)")
            }
            if shortcut.context == .window, let owner = textEditing[shortcut.combo] {
                found.append("\(shortcut.title) takes \(shortcut.combo.symbols) from text editing (\(owner))")
            }
        }
        return found
    }

    // MARK: Help

    /// One row of the Keyboard Shortcuts window.
    package struct HelpRow: Identifiable, Hashable, Sendable {
        package let title: String
        package let keys: String
        package var id: String { title }
    }

    /// The Keyboard Shortcuts window's sections, in menu order. The sidebar sections fold into one
    /// row, since they follow the sidebar rather than a list to learn.
    package static var helpSections: [(title: String, rows: [HelpRow])] {
        var sections: [(title: String, rows: [HelpRow])] = []
        for shortcut in all {
            let row: HelpRow
            if case .domain = shortcut.id {
                guard sections.last?.title != shortcut.menu else { continue }
                row = HelpRow(title: "Sidebar sections, in order", keys: domainRangeSymbols)
            } else if shortcut.id == .compareSide {
                row = HelpRow(title: shortcut.title, keys: "1–8")
            } else {
                row = HelpRow(title: shortcut.title, keys: shortcut.combo.symbols)
            }
            if sections.last?.title == shortcut.menu {
                sections[sections.count - 1].rows.append(row)
            } else {
                sections.append((shortcut.menu, [row]))
            }
        }
        return sections
    }

    /// "⌘1–⌘9, ⌥⌘1–⌥⌘6".
    private static var domainRangeSymbols: String {
        let combos = StudioDomain.allCases.compactMap(domainCombo)
        let plain = combos.filter { $0.modifiers == .command }
        let optioned = combos.filter { $0.modifiers != .command }
        return [plain, optioned].compactMap { group -> String? in
            guard let first = group.first, let last = group.last else { return nil }
            return first == last ? first.symbols : "\(first.symbols)–\(last.symbols)"
        }.joined(separator: ", ")
    }
}
