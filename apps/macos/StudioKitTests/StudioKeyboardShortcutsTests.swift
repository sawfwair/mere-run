import Foundation
@testable import StudioKit
import XCTest

/// The shortcut table the menus bind and Help ▸ Keyboard Shortcuts lists: every key taken once,
/// none taken from macOS or from a focused text field, and the keys the design settled on.
final class StudioKeyboardShortcutsTests: XCTestCase {
    func testEveryShortcutIsUniqueAndClearOfTheSystemAndTextEditing() {
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(), [])
        let ids = StudioKeyboardShortcuts.all.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count, "each action is in the table once")
    }

    func testTheConflictCheckCatchesEachKindOfClash() {
        func shortcut(_ title: String, _ combo: StudioKeyCombo, _ context: StudioShortcutContext) -> StudioShortcut {
            StudioShortcut(id: .find, menu: "Test", title: title, combo: combo, context: context)
        }
        let delete = StudioKeyCombo(.delete, .command)
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [shortcut("Trash", delete, .window)]).count, 1,
                       "⌘⌫ in a menu would take delete-to-line-start from every text field")
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [shortcut("Trash", delete, .list)]), [],
                       "while a list has focus no text field is listening")
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [shortcut("Quit", .init("q", .command), .list)]).count, 1)
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [
            shortcut("One", .init("k", .command), .window), shortcut("Two", .init("k", .command), .prompt),
        ]).count, 1, "a menu item shadows the same key in any view")
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [
            shortcut("One", .init(.space), .list), shortcut("Two", .init(.space), .prompt),
        ]), [], "two views never have focus at once")
    }

    func testTheSettledKeys() {
        func symbols(_ id: StudioShortcutID) -> String { StudioKeyboardShortcuts.combo(id).symbols }
        XCTAssertEqual(symbols(.run), "⌘↩")
        XCTAssertEqual(symbols(.searchLibrary), "⌘L")
        XCTAssertEqual(symbols(.find), "⌘F")
        XCTAssertEqual(symbols(.quickLookSelection), "Space")
        XCTAssertEqual(symbols(.deleteFromLibrary), "⌘⌫")
        XCTAssertEqual(symbols(.showInspector), "⌘E")
        XCTAssertEqual(symbols(.showCommandView), "⌥⌘C")
        XCTAssertEqual(symbols(.showLibrary), "⇧⌘L")
        XCTAssertEqual(StudioKeyboardShortcuts.shortcut(.deleteFromLibrary).context, .list)
        XCTAssertEqual(StudioKeyboardShortcuts.shortcut(.olderPrompt).context, .prompt)
        XCTAssertEqual(StudioKeyboardShortcuts.help("Show Library", .showLibrary), "Show Library (⇧⌘L)")
    }

    func testSidebarSectionsTakeTheDigitsInSidebarOrder() {
        let combos = StudioDomain.allCases.map { StudioKeyboardShortcuts.domainCombo($0)?.symbols }
        XCTAssertEqual(combos.prefix(9).map { $0 ?? "" }, (1...9).map { "⌘\($0)" })
        XCTAssertEqual(combos.dropFirst(9).map { $0 ?? "" }, (1...(StudioDomain.allCases.count - 9)).map { "⌥⌘\($0)" })
        let sidebarOrder = StudioDomainGroup.allCases.flatMap(\.domains)
        XCTAssertEqual(sidebarOrder, StudioDomain.allCases, "the digits follow the sidebar")
    }

    func testFindSearchesThePagesListOrTheLibraryBesideIt() {
        XCTAssertEqual(StudioKeyboardShortcuts.findField(for: .modelsInstalled), .page)
        XCTAssertEqual(StudioKeyboardShortcuts.findField(for: .pluginsCatalog), .page)
        XCTAssertEqual(StudioKeyboardShortcuts.findField(for: .imageGenerate), .library)
        XCTAssertEqual(StudioKeyboardShortcuts.findField(for: .chatChat), .library, "Chat's thread list stands in for the Library")
        XCTAssertEqual(StudioKeyboardShortcuts.findField(for: .audioEnhance), .library)
        XCTAssertNil(StudioKeyboardShortcuts.findField(for: .serverServing), "nothing to search")
    }

    func testTheHelpWindowListsEveryShortcutWithTheSidebarFolded() {
        let sections = StudioKeyboardShortcuts.helpSections
        XCTAssertEqual(sections.map(\.title), ["File", "Edit", "View", "Go", "Run", "Window", "Help", "Library and feed", "Prompt"])
        let rows = sections.flatMap(\.rows)
        let listed = StudioKeyboardShortcuts.all.filter {
            if case .domain = $0.id { return false }
            return true
        }
        XCTAssertEqual(rows.count, listed.count + 1)
        XCTAssertEqual(sections.first { $0.title == "Go" }?.rows.map(\.keys), ["⌘1–⌘9, ⌥⌘1–⌥⌘6"])
    }
}
