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
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [shortcut("Inspect", .init("e", .command), .window)]).count, 1,
                       "⌘E is Use Selection for Find in every text view")
        XCTAssertEqual(StudioKeyboardShortcuts.conflicts(in: [shortcut("Go", .init("g", .command), .window)]).count, 1,
                       "⌘G is Find Next")
    }

    func testTheSettledKeys() {
        func symbols(_ id: StudioShortcutID) -> String { StudioKeyboardShortcuts.combo(id).symbols }
        XCTAssertEqual(symbols(.run), "⌘↩")
        XCTAssertEqual(symbols(.searchLibrary), "⌘L")
        XCTAssertEqual(symbols(.find), "⌘F")
        XCTAssertEqual(symbols(.quickLookSelection), "Space")
        XCTAssertEqual(symbols(.deleteFromLibrary), "⌘⌫")
        XCTAssertEqual(symbols(.showInspector), "⌥⌘I")
        XCTAssertEqual(symbols(.nextTask), "⇧⌘]")
        XCTAssertEqual(symbols(.previousTask), "⇧⌘[")
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
        XCTAssertEqual(sections.map(\.title), ["File", "Edit", "View", "Go", "Run", "Window", "Help", "Library and feed", "Prompt", "Compare"])
        XCTAssertEqual(sections.last?.rows.map(\.keys), ["Space", "1–8"], "Compare's keys, live while Compare has focus")
        let rows = sections.flatMap(\.rows)
        let listed = StudioKeyboardShortcuts.all.filter {
            if case .domain = $0.id { return false }
            return true
        }
        XCTAssertEqual(rows.count, listed.count + 1)
        XCTAssertEqual(sections.first { $0.title == "Go" }?.rows.map(\.keys), ["⌘1–⌘9, ⌥⌘1–⌥⌘6", "⇧⌘]", "⇧⌘["])
    }

    func testNextAndPreviousTaskCycleTheDomainsTasks() throws {
        for domain in StudioDomain.allCases {
            let tasks = domain.tasks
            guard tasks.count > 1 else {
                XCTAssertNil(StudioKeyboardShortcuts.adjacentTask(to: try XCTUnwrap(tasks.first), offset: 1),
                             "\(domain.title) has nothing to cycle")
                continue
            }
            var forward: [StudioTask] = []
            var task = try XCTUnwrap(tasks.first)
            for _ in tasks {
                forward.append(task)
                task = try XCTUnwrap(StudioKeyboardShortcuts.adjacentTask(to: task, offset: 1))
            }
            XCTAssertEqual(forward, tasks, "Next Task walks \(domain.title) in order")
            XCTAssertEqual(task, tasks.first, "and wraps from the last task to the first")
            XCTAssertEqual(StudioKeyboardShortcuts.adjacentTask(to: tasks[0], offset: -1), tasks.last,
                           "Previous Task wraps from the first task to the last")
            XCTAssertEqual(StudioKeyboardShortcuts.adjacentTask(to: tasks[1], offset: -1), tasks[0])
        }
    }

    func testEveryTaskHasATooltipThatTellsItFromItsSiblings() {
        for domain in StudioDomain.allCases {
            let tips = domain.tasks.map(\.helpText)
            for (task, tip) in zip(domain.tasks, tips) {
                XCTAssertFalse(tip.isEmpty, "\(task) has a tooltip")
                XCTAssertFalse(tip.hasSuffix("."), "\(task)'s tooltip is a phrase, not a sentence")
            }
            XCTAssertEqual(Set(tips).count, tips.count, "\(domain.title)'s tasks have distinct tooltips: \(tips)")
        }
        XCTAssertEqual(StudioTask.musicSeparate.helpText, "Separate a mix into stems")
    }
}
