import AppKit
import StudioKit
import SwiftUI

// The composer's prompt history (`StudioPromptHistory`): ↑ and ↓ in the prompt field, and the
// Recent prompts menu beside Run. Both recall through the page's own write, which registers the
// recall as an undo step.

/// The clock button beside Run that lists the page's recent prompts, newest first.
struct StudioRecentPromptsMenu: View {
    let prompts: [String]
    let onPick: (String) -> Void

    var body: some View {
        Menu {
            Section("Recent prompts") {
                ForEach(prompts.prefix(StudioPromptHistory.menuLimit), id: \.self) { prompt in
                    Button(StudioPromptHistory.menuTitle(for: prompt)) { onPick(prompt) }
                }
            }
        } label: {
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.mereIcon)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Recent prompts · ↑ in the prompt recalls them too")
        .accessibilityLabel("Recent prompts")
        .accessibilityValue(prompts.count == 1 ? "1 prompt" : "\(prompts.count) prompts")
    }
}

extension View {
    /// ↑ on the prompt's first line recalls an older prompt from `history`, and ↓ on its last
    /// line a newer one and then what was typed, while `isActive` (the prompt has focus). Any
    /// other ↑ or ↓ moves the caret as usual. `recall` writes the chosen text.
    func studioPromptHistoryKeys(
        isActive: Bool,
        history: [String],
        text: String,
        recall: @escaping (String) -> Void
    ) -> some View {
        modifier(StudioPromptHistoryKeys(isActive: isActive, history: history, text: text, recall: recall))
    }
}

private struct StudioPromptHistoryKeys: ViewModifier {
    let isActive: Bool
    let history: [String]
    let text: String
    let recall: (String) -> Void

    @State private var monitor = Monitor()

    /// The installed key monitor and what it steps through. It reads the latest history, text,
    /// and write on every key, and keeps the recall position between keys.
    @MainActor
    final class Monitor {
        var history: [String] = []
        var text = ""
        var recall: (String) -> Void = { _ in }
        var position = StudioPromptRecall()
        weak var window: NSWindow?
        /// Removes the installed monitor; nil while none is installed.
        private var removeMonitor: (() -> Void)?

        func install() {
            guard removeMonitor == nil,
                  let token = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] event in
                      guard let self, let window = self.window, event.window === window,
                            event.modifierFlags.intersection([.command, .shift, .option, .control]).isEmpty,
                            let editor = window.firstResponder as? NSTextView,
                            self.handle(event, in: editor) else { return event }
                      return nil
                  }) else { return }
            removeMonitor = { NSEvent.removeMonitor(token) }
        }

        func remove() {
            removeMonitor?()
            removeMonitor = nil
        }

        /// Whether the key was a recall: ↑ on the first line or ↓ on the last, with no selection,
        /// and a prompt to show.
        private func handle(_ event: NSEvent, in editor: NSTextView) -> Bool {
            let selection = editor.selectedRange()
            guard selection.length == 0 else { return false }
            let next: String?
            switch event.keyCode {
            case Self.upArrowKeyCode:
                guard Self.caretIsOnFirstLine(editor, caret: selection.location) else { return false }
                next = position.older(than: text, in: history)
            case Self.downArrowKeyCode:
                guard Self.caretIsOnLastLine(editor, caret: selection.location) else { return false }
                next = position.newer(than: text, in: history)
            default:
                return false
            }
            guard let next else { return false }
            text = next
            recall(next)
            return true
        }

        /// No line break before the caret, and the caret on the field's first visual line too,
        /// so ↑ inside a long wrapped first paragraph still moves up a line.
        private static func caretIsOnFirstLine(_ editor: NSTextView, caret: Int) -> Bool {
            StudioPromptRecall.isOnFirstLine(editor.string, caret: caret)
                && abs(lineMid(editor, at: caret) - lineMid(editor, at: 0)) < lineTolerance
        }

        private static func caretIsOnLastLine(_ editor: NSTextView, caret: Int) -> Bool {
            let end = (editor.string as NSString).length
            return StudioPromptRecall.isOnLastLine(editor.string, caret: caret)
                && abs(lineMid(editor, at: caret) - lineMid(editor, at: end)) < lineTolerance
        }

        /// How high on screen the line holding `offset` sits, through the text input client,
        /// which answers the same for either text system.
        private static func lineMid(_ editor: NSTextView, at offset: Int) -> CGFloat {
            editor.firstRect(forCharacterRange: NSRange(location: offset, length: 0), actualRange: nil).midY
        }

        /// Less than a line apart: two offsets on the same visual line.
        private static let lineTolerance: CGFloat = 4

        /// The arrow keys' virtual key codes, as `NSEvent.keyCode` reports them.
        private static let upArrowKeyCode: UInt16 = 126
        private static let downArrowKeyCode: UInt16 = 125
    }

    func body(content: Content) -> some View {
        monitor.history = history
        monitor.text = text
        monitor.recall = recall
        return content
            .background(StudioWindowReader { monitor.window = $0 })
            .onChange(of: isActive, initial: true) { _, active in
                if active { monitor.install() } else { monitor.remove() }
            }
            .onDisappear { monitor.remove() }
    }
}
