import Foundation

/// The prompts a page has run, newest first and each once: what ↑ in the composer's prompt
/// recalls and the composer's Recent prompts menu lists. It is read from the Library, so it is
/// the history the Library already keeps, never a second record: a deleted run's prompt leaves
/// it with the run. A page's prompts are those of the runs its feed shows
/// (`StudioFeedCardBuilder`), and a thread's prompts are its user turns.
package enum StudioPromptHistory {
    /// How many prompts the Recent prompts menu lists; ↑ reaches every one.
    package static let menuLimit = 12

    /// The longest a menu title runs before it is cut with an ellipsis.
    package static let menuTitleLength = 60

    package static func prompts(for task: StudioTask, in items: [StudioLibraryItem]) -> [String] {
        var entries: [(text: String, at: Date, order: Int)] = []
        for item in items where belongs(item, to: task) {
            if let messages = item.messages {
                for message in messages where message.role == .user {
                    entries.append((message.content, message.createdAt, entries.count))
                }
            } else {
                entries.append((item.prompt, item.createdAt, entries.count))
            }
        }
        // Newest first; two prompts from the same instant keep the Library's order.
        entries.sort { $0.at == $1.at ? $0.order < $1.order : $0.at > $1.at }
        var seen: Set<String> = []
        return entries.compactMap { entry in
            let text = entry.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, seen.insert(text).inserted else { return nil }
            return text
        }
    }

    /// The feed's rule: a prompt page shows its mode's runs and threads; a task on the shared
    /// workspace the runs of the templates it runs, and older rows its mode filed.
    private static func belongs(_ item: StudioLibraryItem, to task: StudioTask) -> Bool {
        if let mode = task.mode { return item.mode == mode }
        if let templateID = item.templateID { return task.runs(templateID) }
        return item.mode.task == task
    }

    /// A prompt as one menu line: line breaks folded into spaces, cut at `menuTitleLength`.
    package static func menuTitle(for prompt: String) -> String {
        let line = prompt.components(separatedBy: .newlines).filter { !$0.isBlank }.joined(separator: " ")
        guard line.count > menuTitleLength else { return line }
        return line.prefix(menuTitleLength - 1).trimmingCharacters(in: .whitespaces) + "…"
    }
}

/// Stepping through a page's history from the prompt field, the way a shell's ↑ and ↓ do: the
/// first ↑ keeps what the field held and shows the newest prompt that differs from it, each
/// further ↑ an older one, and ↓ walks back until it puts the kept text back. Editing a recalled
/// prompt starts over from the newest.
package struct StudioPromptRecall: Equatable {
    /// The history entry the field shows; nil while it shows the user's own text.
    package private(set) var position: Int?
    /// What the field held at the first ↑, which ↓ past the newest entry puts back.
    package private(set) var kept = ""

    package init() {}

    /// The prompt ↑ shows in place of `current`, or nil when there is nothing older, so the key
    /// does what it would have.
    package mutating func older(than current: String, in history: [String]) -> String? {
        let start: Int
        if let position, history.indices.contains(position), history[position] == current {
            start = position + 1
        } else {
            kept = current
            start = 0
        }
        guard let next = history.indices.first(where: { $0 >= start && history[$0] != current }) else { return nil }
        position = next
        return history[next]
    }

    /// The text ↓ shows in place of `current`: a newer prompt, then what was kept. nil while the
    /// field is not showing a recalled prompt, so the key does what it would have.
    package mutating func newer(than current: String, in history: [String]) -> String? {
        guard let position, history.indices.contains(position), history[position] == current else {
            self.position = nil
            return nil
        }
        if let next = history.indices.last(where: { $0 < position && history[$0] != kept }) {
            self.position = next
            return history[next]
        }
        self.position = nil
        return kept
    }

    /// Whether ↑ at `caret` (a UTF-16 offset into `text`) would leave the first line: no line
    /// break comes before it. Only then does ↑ recall; on any later line it moves the caret.
    package static func isOnFirstLine(_ text: String, caret: Int) -> Bool {
        let string = text as NSString
        return string.rangeOfCharacter(from: .newlines, options: [], range: NSRange(location: 0, length: caret)).location == NSNotFound
    }

    /// Whether ↓ at `caret` would leave the last line: no line break comes after it.
    package static func isOnLastLine(_ text: String, caret: Int) -> Bool {
        let string = text as NSString
        let rest = NSRange(location: caret, length: string.length - caret)
        return string.rangeOfCharacter(from: .newlines, options: [], range: rest).location == NSNotFound
    }
}
