import Foundation

/// Pure rules for the Converse thread list: which Library rows are threads, how they group,
/// and what their meta line says. Everything here is unit-testable without a view.
package enum StudioThreadListPresenter {
    package struct Section: Equatable {
        package let title: String
        package let threads: [StudioLibraryItem]
    }

    /// The conversation rows, most recently active first. Chat and Code threads share one list:
    /// Code is a preset inside Converse, not a second thread pool.
    package static func threads(in items: [StudioLibraryItem]) -> [StudioLibraryItem] {
        items.filter(\.isConversation).sorted { $0.updatedAt > $1.updatedAt }
    }

    /// The rows the media Library shows: everything that is not a thread.
    package static func mediaItems(in items: [StudioLibraryItem]) -> [StudioLibraryItem] {
        items.filter { !$0.isConversation }
    }

    /// Threads whose title or any turn contains `query` (case-insensitive); all of them for an
    /// empty query.
    package static func filter(_ threads: [StudioLibraryItem], query: String) -> [StudioLibraryItem] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return threads }
        return threads.filter { thread in
            thread.displayTitle.lowercased().contains(needle)
                || (thread.messages ?? []).contains { $0.content.lowercased().contains(needle) }
        }
    }

    /// "Today" for threads active today, "Earlier" for the rest, in that order; a group is
    /// omitted when empty.
    package static func sections(
        _ threads: [StudioLibraryItem],
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> [Section] {
        let today = threads.filter { calendar.isDate($0.updatedAt, inSameDayAs: now) }
        let earlier = threads.filter { !calendar.isDate($0.updatedAt, inSameDayAs: now) }
        var sections: [Section] = []
        if !today.isEmpty { sections.append(Section(title: "Today", threads: today)) }
        if !earlier.isEmpty { sections.append(Section(title: "Earlier", threads: earlier)) }
        return sections
    }

    /// "Qwen3.6 4B · 1:20 PM" for a chat thread active today; "Code · Gemma 4 · Yesterday" for a
    /// Code thread from yesterday; older threads show their day ("Aug 30").
    package static func meta(
        for thread: StudioLibraryItem,
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> String {
        var parts: [String] = []
        if thread.mode == .code { parts.append("Code") }
        parts.append(modelLabel(for: thread))
        parts.append(activityLabel(for: thread.updatedAt, now: now, calendar: calendar))
        return parts.joined(separator: " · ")
    }

    /// The friendly name of the model the thread last ran with, or the preset's default.
    package static func modelLabel(for thread: StudioLibraryItem) -> String {
        let identity = StudioModelNaming.resolvedModelID(for: thread.mode, model: thread.model ?? "")
        return identity.isEmpty ? "Auto" : StudioModelNaming.displayName(identity)
    }

    package static func activityLabel(for date: Date, now: Date, calendar: Calendar) -> String {
        if calendar.isDate(date, inSameDayAs: now) {
            return timeFormatter.string(from: date)
        }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
           calendar.isDate(date, inSameDayAs: yesterday) {
            return "Yesterday"
        }
        return dayFormatter.string(from: date)
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.setLocalizedDateFormatFromTemplate("MMM d")
        return formatter
    }()
}
