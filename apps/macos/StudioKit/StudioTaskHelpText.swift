import Foundation

extension StudioTask {
    /// The header task control's tooltip: the task's empty-state line ("Pull the mix apart"), or,
    /// where its domain's tasks share one line (Models, Server), its primary command's catalog
    /// subtitle. Sentence case, without the closing period.
    package var helpText: String {
        let line = Self.tooltip(presentation.emptyTitle)
        let siblings = domain.tasks.filter { $0 != self }
        let shared = siblings.contains { Self.tooltip($0.presentation.emptyTitle) == line }
        if !line.isEmpty, line != title, !shared { return line }
        let templates = commandTemplates
        let primary = primaryTemplateID.flatMap { id in templates.first { $0.id == id } } ?? templates.first
        if let subtitle = primary?.subtitle, !subtitle.isEmpty { return subtitle }
        return title
    }

    private static func tooltip(_ line: String) -> String {
        line.hasSuffix(".") ? String(line.dropLast()) : line
    }
}
