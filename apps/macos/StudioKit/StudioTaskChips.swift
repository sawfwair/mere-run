import Foundation

// The words a task draft's chips show, shared by the composer's chip strip and the feed card's
// header so a finished run reads the way its settings did before it ran.

extension StudioContractField where Draft == StudioTaskDraft {
    /// "Threshold 0.3", "Native", "Latency offline": the label with the value, or the label alone
    /// for a switch that is on, or "Label auto" when the CLI picks.
    package func chipTitle(in draft: StudioTaskDraft) -> String {
        switch value(in: draft) {
        case .flag(let on):
            return on ? label : "\(label) off"
        case .unset:
            if let fallback = option.defaultValue, !fallback.isEmpty {
                return "\(label) \(StudioContractChoiceTitles.title(for: fallback, flag: flag).lowercased())"
            }
            return "\(label) auto"
        case .text(let text):
            if text.isEmpty { return "\(label) auto" }
            let shown = option.choices.isEmpty ? text : StudioContractChoiceTitles.title(for: text, flag: flag)
            return "\(label) \(shown)"
        case .integer(let integer):
            return "\(label) \(integer)"
        case .number(let number):
            return "\(label) \(StudioComposerPresets.decimalText(number))"
        }
    }
}

package enum StudioTaskChips {
    /// The chips a Library row of a task-draft task shows on its card: the template's essential
    /// options (the composer's chips, minus the variant the card is headed by) valued from the
    /// recorded command, then the model. Empty for a row whose command was not recorded.
    package static func chips(for item: StudioLibraryItem, titles: StudioModelTitles) -> [String] {
        guard let task = item.templateID?.studioTask,
              let recorded = StudioLibraryDraftRestoration.taskDraft(from: item) else { return [] }
        var chips = StudioTaskSchema.essentials(for: task, draft: recorded)
            .filter { $0.overrideID != .variant }
            .map { $0.chipTitle(in: recorded) }
        if !recorded.model.isEmpty {
            chips.append(StudioModelNaming.displayName(recorded.model, titles: titles))
        }
        return chips
    }
}
