import Foundation
import MereRunContract

/// A page's own starting point: the option values the user kept with "Save as my defaults",
/// which the page's fresh drafts and its inspector's Reset start from instead of the app's.
///
/// Only settings are kept, never what a run is about: the inputs, the prompt, the seed, the
/// model (Models ▸ "Use for … by default" owns that), output paths, and credentials stay with
/// the draft. Saving keeps the options the model the draft runs uses, the ones its inspector
/// shows. A kept value the model a later draft runs does not use stays in that draft the way
/// any hidden value does (`StudioOptionScope`): hidden, and left out of the run.
package struct StudioPageDefaults: Codable, Equatable {
    /// Each kept value, by the field it is written back to: a prompt mode's `StudioDraft`
    /// property name, or a task template's flag.
    package var values: [String: StudioContractValue]

    package init(values: [String: StudioContractValue]) {
        self.values = values
    }

    /// Whether a flag names a seed (`--seed`, `--retake-seed`): the one number a run is meant
    /// to change.
    private static func isSeed(_ flag: String) -> Bool {
        flag == "--seed" || flag.hasSuffix("-seed")
    }

    /// Whether an option holds a setting a page can start from, rather than a path, the model,
    /// or a seed.
    private static func keepsKind(of option: MereRunCapabilityOption) -> Bool {
        option.kind != .file && option.kind != .directory && option.flag != "--model" && !isSeed(option.flag)
    }
}

/// What the inspector's defaults row offers for the draft it shows.
package struct StudioPageDefaultsStatus: Equatable {
    /// The page has saved defaults.
    package let hasSaved: Bool
    /// The draft sets a kept option differently from where a fresh draft starts.
    package let canSave: Bool
    /// There are saved defaults to forget, or the draft sets a kept option differently from
    /// the app's defaults.
    package let canRestore: Bool
}

// MARK: - Prompt modes

extension StudioPageDefaults {
    /// Keeps the options `draft` sets that the model it runs uses.
    package init(capturing draft: StudioDraft, mode: StudioMode, source: StudioScopeSource) {
        let bindings = Self.keptBindings(for: mode, draft: draft, source: source)
        self.init(values: Dictionary(bindings.map { ($0.fieldID, $0.read(draft)) }, uniquingKeysWith: { first, _ in first }))
    }

    /// Writes the kept values over `draft`. A value for a field this build no longer has is
    /// passed over.
    package func apply(to draft: inout StudioDraft, mode: StudioMode) {
        let bindings = Self.allBindings(for: mode)
        for (fieldID, value) in values {
            bindings[fieldID]?.write(&draft, value)
        }
    }

    /// The draft fields a save keeps from `draft`: the options of the model it runs, less the
    /// ones that are not settings, plus the editors' own fields beside them.
    package static func keptBindings(
        for mode: StudioMode, draft: StudioDraft, source: StudioScopeSource
    ) -> [StudioContractBinding<StudioDraft>] {
        guard let scope = source.scope(mode: mode, draft: draft) else { return [] }
        return keptBindings(for: mode, options: scope.options)
    }

    /// Every draft field a save could keep, whatever the model: what "Restore app defaults"
    /// puts back. Read Image's are its three tasks' together.
    package static func restorableBindings(
        for mode: StudioMode, draft: StudioDraft, source: StudioScopeSource
    ) -> [StudioContractBinding<StudioDraft>] {
        let actions = mode == .readImage ? StudioReadImageAction.allCases : [draft.readImageAction]
        let options = actions.flatMap { action -> [MereRunCapabilityOption] in
            var probe = draft
            probe.readImageAction = action
            return source.scope(mode: mode, draft: probe)?.capability.options ?? []
        }
        return keptBindings(for: mode, options: options)
    }

    private static func keptBindings(
        for mode: StudioMode, options: [MereRunCapabilityOption]
    ) -> [StudioContractBinding<StudioDraft>] {
        let table = StudioContractBindings.bindings(for: mode)
        var kept: [StudioContractBinding<StudioDraft>] = []
        var seen: Set<String> = []
        func keep(_ binding: StudioContractBinding<StudioDraft>) {
            if seen.insert(binding.fieldID).inserted { kept.append(binding) }
        }
        for option in options where keepsKind(of: option) {
            let override = StudioContractOverrides.override(forFlag: option.flag, mode: mode)
            switch override?.id {
            case .model, .seed, .attachment, .regionPrompts, .imageCanvas, .orderedReferences:
                // The model, the seed, and what the run reads — the composer's well, the drawn
                // prompts, the mask, the ordered references — belong to the draft.
                continue
            default:
                if let binding = table[option.flag] { keep(binding) }
                override?.companions.forEach(keep)
            }
        }
        StudioContractSchema.uncoveredFields(for: mode).flatMap(\.bindings).forEach(keep)
        return kept
    }

    /// Every field of the mode's draft by id: the flags' bindings, the editors' own fields, and
    /// Read Image's task picker.
    private static func allBindings(for mode: StudioMode) -> [String: StudioContractBinding<StudioDraft>] {
        let bindings = Array(StudioContractBindings.bindings(for: mode).values)
            + StudioContractOverrides.overrides(for: mode).flatMap(\.companions)
            + StudioContractSchema.uncoveredFields(for: mode).flatMap(\.bindings)
        return Dictionary(bindings.map { ($0.fieldID, $0) }, uniquingKeysWith: { first, _ in first })
    }
}

// MARK: - Task drafts

extension StudioPageDefaults {
    /// Keeps the options `draft`'s form sets that the model it runs uses.
    package init(capturing draft: StudioTaskDraft, source: StudioScopeSource) {
        let flags = source.scope(for: draft).map { Self.keptFlags(of: $0.options, templateID: draft.templateID) } ?? []
        self.init(values: Dictionary(uniqueKeysWithValues: flags.map { ($0, draft.form[$0]) }))
    }

    /// Writes the kept values over `draft`'s form.
    package func apply(to draft: inout StudioTaskDraft) {
        for (flag, value) in values {
            draft.form.values[flag] = value == .unset ? nil : value
        }
    }

    /// The flags a save keeps from a task template's `options`: everything the inspector edits
    /// except the model, the seed, paths, credentials, and the picks made on the input (which
    /// face, each view's camera). The well's slots, the prompt, and the destinations are never
    /// inspector options to begin with.
    package static func keptFlags(of options: [MereRunCapabilityOption], templateID: CommandTemplateID) -> [String] {
        guard let capability = templateID.capability else { return [] }
        let slotFlags = StudioTaskSchema.slots(for: templateID).compactMap { slot -> String? in
            switch slot.storage {
            case .flag(let flag), .flagList(let flag): return flag
            case .argument, .argumentList, .path, .pathList: return nil
            }
        }
        var promptFlag: [String] = []
        if case .flag(let flag) = StudioTaskSchema.promptField(for: capability) { promptFlag = [flag] }
        let routing = capability.routing
        let modelFlags = (routing?.modelFlags ?? []) + (routing?.families.compactMap(\.modelFlag) ?? [])
        let excluded = StudioTaskSchema.hiddenFlags(for: capability)
            .union(StudioTaskSchema.chosenOutputFlags)
            .union(slotFlags)
            .union(promptFlag)
            .union(modelFlags)
            .union([String].secretFlags)
            .union(CommandLaunchEnvironment.secretFlags(for: templateID).keys)
        return options.filter { option in
            keepsKind(of: option) && !excluded.contains(option.flag)
                && ![.faceIndex, .cameras].contains(StudioTaskSchema.overrideID(forFlag: option.flag, templateID: templateID))
        }.map(\.flag)
    }
}

// MARK: - Storage and undo

extension StudioTaskSessions {
    /// Where a page's defaults are kept: `"<task>.pageDefaults"` for a prompt task, and one per
    /// template (`"<task>.pageDefaults.<template>"`) for a task whose variants run different
    /// commands.
    package static func pageDefaultsKey(_ task: StudioTask, templateID: CommandTemplateID? = nil) -> String {
        task.rawValue + ".pageDefaults" + (templateID.map { "." + $0.rawValue } ?? "")
    }

    package func pageDefaults(for task: StudioTask, templateID: CommandTemplateID? = nil) -> StudioPageDefaults? {
        value(for: Self.pageDefaultsKey(task, templateID: templateID), default: Optional<StudioPageDefaults>.none)
    }

    /// A fresh draft for one of `task`'s templates: the app's, with the page's saved defaults
    /// for that template over it. What a new draft starts from and the task inspector resets to.
    package func freshTaskDraft(for task: StudioTask, templateID: CommandTemplateID) -> StudioTaskDraft {
        var draft = StudioTaskDraft(templateID: templateID)
        pageDefaults(for: task, templateID: templateID)?.apply(to: &draft)
        return draft
    }

    /// A fresh draft on the task's first variant, or nil for a task with none.
    package func freshTaskDraft(for task: StudioTask) -> StudioTaskDraft? {
        task.variantTemplates.first.map { freshTaskDraft(for: task, templateID: $0.id) }
    }

    package func pageDefaultsStatus(for task: StudioTask, draft: StudioTaskDraft, source: StudioScopeSource) -> StudioPageDefaultsStatus {
        let kept = source.scope(for: draft).map { StudioPageDefaults.keptFlags(of: $0.options, templateID: draft.templateID) } ?? []
        let fresh = freshTaskDraft(for: task, templateID: draft.templateID)
        let app = StudioTaskDraft(templateID: draft.templateID)
        let hasSaved = pageDefaults(for: task, templateID: draft.templateID) != nil
        return StudioPageDefaultsStatus(
            hasSaved: hasSaved,
            canSave: kept.contains { draft.form[$0] != fresh.form[$0] },
            canRestore: hasSaved || restorableFlags(of: draft, source: source).contains { draft.form[$0] != app.form[$0] }
        )
    }

    /// "Save as my defaults" on a task draft's inspector, as one undo step.
    package func savePageDefaults(for task: StudioTask, draft: StudioTaskDraft, source: StudioScopeSource) {
        let key = Self.pageDefaultsKey(task, templateID: draft.templateID)
        undoably(StudioPageDefaults.saveUndoName, keys: [key]) {
            set(Optional(StudioPageDefaults(capturing: draft, source: source)), for: key)
        }
    }

    /// "Restore app defaults" on a task draft's inspector: forgets the page's saved defaults for
    /// the draft's template and puts every option they could hold back to the app's, leaving the
    /// inputs, prompt, seed, and model as they are. One undo step brings both back.
    package func restoreAppDefaults(for task: StudioTask, source: StudioScopeSource) {
        guard var next = taskDraft(for: task) else { return }
        let key = Self.pageDefaultsKey(task, templateID: next.templateID)
        let app = StudioTaskDraft(templateID: next.templateID)
        for flag in restorableFlags(of: next, source: source) {
            next.form.values[flag] = app.form.values[flag]
        }
        undoably(StudioPageDefaults.restoreUndoName, keys: [key, Self.taskDraftKey(task)]) {
            set(Optional<StudioPageDefaults>.none, for: key)
            setTaskDraft(next, for: task)
        }
    }

    /// Every flag of the draft's template a save could keep, whatever the model.
    private func restorableFlags(of draft: StudioTaskDraft, source: StudioScopeSource) -> [String] {
        guard let capability = source.capability(for: draft.templateID) else { return [] }
        return StudioPageDefaults.keptFlags(of: capability.options, templateID: draft.templateID)
    }

    /// A prompt recalled into a task draft's composer (↑, or the Recent prompts menu), as an
    /// undo step: the field did not type it, so its own undo does not have it.
    package func recallPrompt(_ prompt: String, for task: StudioTask) {
        guard var next = taskDraft(for: task) else { return }
        next.prompt = prompt
        undoably(StudioPageDefaults.recallUndoName, keys: [Self.taskDraftKey(task)]) {
            setTaskDraft(next, for: task)
        }
    }
}

extension StudioPageDefaults {
    /// The Edit menu's names for the steps this file registers.
    package static let saveUndoName = "Save as My Defaults"
    package static let restoreUndoName = "Restore App Defaults"
    package static let recallUndoName = "Recall Prompt"
}

extension StudioPromptTaskController {
    /// The app's fresh draft for `mode`, before the page's saved defaults: what "Restore app
    /// defaults" puts the options back to.
    package func appFreshDraft(for mode: StudioMode) -> StudioDraft {
        var draft = StudioDraft()
        draft.reset(for: mode)
        controller.applyRecommendedDefaults(to: &draft, for: mode)
        if mode.isConversational { draft.prompt = "" }
        return draft
    }

    package func pageDefaultsStatus(for mode: StudioMode) -> StudioPageDefaultsStatus {
        let source = controller.scopeSource
        let fresh = freshDraft(for: mode)
        let app = appFreshDraft(for: mode)
        let hasSaved = sessions.pageDefaults(for: mode.task) != nil
        return StudioPageDefaultsStatus(
            hasSaved: hasSaved,
            canSave: StudioPageDefaults.keptBindings(for: mode, draft: draft, source: source).contains { $0.isChanged(draft, fresh) },
            canRestore: hasSaved
                || StudioPageDefaults.restorableBindings(for: mode, draft: draft, source: source).contains { $0.isChanged(draft, app) }
        )
    }

    /// "Save as my defaults" on the open prompt task's inspector, as one undo step.
    package func savePageDefaults() {
        guard let mode = activatedMode else { return }
        let key = StudioTaskSessions.pageDefaultsKey(mode.task)
        let defaults = StudioPageDefaults(capturing: draft, mode: mode, source: controller.scopeSource)
        sessions.undoably(StudioPageDefaults.saveUndoName, keys: [key]) {
            sessions.set(Optional(defaults), for: key)
        }
    }

    /// "Restore app defaults" on the open prompt task's inspector: forgets the page's saved
    /// defaults and puts every option they could hold back to the app's, leaving the inputs,
    /// prompt, seed, and model as they are. One undo step brings both back.
    package func restoreAppDefaults() {
        guard let mode = activatedMode else { return }
        let key = StudioTaskSessions.pageDefaultsKey(mode.task)
        let app = appFreshDraft(for: mode)
        var next = draft
        for binding in StudioPageDefaults.restorableBindings(for: mode, draft: draft, source: controller.scopeSource) {
            binding.reset(&next, to: app)
        }
        sessions.undoably(StudioPageDefaults.restoreUndoName, keys: Array(Set([key] + draftKeys(for: mode)))) {
            sessions.set(Optional<StudioPageDefaults>.none, for: key)
            draft = next
        }
    }

    /// A prompt recalled into the open prompt task's composer (↑, or the Recent prompts menu),
    /// as an undo step: the field did not type it, so its own undo does not have it.
    package func recallPrompt(_ prompt: String) {
        guard let mode = activatedMode else { return }
        var next = draft
        next.prompt = prompt
        sessions.undoably(StudioPageDefaults.recallUndoName, keys: draftKeys(for: mode)) {
            draft = next
        }
    }
}
