import Foundation

// Draft undo works on what a change changed, not on whole values. A text field keeps its own
// undo while it is being typed in, so typing is never registered here: registering it too would
// put two steps on the stack for one keystroke, and undoing ours would rewrite the text under the
// field's own step. Undoing a model switch must then put back the model and nothing else, or it
// would also take back what was typed after it. A draft change is therefore recorded as its two
// values and undone as a patch: every JSON path that differs between them goes back, and every
// other path keeps what it holds now.

/// A draft the task sessions register undo for: the prompt tasks' `StudioDraft` and the shared
/// task workspace's `StudioTaskDraft`.
package protocol StudioUndoableDraft: StudioAttachmentDraft, Codable, Equatable {
    /// The composer's free text. Its text field keeps its own undo.
    var prompt: String { get set }
    /// The model the draft runs, as the model chip binds it.
    var model: String { get set }
    /// Whether every value that differs from `previous` is free text: what a text field types.
    func differsOnlyInText(from previous: Self) -> Bool
}

extension StudioDraft: StudioUndoableDraft {
    /// Every changed leaf is a string (or an optional string that was or became nil).
    package func differsOnlyInText(from previous: StudioDraft) -> Bool {
        StudioDraftUndo.changedLeaves(from: previous, to: self).allSatisfy { leaf in
            leaf.before?.isText != false && leaf.after?.isText != false
        }
    }

    /// The Edit menu's name for the change from `previous` to this draft ("Change Steps"), or nil
    /// when the change is only text typed into a field that keeps its own undo.
    package func undoName(from previous: StudioDraft, mode: StudioMode, source: StudioScopeSource) -> String? {
        let fields = StudioInspectorSchema.sections(for: mode, draft: self, source: source).flatMap(\.fields)
            + StudioInspectorSchema.advancedFields(for: mode, draft: self, source: source)
        return StudioDraftUndo.name(from: previous, to: self, fields: fields,
                                    slots: mode.attachmentSlots(for: self, source: source))
    }
}

extension StudioTaskDraft: StudioUndoableDraft {
    /// Same variant, parked forms, and batch, and every changed flag holds text or nothing on both
    /// sides; positionals and the Command view's extra arguments are text.
    package func differsOnlyInText(from previous: StudioTaskDraft) -> Bool {
        guard templateID == previous.templateID, parked == previous.parked,
              batchInputPaths == previous.batchInputPaths else { return false }
        return Set(form.values.keys).union(previous.form.values.keys).allSatisfy { flag in
            form[flag].isTextOrUnset && previous.form[flag].isTextOrUnset
        }
    }

    /// The Edit menu's name for the change from `previous` to this draft, or nil when the change
    /// is only text typed into the prompt, a text option, or the Command view's extra arguments.
    package func undoName(from previous: StudioTaskDraft, task: StudioTask, source: StudioScopeSource) -> String? {
        StudioDraftUndo.name(from: previous, to: self, fields: StudioTaskSchema.fields(for: task, draft: self, source: source),
                             slots: slots(source: source))
    }
}

extension StudioContractValue {
    fileprivate var isTextOrUnset: Bool {
        switch self {
        case .text, .unset: return true
        case .integer, .number, .flag: return false
        }
    }
}

extension StudioLibraryJSON {
    fileprivate var isText: Bool {
        if case .string = self { return true }
        return false
    }
}

enum StudioDraftUndo {
    /// What a change is called in the Edit menu, or nil when it only typed text. A picked value
    /// — an attachment, the model, a toggle, a choice, a slider, a stepper, a file well — names
    /// the step after itself; a text field's value never does, since that field has its own undo.
    static func name<Draft: StudioUndoableDraft>(
        from previous: Draft,
        to next: Draft,
        fields: [StudioContractField<Draft>],
        slots: [StudioAttachmentSlot]
    ) -> String? {
        if let batch = batchUndoName(
            from: previous.batchInputPaths, to: next.batchInputPaths,
            emptied: slots.first(where: \.batches).map { $0.paths(in: next).isEmpty } ?? false
        ) {
            return batch
        }
        for slot in slots {
            let before = slot.paths(in: previous)
            let after = slot.paths(in: next)
            if after.count > before.count { return "Add Attachment" }
            if after.count < before.count { return "Remove Attachment" }
            if after != before { return "Change Attachment" }
        }
        if previous.model != next.model { return "Change Model" }
        let picked = fields.filter { field in
            field.changedCount(draft: next, baseline: previous) > 0 && !isTyped(field, from: previous, to: next)
        }
        if picked.count == 1 { return "Change \(picked[0].label)" }
        if picked.count > 1 || !next.differsOnlyInText(from: previous) { return "Change Settings" }
        return nil
    }

    /// A batch's own steps: files added to it, one taken out, or the whole batch cleared
    /// (`emptied`: the slot holds nothing now). A batch that shrinks to one file is a file taken
    /// out; nil when the batch did not change.
    static func batchUndoName(from previous: [String], to next: [String], emptied: Bool) -> String? {
        guard previous != next else { return nil }
        if next.count > previous.count { return "Add Files" }
        if emptied { return "Clear Files" }
        return next.count < previous.count ? "Remove File" : "Change Files"
    }

    /// Whether this field's change was typed into a text field: a free-text option, the seed row,
    /// or a number the Command view takes as the text the argv carries (text on both sides,
    /// where the inspector's slider and stepper write numbers).
    private static func isTyped<Draft>(_ field: StudioContractField<Draft>, from previous: Draft, to next: Draft) -> Bool {
        switch field.control {
        case .field:
            return true
        case .override:
            return field.overrideID == .seed
        case .slider, .stepper:
            return field.bindings.allSatisfy { $0.read(previous).text != nil && $0.read(next).text != nil }
        case .toggle, .segmented, .picker, .path:
            return false
        }
    }

    /// The JSON paths whose values differ between two drafts, one per changed leaf, sorted. A
    /// run of changes to the same paths (a slider drag) coalesces into one step on this.
    static func changedPaths<Draft: Encodable>(from previous: Draft, to next: Draft) -> [String] {
        changedLeaves(from: previous, to: next).map(\.path).sorted()
    }

    /// Each value that differs between two drafts, down to the leaf; nil where the encoder left
    /// the key out.
    static func changedLeaves<Draft: Encodable>(
        from previous: Draft,
        to next: Draft
    ) -> [(path: String, before: StudioLibraryJSON?, after: StudioLibraryJSON?)] {
        var leaves: [(path: String, before: StudioLibraryJSON?, after: StudioLibraryJSON?)] = []
        collectChanges(json(previous), json(next), at: "", into: &leaves)
        return leaves
    }

    /// `current` with every value that differs between `previous` and `next` put back to its
    /// `previous` value, and everything else as `current` holds it.
    static func reverting<Draft: Codable>(_ current: Draft, from previous: Draft, to next: Draft) -> Draft {
        let merged = revert(json(current), previous: json(previous), next: json(next))
        guard let merged, let data = try? JSONEncoder.mereRunApp.encode(merged),
              let draft = try? JSONDecoder.mereRunApp.decode(Draft.self, from: data) else { return previous }
        return draft
    }

    private static func json<Value: Encodable>(_ value: Value) -> StudioLibraryJSON? {
        guard let data = try? JSONEncoder.mereRunApp.encode(value) else { return nil }
        return try? JSONDecoder.mereRunApp.decode(StudioLibraryJSON.self, from: data)
    }

    /// A missing value is a key the encoder left out (an optional that is nil).
    private static func revert(_ current: StudioLibraryJSON?, previous: StudioLibraryJSON?, next: StudioLibraryJSON?) -> StudioLibraryJSON? {
        guard previous != next else { return current }
        switch (current, previous, next) {
        case (.object(var now)?, .object(let before)?, .object(let after)?):
            for key in Set(before.keys).union(after.keys) {
                now[key] = revert(now[key], previous: before[key], next: after[key])
            }
            return .object(now)
        case (.array(var now)?, .array(let before)?, .array(let after)?) where now.count == after.count && before.count == after.count:
            for index in now.indices {
                now[index] = revert(now[index], previous: before[index], next: after[index]) ?? .null
            }
            return .array(now)
        default:
            return previous
        }
    }

    private static func collectChanges(
        _ previous: StudioLibraryJSON?,
        _ next: StudioLibraryJSON?,
        at path: String,
        into leaves: inout [(path: String, before: StudioLibraryJSON?, after: StudioLibraryJSON?)]
    ) {
        guard previous != next else { return }
        switch (previous, next) {
        case (.object(let before)?, .object(let after)?):
            for key in Set(before.keys).union(after.keys) {
                collectChanges(before[key], after[key], at: path + "/" + key, into: &leaves)
            }
        case (.array(let before)?, .array(let after)?) where before.count == after.count:
            for index in before.indices {
                collectChanges(before[index], after[index], at: path + "/\(index)", into: &leaves)
            }
        default:
            leaves.append((path, previous, next))
        }
    }
}
