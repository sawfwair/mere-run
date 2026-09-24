import Foundation
import MereRunContract
import UniformTypeIdentifiers

// The specialist tasks' surface, read from `MereRunCapabilityCatalog` the way the prompt tasks'
// inspector reads theirs from `StudioContractSchema`: the attachment well's slots are the
// capability's file inputs, the chips are its `essential` options, the inspector's sections are
// its `standard` options by group, and Advanced holds the `expert` tier. Every control binds the
// task draft's console form, so the Command view and the argv show what the form holds.

/// One option of a task's template bound to the task draft: the console's per-flag entry, or a
/// positional, or the variant the draft runs.
extension StudioContractBinding where Draft == StudioTaskDraft {
    package static func flag(_ flag: String) -> Self {
        let inner = StudioContractBinding<StudioConsoleDraft>.flag(flag)
        return Self(fieldID: flag, read: { inner.read($0.form) }, write: { draft, value in inner.write(&draft.form, value) })
    }

    package static func argument(_ index: Int) -> Self {
        let inner = StudioContractBinding<StudioConsoleDraft>.argument(index)
        return Self(
            fieldID: "argument.\(index)",
            read: { inner.read($0.form) },
            write: { draft, value in inner.write(&draft.form, value) }
        )
    }

    /// The variant: reads and writes `templateID`, carrying shared values across the switch.
    package static var variant: Self {
        Self(
            fieldID: StudioTaskSchema.variantFlag,
            read: { .text($0.templateID.rawValue) },
            write: { draft, value in
                guard let text = value.text, let next = CommandTemplateID(rawValue: text) else { return }
                draft.switchTemplate(to: next)
            }
        )
    }
}

/// Where a task's free text goes: a positional (`prompt`, or the repeatable `texts` one line per
/// argument) or a `--prompt`-style option.
package enum StudioTaskPromptField: Equatable {
    case argument(Int, repeatable: Bool)
    case flag(String)
}

/// One group of task fields, in the contract's group order.
package struct StudioTaskSection: Identifiable {
    package let group: StudioContractGroup
    package let fields: [StudioContractField<StudioTaskDraft>]

    package var id: StudioContractGroup { group }
    package var title: String { group.title }

    package func changedCount(draft: StudioTaskDraft, baseline: StudioTaskDraft) -> Int {
        fields.reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
    }

    package func reset(_ draft: inout StudioTaskDraft, to baseline: StudioTaskDraft) {
        for field in fields { field.reset(&draft, to: baseline) }
    }
}

package enum StudioTaskSchema {
    /// The synthetic option the variant chip and inspector row are declared as.
    package static let variantFlag = "task-variant"

    // MARK: Slots

    /// The attachment well's slots for a template: every positional the contract declares as a
    /// file (a repeatable one is an ordered list), then every file or directory option in the
    /// Inputs group that is not an output, a model location, or a composite editor's flag.
    /// Accepted types come from the template's own `inputKind` for the primary input and from a
    /// per-flag table for the rest; a directory option takes a folder.
    package static func slots(for templateID: CommandTemplateID) -> [StudioAttachmentSlot] {
        guard let capability = templateID.capability, let template = CommandCatalog.template(id: templateID) else {
            return []
        }
        // A variant that takes no input (`image validate`) has no well; its folders are options.
        if templateID.studioTask.analyzeArchetype?.inputKind(for: templateID) == StudioAnalyzeInputKind.none {
            return []
        }
        var slots: [StudioAttachmentSlot] = []
        for (index, argument) in capability.arguments.enumerated() where argument.kind == .file || argument.kind == .directory {
            let types: [UTType]
            if argument.kind == .directory {
                types = [.folder]
            } else if slots.isEmpty, !template.inputKind.allowedTypes.isEmpty {
                // The first file positional is the template's own input, wherever the contract
                // places it (`sfx video generate <prompt> <input>`).
                types = template.inputKind.allowedTypes
            } else {
                types = acceptedTypes(forArgument: argument.name, templateID: templateID)
            }
            slots.append(StudioAttachmentSlot(
                id: argument.name,
                label: argument.label,
                acceptedTypes: types,
                storage: argument.repeatable ? .argumentList(index) : .argument(index),
                isRequired: argument.required
            ))
        }
        let excluded = outputFlags(for: capability).union(chosenOutputFlags).union(modelLocationFlags)
            .union(overrideFlags(for: templateID))
        for option in capability.options where [.file, .directory].contains(option.kind) {
            guard StudioContractGroup(contractGroup: option.group) == .inputs, !excluded.contains(option.flag) else { continue }
            let types: [UTType] = option.kind == .directory ? [.folder] : acceptedTypes(forFlag: option.flag)
            slots.append(StudioAttachmentSlot(
                id: option.flag,
                label: option.label,
                acceptedTypes: types,
                storage: option.repeatable ? .flagList(option.flag) : .flag(option.flag),
                isRequired: option.required
            ))
        }
        return slots
    }

    /// The slot the Analyze canvas shows large: the first one.
    package static func primarySlot(for templateID: CommandTemplateID) -> StudioAttachmentSlot? {
        slots(for: templateID).first
    }

    private static func acceptedTypes(forArgument name: String, templateID: CommandTemplateID) -> [UTType] {
        switch templateID {
        case .visionFlow, .visionFaceCompare, .visionFaceBatch, .visionGeometryMultiview: return [.image]
        default: return [.data]
        }
    }

    private static func acceptedTypes(forFlag flag: String) -> [UTType] {
        switch flag {
        case "--input-list": return [.plainText]
        case "--view", "--second", "--image", "--ref-image": return [.image]
        case "--audio", "--ref-audio", "--source-audio", "--reference-audio", "--driving-audio": return [.audio]
        case "--video", "--driving-video": return [.movie, .video, .audiovisualContent]
        case "--plan", "--dataset", "--manifest", "--cameras": return [.json]
        default: return [.data]
        }
    }

    /// Flags whose value is a model location rather than an input: the model chip's business.
    private static let modelLocationFlags: Set<String> = ["--model-path", "--model-root", "--lora", "--adapter"]

    // MARK: Output

    /// The destination flags routing fills (`StudioOutputLocation.destination(for:)`): the
    /// capability's own output flag and every sidecar derived beside it.
    package static func outputFlags(for capability: MereRunCommandCapability) -> Set<String> {
        var flags = StudioOutputLocation.sidecarFlags
        if let flag = capability.output.flag { flags.insert(flag) }
        return flags.intersection(capability.options.map(\.flag))
    }

    /// Destinations routing does not fill because they change what the run does or where a
    /// secondary result goes (`image run-plan --materialize`, `image dataset discover
    /// --training-output-root`, the music sidecars): shown in the inspector's Output section as
    /// path rows, never as well slots.
    package static let chosenOutputFlags: Set<String> = [
        "--materialize", "--training-output-root", "--structured-prompt-output", "--recipe-output", "--lrc-output",
        "--daw-bundle",
    ]

    // MARK: Fields

    /// Flags the inspector never shows: the destinations routing fills, and the machine-readable
    /// switches the launcher owns (`--json` comes from `StudioTaskDraft.launcherDefaults`;
    /// `--receipt` and `--progress-json` are added at launch). `--preflight` and `--dry-run` stay
    /// visible: the pages offered them, and a preflight is how a run plan is checked.
    package static func hiddenFlags(for capability: MereRunCommandCapability) -> Set<String> {
        outputFlags(for: capability).union(["--json", "--receipt", "--progress-json"])
    }

    /// The composite editors a template's options render as, keyed by flag. `--model` is the
    /// filtered picker everywhere; the rest are the editors the pages drew by hand, which the
    /// task inspector renders through its override builder (falling back to the plain control
    /// until a page PR lands the editor).
    package static func overrideID(forFlag flag: String, templateID: CommandTemplateID) -> StudioContractOverrideID? {
        switch flag {
        case "--model": return .model
        case "--cameras": return .cameras
        // `--view` stays a well slot (an ordered list the filmstrip draws); `.orderedViews` is
        // the reordering editor the 3D page PR adds beside it.
        case "--manifest": return .musicManifest
        case "--face-index", "--reference-face-index", "--candidate-face-index": return .faceIndex
        case "--instruments": return .instruments
        case "--renoise", "--renoise-strength": return .renoise
        case "--target-rank": return .targetRanks
        default: return nil
        }
    }

    private static func overrideFlags(for templateID: CommandTemplateID) -> Set<String> {
        guard let capability = templateID.capability else { return [] }
        return Set(capability.options.map(\.flag).filter { overrideID(forFlag: $0, templateID: templateID) != nil })
    }

    /// Every option of the draft's template the inspector and chips can edit, in contract order:
    /// the variant first when the task has several templates, then the options minus the slots
    /// the well owns, the destinations routing fills, and the prompt the composer shows.
    package static func fields(for task: StudioTask, draft: StudioTaskDraft) -> [StudioContractField<StudioTaskDraft>] {
        guard let capability = draft.capability else { return [] }
        var fields: [StudioContractField<StudioTaskDraft>] = []
        if let variant = variantField(for: task) { fields.append(variant) }
        let slotFlags = Set(slots(for: draft.templateID).compactMap { slot -> String? in
            if case .flag(let flag) = slot.storage { return flag }
            if case .flagList(let flag) = slot.storage { return flag }
            return nil
        })
        let hidden = hiddenFlags(for: capability).union(slotFlags)
        var prompt: String?
        if case .flag(let flag) = promptField(for: capability) { prompt = flag }
        var claimed: Set<StudioContractOverrideID> = []
        for declared in capability.options where !hidden.contains(declared.flag) && declared.flag != prompt {
            // A model location the contract filed under Inputs (a `.directory` option) belongs
            // with the model it points at; a destination the user chooses belongs under Output.
            let option: MereRunCapabilityOption
            if modelLocationFlags.contains(declared.flag) {
                option = declared.filed(under: .model)
            } else if chosenOutputFlags.contains(declared.flag) {
                option = declared.filed(under: .output)
            } else {
                option = declared
            }
            if let override = overrideID(forFlag: option.flag, templateID: draft.templateID) {
                // A composite editor renders once, where the first of its flags is declared.
                guard claimed.insert(override).inserted else { continue }
                let owned = capability.options.filter { overrideID(forFlag: $0.flag, templateID: draft.templateID) == override }
                fields.append(StudioContractField(
                    option: option, bindings: owned.map { .flag($0.flag) }, overrideID: override
                ))
                continue
            }
            fields.append(StudioContractField(option: option, bindings: [.flag(option.flag)]))
        }
        return fields
    }

    /// Which of the task's templates runs, as a choice the form renders like Read Image's task
    /// picker. Nil for a task with one template.
    package static func variantField(for task: StudioTask) -> StudioContractField<StudioTaskDraft>? {
        let templates = task.variantTemplates
        guard templates.count > 1 else { return nil }
        return StudioContractField(
            option: MereRunCapabilityOption(
                flag: variantFlag,
                label: task == .threeDFromImage ? "Engine" : "Operation",
                kind: .choice,
                choices: templates.map(\.id.rawValue),
                defaultValue: templates[0].id.rawValue,
                group: MereRunCapabilityOptionGroup.prompt,
                tier: .essential
            ),
            bindings: [.variant],
            overrideID: .variant
        )
    }

    /// The user-facing name of a variant choice: the template's title.
    package static func variantTitle(_ choice: String) -> String {
        CommandTemplateID(rawValue: choice).flatMap(CommandCatalog.template(id:))?.title ?? choice
    }

    /// The two to four essentials shown as chips under the prompt, in contract order.
    package static func essentials(for task: StudioTask, draft: StudioTaskDraft) -> [StudioContractField<StudioTaskDraft>] {
        fields(for: task, draft: draft).filter { $0.tier == .essential && $0.overrideID != .model }
    }

    /// The inspector's sections: `essential` and `standard` fields by group, in group order.
    package static func sections(for task: StudioTask, draft: StudioTaskDraft) -> [StudioTaskSection] {
        let fields = fields(for: task, draft: draft).filter { $0.tier != .expert }
        return StudioContractGroup.allCases.compactMap { group in
            let grouped = fields.filter { $0.group == group }
            guard !grouped.isEmpty else { return nil }
            return StudioTaskSection(group: group, fields: grouped)
        }
    }

    /// Everything the template takes that collapses under "Advanced · N more".
    package static func advanced(for task: StudioTask, draft: StudioTaskDraft) -> [StudioContractField<StudioTaskDraft>] {
        fields(for: task, draft: draft).filter { $0.tier == .expert }
    }

    /// Whether each flag carries a value and what it depends on, for `ContractForm`'s gating.
    package static func dependencies(for draft: StudioTaskDraft) -> [String: (carries: Bool, dependsOn: String?)] {
        guard let capability = draft.capability else { return [:] }
        return StudioConsoleCommand.dependencies(for: capability, draft: draft.form)
    }

    /// How many fields differ from the template's fresh draft; the inspector's badge.
    package static func changedCount(for task: StudioTask, draft: StudioTaskDraft) -> Int {
        let baseline = StudioTaskDraft(templateID: draft.templateID)
        return fields(for: task, draft: draft).reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
    }

    // MARK: Prompt

    /// Where the template's free text goes, or nil when it takes none.
    package static func promptField(for capability: MereRunCommandCapability) -> StudioTaskPromptField? {
        if let index = capability.arguments.firstIndex(where: {
            $0.kind == .string && ["prompt", "texts", "text", "caption", "query"].contains($0.name)
        }) {
            return .argument(index, repeatable: capability.arguments[index].repeatable)
        }
        if let option = capability.options.first(where: { ["--prompt", "--text", "--query"].contains($0.flag) }) {
            return .flag(option.flag)
        }
        return nil
    }

    /// The model scope the chip, the inspector row, and the readiness card share for a draft.
    package static func modelScope(for draft: StudioTaskDraft) -> StudioModelScope {
        StudioModelScope(templateID: draft.templateID)
    }

    /// The model the draft will run: its `--model`, else the template's default; empty when the
    /// template runs no managed model.
    package static func modelID(for draft: StudioTaskDraft) -> String {
        modelScope(for: draft).resolvedModelID(model: draft.text("--model"))
    }
}

extension MereRunCapabilityOption {
    /// The same option under another of the contract's groups.
    fileprivate func filed(under group: StudioContractGroup) -> MereRunCapabilityOption {
        MereRunCapabilityOption(
            flag: flag, label: label, kind: kind, required: required, repeatable: repeatable, choices: choices,
            defaultValue: defaultValue, group: group.rawValue, tier: tier, range: range, dependsOn: dependsOn
        )
    }
}

extension StudioTaskDraft {
    /// The free text the composer's prompt field edits: the prompt positional (or one line per
    /// argument for a repeatable one) or the `--prompt` option; empty for a template with none.
    package var prompt: String {
        get {
            switch capability.flatMap(StudioTaskSchema.promptField) {
            case .argument(let index, let repeatable):
                return repeatable ? form.arguments.dropFirst(index).joined(separator: "\n") : argument(index)
            case .flag(let flag):
                return text(flag)
            case nil:
                return ""
            }
        }
        set {
            switch capability.flatMap(StudioTaskSchema.promptField) {
            case .argument(let index, let repeatable):
                if repeatable {
                    let lines = newValue.components(separatedBy: .newlines).filter { !$0.isBlank }
                    form.arguments = Array(form.arguments.prefix(index)) + lines
                } else {
                    setArgument(index, newValue)
                }
            case .flag(let flag):
                form[flag] = newValue.isEmpty ? .unset : .text(newValue)
            case nil:
                break
            }
        }
    }

    /// The path of the primary input, for the Analyze canvas and Library input identity.
    package var primaryInputPath: String {
        StudioTaskSchema.primarySlot(for: templateID)?.paths(in: self).first ?? ""
    }

    package var slots: [StudioAttachmentSlot] {
        StudioTaskSchema.slots(for: templateID)
    }

    /// The model this draft names, as the model chip binds it.
    package var model: String {
        get { text("--model") }
        set { form["--model"] = newValue.isEmpty ? .unset : .text(newValue) }
    }
}
