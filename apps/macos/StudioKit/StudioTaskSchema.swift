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
    /// One flag's entry. Setting a flag the CLI refuses beside another
    /// (`StudioTaskSchema.exclusiveFlags`) clears that other one.
    package static func flag(_ flag: String) -> Self {
        let inner = StudioContractBinding<StudioConsoleDraft>.flag(flag)
        return Self(fieldID: flag, read: { inner.read($0.form) }, write: { draft, value in
            inner.write(&draft.form, value)
            if !draft.text(flag).isBlank, let other = StudioTaskSchema.exclusiveFlags(for: draft.templateID)[flag] {
                draft.form.values[other] = nil
            }
        })
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
    /// Inputs group (`image dataset discover --root` among them) that is not an output, a model
    /// location, or a composite editor's flag. Accepted types come from the template's own
    /// `inputKind` for the primary input and from a per-flag table for the rest; a directory
    /// option takes a folder. With a scope, an option slot the model does not take is left out;
    /// a file in it stays in the draft for when the user switches back.
    package static func slots(for templateID: CommandTemplateID, scope: StudioOptionScope? = nil) -> [StudioAttachmentSlot] {
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
            } else if let explicit = acceptedTypes(forArgument: argument.name, templateID: templateID) {
                types = explicit
            } else if slots.isEmpty, !template.inputKind.allowedTypes.isEmpty {
                // The first file positional is the template's own input, wherever the contract
                // places it (`sfx video generate <prompt> <input>`).
                types = template.inputKind.allowedTypes
            } else {
                types = [.data]
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
        var optionSlots: [StudioAttachmentSlot] = []
        for option in scope?.options ?? capability.options where [.file, .directory].contains(option.kind) {
            guard StudioContractGroup(contractGroup: option.group) == .inputs, !excluded.contains(option.flag) else { continue }
            let types: [UTType] = option.kind == .directory ? [.folder] : acceptedTypes(forFlag: option.flag)
            optionSlots.append(StudioAttachmentSlot(
                id: option.flag,
                label: option.label,
                acceptedTypes: types,
                storage: option.repeatable ? .flagList(option.flag) : .flag(option.flag),
                isRequired: option.required
            ))
        }
        // The required option comes first in the well, whatever the contract's declaration
        // order: it is the input the run is about (`text train-lora --data`, not its optional
        // resume checkpoint), so the canvas shows it large and the output is named after it.
        return slots + optionSlots.filter(\.isRequired) + optionSlots.filter { !$0.isRequired }
    }

    /// The slot the Analyze canvas shows large: the first one.
    package static func primarySlot(for templateID: CommandTemplateID) -> StudioAttachmentSlot? {
        slots(for: templateID).first
    }

    /// What a template's file positional takes when its `inputKind` does not say: the second
    /// image of a pair, or Foley's clip that may also be the Synchformer features the CLI
    /// accepts in its place (`.npy`). Nil leaves it to the template's input kind.
    private static func acceptedTypes(forArgument name: String, templateID: CommandTemplateID) -> [UTType]? {
        switch templateID {
        case .visionFlow, .visionFaceCompare, .visionFaceBatch, .visionGeometryMultiview: return [.image]
        case .sfxVideo: return [.movie, .video, .audiovisualContent, .data]
        default: return nil
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
    /// The contract files an ungrouped directory option under Inputs, so ACE-Step's checkpoint
    /// root is named here to keep it out of the well and in the Model section.
    private static let modelLocationFlags: Set<String> = [
        "--model-path", "--model-root", "--lora", "--adapter", "--checkpoints-root",
    ]

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

    /// Options of one template the CLI refuses together, each mapped to the one it excludes:
    /// `image run-plan` either checks a plan (`--preflight`) or writes its run folder
    /// (`--materialize`). `StudioCommandChecks` refuses a form that holds both anyway.
    package static func exclusiveFlags(for templateID: CommandTemplateID) -> [String: String] {
        switch templateID {
        case .imageRunPlan: return ["--preflight": "--materialize", "--materialize": "--preflight"]
        default: return [:]
        }
    }

    // MARK: Fields

    /// Flags the inspector never shows: the destinations routing fills, and the machine-readable
    /// switches the launcher owns (`--json` and `--pretty` come from
    /// `StudioTaskDraft.launcherDefaults`; `--receipt` and `--progress-json` are added at
    /// launch). `--preflight` and `--dry-run` stay visible: the pages offered them, and a
    /// preflight is how a run plan is checked.
    package static func hiddenFlags(for capability: MereRunCommandCapability) -> Set<String> {
        outputFlags(for: capability).union(["--json", "--pretty", "--receipt", "--progress-json"])
    }

    /// The composite editors a template's options render as, keyed by flag. `--model` is the
    /// filtered picker everywhere; the task inspector renders the other composite editors
    /// through its override builder.
    package static func overrideID(forFlag flag: String, templateID: CommandTemplateID) -> StudioContractOverrideID? {
        switch flag {
        case "--model": return .model
        case "--cameras": return .cameras
        // `--view` stays a well slot (an ordered list the filmstrip draws); `.orderedViews` is
        // the reordering editor the 3D inspector draws beside it.
        case "--manifest": return .musicManifest
        // Music ▸ Train's dataset is the clip list the page edits and writes beside the adapter.
        case "--dataset" where templateID == .musicTrainAdapter: return .musicManifest
        case "--face-index", "--reference-face-index", "--candidate-face-index": return .faceIndex
        // `--list-instruments` is how the editor reads its choices, not a setting of the run.
        case "--instruments", "--list-instruments": return .instruments
        case "--renoise", "--renoise-strength": return .renoise
        case "--lora-target-ranks": return .targetRanks
        case "--dimensions" where templateID == .geoTessera: return .earthDimensions
        case "--patch-size", "--input-resolution" where templateID == .geoOlmoEarth: return .earthSampling
        default: return nil
        }
    }

    private static func overrideFlags(for templateID: CommandTemplateID) -> Set<String> {
        guard let capability = templateID.capability else { return [] }
        return Set(capability.options.map(\.flag).filter { overrideID(forFlag: $0, templateID: templateID) != nil })
    }

    /// Every option of the draft's template the inspector and chips can edit, in contract order:
    /// the variant first when the task has several templates, then the options the model the
    /// draft runs takes (`StudioOptionScope`, narrowed to its family), minus the slots the well
    /// owns, the destinations routing fills, and the prompt the composer shows.
    package static func fields(
        for task: StudioTask,
        draft: StudioTaskDraft,
        source: StudioScopeSource = .live
    ) -> [StudioContractField<StudioTaskDraft>] {
        guard let scope = source.scope(for: draft) else { return [] }
        let capability = scope.capability
        var fields: [StudioContractField<StudioTaskDraft>] = []
        if let variant = variantField(for: task) { fields.append(variant) }
        let slotFlags = Set(slots(for: draft.templateID).compactMap { slot -> String? in
            if case .flag(let flag) = slot.storage { return flag }
            if case .flagList(let flag) = slot.storage { return flag }
            return nil
        })
        let hidden = hiddenFlags(for: capability).union(slotFlags)
        let options = scope.options
        var prompt: String?
        if case .flag(let flag) = promptField(for: capability) { prompt = flag }
        var claimed: Set<StudioContractOverrideID> = []
        for declared in options where !hidden.contains(declared.flag) && declared.flag != prompt {
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
                let owned = options.filter { overrideID(forFlag: $0.flag, templateID: draft.templateID) == override }
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
        // Filed with the prompt when the task takes one, else with the inputs, so a prompt-less
        // task's inspector never opens with a "Prompt" section.
        let takesPrompt = templates[0].id.capability.flatMap(promptField) != nil
        return StudioContractField(
            option: MereRunCapabilityOption(
                flag: variantFlag,
                label: task == .threeDFromImage ? "Engine" : "Operation",
                kind: .choice,
                choices: templates.map(\.id.rawValue),
                defaultValue: templates[0].id.rawValue,
                group: takesPrompt ? MereRunCapabilityOptionGroup.prompt : MereRunCapabilityOptionGroup.inputs,
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
    package static func essentials(
        for task: StudioTask,
        draft: StudioTaskDraft,
        source: StudioScopeSource = .live
    ) -> [StudioContractField<StudioTaskDraft>] {
        fields(for: task, draft: draft, source: source).filter { $0.tier == .essential && $0.overrideID != .model }
    }

    /// The inspector's sections: `essential` and `standard` fields by group, in group order.
    package static func sections(
        for task: StudioTask,
        draft: StudioTaskDraft,
        source: StudioScopeSource = .live
    ) -> [StudioTaskSection] {
        let fields = fields(for: task, draft: draft, source: source).filter { $0.tier != .expert }
        return StudioContractGroup.allCases.compactMap { group in
            let grouped = fields.filter { $0.group == group }
            guard !grouped.isEmpty else { return nil }
            return StudioTaskSection(group: group, fields: grouped)
        }
    }

    /// Everything the template takes that collapses under "Advanced · N more".
    package static func advanced(
        for task: StudioTask,
        draft: StudioTaskDraft,
        source: StudioScopeSource = .live
    ) -> [StudioContractField<StudioTaskDraft>] {
        fields(for: task, draft: draft, source: source).filter { $0.tier == .expert }
    }

    /// Whether each flag carries a value and what it depends on, for `ContractForm`'s gating.
    package static func dependencies(for draft: StudioTaskDraft) -> [String: (carries: Bool, dependsOn: String?)] {
        guard let capability = draft.capability else { return [:] }
        return StudioConsoleCommand.dependencies(for: capability, draft: draft.form)
    }

    /// How many fields differ from the template's fresh draft; the inspector's badge.
    package static func changedCount(for task: StudioTask, draft: StudioTaskDraft, source: StudioScopeSource = .live) -> Int {
        let baseline = StudioTaskDraft(templateID: draft.templateID)
        return fields(for: task, draft: draft, source: source).reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
    }

    /// The note under the task inspector's header and the composer's chips: the values the draft
    /// holds, away from the template's fresh draft, that the model it runs leaves out or replaces.
    package static func notice(for draft: StudioTaskDraft, source: StudioScopeSource = .live) -> StudioScopeNotice? {
        source.scope(for: draft)?.notice(form: draft.form, baseline: StudioTaskDraft(templateID: draft.templateID).form)
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
    /// For a routed command its default is what the form runs with its model cleared, so
    /// "Auto" names the model the contract picks for the options the form holds.
    package static func modelScope(for draft: StudioTaskDraft, source: StudioScopeSource = .live) -> StudioModelScope {
        var scope = StudioModelScope(templateID: draft.templateID, source: source)
        if let capability = source.capability(for: draft.templateID), let routing = capability.routing {
            var unset = draft.form
            for flag in routing.modelFlags { unset.values[flag] = nil }
            if let model = source.scope(capability: capability, form: unset).managedModel { scope.defaultModelID = model }
        }
        return scope
    }

    /// The model the draft will run. For a routed command, the model its scope resolves to: the
    /// managed model it names or defaults to, else what the form names. Otherwise its
    /// `--model`, else the base its training recipe trains (`StudioTrainingRun.recipeBaseModel`),
    /// else the template's default; empty when the template runs no managed model.
    package static func modelID(for draft: StudioTaskDraft, source: StudioScopeSource = .live) -> String {
        let model = draft.text("--model")
        if let scope = source.scope(for: draft), scope.capability.routing != nil {
            return scope.managedModel ?? model
        }
        if model.isBlank, let base = StudioTrainingRun.recipeBaseModel(for: draft) { return base }
        return modelScope(for: draft, source: source).resolvedModelID(model: model)
    }

    /// What the readiness check asks for before a run: the managed model `modelID(for:)` names,
    /// or nothing when the run reads its weights from a folder on disk — a `--model` that is a
    /// path (the Woosh commands take a local checkpoints root there) or a local model location the
    /// trainers take beside the id (`--model-path`, `--checkpoints-root`); the CLI resolves those
    /// itself and `model list` has no row for them. A model the command excludes, or whose
    /// selectors match no family, blocks the run with the CLI gate's reason.
    package static func requirement(for draft: StudioTaskDraft, source: StudioScopeSource = .live) -> StudioCapabilityRequirement? {
        if let reason = source.scope(for: draft)?.blockingReason { return .unavailable(reason) }
        if isLocalPath(draft.text("--model").trimmingCharacters(in: .whitespacesAndNewlines)) { return nil }
        if ["--model-path", "--checkpoints-root"].contains(where: { !draft.text($0).isBlank }) { return nil }
        let model = modelID(for: draft, source: source)
        return model.isBlank ? nil : .managedModel(model)
    }

    /// The managed model `requirement(for:)` asks for; empty when it asks for none.
    package static func requiredModelID(for draft: StudioTaskDraft, source: StudioScopeSource = .live) -> String {
        guard case .managedModel(let model)? = requirement(for: draft, source: source) else { return "" }
        return model
    }

    /// Whether a `--model` value names a folder rather than a managed id: it starts at the root,
    /// the home folder, or the working directory, or has a folder in it.
    package static func isLocalPath(_ model: String) -> Bool {
        model.hasPrefix("~") || model.hasPrefix(".") || model.contains("/")
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
    /// argument for a repeatable one whose template splits lines) or the `--prompt` option;
    /// empty for a template with none.
    package var prompt: String {
        get {
            switch capability.flatMap(StudioTaskSchema.promptField) {
            case .argument(let index, let repeatable):
                return repeatable && template?.promptSplitsLines != false
                    ? form.arguments.dropFirst(index).joined(separator: "\n")
                    : argument(index)
            case .flag(let flag):
                return text(flag)
            case nil:
                return ""
            }
        }
        set {
            switch capability.flatMap(StudioTaskSchema.promptField) {
            case .argument(let index, let repeatable):
                if repeatable, template?.promptSplitsLines != false {
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

    /// The well's slots for the model the draft runs.
    package var slots: [StudioAttachmentSlot] {
        slots(source: .live)
    }

    package func slots(source: StudioScopeSource) -> [StudioAttachmentSlot] {
        StudioTaskSchema.slots(for: templateID, scope: source.scope(for: self))
    }

    /// The model this draft names, as the model chip binds it.
    package var model: String {
        get { text("--model") }
        set { form["--model"] = newValue.isEmpty ? .unset : .text(newValue) }
    }
}
