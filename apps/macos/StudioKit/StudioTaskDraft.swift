import Foundation
import MereRunContract

// A specialist task's canonical draft is a contract form for its template, not a page-local
// `CommandDraft` or a new field set: the composer's well and chips, the inspector, the Command
// view, Library restoration, and the argv all read one `StudioConsoleDraft`, so parity between
// them is structural rather than tested. `StudioConsoleDraftTests` already holds the seed →
// rebuild identity for every template, which is what makes the console's form safe to be the
// draft.

/// The draft of a task whose surface is the shared task workspace: which of the task's templates
/// runs, and one value per flag and positional of that template's capability.
package struct StudioTaskDraft: Codable, Equatable {
    /// The chosen variant, one of `StudioTask.variantTemplates`.
    package var templateID: CommandTemplateID
    /// One entry per flag the run carries, plus the positionals and anything typed by hand.
    package var form: StudioConsoleDraft

    package init(templateID: CommandTemplateID, form: StudioConsoleDraft) {
        self.templateID = templateID
        self.form = form
    }

    /// A fresh draft for `templateID`: the console's reading of the template's default draft, so
    /// the workspace starts on exactly the command the page and the Command view already ran,
    /// plus the launcher switches the page set on every run (`launcherDefaults`).
    package init(templateID: CommandTemplateID) {
        self.templateID = templateID
        guard let template = CommandCatalog.template(id: templateID) else {
            form = StudioConsoleDraft()
            return
        }
        form = StudioConsoleCommand.seed(template: template, draft: template.defaultDraft())
        for flag in Self.launcherDefaults(for: templateID) where form.values[flag] == nil {
            form[flag] = .flag(true)
        }
    }

    /// The switches a page turned on for every run of a template because the surface reads the
    /// command's machine output: `--json` where the result is printed as JSON and a renderer
    /// decodes it. Applied to a fresh draft only; a parked or restored draft keeps what it ran
    /// with.
    package static func launcherDefaults(for templateID: CommandTemplateID) -> [String] {
        switch templateID {
        case .imageDatasetDiscover, .imageRunPlan,
             .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare, .visionPose, .visionFlow,
             .visionDepth, .visionDepthVideo, .visionGeometry, .visionGeometryMultiview, .visionTrackLive:
            return ["--json"]
        default:
            return []
        }
    }

    /// A fresh draft for a task, on its first variant.
    package init?(task: StudioTask) {
        guard let template = task.variantTemplates.first else { return nil }
        self.init(templateID: template.id)
    }

    package var template: CommandTemplate? {
        CommandCatalog.template(id: templateID)
    }

    package var capability: MereRunCommandCapability? {
        templateID.capability
    }

    /// The `CommandDraft` the job lifecycle reads beside the argv: the template's defaults with
    /// the console's projection over them (`StudioConsoleRun`).
    package var seed: CommandDraft {
        template?.defaultDraft() ?? CommandDraft()
    }

    /// Exactly what the Command view's "Will run" shows for the same form.
    package var arguments: [String] {
        guard let capability else { return [] }
        return StudioConsoleCommand.arguments(for: capability, draft: form)
    }

    /// The launch for this draft, or nil for a template the app cannot run itself.
    package var run: StudioConsoleRun? {
        guard let template else { return nil }
        return StudioConsoleRun(template: template, draft: form, seed: seed)
    }

    /// The request the task runner submits: the template's own Library attribution, the console
    /// projection as the draft, and the argv as the execution history and replay keep.
    package func request(id: UUID = UUID(), createdAt: Date = Date()) -> StudioRunRequest? {
        guard let template, let run else { return nil }
        return StudioRunRequest(
            id: id, mode: template.libraryMode, templateID: templateID, template: template,
            draft: run.commandDraft, createdAt: createdAt,
            execution: StudioExecution(templateID: templateID, arguments: run.arguments)
        )
    }

    /// Switches the variant, carrying the values whose flags the new template also declares
    /// (the input, the model, the output) and dropping the rest, so Faces ▸ Compare keeps the
    /// picture Detect was pointed at. The model is cleared when the templates default to
    /// different models: a face model is no use to the pose command.
    package mutating func switchTemplate(to next: CommandTemplateID) {
        guard next != templateID else { return }
        let previous = self
        self = StudioTaskDraft(templateID: next)
        guard let capability, let before = previous.capability else { return }
        let declared = Set(capability.options.map(\.flag))
        for (flag, value) in previous.form.values where declared.contains(flag) {
            form[flag] = value
        }
        // Positionals carry by position while both templates declare one of the same kind. A
        // repeatable source positional (Batch's images) carries only its first value: the
        // values after it were never a second declared argument.
        let positionals = min(capability.arguments.count, before.arguments.count, previous.form.arguments.count)
        for index in 0..<positionals where capability.arguments[index].kind == before.arguments[index].kind {
            form.arguments[safe: index] = previous.form.arguments[index]
        }
        if previous.template?.defaultModel != template?.defaultModel {
            form.values["--model"] = nil
        }
    }

    /// The same draft with every destination routing fills cleared, so
    /// `StudioOutputLocation.destination(for:)` names fresh ones at submit time. A recorded run's
    /// or a legacy page's `--output` and sidecars were that run's, never settings: carrying them
    /// into a draft would write the next run over the last one's files.
    package func withoutDestinations() -> StudioTaskDraft {
        guard let capability else { return self }
        var cleared = self
        for flag in StudioTaskSchema.outputFlags(for: capability) { cleared.form.values[flag] = nil }
        return cleared
    }

    // MARK: Positionals and flags as text

    package func text(_ flag: String) -> String {
        form.text(flag)
    }

    package func argument(_ index: Int) -> String {
        index < form.arguments.count ? form.arguments[index] : ""
    }

    package mutating func setArgument(_ index: Int, _ text: String) {
        form.arguments[safe: index] = text
    }
}

extension StudioTaskDraft: StudioAttachmentDraft {
    package func attachmentText(for storage: StudioAttachmentSlot.Storage) -> String {
        switch storage {
        case .flag(let flag), .flagList(let flag):
            return form.text(flag)
        case .argument(let index):
            return argument(index)
        case .argumentList(let index):
            return form.arguments.dropFirst(index).joined(separator: "\n")
        case .path, .pathList:
            // The prompt tasks' typed fields; a task draft has none.
            return ""
        }
    }

    package mutating func setAttachmentText(_ text: String, for storage: StudioAttachmentSlot.Storage) {
        switch storage {
        case .flag(let flag), .flagList(let flag):
            form[flag] = text.isEmpty ? .unset : .text(text)
        case .argument(let index):
            setArgument(index, text)
        case .argumentList(let index):
            form.arguments = Array(form.arguments.prefix(index)) + StudioAttachmentSlot.separatedPaths(text)
        case .path, .pathList:
            break
        }
    }

    package mutating func didAttach(to slot: StudioAttachmentSlot) {}
}

extension StudioTaskDraft: StudioSessionPersistable {
    /// Persisted settings never contain launch credentials: the same masking the Command
    /// override applies to its form.
    package var withoutSessionSecrets: StudioTaskDraft {
        var saved = self
        for flag in Set(CommandLaunchEnvironment.secretFlags(for: templateID).keys)
            .union(["--api-key", "--infinity-api-key", "--admin-password", "--hf-token"]) {
            saved.form.values[flag] = nil
        }
        saved.form.extraArguments = ShellWords.split(saved.form.extraArguments).maskingSecrets().shellQuoted()
        return saved
    }
}

extension StudioTaskSessions {
    /// The task's parked draft, or a fresh one on its first variant. Written under
    /// `"<task>.taskDraft"`, beside the prompt tasks' `"<task>.draft"`. The workspace, the
    /// inspector column, and the Command view all read this several times per render, so the
    /// decoded value is memoized against the stored bytes rather than decoded on every read.
    package func taskDraft(for task: StudioTask) -> StudioTaskDraft? {
        let key = Self.taskDraftKey(task)
        if let known = cachedTaskDraft(for: key) { return known }
        guard let fresh = StudioTaskDraftMigration.imported(for: task, from: self) ?? StudioTaskDraft(task: task) else {
            return nil
        }
        // Every reader gets this same value until the first edit parks it, so the Command
        // view's edits land on the draft the workspace shows rather than on a second fresh one.
        rememberFreshTaskDraft(fresh, for: key)
        return fresh
    }

    package func setTaskDraft(_ draft: StudioTaskDraft, for task: StudioTask) {
        set(draft, for: Self.taskDraftKey(task))
    }

    package static func taskDraftKey(_ task: StudioTask) -> String {
        task.rawValue + ".taskDraft"
    }
}

/// Reads the `CommandDraft` a legacy page persisted for a task into the task draft the workspace
/// keeps, once, the first time the workspace opens a task with no draft of its own. Pages that
/// persisted scalar keys rather than a whole draft (Vision, 3D, Utility) start fresh.
@MainActor
package enum StudioTaskDraftMigration {
    /// The session key a legacy page kept a template's `CommandDraft` under, scoped by the task
    /// the way `@StudioStoredValue` scopes it.
    package static func legacyKey(for templateID: CommandTemplateID) -> String? {
        switch templateID {
        case .speechDiarize: return "Voice.diarizationDraft"
        case .speechSynthesize: return "Voice.synthesisDraft"
        case .speechProfileCreate: return "Voice.profileDraft"
        case .speechListen: return "Voice.listenDraft"
        case .speechDiarizeLive: return "Voice.liveDiarizationDraft"
        case .musicAnalyze: return "MusicTools.analyzeDraft"
        case .musicTranscribe: return "MusicTools.transcribeDraft"
        case .musicServe: return "MusicTools.serveDraft"
        case .audioEnhance: return "AudioTools.enhanceDraft"
        case .musicSeparate: return "AudioTools.separateDraft"
        case .sfxVideo: return "SFXLab.videoDraft"
        case .sfxConditionText: return "SFXLab.conditionDraft"
        case .sfxAEEncode: return "SFXLab.encodeDraft"
        case .sfxAEDecode: return "SFXLab.decodeDraft"
        case .sfxClapScore: return "SFXLab.scoreDraft"
        case .imageTrainLoRA, .textTrainLoRA, .musicTrainAdapter: return "Training.draft"
        default: return nil
        }
    }

    /// The imported draft for `task`, or nil when no page draft exists for its first variant.
    /// The page stamped a per-run destination (and its sidecars) into the draft it kept; those
    /// were never settings, so they are cleared and routing names fresh ones.
    package static func imported(for task: StudioTask, from sessions: StudioTaskSessions) -> StudioTaskDraft? {
        guard let template = task.variantTemplates.first,
              let key = legacyKey(for: template.id),
              let draft = sessions.value(for: task.rawValue + "." + key, default: Optional<CommandDraft>.none) else {
            return nil
        }
        return StudioTaskDraft(templateID: template.id, form: StudioConsoleCommand.seed(template: template, draft: draft))
            .withoutDestinations()
    }
}

extension Array where Element == String {
    /// Sets `index`, growing the array with empty strings first, the way the console's positional
    /// binding does.
    fileprivate subscript(safe index: Int) -> String {
        get { index < count ? self[index] : "" }
        set {
            while count <= index { append("") }
            self[index] = newValue
        }
    }
}
