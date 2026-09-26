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
    /// The task's other variants as the user left them: switching back to one restores its form
    /// whole — InstantMesh's ordered views and cameras, Compare's second picture — rather than
    /// only what the two templates share.
    package private(set) var parked: [CommandTemplateID: StudioConsoleDraft] = [:]
    /// The files the primary slot runs one at a time when it batches; empty for a single run.
    package var batchInputPaths: [String] = []

    package init(templateID: CommandTemplateID, form: StudioConsoleDraft) {
        self.templateID = templateID
        self.form = form
    }

    private enum CodingKeys: String, CodingKey {
        case templateID, form, parked, batchInputPaths
    }

    /// A draft saved before variants were parked has no `parked` entry; one parked for a
    /// template this build no longer has drops that form. One saved before batches has none.
    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        templateID = try container.decode(CommandTemplateID.self, forKey: .templateID)
        form = try container.decode(StudioConsoleDraft.self, forKey: .form)
        let parked = try container.decodeIfPresent([String: StudioConsoleDraft].self, forKey: .parked) ?? [:]
        for (key, form) in parked {
            guard let id = CommandTemplateID(rawValue: key) else { continue }
            self.parked[id] = form
        }
        batchInputPaths = try container.decodeIfPresent([String].self, forKey: .batchInputPaths) ?? []
    }

    /// Parked forms are keyed by template id, so the file reads as an object of forms.
    package func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(templateID, forKey: .templateID)
        try container.encode(form, forKey: .form)
        if !parked.isEmpty {
            try container.encode(Dictionary(uniqueKeysWithValues: parked.map { ($0.key.rawValue, $0.value) }), forKey: .parked)
        }
        if !batchInputPaths.isEmpty {
            try container.encode(batchInputPaths, forKey: .batchInputPaths)
        }
    }

    /// A fresh draft for `templateID`: the console's reading of the template's default draft, so
    /// the workspace starts on exactly the command the page and the Command view already ran,
    /// plus the launcher switches the page set on every run (`launcherDefaults`). The default
    /// draft's stamped destination is not carried: a draft never holds a destination the app
    /// names, so routing names a fresh one for each run, with the extension its `--format` asks
    /// for and under whatever root is configured when it runs.
    ///
    /// The reading is of the template's argv before any scope: a default only another model uses
    /// (UniverSR's seed and solver while the template defaults to AP-BWE) is in the form, hidden
    /// and left out of the run, so switching to that model runs what its page ran.
    package init(templateID: CommandTemplateID) {
        self.templateID = templateID
        guard let template = CommandCatalog.template(id: templateID) else {
            form = StudioConsoleDraft()
            return
        }
        let defaults = template.defaultDraft()
        form = templateID.capability.map {
            // The template's own defaults name a managed model the contract places, so the
            // shipped contract alone builds the argv a fresh draft starts from.
            StudioConsoleCommand.seed(capability: $0, arguments: template.unscopedArguments(from: defaults, source: .contract))
        } ?? StudioConsoleDraft(extraArguments: defaults.extraArguments)
        for flag in Self.launcherDefaults(for: templateID) where form.values[flag] == nil {
            form[flag] = .flag(true)
        }
        for (flag, value) in Self.pageValues(for: templateID) where form.values[flag] == nil {
            form[flag] = value
        }
        for flag in Self.consoleOnlyDefaults(for: templateID) {
            form.values[flag] = nil
        }
        self = withoutDestinations()
    }

    /// The switches the catalog turns on so the Command Console opens a heavy command safely
    /// (`--dry-run` on the depth and geometry commands) but a page never ran with: a fresh draft
    /// on the workspace runs for real, and the inspector's Dry run row stays for a preflight.
    package static func consoleOnlyDefaults(for templateID: CommandTemplateID) -> [String] {
        switch templateID {
        case .visionDepth, .visionDepthVideo, .visionGeometry, .visionGeometryMultiview:
            return ["--dry-run"]
        default:
            return []
        }
    }

    /// The switches a page turned on for every run of a template because the surface reads the
    /// command's machine output: `--json` where the result is printed as JSON and a renderer
    /// decodes it, `--pretty` where the JSON view shows what was printed. Applied to a fresh
    /// draft only; a parked or restored draft keeps what it ran with.
    package static func launcherDefaults(for templateID: CommandTemplateID) -> [String] {
        switch templateID {
        case .imageDatasetDiscover, .imageRunPlan,
             .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare, .visionPose, .visionFlow,
             .visionDepth, .visionDepthVideo, .visionGeometry, .visionGeometryMultiview:
            return ["--json"]
        case .visionTrackLive:
            // The Live page drew boxes and labels on the clip unless they were turned off.
            return ["--json", "--show-boxes", "--show-labels"]
        case .textEmbed:
            return ["--pretty"]
        case .textAnonymize:
            return ["--json", "--pretty"]
        default:
            return []
        }
    }

    /// The values a page sent on every run where the template leaves the option to the CLI:
    /// the Faces page always named face 0, the one its picker shows as face 1, while the CLI on
    /// its own takes the largest face, which nothing on the page could show.
    package static func pageValues(for templateID: CommandTemplateID) -> [String: StudioContractValue] {
        switch templateID {
        case .visionFaceEmbed:
            return ["--face-index": .integer(0)]
        case .visionFaceCompare:
            return ["--reference-face-index": .integer(0), "--candidate-face-index": .integer(0)]
        default:
            return [:]
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

    /// Exactly what the Command view's "Will run" shows for the same form: the form scoped to the
    /// model it runs, so a value that model does not use is kept in the form but not run.
    package func arguments(source: StudioScopeSource) -> [String] {
        guard let capability = source.capability(for: templateID) else { return [] }
        return StudioConsoleCommand.arguments(for: capability, draft: form.scoped(to: source.scope(capability: capability, form: form)))
    }

    /// The launch for this draft, or nil for a template the app cannot run itself.
    package func run(source: StudioScopeSource) -> StudioConsoleRun? {
        guard let template else { return nil }
        return StudioConsoleRun(template: template, draft: form, seed: seed, source: source)
    }

    /// The request the task runner submits: the template's own Library attribution, the console
    /// projection as the draft, and the argv as the execution history and replay keep.
    package func request(id: UUID = UUID(), createdAt: Date = Date(), source: StudioScopeSource) -> StudioRunRequest? {
        guard let template, let run = run(source: source) else { return nil }
        return StudioRunRequest(
            id: id, mode: template.libraryMode, templateID: templateID, template: template,
            draft: run.commandDraft, createdAt: createdAt,
            execution: StudioExecution(templateID: templateID, arguments: run.arguments)
        )
    }

    /// Switches the variant. The form being left is parked; a variant the user has had before
    /// comes back exactly as it was left. One opened for the first time starts fresh and
    /// carries the values whose flags the new template also declares (the input, the model)
    /// and drops the rest, so Faces ▸ Compare keeps the picture Detect was pointed at. The model
    /// is cleared when the templates default to different models: a face model is no use to the
    /// pose command. A batch carries to a variant whose own input batches and takes every file in
    /// it (Faces ▸ Detect to Embed); any other switch leaves the new variant one file.
    package mutating func switchTemplate(to next: CommandTemplateID) {
        guard next != templateID else { return }
        let batch = batchInputPaths
        switchForm(to: next)
        batchInputPaths = []
        if !batch.isEmpty, let slot = StudioTaskSchema.primarySlot(for: next), slot.batches,
           batch.allSatisfy({ slot.accepts(URL(fileURLWithPath: $0)) }) {
            slot.setBatch(batch, in: &self)
        }
    }

    private mutating func switchForm(to next: CommandTemplateID) {
        let previous = self
        var parked = previous.parked
        parked[previous.templateID] = previous.form
        defer { self.parked = parked }
        if let restored = parked.removeValue(forKey: next) {
            templateID = next
            form = restored
            return
        }
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
        clearingDestinations { _ in true }
    }

    /// The same draft without the destinations that sit in one of the app's own folders
    /// (`StudioOutputLocation.isAppOwned`), whatever the kind of file: a saved draft from before
    /// drafts stopped keeping them, or one named under a root the settings have since moved.
    /// A destination the user chose elsewhere is a setting and stays.
    package func withoutAppDestinations() -> StudioTaskDraft {
        clearingDestinations { StudioOutputLocation.isAppOwned($0) }
    }

    /// Takes a recorded run's settings as the draft ("Use these settings"): its variant and
    /// form, with the variant being left parked like any switch, and the other parked variants
    /// kept.
    package mutating func adopt(_ restored: StudioTaskDraft) {
        switchTemplate(to: restored.templateID)
        form = restored.form
        // The recorded run read one file; a batch left in the well would run over it.
        batchInputPaths = []
    }

    private func clearingDestinations(where clears: (String) -> Bool) -> StudioTaskDraft {
        var cleared = self
        cleared.form = Self.clearingDestinations(of: form, templateID: templateID, where: clears)
        for (id, form) in parked {
            cleared.parked[id] = Self.clearingDestinations(of: form, templateID: id, where: clears)
        }
        return cleared
    }

    private static func clearingDestinations(
        of form: StudioConsoleDraft, templateID: CommandTemplateID, where clears: (String) -> Bool
    ) -> StudioConsoleDraft {
        guard let capability = templateID.capability else { return form }
        var cleared = form
        for flag in StudioTaskSchema.outputFlags(for: capability) where clears(form.text(flag)) {
            cleared.values[flag] = nil
        }
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
    /// Persisted settings never contain launch credentials, the parked variants' included: the
    /// same masking the Command override applies to its form.
    package var withoutSessionSecrets: StudioTaskDraft {
        var saved = self
        saved.form = Self.withoutSecrets(form, templateID: templateID)
        for (id, form) in parked { saved.parked[id] = Self.withoutSecrets(form, templateID: id) }
        return saved
    }

    private static func withoutSecrets(_ form: StudioConsoleDraft, templateID: CommandTemplateID) -> StudioConsoleDraft {
        var saved = form
        for flag in Set(CommandLaunchEnvironment.secretFlags(for: templateID).keys)
            .union(["--api-key", "--infinity-api-key", "--admin-password", "--hf-token"]) {
            saved.values[flag] = nil
        }
        saved.extraArguments = ShellWords.split(saved.extraArguments).maskingSecrets().shellQuoted()
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

    /// Every edit of a task draft lands here — the composer, the inspector, the Command view, a
    /// Reset — so this is where its undo step is registered.
    package func setTaskDraft(_ draft: StudioTaskDraft, for task: StudioTask) {
        let key = Self.taskDraftKey(task)
        let previous = taskDraft(for: task)
        set(draft, for: key)
        guard let previous else { return }
        registerDraftChange(keys: [key], from: previous, to: draft) {
            draft.undoName(from: previous, task: task, source: .contract)
        }
    }

    package static func taskDraftKey(_ task: StudioTask) -> String {
        task.rawValue + ".taskDraft"
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
