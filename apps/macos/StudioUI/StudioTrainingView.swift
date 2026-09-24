import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Which trainer a Train task hosts: Image ▸ Train, Chat ▸ Train, and Music ▸ Train each show one.
enum StudioTrainingKind: String, CaseIterable, Identifiable {
    case image
    case text
    case music

    var id: String { rawValue }

    init?(task: StudioTask) {
        switch task {
        case .imageTrain: self = .image
        case .chatTrain: self = .text
        case .musicTrain: self = .music
        default: return nil
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .image: .imageTrainLoRA
        case .text: .textTrainLoRA
        case .music: .musicTrainAdapter
        }
    }

    var task: StudioTask {
        switch self {
        case .image: .imageTrain
        case .text: .chatTrain
        case .music: .musicTrain
        }
    }
}

struct StudioTrainingDatasetPreview: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
    let imageURL: URL?
    let audioURL: URL?
}

struct StudioTrainingDatasetSnapshot: Equatable {
    let source: URL
    let totalRecords: Int
    let usableRecords: Int
    let previews: [StudioTrainingDatasetPreview]
    let diagnostics: [String]

    /// Image and text datasets are files on disk; the music dataset is the manifest the page edits,
    /// see `inspect(manifest:)`.
    static func inspect(kind: StudioTrainingKind, path: String) -> StudioTrainingDatasetSnapshot? {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        switch kind {
        case .image:
            return inspectImages(at: url)
        case .text:
            return inspectText(at: url)
        case .music:
            return nil
        }
    }

    /// The music trainer's view of the manifest it is about to write: every clip, those the trainer
    /// would accept, and the problems in the page's words.
    static func inspect(manifest: StudioMusicTrainingManifest) -> StudioTrainingDatasetSnapshot {
        let ready = manifest.readyClipCount()
        return .init(
            source: StudioMusicTrainingManifest.draftFolderURL(),
            totalRecords: manifest.clips.count,
            usableRecords: ready,
            previews: [],
            diagnostics: manifest.problems()
        )
    }

    private static func inspectImages(at root: URL) -> StudioTrainingDatasetSnapshot {
        let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "heic"]
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return .init(
                source: root,
                totalRecords: 0,
                usableRecords: 0,
                previews: [],
                diagnostics: ["The dataset directory could not be read."]
            )
        }
        let images = urls.filter { imageExtensions.contains($0.pathExtension.lowercased()) }
        var previews: [StudioTrainingDatasetPreview] = []
        var missingCaptions = 0
        for image in images {
            let captionURL = image.deletingPathExtension().appendingPathExtension("txt")
            let caption = (try? String(contentsOf: captionURL, encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if caption == nil || caption?.isEmpty == true {
                missingCaptions += 1
            }
            if previews.count < 12 {
                previews.append(
                    .init(
                        id: image.path,
                        title: image.lastPathComponent,
                        detail: caption ?? "Missing matching .txt caption",
                        imageURL: image,
                        audioURL: nil
                    )
                )
            }
        }
        let diagnostics = missingCaptions == 0
            ? ["All discovered images have matching caption files."]
            : ["\(missingCaptions) image(s) are missing matching .txt captions."]
        return .init(
            source: root,
            totalRecords: images.count,
            usableRecords: images.count - missingCaptions,
            previews: previews,
            diagnostics: diagnostics
        )
    }

    private static func inspectText(at url: URL) -> StudioTrainingDatasetSnapshot {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return .init(
                source: url,
                totalRecords: 0,
                usableRecords: 0,
                previews: [],
                diagnostics: ["The JSONL dataset could not be read."]
            )
        }
        let lines = text.split(whereSeparator: \.isNewline)
        var valid = 0
        var previews: [StudioTrainingDatasetPreview] = []
        for (index, line) in lines.enumerated() {
            let row = String(line)
            let isValid = row.data(using: .utf8).flatMap {
                try? JSONSerialization.jsonObject(with: $0)
            } != nil
            if isValid { valid += 1 }
            if previews.count < 8 {
                previews.append(
                    .init(
                        id: "\(url.path)#\(index)",
                        title: "Example \(index + 1)",
                        detail: row.count > 280 ? String(row.prefix(280)) + "…" : row,
                        imageURL: nil,
                        audioURL: nil
                    )
                )
            }
        }
        return .init(
            source: url,
            totalRecords: lines.count,
            usableRecords: valid,
            previews: previews,
            diagnostics: valid == lines.count
                ? ["Every JSONL row parses as JSON."]
                : ["\(lines.count - valid) malformed JSONL row(s) need attention."]
        )
    }
}

struct StudioTrainingEvent: Decodable, Identifiable, Equatable {
    let sequence: Int
    let type: String
    let stage: String?
    let message: String?
    let step: Int?
    let totalSteps: Int?
    let loss: Double?
    let fraction: Double?
    let path: String?

    var id: Int { sequence }

    enum CodingKeys: String, CodingKey {
        case sequence
        case type
        case stage
        case message
        case step
        case totalSteps = "total_steps"
        case loss
        case fraction
        case path
    }
}

struct StudioTrainingSnapshot: Equatable {
    let outputURL: URL
    let events: [StudioTrainingEvent]
    let samples: [URL]
    let checkpoints: [URL]
    let artifacts: [URL]

    var lossPoints: [(step: Int, loss: Double)] {
        events.compactMap {
            guard let step = $0.step, let loss = $0.loss else { return nil }
            return (step, loss)
        }
    }

    var latest: StudioTrainingEvent? { events.last }
    var progress: Double? {
        if let fraction = latest?.fraction { return fraction }
        guard let step = latest?.step, let total = latest?.totalSteps, total > 0 else { return nil }
        return Double(step) / Double(total)
    }

    static func load(outputPath: String) -> StudioTrainingSnapshot? {
        guard !outputPath.isBlank else { return nil }
        let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        let directory = outputURL.deletingLastPathComponent()
        let base = outputURL.deletingPathExtension().lastPathComponent
        let eventURL = directory.appendingPathComponent("\(base).events.jsonl")
        let events = loadEvents(from: eventURL)
        let rootFiles = files(in: directory).filter {
            $0.lastPathComponent.hasPrefix(base)
                || $0.lastPathComponent == "run.json"
        }
        let samples = files(in: directory.appendingPathComponent("samples", isDirectory: true))
            .filter { StudioOutputFileKind.classify($0) == .image }
        let checkpoints = files(in: directory.appendingPathComponent("checkpoints", isDirectory: true))
            .filter { ["safetensors", "zip"].contains($0.pathExtension.lowercased()) }
        return StudioTrainingSnapshot(
            outputURL: outputURL,
            events: events,
            samples: samples.sorted { $0.lastPathComponent < $1.lastPathComponent },
            checkpoints: checkpoints.sorted { $0.lastPathComponent < $1.lastPathComponent },
            artifacts: rootFiles.sorted { $0.lastPathComponent < $1.lastPathComponent }
        )
    }

    private static func loadEvents(from url: URL) -> [StudioTrainingEvent] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            guard let data = String(line).data(using: .utf8) else { return nil }
            return try? decoder.decode(StudioTrainingEvent.self, from: data)
        }
        .sorted { $0.sequence < $1.sequence }
    }

    private static func files(in directory: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ))?.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        } ?? []
    }
}

/// The Train tasks' Project surface: a settings column over the task's `StudioTaskDraft` — the
/// dataset in an attachment well (or the clip list for music), the model picker, the template's
/// options in the sections the page always had, where the adapter will be filed, and Preflight
/// and Start — beside the dashboard that follows the run: live metrics, the loss curve, samples,
/// checkpoints, A/B comparison, and history. Runs go through `StudioTaskRunner`, so Stop, the
/// Library row, and the remembered request are the shared ones, and the root's Command view
/// edits the same draft this column does.
struct StudioTrainingView: View {
    let kind: StudioTrainingKind

    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioModelTitles) private var titles
    @ObservedObject private var models: StudioModelStore
    @StateObject private var jobMonitor = StudioJobMonitor()

    /// The clips Music ▸ Train writes as the trainer's manifest, beside each run's adapter.
    @StudioStoredValue("Training.musicManifest") private var musicManifest = StudioMusicTrainingManifest()
    /// The run the dashboard follows: what the runner remembered last, or a history row picked here.
    @StudioStoredValue("requestID") private var requestID: UUID? = nil
    @StudioStoredValue("Training.compareA") private var compareA: UUID? = nil
    @StudioStoredValue("Training.compareB") private var compareB: UUID? = nil
    @State private var datasetSnapshot: StudioTrainingDatasetSnapshot?
    @State private var currentSnapshot: StudioTrainingSnapshot?
    @State private var statusMessage: String?
    @State private var error: String?
    @State private var selectedDatasetPreview: String?
    @State private var showAdvanced = false

    private let refreshTimer = Timer.publish(every: 1.5, on: .main, in: .common).autoconnect()
    private static let trainingTemplates: Set<CommandTemplateID> = [.imageTrainLoRA, .textTrainLoRA, .musicTrainAdapter]

    init(kind: StudioTrainingKind, models: StudioModelStore) {
        self.kind = kind
        _models = ObservedObject(wrappedValue: models)
    }

    private var task: StudioTask { kind.task }

    // MARK: Draft

    /// The task's draft, read and written through the session store so the root's Command view
    /// and Library ▸ "Use these settings" edit the value this column shows.
    private var draft: StudioTaskDraft {
        get { sessions?.taskDraft(for: task) ?? StudioTaskDraft(templateID: kind.templateID) }
        nonmutating set { sessions?.setTaskDraft(newValue, for: task) }
    }

    private var draftBinding: Binding<StudioTaskDraft> {
        Binding(get: { draft }, set: { draft = $0 })
    }

    private var baseline: StudioTaskDraft { StudioTrainingRun.baseline(for: kind.templateID) }
    private var slots: [StudioAttachmentSlot] { draft.slots }
    private var readiness: ModelReadinessState { controller.readiness(for: task) }
    private var dependencies: [String: (carries: Bool, dependsOn: String?)] { StudioTaskSchema.dependencies(for: draft) }

    /// Every option the contract lets the page edit, in contract order; the page's sections pick
    /// from it by flag and Advanced takes the rest.
    private var allFields: [StudioContractField<StudioTaskDraft>] {
        StudioTaskSchema.sections(for: task, draft: draft).flatMap(\.fields) + StudioTaskSchema.advanced(for: task, draft: draft)
    }

    private func fields(for flags: [String]) -> [StudioContractField<StudioTaskDraft>] {
        flags.compactMap { flag in allFields.first { $0.flag == flag } }
    }

    private var pageSections: [StudioTrainingSection] { StudioTrainingRun.sections(for: kind.templateID) }
    private var modelFields: [StudioContractField<StudioTaskDraft>] { fields(for: StudioTrainingRun.modelFlags(for: kind.templateID)) }

    private var advancedFields: [StudioContractField<StudioTaskDraft>] {
        let placed = Set(pageSections.flatMap(\.flags) + StudioTrainingRun.modelFlags(for: kind.templateID))
        return allFields.filter { !placed.contains($0.flag) && $0.overrideID != .model && $0.overrideID != .musicManifest }
    }

    private var isRunning: Bool {
        _ = jobMonitor.generation
        return runner?.currentJob(for: task) != nil
    }

    private var trainingRuns: [StudioLibraryItem] {
        library.items.filter { item in item.templateID.map(Self.trainingTemplates.contains) == true }
    }

    // MARK: Body

    var body: some View {
        HStack(spacing: 0) {
            settingsColumn
                .frame(minWidth: 340, idealWidth: 465, maxWidth: 465)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            dashboard
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onReceive(refreshTimer) { _ in refreshSnapshot() }
        .onReceive(controller.runCompletions) { result in
            guard Self.trainingTemplates.contains(result.templateID) else { return }
            refreshSnapshot()
            seedComparisons()
            refreshReadiness()
        }
        .onChange(of: draft.model) { _, _ in
            error = nil
            refreshReadiness()
        }
        .onChange(of: draft.text("--recipe")) { _, _ in
            // The recipe decides the seeded options from now on; typed values stay as overrides.
            draft = StudioTrainingRun.applyingRecipe(draft)
        }
        .onChange(of: draft.text("--dataset")) { _, _ in adoptExistingMusicManifest() }
        .onAppear {
            jobMonitor.attach(controller.jobs)
            adoptPageDefaults()
            // An imported page draft may carry a recipe beside the seeded options it decides.
            let recipeApplied = StudioTrainingRun.applyingRecipe(draft)
            if recipeApplied != draft { draft = recipeApplied }
            adoptExistingMusicManifest()
            refreshReadiness()
            seedComparisons()
            refreshSnapshot()
        }
        .task(id: musicManifest) { await saveDraftManifest() }
    }

    // MARK: - Settings column

    private var settingsColumn: some View {
        ScrollView {
            VStack(spacing: 0) {
                datasetSection
                modelSection
                ForEach(pageSections) { section in
                    contractSection(section.title, fields: fields(for: section.flags))
                }
                outputSection
                if !advancedFields.isEmpty {
                    advancedSection
                }
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { actionBar }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Training settings")
    }

    // MARK: Dataset

    @ViewBuilder
    private var datasetSection: some View {
        if kind == .music {
            VStack(alignment: .leading, spacing: 10) {
                StudioMusicManifestEditor(manifest: $musicManifest, message: $statusMessage)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .bottom) {
                Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
            }
        } else {
            StudioInspectorSectionView(
                title: "Dataset",
                canReset: slots.contains { $0.isFilled(in: draft) },
                onReset: clearAttachments
            ) {
                attachmentWell
                Text(kind == .image
                    ? "A folder of images with matching .txt captions; a checkpoint to resume from is optional."
                    : "Chat SFT examples as JSONL, one per line; evaluation prompts and a resume checkpoint are optional.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
                if let snapshot = datasetSnapshot {
                    datasetReport(snapshot)
                }
            }
        }
    }

    /// The well the composer draws for a task, with the same slots: the dataset first, then the
    /// optional inputs the contract declares.
    private var attachmentWell: some View {
        HStack(alignment: .center, spacing: 8) {
            ForEach(slots) { slot in
                StudioAttachmentSlotView(slot: slot, draft: draftBinding, onPick: { pick(slot) })
            }
            VStack(alignment: .leading, spacing: 1) {
                ForEach(slots) { slot in
                    Text(slot.caption(in: draft))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .padding(.leading, 4)
            Spacer(minLength: 8)
            Button("Inspect") { inspectDataset() }
                .buttonStyle(.mereSecondary)
                .disabled(draft.primaryInputPath.isBlank)
                .help("Count the records and preview them")
        }
    }

    @ViewBuilder
    private func datasetReport(_ snapshot: StudioTrainingDatasetSnapshot) -> some View {
        HStack(spacing: 8) {
            datasetMetric("Records", snapshot.totalRecords)
            datasetMetric("Usable", snapshot.usableRecords)
            datasetMetric("Issues", snapshot.totalRecords - snapshot.usableRecords)
        }
        ForEach(snapshot.diagnostics, id: \.self) { diagnostic in
            Label(
                diagnostic,
                systemImage: snapshot.usableRecords == snapshot.totalRecords
                    ? "checkmark.circle.fill"
                    : "exclamationmark.triangle.fill"
            )
            .font(MereRunTheme.captionFont)
            .foregroundStyle(
                snapshot.usableRecords == snapshot.totalRecords
                    ? MereRunTheme.green
                    : MereRunTheme.yellow
            )
        }
        if !snapshot.previews.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(snapshot.previews) { preview in
                        datasetPreviewCard(preview)
                    }
                }
            }
        }
    }

    // MARK: Model

    private var modelSection: some View {
        let modelChanged = draft.model != baseline.model
            || modelFields.contains { $0.changedCount(draft: draft, baseline: baseline) > 0 }
        return StudioInspectorSectionView(title: "Model", canReset: modelChanged, onReset: {
            var next = draft
            next.model = baseline.model
            for field in modelFields { field.reset(&next, to: baseline) }
            draft = next
        }) {
            modelPicker
            if readiness.blocksRun {
                readinessRow
            }
            if !modelFields.isEmpty {
                contractForm(modelFields)
            }
        }
    }

    private var modelPicker: some View {
        let scope = StudioTaskSchema.modelScope(for: draft)
        let bases = models.rows.filter { StudioTrainingRun.isTrainableBase($0.id, for: kind.templateID) }
        return StudioModelPicker(scope: scope, model: draftBinding.model, modelInventory: bases,
                                 onShowModels: { navigation.open(task: .modelsInstalled) }) {
            HStack(spacing: 8) {
                if let glyph = modelStatusGlyph {
                    Image(systemName: glyph)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(MereRunTheme.accent)
                }
                Text(scope.displayLabel(model: draft.model, titles: titles))
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, minHeight: 32)
            .background { StudioInspectorFieldChrome() }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        }
        .help(readiness.blocksRun ? readiness.message(titles: titles) : "Base model")
        .accessibilityLabel("Base model")
        .accessibilityValue(scope.resolvedModelID(model: draft.model))
    }

    private var modelStatusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .notChecked, .checking, .ready, .unknown: return nil
        }
    }

    /// Why the model cannot train yet, with the next step: get it, or check again.
    private var readinessRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(readiness.message(titles: titles))
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                if case .missingModel = readiness {
                    Button("Get the model") { pull(modelID: StudioTaskSchema.modelID(for: draft)) }
                        .buttonStyle(.mereSecondary)
                        .disabled(jobMonitor.pullJob(for: StudioTaskSchema.modelID(for: draft)) != nil)
                }
                Button("Check again", action: refreshReadiness)
                    .buttonStyle(.plain)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.accent)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Model readiness")
        .accessibilityValue(readiness.message(titles: titles))
    }

    // MARK: Contract sections

    private func contractSection(_ title: String, fields: [StudioContractField<StudioTaskDraft>]) -> some View {
        StudioInspectorSectionView(
            title: title,
            canReset: fields.contains { $0.changedCount(draft: draft, baseline: baseline) > 0 },
            onReset: {
                var next = draft
                for field in fields { field.reset(&next, to: baseline) }
                draft = next
            }
        ) {
            contractForm(fields)
            if kind == .text, title == "Training" {
                Text("Leave target modules empty for the native family recipe. LFM2.5 v1 is attention-only; Inkling also includes MLP, expert, and unembedding targets.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The contract form's rows, with two differences from the inspector's: a number the
    /// contract gives no range (a size, a step count, a learning rate) is typed, as the page
    /// always had it, so it can be left empty for the CLI's default rather than nudged by a
    /// stepper that cannot reach 0.0003 or be unset; and a free-text option keeps its label
    /// beside the field. While a recipe is chosen, the rows it decides say so.
    private func contractForm(_ fields: [StudioContractField<StudioTaskDraft>]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(fields.filter { StudioContractSchema.isVisible($0, in: draft, dependencies: dependencies) }) { field in
                if let override = field.overrideID {
                    overrideControl(override)
                } else if [.number, .integer].contains(field.kind), field.option.range == nil {
                    typedRow(field, monospaced: true, width: 120)
                } else if field.control == .field {
                    typedRow(field, monospaced: false, width: 220)
                } else {
                    ContractFormControl(
                        field: field, draft: draftBinding,
                        noneTitle: StudioTrainingRun.isRecipeGoverned(field.flag, in: draft) ? "From recipe" : nil
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A labelled text field over one option: empty leaves the flag out, so the placeholder says
    /// what runs then — the contract's default, the recipe's value, or the CLI's own.
    private func typedRow(_ field: StudioContractField<StudioTaskDraft>, monospaced: Bool, width: CGFloat) -> some View {
        let text = Binding<String>(
            get: {
                let value = field.value(in: draft)
                return value.text ?? value.numericValue.map { StudioComposerPresets.argumentText($0) } ?? ""
            },
            set: { typed in
                var next = draft
                field.write(typed.isEmpty ? .unset : .text(typed), to: &next)
                draft = next
            }
        )
        let placeholder = StudioTrainingRun.isRecipeGoverned(field.flag, in: draft)
            ? "From recipe"
            : (field.option.defaultValue ?? "Default")
        return StudioInspectorLabeledRow(field.label) {
            StudioInspectorTextField(placeholder: placeholder, text: text, isMonospaced: monospaced)
                .frame(width: width)
        }
        .help(field.flag)
    }

    /// The editors the page draws itself: Klein's per-target ranks. The model picker and the
    /// clip list have sections of their own, so their rows never reach a form here.
    @ViewBuilder
    private func overrideControl(_ override: StudioContractOverrideID) -> some View {
        switch override {
        case .targetRanks:
            StudioTargetRankEditor(value: Binding(
                get: { draft.text("--lora-target-ranks") },
                set: { text in
                    var next = draft
                    next.form["--lora-target-ranks"] = text.isEmpty ? .unset : .text(text)
                    draft = next
                }
            ))
        default:
            EmptyView()
        }
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Button {
                    withAnimation(MereRunTheme.Motion.quick) { showAdvanced.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: showAdvanced ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text("Advanced · \(advancedFields.count) more")
                            .font(.callout.weight(.medium))
                            .foregroundStyle(MereRunTheme.textSecondary)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(showAdvanced ? "Hide advanced settings" : "Show \(advancedFields.count) advanced settings")
                Spacer(minLength: 0)
                if showAdvanced, advancedFields.contains(where: { $0.changedCount(draft: draft, baseline: baseline) > 0 }) {
                    Button("Reset") {
                        var next = draft
                        for field in advancedFields { field.reset(&next, to: baseline) }
                        draft = next
                    }
                    .buttonStyle(.plain)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .help("Back to the defaults")
                }
            }
            if showAdvanced {
                contractForm(advancedFields)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Output

    /// Where the adapter lands, without a path field: routing names it inside the domain's folder
    /// when the run starts, after the dataset (or the clip list, for music).
    private var outputSection: some View {
        let folder = URL(fileURLWithPath: StudioOutputLocation.destination(for: draft).text("--output")).deletingLastPathComponent()
        return StudioInspectorSectionView(title: "Output", canReset: false, onReset: {}) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Saves to \(StudioOutputLocation.abbreviate(folder))")
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                    Text(kind == .music
                        ? "The adapter is named when the run starts; the clip list is written beside it."
                        : "The adapter is named after the dataset when the run starts; events, samples, and checkpoints land beside it.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 4)
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([folder]) }
                    .buttonStyle(.mereSecondary)
                    .help("Show the folder in Finder")
            }
            SettingsLink {
                Text("Change in Settings…")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.accent)
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: Actions

    private var actionBar: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let error {
                MereBanner(severity: .error, text: error, onDismiss: { self.error = nil })
            }
            HStack(spacing: 10) {
                Button {
                    preflight()
                } label: {
                    Label(kind == .music ? "Validate" : "Preflight", systemImage: "checkmark.shield")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
                .help(kind == .music ? "Check the clips" : "Check the request without training")
                if isRunning {
                    Button {
                        runner?.stop(task: task)
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.mereSecondary)
                    .help("Stop the training run (⌘.)")
                    .keyboardShortcut(".", modifiers: .command)
                    .accessibilityLabel("Stop training")
                } else {
                    Button {
                        startTraining()
                    } label: {
                        Label(resumes ? "Resume training" : "Start training", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.merePrimary)
                    .keyboardShortcut(.return, modifiers: .command)
                    .help(readiness.blocksRun ? readiness.message(titles: titles) : "Start training (⌘↩)")
                }
            }
            if let statusMessage {
                Text(statusMessage)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(MereRunTheme.background)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }

    private var resumes: Bool {
        !draft.text("--resume-from").isBlank
    }

    // MARK: - Dashboard

    private var dashboard: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                currentRunDashboard
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                comparisonDashboard
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                recentRuns
            }
            .padding(18)
        }
    }

    private var currentRunDashboard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Live training")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                if let requestID,
                   let progress = controller.progressByRequestID[requestID] {
                    Text(progress.label)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            if let snapshot = currentSnapshot {
                trainingMetrics(snapshot)
                StudioTrainingLossChart(points: snapshot.lossPoints)
                    .frame(height: 220)
                    .merePanel()
                if !snapshot.samples.isEmpty {
                    Text("Sample progression")
                        .font(MereRunTheme.sectionFont)
                    sampleGallery(snapshot.samples)
                }
                if !snapshot.checkpoints.isEmpty {
                    Text("Checkpoints")
                        .font(MereRunTheme.sectionFont)
                    artifactStrip(snapshot.checkpoints)
                }
            } else if let requestID {
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: [.image, .text, .audio]
                )
                .frame(minHeight: 300)
            } else {
                ContentUnavailableView(
                    "No training run selected",
                    systemImage: "chart.xyaxis.line",
                    description: Text(kind == .music
                        ? "Add the clips, validate them, then start training."
                        : "Attach a dataset, preflight the request, then start training.")
                )
                .frame(minHeight: 280)
            }
        }
    }

    private var comparisonDashboard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Compare runs")
                .font(MereRunTheme.sectionFont)
            HStack {
                runPicker("A", selection: $compareA)
                runPicker("B", selection: $compareB)
            }
            let first = compareA.flatMap(snapshotForRun)
            let second = compareB.flatMap(snapshotForRun)
            if first != nil || second != nil {
                HStack(spacing: 10) {
                    comparisonCard("A", snapshot: first)
                    comparisonCard("B", snapshot: second)
                }
            } else {
                Text("Completed and in-progress training runs will appear here.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }

    private var recentRuns: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Training history")
                .font(MereRunTheme.sectionFont)
            ForEach(trainingRuns.prefix(8)) { item in
                Button {
                    requestID = item.id
                    currentSnapshot = snapshotForRun(item.id)
                } label: {
                    HStack {
                        Image(systemName: item.status == .completed ? "checkmark.circle.fill" : "circle.dotted")
                            .foregroundStyle(item.status == .failed ? MereRunTheme.red : MereRunTheme.accent)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.displayTitle)
                                .font(.system(size: 12.5, weight: .semibold))
                            Text(item.createdAt.formatted(date: .abbreviated, time: .shortened))
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                        Spacer()
                        Text(item.status.rawValue.capitalized)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                    .padding(9)
                    .merePanel()
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(item.displayTitle), \(item.status.rawValue)")
            }
        }
    }

    private func trainingMetrics(_ snapshot: StudioTrainingSnapshot) -> some View {
        let latest = snapshot.latest
        return HStack(spacing: 8) {
            dashboardMetric(
                "Step",
                latest?.totalSteps.map { "\(latest?.step ?? 0) / \($0)" } ?? "\(latest?.step ?? 0)"
            )
            dashboardMetric(
                "Loss",
                (latest?.loss ?? snapshot.lossPoints.last?.loss).map { String(format: "%.6f", $0) } ?? "—"
            )
            dashboardMetric(
                "Progress",
                snapshot.progress.map { "\(Int(($0 * 100).rounded()))%" } ?? "—"
            )
            dashboardMetric("Samples", "\(snapshot.samples.count)")
            dashboardMetric("Checkpoints", "\(snapshot.checkpoints.count)")
        }
    }

    private func comparisonCard(_ label: String, snapshot: StudioTrainingSnapshot?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(MereRunTheme.accent)
            if let snapshot {
                trainingMetricsCompact(snapshot)
                StudioTrainingLossChart(points: snapshot.lossPoints)
                    .frame(height: 150)
                if let sample = snapshot.samples.last {
                    StudioAsyncImagePreview(
                        url: sample,
                        maxPixelSize: 900,
                        contentMode: .fit,
                        fallbackSystemImage: "photo"
                    )
                    .frame(height: 160)
                }
            } else {
                ContentUnavailableView("Choose a run", systemImage: "chart.xyaxis.line")
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 240, alignment: .topLeading)
        .merePanel()
    }

    private func trainingMetricsCompact(_ snapshot: StudioTrainingSnapshot) -> some View {
        HStack {
            Text("step \(snapshot.latest?.step ?? 0)")
            Spacer()
            if let loss = snapshot.latest?.loss {
                Text("loss \(loss, specifier: "%.5f")")
            }
        }
        .font(MereRunTheme.captionFont)
        .foregroundStyle(MereRunTheme.textMuted)
    }

    private func sampleGallery(_ urls: [URL]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(urls.suffix(12), id: \.self) { url in
                    Button {
                        QuickLookCoordinator.shared.preview(url)
                    } label: {
                        StudioAsyncImagePreview(
                            url: url,
                            maxPixelSize: 480,
                            contentMode: .fill,
                            fallbackSystemImage: "photo"
                        )
                        .frame(width: 130, height: 130)
                        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
                    }
                    .buttonStyle(.plain)
                    .help(url.lastPathComponent)
                    .accessibilityLabel("Sample \(url.lastPathComponent)")
                }
            }
        }
    }

    private func artifactStrip(_ urls: [URL]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 7) {
                ForEach(urls, id: \.self) { url in
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    } label: {
                        Label(url.lastPathComponent, systemImage: "shippingbox")
                    }
                    .buttonStyle(.mereSecondary)
                    .help(url.path)
                }
            }
        }
    }

    private func runPicker(_ label: String, selection: Binding<UUID?>) -> some View {
        Picker(label, selection: selection) {
            Text("None").tag(UUID?.none)
            ForEach(trainingRuns) { item in
                Text(item.displayTitle).tag(Optional(item.id))
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func dashboardMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .monospacedDigit()
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
        .accessibilityElement(children: .combine)
    }

    private func datasetMetric(_ label: String, _ value: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text("\(value)")
                .font(.system(size: 15, weight: .bold, design: .rounded))
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
        .accessibilityElement(children: .combine)
    }

    private func datasetPreviewCard(_ preview: StudioTrainingDatasetPreview) -> some View {
        Button {
            selectedDatasetPreview = preview.id
            if let audioURL = preview.audioURL {
                QuickLookCoordinator.shared.preview(audioURL)
            } else if let imageURL = preview.imageURL {
                QuickLookCoordinator.shared.preview(imageURL)
            }
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                if let imageURL = preview.imageURL {
                    StudioAsyncImagePreview(
                        url: imageURL,
                        maxPixelSize: 360,
                        contentMode: .fill,
                        fallbackSystemImage: "photo"
                    )
                    .frame(width: 115, height: 84)
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
                } else {
                    Image(systemName: preview.audioURL == nil ? "curlybraces" : "waveform")
                        .font(.system(size: 24, weight: .semibold))
                        .foregroundStyle(MereRunTheme.accent)
                        .frame(width: 115, height: 50)
                }
                Text(preview.title)
                    .font(.system(size: 10.5, weight: .semibold))
                    .lineLimit(1)
                Text(preview.detail)
                    .font(.system(size: 9.5))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(3)
            }
            .padding(7)
            .frame(width: 130, height: preview.imageURL == nil ? 120 : 155, alignment: .topLeading)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(selectedDatasetPreview == preview.id ? MereRunTheme.accentSoft : MereRunTheme.surface)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(preview.title)
    }

    // MARK: - Draft upkeep

    /// A draft nothing has touched yet takes the page's own defaults once (`StudioTrainingRun`).
    private func adoptPageDefaults() {
        guard let sessions, !sessions.contains(StudioTaskSessions.taskDraftKey(task)),
              draft == StudioTaskDraft(templateID: kind.templateID) else { return }
        draft = StudioTrainingRun.applyingPageDefaults(draft)
    }

    /// A manifest the draft names that this page did not write — one chosen in the Command view,
    /// or a run's `<adapter>.dataset.jsonl` restored with Library ▸ "Use these settings" — becomes
    /// the clip list, so the clips, the Command view, and Start never disagree. The page then
    /// saves its own copy back into `--dataset`. A path that no longer exists is forgotten
    /// quietly; one that will not read is reported once, then forgotten.
    private func adoptExistingMusicManifest() {
        guard kind == .music else { return }
        let path = draft.text("--dataset")
        guard !path.isBlank else { return }
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL
        guard !StudioMusicTrainingManifest.isDraftURL(url) else { return }
        var next = draft
        next.form["--dataset"] = .unset
        draft = next
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            musicManifest = try StudioMusicTrainingManifest.importing(Data(contentsOf: url), from: url)
            statusMessage = nil
        } catch {
            statusMessage = "The manifest at \(url.lastPathComponent) could not be read into the clip list: \(error.localizedDescription)"
        }
    }

    /// Keeps the Command view's manifest current, a moment after editing stops: a ready clip list
    /// is saved and its path becomes the draft's `--dataset`, so the Command view's Run has a real
    /// file to pass; a list with problems leaves the flag empty, so it never runs a broken one.
    private func saveDraftManifest() async {
        guard kind == .music else { return }
        guard musicManifest.problems().isEmpty else {
            setDraftManifestPath(nil)
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        do {
            let url = try StudioMusicTrainingManifest.storeDraft(content: musicManifest.jsonl())
            // Rows the Library still names (queued Command-view runs included) keep their files.
            let referenced = Set(library.items.compactMap { $0.commandDraft?.inputPath })
            StudioMusicTrainingManifest.pruneDrafts(current: url, referenced: referenced)
            setDraftManifestPath(url.path)
        } catch {
            setDraftManifestPath(nil)
        }
    }

    /// Writes the saved clip list's path into `--dataset`, leaving a manifest the user chose alone.
    private func setDraftManifestPath(_ path: String?) {
        let current = draft.text("--dataset")
        let isOurs = current.isBlank || StudioMusicTrainingManifest.isDraftURL(URL(fileURLWithPath: current))
        guard isOurs, current != (path ?? "") else { return }
        var next = draft
        next.form["--dataset"] = path.map { .text($0) } ?? .unset
        draft = next
    }

    private func pick(_ slot: StudioAttachmentSlot) {
        var next = draft
        StudioAttachmentPicker.pick(for: slot, into: &next)
        draft = next
        datasetSnapshot = nil
        error = nil
    }

    private func clearAttachments() {
        var next = draft
        for slot in slots { slot.clear(in: &next) }
        draft = next
        datasetSnapshot = nil
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: task, modelID: StudioTaskSchema.modelID(for: draft))
    }

    /// Gets a managed model through the same `model pull` job the readiness row reports.
    private func pull(modelID: String) {
        guard !modelID.isBlank, let template = CommandCatalog.template(id: .modelPull) else { return }
        var commandDraft = template.defaultDraft()
        commandDraft.model = modelID
        let request = StudioRunRequest(mode: template.libraryMode, templateID: .modelPull, template: template, draft: commandDraft)
        if !models.startPull(request) {
            error = controller.status
            refreshReadiness()
        }
    }

    // MARK: - Inspect, preflight, start

    private func inspectDataset() {
        if kind == .music {
            datasetSnapshot = StudioTrainingDatasetSnapshot.inspect(manifest: musicManifest)
            statusMessage = nil
            return
        }
        let path = draft.primaryInputPath
        guard !path.isBlank else {
            statusMessage = "Attach a dataset first."
            return
        }
        datasetSnapshot = StudioTrainingDatasetSnapshot.inspect(kind: kind, path: path)
        selectedDatasetPreview = datasetSnapshot?.previews.first?.id
        statusMessage = "Dataset inspection complete."
    }

    private func preflight() {
        error = nil
        inspectDataset()
        guard validateDraft() else { return }
        guard let checked = StudioTrainingRun.preflightDraft(StudioTrainingRun.launchDraft(draft)) else {
            statusMessage = "The clips are ready. ACE-Step checks its model files when training starts."
            return
        }
        guard let runner else { return }
        do {
            _ = try runner.run(checked, task: task)
            statusMessage = "Preflight submitted."
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func startTraining() {
        error = nil
        inspectDataset()
        guard validateDraft(), let runner else { return }
        do {
            let request: StudioRunRequest
            if kind == .music {
                if readiness.blocksRun { throw StudioValidationError(message: readiness.message(titles: titles)) }
                let launch = try StudioTrainingRun.musicLaunch(draft, manifest: musicManifest)
                guard let built = launch.request() else { throw StudioValidationError(message: "This command can't run from Studio.") }
                request = try runner.run(request: built, task: task)
            } else {
                request = try runner.run(StudioTrainingRun.launchDraft(draft), task: task)
            }
            currentSnapshot = StudioTrainingSnapshot.load(outputPath: request.draft.outputPath)
            statusMessage = resumes ? "Checkpoint resume submitted." : "Training submitted."
            if kind == .image, resumes {
                // The next run starts fresh unless another checkpoint is chosen.
                var next = draft
                next.form["--resume-from"] = .unset
                draft = next
            }
            seedComparisons()
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// The page's own checks before the runner's: the dataset inspected and whole, the target
    /// ranks well formed, the schedule positive. The runner then validates the command itself.
    private func validateDraft() -> Bool {
        if kind == .music {
            if let problem = musicManifest.problems().first {
                statusMessage = problem
                return false
            }
        } else {
            guard let snapshot = datasetSnapshot,
                  snapshot.totalRecords > 0,
                  snapshot.usableRecords == snapshot.totalRecords else {
                statusMessage = "Inspect and repair the dataset before training."
                return false
            }
        }
        if kind == .image {
            let resume = draft.text("--resume-from")
            if !resume.isBlank, !FileManager.default.fileExists(atPath: resume) {
                statusMessage = "The selected resume checkpoint does not exist."
                return false
            }
            if let problem = StudioTargetRank.problems(StudioTargetRank.decode(draft.text("--lora-target-ranks"))).first {
                statusMessage = problem
                return false
            }
        }
        let steps = draft.text(kind == .music ? "--steps" : "--training-steps")
        let rank = draft.text("--rank")
        let learningRate = draft.text("--learning-rate")
        guard Int(steps) ?? 0 > 0, Int(rank) ?? 0 > 0, Double(learningRate) ?? 0 > 0 else {
            statusMessage = "Steps, rank, and learning rate must be positive."
            return false
        }
        return true
    }

    // MARK: - Snapshots and comparison

    private func refreshSnapshot() {
        if let requestID,
           let item = library.items.first(where: { $0.id == requestID }),
           let output = item.commandDraft?.outputPath {
            currentSnapshot = StudioTrainingSnapshot.load(outputPath: output)
        }
    }

    private func snapshotForRun(_ id: UUID) -> StudioTrainingSnapshot? {
        guard let item = library.items.first(where: { $0.id == id }),
              let output = item.commandDraft?.outputPath else { return nil }
        return StudioTrainingSnapshot.load(outputPath: output)
    }

    private func seedComparisons() {
        if compareA == nil { compareA = trainingRuns.first?.id }
        if compareB == nil { compareB = trainingRuns.dropFirst().first?.id }
    }
}
