import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Text ▸ Classify edits GLiNER's text and label tasks directly.
struct StudioGLiNERClassificationView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @StudioStoredValue("GLiNER.document") private var document = StudioClassificationDocument()
    @StudioStoredValue("GLiNER.long") private var processLongText = false
    @StudioStoredValue("requestID") private var requestID: UUID? = nil
    @State private var draftRequestPath = ""
    @State private var message: String?
    @State private var confirmsExample = false

    private let model = "text-classify-gliner25-decide"

    private var draft: CommandDraft {
        var value = CommandDraft()
        value.inputPath = draftRequestPath
        value.model = model
        value.force = true
        value.extraArguments = processLongText ? "--long" : ""
        return value
    }

    private var isRunning: Bool {
        guard let requestID, let item = library.items.first(where: { $0.id == requestID }) else { return false }
        return item.status == .queued || item.status == .running
    }

    var body: some View {
        GeometryReader { geometry in
            if geometry.size.width >= 780 {
                HSplitView {
                    editor.frame(minWidth: 300, idealWidth: 360, maxWidth: 450)
                    result.frame(minWidth: 350, maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                VSplitView {
                    editor.frame(minHeight: 190, idealHeight: geometry.size.height * 0.45)
                    result.frame(maxWidth: .infinity, minHeight: 220, maxHeight: .infinity)
                }
            }
        }
        .background(MereRunTheme.background)
        .studioTaskCommand(.textClassify, draft: draft)
        .task(id: document) { await saveDraftRequest() }
        .confirmationDialog("Replace this request with the example?", isPresented: $confirmsExample) {
            Button("Load the Example", role: .destructive) { document = .example }
        } message: {
            Text("The text and label tasks you have now are replaced.")
        }
    }

    private var editor: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                VStack(alignment: .leading, spacing: 6) {
                    fieldLabel("Text to classify")
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $document.text)
                            .font(MereRunTheme.bodyFont)
                            .scrollContentBackground(.hidden)
                            .padding(4)
                        if document.text.isEmpty {
                            Text("Paste a message, ticket, review, or other text.")
                                .font(MereRunTheme.bodyFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .padding(.horizontal, 9)
                                .padding(.vertical, 4)
                                .allowsHitTesting(false)
                        }
                    }
                    .frame(minHeight: 96)
                    .merePanel()
                    .accessibilityLabel("Text to classify")
                }
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        fieldLabel("Label tasks")
                        Spacer()
                        Button {
                            document.tasks.append(.init())
                        } label: {
                            Label("Add task", systemImage: "plus")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(MereRunTheme.accent)
                        .disabled(document.tasks.count >= 16)
                    }
                    if document.tasks.isEmpty {
                        Text("Add a task, name it, then provide the labels the model should compare.")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                    ForEach($document.tasks) { $task in
                        StudioClassificationTaskEditor(
                            task: $task,
                            number: (document.tasks.firstIndex { $0.id == task.id } ?? 0) + 1,
                            canMoveUp: document.tasks.first?.id != task.id,
                            canMoveDown: document.tasks.last?.id != task.id,
                            onMove: { move(task.id, by: $0) },
                            onDelete: { document.tasks.removeAll { $0.id == task.id } }
                        )
                    }
                }
                actions
            }
            .padding(18)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Classify")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Menu {
                    Button("Load the example", action: loadExample)
                    Divider()
                    Button("Import Request…", action: importRequest)
                    Button("Export Request…", action: exportRequest)
                        .disabled(!document.problems.isEmpty)
                    Divider()
                    Button("Clear", role: .destructive) { document = StudioClassificationDocument() }
                        .disabled(document == StudioClassificationDocument())
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Example, import, and export")
            }
            Text("GLiNER2.5 Decide scores labels you supply for each task. Select one label or multiple labels above a threshold.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Model · GLiNER2.5 Decide")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Process long text in overlapping chunks", isOn: $processLongText)
                .font(MereRunTheme.captionFont)
            if !document.problems.isEmpty, !(document.text.isEmpty && document.tasks.isEmpty) {
                ForEach(document.problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.circle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
            }
            HStack(spacing: 8) {
                Button("Check fit") { run(preflight: true) }
                    .buttonStyle(.mereSecondary)
                    .help("Count input tokens without loading the model")
                Spacer()
                Button("Classify") { run(preflight: false) }
                    .buttonStyle(.merePrimary)
            }
            .disabled(!document.problems.isEmpty || isRunning)
            if let message {
                Text(message)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var result: some View {
        StudioClassificationResultPane(requestID: requestID, onLoadExample: loadExample)
            .padding(18)
    }

    private func fieldLabel(_ title: String) -> some View {
        Text(title)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
    }

    private func move(_ id: UUID, by offset: Int) {
        guard let index = document.tasks.firstIndex(where: { $0.id == id }) else { return }
        let target = index + offset
        guard document.tasks.indices.contains(target) else { return }
        document.tasks.swapAt(index, target)
    }

    private func run(preflight: Bool) {
        message = nil
        let directory = StudioOutputLocation.outputDirectoryURL(
            domain: .text,
            prompt: document.tasks.first?.name ?? document.text,
            fallbackStem: "classifications"
        )
        let requestURL = directory.appendingPathComponent("request.json")
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try document.requestJSON().write(to: requestURL, options: .atomic)
        } catch {
            message = "Studio could not write the request: \(error.localizedDescription)"
            return
        }
        var command = CommandDraft()
        command.inputPath = requestURL.path
        command.model = model
        command.outputPath = directory.appendingPathComponent(preflight ? "fit.json" : "classifications.json").path
        command.preflight = preflight
        command.force = true
        command.extraArguments = processLongText ? "--long" : ""
        guard let template = CommandCatalog.template(id: .textClassify) else { return }
        let request = StudioRunRequest(mode: .chat, templateID: template.id, template: template, draft: command)
        requestID = (try? StudioTaskRunner(controller: controller, library: library)
            .run(request: request, task: .textClassify, validating: false))?.id
    }

    private func saveDraftRequest() async {
        guard document.problems.isEmpty else {
            draftRequestPath = ""
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        let url = StudioClassificationDocument.draftRequestURL()
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try document.requestJSON().write(to: url, options: .atomic)
            draftRequestPath = url.path
        } catch {
            draftRequestPath = ""
        }
    }

    private func loadExample() {
        if document == StudioClassificationDocument() || document == .example {
            document = .example
        } else {
            confirmsExample = true
        }
    }

    private func importRequest() {
        guard let url = StudioFilePanels.chooseFile(title: "Import a classification request",
                                                     allowedContentTypes: [.json]).first else { return }
        do {
            document = try StudioClassificationDocument.importing(Data(contentsOf: url))
            message = nil
        } catch {
            message = "That file is not a classification request: \(error.localizedDescription)"
        }
    }

    private func exportRequest() {
        guard let url = StudioFilePanels.saveFile(title: "Export the classification request",
                                                  suggestedName: "request.json", allowedContentTypes: [.json]) else { return }
        do {
            try document.requestJSON().write(to: url, options: .atomic)
        } catch {
            message = "Studio could not save the request: \(error.localizedDescription)"
        }
    }
}

private struct StudioClassificationTaskEditor: View {
    @Binding var task: StudioClassificationDocument.Task
    let number: Int
    let canMoveUp: Bool
    let canMoveDown: Bool
    let onMove: (Int) -> Void
    let onDelete: () -> Void
    @State private var showsPrompt = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("\(number)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(MereRunTheme.textMuted)
                TextField("Task name, e.g. department", text: $task.name)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Task name")
                Menu {
                    Button("Move Up") { onMove(-1) }.disabled(!canMoveUp)
                    Button("Move Down") { onMove(1) }.disabled(!canMoveDown)
                    Button(showsPrompt ? "Hide prompt" : "Edit prompt") { showsPrompt.toggle() }
                    Divider()
                    Button("Delete Task", role: .destructive, action: onDelete)
                } label: {
                    Image(systemName: "ellipsis")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .accessibilityLabel("Task \(number) options")
            }
            if showsPrompt || !task.prompt.isEmpty {
                TextField("Optional prompt", text: $task.prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...3)
                    .accessibilityLabel("Task prompt")
            }
            HStack {
                Text("Labels")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Button("Add label") { task.labels.append(.init(name: "")) }
                    .buttonStyle(.plain)
                    .foregroundStyle(MereRunTheme.accent)
                    .disabled(task.labels.count >= 64)
            }
            ForEach($task.labels) { $label in
                HStack(alignment: .top, spacing: 6) {
                    VStack(spacing: 4) {
                        TextField("Label", text: $label.name)
                            .textFieldStyle(.roundedBorder)
                        TextField("Optional description", text: $label.detail)
                            .textFieldStyle(.roundedBorder)
                    }
                    Button {
                        task.labels.removeAll { $0.id == label.id }
                    } label: {
                        Image(systemName: "minus.circle")
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Remove label")
                }
            }
            Toggle("Select multiple labels", isOn: $task.multiLabel)
                .font(MereRunTheme.captionFont)
            if task.multiLabel {
                HStack {
                    Text("Threshold")
                    Slider(value: $task.threshold, in: 0...1, step: 0.01)
                    Text(task.threshold.formatted(.percent.precision(.fractionLength(0))))
                        .monospacedDigit()
                }
                .font(MereRunTheme.captionFont)
            }
        }
        .padding(12)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
        }
    }
}

private struct StudioClassificationResultPane: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    let requestID: UUID?
    let onLoadExample: () -> Void

    private var item: StudioLibraryItem? {
        requestID.flatMap { id in library.items.first { $0.id == id } }
    }

    var body: some View {
        if let item {
            switch item.status {
            case .queued, .running:
                VStack(spacing: 10) {
                    ProgressView()
                    Text(item.commandDraft?.preflight == true ? "Checking the fit…" : "Classifying…")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    if let progress = controller.progressByRequestID[item.id] {
                        Text(progress.label)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .completed:
                StudioClassificationAnswers(item: item)
            case .failed, .interrupted:
                StudioRunFailureDetail(models: controller.modelStore, item: item)
            case .cancelled:
                ContentUnavailableView("Cancelled", systemImage: "stop.circle")
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "tag")
                    .font(.system(size: 30, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Classifications appear here")
                    .font(MereRunTheme.sectionFont)
                Text("Add text and a label task, then Classify.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                Button("Load the example", action: onLoadExample)
                    .buttonStyle(.mereSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct StudioClassificationAnswers: View {
    let item: StudioLibraryItem
    @State private var loaded: (output: StudioClassificationOutput?, request: StudioClassificationDocument?)?

    private var outputURL: URL? {
        item.commandDraft.flatMap { $0.outputPath.isBlank ? nil : URL(fileURLWithPath: $0.outputPath) }
    }

    var body: some View {
        Group {
            if let loaded {
                if let output = loaded.output {
                    content(output: output, request: loaded.request)
                } else {
                    ContentUnavailableView("No result to show", systemImage: "doc.questionmark",
                                           description: Text("The run finished without a classification document."))
                }
            } else {
                Color.clear
            }
        }
        .task(id: item.id) { loaded = load() }
    }

    private func load() -> (StudioClassificationOutput?, StudioClassificationDocument?) {
        let output = outputURL.flatMap { try? Data(contentsOf: $0) }.flatMap(StudioClassificationOutput.init(data:))
            ?? item.outputText.flatMap(StudioClassificationOutput.init(outputText:))
        let request = item.commandDraft.flatMap { draft in
            (try? Data(contentsOf: URL(fileURLWithPath: draft.inputPath)))
                .flatMap { try? StudioClassificationDocument.importing($0) }
        }
        return (output, request)
    }

    private func content(output: StudioClassificationOutput, request: StudioClassificationDocument?) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(item.commandDraft?.preflight == true ? "Fit" : "Classifications")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    if let outputURL {
                        Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([outputURL]) }
                            .buttonStyle(.mereSecondary)
                    }
                }
                switch output {
                case .fit(let plan):
                    Text("\(plan.inputTokens) input tokens · \(plan.labelCount) labels · \(plan.taskNames.count) tasks")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    ForEach(plan.taskNames, id: \.self) { name in
                        Label(name, systemImage: "checkmark.circle")
                            .font(MereRunTheme.bodyFont)
                    }
                case .fits(let plans):
                    Text("\(plans.count) chunks · \(plans.reduce(0) { $0 + $1.inputTokens }) input tokens")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    ForEach(plans.indices, id: \.self) { index in
                        Text("Chunk \(index + 1): \(plans[index].inputTokens) tokens")
                            .font(MereRunTheme.bodyFont)
                    }
                case .result(let result):
                    Text("\(result.inputTokens) input tokens · \(result.runtime)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    let names = request?.tasks.map(\.name) ?? result.heads.keys.sorted()
                    ForEach(names, id: \.self) { name in
                        if let head = result.heads[name] {
                            let task = request?.tasks.first { $0.name == name }
                            StudioClassificationHeadCard(name: name, head: head, task: task)
                        }
                    }
                }
            }
        }
    }
}

private struct StudioClassificationHeadCard: View {
    let name: String
    let head: StudioClassificationResult.Head
    let task: StudioClassificationDocument.Task?

    private var labels: [String] {
        task?.labels.map(\.name) ?? head.probabilities.keys.sorted()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(name)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
            Text(head.labels.joined(separator: ", "))
                .font(.system(size: 17, weight: .semibold))
            ForEach(labels, id: \.self) { label in
                let probability = head.probabilities[label] ?? 0
                HStack(spacing: 8) {
                    Text(label)
                        .font(.caption.weight(head.labels.contains(label) ? .semibold : .regular))
                        .frame(width: 120, alignment: .leading)
                        .lineLimit(1)
                    GeometryReader { geometry in
                        Capsule()
                            .fill(head.labels.contains(label) ? MereRunTheme.accent : MereRunTheme.textMuted.opacity(0.45))
                            .frame(width: max(3, geometry.size.width * min(max(probability, 0), 1)))
                    }
                    .frame(height: 6)
                    Text(probability.formatted(.percent.precision(.fractionLength(1))))
                        .font(MereRunTheme.captionFont)
                        .monospacedDigit()
                }
                .help(task?.labels.first { $0.name == label }?.detail ?? "")
            }
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
        }
        .accessibilityElement(children: .combine)
    }
}
