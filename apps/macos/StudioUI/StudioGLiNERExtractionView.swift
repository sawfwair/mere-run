import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Text > Extract edits the schema and displays native GLiNER spans and relations.
struct StudioGLiNERExtractionView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @StudioStoredValue("GLiNER.extraction.document") private var document = StudioExtractionDocument()
    @StudioStoredValue("GLiNER.extraction.long") private var processLongText = false
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
                    editor.frame(minWidth: 300, idealWidth: 370, maxWidth: 480)
                    result.frame(minWidth: 340, maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                VSplitView {
                    editor.frame(minHeight: 190, idealHeight: geometry.size.height * 0.45)
                    result.frame(maxWidth: .infinity, minHeight: 220, maxHeight: .infinity)
                }
            }
        }
        .background(MereRunTheme.background)
        .studioTaskCommand(.textExtract, draft: draft)
        .task(id: document) { await saveDraftRequest() }
        .confirmationDialog("Replace this request with the example?", isPresented: $confirmsExample) {
            Button("Load the Example", role: .destructive) { document = .example }
        } message: {
            Text("The text and extraction schema you have now are replaced.")
        }
    }

    private var editor: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                Text("Text to extract from")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                TextEditor(text: $document.text)
                    .font(MereRunTheme.bodyFont)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 110)
                    .merePanel()
                    .accessibilityLabel("Text to extract from")
                section(title: "Entities", addLabel: "Add entity", add: { document.entities.append(.init()) }) {
                    ForEach($document.entities) { $term in
                        termEditor(name: $term.name, detail: $term.detail, nameHint: "Entity type") {
                            document.entities.removeAll { $0.id == term.id }
                        }
                    }
                }
                section(title: "Relations", addLabel: "Add relation", add: { document.relations.append(.init()) }) {
                    ForEach($document.relations) { $term in
                        termEditor(name: $term.name, detail: $term.detail, nameHint: "Relation name") {
                            document.relations.removeAll { $0.id == term.id }
                        }
                    }
                }
                section(title: "Structures", addLabel: "Add structure", add: { document.structures.append(.init()) }) {
                    ForEach($document.structures) { $structure in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                TextField("Structure name", text: $structure.name)
                                    .textFieldStyle(.roundedBorder)
                                Button {
                                    document.structures.removeAll { $0.id == structure.id }
                                } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel("Remove structure")
                            }
                            ForEach($structure.fields) { $field in
                                VStack(alignment: .leading, spacing: 5) {
                                    termEditor(name: $field.name, detail: $field.detail, nameHint: "Field") {
                                        structure.fields.removeAll { $0.id == field.id }
                                    }
                                    Toggle("Allow multiple spans", isOn: $field.multiple)
                                        .font(MereRunTheme.captionFont)
                                }
                            }
                            Button("Add field") { structure.fields.append(.init()) }
                                .buttonStyle(.plain)
                                .foregroundStyle(MereRunTheme.accent)
                        }
                        .padding(12)
                        .merePanel()
                    }
                }
                section(title: "Classifications", addLabel: "Add task",
                        add: { document.classifications.append(.init()) }) {
                    ForEach($document.classifications) { $task in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                TextField("Task name", text: $task.name).textFieldStyle(.roundedBorder)
                                Button {
                                    document.classifications.removeAll { $0.id == task.id }
                                } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel("Remove classification task")
                            }
                            TextField("Optional prompt", text: $task.prompt)
                                .textFieldStyle(.roundedBorder)
                            ForEach($task.labels) { $label in
                                termEditor(name: $label.name, detail: $label.detail, nameHint: "Label") {
                                    task.labels.removeAll { $0.id == label.id }
                                }
                            }
                            Button("Add label") { task.labels.append(.init(name: "")) }
                                .buttonStyle(.plain)
                                .foregroundStyle(MereRunTheme.accent)
                            Toggle("Select multiple labels", isOn: $task.multiLabel)
                                .font(MereRunTheme.captionFont)
                            if task.multiLabel {
                                HStack {
                                    Text("Label threshold")
                                    Slider(value: $task.threshold, in: 0...1, step: 0.01)
                                    Text(task.threshold.formatted(.percent.precision(.fractionLength(0))))
                                        .monospacedDigit()
                                }
                                .font(MereRunTheme.captionFont)
                            }
                        }
                        .padding(12)
                        .merePanel()
                    }
                }
                HStack {
                    Text("Confidence threshold")
                    Slider(value: $document.threshold, in: 0...1, step: 0.01)
                    Text(document.threshold.formatted(.percent.precision(.fractionLength(0))))
                        .monospacedDigit()
                }
                .font(MereRunTheme.captionFont)
                Toggle("Process long text in overlapping chunks", isOn: $processLongText)
                    .font(MereRunTheme.captionFont)
                actions
            }
            .padding(18)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Extract")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Menu {
                    Button("Load the example", action: loadExample)
                    Divider()
                    Button("Import Request…", action: importRequest)
                    Button("Export Request…", action: exportRequest).disabled(!document.problems.isEmpty)
                    Divider()
                    Button("Clear", role: .destructive) { document = StudioExtractionDocument() }
                        .disabled(document == StudioExtractionDocument())
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
            }
            Text("Find named entities, relation pairs, and structured fields using GLiNER2.5 Decide.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
            Text("Model · GLiNER2.5 Decide")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private func section<Content: View>(title: String, addLabel: String, add: @escaping () -> Void,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(title).font(MereRunTheme.captionFont).foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Button(addLabel, action: add)
                    .buttonStyle(.plain)
                    .foregroundStyle(MereRunTheme.accent)
            }
            content()
        }
    }

    private func termEditor(name: Binding<String>, detail: Binding<String>, nameHint: String,
                            remove: @escaping () -> Void) -> some View {
        HStack(alignment: .top, spacing: 6) {
            VStack(spacing: 4) {
                TextField(nameHint, text: name).textFieldStyle(.roundedBorder)
                TextField("Optional description", text: detail).textFieldStyle(.roundedBorder)
            }
            Button(action: remove) { Image(systemName: "minus.circle") }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove \(nameHint.lowercased())")
        }
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !document.problems.isEmpty, !(document.text.isEmpty && document.entities.isEmpty &&
                                              document.relations.isEmpty && document.structures.isEmpty &&
                                              document.classifications.isEmpty) {
                ForEach(document.problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.circle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
            }
            HStack {
                Button("Check fit") { run(preflight: true) }.buttonStyle(.mereSecondary)
                Spacer()
                Button("Extract") { run(preflight: false) }.buttonStyle(.merePrimary)
            }
            .disabled(!document.problems.isEmpty || isRunning)
            if let message {
                Text(message).font(MereRunTheme.captionFont).foregroundStyle(MereRunTheme.red)
            }
        }
    }

    private var result: some View {
        StudioExtractionResultPane(requestID: requestID, onLoadExample: loadExample)
            .padding(18)
    }

    private func run(preflight: Bool) {
        message = nil
        let directory = StudioOutputLocation.outputDirectoryURL(
            domain: .text, prompt: document.entities.first?.name ?? document.text,
            fallbackStem: "extractions")
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
        command.outputPath = directory.appendingPathComponent(preflight ? "fit.json" : "extractions.json").path
        command.preflight = preflight
        command.force = true
        command.extraArguments = processLongText ? "--long" : ""
        guard let template = CommandCatalog.template(id: .textExtract) else { return }
        let request = StudioRunRequest(mode: .chat, templateID: template.id, template: template, draft: command)
        requestID = (try? StudioTaskRunner(controller: controller, library: library)
            .run(request: request, task: .textExtract, validating: false))?.id
    }

    private func saveDraftRequest() async {
        guard document.problems.isEmpty else { draftRequestPath = ""; return }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        let url = StudioExtractionDocument.draftRequestURL()
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try document.requestJSON().write(to: url, options: .atomic)
            draftRequestPath = url.path
        } catch {
            draftRequestPath = ""
        }
    }

    private func loadExample() {
        if document == StudioExtractionDocument() || document == .example {
            document = .example
        } else {
            confirmsExample = true
        }
    }

    private func importRequest() {
        guard let url = StudioFilePanels.chooseFile(title: "Import an extraction request",
                                                     allowedContentTypes: [.json]).first else { return }
        do {
            document = try StudioExtractionDocument.importing(Data(contentsOf: url))
            message = nil
        } catch {
            message = "That file is not an extraction request: \(error.localizedDescription)"
        }
    }

    private func exportRequest() {
        guard let url = StudioFilePanels.saveFile(title: "Export the extraction request",
                                                  suggestedName: "request.json", allowedContentTypes: [.json]) else { return }
        do {
            try document.requestJSON().write(to: url, options: .atomic)
        } catch {
            message = "Studio could not save the request: \(error.localizedDescription)"
        }
    }
}

private struct StudioExtractionResultPane: View {
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
                    Text(item.commandDraft?.preflight == true ? "Checking the fit…" : "Extracting…")
                        .font(MereRunTheme.captionFont)
                    if let progress = controller.progressByRequestID[item.id] {
                        Text(progress.label).font(MereRunTheme.captionFont)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .completed:
                StudioExtractionAnswers(item: item)
            case .failed, .interrupted:
                StudioRunFailureDetail(models: controller.modelStore, item: item)
            case .cancelled:
                ContentUnavailableView("Cancelled", systemImage: "stop.circle")
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "text.viewfinder")
                    .font(.system(size: 30, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Extractions appear here").font(MereRunTheme.sectionFont)
                Text("Add text and an extraction schema, then Extract.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                Button("Load the example", action: onLoadExample).buttonStyle(.mereSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct StudioExtractionAnswers: View {
    let item: StudioLibraryItem
    @State private var plan: StudioExtractionPlan?
    @State private var plans: [StudioExtractionPlan]?
    @State private var result: StudioExtractionResult?

    private var outputURL: URL? {
        item.commandDraft.flatMap { $0.outputPath.isBlank ? nil : URL(fileURLWithPath: $0.outputPath) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text(item.commandDraft?.preflight == true ? "Fit" : "Extractions")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    if let outputURL {
                        Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([outputURL]) }
                            .buttonStyle(.mereSecondary)
                    }
                }
                if let plan {
                    Text("\(plan.inputTokens) input tokens · \(plan.wordCount) words")
                    ForEach(plan.schemaNames, id: \.self) { name in
                        Label(name, systemImage: "checkmark.circle")
                    }
                } else if let plans {
                    Text("\(plans.count) chunks · \(plans.reduce(0) { $0 + $1.inputTokens }) input tokens")
                    ForEach(plans.indices, id: \.self) { index in
                        Text("Chunk \(index + 1): \(plans[index].inputTokens) tokens")
                    }
                } else if let result {
                    Text("\(result.inputTokens) input tokens · \(result.runtime)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    ForEach(result.entities.keys.sorted(), id: \.self) { name in
                        if let spans = result.entities[name] {
                            StudioExtractionCard(title: name) {
                                ForEach(spans, id: \.start) { span in
                                    spanRow(span)
                                }
                            }
                        }
                    }
                    ForEach(result.relations.keys.sorted(), id: \.self) { name in
                        if let relations = result.relations[name] {
                            StudioExtractionCard(title: name) {
                                ForEach(relations.indices, id: \.self) { index in
                                    let relation = relations[index]
                                    Text("\(relation.head.text) → \(relation.tail.text)")
                                        .font(MereRunTheme.bodyFont)
                                }
                            }
                        }
                    }
                    ForEach(result.structures.keys.sorted(), id: \.self) { name in
                        if let records = result.structures[name] {
                            StudioExtractionCard(title: name) {
                                ForEach(records.indices, id: \.self) { index in
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Record \(index + 1)").font(.caption.weight(.semibold))
                                        ForEach(records[index].keys.sorted(), id: \.self) { field in
                                            Text("\(field): \(records[index][field, default: []].map(\.text).joined(separator: ", "))")
                                                .font(MereRunTheme.bodyFont)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ForEach(result.classifications.keys.sorted(), id: \.self) { name in
                        if let head = result.classifications[name] {
                            StudioExtractionCard(title: name) {
                                Text(head.labels.joined(separator: ", "))
                                    .font(MereRunTheme.bodyFont)
                            }
                        }
                    }
                } else {
                    ContentUnavailableView("No result to show", systemImage: "doc.questionmark")
                }
            }
        }
        .task(id: item.id) { load() }
    }

    private func spanRow(_ span: StudioExtractionResult.Span) -> some View {
        HStack {
            Text(span.text).font(MereRunTheme.bodyFont)
            Spacer()
            Text("\(span.start)–\(span.end)")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(span.confidence.formatted(.percent.precision(.fractionLength(1))))
                .font(MereRunTheme.captionFont)
                .monospacedDigit()
        }
    }

    private func load() {
        guard let outputURL, let data = try? Data(contentsOf: outputURL) else { return }
        if item.commandDraft?.preflight == true {
            plan = try? JSONDecoder().decode(StudioExtractionPlan.self, from: data)
            plans = try? JSONDecoder().decode([StudioExtractionPlan].self, from: data)
        } else {
            result = try? JSONDecoder().decode(StudioExtractionResult.self, from: data)
        }
    }
}

private struct StudioExtractionCard<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(MereRunTheme.captionFont).foregroundStyle(MereRunTheme.textSecondary)
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }
}
