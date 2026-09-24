import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Text ▸ Decisions: build a Laya request — the text, then choice, score, and yes-or-no questions
/// about it — and read the answers as probabilities. Studio writes the request file each run
/// needs; importing and exporting one stays available for requests made elsewhere.
struct StudioLayaDecisionView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @StudioStoredValue("Laya.document") private var document = StudioDecisionDocument()
    @StudioStoredValue("Laya.model") private var model = "text-decide-laya"
    /// Shared with the Command view's Run, so either kind of run shows here.
    @StudioStoredValue("requestID") private var requestID: UUID? = nil
    @State private var message: String?
    /// The saved copy of a ready request, for the Command view; empty while the request has problems.
    @State private var draftRequestPath = ""
    @State private var confirmsExample = false

    /// The command the Command view shows and runs: the saved copy of this request, printed.
    private var draft: CommandDraft {
        var value = CommandDraft()
        value.inputPath = draftRequestPath
        value.model = model
        value.force = true
        return value
    }

    private var isRunning: Bool {
        guard let requestID, let item = library.items.first(where: { $0.id == requestID }) else { return false }
        return item.status == .queued || item.status == .running
    }

    var body: some View {
        StudioAnalysisLayout {
            editor
        } result: {
            StudioDecisionResultPane(requestID: requestID, onLoadExample: loadExample)
                .padding(18)
        }
        .background(MereRunTheme.background)
        .studioTaskCommand(.textDecide, draft: draft)
        .task(id: document) { await saveDraftRequest() }
        .confirmationDialog("Replace this request with the example?", isPresented: $confirmsExample) {
            Button("Load the Example", role: .destructive) { document = .example }
        } message: {
            Text("The text and questions you have now are replaced.")
        }
    }

    // MARK: Editor

    private var editor: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                VStack(alignment: .leading, spacing: 6) {
                    fieldLabel("Text to judge")
                    StudioDecisionTextEditor(
                        text: $document.text,
                        placeholder: "Paste a message, a ticket, a review — whatever the questions are about.",
                        minHeight: 96
                    )
                }
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        fieldLabel("Questions")
                        Spacer()
                        addQuestionMenu
                    }
                    if document.questions.isEmpty {
                        Text("Ask which option fits, where the text sits on a scale, or whether a statement holds.")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach($document.questions) { $question in
                        StudioDecisionQuestionEditor(
                            question: $question,
                            number: (document.questions.firstIndex { $0.id == question.id } ?? 0) + 1,
                            resolvedKey: resolvedKey(for: question),
                            canMoveUp: document.questions.first?.id != question.id,
                            canMoveDown: document.questions.last?.id != question.id,
                            onMove: { move(question.id, by: $0) },
                            onDelete: { document.questions.removeAll { $0.id == question.id } }
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
                Text("Decisions")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Menu {
                    Button("Load the example", action: loadExample)
                    Divider()
                    Button("Import Request…", action: importRequest)
                    Button("Export Request…", action: exportRequest)
                        .disabled(document == StudioDecisionDocument())
                    Divider()
                    Button("Clear", role: .destructive) { document = StudioDecisionDocument() }
                        .disabled(document == StudioDecisionDocument())
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Example, import, and export")
            }
            Text("Ask choice, score, and yes-or-no questions about a text. Each answer comes with its probabilities.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Picker("Model", selection: $model) {
                Text("Laya · English").tag("text-decide-laya")
                Text("Laya · Multilingual").tag("text-decide-laya-multilingual")
                Text("Laya · Typed decisions").tag("text-decide-laya-typed-decisions")
            }
            .pickerStyle(.menu)
        }
    }

    private var addQuestionMenu: some View {
        Menu {
            ForEach(StudioDecisionQuestion.Kind.allCases) { kind in
                Button(kind.title) { document.questions.append(.blank(kind)) }
            }
        } label: {
            Label("Add question", systemImage: "plus")
                .font(.caption.weight(.semibold))
                .foregroundStyle(MereRunTheme.accent)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(document.questions.count >= 256)
    }

    private var actions: some View {
        let problems = document.problems
        return VStack(alignment: .leading, spacing: 8) {
            if !problems.isEmpty, !(document.text.isBlank && document.questions.isEmpty) {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(problems, id: \.self) { problem in
                        Label(problem, systemImage: "exclamationmark.circle")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textSecondary)
                    }
                }
            }
            HStack(spacing: 8) {
                Button("Check fit") { run(preflight: true) }
                    .buttonStyle(.mereSecondary)
                    .help("Count tokens and show what the model would cut, without loading it")
                Spacer()
                Button("Decide") { run(preflight: false) }
                    .buttonStyle(.merePrimary)
            }
            .disabled(!problems.isEmpty || isRunning)
            if let message {
                Text(message)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func fieldLabel(_ title: String) -> some View {
        Text(title)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
    }

    private func resolvedKey(for question: StudioDecisionQuestion) -> String {
        guard let index = document.questions.firstIndex(where: { $0.id == question.id }) else { return "" }
        return document.resolvedKeys[index]
    }

    private func move(_ id: UUID, by offset: Int) {
        guard let index = document.questions.firstIndex(where: { $0.id == id }) else { return }
        let target = index + offset
        guard document.questions.indices.contains(target) else { return }
        document.questions.swapAt(index, target)
    }

    // MARK: Actions

    /// Writes this request beside the run's output, in the Text domain's folder, and runs it.
    private func run(preflight: Bool) {
        message = nil
        let directory = StudioOutputLocation.outputDirectoryURL(
            domain: .text,
            prompt: document.questions.first?.prompt ?? document.text,
            fallbackStem: "decisions"
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
        command.outputPath = directory.appendingPathComponent(preflight ? "fit.json" : "decisions.json").path
        command.preflight = preflight
        command.force = true
        guard let template = CommandCatalog.template(id: .textDecide) else { return }
        let request = StudioRunRequest(mode: .chat, templateID: template.id, template: template, draft: command)
        requestID = (try? StudioTaskRunner(controller: controller, library: library)
            .run(request: request, task: .textDecide, validating: false))?.id
    }

    /// Loads the example, asking first when it would replace work.
    private func loadExample() {
        if document == StudioDecisionDocument() || document == .example {
            document = .example
        } else {
            confirmsExample = true
        }
    }

    /// Keeps the Command view's request file current, a moment after typing stops.
    private func saveDraftRequest() async {
        guard document.problems.isEmpty else {
            draftRequestPath = ""
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        let url = StudioDecisionDocument.draftRequestURL()
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try document.requestJSON().write(to: url, options: .atomic)
            draftRequestPath = url.path
        } catch {
            draftRequestPath = ""
        }
    }

    private func importRequest() {
        guard let url = StudioSpecialistFiles.chooseFile(title: "Import a decision request", allowedContentTypes: [.json]).first else {
            return
        }
        do {
            document = try StudioDecisionDocument.importing(Data(contentsOf: url))
            message = nil
        } catch {
            message = "That file is not a decision request: \(error.localizedDescription)"
        }
    }

    private func exportRequest() {
        guard let url = StudioSpecialistFiles.saveFile(
            title: "Export the decision request",
            suggestedName: "request.json",
            allowedContentTypes: [.json]
        ) else { return }
        do {
            try document.requestJSON().write(to: url, options: .atomic)
        } catch {
            message = "Studio could not save the request: \(error.localizedDescription)"
        }
    }
}

// MARK: - Question editor

/// One question: its kind, what it asks, and its options, levels, or yes-and-no wording.
private struct StudioDecisionQuestionEditor: View {
    @Binding var question: StudioDecisionQuestion
    let number: Int
    let resolvedKey: String
    let canMoveUp: Bool
    let canMoveDown: Bool
    let onMove: (Int) -> Void
    let onDelete: () -> Void
    @State private var showsKey = false

    private var promptPlaceholder: String {
        switch question.kind {
        case .choice: return "Which department should handle this?"
        case .score: return "How urgently does this need attention?"
        case .yesNo: return "The customer asks for a refund."
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Text("\(number)")
                    .font(.caption.weight(.bold))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(minWidth: 14, alignment: .leading)
                Picker("Kind", selection: kindBinding) {
                    ForEach(StudioDecisionQuestion.Kind.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Menu {
                    Button("Move Up") { onMove(-1) }.disabled(!canMoveUp)
                    Button("Move Down") { onMove(1) }.disabled(!canMoveDown)
                    Button(showsKey ? "Hide ID" : "Edit ID") { showsKey.toggle() }
                    Divider()
                    Button("Delete Question", role: .destructive, action: onDelete)
                } label: {
                    Image(systemName: "ellipsis")
                        .frame(width: 20, height: 20)
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .accessibilityLabel("Question \(number) options")
            }
            TextField(promptPlaceholder, text: $question.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .padding(8)
                .merePanel()
                .accessibilityLabel(question.kind == .yesNo ? "Statement" : "Question")
            switch question.kind {
            case .choice:
                optionList(title: "Options", placeholder: "Option", addTitle: "Add option", numbered: false)
            case .score:
                optionList(title: "Levels, lowest first", placeholder: "Level", addTitle: "Add level", numbered: true)
            case .yesNo:
                yesNoWording
            }
            if showsKey {
                HStack(spacing: 6) {
                    Text("ID")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField(resolvedKey, text: $question.key)
                        .textFieldStyle(.plain)
                        .font(.system(size: 11.5, design: .monospaced))
                        .padding(6)
                        .merePanel()
                }
                .help("The question's id in the request and the result. Blank derives one from the question.")
            }
        }
        .padding(12)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border.opacity(0.55), lineWidth: 1)
                }
        }
    }

    /// Changing kind keeps what carries over — the prompt, and options between choice and score —
    /// and gives yes-or-no its own two slots.
    private var kindBinding: Binding<StudioDecisionQuestion.Kind> {
        Binding(
            get: { question.kind },
            set: { kind in
                guard kind != question.kind else { return }
                let carried = question.kind == .yesNo ? [] : question.options
                question.kind = kind
                question.options = kind == .yesNo ? [] : (carried.isEmpty ? StudioDecisionQuestion.blank(kind).options : carried)
            }
        )
    }

    /// A choice option's description follows its name; a score level's replaces it.
    private var detailPlaceholder: String {
        question.kind == .score ? "What the model reads instead of the name (optional)" : "Description (optional)"
    }

    private func optionList(title: String, placeholder: String, addTitle: String, numbered: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            ForEach($question.options) { $option in
                let index = question.options.firstIndex { $0.id == option.id } ?? 0
                HStack(alignment: .top, spacing: 6) {
                    if numbered {
                        Text("\(index)")
                            .font(.caption.weight(.semibold))
                            .monospacedDigit()
                            .foregroundStyle(MereRunTheme.textMuted)
                            .frame(width: 14)
                            .padding(.top, 7)
                    }
                    VStack(alignment: .leading, spacing: 3) {
                        TextField(placeholder, text: $option.label)
                            .textFieldStyle(.plain)
                            .padding(6)
                            .merePanel()
                        TextField(detailPlaceholder, text: $option.detail)
                            .textFieldStyle(.plain)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textSecondary)
                            .padding(.horizontal, 6)
                    }
                    Button {
                        question.options.removeAll { $0.id == option.id }
                    } label: {
                        Image(systemName: "minus.circle")
                            .frame(width: 22, height: 22)
                    }
                    .buttonStyle(.mereIcon)
                    .disabled(question.options.count <= 1)
                    .help("Remove")
                    .accessibilityLabel("Remove \(placeholder.lowercased()) \(numbered ? index : index + 1)")
                }
            }
            Button {
                question.options.append(.init(label: ""))
            } label: {
                Label(addTitle, systemImage: "plus")
                    .font(.caption.weight(.medium))
            }
            .buttonStyle(.plain)
            .foregroundStyle(MereRunTheme.accent)
            .disabled(question.options.count >= 255)
        }
    }

    /// Yes or no carries no options, only optional wording for what each answer means.
    private var yesNoWording: some View {
        VStack(alignment: .leading, spacing: 6) {
            wording("Yes means (optional)", label: "true")
            wording("No means (optional)", label: "false")
        }
    }

    private func wording(_ placeholder: String, label: String) -> some View {
        TextField(placeholder, text: Binding(
            get: { question.options.first { $0.label == label }?.detail ?? "" },
            set: { text in
                if let index = question.options.firstIndex(where: { $0.label == label }) {
                    question.options[index].detail = text
                } else {
                    question.options.append(.init(label: label, detail: text))
                }
            }
        ))
        .textFieldStyle(.plain)
        .font(MereRunTheme.captionFont)
        .padding(6)
        .merePanel()
    }
}

/// A multi-line text field with a placeholder, drawn like the Studio's other fields.
private struct StudioDecisionTextEditor: View {
    @Binding var text: String
    let placeholder: String
    let minHeight: CGFloat

    var body: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $text)
                .font(MereRunTheme.bodyFont)
                .scrollContentBackground(.hidden)
                .padding(4)
            if text.isEmpty {
                Text(placeholder)
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .allowsHitTesting(false)
            }
        }
        .frame(minHeight: minHeight)
        .merePanel()
        .accessibilityLabel("Text to judge")
    }
}

// MARK: - Result

/// The run's answers — or, for Check fit, how each question fits the model — read from the JSON
/// the CLI wrote, with the questions taken from the request that run used.
private struct StudioDecisionResultPane: View {
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
                    Text(item.commandDraft?.preflight == true ? "Checking the fit…" : "Deciding…")
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
                StudioDecisionAnswers(item: item)
            case .failed, .interrupted:
                StudioSpecialistFailureView(item: item, models: controller.modelStore)
            case .cancelled:
                ContentUnavailableView("Cancelled", systemImage: "stop.circle")
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "checklist")
                    .font(.system(size: 30, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Answers appear here")
                    .font(MereRunTheme.sectionFont)
                Text("Add the text and a question or two, then Decide.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                Button("Load the example", action: onLoadExample)
                    .buttonStyle(.mereSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct StudioDecisionAnswers: View {
    let item: StudioLibraryItem
    @Environment(\.studioModelTitles) private var titles
    /// Read once per run: nil until loaded, then the output (nil inside when there is none).
    @State private var loaded: (output: StudioDecisionOutput?, request: StudioDecisionDocument?)?

    private var outputURL: URL? {
        item.commandDraft.flatMap { $0.outputPath.isBlank ? nil : URL(fileURLWithPath: $0.outputPath) }
    }

    var body: some View {
        Group {
            if let loaded {
                if let output = loaded.output {
                    content(plan: output.plan, result: output.result, request: loaded.request)
                } else {
                    ContentUnavailableView(
                        "No result to show",
                        systemImage: "doc.questionmark",
                        description: Text("The run finished without a decision document.")
                    )
                }
            } else {
                Color.clear
            }
        }
        .task(id: item.id) { loaded = load() }
    }

    /// The run's document — its output file, or what it printed — and the request it used, so
    /// each answer shows the question it answers.
    private func load() -> (StudioDecisionOutput?, StudioDecisionDocument?) {
        let output = outputURL.flatMap { try? Data(contentsOf: $0) }.flatMap(StudioDecisionOutput.init(data:))
            ?? item.outputText.flatMap(StudioDecisionOutput.init(outputText:))
        let request = item.commandDraft.flatMap { draft in
            (try? Data(contentsOf: URL(fileURLWithPath: draft.inputPath))).flatMap { try? StudioDecisionDocument.importing($0) }
        }
        return (output, request)
    }

    private func content(plan: StudioDecisionPlan, result: StudioDecisionResult?, request: StudioDecisionDocument?) -> some View {
        let questions = request?.questions ?? []
        let keys = request?.resolvedKeys ?? []
        return ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline) {
                    Text(result == nil ? "Fit" : "Answers")
                        .font(MereRunTheme.sectionFont)
                    Text(result == nil
                        ? "\(plan.questions.count) questions · up to \(plan.maxTokens) tokens each"
                        : "\(plan.questions.count) questions · \(StudioModelNaming.displayName(plan.model, titles: titles))")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Spacer()
                    if let outputURL {
                        Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([outputURL]) }
                            .buttonStyle(.mereSecondary)
                    }
                }
                ForEach(plan.questions, id: \.id) { planned in
                    let question = keys.firstIndex(of: planned.id).map { questions[$0] }
                    if let answer = result?.answers[planned.id] {
                        StudioDecisionAnswerCard(key: planned.id, question: question, answer: answer, plan: planned)
                    } else {
                        StudioDecisionFitCard(key: planned.id, question: question, plan: planned, budget: plan.maxTokens)
                    }
                }
            }
        }
    }
}

/// One answer: the winning option, level, or verdict first, then every probability behind it.
private struct StudioDecisionAnswerCard: View {
    let key: String
    let question: StudioDecisionQuestion?
    let answer: StudioDecisionResult.Answer
    let plan: StudioDecisionPlan.Question

    /// Options in the question's own order, with their probabilities. Score levels may share a
    /// name, so rows are told apart by position.
    private var rows: [(label: String, probability: Double)] {
        switch answer.type {
        case "score":
            let levels = question?.options.map(\.label) ?? []
            return answer.probabilities
                .compactMap { key, value in Int(key).map { ($0, value) } }
                .sorted { $0.0 < $1.0 }
                .map { index, value in (levels.indices.contains(index) ? levels[index] : "Level \(index)", value) }
        case "noul":
            return [("Yes", answer.probabilities["true"] ?? 0), ("No", answer.probabilities["false"] ?? 0)]
        default:
            let order = question?.options.map(\.label) ?? answer.probabilities.keys.sorted()
            return order.map { ($0, answer.probabilities[$0] ?? 0) }
        }
    }

    private var headline: String {
        switch answer.type {
        case "score":
            guard let score = answer.score else { return "—" }
            let levels = question?.options.map(\.label) ?? []
            let nearest = Int(score.rounded())
            let name = levels.indices.contains(nearest) ? levels[nearest] : "Level \(nearest)"
            let top = max((levels.isEmpty ? plan.optionCount : levels.count) - 1, 0)
            return "\(name) · \(String(format: "%.1f", score)) of \(top)"
        case "noul":
            let yes = answer.noul ?? answer.probabilities["true"] ?? 0
            return yes >= 0.5 ? "Yes · \(percent(yes))" : "No · \(percent(1 - yes))"
        default:
            return answer.choice ?? "—"
        }
    }

    /// The row the headline names: the level nearest the expected score, or the likeliest option.
    private var winner: Int? {
        if answer.type == "score", let score = answer.score {
            let nearest = Int(score.rounded())
            return rows.indices.contains(nearest) ? nearest : nil
        }
        return rows.indices.max { rows[$0].probability < rows[$1].probability }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(question?.prompt ?? key)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(alignment: .firstTextBaseline) {
                Text(headline)
                    .font(.system(size: 17, weight: .semibold))
                Spacer()
                Text("Confidence \(percent(answer.confidence))")
                    .font(MereRunTheme.captionFont)
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .help("How peaked the distribution is: one minus its normalized entropy, or the larger of yes and no")
            }
            VStack(alignment: .leading, spacing: 5) {
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    StudioProbabilityRow(label: row.label, probability: row.probability, isWinner: index == winner)
                }
            }
            if let note = plan.truncationNote {
                Label(note, systemImage: "scissors")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
            }
            if answer.temperatureClamped {
                Text("Calibration was clamped for this question; treat its confidence as approximate.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border.opacity(0.55), lineWidth: 1)
                }
        }
        .accessibilityElement(children: .combine)
    }
}

/// One question's fit before a run: the tokens it takes, and anything that would be cut.
private struct StudioDecisionFitCard: View {
    let key: String
    let question: StudioDecisionQuestion?
    let plan: StudioDecisionPlan.Question
    let budget: Int

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: plan.wasTruncated ? "scissors" : "checkmark.circle.fill")
                .foregroundStyle(plan.wasTruncated ? MereRunTheme.yellow : MereRunTheme.green)
            VStack(alignment: .leading, spacing: 4) {
                Text(question?.prompt ?? key)
                    .font(.callout.weight(.medium))
                    .fixedSize(horizontal: false, vertical: true)
                Text(plan.truncationNote ?? "Fits: \(plan.inputTokens) of \(budget) tokens, \(plan.optionCount) options.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(plan.wasTruncated ? MereRunTheme.yellow : MereRunTheme.textSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
        }
    }
}

/// A label, a bar filled to its probability, and the percentage; the winner in the accent.
private struct StudioProbabilityRow: View {
    let label: String
    let probability: Double
    let isWinner: Bool

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption.weight(isWinner ? .semibold : .regular))
                .foregroundStyle(isWinner ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(width: 110, alignment: .leading)
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(MereRunTheme.surfaceRaised)
                    Capsule()
                        .fill(isWinner ? MereRunTheme.accent : MereRunTheme.textMuted.opacity(0.45))
                        .frame(width: max(3, geometry.size.width * min(max(probability, 0), 1)))
                }
            }
            .frame(height: 6)
            Text(percent(probability))
                .font(.caption.weight(.medium))
                .monospacedDigit()
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 38, alignment: .trailing)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
        .accessibilityValue(percent(probability))
    }
}

private func percent(_ value: Double) -> String {
    "\(Int((min(max(value, 0), 1) * 100).rounded()))%"
}
