import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

enum StudioUtilityTask: String, CaseIterable, Identifiable {
    case embeddings
    case anonymize
    case imageValidation
    case datasetDiscovery
    case runPlan

    var id: String { rawValue }

    var title: String {
        switch self {
        case .embeddings: "Embeddings"
        case .anonymize: "Anonymize"
        case .imageValidation: "Image validation"
        case .datasetDiscovery: "Dataset discovery"
        case .runPlan: "Run plan"
        }
    }

    var systemImage: String {
        switch self {
        case .embeddings: "point.3.connected.trianglepath.dotted"
        case .anonymize: "eye.slash"
        case .imageValidation: "checkmark.seal"
        case .datasetDiscovery: "folder.badge.questionmark"
        case .runPlan: "list.bullet.clipboard"
        }
    }
}

struct StudioEmbeddingDocument: Equatable {
    struct Vector: Identifiable, Equatable {
        let id: Int
        let values: [Double]

        var norm: Double {
            sqrt(values.reduce(0) { $0 + ($1 * $1) })
        }
    }

    let model: String
    let promptTokens: Int
    let vectors: [Vector]

    static func decode(_ data: Data) -> StudioEmbeddingDocument? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rows = object["data"] as? [[String: Any]]
        else { return nil }

        let vectors = rows.compactMap { row -> Vector? in
            guard let values = row["embedding"] as? [NSNumber] else { return nil }
            return Vector(
                id: (row["index"] as? NSNumber)?.intValue ?? 0,
                values: values.map(\.doubleValue)
            )
        }
        let usage = object["usage"] as? [String: Any]
        return StudioEmbeddingDocument(
            model: object["model"] as? String ?? "Embedding model",
            promptTokens: (usage?["prompt_tokens"] as? NSNumber)?.intValue ?? 0,
            vectors: vectors
        )
    }

    func cosineSimilarity(_ lhs: Vector, _ rhs: Vector) -> Double {
        guard lhs.values.count == rhs.values.count, !lhs.values.isEmpty else { return 0 }
        let dot = zip(lhs.values, rhs.values).reduce(0) { $0 + ($1.0 * $1.1) }
        let denominator = lhs.norm * rhs.norm
        return denominator > 0 ? dot / denominator : 0
    }
}

struct StudioAnonymizationDocument: Equatable {
    struct Span: Identifiable, Equatable {
        let id: Int
        let label: String
        let text: String
        let startToken: Int
        let endToken: Int
    }

    struct Result: Identifiable, Equatable {
        let id: Int
        let text: String
        let anonymizedText: String
        let tokenCount: Int
        let spans: [Span]
    }

    let model: String
    let results: [Result]

    static func decode(_ data: Data) -> StudioAnonymizationDocument? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rows = object["data"] as? [[String: Any]]
        else { return nil }

        let results = rows.enumerated().map { index, row in
            let spans = (row["spans"] as? [[String: Any]] ?? []).enumerated().map { spanIndex, span in
                Span(
                    id: spanIndex,
                    label: span["label"] as? String ?? "PII",
                    text: span["text"] as? String ?? "",
                    startToken: (span["startToken"] as? NSNumber)?.intValue
                        ?? (span["start_token"] as? NSNumber)?.intValue
                        ?? 0,
                    endToken: (span["endToken"] as? NSNumber)?.intValue
                        ?? (span["end_token"] as? NSNumber)?.intValue
                        ?? 0
                )
            }
            return Result(
                id: index,
                text: row["text"] as? String ?? "",
                anonymizedText: row["anonymized_text"] as? String ?? "",
                tokenCount: (row["token_count"] as? NSNumber)?.intValue ?? 0,
                spans: spans
            )
        }
        return StudioAnonymizationDocument(
            model: object["model"] as? String ?? "Privacy Filter",
            results: results
        )
    }
}

struct StudioDatasetDiscoveryDocument: Equatable {
    struct Candidate: Identifiable, Equatable {
        let id: String
        let name: String
        let path: String
        let status: String
        let trainable: Bool
        let images: Int
        let captions: Int
        let usablePairs: Int
        let problems: [String]
    }

    let summary: String
    let scannedDirectories: Int
    let candidates: [Candidate]
    let diagnostics: [String]

    static func decode(_ data: Data) -> StudioDatasetDiscoveryDocument? {
        guard
            let envelope = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let result = envelope["result"] as? [String: Any]
        else { return nil }

        let candidates = (result["candidates"] as? [[String: Any]] ?? []).map { row in
            let diagnostics = row["diagnostics"] as? [[String: Any]] ?? []
            return Candidate(
                id: row["id"] as? String ?? UUID().uuidString,
                name: row["name"] as? String ?? row["relative_path"] as? String ?? "Dataset",
                path: row["path"] as? String ?? "",
                status: row["status"] as? String ?? "unknown",
                trainable: row["trainable"] as? Bool ?? false,
                images: (row["image_count"] as? NSNumber)?.intValue ?? 0,
                captions: (row["caption_count"] as? NSNumber)?.intValue ?? 0,
                usablePairs: (row["usable_pair_count"] as? NSNumber)?.intValue ?? 0,
                problems: diagnostics.compactMap { diagnosticText($0) }
            )
        }
        let diagnostics = (envelope["diagnostics"] as? [[String: Any]] ?? [])
            .compactMap { diagnosticText($0) }
        return StudioDatasetDiscoveryDocument(
            summary: envelope["summary"] as? String ?? "Dataset discovery complete.",
            scannedDirectories: (result["scanned_directory_count"] as? NSNumber)?.intValue ?? 0,
            candidates: candidates,
            diagnostics: diagnostics
        )
    }

    private static func diagnosticText(_ value: [String: Any]) -> String? {
        let title = value["title"] as? String ?? ""
        let message = value["message"] as? String ?? ""
        let combined = [title, message].filter { !$0.isEmpty }.joined(separator: ": ")
        return combined.isEmpty ? nil : combined
    }
}

struct StudioRunPlanDocument: Equatable {
    let summary: String
    let status: String
    let paths: [(label: String, path: String)]
    let diagnostics: [String]

    static func == (lhs: StudioRunPlanDocument, rhs: StudioRunPlanDocument) -> Bool {
        lhs.summary == rhs.summary
            && lhs.status == rhs.status
            && lhs.paths.elementsEqual(rhs.paths, by: ==)
            && lhs.diagnostics == rhs.diagnostics
    }

    static func decode(_ data: Data) -> StudioRunPlanDocument? {
        guard let envelope = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        var paths: [(String, String)] = []
        collectPaths(in: envelope["result"], prefix: nil, into: &paths)
        let diagnostics = (envelope["diagnostics"] as? [[String: Any]] ?? []).compactMap { value -> String? in
            let title = value["title"] as? String ?? ""
            let message = value["message"] as? String ?? ""
            let combined = [title, message].filter { !$0.isEmpty }.joined(separator: ": ")
            return combined.isEmpty ? nil : combined
        }
        return StudioRunPlanDocument(
            summary: envelope["summary"] as? String ?? "Plan processed.",
            status: envelope["status"] as? String ?? "unknown",
            paths: paths,
            diagnostics: diagnostics
        )
    }

    private static func collectPaths(
        in value: Any?,
        prefix: String?,
        into paths: inout [(String, String)]
    ) {
        if let dictionary = value as? [String: Any] {
            for (key, nested) in dictionary.sorted(by: { $0.key < $1.key }) {
                if let path = nested as? String,
                   key.lowercased().contains("path") || key.lowercased().contains("directory") {
                    paths.append((key.humanizedUtilityKey, path))
                } else {
                    collectPaths(in: nested, prefix: key, into: &paths)
                }
            }
        } else if let array = value as? [Any] {
            for nested in array {
                collectPaths(in: nested, prefix: prefix, into: &paths)
            }
        }
    }
}

struct StudioUtilityLabView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Owned by the host. Image ▸ Datasets shows its own picker over the dataset utilities;
    /// Text drives Embeddings and Anonymize from the shell's task control.
    @Binding var task: StudioUtilityTask
    let tasks: [StudioUtilityTask]
    let showsTaskPicker: Bool
    @State private var requestID: UUID?
    @State private var inputText = "semantic search query\nrelated document"
    @State private var model = ""
    @State private var maxTokens = 2_048
    @State private var replacement = "[{label}]"
    @State private var validationTest = "all"
    @State private var validationFamily = "zimage"
    @State private var saveReference = false
    @State private var compareReference = false
    @State private var referenceDirectory = ""
    @State private var datasetRoot = ""
    @State private var maxDepth = 4
    @State private var minUsablePairs = 1
    @State private var trainingOutputRoot = ""
    @State private var trainingModel = ""
    @State private var trainingRecipe = ""
    @State private var excludePreviewImages = false
    @State private var planPath = ""
    @State private var planMode = "Preflight"
    @State private var materializePath = ""
    @State private var outputPath = ""

    init(task: Binding<StudioUtilityTask>, tasks: [StudioUtilityTask], showsTaskPicker: Bool) {
        _task = task
        self.tasks = tasks
        self.showsTaskPicker = showsTaskPicker
    }

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    private var resultData: Data? {
        if let outputURL = item?.outputURL, let data = try? Data(contentsOf: outputURL) {
            return data
        }
        if let outputText = item?.outputText {
            return Data(outputText.utf8)
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            if showsTaskPicker {
                Picker("Utility", selection: $task) {
                    ForEach(tasks) { candidate in
                        Label(candidate.title, systemImage: candidate.systemImage)
                            .tag(candidate)
                    }
                }
                .pickerStyle(.segmented)
                .padding(16)
            }

            HSplitView {
                ScrollView {
                    controls
                        .padding(18)
                }
                .frame(minWidth: 340, idealWidth: 390, maxWidth: 470)

                result
                    .padding(18)
                    .frame(minWidth: 520, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(MereRunTheme.background)
        .onChange(of: task) { _, _ in
            requestID = nil
            applyTaskDefaults()
        }
        .onAppear {
            applyTaskDefaults()
        }
    }

    @ViewBuilder
    private var controls: some View {
        VStack(alignment: .leading, spacing: 16) {
            switch task {
            case .embeddings:
                embeddingsControls
            case .anonymize:
                anonymizeControls
            case .imageValidation:
                validationControls
            case .datasetDiscovery:
                datasetControls
            case .runPlan:
                runPlanControls
            }
            Button {
                run()
            } label: {
                Label(runButtonTitle, systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!canRun)
        }
    }

    private var embeddingsControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Vectorize local text", detail: "Enter one document per line. The result view compares every vector.")
            TextEditor(text: $inputText)
                .font(MereRunTheme.bodyFont)
                .scrollContentBackground(.hidden)
                .padding(8)
                .frame(minHeight: 150)
                .background(MereRunTheme.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
            TextField("Model id or local path (optional)", text: $model)
                .mereField()
            Stepper("Maximum tokens: \(maxTokens)", value: $maxTokens, in: 64...32_768, step: 64)
        }
    }

    private var anonymizeControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Find and redact PII", detail: "Review every detected span beside the original and protected text.")
            TextEditor(text: $inputText)
                .font(MereRunTheme.bodyFont)
                .scrollContentBackground(.hidden)
                .padding(8)
                .frame(minHeight: 180)
                .background(MereRunTheme.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
            TextField("Replacement, using {label} and {index}", text: $replacement)
                .mereField()
            TextField("Model id or local path (optional)", text: $model)
                .mereField()
            Stepper("Maximum tokens: \(maxTokens)", value: $maxTokens, in: 64...32_768, step: 64)
        }
    }

    private var validationControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Validate the image runtime", detail: "Deterministic VAE, encoder, transformer, and end-to-end checks.")
            Picker("Suite", selection: $validationTest) {
                ForEach(["all", "vae", "encoder", "transformer", "pipeline"], id: \.self) {
                    Text($0.capitalized).tag($0)
                }
            }
            Picker("Family", selection: $validationFamily) {
                Text("Z-Image").tag("zimage")
                Text("FLUX.2 Klein").tag("klein")
            }
            Toggle("Save this run as a reference", isOn: $saveReference)
            Toggle("Compare with an existing reference", isOn: $compareReference)
            if compareReference {
                StudioPathField(
                    label: "Reference directory",
                    placeholder: "/path/to/reference",
                    path: $referenceDirectory,
                    picksDirectory: true
                )
            }
            StudioPathField(
                label: "Artifact directory",
                placeholder: "/path/to/validation-output",
                path: $outputPath,
                picksDirectory: true
            )
        }
    }

    private var datasetControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Discover training datasets", detail: "Scan nested folders and rank usable image-caption candidates.")
            StudioPathField(
                label: "Search root",
                placeholder: "/path/to/datasets",
                path: $datasetRoot,
                picksDirectory: true
            )
            Stepper("Maximum depth: \(maxDepth)", value: $maxDepth, in: 0...20)
            Stepper("Minimum usable pairs: \(minUsablePairs)", value: $minUsablePairs, in: 1...100_000)
            Toggle("Exclude preview images", isOn: $excludePreviewImages)
            DisclosureGroup("Training action suggestions") {
                VStack(spacing: 10) {
                    StudioPathField(
                        label: "Per-dataset output root",
                        placeholder: "Optional",
                        path: $trainingOutputRoot,
                        picksDirectory: true
                    )
                    TextField("Training model (optional)", text: $trainingModel)
                        .mereField()
                    TextField("Training recipe (optional)", text: $trainingRecipe)
                        .mereField()
                }
                .padding(.top, 8)
            }
        }
    }

    private var runPlanControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Materialize an image run plan", detail: "Inspect a saved plan or turn it into a durable run directory and action bundle.")
            StudioPathField(
                label: "Plan JSON",
                placeholder: "/path/to/plan.json",
                path: $planPath,
                allowedContentTypes: [.json]
            )
            Picker("Action", selection: $planMode) {
                Text("Preflight").tag("Preflight")
                Text("Materialize").tag("Materialize")
            }
            .pickerStyle(.segmented)
            if planMode == "Materialize" {
                StudioPathField(
                    label: "Run directory",
                    placeholder: "/path/to/run",
                    path: $materializePath,
                    picksDirectory: true
                )
            }
        }
    }

    @ViewBuilder
    private var result: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("\(task.title) result")
                .font(MereRunTheme.sectionFont)
            if let data = resultData {
                switch task {
                case .embeddings:
                    if let document = StudioEmbeddingDocument.decode(data) {
                        embeddingResult(document)
                    } else {
                        fallbackResult
                    }
                case .anonymize:
                    if let document = StudioAnonymizationDocument.decode(data) {
                        anonymizationResult(document)
                    } else {
                        fallbackResult
                    }
                case .imageValidation:
                    validationResult
                case .datasetDiscovery:
                    if let document = StudioDatasetDiscoveryDocument.decode(data) {
                        datasetResult(document)
                    } else {
                        fallbackResult
                    }
                case .runPlan:
                    if let document = StudioRunPlanDocument.decode(data) {
                        runPlanResult(document)
                    } else {
                        fallbackResult
                    }
                }
            } else if task == .imageValidation, requestID != nil {
                validationResult
            } else {
                StudioSpecialistResultView(requestID: requestID)
            }
        }
    }

    private func embeddingResult(_ document: StudioEmbeddingDocument) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                metricStrip([
                    ("Vectors", "\(document.vectors.count)"),
                    ("Dimensions", "\(document.vectors.first?.values.count ?? 0)"),
                    ("Tokens", "\(document.promptTokens)")
                ])
                Text(document.model)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .textSelection(.enabled)

                if document.vectors.count > 1 {
                    Text("Cosine similarity")
                        .font(MereRunTheme.sectionFont)
                    Grid(alignment: .trailing, horizontalSpacing: 8, verticalSpacing: 8) {
                        GridRow {
                            Text("")
                            ForEach(document.vectors) { vector in
                                Text("#\(vector.id + 1)").font(MereRunTheme.captionFont)
                            }
                        }
                        ForEach(document.vectors) { lhs in
                            GridRow {
                                Text("#\(lhs.id + 1)").font(MereRunTheme.captionFont)
                                ForEach(document.vectors) { rhs in
                                    let similarity = document.cosineSimilarity(lhs, rhs)
                                    Text(similarity.formatted(.number.precision(.fractionLength(3))))
                                        .font(MereRunTheme.monoFont)
                                        .padding(6)
                                        .background(similarityColor(similarity))
                                        .clipShape(RoundedRectangle(cornerRadius: 5))
                                }
                            }
                        }
                    }
                    .padding(12)
                    .merePanel()
                }

                ForEach(document.vectors) { vector in
                    VStack(alignment: .leading, spacing: 7) {
                        HStack {
                            Text("Vector \(vector.id + 1)")
                                .font(MereRunTheme.sectionFont)
                            Spacer()
                            Text("L2 \(vector.norm.formatted(.number.precision(.fractionLength(4))))")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                        Text(vector.values.prefix(12).map { String(format: "%.5f", $0) }.joined(separator: "  "))
                            .font(MereRunTheme.monoFont)
                            .lineLimit(3)
                            .textSelection(.enabled)
                    }
                    .padding(12)
                    .merePanel()
                }
            }
        }
    }

    private func anonymizationResult(_ document: StudioAnonymizationDocument) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                metricStrip([
                    ("Documents", "\(document.results.count)"),
                    ("PII spans", "\(document.results.reduce(0) { $0 + $1.spans.count })"),
                    ("Tokens", "\(document.results.reduce(0) { $0 + $1.tokenCount })")
                ])
                ForEach(document.results) { result in
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Original")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text(result.text).textSelection(.enabled)
                        Text("Protected")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text(result.anonymizedText)
                            .font(MereRunTheme.bodyFont)
                            .foregroundStyle(MereRunTheme.green)
                            .textSelection(.enabled)
                        if !result.spans.isEmpty {
                            FlowLayout(spacing: 6) {
                                ForEach(result.spans) { span in
                                    Text("\(span.label): \(span.text)")
                                        .font(MereRunTheme.captionFont)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 5)
                                        .background(MereRunTheme.yellow.opacity(0.14))
                                        .clipShape(Capsule())
                                        .help("Tokens \(span.startToken)–\(span.endToken)")
                                }
                            }
                        }
                    }
                    .padding(14)
                    .merePanel()
                }
            }
        }
    }

    private var validationResult: some View {
        StudioSpecialistResultView(
            requestID: requestID,
            preferredKinds: [.image, .text]
        )
    }

    private func datasetResult(_ document: StudioDatasetDiscoveryDocument) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text(document.summary)
                    .font(MereRunTheme.bodyFont)
                metricStrip([
                    ("Scanned", "\(document.scannedDirectories)"),
                    ("Candidates", "\(document.candidates.count)"),
                    ("Trainable", "\(document.candidates.filter(\.trainable).count)")
                ])
                ForEach(document.candidates) { candidate in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: candidate.trainable ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(candidate.trainable ? MereRunTheme.green : MereRunTheme.yellow)
                            Text(candidate.name)
                                .font(MereRunTheme.sectionFont)
                            Spacer()
                            Text(candidate.status.uppercased())
                                .font(MereRunTheme.captionFont)
                        }
                        Text(candidate.path)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .textSelection(.enabled)
                        metricStrip([
                            ("Images", "\(candidate.images)"),
                            ("Captions", "\(candidate.captions)"),
                            ("Usable", "\(candidate.usablePairs)")
                        ])
                        ForEach(candidate.problems, id: \.self) { problem in
                            Label(problem, systemImage: "exclamationmark.circle")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textSecondary)
                        }
                    }
                    .padding(12)
                    .merePanel()
                }
                ForEach(document.diagnostics, id: \.self) {
                    MereBanner(severity: .warning, text: $0)
                }
            }
        }
    }

    private func runPlanResult(_ document: StudioRunPlanDocument) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: document.status == "ok" ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(document.status == "ok" ? MereRunTheme.green : MereRunTheme.yellow)
                    Text(document.summary)
                        .font(MereRunTheme.bodyFont)
                }
                ForEach(Array(document.paths.enumerated()), id: \.offset) { _, row in
                    HStack(alignment: .top) {
                        Text(row.label)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .frame(width: 150, alignment: .leading)
                        Text(row.path)
                            .font(MereRunTheme.monoFont)
                            .textSelection(.enabled)
                        Spacer()
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: row.path)])
                        } label: {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.borderless)
                    }
                    .padding(10)
                    .merePanel()
                }
                ForEach(document.diagnostics, id: \.self) {
                    MereBanner(severity: .info, text: $0)
                }
            }
        }
    }

    private var fallbackResult: some View {
        StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text, .image])
    }

    private var runButtonTitle: String {
        switch task {
        case .embeddings: "Generate embeddings"
        case .anonymize: "Protect text"
        case .imageValidation: "Run validation"
        case .datasetDiscovery: "Discover datasets"
        case .runPlan: planMode == "Preflight" ? "Preflight plan" : "Materialize run"
        }
    }

    private var canRun: Bool {
        switch task {
        case .embeddings, .anonymize:
            !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .imageValidation:
            !outputPath.isEmpty && (!compareReference || !referenceDirectory.isEmpty)
        case .datasetDiscovery:
            !datasetRoot.isEmpty
        case .runPlan:
            !planPath.isEmpty && (planMode == "Preflight" || !materializePath.isEmpty)
        }
    }

    private func run() {
        var draft = CommandDraft()
        switch task {
        case .embeddings:
            draft.prompt = inputText
            draft.model = model
            draft.maxTokens = maxTokens
            draft.force = true
            draft.outputPath = outputPathForUtility(filename: "embeddings.json")
            requestID = submit(.textEmbed, mode: .chat, draft: draft)
        case .anonymize:
            draft.prompt = inputText
            draft.model = model
            draft.maxTokens = maxTokens
            draft.replacement = replacement
            draft.all = true
            draft.force = true
            draft.outputPath = outputPathForUtility(filename: "anonymized.json")
            requestID = submit(.textAnonymize, mode: .chat, draft: draft)
        case .imageValidation:
            draft.backend = validationTest
            draft.variant = validationFamily
            draft.outputPath = outputPath
            draft.force = saveReference
            draft.all = compareReference
            draft.referenceDirectoryPath = referenceDirectory
            requestID = submit(.imageValidate, mode: .createImage, draft: draft)
        case .datasetDiscovery:
            draft.inputPath = datasetRoot
            draft.maxDepth = maxDepth
            draft.minUsablePairs = minUsablePairs
            draft.trainingOutputRoot = trainingOutputRoot
            draft.trainingModel = trainingModel
            draft.trainingRecipe = trainingRecipe
            draft.excludePreviewImages = excludePreviewImages
            draft.json = true
            requestID = submit(.imageDatasetDiscover, mode: .createImage, draft: draft)
        case .runPlan:
            draft.inputPath = planPath
            draft.preflight = planMode == "Preflight"
            draft.materializePath = planMode == "Materialize" ? materializePath : ""
            draft.json = true
            requestID = submit(.imageRunPlan, mode: .createImage, draft: draft)
        }
    }

    private func submit(_ templateID: CommandTemplateID, mode: StudioMode, draft: CommandDraft) -> UUID? {
        StudioSpecialistRunner.submit(
            templateID: templateID,
            mode: mode,
            draft: draft,
            controller: controller,
            library: library
        )
    }

    private func applyTaskDefaults() {
        model = ""
        maxTokens = 2_048
        switch task {
        case .embeddings:
            inputText = "semantic search query\nrelated document"
        case .anonymize:
            inputText = "My name is Alice Smith and my email is alice@example.com"
        case .imageValidation:
            if outputPath.isEmpty {
                outputPath = StudioSpecialistFiles.timestampedDirectory(component: "validation").path
            }
        case .datasetDiscovery:
            break
        case .runPlan:
            if materializePath.isEmpty {
                materializePath = StudioSpecialistFiles.timestampedDirectory(component: "run-plan").path
            }
        }
    }

    private func outputPathForUtility(filename: String) -> String {
        StudioSpecialistFiles.timestampedDirectory(component: "utilities")
            .appendingPathComponent(filename)
            .path
    }

    private func sectionTitle(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(MereRunTheme.sectionFont)
            Text(detail)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private func metricStrip(_ values: [(String, String)]) -> some View {
        HStack(spacing: 8) {
            ForEach(Array(values.enumerated()), id: \.offset) { _, value in
                VStack(alignment: .leading, spacing: 2) {
                    Text(value.0)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text(value.1)
                        .font(MereRunTheme.sectionFont)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(9)
                .merePanel()
            }
        }
    }

    private func similarityColor(_ similarity: Double) -> Color {
        if similarity >= 0.9 { return MereRunTheme.green.opacity(0.2) }
        if similarity >= 0.6 { return MereRunTheme.accent.opacity(0.16) }
        return MereRunTheme.surfaceRaised
    }
}

private extension String {
    var humanizedUtilityKey: String {
        replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}
