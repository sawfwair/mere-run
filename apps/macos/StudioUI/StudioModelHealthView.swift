import AppKit
import StudioKit
import SwiftUI

struct StudioModelRepairReport: Decodable, Equatable {
    let mode: String
    let wroteCount: Int
    let alreadyCount: Int
    let skippedCount: Int
    let entries: [StudioModelRepairEntry]

    enum CodingKeys: String, CodingKey {
        case mode
        case wroteCount = "wrote_count"
        case alreadyCount = "already_count"
        case skippedCount = "skipped_count"
        case entries
    }

    var actionableEntries: [StudioModelRepairEntry] {
        entries.filter { $0.status == "would_write" || $0.status == "wrote" }
    }

    var errorEntries: [StudioModelRepairEntry] {
        entries.filter { $0.status == "skipped" && $0.message != "model directory not found" }
    }
}

struct StudioModelRepairEntry: Decodable, Equatable, Identifiable {
    let modelID: String
    let status: String
    let path: String?
    let message: String?

    var id: String { "\(modelID):\(path ?? status)" }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case status
        case path
        case message
    }
}

enum StudioModelHealthConfirmation: Identifiable {
    case repair
    case updateBaselines

    var id: String {
        switch self {
        case .repair: "repair"
        case .updateBaselines: "baselines"
        }
    }
}

/// The benchmark family, as an operator would pick it: what the suite measures, not
/// which decode path it exercises.
enum StudioBenchmarkSuite: String, CaseIterable, Identifiable {
    case fused
    case chat
    case code
    case vlm
    case toolCalls
    case toolContinuations
    case gemma4KV
    case gemma4MTP
    case q36MTP
    case lagunaDFlash
    case apiWorkload
    case fusedFixture

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fused: "Fused quality suite"
        case .chat: "Chat slice"
        case .code: "Code slice"
        case .vlm: "Vision-language"
        case .toolCalls: "Tool calls"
        case .toolContinuations: "Tool continuations"
        case .gemma4KV: "Gemma4 KV cache"
        case .gemma4MTP: "Gemma4 MTP"
        case .q36MTP: "Qwen3.6 MTP"
        case .lagunaDFlash: "Laguna DFlash"
        case .apiWorkload: "API workload"
        case .fusedFixture: "Fixture hashes"
        }
    }

    var summary: String {
        switch self {
        case .fused:
            return "The versioned Mere Lite or Mere Comprehensive suite across every scored capability."
        case .chat:
            return "A grounded-chat evaluation slice against local assistant models."
        case .code:
            return "A real coding-evaluation slice. Execution stays inside the selected sandbox."
        case .vlm:
            return "Vision-language models over synthetic or lmms-eval datasets."
        case .toolCalls:
            return "Tool-selection accuracy across local chat models."
        case .toolContinuations:
            return "Gemma 4 continuation quality after completed tool calls."
        case .gemma4KV:
            return "Gemma4 default KV cache decode against packed PolarKV."
        case .gemma4MTP:
            return "Gemma4 serial decode against verified MTP speculative decode."
        case .q36MTP:
            return "Qwen-family serial decode against adaptive and forced MTP speculative decode."
        case .lagunaDFlash:
            return "Laguna target-only, fixed DFlash, and adaptive routing in one resident process."
        case .apiWorkload:
            return "Replays a chat workload against a running API server and reads cache counters."
        case .fusedFixture:
            return "Stamps or verifies normalized fused-benchmark JSONL fixture hashes."
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .fused: .modelBenchmarkFused
        case .chat: .modelBenchmarkChat
        case .code: .modelBenchmarkCode
        case .vlm: .modelBenchmarkVLM
        case .toolCalls: .modelBenchmarkToolCalls
        case .toolContinuations: .modelBenchmarkToolContinuations
        case .gemma4KV: .modelBenchmarkGemma4KV
        case .gemma4MTP: .modelBenchmarkGemma4MTP
        case .q36MTP: .modelBenchmark
        case .lagunaDFlash: .modelBenchmarkLagunaDFlash
        case .apiWorkload: .modelBenchmarkAPIWorkload
        case .fusedFixture: .modelBenchmarkFusedFixture
        }
    }

    /// Only the multi-model suites take a `--models` list.
    var acceptsModels: Bool {
        switch self {
        case .fused, .chat, .code, .vlm, .toolCalls: true
        default: false
        }
    }

    var acceptsDryRun: Bool {
        switch self {
        case .fused, .chat, .code, .vlm, .toolCalls, .toolContinuations, .apiWorkload: true
        default: false
        }
    }
}

/// Which half of the former Health & Repair sheet a Models task shows.
enum StudioModelHealthScope: Equatable {
    /// Manifest audit and repair plus the installed-model quality gate.
    case health
    /// The benchmark family.
    case benchmarks
}

struct StudioModelHealthView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    let scope: StudioModelHealthScope
    let onModelsChanged: () -> Void

    @State private var repairReport: StudioModelRepairReport?
    @State private var repairOutput = ""
    @State private var gateList = ""
    @State private var selectedSuites: Set<String> = Set(Self.suites)
    @State private var strictPerformance = false
    @State private var updateBaselines = false
    @State private var qualityDraft: CommandDraft
    @State private var qualityRequestID: UUID?
    @State private var isAuditing = false
    @State private var isRepairing = false
    @State private var confirmation: StudioModelHealthConfirmation?
    @State private var benchmark: StudioBenchmarkSuite = .fused
    @State private var benchmarkDraft = CommandCatalog.template(id: .modelBenchmarkFused)?
        .defaultDraft() ?? CommandDraft()
    @State private var benchmarkRequestID: UUID?
    @State private var benchmarkModels = ""
    @State private var benchmarkDryRun = false

    static let suites = ["text", "speech", "vision", "image", "embed"]

    init(scope: StudioModelHealthScope, onModelsChanged: @escaping () -> Void) {
        self.scope = scope
        self.onModelsChanged = onModelsChanged
        let template = CommandCatalog.template(id: .qualityGate)
        _qualityDraft = State(initialValue: template?.defaultDraft() ?? CommandDraft())
    }

    private var qualityItem: StudioLibraryItem? {
        guard let qualityRequestID else { return nil }
        return library.items.first { $0.id == qualityRequestID }
    }

    private var suiteArgument: String {
        selectedSuites.count == Self.suites.count
            ? "all"
            : Self.suites.filter(selectedSuites.contains).joined(separator: ",")
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.xl) {
                switch scope {
                case .health:
                    manifestSection
                    qualitySection
                case .benchmarks:
                    benchmarkSection
                }
            }
            .padding(MereRunTheme.Spacing.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            guard scope == .health else { return }
            await auditManifests()
            await loadGateList()
        }
        .alert(item: $confirmation) { pending in
            switch pending {
            case .repair:
                Alert(
                    title: Text("Repair missing model manifests?"),
                    message: Text(
                        "This writes only missing mererun_model.json files for known installed models. "
                            + "Model weights and existing manifests are left untouched."
                    ),
                    primaryButton: .default(Text("Apply Repair")) {
                        Task { await repairManifests() }
                    },
                    secondaryButton: .cancel()
                )
            case .updateBaselines:
                Alert(
                    title: Text("Replace quality baselines?"),
                    message: Text(
                        "Baseline updates change the expected local quality results. "
                            + "Review the completed JSON report before treating the new values as trusted."
                    ),
                    primaryButton: .destructive(Text("Run & Update")) {
                        startQualityGate(updatingBaselines: true)
                    },
                    secondaryButton: .cancel {
                        updateBaselines = false
                    }
                )
            }
        }
    }

    private var manifestSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Manifest integrity")
                        .font(MereRunTheme.sectionFont)
                    Text("Verify the metadata that lets mere.run identify and operate installed models.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Button {
                    Task { await auditManifests() }
                } label: {
                    Label("Audit", systemImage: "magnifyingglass")
                }
                .buttonStyle(.mereSecondary)
                .disabled(isAuditing || isRepairing)
                Button("Repair") { confirmation = .repair }
                    .buttonStyle(.merePrimary)
                    .disabled((repairReport?.wroteCount ?? 0) == 0 || isAuditing || isRepairing)
                    .opacity((repairReport?.wroteCount ?? 0) == 0 ? 0.45 : 1)
            }

            if isAuditing || isRepairing {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(isRepairing ? "Writing missing manifests" : "Auditing local model manifests")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            } else if let report = repairReport {
                HStack(spacing: 10) {
                    healthMetric("Healthy", "\(report.alreadyCount)", color: MereRunTheme.green)
                    healthMetric("Needs repair", "\(report.wroteCount)", color: report.wroteCount == 0 ? MereRunTheme.green : MereRunTheme.yellow)
                    healthMetric("Write errors", "\(report.errorEntries.count)", color: report.errorEntries.isEmpty ? MereRunTheme.textMuted : MereRunTheme.red)
                }

                if report.actionableEntries.isEmpty && report.errorEntries.isEmpty {
                    Label("Installed known models have the manifests they need.", systemImage: "checkmark.seal.fill")
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.green)
                } else {
                    VStack(spacing: 5) {
                        ForEach(report.actionableEntries + report.errorEntries) { entry in
                            HStack(spacing: 9) {
                                Image(systemName: entry.status == "skipped" ? "exclamationmark.triangle.fill" : "doc.badge.plus")
                                    .foregroundStyle(entry.status == "skipped" ? MereRunTheme.red : MereRunTheme.yellow)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(entry.modelID)
                                        .font(.system(size: 12, weight: .semibold))
                                    Text(entry.message ?? entry.path ?? "")
                                        .font(MereRunTheme.captionFont)
                                        .foregroundStyle(MereRunTheme.textMuted)
                                        .lineLimit(2)
                                }
                                Spacer()
                            }
                            .padding(9)
                            .background(MereRunTheme.surface)
                            .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
                        }
                    }
                }
            } else {
                Text(repairOutput.isEmpty ? "Manifest audit is unavailable." : repairOutput)
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.red)
                    .textSelection(.enabled)
            }
        }
        .padding(16)
        .merePanel()
    }

    private var qualitySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Installed-model quality gate")
                        .font(MereRunTheme.sectionFont)
                    Text("Run the CLI's correctness fixtures and optional performance thresholds as a durable Studio job.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                if let item = qualityItem {
                    Label(item.status.rawValue.capitalized, systemImage: qualityStatusIcon(item.status))
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(qualityStatusColor(item.status))
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Suites")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textSecondary)
                HStack(spacing: 14) {
                    ForEach(Self.suites, id: \.self) { suite in
                        Toggle(
                            suite.capitalized,
                            isOn: Binding(
                                get: { selectedSuites.contains(suite) },
                                set: { enabled in
                                    if enabled {
                                        selectedSuites.insert(suite)
                                    } else {
                                        selectedSuites.remove(suite)
                                    }
                                }
                            )
                        )
                        .toggleStyle(.checkbox)
                    }
                }
            }

            HStack(spacing: 20) {
                Toggle("Strict performance thresholds", isOn: $strictPerformance)
                    .toggleStyle(.checkbox)
                Toggle("Update baselines", isOn: $updateBaselines)
                    .toggleStyle(.checkbox)
                Spacer()
                Button {
                    revealQualityReport()
                } label: {
                    Label("Report", systemImage: "doc.text")
                }
                .buttonStyle(.mereSecondary)
                .disabled(!FileManager.default.fileExists(atPath: qualityDraft.outputPath))
                Button {
                    if updateBaselines {
                        confirmation = .updateBaselines
                    } else {
                        startQualityGate(updatingBaselines: false)
                    }
                } label: {
                    Label("Run Quality Gate", systemImage: "play.fill")
                }
                .buttonStyle(.merePrimary)
                .disabled(selectedSuites.isEmpty || qualityItem?.status == .running || qualityItem?.status == .queued)
            }

            HStack {
                Text("JSON report")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Text(qualityDraft.outputPath)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                Spacer()
            }

            if let item = qualityItem, let output = item.outputText, !output.isEmpty {
                Text(output)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(10)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .merePanel()
            } else if !gateList.isEmpty {
                DisclosureGroup("Available checks") {
                    Text(gateList)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .textSelection(.enabled)
                        .padding(.top, 6)
                }
                .font(MereRunTheme.captionFont)
            }
        }
        .padding(16)
        .merePanel()
    }

    // MARK: - Benchmarks

    private var benchmarkItem: StudioLibraryItem? {
        guard let benchmarkRequestID else { return nil }
        return library.items.first { $0.id == benchmarkRequestID }
    }

    /// The benchmark family runs as durable Library jobs beside the quality gate, so a
    /// suite and a gate produce the same kind of reviewable, revealable JSON report.
    private var benchmarkSection: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            HStack {
                Text("Benchmarks")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                if let item = benchmarkItem {
                    Label(
                        item.status.rawValue.capitalized,
                        systemImage: qualityStatusIcon(item.status)
                    )
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(qualityStatusColor(item.status))
                }
            }

            Text(benchmark.summary)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)

            Picker("Suite", selection: $benchmark) {
                ForEach(StudioBenchmarkSuite.allCases) { suite in
                    Text(suite.title).tag(suite)
                }
            }
            .frame(maxWidth: 420)

            if benchmark.acceptsModels {
                TextField(
                    "Models (comma-separated, blank runs the suite default)",
                    text: $benchmarkModels
                )
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }

            if benchmark == .fused {
                Picker("Depth", selection: $benchmarkDraft.benchmarkSuite) {
                    Text("Mere Lite").tag("lite")
                    Text("Mere Comprehensive").tag("comprehensive")
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
            }
            if benchmark == .fusedFixture {
                StudioPathField(
                    label: "Fused benchmark JSONL fixture",
                    placeholder: "Benchmark result JSONL",
                    path: $benchmarkDraft.inputPath
                )
                Toggle("Verify existing hashes", isOn: $benchmarkDraft.benchmarkFixtureCheck)
                    .help("Exit unsuccessfully when a stored fixture hash does not match")
            }

            if benchmark.acceptsDryRun {
                Toggle("Plan only (dry run)", isOn: $benchmarkDryRun)
                    .help("Resolve models and cases without running inference")
            }

            HStack(spacing: 8) {
                Button {
                    startBenchmark()
                } label: {
                    Label("Run \(benchmark.title)", systemImage: "speedometer")
                }
                .buttonStyle(.merePrimary)
                .disabled(
                    benchmarkItem?.status == .running
                        || (benchmark == .fusedFixture && benchmarkDraft.inputPath.isBlank)
                )

                if let item = benchmarkItem, item.status == .completed {
                    Button("Reveal report") {
                        if let url = item.allArtifactURLs.first {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        }
                    }
                    .buttonStyle(.mereSecondary)
                }
                Spacer()
            }

            if benchmarkRequestID != nil {
                StudioSpecialistResultView(
                    requestID: benchmarkRequestID,
                    preferredKinds: [.text]
                )
                .frame(minHeight: 220)
            }
        }
    }

    private func startBenchmark() {
        guard let template = CommandCatalog.template(id: benchmark.templateID) else { return }
        var draft = template.defaultDraft()
        draft.json = true
        draft.dryRun = benchmark.acceptsDryRun && benchmarkDryRun
        if benchmark.acceptsModels {
            draft.benchmarkModels = benchmarkModels.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if benchmark == .fused, !benchmarkDraft.benchmarkSuite.isBlank {
            draft.benchmarkSuite = benchmarkDraft.benchmarkSuite
        }
        if benchmark == .fusedFixture {
            draft.inputPath = benchmarkDraft.inputPath.trimmingCharacters(in: .whitespacesAndNewlines)
            draft.benchmarkFixtureCheck = benchmarkDraft.benchmarkFixtureCheck
        }
        benchmarkRequestID = StudioSpecialistRunner.submit(
            templateID: benchmark.templateID,
            mode: .chat,
            draft: draft,
            controller: controller,
            library: library
        )
    }

    private func healthMetric(_ title: String, _ value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title.uppercased())
                .font(.system(size: 9.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 20, weight: .semibold, design: .rounded))
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(MereRunTheme.surface)
        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
    }

    private func auditManifests() async {
        guard !isAuditing else { return }
        isAuditing = true
        defer { isAuditing = false }
        let result = await controller.utilityCommandResult(
            args: ["model", "repair-manifests", "--dry-run", "--json"]
        )
        repairOutput = result.outputText
        repairReport = StudioModelHealthJSON.decodeRepairReport(result.stdout)
    }

    private func repairManifests() async {
        guard !isRepairing else { return }
        isRepairing = true
        defer { isRepairing = false }
        let result = await controller.utilityCommandResult(
            args: ["model", "repair-manifests", "--json"]
        )
        repairOutput = result.outputText
        repairReport = StudioModelHealthJSON.decodeRepairReport(result.stdout)
        onModelsChanged()
        await auditManifests()
    }

    private func loadGateList() async {
        let result = await controller.utilityCommandResult(args: ["gate", "--list"])
        if result.exitCode == 0 {
            gateList = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private func startQualityGate(updatingBaselines: Bool) {
        qualityDraft.operationsGateSuite = suiteArgument
        qualityDraft.operationsStrictPerformance = strictPerformance
        qualityDraft.operationsUpdateBaselines = updatingBaselines
        qualityDraft.operationsListOnly = false
        qualityRequestID = StudioSpecialistRunner.submit(
            templateID: .qualityGate,
            mode: .chat,
            draft: qualityDraft,
            controller: controller,
            library: library
        )
        updateBaselines = false
    }

    private func revealQualityReport() {
        let url = URL(fileURLWithPath: qualityDraft.outputPath)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func qualityStatusIcon(_ status: StudioLibraryStatus) -> String {
        switch status {
        case .queued: "clock"
        case .running: "progress.indicator"
        case .completed: "checkmark.seal.fill"
        case .failed: "xmark.octagon.fill"
        }
    }

    private func qualityStatusColor(_ status: StudioLibraryStatus) -> Color {
        switch status {
        case .queued: MereRunTheme.textMuted
        case .running: MereRunTheme.accent
        case .completed: MereRunTheme.green
        case .failed: MereRunTheme.red
        }
    }
}

enum StudioModelHealthJSON {
    static func decodeRepairReport(_ text: String) -> StudioModelRepairReport? {
        guard let data = StudioOperationsJSON.objectData(text) else { return nil }
        return try? JSONDecoder().decode(StudioModelRepairReport.self, from: data)
    }
}
