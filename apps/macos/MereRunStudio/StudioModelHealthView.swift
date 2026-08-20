import AppKit
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

struct StudioModelHealthSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.dismiss) private var dismiss

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

    static let suites = ["text", "speech", "vision", "image", "embed"]

    init(onModelsChanged: @escaping () -> Void) {
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
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            ScrollView {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.xl) {
                    manifestSection
                    qualitySection
                }
                .padding(MereRunTheme.Spacing.xl)
            }
        }
        .frame(minWidth: 930, idealWidth: 1_020, minHeight: 680, idealHeight: 760)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
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

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            ZStack {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft)
                Image(systemName: "checkmark.shield")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
            }
            .frame(width: 42, height: 42)

            VStack(alignment: .leading, spacing: 2) {
                Text("Model Health & Repair")
                    .font(MereRunTheme.titleFont)
                Text("Audit manifests and run installed-model quality gates")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            Button("Done") { dismiss() }
                .buttonStyle(.merePrimary)
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, MereRunTheme.Spacing.xl)
        .padding(.vertical, MereRunTheme.Spacing.md)
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
