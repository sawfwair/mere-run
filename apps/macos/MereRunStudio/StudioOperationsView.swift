import AppKit
import SwiftUI

enum StudioOperationsScope: String, CaseIterable, Identifiable {
    case local = "This Mac"
    case relay = "Relay"

    var id: String { rawValue }
}

struct StudioExecutorProfiles: Decodable, Equatable {
    let profiles: [StudioExecutorProfile]
}

struct StudioExecutorProfile: Decodable, Equatable, Identifiable {
    let name: String
    let kind: String
    let destination: String?
    let url: String?

    var id: String { "\(kind):\(name)" }
    var reference: String { "\(kind):\(name)" }
}

struct StudioRelayAuthStatus: Decodable, Equatable {
    let executor: String
    let authenticated: Bool
    let refreshable: Bool
}

struct StudioLocalRunListEnvelope: Decodable, Equatable {
    let summary: String
    let result: StudioLocalRunListResult
}

struct StudioLocalRunListResult: Decodable, Equatable {
    let root: String
    let scannedDirectoryCount: Int
    let entries: [StudioLocalRunEntry]

    enum CodingKeys: String, CodingKey {
        case root
        case scannedDirectoryCount = "scanned_directory_count"
        case entries
    }
}

struct StudioLocalRunEntry: Decodable, Equatable, Identifiable {
    let id: String
    let kind: String
    let path: String
    let relativePath: String
    let status: String
    let state: String?
    let summary: String
    let createdAt: String?
    let updatedAt: String?
    let eventCount: Int?
    let artifactCount: Int?
    let diagnosticCount: Int
    let blockerCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case path
        case relativePath = "relative_path"
        case status
        case state
        case summary
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case eventCount = "event_count"
        case artifactCount = "artifact_count"
        case diagnosticCount = "diagnostic_count"
        case blockerCount = "blocker_count"
    }

    var reference: String { path }
    var displayState: String { state ?? status }
}

struct StudioRemoteRunList: Decodable, Equatable {
    let executor: String
    let jobs: [StudioRemoteRunJob]
}

struct StudioRemoteRunJob: Decodable, Equatable, Identifiable {
    let jobID: String
    let jobReference: String
    let state: String
    let executor: String
    let runDirectory: String?
    let createdAt: String?
    let updatedAt: String?
    let artifacts: [StudioRemoteRunArtifact]
    let error: String?

    var id: String { jobReference }

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case jobReference = "job_reference"
        case state
        case executor
        case runDirectory = "run_directory"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case artifacts
        case error
    }
}

struct StudioRemoteRunArtifact: Decodable, Equatable, Identifiable {
    let name: String
    let kind: String
    let path: String
    let contentType: String
    let sizeBytes: Int64

    var id: String { "\(name):\(path)" }

    enum CodingKeys: String, CodingKey {
        case name
        case kind
        case path
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
    }
}

enum StudioOperationsConfirmation: Identifiable {
    case cancel(String)
    case retry(String)

    var id: String {
        switch self {
        case .cancel(let reference): "cancel:\(reference)"
        case .retry(let reference): "retry:\(reference)"
        }
    }
}

struct StudioOperationsView: View {
    @EnvironmentObject private var controller: MereRunController

    @State private var scope: StudioOperationsScope = .local
    @State private var localRoot = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("runs", isDirectory: true).path
    @State private var maxDepth = 5
    @State private var remoteLimit = 50
    @State private var profiles: [StudioExecutorProfile] = []
    @State private var selectedExecutor = ""
    @State private var localRuns: [StudioLocalRunEntry] = []
    @State private var remoteRuns: [StudioRemoteRunJob] = []
    @State private var selectedReference: String?
    @State private var detailText = "Select a run to inspect its durable report."
    @State private var fetchDirectory = FileManager.default.urls(
        for: .downloadsDirectory,
        in: .userDomainMask
    ).first?.appendingPathComponent("mere.run fetches", isDirectory: true).path ?? "/tmp/mere-run-fetches"
    @State private var statusMessage = "Discovering executors and durable runs"
    @State private var isRefreshing = false
    @State private var busyAction: String?
    @State private var autoRefresh = true
    @State private var confirmation: StudioOperationsConfirmation?
    @State private var showProfileSetup = false
    @State private var newRelayName = "fleet"
    @State private var newRelayURL = "https://relay.mere.run"
    @State private var relayAuthenticated = false
    @State private var relayProgress = ""
    @State private var relayApprovalURL: URL?

    private var selectedLocalRun: StudioLocalRunEntry? {
        localRuns.first { $0.reference == selectedReference }
    }

    private var selectedRemoteRun: StudioRemoteRunJob? {
        remoteRuns.first { $0.jobReference == selectedReference }
    }

    private var relayProfiles: [StudioExecutorProfile] {
        profiles.filter { $0.kind == "relay" }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            controls
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            HStack(spacing: 0) {
                runList
                    .frame(minWidth: 300, idealWidth: 430, maxWidth: 430)
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                detailPane
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            await loadProfiles()
            await refresh()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                if autoRefresh, busyAction == nil {
                    await refresh(preserveDetail: true)
                }
            }
        }
        .onChange(of: scope) {
            selectedReference = nil
            detailText = "Select a run to inspect its durable report."
            Task { await refresh() }
        }
        .onChange(of: selectedExecutor) {
            guard scope == .relay else { return }
            selectedReference = nil
            Task {
                await refreshRelayAuth()
                await refresh()
            }
        }
        .alert(item: $confirmation) { pending in
            switch pending {
            case .cancel(let reference):
                Alert(
                    title: Text("Cancel this run?"),
                    message: Text(reference),
                    primaryButton: .destructive(Text("Cancel Run")) {
                        Task { await perform(["run", "cancel", reference, "--json"], label: "Cancelling") }
                    },
                    secondaryButton: .cancel()
                )
            case .retry(let reference):
                Alert(
                    title: Text("Retry this Relay run?"),
                    message: Text("Relay will submit the same immutable job bundle as a new attempt."),
                    primaryButton: .default(Text("Retry")) {
                        Task { await perform(["run", "retry", reference, "--json"], label: "Retrying") }
                    },
                    secondaryButton: .cancel()
                )
            }
        }
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            Text(statusMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            Toggle("Live", isOn: $autoRefresh)
                .toggleStyle(.switch)
                .help("Refresh the current run list every five seconds")
            Button {
                Task { await refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.mereSecondary)
            .disabled(isRefreshing)
        }
        .padding(.horizontal, MereRunTheme.Spacing.xl)
        .padding(.vertical, MereRunTheme.Spacing.sm)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Picker("Source", selection: $scope) {
                    ForEach(StudioOperationsScope.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)

                if scope == .local {
                    TextField("Run root", text: $localRoot)
                        .textFieldStyle(.roundedBorder)
                    Button("Choose…", action: chooseRoot)
                        .buttonStyle(.mereSecondary)
                    Stepper("Depth \(maxDepth)", value: $maxDepth, in: 0...20)
                        .fixedSize()
                } else {
                    Picker("Relay executor", selection: $selectedExecutor) {
                        if relayProfiles.isEmpty {
                            Text("No Relay profiles").tag("")
                        }
                        ForEach(relayProfiles) { profile in
                            Text(profile.name).tag(profile.reference)
                        }
                    }
                    .frame(minWidth: 220)
                    Button(showProfileSetup ? "Hide Setup" : "Add Relay…") {
                        showProfileSetup.toggle()
                    }
                    .buttonStyle(.mereSecondary)
                    Button {
                        Task { await signInToRelay() }
                    } label: {
                        Label(relayAuthenticated ? "Signed In" : "Sign In", systemImage: relayAuthenticated ? "checkmark.circle.fill" : "person.crop.circle.badge.checkmark")
                    }
                    .buttonStyle(.mereSecondary)
                    .disabled(selectedExecutor.isEmpty || busyAction != nil || relayAuthenticated)
                    Stepper("Limit \(remoteLimit)", value: $remoteLimit, in: 1...500)
                        .fixedSize()
                    Link("Manage fleet in Relay", destination: URL(string: "https://relay.mere.run")!)
                        .font(MereRunTheme.captionFont)
                }
            }

            if scope == .relay, showProfileSetup {
                HStack(spacing: 10) {
                    TextField("Profile name", text: $newRelayName)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 170)
                    TextField("Relay base URL", text: $newRelayURL)
                        .textFieldStyle(.roundedBorder)
                    Button("Save Profile") {
                        Task { await saveRelayProfile() }
                    }
                    .buttonStyle(.merePrimary)
                    .disabled(
                        newRelayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || newRelayURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || busyAction != nil
                    )
                }
            }

            if scope == .relay, !relayProgress.isEmpty {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Image(systemName: relayAuthenticated ? "checkmark.circle.fill" : "person.badge.key")
                        .foregroundStyle(relayAuthenticated ? MereRunTheme.green : MereRunTheme.accent)
                    Text(relayProgress)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .lineLimit(2)
                        .textSelection(.enabled)
                    if let relayApprovalURL {
                        Link("Open sign-in", destination: relayApprovalURL)
                            .font(MereRunTheme.captionFont)
                    }
                    Spacer()
                }
            }

            HStack(spacing: 8) {
                Image(systemName: scope == .relay ? "arrow.triangle.branch" : "externaldrive")
                    .foregroundStyle(MereRunTheme.accent)
                Text(
                    scope == .relay
                        ? "Studio controls your runs. Relay owns nodes, placement, scheduling policy, model distribution, and fleet telemetry."
                        : "Local discovery reads durable run folders and structured reports; it never starts or manages a node worker."
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                Spacer()
                if let busyAction {
                    ProgressView()
                        .controlSize(.small)
                    Text(busyAction)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
        }
        .padding(.horizontal, MereRunTheme.Spacing.xl)
        .padding(.vertical, MereRunTheme.Spacing.sm)
        .background(MereRunTheme.surface.opacity(0.55))
    }

    private var runList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(scope == .local ? "Durable runs" : "Relay jobs")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Text("\(scope == .local ? localRuns.count : remoteRuns.count)")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(16)

            Divider().overlay(MereRunTheme.border.opacity(0.4))

            if scope == .local, localRuns.isEmpty {
                emptyList("No durable runs found", detail: "Choose a narrower run root or increase scan depth.")
            } else if scope == .relay, relayProfiles.isEmpty {
                emptyList("No Relay executor", detail: "Create one with `mere.run executor add relay …`, then refresh.")
            } else if scope == .relay, remoteRuns.isEmpty {
                emptyList("No Relay jobs", detail: "This executor has no recent work.")
            } else {
                ScrollView {
                    LazyVStack(spacing: 5) {
                        if scope == .local {
                            ForEach(localRuns) { run in
                                runRow(
                                    reference: run.reference,
                                    title: run.relativePath,
                                    state: run.displayState,
                                    subtitle: run.summary
                                )
                            }
                        } else {
                            ForEach(remoteRuns) { run in
                                runRow(
                                    reference: run.jobReference,
                                    title: run.jobID,
                                    state: run.state,
                                    subtitle: run.error ?? "\(run.artifacts.count) artifact(s)"
                                )
                            }
                        }
                    }
                    .padding(10)
                }
            }

            Divider().overlay(MereRunTheme.border.opacity(0.4))
            Text(statusMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(2)
                .padding(12)
        }
    }

    private func emptyList(_ title: String, detail: String) -> some View {
        ContentUnavailableView(title, systemImage: "tray", description: Text(detail))
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func runRow(reference: String, title: String, state: String, subtitle: String) -> some View {
        Button {
            selectedReference = reference
            Task { await inspect(reference) }
        } label: {
            HStack(spacing: 10) {
                Circle()
                    .fill(stateColor(state))
                    .frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(title)
                            .font(.system(size: 12.5, weight: .semibold))
                            .lineLimit(1)
                        Spacer()
                        Text(state.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.system(size: 10.5, weight: .medium))
                            .foregroundStyle(stateColor(state))
                    }
                    Text(subtitle)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(2)
                }
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(selectedReference == reference ? MereRunTheme.accentSoft : MereRunTheme.surface)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var detailPane: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let reference = selectedReference {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedRemoteRun?.jobID ?? selectedLocalRun?.relativePath ?? "Run")
                            .font(MereRunTheme.titleFont)
                        Text(reference)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(MereRunTheme.textMuted)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    actionButtons(reference)
                }

                if let remote = selectedRemoteRun {
                    HStack(spacing: 10) {
                        metric("State", remote.state.capitalized)
                        metric("Artifacts", "\(remote.artifacts.count)")
                        metric("Executor", remote.executor)
                        if let updatedAt = remote.updatedAt {
                            metric("Updated", compactDate(updatedAt))
                        }
                    }
                    if !remote.artifacts.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            Text("Artifacts").font(MereRunTheme.sectionFont)
                            ForEach(remote.artifacts) { artifact in
                                HStack {
                                    Image(systemName: "doc")
                                    Text(artifact.name)
                                    Spacer()
                                    Text(ByteCountFormatter.string(fromByteCount: artifact.sizeBytes, countStyle: .file))
                                        .foregroundStyle(MereRunTheme.textMuted)
                                }
                                .font(MereRunTheme.captionFont)
                            }
                        }
                        .padding(12)
                        .merePanel()
                    }
                } else if let local = selectedLocalRun {
                    HStack(spacing: 10) {
                        metric("State", local.displayState.capitalized)
                        metric("Events", "\(local.eventCount ?? 0)")
                        metric("Artifacts", "\(local.artifactCount ?? 0)")
                        metric("Diagnostics", "\(local.diagnosticCount)")
                    }
                }

                Text("Inspection")
                    .font(MereRunTheme.sectionFont)
                ScrollView {
                    Text(detailText)
                        .font(.system(size: 11.5, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                }
                .merePanel()
            } else {
                ContentUnavailableView(
                    "Select a run",
                    systemImage: "list.bullet.rectangle.portrait",
                    description: Text("Inspect durable state and use only the controls supported by its executor.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(MereRunTheme.Spacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func actionButtons(_ reference: String) -> some View {
        HStack(spacing: 7) {
            if reference.hasPrefix("relay://") || reference.hasPrefix("ssh://") {
                Button("Fetch…") { chooseFetchDirectoryAndRun(reference) }
                    .buttonStyle(.mereSecondary)
                Button("Cancel") { confirmation = .cancel(reference) }
                    .buttonStyle(.mereSecondary)
                if reference.hasPrefix("relay://") {
                    Button("Retry") { confirmation = .retry(reference) }
                        .buttonStyle(.mereSecondary)
                }
            } else {
                Button("Reveal") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: reference)])
                }
                .buttonStyle(.mereSecondary)
            }
            Button {
                Task { await inspect(reference) }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.mereSecondary)
            .accessibilityLabel("Refresh inspection")
        }
        .disabled(busyAction != nil)
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title.uppercased())
                .font(.system(size: 9.5, weight: .semibold))
                .kerning(0.5)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 12.5, weight: .semibold))
                .lineLimit(1)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private func loadProfiles() async {
        let result = await controller.utilityCommandResult(args: ["executor", "list", "--json"])
        guard result.exitCode == 0,
              let data = StudioOperationsJSON.objectData(result.stdout),
              let decoded = try? JSONDecoder().decode(StudioExecutorProfiles.self, from: data) else {
            statusMessage = result.outputText.isEmpty ? "Executor profiles unavailable" : result.outputText
            return
        }
        profiles = decoded.profiles
        if selectedExecutor.isEmpty {
            selectedExecutor = relayProfiles.first?.reference ?? ""
        }
        await refreshRelayAuth()
    }

    private func refresh(preserveDetail: Bool = false) async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        if scope == .local {
            let result = await controller.utilityCommandResult(
                args: ["run", "list", "--root", localRoot, "--max-depth", String(maxDepth), "--json"]
            )
            guard let data = StudioOperationsJSON.objectData(result.stdout),
                  let decoded = try? JSONDecoder().decode(StudioLocalRunListEnvelope.self, from: data) else {
                localRuns = []
                statusMessage = StudioOperationsJSON.failureMessage(result)
                return
            }
            localRuns = decoded.result.entries
            statusMessage = decoded.summary
        } else {
            guard !selectedExecutor.isEmpty else {
                remoteRuns = []
                statusMessage = "Add a Relay executor profile to list remote work."
                return
            }
            let result = await controller.utilityCommandResult(
                args: [
                    "run", "list", "--executor", selectedExecutor,
                    "--limit", String(remoteLimit), "--json",
                ]
            )
            guard result.exitCode == 0,
                  let data = StudioOperationsJSON.objectData(result.stdout),
                  let decoded = try? JSONDecoder().decode(StudioRemoteRunList.self, from: data) else {
                remoteRuns = []
                statusMessage = StudioOperationsJSON.failureMessage(result)
                return
            }
            remoteRuns = decoded.jobs
            statusMessage = "\(decoded.jobs.count) recent Relay job(s)"
        }

        if preserveDetail, let selectedReference {
            await inspect(selectedReference, announce: false)
        }
    }

    private func inspect(_ reference: String, announce: Bool = true) async {
        if announce { busyAction = "Inspecting" }
        defer { if announce { busyAction = nil } }
        let result = await controller.utilityCommandResult(args: ["run", "inspect", reference, "--json"])
        detailText = result.outputText.isEmpty ? "No inspection output." : result.outputText
    }

    private func perform(_ args: [String], label: String) async {
        busyAction = label
        defer { busyAction = nil }
        let result = await controller.utilityCommandResult(args: args)
        detailText = result.outputText
        statusMessage = result.exitCode == 0 ? "\(label) completed" : "\(label) failed"
        await refresh()
    }

    private func chooseRoot() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Scan"
        if panel.runModal() == .OK, let url = panel.url {
            localRoot = url.path
            Task { await refresh() }
        }
    }

    private func saveRelayProfile() async {
        busyAction = "Saving Relay profile"
        defer { busyAction = nil }
        let name = newRelayName.trimmingCharacters(in: .whitespacesAndNewlines)
        let url = newRelayURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let result = await controller.utilityCommandResult(
            args: ["executor", "add", "relay", name, "--url", url, "--json"]
        )
        relayProgress = result.outputText
        guard result.exitCode == 0 else { return }
        await loadProfiles()
        selectedExecutor = "relay:\(name)"
        showProfileSetup = false
        statusMessage = "Saved relay:\(name). Sign in to list its jobs."
    }

    private func refreshRelayAuth() async {
        guard !selectedExecutor.isEmpty else {
            relayAuthenticated = false
            return
        }
        let result = await controller.utilityCommandResult(
            args: ["executor", "auth-status", selectedExecutor, "--json"]
        )
        guard result.exitCode == 0,
              let data = StudioOperationsJSON.objectData(result.stdout),
              let status = try? JSONDecoder().decode(StudioRelayAuthStatus.self, from: data) else {
            relayAuthenticated = false
            return
        }
        relayAuthenticated = status.authenticated || status.refreshable
    }

    private func signInToRelay() async {
        guard !selectedExecutor.isEmpty else { return }
        busyAction = "Waiting for Relay sign-in"
        relayProgress = "Starting device authorization…"
        relayApprovalURL = nil
        defer { busyAction = nil }

        let result = await controller.utilityCommandResult(
            args: ["executor", "login", selectedExecutor, "--json"],
            onOutput: { chunk in
                relayProgress = (relayProgress + "\n" + chunk)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if relayApprovalURL == nil,
                   let url = StudioOperationsJSON.firstHTTPURL(in: chunk) {
                    relayApprovalURL = url
                    NSWorkspace.shared.open(url)
                }
            }
        )
        relayProgress = result.outputText
        await refreshRelayAuth()
        if relayAuthenticated {
            statusMessage = "Signed in to \(selectedExecutor)"
            await refresh()
        }
    }

    private func chooseFetchDirectoryAndRun(_ reference: String) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Fetch"
        panel.directoryURL = URL(fileURLWithPath: fetchDirectory, isDirectory: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        fetchDirectory = url.path
        Task {
            await perform(
                ["run", "fetch", reference, "--into", url.path, "--all-artifacts", "--json"],
                label: "Fetching"
            )
            if FileManager.default.fileExists(atPath: url.path) {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        }
    }

    private func stateColor(_ state: String) -> Color {
        switch state.lowercased() {
        case "finished", "completed", "ok": MereRunTheme.green
        case "failed", "blocked": MereRunTheme.red
        case "running", "assigned": MereRunTheme.accent
        case "queued", "planned", "preflighting": MereRunTheme.yellow
        case "cancelled": MereRunTheme.textMuted
        default: MereRunTheme.textSecondary
        }
    }

    private func compactDate(_ raw: String) -> String {
        raw.replacingOccurrences(of: "T", with: " ").prefix(16).description
    }
}

enum StudioOperationsJSON {
    static func objectData(_ text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start <= end else {
            return nil
        }
        return String(text[start...end]).data(using: .utf8)
    }

    static func failureMessage(_ result: MereRunUtilityCommandResult) -> String {
        let detail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        return detail.isEmpty ? "The command did not return a compatible structured report." : detail
    }

    static func firstHTTPURL(in text: String) -> URL? {
        let delimiters = CharacterSet.whitespacesAndNewlines
            .union(CharacterSet(charactersIn: "<>()[]{}\"'"))
        for token in text.components(separatedBy: delimiters) {
            let trimmed = token.trimmingCharacters(in: CharacterSet(charactersIn: ".,;"))
            if (trimmed.hasPrefix("https://") || trimmed.hasPrefix("http://")),
               let url = URL(string: trimmed) {
                return url
            }
        }
        return nil
    }
}
