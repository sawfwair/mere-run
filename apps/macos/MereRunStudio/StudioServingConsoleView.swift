import AppKit
import SwiftUI

enum StudioServingSection: String, CaseIterable, Identifiable {
    case overview = "Overview"
    case modelPool = "Model Pool"
    case resources = "Resources"
    case traffic = "Traffic"
    case agents = "Agents & Clients"
    case activity = "Activity"
    case configuration = "Configuration"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .overview: "rectangle.3.group"
        case .modelPool: "shippingbox.and.arrow.backward"
        case .resources: "gauge.with.dots.needle.67percent"
        case .traffic: "chart.xyaxis.line"
        case .agents: "person.2.badge.gearshape"
        case .activity: "waveform.path.ecg"
        case .configuration: "slider.horizontal.3"
        }
    }
}

@MainActor
final class StudioServingMonitor: ObservableObject {
    @Published private(set) var runtime: StudioRuntimeSnapshot?
    @Published private(set) var agentStatus: StudioAgentStatus?
    @Published private(set) var isReachable = false
    @Published private(set) var isRefreshing = false
    @Published private(set) var connectionDetail = "Waiting for runtime"
    @Published private(set) var agentDetail = "Agent readiness has not been checked"
    @Published private(set) var lastUpdated: Date?
    @Published private(set) var activities: [StudioServiceActivity] = []

    private var pollingTask: Task<Void, Never>?

    func start(controller: MereRunController) {
        stop()
        pollingTask = Task { @MainActor [weak self, weak controller] in
            guard let self, let controller else { return }
            await refreshAgent(controller: controller)
            var iteration = 0
            while !Task.isCancelled {
                await refreshRuntime(controller: controller)
                if iteration > 0, iteration.isMultiple(of: 8) {
                    await refreshAgent(controller: controller)
                }
                iteration += 1
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func stop() {
        pollingTask?.cancel()
        pollingTask = nil
    }

    func refreshNow(controller: MereRunController) async {
        await refreshRuntime(controller: controller)
        await refreshAgent(controller: controller)
    }

    func refreshAgent(controller: MereRunController) async {
        let result = await controller.utilityCommandResult(args: ["agent", "status", "--json"])
        guard result.exitCode == 0,
              let data = Self.jsonObjectData(in: result.stdout) else {
            agentDetail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Agent readiness is unavailable"
                : StudioActivitySanitizer.sanitize(result.stderr)
            return
        }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            agentStatus = try decoder.decode(StudioAgentStatus.self, from: data)
            agentDetail = "Readiness checked"
        } catch {
            agentDetail = "This CLI does not expose typed agent readiness yet"
        }
    }

    func note(
        _ title: String,
        detail: String? = nil,
        level: StudioServiceActivity.Level = .info
    ) {
        append([StudioServiceActivity(level: level, title: title, detail: detail)])
    }

    private func refreshRuntime(controller: MereRunController) async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        var request = URLRequest(url: controller.runtimeURL(path: "/runtime/status"))
        request.timeoutInterval = 3
        if let authorization = controller.runtimeAuthorizationHeader {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard (200..<300).contains(code) else {
                let wasReachable = isReachable
                isReachable = false
                connectionDetail = code == 401
                    ? "Authentication failed — check the API key"
                    : "Runtime returned HTTP \(code)"
                if wasReachable {
                    append([.init(level: .warning, title: "Runtime disconnected", detail: connectionDetail)])
                }
                return
            }

            let decoded = try JSONDecoder().decode(StudioRuntimeSnapshot.self, from: data)
            let previous = runtime
            let wasReachable = isReachable
            runtime = decoded
            isReachable = true
            connectionDetail = "Connected"
            lastUpdated = Date()
            if wasReachable {
                append(StudioServiceActivityDiff.events(previous: previous, current: decoded))
            } else {
                append(StudioServiceActivityDiff.events(previous: nil, current: decoded))
            }
        } catch {
            let wasReachable = isReachable
            isReachable = false
            connectionDetail = "Runtime is not reachable"
            if wasReachable {
                append([.init(level: .warning, title: "Runtime disconnected", detail: error.localizedDescription)])
            }
        }
    }

    private func append(_ events: [StudioServiceActivity]) {
        guard !events.isEmpty else { return }
        activities.insert(contentsOf: events.reversed(), at: 0)
        if activities.count > 200 {
            activities.removeLast(activities.count - 200)
        }
    }

    private static func jsonObjectData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start <= end else {
            return nil
        }
        return String(text[start...end]).data(using: .utf8)
    }
}

struct StudioServingConsoleSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.dismiss) private var dismiss

    @StateObject private var monitor = StudioServingMonitor()
    @State private var section: StudioServingSection = .overview
    @State private var draft: CommandDraft
    @State private var serviceRequestID: UUID?
    @State private var agentRequestID: UUID?
    @State private var selectedAgentModel = ""
    @State private var agentPrompt = "Help me configure and use mere.run on this machine."
    @State private var selectedPoolID: String?
    @State private var settingsAlias = ""
    @State private var settingsTTL = ""
    @State private var settingsContext = ""
    @State private var settingsMaxTokens = ""
    @State private var settingsTemperature = ""
    @State private var settingsTopP = ""
    @State private var settingsMinP = ""
    @State private var settingsPinned = false
    @State private var operationMessage: String?
    @State private var busyOperation: String?

    init() {
        let template = CommandCatalog.template(id: .apiServe)
        _draft = State(initialValue: template?.defaultDraft() ?? CommandDraft())
    }

    private var ownedServiceID: UUID? {
        serviceRequestID ?? controller.runningRequestID(for: .apiServe)
    }

    private var ownsRunningService: Bool {
        guard let ownedServiceID else { return false }
        return controller.isRequestRunning(ownedServiceID)
    }

    private var serviceStateTitle: String {
        if ownsRunningService, monitor.isReachable { return "Running in Studio" }
        if ownsRunningService { return "Starting in Studio" }
        if monitor.isReachable { return "Connected to external server" }
        return "Server stopped"
    }

    private var safety: StudioServingSafety {
        StudioServingSafety.evaluate(host: draft.host, apiKey: draft.apiKey)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            HStack(spacing: 0) {
                navigation
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                ScrollView {
                    sectionContent
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(MereRunTheme.Spacing.xl)
                }
            }
        }
        .frame(minWidth: 1_100, idealWidth: 1_180, minHeight: 700, idealHeight: 760)
        .background(MereRunTheme.background)
        .task {
            syncDraftFromController()
            serviceRequestID = controller.runningRequestID(for: .apiServe)
            monitor.start(controller: controller)
        }
        .onDisappear { monitor.stop() }
        .onChange(of: monitor.agentStatus) { _, status in
            guard selectedAgentModel.isEmpty else { return }
            selectedAgentModel = status?.recommendedModelID
                ?? status?.models.first(where: \.startableByMereRun)?.id
                ?? ""
        }
        .onChange(of: monitor.runtime) { _, _ in
            if selectedPoolID == nil {
                selectedPoolID = monitor.runtime?.textModels.first?.id
                    ?? monitor.runtime?.sidecars?.residents.first?.modelID
            }
        }
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            ZStack {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft)
                Image(systemName: "network")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
            }
            .frame(width: 42, height: 42)

            VStack(alignment: .leading, spacing: 2) {
                Text("Serving & Agents")
                    .font(MereRunTheme.titleFont)
                Text("Operate the local API, model pool, resources, and connected tools")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }

            Spacer()

            Button {
                Task { await monitor.refreshNow(controller: controller) }
            } label: {
                Label("Refresh", systemImage: monitor.isRefreshing ? "arrow.triangle.2.circlepath" : "arrow.clockwise")
            }
            .buttonStyle(.mereSecondary)
            .disabled(monitor.isRefreshing)

            Button("Done") { dismiss() }
                .buttonStyle(.merePrimary)
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, MereRunTheme.Spacing.xl)
        .padding(.vertical, MereRunTheme.Spacing.md)
    }

    private var navigation: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(StudioServingSection.allCases) { candidate in
                Button {
                    section = candidate
                } label: {
                    HStack(spacing: 9) {
                        Image(systemName: candidate.systemImage)
                            .frame(width: 18)
                        Text(candidate.rawValue)
                        Spacer(minLength: 0)
                    }
                    .font(.system(size: 12.5, weight: candidate == section ? .semibold : .medium))
                    .foregroundStyle(candidate == section ? MereRunTheme.background : MereRunTheme.textSecondary)
                    .padding(.horizontal, 10)
                    .frame(height: 34)
                    .background {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                            .fill(candidate == section ? MereRunTheme.accent : .clear)
                    }
                }
                .buttonStyle(.plain)
            }
            Spacer()
            VStack(alignment: .leading, spacing: 5) {
                Label(serviceStateTitle, systemImage: monitor.isReachable ? "checkmark.circle.fill" : "circle")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(monitor.isReachable ? MereRunTheme.green : MereRunTheme.textMuted)
                Text(endpoint)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(2)
            }
            .padding(10)
        }
        .padding(MereRunTheme.Spacing.sm)
        .frame(width: 200)
        .background(MereRunTheme.surface.opacity(0.35))
    }

    @ViewBuilder
    private var sectionContent: some View {
        switch section {
        case .overview: overview
        case .modelPool: modelPool
        case .resources: resources
        case .traffic: traffic
        case .agents: agentsAndClients
        case .activity: activity
        case .configuration: configuration
        }
    }

    private var overview: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle("Local inference service", subtitle: "One operational surface for the API and every resident model lane")

            StudioServingCard {
                HStack(alignment: .top, spacing: MereRunTheme.Spacing.lg) {
                    statusOrb
                    VStack(alignment: .leading, spacing: 5) {
                        Text(serviceStateTitle)
                            .font(.system(size: 18, weight: .semibold))
                        Text(monitor.connectionDetail)
                            .font(MereRunTheme.bodyFont)
                            .foregroundStyle(MereRunTheme.textSecondary)
                        HStack(spacing: 7) {
                            Text(endpoint)
                                .font(.system(size: 12, design: .monospaced))
                                .textSelection(.enabled)
                            copyButton(endpoint)
                        }
                    }
                    Spacer()
                    serviceControls
                }
            }

            if safety == .exposedWithoutAuthentication {
                warningBanner(
                    "Authentication required",
                    detail: "\(draft.host) is reachable beyond this Mac. Add an API key before starting the server."
                )
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 12) {
                metricCard("Text models", value: "\(monitor.runtime?.loadedTextModels.count ?? 0)", detail: "resident")
                metricCard("Sidecars", value: "\(monitor.runtime?.loadedSidecars.count ?? 0)", detail: "resident")
                metricCard(
                    "Requests",
                    value: "\(monitor.runtime?.admission?.activeRequests ?? monitor.runtime?.activeRequests ?? 0)",
                    detail: "\(monitor.runtime?.admission?.queuedRequests ?? 0) queued"
                )
                metricCard(
                    "Memory pressure",
                    value: (monitor.runtime?.memory?.pressure ?? "Unavailable").capitalized,
                    detail: monitor.runtime?.memory?.guardTier.map { "\($0) guard" } ?? "older runtime"
                )
            }

            HStack(alignment: .top, spacing: MereRunTheme.Spacing.md) {
                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                        cardTitle("Resident lanes", systemImage: "square.stack.3d.up")
                        if monitor.runtime == nil {
                            unavailable("Start or connect to a server to inspect the model pool.")
                        } else {
                            ForEach(monitor.runtime?.loadedTextModels ?? []) { model in
                                compactLane(model.id, state: model.state, active: model.activeRequests)
                            }
                            ForEach(monitor.runtime?.loadedSidecars ?? []) { sidecar in
                                compactLane(sidecar.kind.capitalized, state: sidecar.state, active: sidecar.activeRequests)
                            }
                            if monitor.runtime?.loadedTextModels.isEmpty != false,
                               monitor.runtime?.loadedSidecars.isEmpty != false {
                                unavailable("No models are resident. Models load when requested or from Model Pool.")
                            }
                        }
                    }
                }
                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                        cardTitle("Fast actions", systemImage: "bolt")
                        Button("Open Model Pool") { section = .modelPool }
                            .buttonStyle(.mereSecondary)
                        Button("Set up an agent or client") { section = .agents }
                            .buttonStyle(.mereSecondary)
                        Button("Review live activity") { section = .activity }
                            .buttonStyle(.mereSecondary)
                        Button("Configure server") { section = .configuration }
                            .buttonStyle(.mereSecondary)
                    }
                }
            }
        }
    }

    private var modelPool: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle(
                "Model Pool",
                subtitle: "Text and modality sidecars share one memory-aware runtime"
            )

            if let runtime = monitor.runtime {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                    poolHeader("Text models", detail: "\(runtime.loadedTextModels.count) loaded")
                    ForEach(runtime.textModels) { model in
                        textModelRow(model)
                    }

                    poolHeader(
                        "Sidecar services",
                        detail: "\(runtime.sidecars?.loadedCount ?? 0) loaded · \(runtime.sidecars?.defaultIdleTTLSeconds ?? 0)s default TTL"
                    )
                    ForEach(runtime.sidecars?.residents ?? []) { sidecar in
                        sidecarRow(sidecar)
                    }
                }

                if selectedPoolID != nil {
                    runtimeSettings
                }
            } else {
                StudioServingCard {
                    unavailable("The runtime pool will appear here when a server is reachable.")
                }
            }
        }
    }

    private var resources: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle(
                "Resources",
                subtitle: "Host memory, process CPU, Metal allocation, and thermal state"
            )

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 3), spacing: 12) {
                metricCard("Runtime footprint", value: StudioServingFormat.bytes(monitor.runtime?.memory?.currentBytes), detail: "unified memory")
                metricCard("Resident process", value: StudioServingFormat.bytes(monitor.runtime?.memory?.residentBytes), detail: "host resident set")
                metricCard("Available memory", value: StudioServingFormat.bytes(monitor.runtime?.memory?.availableBytes), detail: "host estimate")
                metricCard("Soft ceiling", value: StudioServingFormat.bytes(monitor.runtime?.memory?.softLimitBytes), detail: monitor.runtime?.memory?.guardTier ?? "guard unavailable")
                metricCard("Hard ceiling", value: StudioServingFormat.bytes(monitor.runtime?.memory?.hardLimitBytes), detail: monitor.runtime?.memory?.pressure ?? "pressure unavailable")
                metricCard("Process CPU", value: StudioServingFormat.number(monitor.runtime?.process?.cpuPercent, suffix: "%"), detail: "sampled process usage")
                metricCard("Metal allocation", value: StudioServingFormat.bytes(monitor.runtime?.process?.metalCurrentAllocatedBytes), detail: "allocated, not utilization")
                metricCard("Metal working set", value: StudioServingFormat.bytes(monitor.runtime?.process?.metalRecommendedMaxWorkingSetBytes), detail: monitor.runtime?.process?.metalDeviceName ?? "unavailable")
                metricCard("Thermal state", value: (monitor.runtime?.process?.thermalState ?? "Unavailable").capitalized, detail: monitor.runtime?.process?.lowPowerModeEnabled == true ? "Low Power Mode" : "normal power mode")
            }

            StudioServingCard {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                    cardTitle("Memory guard", systemImage: "shield.lefthalf.filled")
                    resourceProgress(
                        current: monitor.runtime?.memory?.currentBytes,
                        limit: monitor.runtime?.memory?.hardLimitBytes
                    )
                    keyValue("Physical memory", StudioServingFormat.bytes(monitor.runtime?.memory?.physicalBytes))
                    keyValue("Configured ceiling", StudioServingFormat.bytes(monitor.runtime?.memory?.ceilingBytes))
                    keyValue("Active models", "\(monitor.runtime?.memory?.activeModelCount ?? 0)")
                    keyValue("Server uptime", StudioServingFormat.duration(monitor.runtime?.process?.uptimeSeconds))
                    Text("macOS does not expose a stable public GPU-utilization percentage. Studio reports real Metal allocation and the device-recommended working set instead.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
        }
    }

    private var traffic: some View {
        let admission = monitor.runtime?.admission
        let traffic = monitor.runtime?.benchmarkStats
        let prefix = monitor.runtime?.cacheStats?.prefixKVReuse
        let batching = monitor.runtime?.cacheStats?.decodeBatching

        return VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle(
                "Traffic",
                subtitle: "Observed requests and runtime efficiency—not a synthetic benchmark"
            )

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 12) {
                metricCard("Active", value: "\(admission?.activeRequests ?? 0)", detail: "\(admission?.maxActiveRequests ?? 0) capacity")
                metricCard("Queued", value: "\(admission?.queuedRequests ?? 0)", detail: admission?.admissionPaused == true ? "admission paused" : "FIFO admission")
                metricCard("Completed", value: "\(traffic?.completedRequests ?? admission?.totalCompletedRequests ?? 0)", detail: "\(traffic?.failedRequests ?? 0) failed")
                metricCard("Generated", value: "\(traffic?.generatedTokens ?? 0)", detail: "tokens")
                metricCard("Decode rate", value: StudioServingFormat.number(traffic?.decodeTokensPerSecond, suffix: " tok/s"), detail: "observed average")
                metricCard("Total latency", value: StudioServingFormat.number(traffic?.averageTotalSeconds, suffix: "s"), detail: "average")
                metricCard("Prefix reuse", value: prefix?.hitRate.map { String(format: "%.0f%%", $0 * 100) } ?? "Unavailable", detail: "\(prefix?.reusedTokens ?? 0) tokens reused")
                metricCard("Max batch", value: "\(batching?.maxBatchSize ?? 0)", detail: "\(batching?.totalBatchedRows ?? 0) rows batched")
            }

            HStack(alignment: .top, spacing: MereRunTheme.Spacing.md) {
                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                        cardTitle("Request lifecycle", systemImage: "arrow.triangle.branch")
                        keyValue("Admitted", "\(admission?.totalAdmittedRequests ?? 0)")
                        keyValue("Completed", "\(admission?.totalCompletedRequests ?? 0)")
                        keyValue("Cancelled", "\(admission?.totalCancelledRequests ?? 0)")
                        keyValue("Load", StudioServingFormat.number(traffic?.averageLoadSeconds, suffix: "s"))
                        keyValue("Prefill", StudioServingFormat.number(traffic?.averagePrefillSeconds, suffix: "s"))
                        keyValue("Decode", StudioServingFormat.number(traffic?.averageDecodeSeconds, suffix: "s"))
                    }
                }
                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                        cardTitle("Runtime efficiency", systemImage: "speedometer")
                        keyValue("Prefix cache entries", "\(prefix?.entries ?? 0) / \(prefix?.maxEntries ?? 0)")
                        keyValue("Cache hits / misses", "\(prefix?.hits ?? 0) / \(prefix?.misses ?? 0)")
                        keyValue("Batched decode steps", "\(batching?.batchedDecodeSteps ?? 0)")
                        keyValue("Variable-position steps", "\(batching?.variablePositionBatchedSteps ?? 0)")
                        keyValue("Single decode steps", "\(batching?.singleDecodeSteps ?? 0)")
                        keyValue("MTP accepted tokens", "\(mtpAcceptedTokens)")
                    }
                }
            }
        }
    }

    private var agentsAndClients: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle(
                "Agents & Clients",
                subtitle: "Install Pi, connect coding agents, or use any OpenAI-compatible client"
            )

            HStack(alignment: .top, spacing: MereRunTheme.Spacing.md) {
                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                        cardTitle("Pi agent", systemImage: "person.crop.circle.badge.gearshape")
                        readinessRow(
                            "Pi installed",
                            ready: monitor.agentStatus?.pi.installed == true,
                            detail: monitor.agentStatus?.pi.version ?? monitor.agentStatus?.pi.path
                        )
                        readinessRow(
                            "Provider configured",
                            ready: monitor.agentStatus?.provider.configured == true,
                            detail: monitor.agentStatus?.provider.modelID
                        )
                        readinessRow(
                            "API reachable",
                            ready: monitor.isReachable,
                            detail: endpoint
                        )

                        Picker("Agent model", selection: $selectedAgentModel) {
                            ForEach(monitor.agentStatus?.models.filter(\.startableByMereRun) ?? []) { model in
                                Text("\(model.displayName)\(model.installed ? "" : " · download needed")")
                                    .tag(model.id)
                            }
                        }

                        TextField("Opening request", text: $agentPrompt, axis: .vertical)
                            .lineLimit(2...4)
                            .mereField()

                        HStack {
                            Button("Install Pi") { installPi() }
                                .buttonStyle(.mereSecondary)
                                .disabled(monitor.agentStatus?.pi.autoInstallSupported == false)
                            Button("Configure provider") { configurePi() }
                                .buttonStyle(.mereSecondary)
                                .disabled(selectedAgentModel.isEmpty)
                            Button("Start session") { startAgent() }
                                .buttonStyle(.merePrimary)
                                .disabled(selectedAgentModel.isEmpty)
                        }
                    }
                }

                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                        cardTitle("Machine fit", systemImage: "memorychip")
                        keyValue("Processor", monitor.agentStatus?.machine.processor ?? "Unavailable")
                        keyValue("Unified memory", monitor.agentStatus.map { "\($0.machine.unifiedMemoryGB) GB" } ?? "Unavailable")
                        keyValue("Recommended", monitor.agentStatus?.recommendedModelID ?? "Unavailable")
                        if let selected = selectedAgent {
                            Divider().overlay(MereRunTheme.border.opacity(0.5))
                            Text(selected.summary)
                                .font(MereRunTheme.bodyFont)
                            keyValue("Serving engine", selected.servingEngine)
                            keyValue("Minimum memory", "\(selected.minimumUnifiedMemoryGB) GB")
                            keyValue("Recommended memory", "\(selected.recommendedUnifiedMemoryGB) GB")
                            if let reason = selected.reason {
                                Text(reason)
                                    .font(MereRunTheme.captionFont)
                                    .foregroundStyle(MereRunTheme.textMuted)
                            }
                        }
                    }
                }
            }

            clientConnections

            if let agentRequestID {
                StudioServingCard {
                    VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                        cardTitle("Current agent task", systemImage: "terminal")
                        Text(agentOutput(for: agentRequestID))
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(MereRunTheme.textSecondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, minHeight: 90, alignment: .topLeading)
                    }
                }
            }
        }
    }

    private var clientConnections: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
            poolHeader("Connect clients", detail: "OpenAI-compatible \(endpoint)")
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                clientCard(
                    title: "OpenAI SDKs",
                    systemImage: "curlybraces",
                    detail: "Set base URL to \(endpoint)/v1, API key to your configured key, and model to \(selectedClientModel).",
                    copy: "OPENAI_BASE_URL=\(endpoint)/v1\nOPENAI_API_KEY=$MERERUN_API_KEY\nOPENAI_MODEL=\(selectedClientModel)"
                )
                clientCard(
                    title: "curl",
                    systemImage: "terminal",
                    detail: "Test the chat endpoint without exposing request contents in Studio activity.",
                    copy: curlCommand
                )
                clientCard(
                    title: "Codex / Claude BYOA",
                    systemImage: "chevron.left.forwardslash.chevron.right",
                    detail: "Use the guided BYOA setup, then point the client at the local OpenAI-compatible endpoint.",
                    copy: "mere.run setup --mode byoa --host \(controller.runtimeHost) --port \(controller.runtimePort)"
                )
                clientCard(
                    title: "Open WebUI",
                    systemImage: "globe",
                    detail: "Launch the companion UI with this server, model, vision, image, speech, and embedding defaults.",
                    copy: "mere.run open-webui --host \(controller.runtimeHost) --port \(controller.runtimePort) --skip-server"
                ) {
                    launchOpenWebUI()
                }
            }
        }
    }

    private var activity: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle(
                "Activity",
                subtitle: "Sanitized lifecycle, pool, queue, pressure, and failure events"
            )

            if monitor.activities.isEmpty {
                StudioServingCard {
                    unavailable("No service activity yet. Request prompts and message bodies are never recorded here.")
                }
            } else {
                StudioServingCard {
                    VStack(spacing: 0) {
                        ForEach(monitor.activities) { event in
                            HStack(alignment: .top, spacing: MereRunTheme.Spacing.sm) {
                                Circle()
                                    .fill(activityColor(event.level))
                                    .frame(width: 8, height: 8)
                                    .padding(.top, 5)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(event.title)
                                        .font(.system(size: 12.5, weight: .semibold))
                                    if let detail = event.detail {
                                        Text(detail)
                                            .font(MereRunTheme.captionFont)
                                            .foregroundStyle(MereRunTheme.textMuted)
                                            .textSelection(.enabled)
                                    }
                                }
                                Spacer()
                                Text(event.date, style: .time)
                                    .font(.system(size: 10.5, design: .monospaced))
                                    .foregroundStyle(MereRunTheme.textMuted)
                            }
                            .padding(.vertical, 9)
                            if event.id != monitor.activities.last?.id {
                                Divider().overlay(MereRunTheme.border.opacity(0.45))
                            }
                        }
                    }
                }
            }

            Text("Privacy: this feed derives only from runtime counters, lifecycle state, sanitized errors, and operator actions. Prompts, messages, audio, images, and generated content are excluded.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private var configuration: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
            sectionTitle(
                "Configuration",
                subtitle: "Preflight, secure, and launch the app-owned API service"
            )

            StudioServingCard {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                    cardTitle("Connection & safety", systemImage: "network.badge.shield.half.filled")
                    HStack {
                        TextField("Host", text: $draft.host)
                            .mereField()
                        TextField("Port", value: $draft.port, format: .number)
                            .mereField()
                            .frame(width: 110)
                    }
                    SecureField("API key (required for LAN)", text: $draft.apiKey)
                        .mereField()
                    safetySummary
                    HStack {
                        Button("Apply & reconnect") {
                            applyConnection()
                            monitor.note("Runtime endpoint changed", detail: endpoint)
                            monitor.start(controller: controller)
                        }
                        .buttonStyle(.mereSecondary)
                        Spacer()
                        Button("Run preflight") { runPreflight() }
                            .buttonStyle(.mereSecondary)
                        Button(ownsRunningService ? "Restart server" : "Start server") {
                            if ownsRunningService {
                                restartService()
                            } else {
                                startService()
                            }
                        }
                        .buttonStyle(.merePrimary)
                        .disabled(safety == .exposedWithoutAuthentication)
                    }
                }
            }

            StudioServingCard {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                    cardTitle("Runtime defaults", systemImage: "cpu")
                    Picker("Engine", selection: $draft.engine) {
                        Text("Gemma 4").tag("text-chat-gemma4")
                        Text("Qwen 3.6").tag("text-chat-q36")
                        Text("Klein").tag("text-chat-klein")
                        Text("Code").tag("text-code")
                        Text("Laguna S 2.1").tag("text-chat-laguna")
                        Text("LFM2").tag("text-chat-lfm2")
                        Text("DeepSeek V4").tag("text-chat-deepseek-v4-flash")
                    }
                    TextField("Default model id", text: $draft.model)
                        .mereField()
                    TextField("Default adapter id or LoRA path", text: $draft.apiLoRA)
                        .mereField()
                    HStack {
                        integerField("Requests / min", value: $draft.apiRateLimitPerMinute)
                        integerField("Concurrent", value: $draft.apiMaxActiveRequests)
                        integerField("Context tokens", value: $draft.contextSize)
                    }
                    Picker("Memory guard", selection: $draft.apiMemoryGuard) {
                        ForEach(["off", "safe", "balanced", "aggressive", "custom"], id: \.self) {
                            Text($0.capitalized).tag($0)
                        }
                    }
                    if draft.apiMemoryGuard == "custom" {
                        TextField("Custom ceiling (GiB)", text: $draft.apiMemoryGuardCustomCeilingGB)
                            .mereField()
                    }
                }
            }

            StudioServingCard {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                    cardTitle("KV cache", systemImage: "memorychip")
                    HStack {
                        integerField("KV bits", value: $draft.kvBits)
                        TextField("Scheme", text: $draft.kvQuantScheme)
                            .mereField()
                        integerField("Group size", value: $draft.kvGroupSize)
                        integerField("Quantize after", value: $draft.quantizedKVStart)
                    }
                }
            }

            if let operationMessage {
                Text(operationMessage)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
            }
        }
    }

    private var serviceControls: some View {
        HStack(spacing: 8) {
            if ownsRunningService {
                Button("Stop") { stopService() }
                    .buttonStyle(.mereSecondary)
                Button("Restart") { restartService() }
                    .buttonStyle(.mereSecondary)
            } else if monitor.isReachable {
                Button("Reconnect") {
                    monitor.start(controller: controller)
                }
                .buttonStyle(.mereSecondary)
            } else {
                Button("Preflight") { runPreflight() }
                    .buttonStyle(.mereSecondary)
                Button("Start") { startService() }
                    .buttonStyle(.merePrimary)
                    .disabled(safety == .exposedWithoutAuthentication)
            }
        }
    }

    private var statusOrb: some View {
        ZStack {
            Circle()
                .fill((monitor.isReachable ? MereRunTheme.green : MereRunTheme.textMuted).opacity(0.14))
            Circle()
                .fill(monitor.isReachable ? MereRunTheme.green : MereRunTheme.textMuted)
                .frame(width: 13, height: 13)
        }
        .frame(width: 48, height: 48)
    }

    private var runtimeSettings: some View {
        StudioServingCard {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                cardTitle("Selected runtime settings", systemImage: "slider.horizontal.3")
                Text(selectedPoolID ?? "")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                HStack {
                    TextField("Alias", text: $settingsAlias).mereField()
                    TextField("TTL seconds", text: $settingsTTL).mereField()
                    Toggle("Pinned", isOn: $settingsPinned)
                }
                if selectedTextModel != nil {
                    HStack {
                        TextField("Context", text: $settingsContext).mereField()
                        TextField("Max tokens", text: $settingsMaxTokens).mereField()
                        TextField("Temperature", text: $settingsTemperature).mereField()
                        TextField("Top P", text: $settingsTopP).mereField()
                        TextField("Min P", text: $settingsMinP).mereField()
                    }
                } else {
                    Text("Sidecars use lifecycle settings (pin and TTL); generation settings remain request-scoped.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                HStack {
                    Spacer()
                    Button("Save settings") { saveRuntimeSettings() }
                        .buttonStyle(.merePrimary)
                        .disabled(selectedRuntimeModelID == nil || busyOperation != nil)
                }
            }
        }
    }

    private func textModelRow(_ model: StudioRuntimeModel) -> some View {
        StudioServingCard {
            HStack(spacing: MereRunTheme.Spacing.md) {
                stateIcon(model.state)
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(model.id)
                            .font(.system(size: 12.5, weight: .semibold, design: .monospaced))
                        if model.id == monitor.runtime?.defaultModel {
                            badge("Default")
                        }
                        if model.pinned { badge("Pinned") }
                    }
                    Text([
                        model.engine,
                        model.alias.map { "alias \($0)" },
                        model.ttlSeconds.map { "TTL \($0)s" },
                        "\(model.activeRequests) active",
                    ].compactMap { $0 }.joined(separator: " · "))
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    if let error = model.lastError {
                        Text(StudioActivitySanitizer.sanitize(error))
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.red)
                    }
                }
                Spacer()
                Text(model.state)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(stateColor(model.state))
                Button("Settings") { select(model) }
                    .buttonStyle(.mereSecondary)
                Button(model.loaded ? "Unload" : "Load") {
                    runtimeAction(model: model, action: model.loaded ? "unload" : "load")
                }
                .buttonStyle(.mereSecondary)
                .disabled(busyOperation != nil || (model.activeRequests > 0 && model.loaded))
            }
        }
    }

    private func sidecarRow(_ sidecar: StudioRuntimeSidecar) -> some View {
        StudioServingCard {
            HStack(spacing: MereRunTheme.Spacing.md) {
                stateIcon(sidecar.state)
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(sidecar.kind.capitalized)
                            .font(.system(size: 12.5, weight: .semibold))
                        if sidecar.pinned { badge("Pinned") }
                    }
                    Text("\(sidecar.displayModel) · \(sidecar.activeRequests) active · \(sidecar.queuedRequests) queued · TTL \(sidecar.ttlSeconds)s")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    if let reason = sidecar.lastEvictionReason {
                        Text("Last eviction: \(reason.replacingOccurrences(of: "_", with: " "))")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
                Spacer()
                Text(sidecar.state)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(stateColor(sidecar.state))
                Button("Lifecycle") { select(sidecar) }
                    .buttonStyle(.mereSecondary)
                    .disabled(sidecar.modelID == nil)
            }
        }
    }

    private var safetySummary: some View {
        Group {
            switch safety {
            case .loopback:
                Label("Loopback only — accessible from this Mac", systemImage: "lock.fill")
                    .foregroundStyle(MereRunTheme.green)
            case .protectedLAN:
                Label("LAN-visible and protected by bearer authentication", systemImage: "network.badge.shield.half.filled")
                    .foregroundStyle(MereRunTheme.green)
            case .exposedWithoutAuthentication:
                Label("Unsafe: non-loopback servers require an API key", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(MereRunTheme.red)
            }
        }
        .font(MereRunTheme.captionFont)
    }

    private var selectedTextModel: StudioRuntimeModel? {
        monitor.runtime?.textModels.first { $0.id == selectedPoolID }
    }

    private var selectedSidecar: StudioRuntimeSidecar? {
        monitor.runtime?.sidecars?.residents.first {
            $0.modelID == selectedPoolID || ($0.modelID == nil && $0.kind == selectedPoolID)
        }
    }

    private var selectedRuntimeModelID: String? {
        selectedTextModel?.id ?? selectedSidecar?.modelID
    }

    private var selectedAgent: StudioAgentModel? {
        monitor.agentStatus?.models.first { $0.id == selectedAgentModel }
    }

    private var selectedClientModel: String {
        monitor.runtime?.defaultModel
            ?? monitor.runtime?.loadedTextModels.first?.id
            ?? draft.model
    }

    private var endpoint: String {
        "http://\(controller.runtimeHost):\(controller.runtimePort)"
    }

    private var curlCommand: String {
        let authorization = controller.runtimeAPIKey.isBlank
            ? ""
            : " -H 'Authorization: Bearer $MERERUN_API_KEY'"
        return "curl \(endpoint)/v1/chat/completions\(authorization) -H 'Content-Type: application/json' -d '{\"model\":\"\(selectedClientModel)\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
    }

    private var mtpAcceptedTokens: Int {
        monitor.runtime?.textModels.reduce(0) { $0 + ($1.mtp?.acceptedTokens ?? 0) } ?? 0
    }

    private func syncDraftFromController() {
        draft.host = controller.runtimeHost
        draft.port = controller.runtimePort
        draft.apiKey = controller.runtimeAPIKey
    }

    private func applyConnection() {
        controller.runtimeHost = draft.host.trimmingCharacters(in: .whitespacesAndNewlines)
        controller.runtimePort = min(65_535, max(1, draft.port))
        controller.runtimeAPIKey = draft.apiKey
    }

    private func startService() {
        guard safety != .exposedWithoutAuthentication else {
            operationMessage = "Add an API key before exposing the server beyond loopback."
            return
        }
        applyConnection()
        draft.prompt = endpoint
        serviceRequestID = StudioSpecialistRunner.submit(
            templateID: .apiServe,
            mode: .chat,
            draft: draft,
            controller: controller,
            library: library
        )
        monitor.note("API server start requested", detail: endpoint)
        operationMessage = "Starting API server…"
    }

    private func stopService() {
        guard let requestID = ownedServiceID, controller.cancel(requestID: requestID) else {
            operationMessage = "This server was started outside Studio and cannot be stopped here."
            return
        }
        monitor.note("API server stop requested", detail: endpoint)
        operationMessage = "Stopping API server…"
    }

    private func restartService() {
        guard let requestID = ownedServiceID, controller.cancel(requestID: requestID) else {
            operationMessage = "Only an app-owned server can be restarted."
            return
        }
        monitor.note("API server restart requested", detail: endpoint)
        Task { @MainActor in
            for _ in 0..<40 {
                if !controller.isRequestRunning(requestID) { break }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            serviceRequestID = nil
            startService()
        }
    }

    private func runPreflight() {
        guard safety != .exposedWithoutAuthentication else {
            operationMessage = "Add an API key before preflighting a LAN-visible server."
            return
        }
        applyConnection()
        operationMessage = "Running server preflight…"
        Task { @MainActor in
            var preflight = draft
            preflight.preflight = true
            preflight.json = true
            guard let template = CommandCatalog.template(id: .apiServe) else { return }
            let result = await controller.utilityCommandResult(
                args: template.arguments(from: preflight),
                environmentOverrides: CommandLaunchEnvironment.overrides(templateID: .apiServe, draft: preflight)
            )
            operationMessage = result.exitCode == 0
                ? "Preflight passed. The server can start with this configuration."
                : StudioActivitySanitizer.sanitize(result.outputText)
            monitor.note(
                result.exitCode == 0 ? "Server preflight passed" : "Server preflight failed",
                detail: result.exitCode == 0 ? endpoint : result.outputText,
                level: result.exitCode == 0 ? .success : .error
            )
        }
    }

    private func runtimeAction(model: StudioRuntimeModel, action: String) {
        busyOperation = "\(action):\(model.id)"
        Task { @MainActor in
            let encoded = model.id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model.id
            var request = URLRequest(url: controller.runtimeURL(path: "/runtime/models/\(encoded)/\(action)"))
            request.httpMethod = "POST"
            request.timeoutInterval = 60
            if let authorization = controller.runtimeAuthorizationHeader {
                request.setValue(authorization, forHTTPHeaderField: "Authorization")
            }
            do {
                let (data, response) = try await URLSession.shared.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                guard (200..<300).contains(code) else {
                    throw StudioServingError.operationFailed(
                        String(data: data, encoding: .utf8) ?? "HTTP \(code)"
                    )
                }
                operationMessage = "\(model.id) \(action) succeeded."
                monitor.note(
                    action == "load" ? "Model load requested" : "Model unload requested",
                    detail: model.id,
                    level: .success
                )
                await monitor.refreshNow(controller: controller)
            } catch {
                operationMessage = StudioActivitySanitizer.sanitize(error.localizedDescription)
                monitor.note("Model \(action) failed", detail: error.localizedDescription, level: .error)
            }
            busyOperation = nil
        }
    }

    private func select(_ model: StudioRuntimeModel) {
        selectedPoolID = model.id
        settingsAlias = model.alias ?? ""
        settingsTTL = model.ttlSeconds.map(String.init) ?? ""
        settingsContext = model.maxContextTokens.map(String.init) ?? ""
        settingsMaxTokens = model.maxTokens.map(String.init) ?? ""
        settingsTemperature = model.temperature.map { String($0) } ?? ""
        settingsTopP = model.topP.map { String($0) } ?? ""
        settingsMinP = model.minP.map { String($0) } ?? ""
        settingsPinned = model.pinned
    }

    private func select(_ sidecar: StudioRuntimeSidecar) {
        selectedPoolID = sidecar.modelID ?? sidecar.kind
        settingsAlias = ""
        settingsTTL = String(sidecar.ttlSeconds)
        settingsContext = ""
        settingsMaxTokens = ""
        settingsTemperature = ""
        settingsTopP = ""
        settingsMinP = ""
        settingsPinned = sidecar.pinned
    }

    private func saveRuntimeSettings() {
        guard let modelID = selectedRuntimeModelID else { return }
        busyOperation = "settings:\(modelID)"
        Task { @MainActor in
            var args = ["model", "runtime", "set", modelID, settingsPinned ? "--pinned" : "--unpinned"]
            appendSetting(settingsTTL, set: "--ttl-seconds", clear: "--clear-ttl", to: &args)
            if selectedTextModel != nil {
                appendSetting(settingsAlias, set: "--alias", clear: "--clear-alias", to: &args)
                appendSetting(settingsContext, set: "--max-context-tokens", clear: "--clear-max-context-tokens", to: &args)
                appendSetting(settingsMaxTokens, set: "--max-tokens", clear: "--clear-max-tokens", to: &args)
                appendSetting(settingsTemperature, set: "--temperature", clear: "--clear-temperature", to: &args)
                appendSetting(settingsTopP, set: "--top-p", clear: "--clear-top-p", to: &args)
                appendSetting(settingsMinP, set: "--min-p", clear: "--clear-min-p", to: &args)
            }
            let result = await controller.utilityCommandResult(args: args)
            operationMessage = result.exitCode == 0
                ? "Saved runtime settings for \(modelID)."
                : StudioActivitySanitizer.sanitize(result.outputText)
            monitor.note(
                result.exitCode == 0 ? "Runtime settings saved" : "Runtime settings failed",
                detail: modelID,
                level: result.exitCode == 0 ? .success : .error
            )
            busyOperation = nil
            await monitor.refreshNow(controller: controller)
        }
    }

    private func appendSetting(_ value: String, set: String, clear: String, to args: inout [String]) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        args += trimmed.isEmpty ? [clear] : [set, trimmed]
    }

    private func installPi() {
        guard let template = CommandCatalog.template(id: .agentInstallPi) else { return }
        agentRequestID = StudioSpecialistRunner.submit(
            templateID: template.id,
            mode: .chat,
            draft: template.defaultDraft(),
            controller: controller,
            library: library
        )
        monitor.note("Pi installation submitted")
    }

    private func configurePi() {
        guard let template = CommandCatalog.template(id: .agentOnboard) else { return }
        var agentDraft = template.defaultDraft()
        agentDraft.stream = true
        agentDraft.model = selectedAgentModel
        agentDraft.host = controller.runtimeHost
        agentDraft.port = controller.runtimePort
        agentRequestID = StudioSpecialistRunner.submit(
            templateID: template.id,
            mode: .chat,
            draft: agentDraft,
            controller: controller,
            library: library
        )
        monitor.note("Pi provider configuration submitted", detail: selectedAgentModel)
    }

    private func startAgent() {
        guard let template = CommandCatalog.template(id: .agentStart) else { return }
        var agentDraft = template.defaultDraft()
        agentDraft.model = selectedAgentModel
        agentDraft.prompt = agentPrompt
        agentDraft.host = controller.runtimeHost
        agentDraft.port = controller.runtimePort
        agentDraft.stream = monitor.isReachable
        agentRequestID = StudioSpecialistRunner.submit(
            templateID: template.id,
            mode: .chat,
            draft: agentDraft,
            controller: controller,
            library: library
        )
        monitor.note("Agent session submitted", detail: selectedAgentModel)
    }

    private func launchOpenWebUI() {
        guard let template = CommandCatalog.template(id: .openWebui) else { return }
        var webDraft = template.defaultDraft()
        webDraft.host = controller.runtimeHost
        webDraft.port = controller.runtimePort
        webDraft.apiKey = controller.runtimeAPIKey
        webDraft.model = selectedClientModel
        webDraft.openWebUISkipServer = monitor.isReachable
        agentRequestID = StudioSpecialistRunner.submit(
            templateID: template.id,
            mode: .chat,
            draft: webDraft,
            controller: controller,
            library: library
        )
        monitor.note("Open WebUI setup submitted")
    }

    private func agentOutput(for requestID: UUID) -> String {
        let logs = controller.logs(for: requestID).suffix(50).map {
            "[\($0.stream.label)] \(StudioActivitySanitizer.sanitize($0.text))"
        }
        return logs.isEmpty ? "Waiting for output…" : logs.joined(separator: "\n")
    }

    private func sectionTitle(_ title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(MereRunTheme.displaySmallFont)
            Text(subtitle)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private func cardTitle(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(MereRunTheme.sectionFont)
            .foregroundStyle(MereRunTheme.textPrimary)
    }

    private func metricCard(_ title: String, value: String, detail: String) -> some View {
        StudioServingCard {
            VStack(alignment: .leading, spacing: 5) {
                Text(title.uppercased())
                    .font(.system(size: 9.5, weight: .semibold))
                    .kerning(0.6)
                    .foregroundStyle(MereRunTheme.textMuted)
                Text(value)
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Text(detail)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
        }
    }

    private func poolHeader(_ title: String, detail: String) -> some View {
        HStack {
            Text(title)
                .font(MereRunTheme.sectionFont)
            Spacer()
            Text(detail)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .padding(.top, 4)
    }

    private func compactLane(_ name: String, state: String, active: Int) -> some View {
        HStack {
            Circle()
                .fill(stateColor(state))
                .frame(width: 7, height: 7)
            Text(name)
                .font(.system(size: 11.5, weight: .medium))
                .lineLimit(1)
            Spacer()
            Text(active > 0 ? "\(active) active" : state)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private func keyValue(_ key: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(key)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            Text(value)
                .font(.system(size: 11.5, weight: .medium))
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
    }

    private func warningBanner(_ title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(MereRunTheme.red)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(MereRunTheme.sectionFont)
                Text(detail).font(MereRunTheme.captionFont)
            }
        }
        .padding(MereRunTheme.Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(MereRunTheme.red.opacity(0.09), in: RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
    }

    private func unavailable(_ text: String) -> some View {
        Text(text)
            .font(MereRunTheme.bodyFont)
            .foregroundStyle(MereRunTheme.textMuted)
            .frame(maxWidth: .infinity, minHeight: 64, alignment: .center)
    }

    private func stateIcon(_ state: String) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(stateColor(state).opacity(0.12))
            Image(systemName: state == "Active" ? "bolt.fill" : "shippingbox")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(stateColor(state))
        }
        .frame(width: 36, height: 36)
    }

    private func stateColor(_ state: String) -> Color {
        switch state.lowercased() {
        case "active", "ready": MereRunTheme.green
        case "loading", "queued": MereRunTheme.yellow
        case "failed", "error": MereRunTheme.red
        default: MereRunTheme.textMuted
        }
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9.5, weight: .semibold))
            .foregroundStyle(MereRunTheme.accent)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(MereRunTheme.accentSoft, in: Capsule())
    }

    private func copyButton(_ text: String) -> some View {
        Button {
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.setString(text, forType: .string)
        } label: {
            Image(systemName: "doc.on.doc")
        }
        .buttonStyle(.mereIcon)
        .help("Copy")
    }

    private func clientCard(
        title: String,
        systemImage: String,
        detail: String,
        copy: String,
        action: (() -> Void)? = nil
    ) -> some View {
        StudioServingCard {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                cardTitle(title, systemImage: systemImage)
                Text(detail)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 4)
                HStack {
                    Button("Copy setup") {
                        let pasteboard = NSPasteboard.general
                        pasteboard.clearContents()
                        pasteboard.setString(copy, forType: .string)
                    }
                    .buttonStyle(.mereSecondary)
                    if let action {
                        Button("Launch") { action() }
                            .buttonStyle(.merePrimary)
                    }
                }
            }
            .frame(minHeight: 125, alignment: .topLeading)
        }
    }

    private func readinessRow(_ label: String, ready: Bool, detail: String?) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: ready ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(ready ? MereRunTheme.green : MereRunTheme.textMuted)
            VStack(alignment: .leading, spacing: 1) {
                Text(label).font(.system(size: 12, weight: .medium))
                if let detail {
                    Text(detail)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(2)
                }
            }
        }
    }

    private func resourceProgress(current: UInt64?, limit: UInt64?) -> some View {
        let ratio: Double = {
            guard let current, let limit, limit > 0 else { return 0 }
            return min(1, Double(current) / Double(limit))
        }()
        return VStack(alignment: .leading, spacing: 5) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(MereRunTheme.surfaceRaised)
                    Capsule()
                        .fill(ratio >= 0.95 ? MereRunTheme.red : ratio >= 0.9 ? MereRunTheme.yellow : MereRunTheme.green)
                        .frame(width: geometry.size.width * ratio)
                }
            }
            .frame(height: 8)
            Text("\(StudioServingFormat.bytes(current)) of \(StudioServingFormat.bytes(limit))")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private func integerField(_ title: String, value: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(title, value: value, format: .number)
                .mereField()
        }
    }

    private func activityColor(_ level: StudioServiceActivity.Level) -> Color {
        switch level {
        case .info: MereRunTheme.accent
        case .success: MereRunTheme.green
        case .warning: MereRunTheme.yellow
        case .error: MereRunTheme.red
        }
    }
}

private struct StudioServingCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        content
            .padding(MereRunTheme.Spacing.md)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.surface.opacity(0.72))
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                            .strokeBorder(MereRunTheme.border.opacity(0.55), lineWidth: 1)
                    }
            }
    }
}

private enum StudioServingError: LocalizedError {
    case operationFailed(String)

    var errorDescription: String? {
        switch self {
        case .operationFailed(let detail): detail
        }
    }
}
