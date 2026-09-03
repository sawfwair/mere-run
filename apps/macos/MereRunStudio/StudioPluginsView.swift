import AppKit
import SwiftUI

struct StudioPluginCatalogSnapshot: Decodable, Equatable {
    let contractVersion: String
    let updatedAt: String
    let defaultChannel: String
    let source: String
    let plugins: [StudioPluginCatalogEntry]
}

struct StudioPluginCatalogEntry: Decodable, Equatable, Identifiable {
    let id: String
    let name: String
    let description: String
    let repo: String
    let package: String
    let subdirectory: String
    let entrypoint: String
    let capabilities: [String]
    let channels: [String: StudioPluginInstall]
    let installed: Bool
    let verified: Bool
    let installedVersion: String?
    let installedPath: String?
    let verificationError: String?
    let installCommand: String?
}

struct StudioPluginInstall: Decodable, Equatable {
    let manager: String
    let spec: String
    let ref: String?
}

enum StudioPluginConfirmationKind: String {
    case install
    case update
    case rollback
}

struct StudioPluginConfirmation: Identifiable {
    let plugin: StudioPluginCatalogEntry
    let channel: String
    let force: Bool
    var kind: StudioPluginConfirmationKind = .install

    var id: String { "\(plugin.id):\(channel):\(force):\(kind.rawValue)" }
}

struct StudioPluginsView: View {
    @EnvironmentObject private var controller: MereRunController

    @State private var catalog: StudioPluginCatalogSnapshot?
    @State private var selectedID: String?
    @State private var selectedChannel = ""
    @State private var searchText = ""
    @State private var catalogURL = ""
    @State private var statusMessage = "Loading the official plugin catalog"
    @State private var operationOutput = ""
    @State private var isRefreshing = false
    @State private var busyAction: String?
    @State private var confirmation: StudioPluginConfirmation?
    @State private var runArguments = ""

    private var visiblePlugins: [StudioPluginCatalogEntry] {
        guard let plugins = catalog?.plugins else { return [] }
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return plugins }
        return plugins.filter { plugin in
            plugin.id.lowercased().contains(query)
                || plugin.name.lowercased().contains(query)
                || plugin.description.lowercased().contains(query)
                || plugin.capabilities.contains { $0.lowercased().contains(query) }
        }
    }

    private var selectedPlugin: StudioPluginCatalogEntry? {
        catalog?.plugins.first { $0.id == selectedID }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            HStack(spacing: 0) {
                pluginList
                    .frame(minWidth: 300, idealWidth: 390, maxWidth: 390)
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                detailPane
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task { await refresh() }
        .onChange(of: selectedID) {
            guard let plugin = selectedPlugin else { return }
            selectedChannel = plugin.channels.keys.contains(catalog?.defaultChannel ?? "")
                ? catalog?.defaultChannel ?? ""
                : plugin.channels.keys.sorted().first ?? ""
            operationOutput = plugin.verificationError ?? ""
        }
        .alert(item: $confirmation) { pending in
            if pending.kind == .rollback {
                return Alert(
                    title: Text("Roll back \(pending.plugin.name)?"),
                    message: Text(
                        "mere.run will restore the retained signed bundle for \(pending.plugin.id) and "
                            + "make it the active version. The current version stays on disk."
                    ),
                    primaryButton: .destructive(Text("Roll back")) {
                        Task { await rollback(pending.plugin) }
                    },
                    secondaryButton: .cancel()
                )
            }
            return Alert(
                title: Text(pending.force ? "Update \(pending.plugin.name)?" : "Install \(pending.plugin.name)?"),
                message: Text(
                    "mere.run will execute the catalog-pinned \(pending.channel) install through pipx, "
                        + "then verify the plugin manifest."
                ),
                primaryButton: .default(Text(pending.force ? "Update" : "Install")) {
                    Task { await install(pending) }
                },
                secondaryButton: .cancel()
            )
        }
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            Text(statusMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            if let busyAction {
                ProgressView().controlSize(.small)
                Text(busyAction)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Button {
                Task { await refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.mereSecondary)
            .disabled(isRefreshing || busyAction != nil)
        }
        .padding(.horizontal, MereRunTheme.Spacing.xl)
        .padding(.vertical, MereRunTheme.Spacing.sm)
    }

    private var pluginList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 7) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(MereRunTheme.textMuted)
                TextField("Search plugins or capabilities", text: $searchText)
                    .textFieldStyle(.plain)
            }
            .padding(10)
            .merePanel()
            .padding(12)

            HStack {
                Text("Official catalog")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Text("\(visiblePlugins.count)")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)

            if visiblePlugins.isEmpty {
                ContentUnavailableView(
                    catalog == nil ? "Catalog unavailable" : "No matching plugins",
                    systemImage: "puzzlepiece.extension",
                    description: Text(statusMessage)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(visiblePlugins) { plugin in
                            pluginRow(plugin)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.bottom, 12)
                }
            }

            Divider().overlay(MereRunTheme.border.opacity(0.4))
            VStack(alignment: .leading, spacing: 7) {
                TextField("Custom catalog URL or JSON path (optional)", text: $catalogURL)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await refresh() } }
                Text(statusMessage)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(2)
            }
            .padding(12)
        }
    }

    private func pluginRow(_ plugin: StudioPluginCatalogEntry) -> some View {
        Button {
            selectedID = plugin.id
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: plugin.installed ? "checkmark.seal.fill" : "puzzlepiece")
                        .foregroundStyle(plugin.verified ? MereRunTheme.green : MereRunTheme.accent)
                    Text(plugin.name)
                        .font(.system(size: 12.5, weight: .semibold))
                    Spacer()
                    Text(plugin.installed ? plugin.installedVersion ?? "Installed" : "Available")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(plugin.verified ? MereRunTheme.green : MereRunTheme.textMuted)
                }
                Text(plugin.description)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(2)
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(selectedID == plugin.id ? MereRunTheme.accentSoft : MereRunTheme.surface)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var detailPane: some View {
        if let plugin = selectedPlugin {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 5) {
                            Text(plugin.name)
                                .font(MereRunTheme.titleFont)
                            Text(plugin.id)
                                .font(.system(size: 11.5, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                        Spacer()
                        statusBadge(plugin)
                    }

                    Text(plugin.description)
                        .font(.system(size: 13.5))
                        .foregroundStyle(MereRunTheme.textSecondary)

                    HStack(spacing: 10) {
                        detailMetric("Entrypoint", plugin.entrypoint)
                        detailMetric("Package", plugin.package)
                        detailMetric("Version", plugin.installedVersion ?? "—")
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Capabilities")
                            .font(MereRunTheme.sectionFont)
                        FlowLayout(spacing: 6) {
                            ForEach(plugin.capabilities, id: \.self) { capability in
                                Text(capability)
                                    .font(.system(size: 10.5, weight: .medium))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(MereRunTheme.accentSoft)
                                    .clipShape(Capsule())
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Install source")
                            .font(MereRunTheme.sectionFont)
                        HStack {
                            Picker("Channel", selection: $selectedChannel) {
                                ForEach(plugin.channels.keys.sorted(), id: \.self) { channel in
                                    Text(channel).tag(channel)
                                }
                            }
                            .frame(width: 220)
                            Link("Repository", destination: URL(string: plugin.repo)!)
                            Spacer()
                            Button("Details") {
                                Task { await details(plugin) }
                            }
                            .buttonStyle(.mereSecondary)
                            .disabled(busyAction != nil)
                            .help("Show this plugin's catalog entry and resolved install command")
                            Button("Copy command") { copyInstallCommand(plugin) }
                                .buttonStyle(.mereSecondary)
                            Button(plugin.installed ? "Update" : "Install") {
                                confirmation = StudioPluginConfirmation(
                                    plugin: plugin,
                                    channel: selectedChannel,
                                    force: plugin.installed,
                                    kind: plugin.installed ? .update : .install
                                )
                            }
                            .buttonStyle(.merePrimary)
                        }
                        if let command = selectedInstallCommand(plugin) {
                            Text(command)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                                .textSelection(.enabled)
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .merePanel()
                        }
                    }

                    if plugin.installed {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Installed plugin")
                                    .font(MereRunTheme.sectionFont)
                                Spacer()
                                if let path = plugin.installedPath {
                                    Button("Reveal") {
                                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                                    }
                                    .buttonStyle(.mereSecondary)
                                }
                                Button("Run Doctor") {
                                    Task { await doctor(plugin) }
                                }
                                .buttonStyle(.mereSecondary)
                                .disabled(busyAction != nil)
                                Button("Roll back") {
                                    confirmation = StudioPluginConfirmation(
                                        plugin: plugin,
                                        channel: selectedChannel,
                                        force: false,
                                        kind: .rollback
                                    )
                                }
                                .buttonStyle(.mereSecondary)
                                .disabled(busyAction != nil)
                                .help("Restore the retained signed bundle for this plugin")
                            }

                            HStack(spacing: 8) {
                                TextField(
                                    "Arguments passed to \(plugin.entrypoint)",
                                    text: $runArguments
                                )
                                .mereField(cornerRadius: MereRunTheme.Radius.sm)
                                Button("Run") {
                                    Task { await run(plugin) }
                                }
                                .buttonStyle(.mereSecondary)
                                .disabled(busyAction != nil)
                                .help("Run the installed plugin without changing PATH")
                            }
                            if let path = plugin.installedPath {
                                Text(path)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(MereRunTheme.textMuted)
                                    .textSelection(.enabled)
                            }
                            if let error = plugin.verificationError {
                                Label(error, systemImage: "exclamationmark.triangle.fill")
                                    .font(MereRunTheme.captionFont)
                                    .foregroundStyle(MereRunTheme.red)
                            }
                        }
                        .padding(12)
                        .merePanel()
                    }

                    if !operationOutput.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Latest operation")
                                .font(MereRunTheme.sectionFont)
                            Text(operationOutput)
                                .font(.system(size: 11.5, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(12)
                                .merePanel()
                        }
                    }
                }
                .padding(MereRunTheme.Spacing.xl)
            }
        } else {
            ContentUnavailableView(
                "Select a plugin",
                systemImage: "puzzlepiece.extension",
                description: Text("Review capabilities and installation state before changing this Mac.")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func statusBadge(_ plugin: StudioPluginCatalogEntry) -> some View {
        Label(
            plugin.verified ? "Installed & verified" : plugin.installed ? "Needs attention" : "Not installed",
            systemImage: plugin.verified ? "checkmark.seal.fill" : plugin.installed
                ? "exclamationmark.triangle.fill" : "arrow.down.circle"
        )
        .font(.system(size: 11.5, weight: .semibold))
        .foregroundStyle(plugin.verified ? MereRunTheme.green : plugin.installed ? MereRunTheme.red : MereRunTheme.accent)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(MereRunTheme.surface)
        .clipShape(Capsule())
    }

    private func detailMetric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title.uppercased())
                .font(.system(size: 9.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 12, weight: .medium))
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .merePanel()
    }

    private func refresh() async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        var args = ["plugin", "list"]
        if !catalogURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--catalog-url", catalogURL]
        }
        args.append("--json")
        let result = await controller.utilityCommandResult(args: args)
        guard result.exitCode == 0,
              let data = StudioOperationsJSON.objectData(result.stdout),
              let decoded = try? JSONDecoder().decode(StudioPluginCatalogSnapshot.self, from: data) else {
            catalog = nil
            statusMessage = StudioOperationsJSON.failureMessage(result)
            return
        }
        catalog = decoded
        statusMessage = "Updated \(decoded.updatedAt) · \(decoded.plugins.filter(\.installed).count) installed"
        if selectedID == nil || !decoded.plugins.contains(where: { $0.id == selectedID }) {
            selectedID = decoded.plugins.first?.id
        }
    }

    private func install(_ pending: StudioPluginConfirmation) async {
        busyAction = pending.force ? "Updating \(pending.plugin.name)" : "Installing \(pending.plugin.name)"
        defer { busyAction = nil }
        var args = ["plugin", "install", pending.plugin.id]
        if !catalogURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--catalog-url", catalogURL]
        }
        if !pending.channel.isEmpty {
            args += ["--channel", pending.channel]
        }
        args.append("--yes")
        if pending.force { args.append("--force") }
        let result = await controller.utilityCommandResult(args: args)
        operationOutput = result.outputText
        await refresh()
    }

    private func doctor(_ plugin: StudioPluginCatalogEntry) async {
        busyAction = "Checking \(plugin.name)"
        defer { busyAction = nil }
        var args = ["plugin", "doctor", plugin.id]
        if !catalogURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--catalog-url", catalogURL]
        }
        let result = await controller.utilityCommandResult(args: args)
        operationOutput = result.outputText
        await refresh()
    }

    /// `plugin info` — the catalog entry and resolved install command for one plugin.
    private func details(_ plugin: StudioPluginCatalogEntry) async {
        busyAction = "Reading \(plugin.name)"
        defer { busyAction = nil }
        var args = ["plugin", "info", plugin.id]
        if !catalogURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--catalog-url", catalogURL]
        }
        if !selectedChannel.isEmpty { args += ["--channel", selectedChannel] }
        let result = await controller.utilityCommandResult(args: args)
        operationOutput = result.outputText
        statusMessage = result.exitCode == 0
            ? "Read the catalog entry for \(plugin.id)"
            : "Could not read \(plugin.id)"
    }

    /// `plugin run` — runs the installed entrypoint out of process without touching PATH.
    private func run(_ plugin: StudioPluginCatalogEntry) async {
        let entrypoint = plugin.entrypoint
        busyAction = "Running \(entrypoint)"
        defer { busyAction = nil }
        let forwarded = ShellWords.split(runArguments)
        let result = await controller.utilityCommandResult(args: ["plugin", "run", entrypoint] + forwarded)
        operationOutput = result.outputText
        statusMessage = result.exitCode == 0
            ? "\(entrypoint) finished"
            : "\(entrypoint) exited with code \(result.exitCode)"
    }

    /// `plugin rollback --yes` — activates the retained signed bundle. Always confirmed first.
    private func rollback(_ plugin: StudioPluginCatalogEntry) async {
        busyAction = "Rolling back \(plugin.name)"
        defer { busyAction = nil }
        let result = await controller.utilityCommandResult(
            args: ["plugin", "rollback", plugin.id, "--yes"]
        )
        operationOutput = result.outputText
        statusMessage = result.exitCode == 0
            ? "Restored the retained bundle for \(plugin.id)"
            : "Could not roll back \(plugin.id)"
        await refresh()
    }

    private func copyInstallCommand(_ plugin: StudioPluginCatalogEntry) {
        guard let command = selectedInstallCommand(plugin) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(command, forType: .string)
        statusMessage = "Copied install command for \(plugin.id)"
    }

    private func selectedInstallCommand(_ plugin: StudioPluginCatalogEntry) -> String? {
        guard let install = plugin.channels[selectedChannel] else {
            return plugin.installCommand
        }
        switch install.manager {
        case "pipx":
            return "pipx install \(plugin.installed ? "--force " : "")\(install.spec)"
        default:
            return "\(install.manager) \(install.spec)"
        }
    }
}
