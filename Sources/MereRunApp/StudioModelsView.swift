import AppKit
import SwiftUI

struct StudioModelInventoryRow: Identifiable, Equatable {
    let id: String
    let category: String
    let status: String
    let size: String

    var isInstalled: Bool {
        status.lowercased() == "installed"
    }
}

struct StudioRuntimeSettings: Codable, Equatable {
    var alias: String?
    var pinned: Bool
    var ttlSeconds: Int?
    var maxContextTokens: Int?
    var maxTokens: Int?
    var temperature: Double?
    var topP: Double?
    var engineOverride: String?

    init(
        alias: String? = nil,
        pinned: Bool = false,
        ttlSeconds: Int? = nil,
        maxContextTokens: Int? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        engineOverride: String? = nil
    ) {
        self.alias = alias
        self.pinned = pinned
        self.ttlSeconds = ttlSeconds
        self.maxContextTokens = maxContextTokens
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.engineOverride = engineOverride
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        alias = try container.decodeIfPresent(String.self, forKey: .alias)
        pinned = try container.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
        ttlSeconds = try container.decodeIfPresent(Int.self, forKey: .ttlSeconds)
        maxContextTokens = try container.decodeIfPresent(Int.self, forKey: .maxContextTokens)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        engineOverride = try container.decodeIfPresent(String.self, forKey: .engineOverride)
    }
}

enum StudioModelInventoryParser {
    static func rows(from output: String) -> [StudioModelInventoryRow] {
        output
            .components(separatedBy: .newlines)
            .compactMap { line -> StudioModelInventoryRow? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty,
                      !trimmed.hasPrefix("-"),
                      !trimmed.hasPrefix("ID ") else {
                    return nil
                }

                let fields = trimmed.split(whereSeparator: \.isWhitespace).map(String.init)
                guard fields.count >= 4 else { return nil }
                return StudioModelInventoryRow(
                    id: fields[0],
                    category: fields[1],
                    status: fields[2],
                    size: fields.dropFirst(3).joined(separator: " ")
                )
            }
    }

    static func modelRoot(from output: String) -> URL? {
        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            let marker = "Model Root:"
            guard trimmed.hasPrefix(marker) else { continue }
            let path = trimmed.dropFirst(marker.count).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else { return nil }
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return nil
    }
}

struct StudioModelsSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @Environment(\.dismiss) private var dismiss

    let onModelsChanged: () -> Void

    @State private var rows: [StudioModelInventoryRow] = []
    @State private var selectedID: String?
    @State private var detailText = ""
    @State private var statusMessage = "Loading models"
    @State private var showAll = false
    @State private var isRefreshing = false
    @State private var loadingInfoID: String?
    @State private var loadingRuntimeID: String?
    @State private var removingID: String?
    @State private var pendingRemoval: StudioModelInventoryRow?
    @State private var runtimeSettingsByID: [String: StudioRuntimeSettings] = [:]
    @State private var runtimeAlias = ""
    @State private var runtimeTTL = ""
    @State private var runtimeMaxContext = ""
    @State private var runtimeMaxTokens = ""
    @State private var runtimeTemperature = ""
    @State private var runtimeTopP = ""
    @State private var runtimePinned = false

    private var installedRows: [StudioModelInventoryRow] {
        rows.filter(\.isInstalled)
    }

    private var visibleRows: [StudioModelInventoryRow] {
        showAll ? rows : installedRows
    }

    private var selectedRow: StudioModelInventoryRow? {
        rows.first { $0.id == selectedID }
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            Divider()
                .overlay(MereRunTheme.border.opacity(0.6))

            HStack(spacing: 0) {
                modelList
                    .frame(width: 360)

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.6))

                detailPane
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: 920, height: 600)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            await refresh()
        }
        .onChange(of: showAll) {
            if let selectedID, visibleRows.contains(where: { $0.id == selectedID }) {
                return
            }
            guard let first = visibleRows.first else {
                selectedID = nil
                detailText = ""
                return
            }
            select(first)
        }
        .alert(item: $pendingRemoval) { row in
            Alert(
                title: Text("Purge \(row.id)?"),
                message: Text("This deletes \(row.size) from the local model store."),
                primaryButton: .destructive(Text("Purge")) {
                    Task { await purge(row) }
                },
                secondaryButton: .cancel()
            )
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Models")
                    .font(MereRunTheme.titleFont)
                Text(statusMessage)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }

            Spacer()

            Picker("Scope", selection: $showAll) {
                Text("Downloaded").tag(false)
                Text("All").tag(true)
            }
            .pickerStyle(.segmented)
            .frame(width: 230)

            Button {
                revealStore()
            } label: {
                Label("Store", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .help("Reveal the model store in Finder")

            Button {
                Task { await refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .disabled(isRefreshing)

            Button("Done") {
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .tint(MereRunTheme.accent)
        }
        .padding(18)
    }

    private var modelList: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(showAll ? "All known models" : "Downloaded")
                    .font(MereRunTheme.sectionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Text("\(visibleRows.count)")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)

            if visibleRows.isEmpty {
                emptyList
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(visibleRows) { row in
                            modelRow(row)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 14)
                }
            }
        }
    }

    private var emptyList: some View {
        VStack(spacing: 10) {
            Image(systemName: "shippingbox")
                .font(.system(size: 28, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(showAll ? "No models reported." : "No downloaded models yet.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func modelRow(_ row: StudioModelInventoryRow) -> some View {
        Button {
            select(row)
        } label: {
            HStack(spacing: 10) {
                Circle()
                    .fill(row.isInstalled ? MereRunTheme.green : MereRunTheme.border)
                    .frame(width: 8, height: 8)

                VStack(alignment: .leading, spacing: 4) {
                    Text(row.id)
                        .font(.system(size: 13, weight: .semibold))
                        .lineLimit(1)
                    HStack(spacing: 8) {
                        Text(row.category)
                        Text(row.status)
                        Text(row.size)
                        if let runtime = runtimeSettingsByID[row.id] {
                            if runtime.pinned {
                                Text("pinned")
                            }
                            if let alias = runtime.alias {
                                Text(alias)
                            }
                        }
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                }

                Spacer(minLength: 0)
            }
            .padding(10)
            .background {
                RoundedRectangle(cornerRadius: 9)
                    .fill(selectedID == row.id ? MereRunTheme.surfaceRaised : MereRunTheme.surface.opacity(0.42))
                    .overlay {
                        RoundedRectangle(cornerRadius: 9)
                            .strokeBorder(
                                selectedID == row.id ? MereRunTheme.accent.opacity(0.85) : MereRunTheme.border.opacity(0.55),
                                lineWidth: 1
                            )
                    }
            }
        }
        .buttonStyle(.plain)
    }

    private var detailPane: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let selectedRow {
                selectedHeader(selectedRow)

                ScrollView {
                    Text(detailBody(for: selectedRow))
                        .font(MereRunTheme.monoFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(14)
                }
                .merePanel()
            } else {
                Spacer()
                emptyDetail
                Spacer()
            }
        }
        .padding(18)
    }

    private func selectedHeader(_ row: StudioModelInventoryRow) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.id)
                        .font(.system(size: 18, weight: .semibold))
                    Text("\(row.category) · \(row.status) · \(row.size)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }

                Spacer()

                if loadingInfoID == row.id || removingID == row.id {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            HStack(spacing: 10) {
                Button {
                    Task { await runtimeLoad(row) }
                } label: {
                    Label("Load", systemImage: "play.fill")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || loadingRuntimeID != nil)

                Button {
                    Task { await runtimeUnload(row) }
                } label: {
                    Label("Unload", systemImage: "stop.fill")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || loadingRuntimeID != nil)

                Button {
                    Task { await saveRuntimeSettings(for: row, pinned: true) }
                } label: {
                    Label("Pin", systemImage: "pin")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || loadingRuntimeID != nil)

                Button {
                    Task { await saveRuntimeSettings(for: row, pinned: false) }
                } label: {
                    Label("Unpin", systemImage: "pin.slash")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || loadingRuntimeID != nil)
            }

            runtimeSettingsEditor(row)

            HStack(spacing: 10) {
                Button {
                    Task { await loadInfo(for: row) }
                } label: {
                    Label("Inspect", systemImage: "info.circle")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || loadingInfoID != nil)

                Button {
                    Task { await reveal(row) }
                } label: {
                    Label("Finder", systemImage: "magnifyingglass")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || loadingInfoID != nil)
                .help("Reveal this model's source folder in Finder")

                Button(role: .destructive) {
                    pendingRemoval = row
                } label: {
                    Label("Purge", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .disabled(!row.isInstalled || removingID != nil)
            }
        }
    }

    private func runtimeSettingsEditor(_ row: StudioModelInventoryRow) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                TextField("Alias", text: $runtimeAlias)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                TextField("TTL", text: $runtimeTTL)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    .frame(width: 72)
                TextField("Context", text: $runtimeMaxContext)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    .frame(width: 88)
                TextField("Max tokens", text: $runtimeMaxTokens)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    .frame(width: 96)
                TextField("Temp", text: $runtimeTemperature)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    .frame(width: 72)
                TextField("Top P", text: $runtimeTopP)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    .frame(width: 72)
                Toggle("Pinned", isOn: $runtimePinned)
                    .toggleStyle(.checkbox)

                Button {
                    Task { await saveRuntimeSettings(for: row, pinned: runtimePinned) }
                } label: {
                    Label("Save", systemImage: "checkmark")
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .disabled(!row.isInstalled || loadingRuntimeID != nil)
            }
            .font(MereRunTheme.captionFont)
        }
    }

    private var emptyDetail: some View {
        VStack(spacing: 10) {
            Image(systemName: "info.circle")
                .font(.system(size: 28, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text("Select a downloaded model to inspect, reveal, or purge it.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .frame(maxWidth: .infinity)
    }

    private func detailBody(for row: StudioModelInventoryRow) -> String {
        if !row.isInstalled {
            return "\(row.id) is not downloaded."
        }
        if detailText.isEmpty {
            return "Select Inspect to load manifest, validation, and component paths."
        }
        return detailText
    }

    private func select(_ row: StudioModelInventoryRow) {
        selectedID = row.id
        detailText = row.isInstalled ? "" : "\(row.id) is not downloaded."
        guard row.isInstalled else { return }
        Task { await loadInfo(for: row) }
        Task { await loadRuntimeSettings(for: row) }
    }

    @MainActor
    private func refresh() async {
        isRefreshing = true
        statusMessage = "Refreshing model inventory..."
        let result = await controller.utilityCommandResult(args: ["model", "list"])
        isRefreshing = false

        guard result.exitCode == 0 else {
            statusMessage = "Could not list models"
            detailText = result.outputText
            return
        }

        rows = StudioModelInventoryParser.rows(from: result.stdout)
        statusMessage = "\(installedRows.count) downloaded · \(rows.count) known"
        if let selectedID, visibleRows.contains(where: { $0.id == selectedID }) {
            return
        }
        if let first = visibleRows.first {
            select(first)
        } else {
            selectedID = nil
            detailText = ""
        }
    }

    @MainActor
    private func loadRuntimeSettings(for row: StudioModelInventoryRow) async {
        loadingRuntimeID = row.id
        let result = await controller.utilityCommandResult(args: ["model", "runtime", "get", row.id, "--json"])
        loadingRuntimeID = nil
        guard result.exitCode == 0,
              let data = result.stdout.data(using: .utf8),
              let settings = try? JSONDecoder().decode(StudioRuntimeSettings.self, from: data) else {
            return
        }
        runtimeSettingsByID[row.id] = settings
        if selectedID == row.id {
            applyRuntimeSettings(settings)
        }
    }

    private func applyRuntimeSettings(_ settings: StudioRuntimeSettings) {
        runtimeAlias = settings.alias ?? ""
        runtimeTTL = settings.ttlSeconds.map(String.init) ?? ""
        runtimeMaxContext = settings.maxContextTokens.map(String.init) ?? ""
        runtimeMaxTokens = settings.maxTokens.map(String.init) ?? ""
        runtimeTemperature = settings.temperature.map { String($0) } ?? ""
        runtimeTopP = settings.topP.map { String($0) } ?? ""
        runtimePinned = settings.pinned
    }

    @MainActor
    private func saveRuntimeSettings(for row: StudioModelInventoryRow, pinned: Bool) async {
        loadingRuntimeID = row.id
        statusMessage = "Saving runtime settings for \(row.id)..."
        var args = ["model", "runtime", "set", row.id, pinned ? "--pinned" : "--unpinned"]
        appendRuntimeSettingArgs(to: &args)
        let result = await controller.utilityCommandResult(args: args)
        loadingRuntimeID = nil
        if result.exitCode == 0 {
            statusMessage = "Saved runtime settings for \(row.id)"
            await loadRuntimeSettings(for: row)
        } else {
            statusMessage = "Could not save runtime settings for \(row.id)"
            detailText = result.outputText
        }
    }

    private func appendRuntimeSettingArgs(to args: inout [String]) {
        appendStringSetting(runtimeAlias, setFlag: "--alias", clearFlag: "--clear-alias", to: &args)
        appendStringSetting(runtimeTTL, setFlag: "--ttl-seconds", clearFlag: "--clear-ttl", to: &args)
        appendStringSetting(
            runtimeMaxContext,
            setFlag: "--max-context-tokens",
            clearFlag: "--clear-max-context-tokens",
            to: &args
        )
        appendStringSetting(runtimeMaxTokens, setFlag: "--max-tokens", clearFlag: "--clear-max-tokens", to: &args)
        appendStringSetting(
            runtimeTemperature,
            setFlag: "--temperature",
            clearFlag: "--clear-temperature",
            to: &args
        )
        appendStringSetting(runtimeTopP, setFlag: "--top-p", clearFlag: "--clear-top-p", to: &args)
    }

    private func appendStringSetting(
        _ value: String,
        setFlag: String,
        clearFlag: String,
        to args: inout [String]
    ) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            args.append(clearFlag)
        } else {
            args += [setFlag, trimmed]
        }
    }

    @MainActor
    private func runtimeLoad(_ row: StudioModelInventoryRow) async {
        await runtimeServerAction(row, action: "load")
    }

    @MainActor
    private func runtimeUnload(_ row: StudioModelInventoryRow) async {
        await runtimeServerAction(row, action: "unload")
    }

    @MainActor
    private func runtimeServerAction(_ row: StudioModelInventoryRow, action: String) async {
        loadingRuntimeID = row.id
        statusMessage = "\(action.capitalized)ing \(row.id)..."
        do {
            var request = URLRequest(url: controller.runtimeURL(path: "/runtime/models/\(row.id)/\(action)"))
            request.httpMethod = "POST"
            if let authorization = controller.runtimeAuthorizationHeader {
                request.setValue(authorization, forHTTPHeaderField: "Authorization")
            }
            let (_, response) = try await URLSession.shared.data(for: request)
            let http = response as? HTTPURLResponse
            if (200..<300).contains(http?.statusCode ?? 500) {
                statusMessage = "\(action.capitalized)ed \(row.id)"
            } else {
                statusMessage = "Runtime \(action) returned HTTP \(http?.statusCode ?? 0)"
            }
        } catch {
            statusMessage = "Runtime server is not reachable"
        }
        loadingRuntimeID = nil
    }

    @MainActor
    private func loadInfo(for row: StudioModelInventoryRow) async {
        guard row.isInstalled else { return }
        loadingInfoID = row.id
        statusMessage = "Inspecting \(row.id)..."
        let result = await controller.utilityCommandResult(args: ["model", "info", row.id, "--components"])
        loadingInfoID = nil
        detailText = result.outputText
        statusMessage = result.exitCode == 0 ? "\(installedRows.count) downloaded · \(rows.count) known" : "Could not inspect \(row.id)"
    }

    @MainActor
    private func reveal(_ row: StudioModelInventoryRow) async {
        guard row.isInstalled else { return }
        loadingInfoID = row.id
        statusMessage = "Resolving \(row.id) folder..."

        let info: String
        if let root = StudioModelInventoryParser.modelRoot(from: detailText) {
            reveal(url: root)
            loadingInfoID = nil
            statusMessage = "\(installedRows.count) downloaded · \(rows.count) known"
            return
        } else {
            let result = await controller.utilityCommandResult(args: ["model", "info", row.id])
            info = result.outputText
        }

        loadingInfoID = nil
        guard let root = StudioModelInventoryParser.modelRoot(from: info) else {
            statusMessage = "Could not resolve \(row.id) folder"
            detailText = info
            return
        }
        reveal(url: root)
        statusMessage = "\(installedRows.count) downloaded · \(rows.count) known"
    }

    @MainActor
    private func purge(_ row: StudioModelInventoryRow) async {
        removingID = row.id
        statusMessage = "Purging \(row.id)..."
        let result = await controller.utilityCommandResult(args: ["model", "remove", row.id, "--force"])
        removingID = nil
        detailText = result.outputText
        if result.exitCode == 0 {
            onModelsChanged()
            await refresh()
        } else {
            statusMessage = "Could not purge \(row.id)"
        }
    }

    private func revealStore() {
        let url = modelStoreURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.open(url)
    }

    private func reveal(url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private var modelStoreURL: URL {
        if !controller.modelsRoot.isBlank {
            return URL(fileURLWithPath: NSString(string: controller.modelsRoot).expandingTildeInPath, isDirectory: true)
        }
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support", isDirectory: true)
        return base
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
    }
}
