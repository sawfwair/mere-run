import AppKit
import SwiftUI

struct StudioModelUsageTerms: Equatable {
    let summary: String
    let links: [URL]
}

struct StudioModelInventoryRow: Identifiable, Equatable {
    let id: String
    let category: String
    let status: String
    let size: String
    let usageTerms: StudioModelUsageTerms?
    let title: String?
    let summary: String?
    let estimatedDownloadBytes: Int64?
    let minimumUnifiedMemoryGB: Int?
    let recommendedUnifiedMemoryGB: Int?
    let supported: Bool?
    let supportReasons: [String]
    let sourceRepository: String?
    let publisher: String?
    let referencedBytes: Int64?
    let reclaimableBytes: Int64?
    let sharedBytes: Int64?
    let externalBytes: Int64?

    init(
        id: String,
        category: String,
        status: String,
        size: String,
        usageTerms: StudioModelUsageTerms?,
        title: String? = nil,
        summary: String? = nil,
        estimatedDownloadBytes: Int64? = nil,
        minimumUnifiedMemoryGB: Int? = nil,
        recommendedUnifiedMemoryGB: Int? = nil,
        supported: Bool? = nil,
        supportReasons: [String] = [],
        sourceRepository: String? = nil,
        publisher: String? = nil,
        referencedBytes: Int64? = nil,
        reclaimableBytes: Int64? = nil,
        sharedBytes: Int64? = nil,
        externalBytes: Int64? = nil
    ) {
        self.id = id
        self.category = category
        self.status = status
        self.size = size
        self.usageTerms = usageTerms
        self.title = title
        self.summary = summary
        self.estimatedDownloadBytes = estimatedDownloadBytes
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.recommendedUnifiedMemoryGB = recommendedUnifiedMemoryGB
        self.supported = supported
        self.supportReasons = supportReasons
        self.sourceRepository = sourceRepository
        self.publisher = publisher
        self.referencedBytes = referencedBytes
        self.reclaimableBytes = reclaimableBytes
        self.sharedBytes = sharedBytes
        self.externalBytes = externalBytes
    }

    var isInstalled: Bool {
        status.lowercased() == "installed"
    }

    var displayedSize: String {
        guard !isInstalled, let estimatedDownloadBytes else {
            return size
        }
        return ByteCountFormatter.string(fromByteCount: estimatedDownloadBytes, countStyle: .file)
    }
}

struct StudioModelCatalogMetadata: Decodable, Equatable {
    let id: String
    let title: String
    let summary: String
    let minimumUnifiedMemoryGB: Int
    let recommendedUnifiedMemoryGB: Int
    let supported: Bool
    let reasons: [String]
    let estimatedDownloadBytes: Int64?
    let sourceRepository: String?
    let publisher: String?
}

private struct StudioModelCapabilitiesOutput: Decodable {
    let models: [StudioModelCatalogMetadata]
}

enum StudioModelCatalogParser {
    static func metadataByID(from output: String) -> [String: StudioModelCatalogMetadata] {
        guard let data = output.data(using: .utf8),
              let payload = try? JSONDecoder().decode(StudioModelCapabilitiesOutput.self, from: data) else {
            return [:]
        }
        return Dictionary(uniqueKeysWithValues: payload.models.map { ($0.id, $0) })
    }

    static func applying(
        _ metadataByID: [String: StudioModelCatalogMetadata],
        to rows: [StudioModelInventoryRow]
    ) -> [StudioModelInventoryRow] {
        rows.map { row in
            guard let metadata = metadataByID[row.id] else { return row }
            return StudioModelInventoryRow(
                id: row.id,
                category: row.category,
                status: row.status,
                size: row.size,
                usageTerms: row.usageTerms,
                title: metadata.title,
                summary: metadata.summary,
                estimatedDownloadBytes: metadata.estimatedDownloadBytes,
                minimumUnifiedMemoryGB: metadata.minimumUnifiedMemoryGB,
                recommendedUnifiedMemoryGB: metadata.recommendedUnifiedMemoryGB,
                supported: metadata.supported,
                supportReasons: metadata.reasons,
                sourceRepository: metadata.sourceRepository,
                publisher: metadata.publisher,
                referencedBytes: row.referencedBytes,
                reclaimableBytes: row.reclaimableBytes,
                sharedBytes: row.sharedBytes,
                externalBytes: row.externalBytes
            )
        }
    }
}

enum StudioModelDownloadCommand {
    static func arguments(modelID: String, acknowledgingUsageTerms: Bool) -> [String] {
        var arguments = ["model", "pull", modelID]
        if acknowledgingUsageTerms {
            arguments.append("--accept-model-license")
        }
        return arguments
    }

    static func appendingOutput(_ chunk: String, to current: String, limit: Int = 32 * 1024) -> String {
        let normalized = chunk.replacingOccurrences(of: "\r", with: "\n")
        return String((current + normalized).suffix(limit))
    }

    static func latestProgress(in output: String) -> StudioRunProgress? {
        output
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: .newlines)
            .reversed()
            .lazy
            .compactMap(StudioProgressParser.parse)
            .first
    }
}

enum StudioModelOptimizationCommand {
    static func supports(modelID: String) -> Bool {
        let normalized = modelID.lowercased()
        return normalized == "video-minimax-h3-fl2va-mlx"
            || normalized == "video-minimax-h3-fl2va-bf16-mlx"
            || normalized == "video-minimax-h3-fl2va-8bit-mlx"
            || normalized == "video-minimax-h3-ref2va-mlx"
    }

    static func arguments(modelID: String, replacing: Bool) -> [String] {
        var arguments = ["model", "optimize", modelID, "--json"]
        if replacing { arguments.append("--force") }
        return arguments
    }
}

private enum StudioModelsAlert: Identifiable {
    case download(StudioModelInventoryRow)
    case removal(StudioModelInventoryRow)
    case cleanup(StudioModelGarbagePlan)

    var id: String {
        switch self {
        case .download(let row): "download:\(row.id)"
        case .removal(let row): "removal:\(row.id)"
        case .cleanup: "cleanup"
        }
    }
}

private struct StudioModelGarbageCollectOutput: Decodable {
    let plan: StudioModelGarbagePlan
    let result: StudioModelGarbageResult?
}

private struct StudioModelGarbagePlan: Decodable {
    let reclaimableBytes: Int64
    let items: [StudioModelGarbageItem]
}

private struct StudioModelGarbageItem: Decodable {}

private struct StudioModelGarbageResult: Decodable {
    let reclaimedBytes: Int64
}

private struct StudioModelStorageReport: Decodable {
    let applicationSupportBytes: Int64
    let garbageCollectableBytes: Int64
    let models: [StudioModelStorageUsage]
}

private struct StudioModelStorageUsage: Decodable {
    let id: String
    let installed: Bool
    let referencedBytes: Int64
    let reclaimableBytes: Int64
    let sharedBytes: Int64
    let externalBytes: Int64
}

struct StudioRuntimeSettings: Codable, Equatable {
    var alias: String?
    var pinned: Bool
    var ttlSeconds: Int?
    var maxContextTokens: Int?
    var maxTokens: Int?
    var temperature: Double?
    var topP: Double?
    var minP: Double?
    var engineOverride: String?

    init(
        alias: String? = nil,
        pinned: Bool = false,
        ttlSeconds: Int? = nil,
        maxContextTokens: Int? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        minP: Double? = nil,
        engineOverride: String? = nil
    ) {
        self.alias = alias
        self.pinned = pinned
        self.ttlSeconds = ttlSeconds
        self.maxContextTokens = maxContextTokens
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.minP = minP
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
        minP = try container.decodeIfPresent(Double.self, forKey: .minP)
        engineOverride = try container.decodeIfPresent(String.self, forKey: .engineOverride)
    }
}

enum StudioModelInventoryParser {
    static func rows(from output: String) -> [StudioModelInventoryRow] {
        let usageTerms = usageTermsByID(from: output)
        return output
            .components(separatedBy: .newlines)
            .compactMap { line -> StudioModelInventoryRow? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty,
                      !trimmed.hasPrefix("-"),
                      !trimmed.hasPrefix("ID "),
                      !trimmed.hasPrefix("Usage restriction:"),
                      !trimmed.hasPrefix("Usage terms:") else {
                    return nil
                }

                let fields = trimmed.split(whereSeparator: \.isWhitespace).map(String.init)
                guard fields.count >= 4 else { return nil }
                return StudioModelInventoryRow(
                    id: fields[0],
                    category: fields[1],
                    status: fields[2],
                    size: fields.dropFirst(3).joined(separator: " "),
                    usageTerms: usageTerms[fields[0]]
                )
            }
    }

    static func usageTermsByID(from output: String) -> [String: StudioModelUsageTerms] {
        var result: [String: StudioModelUsageTerms] = [:]
        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            let marker = "Usage terms: "
            guard trimmed.hasPrefix(marker) else { continue }
            let payload = String(trimmed.dropFirst(marker.count))
            guard let separator = payload.range(of: " - ") else { continue }
            let id = String(payload[..<separator.lowerBound])
            let summary = String(payload[separator.upperBound...])
            let links = summary
                .split(whereSeparator: \.isWhitespace)
                .compactMap { token -> URL? in
                    let candidate = token.trimmingCharacters(in: CharacterSet(charactersIn: "[];"))
                    guard candidate.hasPrefix("https://") else { return nil }
                    return URL(string: candidate)
                }
            result[id] = StudioModelUsageTerms(summary: summary, links: links)
        }
        return result
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

struct StudioModelsView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    let onModelsChanged: () -> Void

    @State private var rows: [StudioModelInventoryRow] = []
    @State private var selectedID: String?
    @State private var detailText = ""
    @State private var statusMessage = "Loading models"
    @State private var showAll = false
    @State private var searchText = ""
    @State private var isRefreshing = false
    @State private var loadingInfoID: String?
    @State private var loadingRuntimeID: String?
    @State private var downloadingID: String?
    @State private var downloadCommandID: UUID?
    @State private var downloadProgress: StudioRunProgress?
    @State private var downloadProgressOutput = ""
    @State private var cancellingDownloadID: String?
    @State private var removingID: String?
    @State private var optimizingID: String?
    @State private var pendingAlert: StudioModelsAlert?
    @State private var storageReport: StudioModelStorageReport?
    @State private var isCleaningStorage = false
    @State private var runtimeSettingsByID: [String: StudioRuntimeSettings] = [:]
    @State private var runtimeAlias = ""
    @State private var runtimeTTL = ""
    @State private var runtimeMaxContext = ""
    @State private var runtimeMaxTokens = ""
    @State private var runtimeTemperature = ""
    @State private var runtimeTopP = ""
    @State private var runtimeMinP = ""
    @State private var runtimePinned = false

    private var installedRows: [StudioModelInventoryRow] {
        rows.filter(\.isInstalled)
    }

    private var visibleRows: [StudioModelInventoryRow] {
        let scoped = showAll ? rows : installedRows
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return scoped }
        return scoped.filter { row in
            row.id.lowercased().contains(query) || row.category.lowercased().contains(query)
        }
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
        .alert(item: $pendingAlert) { pending in
            switch pending {
            case .download(let row):
                Alert(
                    title: Text("Accept third-party model terms"),
                    message: Text(downloadTermsMessage(row)),
                    primaryButton: .default(Text("Accept & Download")) {
                        Task { await download(row, acknowledgingUsageTerms: true) }
                    },
                    secondaryButton: .cancel()
                )
            case .removal(let row):
                Alert(
                    title: Text("Purge \(row.id)?"),
                    message: Text(removalMessage(row)),
                    primaryButton: .destructive(Text("Purge")) {
                        Task { await purge(row) }
                    },
                    secondaryButton: .cancel()
                )
            case .cleanup(let plan):
                Alert(
                    title: Text("Clean up model storage?"),
                    message: Text(
                        "This deletes \(Self.bytes(plan.reclaimableBytes)) across \(plan.items.count) "
                            + "unreferenced items. Installed and legacy-linked payloads are preserved."
                    ),
                    primaryButton: .destructive(Text("Clean Up")) {
                        Task { await cleanStorage() }
                    },
                    secondaryButton: .cancel()
                )
            }
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            Text(statusMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)

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
                Label("Files", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .help("Open model storage in Finder")

            Button {
                Task { await previewStorageCleanup() }
            } label: {
                Label("Clean Up", systemImage: "externaldrive.badge.minus")
            }
            .buttonStyle(.bordered)
            .disabled(isRefreshing || isCleaningStorage)
            .help("Preview unreferenced payloads and partial downloads before deleting them")

            Button {
                Task { await refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .disabled(isRefreshing)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }

    private var modelList: some View {
        VStack(alignment: .leading, spacing: 0) {
            searchField
                .padding(.horizontal, 12)
                .padding(.top, 12)
                .padding(.bottom, 8)

            HStack {
                Text(showAll ? "All known models" : "Downloaded")
                    .font(.system(size: 10.5, weight: .semibold))
                    .kerning(0.5)
                    .textCase(.uppercase)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Text("\(visibleRows.count)")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 6)

            if visibleRows.isEmpty {
                emptyList
            } else {
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(visibleRows) { row in
                            StudioModelListRow(
                                row: row,
                                isSelected: selectedID == row.id,
                                runtime: runtimeSettingsByID[row.id]
                            ) {
                                select(row)
                            }
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.bottom, 14)
                }
            }
        }
    }

    private var searchField: some View {
        HStack(spacing: 7) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search models", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 12.5))
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 11))
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, MereRunTheme.Spacing.sm)
        .frame(height: 30)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.7), lineWidth: 1)
                }
        }
    }

    private var emptyList: some View {
        VStack(spacing: 10) {
            Image(systemName: "shippingbox")
                .font(.system(size: 28, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(emptyListMessage)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(16)
    }

    private var emptyListMessage: String {
        if !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "No models match the search."
        }
        return showAll ? "No models reported." : "No downloaded models yet."
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
                    Text(row.title ?? row.id)
                        .font(.system(size: 18, weight: .semibold))
                    Text(modelStatusLine(row))
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    if let summary = row.summary {
                        Text(summary)
                            .font(MereRunTheme.bodyFont)
                            .foregroundStyle(MereRunTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    modelFacts(row)
                    if row.supported == false, !row.supportReasons.isEmpty {
                        Text(row.supportReasons.joined(separator: " "))
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.yellow)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if let usageTerms = row.usageTerms {
                        Text("Third-party usage terms · acceptance required for new downloads")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.yellow)
                        Text(usageTerms.summary)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                        HStack(spacing: 8) {
                            ForEach(Array(usageTerms.links.enumerated()), id: \.offset) { index, url in
                                Link("Review terms \(index + 1)", destination: url)
                                    .font(MereRunTheme.captionFont)
                            }
                        }
                    }
                }

                Spacer()

                if loadingInfoID == row.id || downloadingID == row.id || removingID == row.id
                    || optimizingID == row.id {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            if row.isInstalled {
                installedModelActions(row)
                runtimeSettingsEditor(row)
                installedModelFilesActions(row)
            } else {
                missingModelActions(row)
            }
        }
    }

    private func modelStatusLine(_ row: StudioModelInventoryRow) -> String {
        let sizeDescription = row.isInstalled
            ? "\(row.size) referenced"
            : "\(row.displayedSize) estimated download"
        return "\(row.id) · \(row.category) · \(row.status) · \(sizeDescription)"
    }

    private func modelFacts(_ row: StudioModelInventoryRow) -> some View {
        HStack(spacing: 12) {
            if let estimatedDownloadBytes = row.estimatedDownloadBytes {
                Label(
                    "Checkpoint \(Self.bytes(estimatedDownloadBytes))",
                    systemImage: "externaldrive"
                )
            }
            if let minimumUnifiedMemoryGB = row.minimumUnifiedMemoryGB {
                Label("\(minimumUnifiedMemoryGB) GB RAM minimum", systemImage: "memorychip")
            }
            if let sourceRepository = row.sourceRepository,
               let sourceURL = URL(string: "https://huggingface.co/\(sourceRepository)") {
                Link(destination: sourceURL) {
                    Label("By \(row.publisher ?? sourceRepository)", systemImage: "person.crop.circle")
                }
            } else if let publisher = row.publisher {
                Label("By \(publisher)", systemImage: "person.crop.circle")
            }
        }
        .font(MereRunTheme.captionFont)
        .foregroundStyle(MereRunTheme.textMuted)
    }

    private func installedModelActions(_ row: StudioModelInventoryRow) -> some View {
        HStack(spacing: 10) {
            Button {
                Task { await runtimeLoad(row) }
            } label: {
                Label("Load", systemImage: "play.fill")
            }
            .buttonStyle(.bordered)
            .disabled(loadingRuntimeID != nil)

            Button {
                Task { await runtimeUnload(row) }
            } label: {
                Label("Unload", systemImage: "stop.fill")
            }
            .buttonStyle(.bordered)
            .disabled(loadingRuntimeID != nil)

            Button {
                Task { await saveRuntimeSettings(for: row, pinned: true) }
            } label: {
                Label("Pin", systemImage: "pin")
            }
            .buttonStyle(.bordered)
            .disabled(loadingRuntimeID != nil)

            Button {
                Task { await saveRuntimeSettings(for: row, pinned: false) }
            } label: {
                Label("Unpin", systemImage: "pin.slash")
            }
            .buttonStyle(.bordered)
            .disabled(loadingRuntimeID != nil)
        }
    }

    private func installedModelFilesActions(_ row: StudioModelInventoryRow) -> some View {
        HStack(spacing: 10) {
            Button {
                Task { await loadInfo(for: row) }
            } label: {
                Label("Inspect", systemImage: "info.circle")
            }
            .buttonStyle(.bordered)
            .disabled(loadingInfoID != nil)

            Button {
                Task { await reveal(row) }
            } label: {
                Label("Finder", systemImage: "magnifyingglass")
            }
            .buttonStyle(.bordered)
            .disabled(loadingInfoID != nil)
            .help("Reveal this model's source folder in Finder")

            if StudioModelOptimizationCommand.supports(modelID: row.id) {
                Button {
                    Task { await optimize(row, replacing: false) }
                } label: {
                    Label("Optimize", systemImage: "bolt.badge.clock")
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .disabled(optimizingID != nil)
                .help("Build the inference-only MiniMax-H3 AdaLN cache")

                Menu {
                    Button("Rebuild optimization cache") {
                        Task { await optimize(row, replacing: true) }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .disabled(optimizingID != nil)
            }

            Button(role: .destructive) {
                pendingAlert = .removal(row)
            } label: {
                Label("Purge", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .disabled(removingID != nil)
        }
    }

    private func missingModelActions(_ row: StudioModelInventoryRow) -> some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
            Text("Download this model into Mere's managed model storage. The CLI checks hardware, disk space, and the pinned source before transferring weights.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)

            HStack(spacing: MereRunTheme.Spacing.sm) {
                Button {
                    requestDownload(row)
                } label: {
                    if downloadingID == row.id {
                        HStack(spacing: 7) {
                            ProgressView()
                                .controlSize(.small)
                            Text(cancellingDownloadID == row.id ? "Cancelling…" : "Downloading…")
                        }
                    } else {
                        Label("Download", systemImage: "arrow.down.circle.fill")
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .disabled(downloadingID != nil || isRefreshing)

                if downloadingID == row.id {
                    Button("Cancel", role: .cancel) {
                        cancelDownload()
                    }
                    .buttonStyle(.bordered)
                    .disabled(cancellingDownloadID == row.id)
                }

                Spacer()

                if row.displayedSize != "—" {
                    Text("\(row.displayedSize) download")
                        .font(MereRunTheme.monoFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }

            if downloadingID == row.id, let progress = downloadProgress {
                VStack(alignment: .leading, spacing: 5) {
                    if let fraction = progress.fractionCompleted {
                        ProgressView(value: fraction)
                            .progressViewStyle(.linear)
                    } else {
                        ProgressView()
                            .controlSize(.small)
                    }
                    HStack(spacing: 6) {
                        Text(progress.label)
                        if let detail = progress.detail {
                            Text("· \(detail)")
                        }
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                }
            }

            if row.usageTerms != nil {
                Text("Review the linked terms above. Download requires explicit acceptance.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
            }
        }
        .padding(MereRunTheme.Spacing.md)
        .merePanel()
    }

    private func runtimeSettingsEditor(_ row: StudioModelInventoryRow) -> some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
            Text("Runtime settings")
                .font(.system(size: 10.5, weight: .semibold))
                .kerning(0.5)
                .textCase(.uppercase)
                .foregroundStyle(MereRunTheme.textMuted)

            Grid(alignment: .leading, horizontalSpacing: MereRunTheme.Spacing.sm, verticalSpacing: 8) {
                GridRow {
                    runtimeField("Alias", text: $runtimeAlias, width: 150)
                    runtimeField("TTL seconds", text: $runtimeTTL, width: 100)
                    runtimeField("Context", text: $runtimeMaxContext, width: 100)
                }
                GridRow {
                    runtimeField("Max tokens", text: $runtimeMaxTokens, width: 150)
                    runtimeField("Temperature", text: $runtimeTemperature, width: 100)
                    runtimeField("Top P", text: $runtimeTopP, width: 100)
                }
                GridRow {
                    runtimeField("Min P", text: $runtimeMinP, width: 150)
                }
            }

            HStack(spacing: MereRunTheme.Spacing.sm) {
                Toggle("Pinned", isOn: $runtimePinned)
                    .toggleStyle(.checkbox)
                    .font(MereRunTheme.captionFont)

                Spacer()

                Button {
                    Task { await saveRuntimeSettings(for: row, pinned: runtimePinned) }
                } label: {
                    Label("Save settings", systemImage: "checkmark")
                }
                .buttonStyle(.merePrimary)
                .disabled(!row.isInstalled || loadingRuntimeID != nil)
            }
        }
        .padding(MereRunTheme.Spacing.md)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.55))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border.opacity(0.5), lineWidth: 1)
                }
        }
    }

    private func runtimeField(_ label: String, text: Binding<String>, width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(label, text: text)
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
                .frame(width: width)
                .labelsHidden()
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
            return detailText.isEmpty ? "\(row.id) is not downloaded." : detailText
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

    private func requestDownload(_ row: StudioModelInventoryRow) {
        if row.usageTerms != nil {
            pendingAlert = .download(row)
        } else {
            Task { await download(row, acknowledgingUsageTerms: false) }
        }
    }

    @MainActor
    private func download(_ row: StudioModelInventoryRow, acknowledgingUsageTerms: Bool) async {
        let modelID = row.id
        downloadingID = modelID
        cancellingDownloadID = nil
        downloadProgress = nil
        downloadProgressOutput = ""
        let commandID = UUID()
        downloadCommandID = commandID
        statusMessage = "Downloading \(modelID)…"
        detailText = "Starting managed download for \(modelID)…\n"

        let result = await controller.utilityCommandResult(
            args: StudioModelDownloadCommand.arguments(
                modelID: modelID,
                acknowledgingUsageTerms: acknowledgingUsageTerms
            ),
            commandID: commandID,
            onOutput: { chunk in
                downloadProgressOutput = StudioModelDownloadCommand.appendingOutput(
                    chunk,
                    to: downloadProgressOutput
                )
                if let progress = StudioModelDownloadCommand.latestProgress(in: downloadProgressOutput) {
                    downloadProgress = progress
                }
                guard selectedID == modelID else { return }
                detailText = StudioModelDownloadCommand.appendingOutput(chunk, to: detailText)
            }
        )

        let wasCancelled = cancellingDownloadID == modelID
        downloadCommandID = nil
        downloadingID = nil
        cancellingDownloadID = nil
        downloadProgress = nil
        downloadProgressOutput = ""

        if selectedID == modelID, !result.outputText.isEmpty {
            detailText = result.outputText
        }
        if wasCancelled {
            statusMessage = "Cancelled download for \(modelID)"
            if selectedID == modelID {
                detailText = StudioModelDownloadCommand.appendingOutput(
                    "\nDownload cancelled. Its resumable partial payload remains available to the next pull.\n",
                    to: detailText
                )
            }
            return
        }
        guard result.exitCode == 0 else {
            statusMessage = "Could not download \(modelID)"
            return
        }

        let keepSelection = selectedID == modelID
        onModelsChanged()
        await refresh()
        if keepSelection, let installed = rows.first(where: { $0.id == modelID }) {
            select(installed)
        }
        statusMessage = "Downloaded \(modelID)"
    }

    private func cancelDownload() {
        guard let commandID = downloadCommandID,
              let modelID = downloadingID,
              controller.cancelUtilityCommand(commandID) else {
            return
        }
        cancellingDownloadID = modelID
        statusMessage = "Cancelling download for \(modelID)…"
    }

    private func downloadTermsMessage(_ row: StudioModelInventoryRow) -> String {
        let summary = row.usageTerms?.summary ?? "This model has third-party usage terms."
        return """
        \(summary)

        By continuing, you confirm that you reviewed and accept the listed terms and agree to comply with them. \
        Mere does not determine whether your intended use is permitted. You are responsible for compliance.
        """
    }

    @MainActor
    private func refresh() async {
        isRefreshing = true
        statusMessage = "Refreshing model inventory..."
        let result = await controller.utilityCommandResult(args: ["model", "list"])

        guard result.exitCode == 0 else {
            isRefreshing = false
            statusMessage = "Could not list models"
            detailText = result.outputText
            return
        }

        rows = StudioModelInventoryParser.rows(from: result.stdout)
        let capabilitiesResult = await controller.utilityCommandResult(
            args: ["model", "capabilities", "--all", "--json"]
        )
        if capabilitiesResult.exitCode == 0 {
            rows = StudioModelCatalogParser.applying(
                StudioModelCatalogParser.metadataByID(from: capabilitiesResult.stdout),
                to: rows
            )
        }
        let storageResult = await controller.utilityCommandResult(args: ["model", "storage", "--json"])
        if storageResult.exitCode == 0,
           let data = storageResult.stdout.data(using: .utf8),
           let report = try? JSONDecoder().decode(StudioModelStorageReport.self, from: data) {
            storageReport = report
            applyStorageUsage(report)
        }
        isRefreshing = false
        updateStatusMessage()
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
        runtimeMinP = settings.minP.map { String($0) } ?? ""
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
        appendStringSetting(runtimeMinP, setFlag: "--min-p", clearFlag: "--clear-min-p", to: &args)
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

    @MainActor
    private func optimize(_ row: StudioModelInventoryRow, replacing: Bool) async {
        optimizingID = row.id
        statusMessage = replacing ? "Rebuilding optimization cache for \(row.id)…" : "Optimizing \(row.id)…"
        detailText = "Preparing MiniMax-H3 inference cache…\n"
        let result = await controller.utilityCommandResult(
            args: StudioModelOptimizationCommand.arguments(modelID: row.id, replacing: replacing),
            onOutput: { chunk in
                guard selectedID == row.id else { return }
                detailText = StudioModelDownloadCommand.appendingOutput(chunk, to: detailText)
            }
        )
        optimizingID = nil
        detailText = result.outputText.isEmpty ? detailText : result.outputText
        statusMessage = result.exitCode == 0
            ? "Optimized \(row.id)"
            : "Could not optimize \(row.id)"
    }

    @MainActor
    private func previewStorageCleanup() async {
        isCleaningStorage = true
        statusMessage = "Inspecting model storage..."
        let command = await controller.utilityCommandResult(args: ["model", "gc", "--json"])
        isCleaningStorage = false
        guard command.exitCode == 0,
              let data = command.stdout.data(using: .utf8),
              let output = try? JSONDecoder().decode(StudioModelGarbageCollectOutput.self, from: data) else {
            statusMessage = "Could not inspect model storage"
            detailText = command.outputText
            return
        }
        guard !output.plan.items.isEmpty else {
            statusMessage = "Model storage is clean"
            return
        }
        pendingAlert = .cleanup(output.plan)
        statusMessage = "\(Self.bytes(output.plan.reclaimableBytes)) can be cleaned up"
    }

    @MainActor
    private func cleanStorage() async {
        isCleaningStorage = true
        statusMessage = "Cleaning model storage..."
        let command = await controller.utilityCommandResult(args: ["model", "gc", "--force", "--json"])
        isCleaningStorage = false
        guard command.exitCode == 0,
              let data = command.stdout.data(using: .utf8),
              let output = try? JSONDecoder().decode(StudioModelGarbageCollectOutput.self, from: data),
              let result = output.result else {
            statusMessage = "Could not clean model storage"
            detailText = command.outputText
            return
        }
        statusMessage = "Reclaimed \(Self.bytes(result.reclaimedBytes))"
        await refresh()
    }

    private func applyStorageUsage(_ report: StudioModelStorageReport) {
        let usageByID = Dictionary(uniqueKeysWithValues: report.models.map { ($0.id, $0) })
        rows = rows.map { row in
            guard let usage = usageByID[row.id], usage.installed else { return row }
            return StudioModelInventoryRow(
                id: row.id,
                category: row.category,
                status: row.status,
                size: Self.bytes(usage.referencedBytes),
                usageTerms: row.usageTerms,
                title: row.title,
                summary: row.summary,
                estimatedDownloadBytes: row.estimatedDownloadBytes,
                minimumUnifiedMemoryGB: row.minimumUnifiedMemoryGB,
                recommendedUnifiedMemoryGB: row.recommendedUnifiedMemoryGB,
                supported: row.supported,
                supportReasons: row.supportReasons,
                sourceRepository: row.sourceRepository,
                publisher: row.publisher,
                referencedBytes: usage.referencedBytes,
                reclaimableBytes: usage.reclaimableBytes,
                sharedBytes: usage.sharedBytes,
                externalBytes: usage.externalBytes
            )
        }
    }

    private func updateStatusMessage() {
        if let storageReport {
            statusMessage = "\(installedRows.count) downloaded · \(Self.bytes(storageReport.applicationSupportBytes)) used"
                + " · \(Self.bytes(storageReport.garbageCollectableBytes)) cleanable"
        } else {
            statusMessage = "\(installedRows.count) downloaded · \(rows.count) known"
        }
    }

    private func removalMessage(_ row: StudioModelInventoryRow) -> String {
        guard let reclaimable = row.reclaimableBytes else {
            return "This model references \(row.size). Shared payloads used by other models will be preserved."
        }
        var message = "This model references \(row.size) and will reclaim \(Self.bytes(reclaimable)) now."
        if let shared = row.sharedBytes, shared > 0 {
            message += " \(Self.bytes(shared)) shared with other models will be preserved."
        }
        if let external = row.externalBytes, external > 0 {
            message += " \(Self.bytes(external)) stored outside MereRun will be preserved."
        }
        return message
    }

    private static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
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

/// One model in the catalog list: install dot, name, category/runtime facts, size in the
/// trailing column where the eye expects it.
private struct StudioModelListRow: View {
    let row: StudioModelInventoryRow
    let isSelected: Bool
    let runtime: StudioRuntimeSettings?
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: MereRunTheme.Spacing.sm) {
                Circle()
                    .fill(row.isInstalled ? MereRunTheme.green : MereRunTheme.border)
                    .frame(width: 7, height: 7)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 3) {
                    Text(row.id)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    HStack(spacing: 6) {
                        Text(row.category)
                        if row.usageTerms != nil {
                            Label("terms", systemImage: "doc.text")
                                .labelStyle(.titleAndIcon)
                        }
                        if !row.isInstalled {
                            Text("·")
                            Text(row.status)
                        }
                        if let runtime {
                            if runtime.pinned {
                                Label("pinned", systemImage: "pin.fill")
                                    .labelStyle(.titleAndIcon)
                            }
                            if let alias = runtime.alias {
                                Text("→ \(alias)")
                            }
                        }
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                }

                Spacer(minLength: 8)

                Text(row.displayedSize)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(rowFill)
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                            .strokeBorder(
                                isSelected ? MereRunTheme.accent.opacity(0.5) : Color.clear,
                                lineWidth: 1
                            )
                    }
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityLabel("\(row.id), \(row.category), \(row.status), \(row.displayedSize)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var rowFill: Color {
        if isSelected { return MereRunTheme.accentSoft }
        if hovering { return MereRunTheme.hoverFill }
        return MereRunTheme.surface.opacity(0.35)
    }
}
