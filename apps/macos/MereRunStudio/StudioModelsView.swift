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
    /// The model's context window in tokens, when the inventory reports one. Conversation
    /// threads size their transcript budget from it; nil keeps the fixed default budget.
    let contextWindow: Int?

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
        externalBytes: Int64? = nil,
        contextWindow: Int? = nil
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
        self.contextWindow = contextWindow
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


private enum ModelsMetrics {
    static let listWidth: CGFloat = 320
    static let rowHeight: CGFloat = 40
    static let dot: CGFloat = 8
    static let jobBarHeight: CGFloat = 40
    static let detailPadding = EdgeInsets(top: 22, leading: 24, bottom: 22, trailing: 24)
    static let panelRadius: CGFloat = 10
}

/// Models ▸ Installed: a 320pt list of installed (and in-flight) models beside a detail column
/// of facts, health, performance, and the adapters that target the model, with a job bar for
/// pulls, optimizations, and storage clean-ups pinned to the page bottom.
struct StudioModelsView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel

    let onModelsChanged: () -> Void
    /// Reports installed count and store size so the toolbar subtitle can show them.
    let onInventoryChanged: (StudioModelInventorySummary) -> Void
    /// The domain an adapter is applied to ("Image" in "Use in Image").
    let adapterTargetTitle: String
    let onUseAdapter: (StudioAdapterRow) -> Void
    let onTrain: (CommandTemplateID) -> Void

    init(
        onModelsChanged: @escaping () -> Void,
        onInventoryChanged: @escaping (StudioModelInventorySummary) -> Void = { _ in },
        adapterTargetTitle: String = "Image",
        onUseAdapter: @escaping (StudioAdapterRow) -> Void = { _ in },
        onTrain: @escaping (CommandTemplateID) -> Void = { _ in }
    ) {
        self.onModelsChanged = onModelsChanged
        self.onInventoryChanged = onInventoryChanged
        self.adapterTargetTitle = adapterTargetTitle
        self.onUseAdapter = onUseAdapter
        self.onTrain = onTrain
    }

    @State private var rows: [StudioModelInventoryRow] = []
    @State private var adapters: [StudioAdapterRow] = []
    @State private var selectedID: String?
    @State private var detailText = ""
    @State private var statusMessage = ""
    @State private var searchText = ""
    @State private var selectedFamily: StudioModelFamily?
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
    @State private var optimizeCommandID: UUID?
    @State private var cleanupCommandID: UUID?
    @State private var pendingAlert: StudioModelsAlert?
    @State private var storageReport: StudioModelStorageReport?
    @State private var isCleaningStorage = false
    @State private var infoFactsByID: [String: StudioModelInfoFacts] = [:]
    @State private var runtimeSettingsByID: [String: StudioRuntimeSettings] = [:]
    @State private var runtimeAlias = ""
    @State private var runtimeTTL = ""
    @State private var runtimeMaxContext = ""
    @State private var runtimeMaxTokens = ""
    @State private var runtimeTemperature = ""
    @State private var runtimeTopP = ""
    @State private var runtimeMinP = ""
    @State private var runtimePinned = false
    @State private var showPullSheet = false
    @State private var showJobLog = false
    @State private var showRuntimeSettings = false
    @State private var showDetails = false
    @State private var jobLog = ""

    private var installedRows: [StudioModelInventoryRow] {
        rows.filter(\.isInstalled)
    }

    private var missingRows: [StudioModelInventoryRow] {
        rows.filter { !$0.isInstalled && $0.id != downloadingID }
    }

    /// The page's own job first; otherwise a composer-initiated pull from the Library.
    private var activeJob: StudioModelsJob? {
        if let downloadingID {
            return StudioModelsJob(
                kind: .pull,
                modelID: downloadingID,
                subject: displayName(for: downloadingID),
                progress: downloadProgress,
                isCancelling: cancellingDownloadID == downloadingID
            )
        }
        if let optimizingID {
            return StudioModelsJob(kind: .optimize, modelID: optimizingID, subject: displayName(for: optimizingID))
        }
        if isCleaningStorage {
            return StudioModelsJob(kind: .cleanup, modelID: nil, subject: "model storage")
        }
        return StudioModelsPresenter.libraryPullJob(
            in: library.items,
            rows: rows,
            progressByRequestID: controller.progressByRequestID
        )
    }

    private var pullingIDs: Set<String> {
        guard let job = activeJob, job.kind == .pull, let modelID = job.modelID else { return [] }
        return [modelID]
    }

    private var families: [StudioModelFamily] {
        StudioModelsPresenter.families(in: StudioModelsPresenter.listRows(rows, pullingIDs: pullingIDs))
    }

    private var visibleRows: [StudioModelInventoryRow] {
        StudioModelsPresenter.filter(
            StudioModelsPresenter.listRows(rows, pullingIDs: pullingIDs),
            family: selectedFamily,
            query: searchText
        )
    }

    private var selectedRow: StudioModelInventoryRow? {
        rows.first { $0.id == selectedID }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                listColumn
                    .frame(width: ModelsMetrics.listWidth)
                    .overlay(alignment: .trailing) {
                        Rectangle()
                            .fill(MereRunTheme.border.opacity(0.4))
                            .frame(width: 1)
                    }

                detailColumn
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if let job = activeJob {
                StudioModelsJobBar(
                    job: job,
                    onCancel: { cancel(job) },
                    onLog: { showJobLog.toggle() },
                    showLog: $showJobLog,
                    log: log(for: job)
                )
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            await refresh()
        }
        .onChange(of: visibleRows.map(\.id)) { _, ids in
            if let selectedID, ids.contains(selectedID) { return }
            guard let first = visibleRows.first else {
                selectedID = nil
                detailText = ""
                return
            }
            select(first)
        }
        .sheet(isPresented: $showPullSheet) {
            StudioModelPullSheet(rows: missingRows) { row in
                showPullSheet = false
                requestDownload(row)
            }
        }
        .alert(item: $pendingAlert) { pending in
            switch pending {
            case .download(let row):
                Alert(
                    title: Text("Accept third-party model terms"),
                    message: Text(downloadTermsMessage(row)),
                    primaryButton: .default(Text("Accept & Pull")) {
                        Task { await download(row, acknowledgingUsageTerms: true) }
                    },
                    secondaryButton: .cancel()
                )
            case .removal(let row):
                Alert(
                    title: Text("Remove \(displayName(for: row.id))?"),
                    message: Text(removalMessage(row)),
                    primaryButton: .destructive(Text("Remove")) {
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

    // MARK: - List column

    private var listColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                StudioModelsSearchPill(
                    text: $searchText,
                    placeholder: "Search \(installedRows.count) \(installedRows.count == 1 ? "model" : "models")"
                )

                Button("Pull…") {
                    showPullSheet = true
                }
                .buttonStyle(ModelsPrimaryButtonStyle())
                .disabled(isRefreshing)
                .help("Pull a model into managed storage")
            }
            .padding(EdgeInsets(top: 14, leading: 14, bottom: 8, trailing: 14))

            familyChips
                .padding(EdgeInsets(top: 0, leading: 14, bottom: 10, trailing: 14))

            if visibleRows.isEmpty {
                emptyList
            } else {
                ScrollView {
                    LazyVStack(spacing: 1) {
                        ForEach(visibleRows) { row in
                            StudioModelListRow(
                                name: displayName(for: row.id),
                                meta: StudioModelsPresenter.meta(for: row, status: status(of: row)),
                                status: status(of: row),
                                isSelected: selectedID == row.id
                            ) {
                                select(row)
                            }
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.bottom, 12)
                }
            }

            if !statusMessage.isEmpty {
                Text(statusMessage)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(2)
                    .padding(EdgeInsets(top: 6, leading: 14, bottom: 10, trailing: 14))
                    .accessibilityAddTraits(.updatesFrequently)
            }
        }
    }

    private var familyChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                StudioModelsChip(title: "All", isSelected: selectedFamily == nil) {
                    selectedFamily = nil
                }
                ForEach(families) { family in
                    StudioModelsChip(title: family.title, isSelected: selectedFamily == family) {
                        selectedFamily = selectedFamily == family ? nil : family
                    }
                }
            }
        }
        .accessibilityLabel("Model family")
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
        if isRefreshing, rows.isEmpty { return "Loading models…" }
        if !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "No models match the search."
        }
        if selectedFamily != nil { return "No installed models in this family." }
        return "No models installed yet. Pull one to get started."
    }

    // MARK: - Detail column

    private var detailColumn: some View {
        Group {
            if let selectedRow {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        detailHeader(selectedRow)
                        factsPanel(selectedRow)
                        if selectedRow.isInstalled {
                            HStack(alignment: .top, spacing: 14) {
                                healthPanel(selectedRow)
                                performancePanel(selectedRow)
                            }
                            .fixedSize(horizontal: false, vertical: true)
                            adaptersPanel(selectedRow)
                            runtimePanel(selectedRow)
                            detailsPanel(selectedRow)
                        } else {
                            pullPanel(selectedRow)
                        }
                    }
                    .padding(ModelsMetrics.detailPadding)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            } else {
                emptyDetail
            }
        }
    }

    private func detailHeader(_ row: StudioModelInventoryRow) -> some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(displayName(for: row.id))
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .accessibilityAddTraits(.isHeader)

                HStack(spacing: 6) {
                    StudioModelsTag(title: StudioModelsPresenter.family(of: row).title)
                    if let size = StudioModelsPresenter.sizeChip(for: row, facts: infoFactsByID[row.id]) {
                        StudioModelsTag(title: size)
                    }
                    ForEach(StudioModelsPresenter.defaultDomainTitles(for: row.id), id: \.self) { domain in
                        StudioModelsTag(title: "Default for \(domain)", accent: true)
                    }
                    if row.supported == false {
                        StudioModelsTag(title: "Needs attention", accent: true)
                    }
                }
            }

            Spacer(minLength: 14)

            if row.isInstalled {
                headerActions(row)
            } else if downloadingID == row.id || pullingIDs.contains(row.id) {
                ProgressView()
                    .controlSize(.small)
            }
        }
    }

    private func headerActions(_ row: StudioModelInventoryRow) -> some View {
        HStack(spacing: 8) {
            if loadingInfoID == row.id || removingID == row.id || optimizingID == row.id {
                ProgressView()
                    .controlSize(.small)
            }

            Button("Reveal") {
                Task { await reveal(row) }
            }
            .buttonStyle(ModelsSecondaryButtonStyle())
            .disabled(loadingInfoID != nil)
            .help("Reveal this model's folder in Finder")

            if StudioModelOptimizationCommand.supports(modelID: row.id) {
                Button("Optimize") {
                    Task { await optimize(row, replacing: false) }
                }
                .buttonStyle(ModelsSecondaryButtonStyle())
                .disabled(optimizingID != nil)
                .help("Build the inference-only MiniMax-H3 AdaLN cache")
            }

            Button("Remove…") {
                pendingAlert = .removal(row)
            }
            .buttonStyle(ModelsSecondaryButtonStyle())
            .disabled(removingID != nil)

            Menu {
                Button("Refresh inventory") {
                    Task { await refresh() }
                }
                .disabled(isRefreshing)
                Button("Inspect manifest and components") {
                    showDetails = true
                    Task { await loadInfo(for: row) }
                }
                .disabled(loadingInfoID != nil)
                if StudioModelOptimizationCommand.supports(modelID: row.id) {
                    Button("Rebuild optimization cache") {
                        Task { await optimize(row, replacing: true) }
                    }
                    .disabled(optimizingID != nil)
                }
                Divider()
                Button("Open model store in Finder") {
                    revealStore()
                }
                Button("Clean up storage…") {
                    Task { await previewStorageCleanup() }
                }
                .disabled(isRefreshing || isCleaningStorage)
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 26, height: 26)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .frame(width: 26, height: 26)
            .background {
                RoundedRectangle(cornerRadius: 6)
                    .fill(MereRunTheme.surfaceRaised)
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                    }
            }
            .accessibilityLabel("More model actions")
        }
    }

    private func factsPanel(_ row: StudioModelInventoryRow) -> some View {
        let facts = infoFactsByID[row.id]
        let usage = StudioModelsPresenter.usage(of: row.id, in: library.items)
        return VStack(spacing: 0) {
            if row.isInstalled {
                StudioModelsKeyValueRow(
                    key: "Store",
                    value: StudioModelsPresenter.abbreviatedPath(storePath(for: facts)),
                    mono: true
                )
            }
            if let source = StudioModelsPresenter.sourceLine(for: row) {
                StudioModelsKeyValueRow(key: "Source", value: source, mono: true)
            } else if let publisher = row.publisher, !publisher.isBlank {
                StudioModelsKeyValueRow(key: "Publisher", value: publisher, mono: false)
            }
            if let usage {
                StudioModelsKeyValueRow(key: "Last used", value: StudioModelsPresenter.usageLine(usage), mono: false)
            }
            if row.isInstalled, let verified = facts?.verifiedLine {
                StudioModelsKeyValueRow(key: "Verified", value: verified, mono: false)
            }
            if !row.isInstalled, row.displayedSize != "—" {
                StudioModelsKeyValueRow(key: "Download", value: row.displayedSize, mono: true)
            }
            if let summary = row.summary, !summary.isBlank {
                Text(summary)
                    .font(.system(size: 12))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 6)
            }
        }
        .padding(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
        .modelsPanel()
    }

    private func healthPanel(_ row: StudioModelInventoryRow) -> some View {
        let gate = StudioModelsPresenter.gateLine(in: library.items)
        let facts = infoFactsByID[row.id]
        return VStack(alignment: .leading, spacing: 8) {
            Text("Health")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)

            if let gate {
                StudioModelsStatusLine(text: gate.text, color: gate.ok ? MereRunTheme.green : MereRunTheme.red)
            }
            if let facts, let hasManifest = facts.hasManifest {
                if !hasManifest {
                    StudioModelsStatusLine(text: "Manifest missing", color: MereRunTheme.yellow)
                } else if facts.isValid == false {
                    StudioModelsStatusLine(text: "Validation failed", color: MereRunTheme.red)
                } else {
                    StudioModelsStatusLine(text: "Manifest audit clean", color: MereRunTheme.green)
                }
            }
            if row.supported == false {
                StudioModelsStatusLine(
                    text: row.supportReasons.first ?? "Not supported on this Mac",
                    color: MereRunTheme.yellow
                )
            }
            if gate == nil, facts?.hasManifest == nil, row.supported != false {
                Text("No checks recorded yet")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }

            HStack(spacing: 6) {
                Button("Run gate") {
                    navigation.open(task: .modelsHealth)
                }
                .buttonStyle(ModelsSecondaryButtonStyle())
                Button("Benchmark…") {
                    navigation.open(task: .modelsBenchmarks)
                }
                .buttonStyle(ModelsSecondaryButtonStyle())
            }
            .padding(.top, 4)
        }
        .padding(EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16))
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .modelsPanel()
    }

    private func performancePanel(_ row: StudioModelInventoryRow) -> some View {
        let lastRun = StudioModelsPresenter.lastRunDuration(for: row.id, in: library.items)
        let memory = StudioModelsPresenter.memoryLine(for: row)
        let benchmark = StudioModelsPresenter.benchmarkLine(in: library.items)
        return VStack(alignment: .leading, spacing: 8) {
            Text("Performance")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)

            if let lastRun {
                StudioModelsKeyValueRow(key: "Last run", value: lastRun, mono: true)
            }
            if let memory {
                StudioModelsKeyValueRow(key: "Unified memory", value: memory, mono: true)
            }
            if let benchmark {
                StudioModelsKeyValueRow(key: "Last benchmark", value: benchmark, mono: false)
            }
            if lastRun == nil, memory == nil, benchmark == nil {
                Text("No runs recorded yet")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16))
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .modelsPanel()
    }

    private func adaptersPanel(_ row: StudioModelInventoryRow) -> some View {
        let using = adapters.filter { $0.baseModelID == row.id }
        let training = StudioModelsPresenter.family(of: row).trainingTemplateID
        return VStack(spacing: 0) {
            HStack {
                Text("Adapters using this model")
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Spacer()
                if let training {
                    Button("Train new…") {
                        onTrain(training)
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                }
            }
            .padding(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
            .overlay(alignment: .bottom) {
                Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
            }

            if using.isEmpty {
                Text("No adapters target this model yet.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(EdgeInsets(top: 10, leading: 16, bottom: 12, trailing: 16))
            } else {
                ForEach(using) { adapter in
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(adapter.id)
                                .font(.system(size: 12.5, weight: .medium, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textPrimary)
                                .lineLimit(1)
                            Text(adapterMeta(adapter))
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(MereRunTheme.textMuted)
                                .lineLimit(1)
                        }
                        Spacer(minLength: 10)
                        Button("Use in \(adapterTargetTitle)") {
                            onUseAdapter(adapter)
                        }
                        .buttonStyle(ModelsSecondaryButtonStyle())
                        .disabled(!adapter.installed)
                        .help(adapter.installed ? "Apply this adapter to the composer" : "Pull the adapter first")
                    }
                    .padding(EdgeInsets(top: 10, leading: 16, bottom: 10, trailing: 16))
                    .overlay(alignment: .bottom) {
                        Rectangle().fill(MereRunTheme.border.opacity(0.27)).frame(height: 1)
                    }
                }
            }
        }
        .modelsPanel()
    }

    private func adapterMeta(_ adapter: StudioAdapterRow) -> String {
        StudioModelsPresenter.adapterMeta(
            format: adapter.format,
            byteCount: adapter.byteCount,
            version: adapter.version,
            installed: adapter.installed
        )
    }

    private func runtimePanel(_ row: StudioModelInventoryRow) -> some View {
        StudioModelsDisclosurePanel(title: "Runtime settings", isExpanded: $showRuntimeSettings) {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                HStack(spacing: 6) {
                    Button("Load") {
                        Task { await runtimeLoad(row) }
                    }
                    .buttonStyle(ModelsSecondaryButtonStyle())
                    Button("Unload") {
                        Task { await runtimeUnload(row) }
                    }
                    .buttonStyle(ModelsSecondaryButtonStyle())
                    Button(runtimePinned ? "Unpin" : "Pin") {
                        Task { await saveRuntimeSettings(for: row, pinned: !runtimePinned) }
                    }
                    .buttonStyle(ModelsSecondaryButtonStyle())
                    if loadingRuntimeID == row.id {
                        ProgressView().controlSize(.small)
                    }
                }
                .disabled(loadingRuntimeID != nil)

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
                    Button("Save settings") {
                        Task { await saveRuntimeSettings(for: row, pinned: runtimePinned) }
                    }
                    .buttonStyle(ModelsPrimaryButtonStyle())
                    .disabled(loadingRuntimeID != nil)
                }
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

    private func detailsPanel(_ row: StudioModelInventoryRow) -> some View {
        StudioModelsDisclosurePanel(title: "Details", isExpanded: $showDetails) {
            Text(detailBody(for: row))
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func pullPanel(_ row: StudioModelInventoryRow) -> some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
            if let usageTerms = row.usageTerms {
                Text("Third-party usage terms · acceptance required before pulling")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
                Text(usageTerms.summary)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 8) {
                    ForEach(Array(usageTerms.links.enumerated()), id: \.offset) { index, url in
                        Link("Review terms \(index + 1)", destination: url)
                            .font(MereRunTheme.captionFont)
                    }
                }
            }
            if row.supported == false, !row.supportReasons.isEmpty {
                Text(row.supportReasons.joined(separator: " "))
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 8) {
                if downloadingID == row.id {
                    Button("Cancel pull") {
                        cancelDownload()
                    }
                    .buttonStyle(ModelsSecondaryButtonStyle())
                    .disabled(cancellingDownloadID == row.id)
                } else if pullingIDs.contains(row.id) {
                    Text("Pulling from the composer")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                } else {
                    Button("Pull") {
                        requestDownload(row)
                    }
                    .buttonStyle(ModelsPrimaryButtonStyle())
                    .disabled(downloadingID != nil || isRefreshing)
                }
            }

            if !detailText.isEmpty {
                Text(detailText)
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
        .padding(EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16))
        .frame(maxWidth: .infinity, alignment: .leading)
        .modelsPanel()
    }

    private var emptyDetail: some View {
        VStack(spacing: 10) {
            Image(systemName: "shippingbox")
                .font(.system(size: 28, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(rows.isEmpty && isRefreshing ? "Loading models…" : "Select a model to see its facts, health, and adapters.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Presentation helpers

    /// The store that holds the model: its resolved root's parent when `model info` has reported
    /// one (external stores included), otherwise the configured managed store.
    private func storePath(for facts: StudioModelInfoFacts?) -> String {
        guard let root = facts?.root else { return modelStoreURL.path }
        return URL(fileURLWithPath: root, isDirectory: true).deletingLastPathComponent().path
    }

    private func displayName(for modelID: String) -> String {
        rows.first { $0.id == modelID }.map(StudioModelsPresenter.displayName) ?? modelID
    }

    private func status(of row: StudioModelInventoryRow) -> StudioModelRowStatus {
        StudioModelsPresenter.status(of: row, job: activeJob)
    }

    private func detailBody(for row: StudioModelInventoryRow) -> String {
        if detailText.isEmpty {
            return loadingInfoID == row.id ? "Loading manifest, validation, and component paths…" : "No details loaded."
        }
        return detailText
    }

    private func log(for job: StudioModelsJob) -> String {
        if let itemID = job.libraryItemID {
            let item = library.items.first { $0.id == itemID }
            return item?.outputText ?? "Pull started from the composer. Its log is in the Library row."
        }
        return jobLog.isEmpty ? "Waiting for output…" : jobLog
    }

    private func cancel(_ job: StudioModelsJob) {
        if let itemID = job.libraryItemID {
            _ = controller.cancel(requestID: itemID)
            return
        }
        switch job.kind {
        case .pull:
            cancelDownload()
        case .optimize:
            if let optimizeCommandID { _ = controller.cancelUtilityCommand(optimizeCommandID) }
        case .cleanup:
            if let cleanupCommandID { _ = controller.cancelUtilityCommand(cleanupCommandID) }
        }
    }

    private func select(_ row: StudioModelInventoryRow) {
        selectedID = row.id
        detailText = ""
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

    // MARK: - Commands

    @MainActor
    private func download(_ row: StudioModelInventoryRow, acknowledgingUsageTerms: Bool) async {
        let modelID = row.id
        downloadingID = modelID
        cancellingDownloadID = nil
        downloadProgress = nil
        downloadProgressOutput = ""
        jobLog = ""
        let commandID = UUID()
        downloadCommandID = commandID
        statusMessage = ""
        selectedID = modelID
        detailText = ""

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
                jobLog = StudioModelDownloadCommand.appendingOutput(chunk, to: jobLog)
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
            statusMessage = "Cancelled pull for \(displayName(for: modelID)). The partial payload resumes on the next pull."
            return
        }
        guard result.exitCode == 0 else {
            statusMessage = "Could not pull \(displayName(for: modelID))"
            return
        }

        let keepSelection = selectedID == modelID
        onModelsChanged()
        await refresh()
        if keepSelection, let installed = rows.first(where: { $0.id == modelID }) {
            select(installed)
        }
    }

    private func cancelDownload() {
        guard let commandID = downloadCommandID,
              let modelID = downloadingID,
              controller.cancelUtilityCommand(commandID) else {
            return
        }
        cancellingDownloadID = modelID
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
        let adapterResult = await controller.utilityCommandResult(args: ["adapter", "list", "--json"])
        if adapterResult.exitCode == 0,
           let data = StudioOperationsJSON.objectData(adapterResult.stdout),
           let payload = try? JSONDecoder().decode(StudioAdapterCatalogPayload.self, from: data) {
            adapters = payload.adapters
        }
        isRefreshing = false
        statusMessage = ""
        onInventoryChanged(StudioModelInventorySummary(
            installedCount: installedRows.count,
            storageBytes: storageReport?.applicationSupportBytes
        ))
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
        var args = ["model", "runtime", "set", row.id, pinned ? "--pinned" : "--unpinned"]
        appendRuntimeSettingArgs(to: &args)
        let result = await controller.utilityCommandResult(args: args)
        loadingRuntimeID = nil
        if result.exitCode == 0 {
            statusMessage = "Saved runtime settings for \(displayName(for: row.id))"
            await loadRuntimeSettings(for: row)
        } else {
            statusMessage = "Could not save runtime settings for \(displayName(for: row.id))"
            detailText = result.outputText
            showDetails = true
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
        let name = displayName(for: row.id)
        do {
            var request = URLRequest(url: controller.runtimeURL(path: "/runtime/models/\(row.id)/\(action)"))
            request.httpMethod = "POST"
            if let authorization = controller.runtimeAuthorizationHeader {
                request.setValue(authorization, forHTTPHeaderField: "Authorization")
            }
            let (_, response) = try await URLSession.shared.data(for: request)
            let http = response as? HTTPURLResponse
            if (200..<300).contains(http?.statusCode ?? 500) {
                statusMessage = "\(action.capitalized)ed \(name)"
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
        let result = await controller.utilityCommandResult(args: ["model", "info", row.id, "--components"])
        loadingInfoID = nil
        if result.exitCode == 0 {
            infoFactsByID[row.id] = StudioModelsPresenter.facts(fromInfo: result.stdout)
        } else {
            statusMessage = "Could not inspect \(displayName(for: row.id))"
        }
        if selectedID == row.id {
            detailText = result.outputText
        }
    }

    @MainActor
    private func reveal(_ row: StudioModelInventoryRow) async {
        guard row.isInstalled else { return }
        if let root = infoFactsByID[row.id]?.root {
            reveal(url: URL(fileURLWithPath: root, isDirectory: true))
            return
        }
        loadingInfoID = row.id
        let result = await controller.utilityCommandResult(args: ["model", "info", row.id])
        loadingInfoID = nil
        guard let root = StudioModelInventoryParser.modelRoot(from: result.outputText) else {
            statusMessage = "Could not resolve the \(displayName(for: row.id)) folder"
            detailText = result.outputText
            showDetails = true
            return
        }
        reveal(url: root)
    }

    @MainActor
    private func purge(_ row: StudioModelInventoryRow) async {
        removingID = row.id
        let result = await controller.utilityCommandResult(args: ["model", "remove", row.id, "--force"])
        removingID = nil
        detailText = result.outputText
        if result.exitCode == 0 {
            infoFactsByID[row.id] = nil
            onModelsChanged()
            await refresh()
        } else {
            statusMessage = "Could not remove \(displayName(for: row.id))"
            showDetails = true
        }
    }

    @MainActor
    private func optimize(_ row: StudioModelInventoryRow, replacing: Bool) async {
        optimizingID = row.id
        let commandID = UUID()
        optimizeCommandID = commandID
        jobLog = replacing ? "Rebuilding the MiniMax-H3 inference cache…\n" : "Preparing the MiniMax-H3 inference cache…\n"
        let result = await controller.utilityCommandResult(
            args: StudioModelOptimizationCommand.arguments(modelID: row.id, replacing: replacing),
            commandID: commandID,
            onOutput: { chunk in
                jobLog = StudioModelDownloadCommand.appendingOutput(chunk, to: jobLog)
            }
        )
        optimizingID = nil
        optimizeCommandID = nil
        if selectedID == row.id {
            detailText = result.outputText.isEmpty ? jobLog : result.outputText
        }
        statusMessage = result.exitCode == 0
            ? "Optimized \(displayName(for: row.id))"
            : "Could not optimize \(displayName(for: row.id))"
    }

    @MainActor
    private func previewStorageCleanup() async {
        isCleaningStorage = true
        let commandID = UUID()
        cleanupCommandID = commandID
        jobLog = "Inspecting model storage…\n"
        let command = await controller.utilityCommandResult(args: ["model", "gc", "--json"], commandID: commandID)
        isCleaningStorage = false
        cleanupCommandID = nil
        guard command.exitCode == 0,
              let data = command.stdout.data(using: .utf8),
              let output = try? JSONDecoder().decode(StudioModelGarbageCollectOutput.self, from: data) else {
            statusMessage = "Could not inspect model storage"
            detailText = command.outputText
            showDetails = true
            return
        }
        guard !output.plan.items.isEmpty else {
            statusMessage = "Model storage is clean"
            return
        }
        pendingAlert = .cleanup(output.plan)
    }

    @MainActor
    private func cleanStorage() async {
        isCleaningStorage = true
        let commandID = UUID()
        cleanupCommandID = commandID
        jobLog = "Cleaning model storage…\n"
        let command = await controller.utilityCommandResult(
            args: ["model", "gc", "--force", "--json"],
            commandID: commandID,
            onOutput: { chunk in
                jobLog = StudioModelDownloadCommand.appendingOutput(chunk, to: jobLog)
            }
        )
        isCleaningStorage = false
        cleanupCommandID = nil
        guard command.exitCode == 0,
              let data = command.stdout.data(using: .utf8),
              let output = try? JSONDecoder().decode(StudioModelGarbageCollectOutput.self, from: data),
              let result = output.result else {
            statusMessage = "Could not clean model storage"
            detailText = command.outputText
            showDetails = true
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

// MARK: - Pull sheet

/// The models that are not installed yet, each with a Pull action. Usage-terms acceptance and
/// the progress feedback stay with the Installed page: it starts the pull and shows the job bar.
struct StudioModelPullSheet: View {
    let rows: [StudioModelInventoryRow]
    let onPull: (StudioModelInventoryRow) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""
    @State private var selectedFamily: StudioModelFamily?

    private var visibleRows: [StudioModelInventoryRow] {
        StudioModelsPresenter.filter(rows, family: selectedFamily, query: searchText)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Pull a model")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textPrimary)
                    Text("\(rows.count) available · downloads into managed storage")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(ModelsSecondaryButtonStyle())
                    .keyboardShortcut(.cancelAction)
            }
            .padding(EdgeInsets(top: 16, leading: 16, bottom: 10, trailing: 16))

            StudioModelsSearchPill(text: $searchText, placeholder: "Search \(rows.count) models")
                .padding(EdgeInsets(top: 0, leading: 16, bottom: 8, trailing: 16))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    StudioModelsChip(title: "All", isSelected: selectedFamily == nil) { selectedFamily = nil }
                    ForEach(StudioModelsPresenter.families(in: rows)) { family in
                        StudioModelsChip(title: family.title, isSelected: selectedFamily == family) {
                            selectedFamily = selectedFamily == family ? nil : family
                        }
                    }
                }
            }
            .padding(EdgeInsets(top: 0, leading: 16, bottom: 10, trailing: 16))

            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)

            if visibleRows.isEmpty {
                Text(rows.isEmpty ? "Every known model is installed." : "No models match.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 1) {
                        ForEach(visibleRows) { row in
                            pullRow(row)
                        }
                    }
                    .padding(8)
                }
            }
        }
        .frame(width: 560, height: 520)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private func pullRow(_ row: StudioModelInventoryRow) -> some View {
        HStack(spacing: 10) {
            Circle()
                .fill(row.supported == false ? MereRunTheme.yellow : MereRunTheme.textMuted)
                .frame(width: ModelsMetrics.dot, height: ModelsMetrics.dot)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(StudioModelsPresenter.displayName(row))
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                Text(pullMeta(row))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(row.supported == false ? MereRunTheme.yellow : MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            Spacer(minLength: 10)
            Button("Pull") { onPull(row) }
                .buttonStyle(ModelsSecondaryButtonStyle())
                .help(row.usageTerms == nil ? "Pull \(row.id)" : "Requires accepting third-party usage terms")
        }
        .padding(.horizontal, 10)
        .frame(minHeight: ModelsMetrics.rowHeight)
        .accessibilityElement(children: .combine)
    }

    private func pullMeta(_ row: StudioModelInventoryRow) -> String {
        var parts = [StudioModelsPresenter.family(of: row).title]
        if row.displayedSize != "—" { parts.append("\(row.displayedSize) download") }
        if row.usageTerms != nil { parts.append("third-party terms") }
        if row.supported == false, let reason = row.supportReasons.first { parts.append(reason) }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Job bar

/// The page-bottom job bar: dot, label, muted detail, a ≤260pt progress track, Cancel, Log.
private struct StudioModelsJobBar: View {
    let job: StudioModelsJob
    let onCancel: () -> Void
    let onLog: () -> Void
    @Binding var showLog: Bool
    let log: String

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(MereRunTheme.accent)
                .frame(width: ModelsMetrics.dot, height: ModelsMetrics.dot)
                .accessibilityHidden(true)
            Text(job.label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
            if let detail = job.detail {
                Text(detail)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            StudioModelsProgressTrack(fraction: job.fraction)
                .frame(maxWidth: 260)
            Spacer(minLength: 0)
            Button("Cancel", action: onCancel)
                .buttonStyle(ModelsSecondaryButtonStyle())
                .disabled(job.isCancelling)
            Button("Log", action: onLog)
                .buttonStyle(ModelsSecondaryButtonStyle())
                .popover(isPresented: $showLog, arrowEdge: .bottom) {
                    ScrollView {
                        Text(log)
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(MereRunTheme.textSecondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                            .padding(12)
                    }
                    .frame(width: 520, height: 280)
                    .background(MereRunTheme.surface)
                }
        }
        .padding(.horizontal, 16)
        .frame(height: ModelsMetrics.jobBarHeight)
        .frame(maxWidth: .infinity)
        .background(MereRunTheme.background)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.53)).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(job.label). \(job.detail ?? "")")
    }
}

/// A 4pt track; indeterminate progress shows a soft animated sweep, and none under Reduce Motion.
private struct StudioModelsProgressTrack: View {
    let fraction: Double?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var sweep = false

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(MereRunTheme.surfaceRaised)
                if let fraction {
                    Capsule()
                        .fill(MereRunTheme.accent)
                        .frame(width: max(4, proxy.size.width * min(max(fraction, 0), 1)))
                } else if !reduceMotion {
                    Capsule()
                        .fill(MereRunTheme.accent.opacity(0.7))
                        .frame(width: proxy.size.width * 0.3)
                        .offset(x: sweep ? proxy.size.width * 0.7 : 0)
                        .animation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true), value: sweep)
                        .onAppear { sweep = true }
                } else {
                    Capsule().fill(MereRunTheme.accent.opacity(0.35))
                }
            }
        }
        .frame(height: 4)
        .accessibilityElement()
        .accessibilityLabel("Progress")
        .accessibilityValue(fraction.map { "\(Int(($0 * 100).rounded())) percent" } ?? "in progress")
    }
}

// MARK: - Pieces

/// One model in the list: an 8pt status dot, the display name, and "Family · size".
private struct StudioModelListRow: View {
    let name: String
    let meta: String
    let status: StudioModelRowStatus
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Circle()
                    .fill(status.color)
                    .frame(width: ModelsMetrics.dot, height: ModelsMetrics.dot)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 1) {
                    Text(name)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(meta)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .frame(height: ModelsMetrics.rowHeight)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(isSelected ? MereRunTheme.accentSoft : hovering ? MereRunTheme.hoverFill : Color.clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityLabel("\(name), \(meta), \(status.accessibilityDescription)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

/// The 28pt capsule search field with a leading glass.
private struct StudioModelsSearchPill: View {
    @Binding var text: String
    let placeholder: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundStyle(MereRunTheme.textPrimary)
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 11))
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .frame(maxWidth: .infinity)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }
}

/// A 24pt filter chip; selected chips take the accent wash and accent text.
private struct StudioModelsChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(isSelected ? MereRunTheme.accent : MereRunTheme.textPrimary)
                .padding(.horizontal, 9)
                .frame(height: 24)
                .background {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(isSelected ? MereRunTheme.accentSoft : MereRunTheme.surfaceRaised)
                }
                .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

/// A 20pt fact tag under the detail title ("Image", "Q4 · 2.1 GB", "Default for Image").
private struct StudioModelsTag: View {
    let title: String
    var accent = false

    var body: some View {
        Text(title)
            .font(.system(size: 10.5, weight: .medium))
            .foregroundStyle(accent ? MereRunTheme.accent : MereRunTheme.textPrimary)
            .padding(.horizontal, 9)
            .frame(height: 20)
            .background {
                RoundedRectangle(cornerRadius: 6)
                    .fill(accent ? MereRunTheme.accentSoft : MereRunTheme.surfaceRaised)
            }
    }
}

/// A key on the left, a value on the right, 24pt tall.
private struct StudioModelsKeyValueRow: View {
    let key: String
    let value: String
    let mono: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(key)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
            Spacer(minLength: 12)
            Text(value)
                .font(.system(size: 12, weight: .medium, design: mono ? .monospaced : .default))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
        .frame(minHeight: 24)
        .accessibilityElement(children: .combine)
    }
}

/// A status dot beside a sentence, for the Health panel.
private struct StudioModelsStatusLine: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(color)
                .frame(width: ModelsMetrics.dot, height: ModelsMetrics.dot)
                .accessibilityHidden(true)
            Text(text)
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(2)
        }
        .accessibilityElement(children: .combine)
    }
}

/// A panel whose body folds away behind its title, for the secondary editors under the fold.
private struct StudioModelsDisclosurePanel<Content: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(MereRunTheme.Motion.standard) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Text(title)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textPrimary)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .rotationEffect(.degrees(isExpanded ? 0 : -90))
                }
                .padding(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(isExpanded ? [.isButton, .isSelected] : .isButton)
            .accessibilityHint(isExpanded ? "Collapse" : "Expand")

            if isExpanded {
                content()
                    .padding(EdgeInsets(top: 0, leading: 16, bottom: 14, trailing: 16))
            }
        }
        .modelsPanel()
    }
}

/// `btnPrimary`: 28pt, accent fill, on-accent 13pt semibold text, 6pt radius.
private struct ModelsPrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(MereRunTheme.onAccent)
            .padding(.horizontal, 14)
            .frame(height: 28)
            .background {
                RoundedRectangle(cornerRadius: 6)
                    .fill(MereRunTheme.accent.opacity(configuration.isPressed ? 0.8 : 1))
            }
            .opacity(isEnabled ? 1 : 0.5)
            .contentShape(RoundedRectangle(cornerRadius: 6))
    }
}

/// `btnSecondary`: 26pt, raised surface, hairline border, 11.5pt medium text, 6pt radius.
private struct ModelsSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 11.5, weight: .medium))
            .foregroundStyle(MereRunTheme.textPrimary)
            .padding(.horizontal, 10)
            .frame(height: 26)
            .background {
                RoundedRectangle(cornerRadius: 6)
                    .fill(configuration.isPressed ? MereRunTheme.accentSoft : MereRunTheme.surfaceRaised)
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                    }
            }
            .opacity(isEnabled ? 1 : 0.5)
            .contentShape(RoundedRectangle(cornerRadius: 6))
    }
}

private extension View {
    /// `panel`: surface fill, hairline border, 10pt radius.
    func modelsPanel() -> some View {
        background {
            RoundedRectangle(cornerRadius: ModelsMetrics.panelRadius)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: ModelsMetrics.panelRadius)
                        .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }
}
