import Combine
import Foundation

/// Inventory and downloads outlive the Models page. Process state belongs to JobStore.
@MainActor
package final class StudioModelStore: ObservableObject {
    @Published package private(set) var rows: [StudioModelInventoryRow] = []
    @Published package private(set) var storage: StudioModelStorageReport?
    @Published package private(set) var isRefreshing = false
    @Published package private(set) var error: String?
    private weak var controller: MereRunController?
    private var subscriptions = Set<AnyCancellable>()
    private var generation = UUID()
    private var metadata: [String: StudioModelCatalogMetadata] = [:]
    private var snapshotContext: [String]?

    package init(controller: MereRunController) {
        self.controller = controller
        controller.jobs.events.sink { [weak self, weak controller] event in
            let job: Job
            switch event {
            case .started(let value), .changed(let value), .output(let value, _, _), .finished(let value, _): job = value
            }
            guard job.request.templateID == .modelPull, controller?.usesCurrentConfiguration(job) == true else { return }
            self?.objectWillChange.send()
            if case .finished = event {
                self?.storage = nil
                controller?.refreshRequestedReadiness()
                Task { [weak self] in await self?.refresh() }
            }
        }.store(in: &subscriptions)
    }

    package var downloads: [Job] {
        guard let controller else { return [] }
        return controller.jobs.all.filter {
            $0.request.templateID == .modelPull && $0.state.isActive && controller.usesCurrentConfiguration($0)
        }
    }

    package var lastCompletedDownload: Job? {
        guard let controller else { return nil }
        return controller.jobs.all.filter {
            $0.request.templateID == .modelPull && controller.usesCurrentConfiguration($0)
        }.compactMap { job in
            job.result.map { (job: job, completedAt: $0.completedAt) }
        }.max { $0.completedAt < $1.completedAt }?.job
    }

    package var downloadMessage: String? {
        guard let job = lastCompletedDownload else { return nil }
        if case .cancelled = job.state { return "Download cancelled. Pull the model again to resume." }
        return job.exitCode == 0 ? "Download complete." : "Download failed. Open the log, then pull the model again."
    }

    package func download(modelID: String) -> Job? {
        downloads.first { $0.request.draft?.model == modelID }
    }

    @discardableResult
    package func startPull(_ request: StudioRunRequest) -> Bool {
        guard let controller, request.templateID == .modelPull else { return false }
        if download(modelID: request.draft.model) != nil { return true }
        return controller.run(studio: request)
    }

    package func cancelDownload(modelID: String) {
        guard let job = download(modelID: modelID) else { return }
        controller?.jobs.cancel(job.id)
        objectWillChange.send()
    }

    /// A refresh publishes one complete snapshot. Older requests cannot overwrite a newer one
    /// or publish data read from a different CLI or model location.
    package func refresh(includingStorage: Bool = false) async {
        guard let controller else { return }
        let context = [controller.cliPath, controller.modelsRoot, controller.hubCache]
        let token = UUID()
        generation = token
        isRefreshing = true
        if snapshotContext != context {
            rows = []
            storage = nil
            metadata = [:]
            snapshotContext = context
        }
        defer { if generation == token { isRefreshing = false } }
        let inventory = await controller.utilityCommandResult(args: ["model", "list", "--json"])
        guard isCurrent(token, context: context) else { return }
        guard inventory.exitCode == 0 else {
            error = "Could not refresh models. Check the CLI and model location, then refresh."
            return
        }
        let freshRows: [StudioModelInventoryRow]
        do {
            freshRows = try StudioModelInventoryParser.decodeRows(from: inventory.stdout)
        } catch {
            self.error = "Could not read the model inventory. Check the CLI version, then refresh."
            return
        }
        var freshMetadata = metadata
        if freshMetadata.isEmpty || includingStorage {
            let capabilities = await controller.utilityCommandResult(args: ["model", "capabilities", "--all", "--json"])
            guard isCurrent(token, context: context) else { return }
            if capabilities.exitCode == 0 {
                freshMetadata = StudioModelCatalogParser.metadataByID(from: capabilities.stdout)
            }
        }
        var freshStorage = storage
        if includingStorage {
            let result = await controller.utilityCommandResult(args: ["model", "storage", "--json"])
            guard isCurrent(token, context: context) else { return }
            freshStorage = result.exitCode == 0
                ? try? JSONDecoder().decode(StudioModelStorageReport.self, from: Data(result.stdout.utf8)) : nil
        }
        metadata = freshMetadata
        storage = freshStorage
        rows = applyingStorage(freshStorage, to: StudioModelCatalogParser.applying(freshMetadata, to: freshRows))
        error = nil
    }

    private func isCurrent(_ token: UUID, context: [String]) -> Bool {
        guard let controller else { return false }
        return generation == token && context == [controller.cliPath, controller.modelsRoot, controller.hubCache]
    }

    private func applyingStorage(_ report: StudioModelStorageReport?, to rows: [StudioModelInventoryRow]) -> [StudioModelInventoryRow] {
        guard let report else { return rows }
        let usageByID = Dictionary(uniqueKeysWithValues: report.models.map { ($0.id, $0) })
        return rows.map { row in
            guard row.isInstalled, let usage = usageByID[row.id], usage.installed else { return row }
            return StudioModelInventoryRow(
                id: row.id, category: row.category, status: row.status,
                size: ByteCountFormatter.string(fromByteCount: usage.referencedBytes, countStyle: .file),
                usageTerms: row.usageTerms, title: row.title, summary: row.summary,
                estimatedDownloadBytes: row.estimatedDownloadBytes,
                minimumUnifiedMemoryGB: row.minimumUnifiedMemoryGB,
                recommendedUnifiedMemoryGB: row.recommendedUnifiedMemoryGB,
                supported: row.supported, supportReasons: row.supportReasons,
                sourceRepository: row.sourceRepository, publisher: row.publisher,
                referencedBytes: usage.referencedBytes, reclaimableBytes: usage.reclaimableBytes,
                sharedBytes: usage.sharedBytes, externalBytes: usage.externalBytes, contextWindow: row.contextWindow)
        }
    }
}
