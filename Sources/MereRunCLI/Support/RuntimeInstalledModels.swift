import Foundation
import MereRunCore

/// Installed API-servable models, found by reading every configured model location.
struct RuntimeInstalledModels: Equatable, Sendable {
    /// Servable model ids mapped to their resolved install path.
    let installPaths: [String: String]
    let locationIssues: [ModelLocationIssue]

    static func scan() -> RuntimeInstalledModels {
        let locationIssues = ModelResolver().locationIssues()
        var installPaths: [String: String] = [:]
        for spec in ManagedModelCatalog.allSpecs where spec.isAPIServableRuntimeModel {
            if let url = spec.managedRuntimeURL() {
                installPaths[spec.id] = url.path
            }
        }
        return RuntimeInstalledModels(installPaths: installPaths, locationIssues: locationIssues)
    }
}

/// Keeps the model-location scan off the request path. The first caller waits for a scan;
/// after that callers get the latest scan at once and a stale one refreshes in the background.
actor RuntimeInstalledModelsCache {
    private let maxAge: TimeInterval
    private let currentDate: @Sendable () -> Date
    private let scan: @Sendable () -> RuntimeInstalledModels
    private var latest: (models: RuntimeInstalledModels, scannedAt: Date)?
    private var refresh: Task<RuntimeInstalledModels, Never>?

    init(
        maxAge: TimeInterval = 10,
        currentDate: @escaping @Sendable () -> Date = { Date() },
        scan: @escaping @Sendable () -> RuntimeInstalledModels = { RuntimeInstalledModels.scan() }
    ) {
        self.maxAge = maxAge
        self.currentDate = currentDate
        self.scan = scan
    }

    func models() async -> RuntimeInstalledModels {
        guard let latest else {
            return await startRefresh().value
        }
        if currentDate().timeIntervalSince(latest.scannedAt) >= maxAge {
            startRefresh()
        }
        return latest.models
    }

    @discardableResult
    private func startRefresh() -> Task<RuntimeInstalledModels, Never> {
        if let refresh {
            return refresh
        }
        let scan = scan
        let task = Task {
            // The scan blocks on filesystem calls; run it on a GCD thread rather than
            // tying up a cooperative-pool thread or this actor.
            let models = await withCheckedContinuation { continuation in
                DispatchQueue.global(qos: .utility).async {
                    continuation.resume(returning: scan())
                }
            }
            finishRefresh(models)
            return models
        }
        refresh = task
        return task
    }

    private func finishRefresh(_ models: RuntimeInstalledModels) {
        latest = (models, currentDate())
        refresh = nil
    }
}
