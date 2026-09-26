import Combine
import Foundation

/// Owns shared services independently of the presence of any particular window.
@MainActor
package final class StudioAppSession: ObservableObject {
    package let controller: MereRunController
    package let library: StudioLibraryStore
    /// Running plus queued runs, for the Dock badge and the menu bar extra.
    package let runQueue = StudioRunQueueCounter()

    package init() {
        controller = MereRunController(
            secretStore: KeychainSecretStore(),
            taskSessions: StudioTaskSessions(url: StudioTaskSessions.defaultURL)
        )
        library = StudioLibraryStore()
        library.observe(controller: controller)
        runQueue.attach(controller.jobs)
        controller.servingMonitor.start(controller: controller)
        controller.machineMonitor.start()
        _ = controller.localServer
        _ = controller.residentServers
    }
}
