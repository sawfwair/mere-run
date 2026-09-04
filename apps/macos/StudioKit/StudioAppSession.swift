import Combine
import Foundation

/// Owns shared services independently of the presence of any particular window.
@MainActor
package final class StudioAppSession: ObservableObject {
    package let controller: MereRunController
    package let library: StudioLibraryStore

    package init() {
        controller = MereRunController(taskSessions: StudioTaskSessions(url: StudioTaskSessions.defaultURL))
        library = StudioLibraryStore()
        library.observe(controller: controller)
        controller.servingMonitor.start(controller: controller)
    }
}
