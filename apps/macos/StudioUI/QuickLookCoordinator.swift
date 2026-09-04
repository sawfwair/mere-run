import AppKit
import Quartz
import QuickLook
import StudioKit

/// Drives the shared macOS Quick Look panel for a single output file. SwiftUI's `.quickLookPreview`
/// is iOS-only, so the GUI previews via `QLPreviewPanel` directly.
@MainActor
package final class QuickLookCoordinator: NSObject, QLPreviewPanelDataSource {
    package static let shared = QuickLookCoordinator()

    package private(set) var url: URL?

    /// Installs this coordinator as the panel's data source (called from the responder-chain
    /// handshake in MereRunAppDelegate).
    package func install(on panel: QLPreviewPanel) {
        panel.dataSource = self
    }

    /// Shows (or reloads) Quick Look for `url`. The panel resolves its data source via the responder
    /// chain (see MereRunAppDelegate's QLPreviewPanelController methods), so we only stash the URL
    /// and bring the shared panel forward. Returns false if the panel is unavailable.
    @discardableResult
    package func preview(_ url: URL) -> Bool {
        self.url = url
        guard let panel = QLPreviewPanel.shared() else { return false }
        if panel.isVisible {
            panel.reloadData()
        } else {
            panel.makeKeyAndOrderFront(nil)
        }
        return true
    }

    package nonisolated func numberOfPreviewItems(in panel: QLPreviewPanel) -> Int {
        MainActor.assumeIsolated { url == nil ? 0 : 1 }
    }

    package nonisolated func previewPanel(_ panel: QLPreviewPanel, previewItemAt index: Int) -> QLPreviewItem {
        MainActor.assumeIsolated { (url as NSURL?) ?? NSURL() }
    }
}
