import AppKit
import Quartz
import QuickLook

/// Drives the shared macOS Quick Look panel for a single output file. SwiftUI's `.quickLookPreview`
/// is iOS-only, so the GUI previews via `QLPreviewPanel` directly.
@MainActor
final class QuickLookCoordinator: NSObject, QLPreviewPanelDataSource {
    static let shared = QuickLookCoordinator()

    private var url: URL?

    /// Shows (or reloads) Quick Look for `url`. No-op if the panel is unavailable.
    func preview(_ url: URL) {
        self.url = url
        guard let panel = QLPreviewPanel.shared() else { return }
        panel.dataSource = self
        if panel.isVisible {
            panel.reloadData()
        } else {
            panel.makeKeyAndOrderFront(nil)
        }
    }

    nonisolated func numberOfPreviewItems(in panel: QLPreviewPanel) -> Int {
        MainActor.assumeIsolated { url == nil ? 0 : 1 }
    }

    nonisolated func previewPanel(_ panel: QLPreviewPanel, previewItemAt index: Int) -> QLPreviewItem {
        MainActor.assumeIsolated { (url as NSURL?) ?? NSURL() }
    }
}
