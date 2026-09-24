import AppKit
import Quartz
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

package enum StudioFilePanels {
    @MainActor
    static func chooseFile(
        title: String,
        allowedContentTypes: [UTType] = [],
        allowsMultipleSelection: Bool = false
    ) -> [URL] {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = allowsMultipleSelection
        if !allowedContentTypes.isEmpty {
            panel.allowedContentTypes = allowedContentTypes
        }
        return panel.runModal() == .OK ? panel.urls : []
    }

    @MainActor
    static func chooseDirectory(title: String) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        return panel.runModal() == .OK ? panel.url : nil
    }

    @MainActor
    package static func saveFile(
        title: String,
        suggestedName: String,
        allowedContentTypes: [UTType] = []
    ) -> URL? {
        let panel = NSSavePanel()
        panel.title = title
        panel.nameFieldStringValue = suggestedName
        if !allowedContentTypes.isEmpty {
            panel.allowedContentTypes = allowedContentTypes
        }
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// One output file in the domain's configured folder, stamped with the display clock so
    /// snapshot boards render a stable path.
    @MainActor
    static func outputFile(domain: StudioDomain, name: String, fileExtension: String) -> URL {
        StudioOutputLocation.specialistFile(
            domain: domain,
            name: name,
            fileExtension: fileExtension,
            now: StudioDisplayClock.now
        )
    }
}

struct StudioEmbeddedQuickLookPreview: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> QLPreviewView {
        let view = QLPreviewView(frame: .zero, style: .normal)!
        view.autostarts = true
        view.previewItem = url as NSURL
        return view
    }

    func updateNSView(_ view: QLPreviewView, context: Context) {
        view.previewItem = url as NSURL
    }
}
