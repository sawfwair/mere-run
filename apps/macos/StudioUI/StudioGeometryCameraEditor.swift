import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// Vision ▸ Geometry (multi-view) takes optional calibrated cameras. The editor shows one camera
// per view as numeric fields and leaves the JSON to `StudioCameraDocuments`; the page writes the
// file each run needs. Import and export keep files made elsewhere usable. The Vision page PR
// wraps this as the `.cameras` override of the task inspector; the numeric pieces it is built
// from are in `StudioCameraControls.swift`.

/// The cameras for `vision geometry-multiview --cameras`: image size, normalized focal length and
/// principal point, and a world-to-camera rotation and translation per view. Each view's decoded
/// size comes in with it, because the CLI rejects a camera whose image size is not the image's.
struct StudioGeometryCameraEditor: View {
    @Binding var enabled: Bool
    @Binding var document: StudioGeometryCameraDocument
    /// The views in order, with their pixel sizes when readable, so each camera is labelled with
    /// its image and sized to it.
    let views: [StudioCameraView]
    @Binding var message: String?

    var body: some View {
        StudioCameraSection(
            enabled: $enabled,
            count: document.cameras.count,
            viewCount: views.count,
            problems: document.problems(views: views),
            onEnable: { if document.cameras.isEmpty { matchViews() } },
            onMatchViews: matchViews,
            onImport: importDocument,
            onExport: exportDocument,
            canExport: !document.cameras.isEmpty
        ) {
            ForEach($document.cameras) { $camera in
                let index = document.cameras.firstIndex { $0.id == camera.id } ?? 0
                let view = views.indices.contains(index) ? views[index] : nil
                StudioCameraCard(
                    title: "Camera \(index + 1)",
                    subtitle: view?.name ?? "No view",
                    problems: camera.problems(view: view),
                    onRemove: { document.cameras.removeAll { $0.id == camera.id } }
                ) {
                    HStack(alignment: .top, spacing: 14) {
                        StudioNumberGroup("Image size") {
                            StudioIntegerCell(value: $camera.imageWidth, label: "width")
                            Text("×").foregroundStyle(MereRunTheme.textMuted)
                            StudioIntegerCell(value: $camera.imageHeight, label: "height")
                        }
                        StudioNumberGroup("Focal (of size)") {
                            StudioNumberCell(value: $camera.normalizedFX, label: "fx")
                            StudioNumberCell(value: $camera.normalizedFY, label: "fy")
                        }
                        StudioNumberGroup("Center (of size)") {
                            StudioNumberCell(value: $camera.normalizedCX, label: "cx")
                            StudioNumberCell(value: $camera.normalizedCY, label: "cy")
                        }
                    }
                    if let size = view?.pixelSize, size.width != camera.imageWidth || size.height != camera.imageHeight {
                        Button("Use the image's size, \(size.label)") {
                            camera.imageWidth = size.width
                            camera.imageHeight = size.height
                        }
                        .buttonStyle(.mereSecondary)
                        .controlSize(.small)
                    }
                    HStack(alignment: .top, spacing: 14) {
                        StudioNumberGroup("Rotation, world to camera") {
                            StudioMatrixGrid(values: $camera.rotation, columns: 3, label: "rotation")
                        }
                        StudioNumberGroup("Translation") {
                            StudioMatrixGrid(values: $camera.translation, columns: 3, label: "translation")
                        }
                    }
                }
            }
        }
    }

    /// One camera per view, each sized to its image; extra cameras beyond the views are dropped
    /// from the end.
    private func matchViews() {
        let count = views.count
        if document.cameras.count > count {
            document.cameras.removeLast(document.cameras.count - count)
        }
        while document.cameras.count < count {
            document.cameras.append(.identity(size: views[document.cameras.count].pixelSize))
        }
    }

    private func importDocument() {
        guard let url = StudioSpecialistFiles.chooseFile(title: "Import a camera file", allowedContentTypes: [.json]).first else { return }
        do {
            document = try StudioGeometryCameraDocument.importing(Data(contentsOf: url))
            enabled = true
            message = nil
        } catch {
            message = "That file is not a camera document: \(error.localizedDescription)"
        }
    }

    private func exportDocument() {
        guard let url = StudioSpecialistFiles.saveFile(title: "Export the camera file", suggestedName: "cameras.json", allowedContentTypes: [.json]) else {
            return
        }
        do {
            try document.json().write(to: url, options: .atomic)
        } catch {
            message = "Studio could not save the camera file: \(error.localizedDescription)"
        }
    }
}
