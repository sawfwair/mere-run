import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// 3D ▸ InstantMesh takes optional calibrated cameras for its ordered views. The editor shows one
// camera per view as numeric fields and leaves the JSON to `StudioCameraDocuments`; the page
// writes the file each run needs. The 3D page PR wraps this as the `.cameras` override of the
// task inspector; the numeric pieces it is built from are in `StudioCameraControls.swift`.

/// The cameras for `image reconstruct-3d-multiview --cameras`: a 3 × 4 camera-to-world pose and
/// `fx, fy, cx, cy` per view. InstantMesh resizes every view to its conditioning size, so image
/// sizes play no part here.
struct StudioInstantMeshCameraEditor: View {
    @Binding var enabled: Bool
    @Binding var document: StudioInstantMeshCameraDocument
    let viewNames: [String]
    @Binding var message: String?

    var body: some View {
        StudioCameraSection(
            enabled: $enabled,
            count: document.cameras.count,
            viewCount: viewNames.count,
            problems: document.problems(viewCount: viewNames.count),
            onEnable: { if document.cameras.isEmpty { matchViews() } },
            onMatchViews: matchViews,
            onImport: importDocument,
            onExport: exportDocument,
            canExport: !document.cameras.isEmpty
        ) {
            ForEach($document.cameras) { $camera in
                let index = document.cameras.firstIndex { $0.id == camera.id } ?? 0
                StudioCameraCard(
                    title: "Camera \(index + 1)",
                    subtitle: viewNames.indices.contains(index) ? viewNames[index] : "No view",
                    problems: camera.problems,
                    onRemove: { document.cameras.removeAll { $0.id == camera.id } }
                ) {
                    if camera.values.count == 16 {
                        HStack(alignment: .top, spacing: 14) {
                            StudioNumberGroup("Pose, camera to world") {
                                StudioMatrixGrid(values: $camera.values, columns: 4, range: 0..<12, label: "pose")
                            }
                            VStack(alignment: .leading, spacing: 8) {
                                StudioNumberGroup("Focal") {
                                    StudioNumberCell(value: $camera.values[12], label: "fx")
                                    StudioNumberCell(value: $camera.values[13], label: "fy")
                                }
                                StudioNumberGroup("Center") {
                                    StudioNumberCell(value: $camera.values[14], label: "cx")
                                    StudioNumberCell(value: $camera.values[15], label: "cy")
                                }
                            }
                        }
                    } else {
                        Button("Reset to the starting camera") { camera = .example }
                            .buttonStyle(.mereSecondary)
                            .controlSize(.small)
                    }
                }
            }
        }
    }

    private func matchViews() {
        let count = viewNames.count
        if document.cameras.count > count {
            document.cameras.removeLast(document.cameras.count - count)
        }
        while document.cameras.count < count {
            document.cameras.append(.example)
        }
    }

    private func importDocument() {
        guard let url = StudioSpecialistFiles.chooseFile(title: "Import a camera file", allowedContentTypes: [.json]).first else { return }
        do {
            document = try StudioInstantMeshCameraDocument.importing(Data(contentsOf: url))
            enabled = true
            message = nil
        } catch {
            message = "That file is not an InstantMesh camera document: \(error.localizedDescription)"
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
