import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// Vision ▸ Geometry (multi-view) takes optional calibrated cameras. The editor shows one camera
// per view as numeric fields and leaves the JSON to `StudioCameraDocuments`; the page writes the
// file each run needs. Import and export keep files made elsewhere usable.
// `StudioGeometryCameraOverride` below is the `.cameras` override of the task inspector; the
// numeric pieces the editor is built from are in `StudioCameraControls.swift`.

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
                    // Two groups per row, so the cards fit the inspector column.
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
                    }
                    StudioNumberGroup("Center (of size)") {
                        StudioNumberCell(value: $camera.normalizedCX, label: "cx")
                        StudioNumberCell(value: $camera.normalizedCY, label: "cy")
                    }
                    if let size = view?.pixelSize, size.width != camera.imageWidth || size.height != camera.imageHeight {
                        Button("Use the image's size, \(size.label)") {
                            camera.imageWidth = size.width
                            camera.imageHeight = size.height
                        }
                        .buttonStyle(.mereSecondary)
                        .controlSize(.small)
                    }
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

/// The task inspector's `.cameras` override for Vision ▸ Geometry (multi-view): the editor above
/// over a camera document kept beside the task's draft (`"vision.geometry.geometryCameras"`),
/// writing the saved draft file's path into `--cameras` a moment after editing stops, so the
/// Command view's preview and the run carry a real file. Only a document the CLI would accept is
/// written; while the cameras are off or do not match the views, the flag is left empty and the
/// model estimates them. A camera file picked by hand (the Command view, a restored run) is read
/// into the editor once rather than overwritten.
struct StudioGeometryCameraOverride: View {
    @Binding var draft: StudioTaskDraft
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioTaskScope) private var scope
    @StudioStoredValue("geometryCameras") private var document = StudioGeometryCameraDocument()
    @StudioStoredValue("suppliesCameras") private var enabled = false
    @State private var message: String?
    /// Each view's decoded size by path, read when the list changes rather than per render.
    @State private var viewSizes: [String: StudioPixelSize] = [:]

    /// The draft-file folder the page used, so files it wrote keep working.
    static let page = "Vision Geometry"
    private static let flag = "--cameras"

    /// The ordered views: the repeatable positional the well holds.
    private var paths: [String] {
        StudioTaskSchema.primarySlot(for: draft.templateID)?.paths(in: draft) ?? []
    }

    private var views: [StudioCameraView] {
        paths.map { StudioCameraView(name: URL(fileURLWithPath: $0).lastPathComponent, pixelSize: viewSizes[$0]) }
    }

    private struct DraftKey: Equatable {
        let enabled: Bool
        let document: StudioGeometryCameraDocument
        let views: [StudioCameraView]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if paths.count < 2 {
                Text("Add at least two ordered views to the well; the model solves the cameras between them.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            StudioGeometryCameraEditor(enabled: $enabled, document: $document, views: views, message: $message)
            if let message {
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear(perform: adoptExistingCameras)
        .task(id: paths) { refreshViewSizes() }
        .task(id: DraftKey(enabled: enabled, document: document, views: views)) { await saveDraftCameras() }
    }

    private func refreshViewSizes() {
        viewSizes = Dictionary(uniqueKeysWithValues: paths.compactMap { path in
            StudioPixelSize.of(URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)).map { (path, $0) }
        })
    }

    /// Keeps the flag current: a saved copy of a valid document under a name made from its
    /// content, pruned to the recent few plus every file a Library row's argv still names (a
    /// finished run has its own copy beside its output; a queued one still reads the draft).
    private func saveDraftCameras() async {
        guard enabled, !document.cameras.isEmpty, document.problems(views: views).isEmpty else {
            forgetStudioDraft()
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        do {
            let url = try StudioCameraDocuments.storeDraft(page: Self.page, content: document.json())
            let referenced = Set(library.items.flatMap { item in
                StudioCameraDocuments.referencedPaths(in: item.commandArguments ?? []) + [item.commandDraft?.camerasPath].compactMap { $0 }
            })
            StudioCameraDocuments.pruneDrafts(page: Self.page, current: url, referenced: referenced)
            if draft.text(Self.flag) != url.path { draft.form[Self.flag] = .text(url.path) }
        } catch {
            message = "Studio could not write the camera file: \(error.localizedDescription)"
        }
    }

    /// Clears a file this editor wrote; a path chosen by hand stays.
    private func forgetStudioDraft() {
        let current = draft.text(Self.flag)
        guard !current.isEmpty, StudioCameraDocuments.isDraft(current, page: Self.page) else { return }
        draft.form[Self.flag] = .unset
    }

    /// A camera document from before this editor — the page's own (`VisionLab.geometryCameras`)
    /// or a file the flag already names — is read into the editor once.
    private func adoptExistingCameras() {
        guard document.cameras.isEmpty else { return }
        if let sessions {
            let legacyKey = scope + ".VisionLab.geometryCameras"
            let legacy = sessions.value(for: legacyKey, default: StudioGeometryCameraDocument())
            if !legacy.cameras.isEmpty {
                document = legacy
                enabled = sessions.value(for: scope + ".VisionLab.suppliesCameras", default: true)
                sessions.set(Optional<StudioGeometryCameraDocument>.none, for: legacyKey)
                return
            }
        }
        let current = draft.text(Self.flag)
        guard !current.isEmpty, !StudioCameraDocuments.isDraft(current, page: Self.page) else { return }
        let url = URL(fileURLWithPath: NSString(string: current).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            document = try StudioGeometryCameraDocument.importing(Data(contentsOf: url))
            enabled = true
        } catch {
            message = "The camera file at \(url.lastPathComponent) could not be read into the editor: \(error.localizedDescription)"
        }
    }
}
