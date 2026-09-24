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

// MARK: - Task inspector override

/// The `.cameras` override of the task inspector for 3D ▸ InstantMesh: the ordered views the
/// well holds, numbered and reorderable, then the camera editor over them. The task draft is the
/// source of truth — the views are its `--view` list and the cameras its `--cameras` file — so
/// the Command view, Library restoration, and the argv see exactly what the editor shows. The
/// edited document is kept under the keys the 3D page used, so a document edited there carries
/// over; whenever it changes it is saved as a content-named file (`StudioCameraDocuments`) that
/// `--cameras` points at, and a file the draft names from elsewhere (Use these settings) is read
/// back into the editor.
struct StudioInstantMeshCamerasOverride: View {
    @Binding var draft: StudioTaskDraft
    @EnvironmentObject private var library: StudioLibraryStore
    @StudioStoredValue("3DCreation.cameras") private var cameras = StudioInstantMeshCameraDocument()
    @StudioStoredValue("3DCreation.suppliesCameras") private var suppliesCameras = false
    /// The file the editor last saved, so a `--cameras` change that is its own write is not read back.
    @State private var savedPath = ""
    @State private var message: String?

    private static let draftPage = "3D Creation"
    private static let viewFlag = "--view"
    private static let camerasFlag = "--cameras"

    private var viewSlot: StudioAttachmentSlot? {
        draft.slots.first { $0.id == Self.viewFlag }
    }

    private var views: [String] {
        viewSlot?.paths(in: draft) ?? []
    }

    private var camerasPath: String {
        draft.text(Self.camerasFlag)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            orderedViews
            StudioInstantMeshCameraEditor(
                enabled: $suppliesCameras,
                document: $cameras,
                viewNames: views.map { URL(fileURLWithPath: $0).lastPathComponent },
                message: $message
            )
            if let message {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear(perform: adoptNamedFile)
        .onChange(of: camerasPath) { _, _ in adoptNamedFile() }
        .task(id: SaveKey(enabled: suppliesCameras, document: cameras, viewCount: views.count)) { await save() }
    }

    // MARK: Ordered views

    private var orderedViews: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Ordered views")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Text("\(views.count) / 4 or 6")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(views.count == 4 || views.count == 6 ? MereRunTheme.green : MereRunTheme.textMuted)
                    .accessibilityLabel("\(views.count) of 4 or 6 views")
            }
            ForEach(Array(views.enumerated()), id: \.offset) { index, path in
                viewRow(index: index, path: path)
            }
            if views.count < 6, let viewSlot {
                Button {
                    StudioAttachmentPicker.pick(for: viewSlot, into: &draft)
                } label: {
                    Label("Add views…", systemImage: "photo.stack")
                }
                .buttonStyle(.mereSecondary)
                .controlSize(.small)
            }
        }
    }

    private func viewRow(index: Int, path: String) -> some View {
        let name = URL(fileURLWithPath: path).lastPathComponent
        return HStack(spacing: 7) {
            Text("\(index + 1)")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 18)
            Text(name)
                .font(MereRunTheme.captionFont)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            if index > 0 {
                Button { move(index, to: index - 1) } label: { Image(systemName: "arrow.up") }
                    .buttonStyle(.plain)
                    .help("Move up")
                    .accessibilityLabel("Move \(name) up")
            }
            if index + 1 < views.count {
                Button { move(index, to: index + 1) } label: { Image(systemName: "arrow.down") }
                    .buttonStyle(.plain)
                    .help("Move down")
                    .accessibilityLabel("Move \(name) down")
            }
            Button(role: .destructive) { remove(index) } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain)
                .help("Remove")
                .accessibilityLabel("Remove \(name)")
        }
        .padding(8)
        .background(MereRunTheme.surface)
        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
    }

    private func move(_ index: Int, to destination: Int) {
        var ordered = views
        ordered.swapAt(index, destination)
        setViews(ordered)
    }

    private func remove(_ index: Int) {
        var ordered = views
        ordered.remove(at: index)
        setViews(ordered)
    }

    private func setViews(_ ordered: [String]) {
        guard let viewSlot else { return }
        draft.setAttachmentText(ordered.joined(separator: "\n"), for: viewSlot.storage)
    }

    // MARK: The camera file

    private struct SaveKey: Equatable {
        let enabled: Bool
        let document: StudioInstantMeshCameraDocument
        let viewCount: Int
    }

    /// Keeps `--cameras` current a moment after editing stops: the document as a content-named
    /// file while cameras are on (an incomplete one too, so the run is refused with the reason
    /// rather than run without cameras), nothing while they are off.
    private func save() async {
        guard suppliesCameras else {
            if !camerasPath.isEmpty {
                savedPath = ""
                draft.form[Self.camerasFlag] = .unset
            }
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        do {
            let url = try StudioCameraDocuments.storeDraft(page: Self.draftPage, content: cameras.json())
            // Rows the Library still names (queued Command-view runs included) keep their files.
            StudioCameraDocuments.pruneDrafts(page: Self.draftPage, current: url, referenced: referencedCameraFiles())
            savedPath = url.path
            if camerasPath != url.path { draft.form[Self.camerasFlag] = .text(url.path) }
        } catch {
            message = "Studio could not write the camera file: \(error.localizedDescription)"
        }
    }

    /// A camera file the draft names that this editor did not write — restored from a Library
    /// row, or typed in the Command view — is read into the editor once; one that will not read
    /// is reported.
    private func adoptNamedFile() {
        let path = camerasPath
        guard !path.isEmpty, path != savedPath else { return }
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        do {
            cameras = try StudioInstantMeshCameraDocument.importing(Data(contentsOf: url))
            savedPath = path
            suppliesCameras = true
            message = nil
        } catch {
            message = "The camera file at \(url.lastPathComponent) could not be read into the editor: \(error.localizedDescription)"
        }
    }

    private func referencedCameraFiles() -> Set<String> {
        var referenced = Set(library.items.compactMap { $0.commandDraft?.camerasPath }.filter { !$0.isEmpty })
        for arguments in library.items.compactMap(\.commandArguments) {
            for (index, argument) in arguments.enumerated() where argument == Self.camerasFlag && index + 1 < arguments.count {
                referenced.insert(arguments[index + 1])
            }
        }
        return referenced
    }
}
