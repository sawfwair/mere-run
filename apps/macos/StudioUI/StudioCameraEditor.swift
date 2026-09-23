import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// Vision ▸ Geometry (multi-view) and 3D ▸ InstantMesh take optional calibrated cameras. These
// editors show one camera per view as numeric fields and leave the JSON to `StudioCameraDocuments`;
// the page writes the file each run needs. Import and export keep files made elsewhere usable.

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

// MARK: - Pieces

/// The section frame both editors share: the toggle, the cards, a Match views button when the
/// counts differ, the import and export menu, and the CLI's checks.
private struct StudioCameraSection<Cards: View>: View {
    @Binding var enabled: Bool
    let count: Int
    let viewCount: Int
    let problems: [String]
    let onEnable: () -> Void
    let onMatchViews: () -> Void
    let onImport: () -> Void
    let onExport: () -> Void
    let canExport: Bool
    @ViewBuilder let cards: () -> Cards

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Toggle("Supply calibrated cameras", isOn: $enabled)
                    .onChange(of: enabled) { _, enabled in if enabled { onEnable() } }
                Spacer()
                Menu {
                    Button("Import camera file…", action: onImport)
                    Button("Export camera file…", action: onExport)
                        .disabled(!canExport)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Import and export a camera file")
                .accessibilityLabel("Camera file actions")
            }
            if enabled {
                Text("One camera per view, in view order. Without cameras the model estimates them.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
                cards()
                if count != viewCount {
                    Button {
                        onMatchViews()
                    } label: {
                        Label(
                            count < viewCount ? "Add cameras for the other views" : "Drop the extra cameras",
                            systemImage: count < viewCount ? "plus" : "minus"
                        )
                    }
                    .buttonStyle(.mereSecondary)
                    .controlSize(.small)
                }
                ForEach(problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.circle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
            }
        }
    }
}

private struct StudioCameraCard<Fields: View>: View {
    let title: String
    let subtitle: String
    let problems: [String]
    let onRemove: () -> Void
    @ViewBuilder let fields: () -> Fields

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                Text(subtitle)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button(action: onRemove) {
                    Image(systemName: "minus.circle")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.mereIcon)
                .help("Remove")
                .accessibilityLabel("Remove \(title.lowercased())")
            }
            fields()
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(
                            problems.isEmpty ? MereRunTheme.border.opacity(0.55) : MereRunTheme.yellow.opacity(0.6),
                            lineWidth: 1
                        )
                }
        }
    }
}

private struct StudioNumberGroup<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    init(_ title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 10.5))
                .foregroundStyle(MereRunTheme.textMuted)
            HStack(spacing: 4) { content() }
        }
    }
}

/// A grid of number cells over a slice of a flat row-major array.
private struct StudioMatrixGrid: View {
    @Binding var values: [Double]
    let columns: Int
    var range: Range<Int>?
    let label: String

    var body: some View {
        let indices = Array(range ?? values.indices)
        let rows = stride(from: 0, to: indices.count, by: columns).map { Array(indices[$0..<min($0 + columns, indices.count)]) }
        VStack(spacing: 4) {
            ForEach(rows, id: \.first) { row in
                HStack(spacing: 4) {
                    ForEach(row, id: \.self) { index in
                        StudioNumberCell(value: $values[index], label: "\(label) \(index + 1)")
                    }
                }
            }
        }
    }
}

private struct StudioNumberCell: View {
    @Binding var value: Double
    let label: String

    var body: some View {
        TextField("", value: $value, format: .number.precision(.fractionLength(0...6)).grouping(.never))
            .textFieldStyle(.plain)
            .font(.system(size: 11, design: .monospaced))
            .multilineTextAlignment(.trailing)
            .padding(.horizontal, 5)
            .padding(.vertical, 4)
            .frame(width: 58)
            .merePanel()
            .accessibilityLabel(label)
    }
}

private struct StudioIntegerCell: View {
    @Binding var value: Int
    let label: String

    var body: some View {
        TextField("", value: $value, format: .number.grouping(.never))
            .textFieldStyle(.plain)
            .font(.system(size: 11, design: .monospaced))
            .multilineTextAlignment(.trailing)
            .padding(.horizontal, 5)
            .padding(.vertical, 4)
            .frame(width: 58)
            .merePanel()
            .accessibilityLabel(label)
    }
}
