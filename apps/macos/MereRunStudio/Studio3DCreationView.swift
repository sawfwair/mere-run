import SwiftUI
import UniformTypeIdentifiers

private enum Studio3DEngine: String, CaseIterable, Identifiable {
    case trellis = "TRELLIS.2"
    case triposr = "TripoSR"
    case instantMesh = "InstantMesh"

    var id: String { rawValue }

    var templateID: CommandTemplateID {
        switch self {
        case .trellis: .imageReconstruct3DTrellis2
        case .triposr: .imageReconstruct3D
        case .instantMesh: .imageReconstruct3DMultiview
        }
    }

    var subtitle: String {
        switch self {
        case .trellis: "PBR O-Voxel asset from one image"
        case .triposr: "Fast colored mesh from one image"
        case .instantMesh: "Ordered 4- or 6-view reconstruction"
        }
    }
}

struct Studio3DCreationView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    @State private var engine: Studio3DEngine = .trellis
    @State private var sourcePath = ""
    @State private var orderedViews: [String] = []
    @State private var outputDirectory = StudioSpecialistFiles
        .timestampedDirectory(component: "3D")
        .path
    @State private var model = ""
    @State private var resolution = 256
    @State private var densityThreshold = 25.0
    @State private var foregroundRatio = 0.85
    @State private var alreadyFramed = false
    @State private var vertexColors = true
    @State private var seed = "42"
    @State private var textureSeed = "42"
    @State private var maxTokens = 2_097_152
    @State private var remesh = true
    @State private var remeshBand = 1.0
    @State private var sealRadius = 12
    @State private var camerasPath = ""
    @State private var preflight = false
    @State private var requestID: UUID?
    @State private var errorMessage: String?

    private var currentItem: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    var body: some View {
        HStack(spacing: 0) {
            configuration
                .frame(width: 430)
            Divider().overlay(MereRunTheme.border.opacity(0.55))
            VStack(alignment: .leading, spacing: 12) {
                resultHeader
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: [.model3D, .image, .text]
                )
            }
            .padding(18)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onChange(of: engine) { _, _ in
            model = ""
            outputDirectory = StudioSpecialistFiles.timestampedDirectory(component: "3D").path
            errorMessage = nil
        }
    }

    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Picker("Engine", selection: $engine) {
                    ForEach(Studio3DEngine.allCases) { engine in
                        Text(engine.rawValue).tag(engine)
                    }
                }
                .pickerStyle(.segmented)

                Text(engine.subtitle)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)

                if engine == .instantMesh {
                    orderedViewEditor
                } else {
                    StudioPathField(
                        label: "Source image",
                        placeholder: "/path/to/object.png",
                        path: $sourcePath,
                        allowedContentTypes: [.image]
                    )
                }

                StudioPathField(
                    label: "Output directory",
                    placeholder: "/path/to/output",
                    path: $outputDirectory,
                    picksDirectory: true
                )

                DisclosureGroup("Model & engine controls") {
                    VStack(alignment: .leading, spacing: 12) {
                        labeledField("Model override", text: $model, placeholder: "Managed default")
                        Toggle("Input is already framed", isOn: $alreadyFramed)

                        switch engine {
                        case .triposr:
                            Stepper("Grid resolution: \(resolution)", value: $resolution, in: 2...512)
                            valueSlider("Density threshold", value: $densityThreshold, range: 0...100)
                            valueSlider("Foreground ratio", value: $foregroundRatio, range: 0.1...1)
                            Toggle("Export vertex colors", isOn: $vertexColors)
                        case .trellis:
                            labeledField("Shape seed", text: $seed, placeholder: "42")
                            labeledField("Texture seed", text: $textureSeed, placeholder: "42")
                            Stepper("Sparse token ceiling: \(maxTokens)", value: $maxTokens, in: 1...4_194_304, step: 65_536)
                            Toggle("Remesh surface", isOn: $remesh)
                            if remesh {
                                valueSlider("Remesh band", value: $remeshBand, range: 0.1...4)
                                Stepper("Seal radius: \(sealRadius)", value: $sealRadius, in: 0...64)
                            }
                        case .instantMesh:
                            Stepper("Grid resolution: \(resolution)", value: $resolution, in: 2...256)
                            Toggle("Export vertex colors", isOn: $vertexColors)
                            StudioPathField(
                                label: "Camera JSON (optional)",
                                placeholder: "/path/to/cameras.json",
                                path: $camerasPath,
                                allowedContentTypes: [.json]
                            )
                        }
                    }
                    .padding(.top, 10)
                }
                .font(MereRunTheme.bodyFont)

                Toggle("Preflight only", isOn: $preflight)

                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }

                Button {
                    run()
                } label: {
                    Label(preflight ? "Run preflight" : "Create 3D asset", systemImage: "cube.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .controlSize(.large)
            }
            .padding(18)
        }
    }

    private var orderedViewEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Ordered views")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Text("\(orderedViews.count) / 4 or 6")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(
                        orderedViews.count == 4 || orderedViews.count == 6
                            ? MereRunTheme.green
                            : MereRunTheme.textMuted
                    )
            }
            ForEach(Array(orderedViews.enumerated()), id: \.offset) { index, path in
                HStack(spacing: 7) {
                    Text("\(index + 1)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .frame(width: 18)
                    Text(path)
                        .font(MereRunTheme.captionFont)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    if index > 0 {
                        Button { orderedViews.swapAt(index, index - 1) } label: {
                            Image(systemName: "arrow.up")
                        }
                        .buttonStyle(.plain)
                    }
                    if index + 1 < orderedViews.count {
                        Button { orderedViews.swapAt(index, index + 1) } label: {
                            Image(systemName: "arrow.down")
                        }
                        .buttonStyle(.plain)
                    }
                    Button(role: .destructive) { orderedViews.remove(at: index) } label: {
                        Image(systemName: "xmark")
                    }
                    .buttonStyle(.plain)
                }
                .padding(8)
                .background(MereRunTheme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            Button {
                let urls = StudioSpecialistFiles.chooseFile(
                    title: "Add ordered source views",
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: true
                )
                orderedViews.append(contentsOf: urls.map(\.path))
                orderedViews = Array(orderedViews.prefix(6))
            } label: {
                Label("Add views…", systemImage: "photo.stack")
            }
            .buttonStyle(.bordered)
            .disabled(orderedViews.count >= 6)
        }
    }

    private var resultHeader: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("Interactive result")
                    .font(MereRunTheme.sectionFont)
                Text("Orbit the GLB/OBJ with Quick Look; every export remains in Library.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            if let summary = currentItem.flatMap(StudioMeshSummary.load) {
                Text(summary)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
        }
    }

    private func labeledField(
        _ label: String,
        text: Binding<String>,
        placeholder: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: text)
                .mereField()
        }
    }

    private func valueSlider(
        _ label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(value.wrappedValue.formatted(.number.precision(.fractionLength(2))))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .font(MereRunTheme.captionFont)
            Slider(value: value, in: range)
        }
    }

    private func run() {
        errorMessage = nil
        guard !outputDirectory.isBlank else {
            errorMessage = "Choose an output directory."
            return
        }
        let outputURL = URL(fileURLWithPath: NSString(string: outputDirectory).expandingTildeInPath)
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: outputURL.path),
           !contents.isEmpty {
            errorMessage = "Choose a new or empty output directory so this asset remains immutable in Library."
            return
        }
        if engine == .instantMesh {
            guard orderedViews.count == 4 || orderedViews.count == 6 else {
                errorMessage = "InstantMesh needs exactly four or six ordered views."
                return
            }
        } else if sourcePath.isBlank {
            errorMessage = "Choose a source image."
            return
        }

        guard let template = CommandCatalog.template(id: engine.templateID) else {
            errorMessage = "The \(engine.rawValue) command is unavailable."
            return
        }
        var draft = template.defaultDraft()
        draft.inputPath = sourcePath
        draft.referenceImagePaths = orderedViews.joined(separator: "\n")
        draft.outputPath = outputDirectory
        draft.model = model
        draft.reconstructionResolution = resolution
        draft.densityThreshold = densityThreshold
        draft.foregroundRatio = foregroundRatio
        draft.alreadyFramed = alreadyFramed
        draft.noVertexColors = !vertexColors
        draft.seed = seed
        draft.trellisTextureSeed = textureSeed
        draft.maxTokens = maxTokens
        draft.trellisNoRemesh = !remesh
        draft.trellisRemeshBand = remesh ? remeshBand : nil
        draft.trellisSealRadius = remesh ? sealRadius : nil
        draft.camerasPath = camerasPath
        draft.dryRun = preflight
        draft.json = preflight
        requestID = StudioSpecialistRunner.submit(
            templateID: engine.templateID,
            mode: .createImage,
            draft: draft,
            controller: controller,
            library: library
        )
    }
}

private enum StudioMeshSummary {
    static func load(item: StudioLibraryItem) -> String? {
        let manifests = item.allArtifactURLs.filter {
            $0.pathExtension.lowercased() == "json"
                && $0.lastPathComponent.lowercased().contains("manifest")
        }
        guard let data = manifests.lazy.compactMap({ try? Data(contentsOf: $0) }).first,
              let object = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }
        let values = flatten(object)
        let vertex = firstInt(values, keys: ["vertex_count", "vertices", "num_vertices"])
        let triangle = firstInt(values, keys: ["triangle_count", "triangles", "num_faces", "face_count"])
        let pbr = firstInt(values, keys: ["pbr_voxel_count", "voxel_count", "occupied_voxels"])
        let parts: [String] = [
            vertex.map { "\($0.formatted()) vertices" },
            triangle.map { "\($0.formatted()) triangles" },
            pbr.map { "\($0.formatted()) PBR voxels" },
        ].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private static func flatten(_ value: Any) -> [String: Any] {
        if let dictionary = value as? [String: Any] {
            return dictionary.reduce(into: dictionary) { result, entry in
                flatten(entry.value).forEach { result[$0.key] = $0.value }
            }
        }
        if let array = value as? [Any] {
            return array.reduce(into: [:]) { result, element in
                flatten(element).forEach { result[$0.key] = $0.value }
            }
        }
        return [:]
    }

    private static func firstInt(_ values: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let value = values[key] as? Int { return value }
            if let number = values[key] as? NSNumber { return number.intValue }
        }
        return nil
    }
}
