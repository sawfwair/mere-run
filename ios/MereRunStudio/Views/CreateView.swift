import PhotosUI
import SwiftUI
import MereRunRelayKit
import UniformTypeIdentifiers

/// Prompt-first creation in the Studio language: pick a mode, describe or
/// attach the source, run it on your fleet. Modes, fields, and options come
/// from the shared node catalog; availability comes from what the paired
/// fleet reports, and an out-of-date fleet is said out loud rather than
/// hidden. Photos arrive through the system picker and audio through the
/// file importer; both are hashed into the job bundle and uploaded by digest.
struct CreateView: View {
    enum AssetMedia {
        case image
        case audio
    }

    struct Mode: Identifiable {
        let kind: String
        let symbol: String
        let headline: String
        let promptPlaceholder: String?
        let promptField: String?
        let promptRequired: Bool
        let modelPrefix: String
        let assetField: String?
        let assetRequired: Bool
        let assetMedia: AssetMedia

        init(
            kind: String,
            symbol: String,
            headline: String,
            promptPlaceholder: String? = nil,
            promptField: String? = nil,
            promptRequired: Bool = false,
            modelPrefix: String,
            assetField: String? = nil,
            assetRequired: Bool = false,
            assetMedia: AssetMedia = .image
        ) {
            self.kind = kind
            self.symbol = symbol
            self.headline = headline
            self.promptPlaceholder = promptPlaceholder
            self.promptField = promptField
            self.promptRequired = promptRequired
            self.modelPrefix = modelPrefix
            self.assetField = assetField
            self.assetRequired = assetRequired
            self.assetMedia = assetMedia
        }

        var id: String { kind }
        var entry: WorkflowNodeCatalogEntry? { WorkflowNodeRegistry.entry(for: kind) }
    }

    static let modes: [Mode] = [
        Mode(
            kind: "image.generate",
            symbol: "photo",
            headline: "Make something\nvisible.",
            promptPlaceholder: "Describe the image…",
            promptField: "prompt",
            promptRequired: true,
            modelPrefix: "image-",
            assetField: "input"
        ),
        Mode(
            kind: "video.generate",
            symbol: "film",
            headline: "Set it in motion.",
            promptPlaceholder: "Describe the shot…",
            promptField: "prompt",
            promptRequired: true,
            modelPrefix: "video-",
            assetField: "image"
        ),
        Mode(
            kind: "music.generate",
            symbol: "music.note",
            headline: "Score the moment.",
            promptPlaceholder: "Describe the music…",
            promptField: "prompt",
            promptRequired: true,
            modelPrefix: "music-"
        ),
        Mode(
            kind: "sfx.generate",
            symbol: "waveform",
            headline: "Shape a sound.",
            promptPlaceholder: "Describe the sound effect…",
            promptField: "prompt",
            promptRequired: true,
            modelPrefix: "sfx-"
        ),
        Mode(
            kind: "speech.synthesize",
            symbol: "speaker.wave.2",
            headline: "Give it a voice.",
            promptPlaceholder: "Write what should be spoken…",
            promptField: "text",
            promptRequired: true,
            modelPrefix: "speech-tts-"
        ),
        Mode(
            kind: "vision.image-to-3d",
            symbol: "cube",
            headline: "Give it depth.",
            modelPrefix: "image-3d-",
            assetField: "image",
            assetRequired: true
        ),
        Mode(
            kind: "vision.caption",
            symbol: "text.below.photo",
            headline: "Say what you see.",
            promptPlaceholder: "Optional: guide the caption…",
            promptField: "prompt",
            modelPrefix: "vision-",
            assetField: "image",
            assetRequired: true
        ),
        Mode(
            kind: "vision.ocr",
            symbol: "doc.text.viewfinder",
            headline: "Read every word.",
            modelPrefix: "vision-ocr-",
            assetField: "image",
            assetRequired: true
        ),
        Mode(
            kind: "speech.transcribe",
            symbol: "text.quote",
            headline: "Write it down.",
            modelPrefix: "speech-asr-",
            assetField: "audio",
            assetRequired: true,
            assetMedia: .audio
        ),
        Mode(
            kind: "speech.diarize",
            symbol: "person.2.wave.2",
            headline: "Who said what.",
            modelPrefix: "speech-diarization-",
            assetField: "audio",
            assetRequired: true,
            assetMedia: .audio
        ),
        Mode(
            kind: "music.separate",
            symbol: "square.stack.3d.down.right",
            headline: "Pull it apart.",
            modelPrefix: "music-separate-",
            assetField: "audio",
            assetRequired: true,
            assetMedia: .audio
        ),
        Mode(
            kind: "audio.enhance",
            symbol: "wand.and.stars",
            headline: "Make it shine.",
            modelPrefix: "audio-",
            assetField: "audio",
            assetRequired: true,
            assetMedia: .audio
        ),
    ]

    @EnvironmentObject private var relay: RelayStore
    @StateObject private var local = LocalEngine()
    @State private var runLocally = false
    @State private var selected = Self.modes[0]
    @State private var prompt = ""
    @State private var model = ""
    @State private var textFields: [String: String] = [:]
    @State private var numberFields: [String: String] = [:]
    @State private var enumFields: [String: String] = [:]
    @State private var showOptions = false
    @State private var submitting = false
    @State private var errorMessage: String?
    @State private var submittedJobID: String?

    @State private var photoItem: PhotosPickerItem?
    @State private var showAudioImporter = false
    @State private var assetURL: URL?
    @State private var assetLabel: String?
    @State private var assetPreview: UIImage?

    private var fleetKinds: Set<String>? {
        relay.workerProbe.map { Set($0.nodeKinds) }
    }

    private func isAvailable(_ mode: Mode) -> Bool {
        fleetKinds?.contains(mode.kind) ?? true
    }

    private var installedModels: [String] {
        (relay.workerProbe?.installedModelIDs ?? []).filter { $0.hasPrefix(selected.modelPrefix) }
    }

    private var optionalStringInputs: [WorkflowNodeField] {
        selected.entry?.inputs.filter {
            $0.type == .string && !$0.required && $0.name != "model" && $0.name != selected.promptField
        } ?? []
    }

    private var numericInputs: [WorkflowNodeField] {
        selected.entry?.inputs.filter { ($0.type == .integer || $0.type == .number) && !$0.required } ?? []
    }

    private var enumInputs: [WorkflowNodeField] {
        selected.entry?.inputs.filter { $0.type == .enumeration && !$0.required } ?? []
    }

    private var canRun: Bool {
        let promptSatisfied = !selected.promptRequired || !prompt.isEmpty
        let assetSatisfied = !selected.assetRequired || assetURL != nil
        return promptSatisfied && assetSatisfied && !submitting && isAvailable(selected)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: MereTheme.Spacing.xl) {
                    Text(selected.headline)
                        .font(.system(.largeTitle, design: .serif))
                        .foregroundStyle(MereTheme.textPrimary)
                        .padding(.top, MereTheme.Spacing.l)
                        .animation(nil, value: selected.kind)

                    modeRail

                    if selected.kind == "image.generate" {
                        destinationToggle
                    }

                    composer

                    if runLocally, selected.kind == "image.generate" {
                        localLane
                    }

                    if !isAvailable(selected) {
                        MereBannerView(
                            text: "Your fleet's nodes don't offer \(selected.entry?.title.lowercased() ?? selected.kind) yet. Update mere.run on your nodes to add it.",
                            color: MereTheme.caution
                        )
                    }

                    if let errorMessage {
                        MereBannerView(text: errorMessage, color: MereTheme.failure)
                    }

                    runButton

                    Text("Runs on your machines. Prompts, photos, and outputs stay between your devices.")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textMuted)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                .padding(.horizontal, MereTheme.Spacing.xl)
                .padding(.bottom, MereTheme.Spacing.xxl)
            }
            .background(MereTheme.background.ignoresSafeArea())
            .scrollDismissesKeyboard(.interactively)
            .navigationDestination(item: $submittedJobID) { jobID in
                RunDetailView(jobID: jobID)
            }
            .sheet(isPresented: $showOptions) { optionsSheet }
            .fileImporter(
                isPresented: $showAudioImporter,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    importAudio(url)
                }
            }
            .onChange(of: photoItem) {
                Task { await importPhoto() }
            }
            .task {
                await relay.refreshWorkerProbe()
            }
        }
    }

    private var modeRail: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: MereTheme.Spacing.s) {
                ForEach(Self.modes) { mode in
                    let isSelected = mode.kind == selected.kind
                    Button {
                        select(mode)
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: mode.symbol)
                            Text(mode.entry?.title ?? mode.kind)
                        }
                        .font(.subheadline.weight(isSelected ? .semibold : .regular))
                        .padding(.horizontal, MereTheme.Spacing.m)
                        .padding(.vertical, MereTheme.Spacing.s)
                        .background(
                            Capsule().fill(isSelected ? MereTheme.surfaceRaised : MereTheme.surface)
                        )
                        .overlay(
                            Capsule().stroke(
                                isSelected ? MereTheme.accent : MereTheme.border.opacity(0.6),
                                lineWidth: 1
                            )
                        )
                        .foregroundStyle(isAvailable(mode) ? MereTheme.textPrimary : MereTheme.textMuted)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 2)
        }
        .defaultScrollAnchor(.leading)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: MereTheme.Spacing.m) {
            if selected.assetField != nil {
                attachRow
            }
            if let placeholder = selected.promptPlaceholder {
                TextField(
                    "",
                    text: $prompt,
                    prompt: Text(placeholder).foregroundColor(MereTheme.textMuted),
                    axis: .vertical
                )
                .font(.body)
                .foregroundStyle(MereTheme.textPrimary)
                .lineLimit(3...9)
            }

            Divider().overlay(MereTheme.border.opacity(0.5))

            HStack {
                Menu {
                    Button("Fleet default") { model = "" }
                    ForEach(installedModels, id: \.self) { id in
                        Button(id) { model = id }
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "cpu")
                        Text(model.isEmpty ? "Fleet default" : model)
                            .lineLimit(1)
                    }
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textSecondary)
                }
                Spacer()
                Button {
                    showOptions = true
                } label: {
                    Label("Options", systemImage: "slider.horizontal.3")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textSecondary)
                }
            }
        }
        .padding(MereTheme.Spacing.l)
        .merePanel()
    }

    @ViewBuilder
    private var attachRow: some View {
        if let assetURL {
            HStack(spacing: MereTheme.Spacing.s) {
                if let assetPreview {
                    Image(uiImage: assetPreview)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 52, height: 52)
                        .clipShape(RoundedRectangle(cornerRadius: MereTheme.Radius.field))
                } else {
                    Image(systemName: "waveform")
                        .frame(width: 52, height: 52)
                        .background(
                            RoundedRectangle(cornerRadius: MereTheme.Radius.field)
                                .fill(MereTheme.surfaceRaised)
                        )
                        .foregroundStyle(MereTheme.accent)
                }
                Text(assetLabel ?? assetURL.lastPathComponent)
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textSecondary)
                    .lineLimit(1)
                Spacer()
                Button {
                    clearAsset()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(MereTheme.textMuted)
                }
                .accessibilityLabel("Remove attachment")
            }
        } else {
            switch selected.assetMedia {
            case .image:
                PhotosPicker(selection: $photoItem, matching: .images) {
                    AttachLabel(
                        title: selected.assetRequired ? "Choose a photo" : "Add a photo (optional)",
                        symbol: "photo.badge.plus"
                    )
                }
            case .audio:
                Button {
                    showAudioImporter = true
                } label: {
                    AttachLabel(title: "Choose an audio file", symbol: "waveform.badge.plus")
                }
            }
        }
    }


    private var destinationToggle: some View {
        Picker("Runs on", selection: $runLocally) {
            Text("Your fleet").tag(false)
            Text("This iPhone").tag(true)
        }
        .pickerStyle(.segmented)
    }

    @ViewBuilder
    private var localLane: some View {
        switch local.state {
        case .checking:
            EmptyView()
        case .notInstalled:
            MereBannerView(
                text: "Klein nano (about 2 GB) runs entirely on this iPhone. Download once over Wi-Fi; it stays in this app's storage.",
                color: MereTheme.accent
            )
            Button {
                Task { await local.download() }
            } label: {
                Text("Download Klein nano").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
        case .downloading(let label):
            HStack(spacing: MereTheme.Spacing.s) {
                ProgressView()
                Text("Downloading — \(label)")
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textSecondary)
            }
        case .failed(let message):
            MereBannerView(text: message, color: MereTheme.failure)
        case .ready, .generating:
            if let image = local.lastImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: MereTheme.Radius.panel))
            }
            if local.state == .generating {
                HStack(spacing: MereTheme.Spacing.s) {
                    ProgressView()
                    Text("Generating on this iPhone…")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textSecondary)
                }
            }
        }
    }

    private var runButton: some View {
        Button {
            if runLocally, selected.kind == "image.generate" {
                Task { await local.generate(prompt: prompt) }
            } else {
                Task { await submit() }
            }
        } label: {
            Group {
                if submitting || local.state == .generating {
                    ProgressView().tint(.white)
                } else {
                    Text(runLocally && selected.kind == "image.generate" ? "Run on this iPhone" : "Run on your fleet")
                        .font(.body.weight(.semibold))
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(runLocally && selected.kind == "image.generate"
            ? (prompt.isEmpty || local.state != .ready)
            : !canRun)
    }

    private var optionsSheet: some View {
        NavigationStack {
            List {
                if !optionalStringInputs.isEmpty {
                    Section {
                        ForEach(optionalStringInputs, id: \.name) { field in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(fieldTitle(field.name))
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textMuted)
                                TextField(
                                    "",
                                    text: binding(for: field.name, in: $textFields),
                                    axis: field.multiline == true ? .vertical : .horizontal
                                )
                                .lineLimit(field.multiline == true ? 2...6 : 1...1)
                            }
                        }
                    }
                }
                if !numericInputs.isEmpty {
                    Section {
                        ForEach(numericInputs, id: \.name) { field in
                            LabeledContent(fieldTitle(field.name)) {
                                TextField("auto", text: binding(for: field.name, in: $numberFields))
                                    .keyboardType(field.type == .integer ? .numberPad : .decimalPad)
                                    .multilineTextAlignment(.trailing)
                                    .frame(maxWidth: 120)
                            }
                        }
                    }
                }
                if !enumInputs.isEmpty {
                    Section {
                        ForEach(enumInputs, id: \.name) { field in
                            Picker(fieldTitle(field.name), selection: binding(for: field.name, in: $enumFields)) {
                                Text("auto").tag("")
                                ForEach(field.values ?? [], id: \.self) { value in
                                    Text(value).tag(value)
                                }
                            }
                        }
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(MereTheme.background)
            .navigationTitle("Options")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showOptions = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func select(_ mode: Mode) {
        selected = mode
        model = ""
        textFields = [:]
        numberFields = [:]
        enumFields = [:]
        errorMessage = nil
        clearAsset()
    }

    private func clearAsset() {
        if let assetURL {
            try? FileManager.default.removeItem(at: assetURL)
        }
        assetURL = nil
        assetLabel = nil
        assetPreview = nil
        photoItem = nil
    }

    private func importPhoto() async {
        guard let photoItem else { return }
        guard let data = try? await photoItem.loadTransferable(type: Data.self),
              let image = UIImage(data: data),
              let jpeg = image.jpegData(compressionQuality: 0.92) else {
            errorMessage = "Could not load that photo."
            return
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("attachments", isDirectory: true)
            .appendingPathComponent("photo-\(UUID().uuidString).jpg")
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try jpeg.write(to: url)
            assetURL = url
            assetLabel = "Photo"
            assetPreview = image
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func importAudio(_ source: URL) {
        let scoped = source.startAccessingSecurityScopedResource()
        defer {
            if scoped { source.stopAccessingSecurityScopedResource() }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("attachments", isDirectory: true)
            .appendingPathComponent("audio-\(UUID().uuidString).\(source.pathExtension.isEmpty ? "wav" : source.pathExtension)")
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try FileManager.default.copyItem(at: source, to: url)
            assetURL = url
            assetLabel = source.lastPathComponent
            assetPreview = nil
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func fieldTitle(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ")
    }

    private func binding(for name: String, in store: Binding<[String: String]>) -> Binding<String> {
        Binding(
            get: { store.wrappedValue[name] ?? "" },
            set: { store.wrappedValue[name] = $0 }
        )
    }

    private func submit() async {
        submitting = true
        defer { submitting = false }
        var arguments: [String: WorkflowValue] = [:]
        if let promptField = selected.promptField, !prompt.isEmpty {
            arguments[promptField] = .string(prompt)
        }
        if let assetField = selected.assetField, let assetURL {
            arguments[assetField] = .string(assetURL.path)
        }
        if !model.isEmpty {
            arguments["model"] = .string(model)
        }
        for (name, raw) in textFields where !raw.isEmpty {
            arguments[name] = .string(raw)
        }
        for (name, raw) in enumFields where !raw.isEmpty {
            arguments[name] = .string(raw)
        }
        for field in numericInputs {
            let raw = (numberFields[field.name] ?? "").trimmingCharacters(in: .whitespaces)
            guard !raw.isEmpty else { continue }
            if field.type == .integer, let value = Int64(raw) {
                arguments[field.name] = .integer(value)
            } else if field.type == .number, let value = Double(raw) {
                arguments[field.name] = .number(value)
            }
        }
        do {
            let job = try await relay.submit(kind: selected.kind, arguments: arguments)
            errorMessage = nil
            submittedJobID = job.jobID
        } catch let error as RelayClientError {
            errorMessage = AppErrorText.presentable(error.message)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

private struct AttachLabel: View {
    let title: String
    let symbol: String

    var body: some View {
        HStack(spacing: MereTheme.Spacing.s) {
            Image(systemName: symbol)
                .foregroundStyle(MereTheme.accent)
            Text(title)
                .font(.subheadline)
                .foregroundStyle(MereTheme.textSecondary)
            Spacer()
        }
        .padding(.vertical, MereTheme.Spacing.s)
    }
}

/// The single inline notice component, in the Studio's one voice.
struct MereBannerView: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: MereTheme.Spacing.s) {
            Circle().fill(color).frame(width: 8, height: 8).padding(.top, 5)
            Text(text)
                .font(.footnote)
                .foregroundStyle(MereTheme.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(MereTheme.Spacing.m)
        .merePanel()
    }
}
