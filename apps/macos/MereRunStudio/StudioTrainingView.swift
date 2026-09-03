import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum StudioTrainingKind: String, CaseIterable, Identifiable {
    case image
    case text
    case music

    var id: String { rawValue }

    var title: String {
        switch self {
        case .image: "Image LoRA"
        case .text: "Text LoRA"
        case .music: "Music LoRA / LoKr"
        }
    }

    var symbol: String {
        switch self {
        case .image: "photo.stack"
        case .text: "text.word.spacing"
        case .music: "music.note.list"
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .image: .imageTrainLoRA
        case .text: .textTrainLoRA
        case .music: .musicTrainAdapter
        }
    }

    var mode: StudioMode {
        switch self {
        case .image: .createImage
        case .text: .chat
        case .music: .music
        }
    }
}

struct StudioTrainingDatasetPreview: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
    let imageURL: URL?
    let audioURL: URL?
}

struct StudioTrainingDatasetSnapshot: Equatable {
    let source: URL
    let totalRecords: Int
    let usableRecords: Int
    let previews: [StudioTrainingDatasetPreview]
    let diagnostics: [String]

    static func inspect(kind: StudioTrainingKind, path: String) -> StudioTrainingDatasetSnapshot {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        switch kind {
        case .image:
            return inspectImages(at: url)
        case .text:
            return inspectText(at: url)
        case .music:
            return inspectMusic(at: url)
        }
    }

    private static func inspectImages(at root: URL) -> StudioTrainingDatasetSnapshot {
        let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "heic"]
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return .init(
                source: root,
                totalRecords: 0,
                usableRecords: 0,
                previews: [],
                diagnostics: ["The dataset directory could not be read."]
            )
        }
        let images = urls.filter { imageExtensions.contains($0.pathExtension.lowercased()) }
        var previews: [StudioTrainingDatasetPreview] = []
        var missingCaptions = 0
        for image in images {
            let captionURL = image.deletingPathExtension().appendingPathExtension("txt")
            let caption = (try? String(contentsOf: captionURL, encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if caption == nil || caption?.isEmpty == true {
                missingCaptions += 1
            }
            if previews.count < 12 {
                previews.append(
                    .init(
                        id: image.path,
                        title: image.lastPathComponent,
                        detail: caption ?? "Missing matching .txt caption",
                        imageURL: image,
                        audioURL: nil
                    )
                )
            }
        }
        let diagnostics = missingCaptions == 0
            ? ["All discovered images have matching caption files."]
            : ["\(missingCaptions) image(s) are missing matching .txt captions."]
        return .init(
            source: root,
            totalRecords: images.count,
            usableRecords: images.count - missingCaptions,
            previews: previews,
            diagnostics: diagnostics
        )
    }

    private static func inspectText(at url: URL) -> StudioTrainingDatasetSnapshot {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return .init(
                source: url,
                totalRecords: 0,
                usableRecords: 0,
                previews: [],
                diagnostics: ["The JSONL dataset could not be read."]
            )
        }
        let lines = text.split(whereSeparator: \.isNewline)
        var valid = 0
        var previews: [StudioTrainingDatasetPreview] = []
        for (index, line) in lines.enumerated() {
            let row = String(line)
            let isValid = row.data(using: .utf8).flatMap {
                try? JSONSerialization.jsonObject(with: $0)
            } != nil
            if isValid { valid += 1 }
            if previews.count < 8 {
                previews.append(
                    .init(
                        id: "\(url.path)#\(index)",
                        title: "Example \(index + 1)",
                        detail: row.count > 280 ? String(row.prefix(280)) + "…" : row,
                        imageURL: nil,
                        audioURL: nil
                    )
                )
            }
        }
        return .init(
            source: url,
            totalRecords: lines.count,
            usableRecords: valid,
            previews: previews,
            diagnostics: valid == lines.count
                ? ["Every JSONL row parses as JSON."]
                : ["\(lines.count - valid) malformed JSONL row(s) need attention."]
        )
    }

    private static func inspectMusic(at url: URL) -> StudioTrainingDatasetSnapshot {
        guard let data = try? Data(contentsOf: url) else {
            return .init(
                source: url,
                totalRecords: 0,
                usableRecords: 0,
                previews: [],
                diagnostics: ["The music dataset manifest could not be read."]
            )
        }
        let records: [[String: Any]]
        if url.pathExtension.lowercased() == "jsonl",
           let text = String(data: data, encoding: .utf8) {
            records = text.split(whereSeparator: \.isNewline).compactMap { line in
                guard let row = String(line).data(using: .utf8) else { return nil }
                return (try? JSONSerialization.jsonObject(with: row)) as? [String: Any]
            }
        } else {
            let object = try? JSONSerialization.jsonObject(with: data)
            records = object as? [[String: Any]]
                ?? (object as? [String: Any]).flatMap { $0["records"] as? [[String: Any]] }
                ?? []
        }
        let root = url.deletingLastPathComponent()
        var usable = 0
        let previews = records.prefix(10).enumerated().map { index, record in
            let rawAudio = (record["audio"] as? String) ?? ""
            let audioURL = rawAudio.isEmpty
                ? nil
                : (rawAudio.hasPrefix("/")
                    ? URL(fileURLWithPath: rawAudio)
                    : root.appendingPathComponent(rawAudio))
            if let audioURL, FileManager.default.fileExists(atPath: audioURL.path),
               (record["caption"] as? String)?.isEmpty == false {
                usable += 1
            }
            let caption = (record["caption"] as? String) ?? "Missing caption"
            let lyrics = (record["lyrics"] as? String).flatMap {
                $0.isEmpty ? nil : String($0.prefix(100))
            }
            return StudioTrainingDatasetPreview(
                id: "\(url.path)#\(index)",
                title: audioURL?.lastPathComponent ?? "Missing audio",
                detail: lyrics.map { "\(caption)\nLyrics: \($0)" } ?? caption,
                imageURL: nil,
                audioURL: audioURL
            )
        }
        // Count all usable records, not only the preview window.
        usable = records.reduce(into: 0) { count, record in
            let rawAudio = (record["audio"] as? String) ?? ""
            let resolved = rawAudio.hasPrefix("/")
                ? URL(fileURLWithPath: rawAudio)
                : root.appendingPathComponent(rawAudio)
            if !rawAudio.isEmpty,
               FileManager.default.fileExists(atPath: resolved.path),
               (record["caption"] as? String)?.isEmpty == false {
                count += 1
            }
        }
        return .init(
            source: url,
            totalRecords: records.count,
            usableRecords: usable,
            previews: Array(previews),
            diagnostics: usable == records.count && !records.isEmpty
                ? ["Every manifest row resolves to audio and a caption."]
                : ["\(records.count - usable) row(s) are missing audio or a caption."]
        )
    }
}

struct StudioTrainingEvent: Decodable, Identifiable, Equatable {
    let sequence: Int
    let type: String
    let stage: String?
    let message: String?
    let step: Int?
    let totalSteps: Int?
    let loss: Double?
    let fraction: Double?
    let path: String?

    var id: Int { sequence }

    enum CodingKeys: String, CodingKey {
        case sequence
        case type
        case stage
        case message
        case step
        case totalSteps = "total_steps"
        case loss
        case fraction
        case path
    }
}

struct StudioTrainingSnapshot: Equatable {
    let outputURL: URL
    let events: [StudioTrainingEvent]
    let samples: [URL]
    let checkpoints: [URL]
    let artifacts: [URL]

    var lossPoints: [(step: Int, loss: Double)] {
        events.compactMap {
            guard let step = $0.step, let loss = $0.loss else { return nil }
            return (step, loss)
        }
    }

    var latest: StudioTrainingEvent? { events.last }
    var progress: Double? {
        if let fraction = latest?.fraction { return fraction }
        guard let step = latest?.step, let total = latest?.totalSteps, total > 0 else { return nil }
        return Double(step) / Double(total)
    }

    static func load(outputPath: String) -> StudioTrainingSnapshot? {
        guard !outputPath.isBlank else { return nil }
        let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        let directory = outputURL.deletingLastPathComponent()
        let base = outputURL.deletingPathExtension().lastPathComponent
        let eventURL = directory.appendingPathComponent("\(base).events.jsonl")
        let events = loadEvents(from: eventURL)
        let rootFiles = files(in: directory).filter {
            $0.lastPathComponent.hasPrefix(base)
                || $0.lastPathComponent == "run.json"
        }
        let samples = files(in: directory.appendingPathComponent("samples", isDirectory: true))
            .filter { StudioOutputFileKind.classify($0) == .image }
        let checkpoints = files(in: directory.appendingPathComponent("checkpoints", isDirectory: true))
            .filter { ["safetensors", "zip"].contains($0.pathExtension.lowercased()) }
        return StudioTrainingSnapshot(
            outputURL: outputURL,
            events: events,
            samples: samples.sorted { $0.lastPathComponent < $1.lastPathComponent },
            checkpoints: checkpoints.sorted { $0.lastPathComponent < $1.lastPathComponent },
            artifacts: rootFiles.sorted { $0.lastPathComponent < $1.lastPathComponent }
        )
    }

    private static func loadEvents(from url: URL) -> [StudioTrainingEvent] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            guard let data = String(line).data(using: .utf8) else { return nil }
            return try? decoder.decode(StudioTrainingEvent.self, from: data)
        }
        .sorted { $0.sequence < $1.sequence }
    }

    private static func files(in directory: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ))?.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        } ?? []
    }
}

struct StudioTrainingView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Fixed per host: Image ▸ Train, Chat ▸ Train, and Music ▸ Train each show one trainer.
    @State private var kind: StudioTrainingKind
    @State private var imageDraft: CommandDraft
    @State private var textDraft: CommandDraft
    @State private var musicDraft: CommandDraft
    @State private var datasetSnapshot: StudioTrainingDatasetSnapshot?
    @State private var currentSnapshot: StudioTrainingSnapshot?
    @State private var requestID: UUID?
    @State private var statusMessage: String?
    @State private var compareA: UUID?
    @State private var compareB: UUID?
    @State private var selectedDatasetPreview: String?

    private let refreshTimer = Timer.publish(every: 1.5, on: .main, in: .common).autoconnect()

    init(kind: StudioTrainingKind) {
        _kind = State(initialValue: kind)

        var image = CommandCatalog.template(id: .imageTrainLoRA)?.defaultDraft() ?? CommandDraft()
        image.outputPath = Self.timestampedOutput(prefix: "image-adapter")
        if image.seed.isBlank { image.seed = "42" }
        image.checkpointInterval = 250
        image.sampleInterval = 250
        _imageDraft = State(initialValue: image)

        var text = CommandCatalog.template(id: .textTrainLoRA)?.defaultDraft() ?? CommandDraft()
        text.outputPath = Self.timestampedOutput(prefix: "text-adapter")
        if text.seed.isBlank { text.seed = "42" }
        _textDraft = State(initialValue: text)

        var music = CommandCatalog.template(id: .musicTrainAdapter)?.defaultDraft() ?? CommandDraft()
        music.outputPath = Self.timestampedOutput(prefix: "music-adapter")
        if music.seed.isBlank { music.seed = "42" }
        _musicDraft = State(initialValue: music)
    }

    private var activeDraft: CommandDraft {
        switch kind {
        case .image: imageDraft
        case .text: textDraft
        case .music: musicDraft
        }
    }

    private var trainingRuns: [StudioLibraryItem] {
        let ids: Set<CommandTemplateID> = [.imageTrainLoRA, .textTrainLoRA, .musicTrainAdapter]
        return library.items.filter { item in
            item.templateID.map(ids.contains) == true
        }
    }

    var body: some View {
        HStack(spacing: 0) {
            leftColumn
                .frame(width: 465)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            dashboard
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onReceive(refreshTimer) { _ in refreshSnapshot() }
        .onReceive(controller.runCompletions) { result in
            guard [.imageTrainLoRA, .textTrainLoRA, .musicTrainAdapter].contains(result.templateID) else {
                return
            }
            refreshSnapshot()
            seedComparisons()
        }
        .onChange(of: kind) { _, _ in
            datasetSnapshot = nil
            currentSnapshot = nil
            statusMessage = nil
        }
        .task { seedComparisons() }
    }

    private var leftColumn: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                datasetSection
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                switch kind {
                case .image:
                    imageControls
                case .text:
                    textControls
                case .music:
                    musicControls
                }
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                actionSection
                if let statusMessage {
                    Text(statusMessage)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .padding(18)
        }
    }

    private var datasetSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Dataset")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Button("Inspect") { inspectDataset() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            StudioPathField(
                label: kind == .image ? "Image-caption directory" : "JSON / JSONL manifest",
                placeholder: kind == .image ? "Folder with image + .txt pairs" : "Training manifest",
                path: activeInputBinding,
                picksDirectory: kind == .image,
                allowedContentTypes: kind == .image ? [] : [.json, .plainText]
            )
            if let snapshot = datasetSnapshot {
                HStack(spacing: 8) {
                    datasetMetric("Records", snapshot.totalRecords)
                    datasetMetric("Usable", snapshot.usableRecords)
                    datasetMetric("Issues", snapshot.totalRecords - snapshot.usableRecords)
                }
                ForEach(snapshot.diagnostics, id: \.self) { diagnostic in
                    Label(
                        diagnostic,
                        systemImage: snapshot.usableRecords == snapshot.totalRecords
                            ? "checkmark.circle.fill"
                            : "exclamationmark.triangle.fill"
                    )
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(
                        snapshot.usableRecords == snapshot.totalRecords
                            ? MereRunTheme.green
                            : MereRunTheme.yellow
                    )
                }
                if !snapshot.previews.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(snapshot.previews) { preview in
                                datasetPreviewCard(preview)
                            }
                        }
                    }
                }
            }
        }
    }

    private var activeInputBinding: Binding<String> {
        switch kind {
        case .image: $imageDraft.inputPath
        case .text: $textDraft.inputPath
        case .music: $musicDraft.inputPath
        }
    }

    private var imageControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            Text("Image training")
                .font(MereRunTheme.sectionFont)
            labeledTextField("Base model", placeholder: "image-krea2-raw or Klein base", text: $imageDraft.model)
            Picker("Recipe", selection: $imageDraft.trainingRecipe) {
                Text("Custom").tag("")
                Text("Krea fast style").tag("krea-fast-style")
                Text("Krea cinematic style").tag("krea-cinematic-style")
                Text("Klein fast style").tag("klein-fast-style")
            }
            if !imageDraft.trainingRecipe.isBlank {
                Toggle("Override recipe values", isOn: $imageDraft.overrideTrainingRecipe)
            }
            HStack {
                Stepper("Width \(imageDraft.width)", value: $imageDraft.width, in: 256...2_048, step: 16)
                Stepper("Height \(imageDraft.height)", value: $imageDraft.height, in: 256...2_048, step: 16)
            }
            coreOptimizerControls(draft: $imageDraft)

            DisclosureGroup("Memory and schedule") {
                VStack(alignment: .leading, spacing: 10) {
                    Toggle("Progressive resolution", isOn: $imageDraft.progressive)
                    Toggle("Low RAM latent cache", isOn: $imageDraft.lowRAM)
                    Toggle("Gradient checkpointing", isOn: $imageDraft.gradientCheckpointing)
                    Toggle("Disable compiled train step", isOn: $imageDraft.disableCompile)
                    Toggle("Lite attention targets", isOn: $imageDraft.trainingLite)
                    Picker("Frozen-base quantization", selection: $imageDraft.baseQuantizationBits) {
                        Text("None").tag("")
                        Text("4-bit").tag("4")
                        Text("8-bit").tag("8")
                    }
                    Stepper(
                        "Scheduler steps \(imageDraft.schedulerSteps)",
                        value: $imageDraft.schedulerSteps,
                        in: 100...10_000,
                        step: 100
                    )
                    Stepper(
                        "Warmup steps \(imageDraft.lrWarmupSteps)",
                        value: $imageDraft.lrWarmupSteps,
                        in: 0...10_000
                    )
                    Toggle("Disable cosine scheduler", isOn: $imageDraft.disableCosineScheduler)
                    numberField("LR minimum factor", value: $imageDraft.lrMinFactor)
                    numberField("Adam weight decay", value: $imageDraft.adamWeightDecay)
                    numberField("Caption dropout", value: $imageDraft.captionDropout)
                    Stepper(
                        "Maximum source resolution \(imageDraft.maxResolution)",
                        value: $imageDraft.maxResolution,
                        in: 0...4_096,
                        step: 64
                    )
                }
                .padding(.top, 9)
            }

            DisclosureGroup("Checkpoints, samples, and resume") {
                VStack(alignment: .leading, spacing: 10) {
                    Stepper(
                        "Checkpoint every \(imageDraft.checkpointInterval) steps",
                        value: $imageDraft.checkpointInterval,
                        in: 0...10_000,
                        step: 50
                    )
                    StudioPathField(
                        label: "Resume from Klein checkpoint",
                        placeholder: "Optional .safetensors or .zip",
                        path: Binding(
                            get: { imageDraft.trainingResumePath ?? "" },
                            set: { imageDraft.trainingResumePath = $0.isBlank ? nil : $0 }
                        ),
                        allowedContentTypes: [.data, .archive]
                    )
                    Stepper(
                        "Preview every \(imageDraft.sampleInterval) steps",
                        value: $imageDraft.sampleInterval,
                        in: 0...10_000,
                        step: 50
                    )
                    labeledTextField(
                        "Preview prompt",
                        placeholder: "Defaults to first caption",
                        text: $imageDraft.samplePrompt
                    )
                    labeledTextField(
                        "Preview model",
                        placeholder: "image-klein-9b",
                        text: $imageDraft.sampleModel
                    )
                    HStack {
                        Stepper("Sample steps \(imageDraft.sampleSteps)", value: $imageDraft.sampleSteps, in: 1...100)
                        numberField("Sample CFG", value: $imageDraft.sampleCFG)
                    }
                    numberField("Sample LoRA scale", value: $imageDraft.sampleLoRAScale)
                    labeledTextField("Sample seed", placeholder: "Random", text: $imageDraft.sampleSeed)
                }
                .padding(.top, 9)
            }

            DisclosureGroup("Klein target and timestep controls") {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Target mode", selection: $imageDraft.loraTargetMode) {
                        Text("Default").tag("")
                        Text("Suffix").tag("suffix")
                        Text("Transformer linear walk").tag("transformer-linear-walk")
                    }
                    Picker("Rank preset", selection: $imageDraft.loraRankPreset) {
                        Text("None").tag("")
                        Text("FLUX.2 style 128").tag("flux2-style-128")
                    }
                    Picker("Target preset", selection: $imageDraft.loraTargetPreset) {
                        Text("None").tag("")
                        Text("fal Klein fast").tag("fal-klein-fast")
                    }
                    labeledTextField(
                        "Per-target ranks",
                        placeholder: ".attn.to_q=128,.ff.linear_in=64",
                        text: $imageDraft.loraTargetRanks
                    )
                    Picker("Timestep sampling", selection: $imageDraft.timestepSampling) {
                        ForEach(
                            ["", "uniform", "bellCurve", "contentFocused", "styleFocused", "logitNormal", "shift"],
                            id: \.self
                        ) { value in
                            Text(value.isEmpty ? "Default" : value).tag(value)
                        }
                    }
                    Picker("Timestep weighting", selection: $imageDraft.timestepLossWeighting) {
                        Text("Default").tag("")
                        Text("None").tag("none")
                        Text("Weighted").tag("weighted")
                    }
                    Picker("Loss weighting", selection: $imageDraft.lossWeighting) {
                        Text("Default").tag("")
                        Text("None").tag("none")
                        Text("SNR").tag("snr")
                        Text("Min SNR").tag("minSNR")
                    }
                    HStack {
                        Stepper("Timestep low \(imageDraft.timestepLow)", value: $imageDraft.timestepLow, in: 0...10_000)
                        Stepper("High \(imageDraft.timestepHigh)", value: $imageDraft.timestepHigh, in: 0...10_000)
                    }
                }
                .padding(.top, 9)
            }
        }
    }

    private var textControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            Text("Text training")
                .font(MereRunTheme.sectionFont)
            Picker("Base family", selection: $textDraft.model) {
                Text("Gemma 4").tag("text-chat-gemma4-12b-4bit")
                Text("Laguna XS 2.1").tag("text-chat-laguna-xs-2-1")
                Text("Inkling-Small").tag("text-chat-inkling-small")
                Text("LFM2.5 A1B").tag("text-chat-lfm25-a1b-8bit")
            }
            .pickerStyle(.segmented)
            labeledTextField("Base model", placeholder: "Managed text model", text: $textDraft.model)
            StudioPathField(
                label: "Local model path",
                placeholder: "Optional local model directory",
                path: $textDraft.modelRoot,
                picksDirectory: true
            )
            labeledTextField("Adapter name", placeholder: "local-assistant", text: $textDraft.adapterName)
            coreOptimizerControls(draft: $textDraft)
            Stepper(
                "Maximum sequence length \(textDraft.maxSequenceLength)",
                value: $textDraft.maxSequenceLength,
                in: 128...65_536,
                step: 128
            )
            labeledTextField(
                "Target modules",
                placeholder: "Model-family defaults",
                text: $textDraft.targetModules
            )
            Text("Leave target modules empty for the native family recipe. LFM2.5 v1 is attention-only; Inkling also includes MLP, expert, and unembedding targets.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            if textDraft.model.localizedCaseInsensitiveContains("inkling") {
                numberField(
                    "Reasoning effort",
                    value: Binding(
                        get: { textDraft.reasoningEffort ?? 0.9 },
                        set: { textDraft.reasoningEffort = min(max($0, 0), 0.99) }
                    )
                )
            }
            StudioPathField(
                label: "Evaluation prompts",
                placeholder: "Optional eval JSON or JSONL",
                path: $textDraft.evalPath,
                allowedContentTypes: [.json, .plainText]
            )
            Toggle("Open local training visualizer", isOn: $textDraft.visualize)
            if textDraft.visualize {
                Stepper(
                    "Visualizer port \(textDraft.visualizePort)",
                    value: $textDraft.visualizePort,
                    in: 1...65_535
                )
            }
        }
    }

    private var musicControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            Text("Music adapter training")
                .font(MereRunTheme.sectionFont)
            labeledTextField("ACE-Step model", placeholder: "music-acestep", text: $musicDraft.model)
            Picker("Adapter kind", selection: $musicDraft.musicTrainingKind) {
                Text("LoRA").tag("lora")
                Text("LoKr").tag("lokr")
            }
            coreOptimizerControls(draft: $musicDraft)
            if musicDraft.musicTrainingKind == "lokr" {
                Stepper(
                    "Factor \(musicDraft.musicTrainingFactor)",
                    value: $musicDraft.musicTrainingFactor,
                    in: -1...1_024
                )
            }
            numberField("AdamW weight decay", value: $musicDraft.musicTrainingWeightDecay)
            VStack(alignment: .leading, spacing: 5) {
                Text("Maximum clip duration \(musicDraft.musicTrainingMaxDuration, specifier: "%.1f")s")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Slider(value: $musicDraft.musicTrainingMaxDuration, in: 1...600, step: 1)
            }
            Stepper(
                "Log every \(musicDraft.musicTrainingLogEvery) steps",
                value: $musicDraft.musicTrainingLogEvery,
                in: 1...1_000
            )
            StudioPathField(
                label: "Checkpoint root",
                placeholder: "Auto-discover",
                path: $musicDraft.musicCheckpointsRoot,
                picksDirectory: true
            )
            HStack {
                labeledTextField(
                    "Decoder",
                    placeholder: "acestep-v15-turbo",
                    text: $musicDraft.musicDecoderSubdirectory
                )
                labeledTextField("VAE", placeholder: "vae", text: $musicDraft.musicVAESubdirectory)
            }
            labeledTextField(
                "Text encoder",
                placeholder: "Auto-discover",
                text: $musicDraft.musicTextSubdirectory
            )
        }
    }

    private func coreOptimizerControls(draft: Binding<CommandDraft>) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Stepper(
                    "Steps \(draft.wrappedValue.steps)",
                    value: draft.steps,
                    in: 1...1_000_000
                )
                Stepper(
                    "Batch \(draft.wrappedValue.batchSize)",
                    value: draft.batchSize,
                    in: 1...256
                )
            }
            HStack {
                Stepper(
                    "Rank \(draft.wrappedValue.rank)",
                    value: draft.rank,
                    in: 1...1_024
                )
                numberField("Alpha", value: draft.alpha)
            }
            numberField("Learning rate", value: draft.learningRate)
            labeledTextField("Seed", placeholder: "42", text: draft.seed)
        }
    }

    private var actionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioPathField(
                label: "Adapter output",
                placeholder: "Output .safetensors",
                path: activeOutputBinding
            )
            HStack {
                Button {
                    preflight()
                } label: {
                    Label(kind == .music ? "Validate" : "Preflight", systemImage: "checkmark.shield")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                Button {
                    startTraining()
                } label: {
                    Label(activeDraft.trainingResumePath?.isBlank == false ? "Resume training" : "Start training", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
            }
        }
    }

    private var activeOutputBinding: Binding<String> {
        switch kind {
        case .image: $imageDraft.outputPath
        case .text: $textDraft.outputPath
        case .music: $musicDraft.outputPath
        }
    }

    private var dashboard: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                currentRunDashboard
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                comparisonDashboard
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                recentRuns
            }
            .padding(18)
        }
    }

    private var currentRunDashboard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Live training")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                if let requestID,
                   let progress = controller.progressByRequestID[requestID] {
                    Text(progress.label)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            if let snapshot = currentSnapshot {
                trainingMetrics(snapshot)
                StudioTrainingLossChart(points: snapshot.lossPoints)
                    .frame(height: 220)
                    .merePanel()
                if !snapshot.samples.isEmpty {
                    Text("Sample progression")
                        .font(MereRunTheme.sectionFont)
                    sampleGallery(snapshot.samples)
                }
                if !snapshot.checkpoints.isEmpty {
                    Text("Checkpoints")
                        .font(MereRunTheme.sectionFont)
                    artifactStrip(snapshot.checkpoints)
                }
            } else if let requestID {
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: [.image, .text, .audio]
                )
                .frame(minHeight: 300)
            } else {
                ContentUnavailableView(
                    "No training run selected",
                    systemImage: "chart.xyaxis.line",
                    description: Text("Inspect a dataset, preflight the request, then start training.")
                )
                .frame(minHeight: 280)
            }
        }
    }

    private var comparisonDashboard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Compare runs")
                .font(MereRunTheme.sectionFont)
            HStack {
                runPicker("A", selection: $compareA)
                runPicker("B", selection: $compareB)
            }
            let first = compareA.flatMap(snapshotForRun)
            let second = compareB.flatMap(snapshotForRun)
            if first != nil || second != nil {
                HStack(spacing: 10) {
                    comparisonCard("A", snapshot: first)
                    comparisonCard("B", snapshot: second)
                }
            } else {
                Text("Completed and in-progress training runs will appear here.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }

    private var recentRuns: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Training history")
                .font(MereRunTheme.sectionFont)
            ForEach(trainingRuns.prefix(8)) { item in
                Button {
                    requestID = item.id
                    currentSnapshot = snapshotForRun(item.id)
                } label: {
                    HStack {
                        Image(systemName: item.status == .completed ? "checkmark.circle.fill" : "circle.dotted")
                            .foregroundStyle(item.status == .failed ? MereRunTheme.red : MereRunTheme.accent)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.displayTitle)
                                .font(.system(size: 12.5, weight: .semibold))
                            Text(item.createdAt.formatted(date: .abbreviated, time: .shortened))
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                        Spacer()
                        Text(item.status.rawValue.capitalized)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                    .padding(9)
                    .merePanel()
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func trainingMetrics(_ snapshot: StudioTrainingSnapshot) -> some View {
        let latest = snapshot.latest
        return HStack(spacing: 8) {
            dashboardMetric(
                "Step",
                latest?.totalSteps.map { "\(latest?.step ?? 0) / \($0)" } ?? "\(latest?.step ?? 0)"
            )
            dashboardMetric(
                "Loss",
                latest?.loss.map { String(format: "%.6f", $0) } ?? "—"
            )
            dashboardMetric(
                "Progress",
                snapshot.progress.map { "\(Int(($0 * 100).rounded()))%" } ?? "—"
            )
            dashboardMetric("Samples", "\(snapshot.samples.count)")
            dashboardMetric("Checkpoints", "\(snapshot.checkpoints.count)")
        }
    }

    private func comparisonCard(_ label: String, snapshot: StudioTrainingSnapshot?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(MereRunTheme.accent)
            if let snapshot {
                trainingMetricsCompact(snapshot)
                StudioTrainingLossChart(points: snapshot.lossPoints)
                    .frame(height: 150)
                if let sample = snapshot.samples.last {
                    StudioAsyncImagePreview(
                        url: sample,
                        maxPixelSize: 900,
                        contentMode: .fit,
                        fallbackSystemImage: "photo"
                    )
                    .frame(height: 160)
                }
            } else {
                ContentUnavailableView("Choose a run", systemImage: "chart.xyaxis.line")
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 240, alignment: .topLeading)
        .merePanel()
    }

    private func trainingMetricsCompact(_ snapshot: StudioTrainingSnapshot) -> some View {
        HStack {
            Text("step \(snapshot.latest?.step ?? 0)")
            Spacer()
            if let loss = snapshot.latest?.loss {
                Text("loss \(loss, specifier: "%.5f")")
            }
        }
        .font(MereRunTheme.captionFont)
        .foregroundStyle(MereRunTheme.textMuted)
    }

    private func sampleGallery(_ urls: [URL]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(urls.suffix(12), id: \.self) { url in
                    Button {
                        QuickLookCoordinator.shared.preview(url)
                    } label: {
                        StudioAsyncImagePreview(
                            url: url,
                            maxPixelSize: 480,
                            contentMode: .fill,
                            fallbackSystemImage: "photo"
                        )
                        .frame(width: 130, height: 130)
                        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
                    }
                    .buttonStyle(.plain)
                    .help(url.lastPathComponent)
                }
            }
        }
    }

    private func artifactStrip(_ urls: [URL]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 7) {
                ForEach(urls, id: \.self) { url in
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    } label: {
                        Label(url.lastPathComponent, systemImage: "shippingbox")
                    }
                    .buttonStyle(.bordered)
                    .help(url.path)
                }
            }
        }
    }

    private func runPicker(_ label: String, selection: Binding<UUID?>) -> some View {
        Picker(label, selection: selection) {
            Text("None").tag(UUID?.none)
            ForEach(trainingRuns) { item in
                Text(item.displayTitle).tag(Optional(item.id))
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func dashboardMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .monospacedDigit()
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private func datasetMetric(_ label: String, _ value: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text("\(value)")
                .font(.system(size: 15, weight: .bold, design: .rounded))
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private func datasetPreviewCard(_ preview: StudioTrainingDatasetPreview) -> some View {
        Button {
            selectedDatasetPreview = preview.id
            if let audioURL = preview.audioURL {
                QuickLookCoordinator.shared.preview(audioURL)
            } else if let imageURL = preview.imageURL {
                QuickLookCoordinator.shared.preview(imageURL)
            }
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                if let imageURL = preview.imageURL {
                    StudioAsyncImagePreview(
                        url: imageURL,
                        maxPixelSize: 360,
                        contentMode: .fill,
                        fallbackSystemImage: "photo"
                    )
                    .frame(width: 115, height: 84)
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
                } else {
                    Image(systemName: preview.audioURL == nil ? "curlybraces" : "waveform")
                        .font(.system(size: 24, weight: .semibold))
                        .foregroundStyle(MereRunTheme.accent)
                        .frame(width: 115, height: 50)
                }
                Text(preview.title)
                    .font(.system(size: 10.5, weight: .semibold))
                    .lineLimit(1)
                Text(preview.detail)
                    .font(.system(size: 9.5))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(3)
            }
            .padding(7)
            .frame(width: 130, height: preview.imageURL == nil ? 120 : 155, alignment: .topLeading)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(selectedDatasetPreview == preview.id ? MereRunTheme.accentSoft : MereRunTheme.surface)
            }
        }
        .buttonStyle(.plain)
    }

    private func inspectDataset() {
        let path = activeDraft.inputPath
        guard !path.isBlank else {
            statusMessage = "Choose a dataset first."
            return
        }
        datasetSnapshot = StudioTrainingDatasetSnapshot.inspect(kind: kind, path: path)
        selectedDatasetPreview = datasetSnapshot?.previews.first?.id
        statusMessage = "Dataset inspection complete."
    }

    private func preflight() {
        inspectDataset()
        guard validateDraft(allowsExistingOutput: true) else { return }
        if kind == .music {
            statusMessage = datasetSnapshot?.usableRecords == datasetSnapshot?.totalRecords
                ? "Music dataset is ready. ACE-Step will validate model resources when training starts."
                : "Resolve the dataset issues before training."
            return
        }
        var draft = activeDraft
        if kind == .image {
            draft.preflight = true
            draft.json = true
        } else {
            draft.dryRun = true
            draft.json = true
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: kind.templateID,
            mode: kind.mode,
            draft: draft,
            controller: controller,
            library: library
        )
        statusMessage = "Preflight submitted."
    }

    private func startTraining() {
        inspectDataset()
        guard validateDraft(allowsExistingOutput: false) else { return }
        requestID = StudioSpecialistRunner.submit(
            templateID: kind.templateID,
            mode: kind.mode,
            draft: activeDraft,
            controller: controller,
            library: library
        )
        currentSnapshot = StudioTrainingSnapshot.load(outputPath: activeDraft.outputPath)
        statusMessage = activeDraft.trainingResumePath?.isBlank == false
            ? "Checkpoint resume submitted."
            : "Training submitted."
        advanceOutput()
        seedComparisons()
    }

    private func validateDraft(allowsExistingOutput: Bool) -> Bool {
        guard let snapshot = datasetSnapshot,
              snapshot.totalRecords > 0,
              snapshot.usableRecords == snapshot.totalRecords else {
            statusMessage = "Inspect and repair the dataset before training."
            return false
        }
        let draft = activeDraft
        guard !draft.outputPath.isBlank,
              draft.outputPath.lowercased().hasSuffix(".safetensors") else {
            statusMessage = "Choose a .safetensors output path."
            return false
        }
        if !allowsExistingOutput, FileManager.default.fileExists(atPath: draft.outputPath) {
            statusMessage = "The output already exists. Choose a new path to preserve the existing adapter."
            return false
        }
        if kind == .image,
           let resume = draft.trainingResumePath,
           !resume.isBlank,
           !FileManager.default.fileExists(atPath: resume) {
            statusMessage = "The selected resume checkpoint does not exist."
            return false
        }
        guard draft.steps > 0, draft.rank > 0, draft.learningRate > 0 else {
            statusMessage = "Steps, rank, and learning rate must be positive."
            return false
        }
        return true
    }

    private func refreshSnapshot() {
        if let requestID,
           let item = library.items.first(where: { $0.id == requestID }),
           let output = item.commandDraft?.outputPath {
            currentSnapshot = StudioTrainingSnapshot.load(outputPath: output)
        }
    }

    private func snapshotForRun(_ id: UUID) -> StudioTrainingSnapshot? {
        guard let item = library.items.first(where: { $0.id == id }),
              let output = item.commandDraft?.outputPath else { return nil }
        return StudioTrainingSnapshot.load(outputPath: output)
    }

    private func seedComparisons() {
        if compareA == nil { compareA = trainingRuns.first?.id }
        if compareB == nil { compareB = trainingRuns.dropFirst().first?.id }
    }

    private func advanceOutput() {
        switch kind {
        case .image:
            imageDraft.outputPath = Self.timestampedOutput(prefix: "image-adapter")
            imageDraft.trainingResumePath = nil
        case .text:
            textDraft.outputPath = Self.timestampedOutput(prefix: "text-adapter")
        case .music:
            musicDraft.outputPath = Self.timestampedOutput(prefix: "music-adapter")
        }
    }

    private func labeledTextField(
        _ label: String,
        placeholder: String,
        text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: text)
                .mereField()
        }
    }

    private func numberField(_ label: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(label, value: value, format: .number.precision(.significantDigits(1...8)))
                .mereField()
        }
    }

    nonisolated private static func timestampedOutput(prefix: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/MereRun/Training", isDirectory: true)
            .appendingPathComponent(
                "\(prefix)-\(formatter.string(from: Date())).safetensors",
                isDirectory: false
            )
            .path
    }
}

private struct StudioTrainingLossChart: View {
    let points: [(step: Int, loss: Double)]

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                let inset: CGFloat = 26
                let rect = CGRect(
                    x: inset,
                    y: 12,
                    width: max(1, size.width - inset - 12),
                    height: max(1, size.height - inset - 18)
                )
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: rect.minX, y: rect.minY))
                        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
                        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
                    },
                    with: .color(MereRunTheme.border),
                    lineWidth: 1
                )
                guard points.count > 1 else {
                    context.draw(
                        Text("Waiting for loss points")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted),
                        at: CGPoint(x: rect.midX, y: rect.midY)
                    )
                    return
                }
                let minStep = points.map(\.step).min() ?? 0
                let maxStep = points.map(\.step).max() ?? 1
                let minLoss = points.map(\.loss).min() ?? 0
                let maxLoss = points.map(\.loss).max() ?? 1
                func x(_ step: Int) -> CGFloat {
                    rect.minX + CGFloat(Double(step - minStep) / Double(max(maxStep - minStep, 1))) * rect.width
                }
                func y(_ loss: Double) -> CGFloat {
                    let range = max(maxLoss - minLoss, 0.000_000_1)
                    return rect.maxY - CGFloat((loss - minLoss) / range) * rect.height
                }
                var path = Path()
                for (index, point) in points.enumerated() {
                    let location = CGPoint(x: x(point.step), y: y(point.loss))
                    if index == 0 {
                        path.move(to: location)
                    } else {
                        path.addLine(to: location)
                    }
                }
                context.stroke(
                    path,
                    with: .linearGradient(
                        Gradient(colors: [MereRunTheme.accent, MereRunTheme.green]),
                        startPoint: CGPoint(x: rect.minX, y: rect.midY),
                        endPoint: CGPoint(x: rect.maxX, y: rect.midY)
                    ),
                    style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round)
                )
            }
        }
        .padding(8)
        .accessibilityElement()
        .accessibilityLabel("Training loss chart with \(points.count) points")
    }
}
