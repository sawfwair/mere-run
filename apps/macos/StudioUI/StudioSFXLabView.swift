import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

enum StudioSFXTask: String, CaseIterable, Identifiable {
    case generate
    case video
    case condition
    case encode
    case decode
    case score

    var id: String { rawValue }

    var title: String {
        switch self {
        case .generate: "Generate"
        case .video: "Video Foley"
        case .condition: "Conditioning"
        case .encode: "Encode"
        case .decode: "Decode"
        case .score: "CLAP Score"
        }
    }

    var symbol: String {
        switch self {
        case .generate: "waveform.badge.plus"
        case .video: "film.stack"
        case .condition: "text.badge.star"
        case .encode: "square.stack.3d.down.forward"
        case .decode: "waveform.path"
        case .score: "checkmark.seal"
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .generate: .sfxGenerate
        case .video: .sfxVideo
        case .condition: .sfxConditionText
        case .encode: .sfxAEEncode
        case .decode: .sfxAEDecode
        case .score: .sfxClapScore
        }
    }
}

struct StudioNPYMetadata: Equatable {
    let version: String
    let descriptor: String
    let shape: String
    let fortranOrder: Bool
    let byteCount: Int

    static func load(from url: URL) -> StudioNPYMetadata? {
        guard let data = try? Data(contentsOf: url),
              data.count >= 10,
              Array(data.prefix(6)) == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] else {
            return nil
        }
        let major = Int(data[6])
        let minor = Int(data[7])
        let headerLength: Int
        let headerStart: Int
        if major <= 1 {
            headerLength = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            guard data.count >= 12 else { return nil }
            headerLength = Int(data[8])
                | (Int(data[9]) << 8)
                | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            headerStart = 12
        }
        guard headerLength >= 0, data.count >= headerStart + headerLength,
              let header = String(
                  data: data.subdata(in: headerStart..<(headerStart + headerLength)),
                  encoding: .ascii
              ) else {
            return nil
        }
        return StudioNPYMetadata(
            version: "\(major).\(minor)",
            descriptor: dictionaryValue("descr", in: header) ?? "unknown",
            shape: tupleValue("shape", in: header) ?? "unknown",
            fortranOrder: header.contains("'fortran_order': True")
                || header.contains("\"fortran_order\": true"),
            byteCount: data.count
        )
    }

    private static func dictionaryValue(_ key: String, in header: String) -> String? {
        for quote in ["'", "\""] {
            let marker = "\(quote)\(key)\(quote)"
            guard let keyRange = header.range(of: marker),
                  let colon = header[keyRange.upperBound...].firstIndex(of: ":") else { continue }
            let tail = header[header.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            guard let first = tail.first, first == "'" || first == "\"",
                  let end = tail.dropFirst().firstIndex(of: first) else { continue }
            return String(tail[tail.index(after: tail.startIndex)..<end])
        }
        return nil
    }

    private static func tupleValue(_ key: String, in header: String) -> String? {
        for quote in ["'", "\""] {
            let marker = "\(quote)\(key)\(quote)"
            guard let keyRange = header.range(of: marker),
                  let open = header[keyRange.upperBound...].firstIndex(of: "("),
                  let close = header[open...].firstIndex(of: ")") else { continue }
            return String(header[open...close])
        }
        return nil
    }
}

struct StudioSFXLabView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Owned by the shell's task control; the rail mirrors it for the tasks this host offers.
    @Binding var task: StudioSFXTask
    let tasks: [StudioSFXTask]
    @State private var generateDraft: CommandDraft
    @State private var videoDraft: CommandDraft
    @State private var conditionDraft: CommandDraft
    @State private var encodeDraft: CommandDraft
    @State private var decodeDraft: CommandDraft
    @State private var scoreDraft: CommandDraft
    @State private var requestID: UUID?
    @State private var statusMessage: String?

    init(task: Binding<StudioSFXTask>, tasks: [StudioSFXTask], initialDraft: StudioDraft) {
        _task = task
        self.tasks = tasks

        var generate = CommandCatalog.template(id: .sfxGenerate)?.defaultDraft() ?? CommandDraft()
        generate.prompt = initialDraft.prompt
        generate.secondaryText = initialDraft.secondaryText
        generate.model = initialDraft.model.localizedCaseInsensitiveContains("sfx")
            ? initialDraft.model
            : generate.model
        generate.outputPath = Self.timestampedOutput(prefix: "sfx", extension: "wav")
        _generateDraft = State(initialValue: generate)

        var video = CommandCatalog.template(id: .sfxVideo)?.defaultDraft() ?? CommandDraft()
        video.prompt = initialDraft.prompt
        video.inputPath = initialDraft.inputPath
        video.outputPath = Self.timestampedOutput(prefix: "foley", extension: "wav")
        _videoDraft = State(initialValue: video)

        var condition = CommandCatalog.template(id: .sfxConditionText)?.defaultDraft() ?? CommandDraft()
        condition.prompt = initialDraft.prompt
        condition.outputPath = Self.timestampedOutput(prefix: "conditioning", extension: "safetensors")
        _conditionDraft = State(initialValue: condition)

        var encode = CommandCatalog.template(id: .sfxAEEncode)?.defaultDraft() ?? CommandDraft()
        encode.inputPath = initialDraft.inputPath
        encode.outputPath = Self.timestampedOutput(prefix: "latents", extension: "npy")
        _encodeDraft = State(initialValue: encode)

        var decode = CommandCatalog.template(id: .sfxAEDecode)?.defaultDraft() ?? CommandDraft()
        decode.outputPath = Self.timestampedOutput(prefix: "decoded", extension: "wav")
        _decodeDraft = State(initialValue: decode)

        var score = CommandCatalog.template(id: .sfxClapScore)?.defaultDraft() ?? CommandDraft()
        score.prompt = initialDraft.prompt
        score.inputPath = initialDraft.inputPath
        _scoreDraft = State(initialValue: score)
    }

    private var activeDraft: CommandDraft {
        switch task {
        case .generate: generateDraft
        case .video: videoDraft
        case .condition: conditionDraft
        case .encode: encodeDraft
        case .decode: decodeDraft
        case .score: scoreDraft
        }
    }

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    var body: some View {
        HStack(spacing: 0) {
            if tasks.count > 1 {
                taskRail
                    .frame(width: 175)
                Divider().overlay(MereRunTheme.border.opacity(0.6))
            }
            configuration
                .frame(minWidth: 300, idealWidth: 420, maxWidth: 420)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            resultPane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private var taskRail: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(tasks) { item in
                Button {
                    task = item
                    statusMessage = nil
                } label: {
                    Label(item.title, systemImage: item.symbol)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                                .fill(task == item ? MereRunTheme.accentSoft : Color.clear)
                        }
                }
                .buttonStyle(.plain)
            }
            Spacer()
            VStack(alignment: .leading, spacing: 5) {
                Text("CLI-backed")
                    .font(.system(size: 10, weight: .bold))
                Text("Woosh and MMAudio stay authoritative; this lab owns preparation and feedback.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(10)
            .merePanel()
        }
        .padding(12)
    }

    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                switch task {
                case .generate:
                    generationControls(videoConditioned: false)
                case .video:
                    generationControls(videoConditioned: true)
                case .condition:
                    conditioningControls
                case .encode:
                    codecControls(isEncoding: true)
                case .decode:
                    codecControls(isEncoding: false)
                case .score:
                    scoreControls
                }
                if let statusMessage {
                    Text(statusMessage)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .padding(18)
        }
    }

    @ViewBuilder
    private func generationControls(videoConditioned: Bool) -> some View {
        let draft = videoConditioned ? $videoDraft : $generateDraft
        if videoConditioned {
            StudioPathField(
                label: "Source video or Synchformer features",
                placeholder: "Video input",
                path: draft.inputPath,
                allowedContentTypes: [.movie, .video, .data]
            )
        }
        labeledTextEditor(
            "Sound direction",
            placeholder: "Heavy rain hitting a metal roof, close perspective",
            text: draft.prompt
        )
        labeledTextField(
            "Avoid",
            placeholder: "music, speech, distortion",
            text: draft.secondaryText
        )
        labeledTextField("Model", placeholder: "Managed model or local root", text: draft.model)

        HStack {
            VStack(alignment: .leading, spacing: 5) {
                Text("Duration \(draft.wrappedValue.durationSeconds, specifier: "%.1f")s")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Slider(value: draft.durationSeconds, in: 0.5...60, step: 0.5)
            }
            Stepper("Steps \(draft.wrappedValue.steps)", value: draft.steps, in: 1...200)
        }
        VStack(alignment: .leading, spacing: 5) {
            Text("Guidance \(draft.wrappedValue.cfgScale, specifier: "%.2f")")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Slider(value: draft.cfgScale, in: 0...20, step: 0.25)
        }
        HStack {
            labeledTextField("Seed", placeholder: "Random", text: draft.seed)
            labeledTextField("Renoise", placeholder: "auto", text: draft.sfxRenoise)
        }

        if videoConditioned {
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            Text("Synchronization")
                .font(MereRunTheme.sectionFont)
            labeledTextField(
                "Synchformer model",
                placeholder: "sfx-woosh-synchformer",
                text: draft.sfxSynchformerModel
            )
            Stepper(
                "Sync batch: \(draft.wrappedValue.sfxSyncBatchSize)",
                value: draft.sfxSyncBatchSize,
                in: 1...64
            )
            Stepper(
                "CLIP batch: \(draft.wrappedValue.sfxClipBatchSize)",
                value: draft.sfxClipBatchSize,
                in: 1...64
            )
            Toggle("Preflight only", isOn: draft.preflight)
        }

        StudioPathField(
            label: "Output WAV",
            placeholder: "Output WAV",
            path: draft.outputPath
        )
        runButton(videoConditioned ? "Generate synchronized foley" : "Generate sound effect")
    }

    private var conditioningControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            labeledTextEditor(
                "Conditioning prompt",
                placeholder: "An enormous stone door grinding open",
                text: $conditionDraft.prompt
            )
            labeledTextField(
                "Model",
                placeholder: "Managed Woosh model or local root",
                text: $conditionDraft.model
            )
            StudioPathField(
                label: "Tensor output",
                placeholder: "Conditioning safetensors",
                path: $conditionDraft.outputPath
            )
            Text("Exports the exact text-conditioning tensors used by Woosh for inspection and reuse.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            runButton("Export conditioning")
        }
    }

    private func codecControls(isEncoding: Bool) -> some View {
        let draft = isEncoding ? $encodeDraft : $decodeDraft
        return VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            StudioPathField(
                label: isEncoding ? "Source audio" : "Latent array",
                placeholder: isEncoding ? "Audio file" : "NumPy .npy file",
                path: draft.inputPath,
                allowedContentTypes: isEncoding ? [.audio] : [.data]
            )
            labeledTextField(
                "Model",
                placeholder: "Managed Woosh model or local root",
                text: draft.model
            )
            StudioPathField(
                label: isEncoding ? "Latent output" : "Audio output",
                placeholder: isEncoding ? "NumPy .npy file" : "WAV file",
                path: draft.outputPath
            )
            Text(
                isEncoding
                    ? "Encode audio into the native [1, 128, frames] Woosh latent representation."
                    : "Decode a native Woosh latent array back to 44.1 kHz audio."
            )
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
            runButton(isEncoding ? "Encode latents" : "Decode audio")
        }
    }

    private var scoreControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            labeledTextEditor(
                "Target description",
                placeholder: "A clean glass bottle breaking on concrete",
                text: $scoreDraft.prompt
            )
            StudioPathField(
                label: "Audio candidate",
                placeholder: "Audio file to score",
                path: $scoreDraft.inputPath,
                allowedContentTypes: [.audio]
            )
            labeledTextField(
                "CLAP model",
                placeholder: "Managed model or local root",
                text: $scoreDraft.model
            )
            Text("CLAP measures semantic prompt/audio alignment. Use it to rank candidates, not as an absolute quality score.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            runButton("Score alignment")
        }
    }

    @ViewBuilder
    private var resultPane: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(task.title)
                    .font(MereRunTheme.sectionFont)
                Spacer()
                if let item {
                    Text(item.status.rawValue.capitalized)
                        .font(.system(size: 10.5, weight: .bold))
                        .foregroundStyle(item.status == .failed ? MereRunTheme.red : MereRunTheme.textMuted)
                }
            }

            if task == .video, !videoDraft.inputPath.isBlank {
                StudioSFXSyncReview(
                    videoURL: URL(fileURLWithPath: videoDraft.inputPath),
                    audioURL: resultAudioURL
                )
            } else if task == .score, let score = clapScore {
                clapScoreView(score)
            } else if let tensorURL, let metadata = StudioNPYMetadata.load(from: tensorURL) {
                tensorInspector(url: tensorURL, metadata: metadata)
            } else {
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: [.audio, .video, .text, .image]
                )
            }
        }
        .padding(18)
    }

    private var resultAudioURL: URL? {
        item?.allArtifactURLs.first { StudioOutputFileKind.classify($0) == .audio }
    }

    private var tensorURL: URL? {
        item?.allArtifactURLs.first { $0.pathExtension.lowercased() == "npy" }
    }

    private var clapScore: Double? {
        guard task == .score, let text = item?.outputText else { return nil }
        let tokens = text.split { $0.isWhitespace || $0 == ":" || $0 == "=" }
        return tokens.reversed().compactMap { Double($0) }.first
    }

    private func clapScoreView(_ score: Double) -> some View {
        VStack(spacing: 22) {
            ZStack {
                Circle()
                    .stroke(MereRunTheme.border, lineWidth: 18)
                Circle()
                    .trim(from: 0, to: min(1, max(0, score)))
                    .stroke(
                        MereRunTheme.accent,
                        style: StrokeStyle(lineWidth: 18, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                VStack {
                    Text(score, format: .number.precision(.fractionLength(3)))
                        .font(.system(size: 38, weight: .bold, design: .rounded))
                    Text("semantic alignment")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .frame(width: 230, height: 230)
            if !scoreDraft.inputPath.isBlank {
                StudioAudioPlayerView(url: URL(fileURLWithPath: scoreDraft.inputPath))
                    .frame(height: 210)
                    .merePanel()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func tensorInspector(url: URL, metadata: StudioNPYMetadata) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Image(systemName: "square.stack.3d.down.forward")
                .font(.system(size: 46, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
            Text(url.lastPathComponent)
                .font(MereRunTheme.titleFont)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                metric("Shape", metadata.shape)
                metric("DType", metadata.descriptor)
                metric("NumPy", metadata.version)
                metric("Layout", metadata.fortranOrder ? "Fortran" : "C order")
                metric("Size", ByteCountFormatter.string(fromByteCount: Int64(metadata.byteCount), countStyle: .file))
            }
            HStack {
                Button("Quick Look") { QuickLookCoordinator.shared.preview(url) }
                    .buttonStyle(.bordered)
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    .buttonStyle(.bordered)
            }
            Spacer()
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .merePanel()
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .textSelection(.enabled)
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private func runButton(_ title: String) -> some View {
        Button {
            submit()
        } label: {
            Label(title, systemImage: task.symbol)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(MereRunTheme.accent)
    }

    private func submit() {
        let draft = activeDraft
        if [.generate, .video, .condition, .score].contains(task), draft.prompt.isBlank {
            statusMessage = "Enter a sound description."
            return
        }
        if [.video, .encode, .decode, .score].contains(task),
           !FileManager.default.fileExists(atPath: draft.inputPath) {
            statusMessage = "Choose a valid input file."
            return
        }
        if task != .score, draft.outputPath.isBlank {
            statusMessage = "Choose an output path."
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: task.templateID,
            mode: .sfx,
            draft: draft,
            controller: controller,
            library: library
        )
        statusMessage = "\(task.title) submitted."
        advanceOutput()
    }

    private func advanceOutput() {
        switch task {
        case .generate:
            generateDraft.outputPath = Self.timestampedOutput(prefix: "sfx", extension: "wav")
        case .video:
            videoDraft.outputPath = Self.timestampedOutput(prefix: "foley", extension: "wav")
        case .condition:
            conditionDraft.outputPath = Self.timestampedOutput(prefix: "conditioning", extension: "safetensors")
        case .encode:
            encodeDraft.outputPath = Self.timestampedOutput(prefix: "latents", extension: "npy")
        case .decode:
            decodeDraft.outputPath = Self.timestampedOutput(prefix: "decoded", extension: "wav")
        case .score:
            break
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

    private func labeledTextEditor(
        _ label: String,
        placeholder: String,
        text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextEditor(text: text)
                .font(MereRunTheme.bodyFont)
                .scrollContentBackground(.hidden)
                .padding(9)
                .frame(minHeight: 90)
                .merePanel()
                .overlay(alignment: .topLeading) {
                    if text.wrappedValue.isEmpty {
                        Text(placeholder)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .padding(14)
                            .allowsHitTesting(false)
                    }
                }
        }
    }

    nonisolated private static func timestampedOutput(prefix: String, extension pathExtension: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Music/MereRun/Sound FX", isDirectory: true)
            .appendingPathComponent(
                "\(prefix)-\(formatter.string(from: Date())).\(pathExtension)",
                isDirectory: false
            )
            .path
    }
}

private struct StudioSFXSyncReview: View {
    let videoURL: URL
    let audioURL: URL?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Picture and generated foley")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            StudioVideoPlayerView(url: videoURL)
                .frame(maxHeight: .infinity)
                .merePanel()
            if let audioURL {
                StudioAudioPlayerView(url: audioURL)
                    .frame(height: 210)
                    .merePanel()
            } else {
                HStack {
                    ProgressView()
                    Text("The generated waveform will appear beneath the picture.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .merePanel()
            }
        }
    }
}
