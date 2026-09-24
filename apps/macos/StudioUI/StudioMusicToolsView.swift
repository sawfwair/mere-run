import AppKit
import Foundation
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

enum StudioMusicTool: String, CaseIterable, Identifiable {
    case analyze
    case transcribe
    case serve

    var id: String { rawValue }

    var title: String {
        switch self {
        case .analyze: "Analyze"
        case .transcribe: "Transcribe"
        case .serve: "Resident Server"
        }
    }

    var symbol: String {
        switch self {
        case .analyze: "waveform.badge.magnifyingglass"
        case .transcribe: "pianokeys"
        case .serve: "bolt.horizontal.circle"
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .analyze: .musicAnalyze
        case .transcribe: .musicTranscribe
        case .serve: .musicServe
        }
    }
}

struct StudioMusicToolsView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Owned by the shell's task control. Music hosts Analyze and Transcribe; Server hosts the
    /// resident music server.
    @Binding var tool: StudioMusicTool
    /// The tools this host offers; the rail is hidden when there is only one.
    let tools: [StudioMusicTool]
    @StudioStoredValue("MusicTools.analyzeDraft") private var analyzeDraft: CommandDraft = CommandDraft()
    @StudioStoredValue("MusicTools.transcribeDraft") private var transcribeDraft: CommandDraft = CommandDraft()
    @StudioStoredValue("MusicTools.serveDraft") private var serveDraft: CommandDraft = CommandDraft()
    @StudioStoredValue("requestID") private var requestID: UUID? = nil
    @State private var statusMessage: String?

    init(tool: Binding<StudioMusicTool>, tools: [StudioMusicTool]) {
        _tool = tool
        self.tools = tools

        var analyze = CommandCatalog.template(id: .musicAnalyze)?.defaultDraft() ?? CommandDraft()
        analyze.model = analyze.model.isBlank ? "music-acestep" : analyze.model
        _analyzeDraft = StudioStoredValue(wrappedValue: analyze, "MusicTools.analyzeDraft")

        var transcribe = CommandCatalog.template(id: .musicTranscribe)?.defaultDraft() ?? CommandDraft()
        transcribe.model = transcribe.model.isBlank ? "music-muscriptor-medium" : transcribe.model
        transcribe.outputPath = Self.timestampedOutput(prefix: "transcription", extension: "mid")
        transcribe.musicContextOutput = Self.timestampedOutput(prefix: "musical-context", extension: "json")
        _transcribeDraft = StudioStoredValue(wrappedValue: transcribe, "MusicTools.transcribeDraft")

        let serve = CommandCatalog.template(id: .musicServe)?.defaultDraft() ?? CommandDraft()
        _serveDraft = StudioStoredValue(wrappedValue: serve, "MusicTools.serveDraft")
    }

    private var activeDraft: CommandDraft {
        switch tool {
        case .analyze: analyzeDraft
        case .transcribe: transcribeDraft
        case .serve: serveDraft
        }
    }

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    var body: some View {
        StudioAnalysisLayout { configuration } result: { resultPane }
        .studioTaskCommand(tool.templateID, draft: activeDraft)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                switch tool {
                case .analyze:
                    analysisControls
                case .transcribe:
                    transcriptionControls
                case .serve:
                    serverControls
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

    private var analysisControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            StudioPathField(
                label: "Source music",
                placeholder: "WAV, MP3, M4A…",
                path: $analyzeDraft.inputPath,
                allowedContentTypes: [.audio]
            )
            labeledTextField("ACE-Step model", placeholder: "music-acestep", text: $analyzeDraft.model)
            Toggle("Limit analysis duration", isOn: $analyzeDraft.useDuration)
            if analyzeDraft.useDuration {
                VStack(alignment: .leading, spacing: 5) {
                    Text("First \(analyzeDraft.durationSeconds, specifier: "%.1f") seconds")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Slider(value: $analyzeDraft.durationSeconds, in: 1...600, step: 1)
                }
            }
            Stepper(
                "Maximum LM tokens \(analyzeDraft.musicAnalysisMaxTokens)",
                value: $analyzeDraft.musicAnalysisMaxTokens,
                in: 64...16_384,
                step: 64
            )
            numberField("LM temperature", value: $analyzeDraft.musicAnalysisTemperature)
            Stepper(
                "LM top-k \(analyzeDraft.musicLMTopK)",
                value: $analyzeDraft.musicLMTopK,
                in: 0...2_048
            )
            numberField("LM top-p", value: $analyzeDraft.musicLMTopP)
            Toggle("Include raw LM response", isOn: $analyzeDraft.musicIncludeRawLM)
            Toggle("Include serialized audio codes", isOn: $analyzeDraft.musicIncludeAudioCodes)
            checkpointControls(draft: $analyzeDraft, includesLanguageModel: true)
            runButton("Analyze music")
        }
    }

    private var transcriptionControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            StudioPathField(
                label: "Source mix",
                placeholder: "WAV, MP3, M4A, FLAC…",
                path: $transcribeDraft.inputPath,
                allowedContentTypes: [.audio]
            )
            labeledTextField(
                "MuScriptor model",
                placeholder: "music-muscriptor-medium",
                text: $transcribeDraft.model
            )
            Picker("Architecture", selection: $transcribeDraft.musicTranscribeVariant) {
                Text("Infer from model").tag("")
                Text("Small").tag("small")
                Text("Medium").tag("medium")
                Text("Large").tag("large")
            }
            Picker("Output format", selection: $transcribeDraft.musicTranscribeFormat) {
                Text("MIDI").tag("midi")
                Text("JSON").tag("json")
                Text("JSON Lines").tag("jsonl")
            }
            .onChange(of: transcribeDraft.musicTranscribeFormat) { _, format in
                transcribeDraft.outputPath = Self.replacingExtension(
                    transcribeDraft.outputPath,
                    with: format == "midi" ? "mid" : format
                )
            }
            StudioInstrumentPicker(value: $transcribeDraft.musicInstruments)
            HStack {
                Stepper(
                    "Beam \(transcribeDraft.musicBeamSize)",
                    value: $transcribeDraft.musicBeamSize,
                    in: 1...32
                )
                Stepper(
                    "Chunk batch \(transcribeDraft.musicChunkBatchSize)",
                    value: $transcribeDraft.musicChunkBatchSize,
                    in: 1...64
                )
            }
            Stepper(
                "Tokens per chunk \(transcribeDraft.musicMaxTokensPerChunk)",
                value: $transcribeDraft.musicMaxTokensPerChunk,
                in: 64...16_000,
                step: 64
            )
            Picker("Compute type", selection: $transcribeDraft.musicDType) {
                Text("BFloat16").tag("bfloat16")
                Text("Float16").tag("float16")
                Text("Float32").tag("float32")
            }
            Toggle("Sample instead of greedy decode", isOn: $transcribeDraft.musicSampling)
            if transcribeDraft.musicSampling {
                numberField("Temperature", value: $transcribeDraft.temperature)
            }
            Toggle("Require EOS for every chunk", isOn: $transcribeDraft.musicStrictEOS)
            Toggle("Detect tempo, meter, key, and beat phase", isOn: Binding(
                get: { !transcribeDraft.musicNoMusicalContext },
                set: { transcribeDraft.musicNoMusicalContext = !$0 }
            ))
            StudioPathField(
                label: "Transcription output",
                placeholder: "MIDI or event output",
                path: $transcribeDraft.outputPath
            )
            if !transcribeDraft.musicNoMusicalContext {
                StudioPathField(
                    label: "Musical context JSON",
                    placeholder: "Tempo/key/beat sidecar",
                    path: $transcribeDraft.musicContextOutput
                )
            }
            runButton("Transcribe to \(transcribeDraft.musicTranscribeFormat.uppercased())")
        }
    }

    private var serverControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            Text("Keep ACE-Step, its language model, and adapter stack warm behind a local API.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)
            HStack {
                labeledTextField("Host", placeholder: "127.0.0.1", text: $serveDraft.host)
                Stepper("Port \(String(serveDraft.port))", value: $serveDraft.port, in: 1...65_535)
            }
            labeledTextField("ACE-Step model", placeholder: "music-acestep", text: $serveDraft.model)
            checkpointControls(draft: $serveDraft, includesLanguageModel: true)
            StudioPathField(
                label: "Adapters",
                placeholder: "One adapter path per line",
                path: $serveDraft.musicAdapterPaths,
                allowsMultipleSelection: true,
                allowedContentTypes: [.data]
            )
            if !serveDraft.musicAdapterPaths.isBlank {
                Picker("Adapter format", selection: $serveDraft.musicAdapterKind) {
                    Text("Automatic").tag("auto")
                    Text("LoRA").tag("lora")
                    Text("LoKr").tag("lokr")
                }
                labeledTextField(
                    "Adapter scales",
                    placeholder: "One scale per line",
                    text: $serveDraft.musicAdapterScales
                )
            }
            SecureField("Optional bearer token", text: $serveDraft.apiKey)
                .mereField()
            Text("The token is injected through MERERUN_API_KEY and is never placed in argv.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            StudioMusicServerControl(server: controller.musicServer, draft: serveDraft)
        }
    }

    private func checkpointControls(
        draft: Binding<CommandDraft>,
        includesLanguageModel: Bool
    ) -> some View {
        DisclosureGroup("Checkpoint layout") {
            VStack(alignment: .leading, spacing: 10) {
                StudioPathField(
                    label: "Checkpoint root",
                    placeholder: "Auto-discover",
                    path: draft.musicCheckpointsRoot,
                    picksDirectory: true
                )
                HStack {
                    labeledTextField(
                        "Decoder",
                        placeholder: "acestep-v15-turbo",
                        text: draft.musicDecoderSubdirectory
                    )
                    labeledTextField("VAE", placeholder: "vae", text: draft.musicVAESubdirectory)
                }
                if includesLanguageModel {
                    labeledTextField(
                        "LM model",
                        placeholder: "music-acestep-lm-1.7b",
                        text: draft.musicLMModel
                    )
                    labeledTextField(
                        "LM subdirectory",
                        placeholder: "Auto-discover",
                        text: draft.musicLMSubdirectory
                    )
                }
                labeledTextField(
                    "Text encoder",
                    placeholder: "Auto-discover",
                    text: draft.musicTextSubdirectory
                )
            }
            .padding(.top, 9)
        }
    }

    @ViewBuilder
    private var resultPane: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(tool.title)
                    .font(MereRunTheme.sectionFont)
                Spacer()
                // The resident server is a process, not a Library run; its state is in the pane.
                if let item, tool != .serve {
                    Text(item.status.rawValue.capitalized)
                        .font(.system(size: 10.5, weight: .bold))
                        .foregroundStyle(item.status == .failed ? MereRunTheme.red : MereRunTheme.textMuted)
                }
            }
            switch tool {
            case .analyze:
                analysisResult
            case .transcribe:
                transcriptionResult
            case .serve:
                serverResult
            }
        }
        .padding(18)
    }

    @ViewBuilder
    private var analysisResult: some View {
        if !analyzeDraft.inputPath.isBlank, FileManager.default.fileExists(atPath: analyzeDraft.inputPath) {
            StudioAudioPlayerView(url: URL(fileURLWithPath: analyzeDraft.inputPath))
                .frame(height: 180)
                .merePanel()
        }
        if let analysis {
            StudioMusicAnalysisView(analysis: analysis)
        } else {
            StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text, .audio])
        }
    }

    @ViewBuilder
    private var transcriptionResult: some View {
        if let midiURL, let summary = StudioMIDISummary.load(from: midiURL) {
            HStack(spacing: 8) {
                resultMetric("Notes", "\(summary.notes.count)")
                resultMetric("Tracks", "\(summary.trackCount)")
                resultMetric("PPQ", "\(summary.ticksPerQuarter)")
                resultMetric(
                    "Tempo",
                    summary.tempoMicrosecondsPerQuarter.map {
                        "\(Int((60_000_000.0 / Double($0)).rounded())) BPM"
                    } ?? "—"
                )
            }
            StudioMIDIPianoRoll(summary: summary)
                .frame(minHeight: 320)
                .merePanel()
            HStack {
                Button("Quick Look") { QuickLookCoordinator.shared.preview(midiURL) }
                    .buttonStyle(.mereSecondary)
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([midiURL]) }
                    .buttonStyle(.mereSecondary)
            }
        } else {
            StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text, .audio])
        }
    }

    private var serverResult: some View {
        StudioMusicServerStatus(server: controller.musicServer, host: serveDraft.host, port: serveDraft.port)
    }

    /// `music analyze` prints its result on stdout, which the Library row keeps as the run's text.
    private var analysis: StudioMusicAnalysisDocument? {
        item?.outputText.flatMap(StudioMusicAnalysisDocument.decode)
    }

    private var midiURL: URL? {
        item?.allArtifactURLs.first {
            ["mid", "midi"].contains($0.pathExtension.lowercased())
        } ?? {
            let url = URL(fileURLWithPath: transcribeDraft.outputPath)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }()
    }

    private func resultMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private func runButton(_ title: String) -> some View {
        Button {
            submit()
        } label: {
            Label(title, systemImage: tool.symbol)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.merePrimary)
    }

    private func submit() {
        let draft = activeDraft
        if tool != .serve, !FileManager.default.fileExists(atPath: draft.inputPath) {
            statusMessage = "Choose a valid source audio file."
            return
        }
        if tool == .transcribe, draft.outputPath.isBlank {
            statusMessage = "Choose a transcription output."
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: tool.templateID,
            mode: .music,
            draft: draft,
            controller: controller,
            library: library
        )
        statusMessage = "\(tool.title) submitted."
        if tool == .transcribe {
            let format = transcribeDraft.musicTranscribeFormat
            transcribeDraft.outputPath = Self.timestampedOutput(
                prefix: "transcription",
                extension: format == "midi" ? "mid" : format
            )
            transcribeDraft.musicContextOutput = Self.timestampedOutput(
                prefix: "musical-context",
                extension: "json"
            )
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

    private static func timestampedOutput(prefix: String, extension pathExtension: String) -> String {
        StudioSpecialistFiles.outputFile(domain: .music, name: prefix, fileExtension: pathExtension).path
    }

    nonisolated private static func replacingExtension(_ path: String, with pathExtension: String) -> String {
        URL(fileURLWithPath: path)
            .deletingPathExtension()
            .appendingPathExtension(pathExtension)
            .path
    }
}

private struct StudioMIDIPianoRoll: View {
    let summary: StudioMIDISummary

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                let notes = summary.notes
                guard !notes.isEmpty,
                      let pitches = summary.pitchRange else {
                    let text = context.resolve(
                        Text("No note events found")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    )
                    context.draw(text, at: CGPoint(x: size.width / 2, y: size.height / 2))
                    return
                }
                let pitchSpan = max(1, pitches.upperBound - pitches.lowerBound + 1)
                let totalTicks = max(1, summary.totalTicks)
                let rowHeight = max(2, size.height / CGFloat(pitchSpan))
                for note in notes {
                    let x = CGFloat(note.startTick) / CGFloat(totalTicks) * size.width
                    let width = max(
                        2,
                        CGFloat(note.durationTicks) / CGFloat(totalTicks) * size.width
                    )
                    let pitchOffset = note.pitch - pitches.lowerBound
                    let y = size.height - CGFloat(pitchOffset + 1) * rowHeight
                    let hue = Double(note.channel) / 16
                    context.fill(
                        Path(
                            roundedRect: CGRect(
                                x: x,
                                y: y,
                                width: width,
                                height: max(1.5, rowHeight - 1)
                            ),
                            cornerRadius: 1.5
                        ),
                        with: .color(
                            Color(
                                hue: hue,
                                saturation: 0.7,
                                brightness: 0.92,
                                opacity: 0.45 + 0.55 * Double(note.velocity) / 127
                            )
                        )
                    )
                }
            }
            .background {
                LinearGradient(
                    colors: [MereRunTheme.surfaceRaised, MereRunTheme.surface],
                    startPoint: .top,
                    endPoint: .bottom
                )
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Piano roll with \(summary.notes.count) notes")
    }
}

/// What ACE-Step understood about a piece: tempo, key, meter, and language as tiles, the caption
/// as prose, the lyrics it heard, and — only when the run asked to keep them — the language
/// model's whole reply and the audio codes, folded away.
private struct StudioMusicAnalysisView: View {
    let analysis: StudioMusicAnalysisDocument
    @Environment(\.studioModelTitles) private var titles
    @State private var showsRawReply = false
    @State private var showsAudioCodes = false

    private var tiles: [(label: String, value: String)] {
        var tiles: [(String, String)] = []
        if let tempo = analysis.tempoDescription { tiles.append(("Tempo", tempo)) }
        if let key = analysis.metadata.keyscale, !key.isBlank { tiles.append(("Key", key)) }
        if let meter = analysis.metadata.timesignature, !meter.isBlank { tiles.append(("Meter", meter)) }
        if let language = analysis.languageDescription { tiles.append(("Language", language)) }
        tiles.append(("Analyzed", analysis.analyzedDescription))
        return tiles
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                HStack(spacing: 8) {
                    ForEach(tiles, id: \.label) { tile in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(tile.label)
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                            Text(tile.value)
                                .font(.system(size: 14, weight: .bold, design: .rounded))
                                .lineLimit(1)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .merePanel()
                        .accessibilityElement(children: .combine)
                    }
                }

                if let caption = analysis.caption {
                    section("What it sounds like") {
                        Text(caption)
                            .font(MereRunTheme.bodyFont)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if let lyrics = analysis.lyrics {
                    section("Lyrics") {
                        Text(lyrics)
                            .font(MereRunTheme.bodyFont)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if analysis.caption == nil, analysis.lyrics == nil, analysis.metadata.bpm == nil {
                    Text("The model found no tempo, key, caption, or lyrics in this recording.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }

                Text("\(StudioModelNaming.displayName(analysis.model, titles: titles)) · \(URL(fileURLWithPath: analysis.audio).lastPathComponent)")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)

                if let reply = analysis.rawLMOutput, !reply.isBlank {
                    disclosure("Model reply", isExpanded: $showsRawReply, text: reply)
                }
                if let codes = analysis.audioCodes, !codes.isBlank {
                    disclosure("Audio codes", isExpanded: $showsAudioCodes, text: codes)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Music analysis")
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(MereRunTheme.sectionFont)
            content()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private func disclosure(_ title: String, isExpanded: Binding<Bool>, text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(MereRunTheme.Motion.quick) { isExpanded.wrappedValue.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: isExpanded.wrappedValue ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                    Text(title)
                        .font(.caption.weight(.medium))
                }
                .foregroundStyle(MereRunTheme.textMuted)
            }
            .buttonStyle(.plain)
            if isExpanded.wrappedValue {
                ScrollView {
                    Text(text)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 220)
                .padding(10)
                .merePanel()
            }
        }
    }
}

/// Start or Stop for the resident music server. It observes `controller.musicServer`, the process
/// the menu bar can stop too, rather than whichever run happens to hold the console.
private struct StudioMusicServerControl: View {
    @ObservedObject var server: StudioServiceProcess
    let draft: CommandDraft

    var body: some View {
        if server.state.isRunning {
            HStack {
                Button { Task { _ = await server.restart(draft: draft) } } label: {
                    Label("Restart", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
                .help("Restart the server with these settings")
                Button { server.stop() } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
            }
            .disabled(server.state.isStopping)
        } else {
            Button { server.start(draft: draft) } label: {
                Label("Start resident server", systemImage: StudioMusicTool.serve.symbol)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.merePrimary)
        }
    }
}

/// The resident music server's state, endpoint, and live log.
private struct StudioMusicServerStatus: View {
    @ObservedObject var server: StudioServiceProcess
    let host: String
    let port: Int

    private var isRunning: Bool { server.state.isRunning && !server.state.isStopping }

    var body: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(isRunning ? MereRunTheme.green.opacity(0.14) : MereRunTheme.surface)
                Image(systemName: isRunning ? "bolt.horizontal.circle.fill" : "bolt.slash.circle")
                    .font(.system(size: 58, weight: .semibold))
                    .foregroundStyle(isRunning ? MereRunTheme.green : MereRunTheme.textMuted)
            }
            .frame(width: 130, height: 130)
            Text("Resident music is \(StudioResidentServerCopy.title(server.state).lowercased())")
                .font(MereRunTheme.titleFont)
            if let failure = StudioResidentServerCopy.failure(server.state) {
                Text(failure)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .multilineTextAlignment(.center)
                    .textSelection(.enabled)
            }
            Text("http://\(host):\(String(port))")
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .textSelection(.enabled)
            Button("Open health endpoint") {
                guard let url = URL(string: "http://\(host):\(port)/health") else { return }
                NSWorkspace.shared.open(url)
            }
            .buttonStyle(.mereSecondary)
            .disabled(!isRunning)
            if let job = server.job {
                StudioServiceLogView(job: job)
                    .frame(minHeight: 200)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
