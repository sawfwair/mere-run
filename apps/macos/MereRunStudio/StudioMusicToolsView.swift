import AppKit
import Foundation
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

struct StudioMIDINote: Identifiable, Equatable {
    let id: Int
    let startTick: Int
    let durationTicks: Int
    let pitch: Int
    let velocity: Int
    let channel: Int
}

struct StudioMIDISummary: Equatable {
    let format: Int
    let trackCount: Int
    let ticksPerQuarter: Int
    let notes: [StudioMIDINote]
    let tempoMicrosecondsPerQuarter: Int?

    var pitchRange: ClosedRange<Int>? {
        guard let minimum = notes.map(\.pitch).min(),
              let maximum = notes.map(\.pitch).max() else { return nil }
        return minimum...maximum
    }

    var totalTicks: Int {
        notes.map { $0.startTick + $0.durationTicks }.max() ?? 0
    }

    static func load(from url: URL) -> StudioMIDISummary? {
        guard let data = try? Data(contentsOf: url), data.count >= 14,
              String(data: data.prefix(4), encoding: .ascii) == "MThd" else {
            return nil
        }
        let headerLength = readUInt32(data, at: 4)
        guard headerLength >= 6, data.count >= 8 + Int(headerLength) else { return nil }
        let format = Int(readUInt16(data, at: 8))
        let trackCount = Int(readUInt16(data, at: 10))
        let division = Int(readUInt16(data, at: 12))
        let ticksPerQuarter = division & 0x8000 == 0 ? division : 480
        var offset = 8 + Int(headerLength)
        var notes: [StudioMIDINote] = []
        var tempo: Int?
        var nextID = 0

        for _ in 0..<trackCount {
            guard offset + 8 <= data.count,
                  String(data: data[offset..<(offset + 4)], encoding: .ascii) == "MTrk" else {
                break
            }
            let length = Int(readUInt32(data, at: offset + 4))
            let end = min(data.count, offset + 8 + length)
            var cursor = offset + 8
            var tick = 0
            var runningStatus: UInt8?
            var openNotes: [Int: (tick: Int, velocity: Int)] = [:]

            while cursor < end {
                guard let delta = readVariableLength(data, cursor: &cursor, end: end) else { break }
                tick += delta
                guard cursor < end else { break }
                var status = data[cursor]
                if status & 0x80 != 0 {
                    cursor += 1
                    if status < 0xF0 { runningStatus = status }
                } else if let runningStatus {
                    status = runningStatus
                } else {
                    break
                }

                if status == 0xFF {
                    guard cursor < end else { break }
                    let type = data[cursor]
                    cursor += 1
                    guard let metaLength = readVariableLength(data, cursor: &cursor, end: end),
                          cursor + metaLength <= end else { break }
                    if type == 0x51, metaLength == 3 {
                        tempo = Int(data[cursor]) << 16
                            | Int(data[cursor + 1]) << 8
                            | Int(data[cursor + 2])
                    }
                    cursor += metaLength
                    continue
                }
                if status == 0xF0 || status == 0xF7 {
                    guard let systemLength = readVariableLength(data, cursor: &cursor, end: end),
                          cursor + systemLength <= end else { break }
                    cursor += systemLength
                    continue
                }

                let command = status & 0xF0
                let channel = Int(status & 0x0F)
                let dataLength = command == 0xC0 || command == 0xD0 ? 1 : 2
                guard cursor + dataLength <= end else { break }
                let first = Int(data[cursor])
                let second = dataLength == 2 ? Int(data[cursor + 1]) : 0
                cursor += dataLength

                let key = channel * 128 + first
                if command == 0x90, second > 0 {
                    openNotes[key] = (tick, second)
                } else if command == 0x80 || (command == 0x90 && second == 0),
                          let opened = openNotes.removeValue(forKey: key) {
                    notes.append(
                        StudioMIDINote(
                            id: nextID,
                            startTick: opened.tick,
                            durationTicks: max(1, tick - opened.tick),
                            pitch: first,
                            velocity: opened.velocity,
                            channel: channel
                        )
                    )
                    nextID += 1
                }
            }
            offset = end
        }
        return StudioMIDISummary(
            format: format,
            trackCount: trackCount,
            ticksPerQuarter: ticksPerQuarter,
            notes: notes.sorted {
                $0.startTick == $1.startTick ? $0.pitch < $1.pitch : $0.startTick < $1.startTick
            },
            tempoMicrosecondsPerQuarter: tempo
        )
    }

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        guard offset + 1 < data.count else { return 0 }
        return UInt16(data[offset]) << 8 | UInt16(data[offset + 1])
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        guard offset + 3 < data.count else { return 0 }
        return UInt32(data[offset]) << 24
            | UInt32(data[offset + 1]) << 16
            | UInt32(data[offset + 2]) << 8
            | UInt32(data[offset + 3])
    }

    private static func readVariableLength(
        _ data: Data,
        cursor: inout Int,
        end: Int
    ) -> Int? {
        var value = 0
        for _ in 0..<4 {
            guard cursor < end else { return nil }
            let byte = data[cursor]
            cursor += 1
            value = (value << 7) | Int(byte & 0x7F)
            if byte & 0x80 == 0 { return value }
        }
        return value
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
    @State private var analyzeDraft: CommandDraft
    @State private var transcribeDraft: CommandDraft
    @State private var serveDraft: CommandDraft
    @State private var requestID: UUID?
    @State private var statusMessage: String?

    init(tool: Binding<StudioMusicTool>, tools: [StudioMusicTool]) {
        _tool = tool
        self.tools = tools

        var analyze = CommandCatalog.template(id: .musicAnalyze)?.defaultDraft() ?? CommandDraft()
        analyze.model = analyze.model.isBlank ? "music-acestep" : analyze.model
        _analyzeDraft = State(initialValue: analyze)

        var transcribe = CommandCatalog.template(id: .musicTranscribe)?.defaultDraft() ?? CommandDraft()
        transcribe.model = transcribe.model.isBlank ? "music-muscriptor-medium" : transcribe.model
        transcribe.outputPath = Self.timestampedOutput(prefix: "transcription", extension: "mid")
        transcribe.musicContextOutput = Self.timestampedOutput(prefix: "musical-context", extension: "json")
        _transcribeDraft = State(initialValue: transcribe)

        let serve = CommandCatalog.template(id: .musicServe)?.defaultDraft() ?? CommandDraft()
        _serveDraft = State(initialValue: serve)
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
        HStack(spacing: 0) {
            if tools.count > 1 {
                toolRail
                    .frame(width: 185)
                Divider().overlay(MereRunTheme.border.opacity(0.6))
            }
            configuration
                .frame(width: 430)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            resultPane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private var toolRail: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(tools) { item in
                Button {
                    tool = item
                    statusMessage = nil
                } label: {
                    Label(item.title, systemImage: item.symbol)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                                .fill(tool == item ? MereRunTheme.accentSoft : Color.clear)
                        }
                }
                .buttonStyle(.plain)
            }
            Spacer()
        }
        .padding(12)
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
                    if controller.isRunning {
                        Button(role: .destructive) {
                            controller.cancel()
                            statusMessage = "Stopping resident music server…"
                        } label: {
                            Label("Stop server", systemImage: "stop.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }
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
            labeledTextField(
                "Expected instruments",
                placeholder: "voice,drums,bass — blank means automatic",
                text: $transcribeDraft.musicInstruments
            )
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
                Stepper("Port \(serveDraft.port)", value: $serveDraft.port, in: 1...65_535)
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
            runButton(controller.isRunning ? "Server already running" : "Start resident server")
                .disabled(controller.isRunning)
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
                if let item {
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
        if let object = analysisJSONObject {
            StudioJSONSummaryView(value: object)
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
                    .buttonStyle(.bordered)
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([midiURL]) }
                    .buttonStyle(.bordered)
            }
        } else {
            StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text, .audio])
        }
    }

    private var serverResult: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(controller.isRunning ? MereRunTheme.green.opacity(0.14) : MereRunTheme.surface)
                Image(systemName: controller.isRunning ? "bolt.horizontal.circle.fill" : "bolt.slash.circle")
                    .font(.system(size: 58, weight: .semibold))
                    .foregroundStyle(controller.isRunning ? MereRunTheme.green : MereRunTheme.textMuted)
            }
            .frame(width: 130, height: 130)
            Text(controller.isRunning ? "Resident music is running" : "Resident music is stopped")
                .font(MereRunTheme.titleFont)
            Text("http://\(serveDraft.host):\(serveDraft.port)")
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .textSelection(.enabled)
            HStack {
                Button("Open health endpoint") {
                    guard let url = URL(string: "http://\(serveDraft.host):\(serveDraft.port)/health") else {
                        return
                    }
                    NSWorkspace.shared.open(url)
                }
                .buttonStyle(.bordered)
                if controller.isRunning {
                    Button(role: .destructive) { controller.cancel() } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.bordered)
                }
            }
            StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text])
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var analysisJSONObject: Any? {
        guard let text = item?.outputText,
              let data = text.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
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
        .buttonStyle(.borderedProminent)
        .tint(MereRunTheme.accent)
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

    nonisolated private static func timestampedOutput(prefix: String, extension pathExtension: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Music/MereRun/Tools", isDirectory: true)
            .appendingPathComponent(
                "\(prefix)-\(formatter.string(from: Date())).\(pathExtension)",
                isDirectory: false
            )
            .path
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

private struct StudioJSONSummaryView: View {
    let value: Any

    private struct Row: Identifiable {
        let id: String
        let label: String
        let value: String?
        let depth: Int
    }

    private var flattenedRows: [Row] {
        var rows: [Row] = []
        Self.flatten(value, key: nil, depth: 0, path: "root", into: &rows)
        return rows
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(flattenedRows) { row in
                    if let value = row.value {
                        HStack(alignment: .firstTextBaseline) {
                            Text(row.label)
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .frame(width: max(90, 150 - CGFloat(row.depth * 12)), alignment: .leading)
                            Text(value)
                                .font(MereRunTheme.bodyFont)
                                .textSelection(.enabled)
                            Spacer()
                        }
                        .padding(10)
                        .padding(.leading, CGFloat(row.depth * 10))
                        .merePanel()
                    } else {
                        Text(row.label)
                            .font(row.depth == 0 ? MereRunTheme.titleFont : MereRunTheme.sectionFont)
                            .padding(.leading, CGFloat(row.depth * 10))
                            .padding(.top, row.depth == 0 ? 0 : 6)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private static func flatten(
        _ value: Any,
        key: String?,
        depth: Int,
        path: String,
        into rows: inout [Row]
    ) {
        if let dictionary = value as? [String: Any] {
            if let key {
                rows.append(Row(id: path, label: key.humanizedMusicKey, value: nil, depth: depth))
            }
            for childKey in dictionary.keys.sorted() {
                if let child = dictionary[childKey] {
                    flatten(
                        child,
                        key: childKey,
                        depth: depth + 1,
                        path: "\(path).\(childKey)",
                        into: &rows
                    )
                }
            }
        } else if let array = value as? [Any] {
            rows.append(
                Row(
                    id: path,
                    label: key?.humanizedMusicKey ?? "Values",
                    value: array.prefix(40).map { String(describing: $0) }.joined(separator: ", "),
                    depth: depth
                )
            )
        } else {
            rows.append(
                Row(
                    id: path,
                    label: key?.humanizedMusicKey ?? "Value",
                    value: String(describing: value),
                    depth: depth
                )
            )
        }
    }
}

private extension String {
    var humanizedMusicKey: String {
        replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}
