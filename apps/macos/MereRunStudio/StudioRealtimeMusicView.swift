import AppKit
import SwiftUI

/// A native transport/control surface for the CLI's long-lived Magenta RT2 session.
/// The CLI remains the source of truth; controls are sent over its documented stdin protocol.
struct StudioRealtimeMusicView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    @State private var prompt: String
    @State private var model: String
    @State private var duration = 300.0
    @State private var capturesAudio = true
    @State private var playsAudio = true
    @State private var outputPath: String
    @State private var temperature = 1.0
    @State private var topK = 100
    @State private var cfgMusicCoCa = 3.0
    @State private var cfgNotes = 5.0
    @State private var cfgDrums = 1.0
    @State private var drumless = false
    @State private var style = "streaming"
    @State private var midiInputs: [String] = []
    @State private var midiInput = ""
    @State private var midiChannel = "all"
    @State private var requestID: UUID?
    @State private var startedAt: Date?
    @State private var statusMessage: String?

    init(initialDraft: StudioDraft) {
        _prompt = State(initialValue: initialDraft.prompt)
        _model = State(initialValue: Self.preferredModel(from: initialDraft.model))
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Music/MereRun", isDirectory: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        _outputPath = State(
            initialValue: directory
                .appendingPathComponent("realtime-\(formatter.string(from: Date())).wav")
                .path
        )
    }

    nonisolated static func preferredModel(from activeModel: String) -> String {
        let normalized = activeModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized.localizedCaseInsensitiveContains("magenta-rt2")
            ? normalized
            : "music-magenta-rt2-small"
    }

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    private var isLive: Bool {
        guard let requestID else { return false }
        return controller.canSteerRealtimeMusic(requestID: requestID)
    }

    private var isSessionPending: Bool {
        item?.status == .queued || item?.status == .running
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                transport
                    .frame(minWidth: 300, idealWidth: 380, maxWidth: 380)
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                liveControls
                    .frame(maxWidth: .infinity)
            }
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            footer
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            midiInputs = await controller.loadMIDIInputs()
        }
    }

    private var liveBadge: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(isLive ? MereRunTheme.green : MereRunTheme.textMuted)
                .frame(width: 8, height: 8)
            Text(isLive ? "LIVE" : item?.status.rawValue.uppercased() ?? "READY")
                .font(.system(size: 10.5, weight: .bold))
                .kerning(0.7)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background { Capsule().fill((isLive ? MereRunTheme.green : MereRunTheme.surfaceRaised).opacity(0.16)) }
    }

    private var transport: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                Text("Session")
                    .font(MereRunTheme.sectionFont)

                VStack(alignment: .leading, spacing: 5) {
                    Text("Starting prompt")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField("Describe the music", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .mereField()
                }

                VStack(alignment: .leading, spacing: 5) {
                    Text("Model")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField("music-magenta-rt2-small", text: $model)
                        .mereField()
                }

                Stepper(
                    "Maximum duration \(Int(duration))s",
                    value: $duration,
                    in: 5...3_600,
                    step: 5
                )
                .font(MereRunTheme.captionFont)

                Toggle("Play through default output", isOn: $playsAudio)
                    .font(MereRunTheme.captionFont)
                Toggle("Record WAV", isOn: $capturesAudio)
                    .font(MereRunTheme.captionFont)

                if capturesAudio {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Recording")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                        HStack(spacing: 7) {
                            TextField("Output WAV", text: $outputPath)
                                .mereField()
                            Button("Choose…", action: chooseOutput)
                                .controlSize(.small)
                        }
                    }
                }

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                VStack(alignment: .leading, spacing: 8) {
                    Text("MIDI input")
                        .font(MereRunTheme.sectionFont)
                    Picker("Device", selection: $midiInput) {
                        Text("None").tag("")
                        ForEach(midiInputs, id: \.self) { Text($0).tag($0) }
                    }
                    Picker("Channel", selection: $midiChannel) {
                        Text("All").tag("all")
                        ForEach(1...16, id: \.self) { Text("\($0)").tag(String($0)) }
                    }
                    .disabled(midiInput.isEmpty)
                    if midiInputs.isEmpty {
                        Text("No CoreMIDI input sources detected.")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }

                if let progress = activeProgress {
                    VStack(alignment: .leading, spacing: 5) {
                        ProgressView(value: progress.fractionCompleted)
                            .tint(MereRunTheme.accent)
                        Text(progress.detail ?? progress.label)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                } else if let startedAt, isLive {
                    TimelineView(.periodic(from: startedAt, by: 1)) { context in
                        let elapsed = context.date.timeIntervalSince(startedAt)
                        VStack(alignment: .leading, spacing: 5) {
                            ProgressView(value: min(1, elapsed / duration))
                                .tint(MereRunTheme.accent)
                            Text("\(Int(elapsed))s of \(Int(duration))s")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                    }
                }

                if let statusMessage {
                    Text(statusMessage)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .padding(20)
        }
    }

    private var liveControls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                HStack {
                    Text("Live steering")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    Button("Apply controls", action: applyControls)
                        .buttonStyle(.borderedProminent)
                        .tint(MereRunTheme.accent)
                        .disabled(!isLive)
                }

                VStack(alignment: .leading, spacing: 5) {
                    Text("Next musical direction")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    HStack(spacing: 8) {
                        TextField("Shift the performance…", text: $prompt)
                            .mereField()
                        Button("Send prompt") {
                            send("prompt \(prompt)")
                        }
                        .disabled(!isLive || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }

                Picker("Style conditioning", selection: $style) {
                    Text("Streaming").tag("streaming")
                    Text("Full").tag("full")
                }
                .pickerStyle(.segmented)

                controlSlider("Temperature", value: $temperature, range: 0.1...2, format: "%.2f")
                Stepper("Top-k \(topK)", value: $topK, in: 0...512)
                    .font(MereRunTheme.captionFont)
                controlSlider("MusicCoCa guidance", value: $cfgMusicCoCa, range: 0...10, format: "%.1f")
                controlSlider("Note guidance", value: $cfgNotes, range: 0...12, format: "%.1f")
                controlSlider("Drum guidance", value: $cfgDrums, range: 0...8, format: "%.1f")
                Toggle("Drumless", isOn: $drumless)
                    .font(MereRunTheme.captionFont)

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                Text("Playable notes")
                    .font(MereRunTheme.sectionFont)
                Text("Tap a note to inject a short MIDI gesture without a controller.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 8), spacing: 6) {
                    ForEach(Array(60...75), id: \.self) { note in
                        Button(noteName(note)) { playNote(note) }
                            .buttonStyle(.bordered)
                            .disabled(!isLive)
                    }
                }

                HStack(spacing: 9) {
                    Button("Reset state") { send("reset") }
                        .disabled(!isLive)
                    Button("Stop session", role: .destructive) { send("quit") }
                        .disabled(!isLive)
                }
            }
            .padding(20)
        }
    }

    private var footer: some View {
        HStack {
            liveBadge
            if let url = item?.outputURL {
                Button {
                    QuickLookCoordinator.shared.preview(url)
                } label: {
                    Label("Preview recording", systemImage: "play.circle")
                }
            }
            Spacer()
            Button(
                isLive ? "Running" : (item?.status == .queued ? "Queued" : "Start session"),
                action: start
            )
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .disabled(
                    isSessionPending
                        || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        || (!playsAudio && !capturesAudio)
                )
        }
        .padding(16)
    }

    private var activeProgress: StudioRunProgress? {
        guard let requestID, controller.activeRunRequestID == requestID else { return nil }
        return controller.currentProgress
    }

    private func controlSlider(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        format: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .font(MereRunTheme.captionFont)
            Slider(value: value, in: range)
        }
    }

    private func start() {
        guard let template = CommandCatalog.template(id: .musicRealtime) else {
            statusMessage = "Realtime music template is unavailable."
            return
        }
        var draft = template.defaultDraft()
        draft.prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        draft.model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        draft.durationSeconds = duration
        draft.musicPlay = playsAudio
        draft.outputPath = capturesAudio ? outputPath : ""
        draft.musicInteractive = true
        draft.musicStyleConditioning = style
        draft.musicTemperature = temperature
        draft.musicTopK = topK
        draft.musicCFGMusicCoCa = cfgMusicCoCa
        draft.musicCFGNotes = cfgNotes
        draft.musicCFGDrums = cfgDrums
        draft.musicDrumless = drumless
        draft.musicMIDIInput = midiInput
        draft.musicMIDIChannel = midiChannel

        let request = StudioRunRequest(
            mode: .music,
            templateID: .musicRealtime,
            template: template,
            draft: draft
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        library.start(request: request, commandPreview: preview, status: status)
        requestID = request.id
        startedAt = Date()
        statusMessage = status == .queued ? "Queued behind the active job." : "Starting Magenta RT2…"
        _ = controller.run(studio: request)
    }

    private func applyControls() {
        send("style \(style)")
        send("temp \(temperature)")
        send("topk \(topK)")
        send("mc \(cfgMusicCoCa)")
        send("notes \(cfgNotes)")
        send("drums \(cfgDrums)")
        send("drumless \(drumless ? "on" : "off")")
        statusMessage = "Live controls queued."
    }

    private func send(_ command: String) {
        guard let requestID else { return }
        if !controller.submitRealtimeMusicCommand(command, requestID: requestID) {
            statusMessage = "The realtime process is not ready for live control."
        }
    }

    private func playNote(_ note: Int) {
        send("noteon \(note)")
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(280))
            send("noteoff \(note)")
        }
    }

    private func noteName(_ midi: Int) -> String {
        let names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
        return "\(names[midi % 12])\(midi / 12 - 1)"
    }

    private func chooseOutput() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = URL(fileURLWithPath: outputPath).lastPathComponent
        if panel.runModal() == .OK, let url = panel.url {
            outputPath = url.pathExtension.lowercased() == "wav"
                ? url.path
                : url.appendingPathExtension("wav").path
        }
    }
}
