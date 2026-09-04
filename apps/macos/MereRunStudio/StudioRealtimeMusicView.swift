import AppKit
import SwiftUI

/// Music ▸ Realtime: the Session archetype for the CLI's long-lived Magenta RT2 process.
///
/// One column of panels — transport, waveform, steering, session log — over a job bar. The CLI
/// remains the source of truth: launch options become `music realtime` arguments, steering is
/// sent over the documented stdin protocol, the clock and log are read back from the run. The
/// view re-attaches to a session that is already running when the user navigates back to it.
struct StudioRealtimeMusicView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.studioRealtimeSteeringSeed) private var steeringSeed

    // Launch options (fixed once a session starts; edits apply to the next session).
    @State private var model: String
    @State private var duration = 300.0
    @State private var capturesAudio = true
    @State private var playsAudio = true
    @State private var outputPath: String
    @State private var midiInputs: [String] = []
    @State private var midiInput = ""
    @State private var midiChannel = "all"

    // Steering (sent live over stdin; also the initial controls at launch).
    @State private var promptA: String
    @State private var promptB = ""
    @State private var blend = 0.0
    @State private var temperature = 1.0
    @State private var topK = 100
    @State private var guidance = 3.0
    @State private var noteGuidance = 5.0
    @State private var drumGuidance = 1.0
    @State private var drumless = false
    @State private var style = "streaming"

    // Session
    @State private var requestID: UUID?
    @State private var peaks: [Float] = []
    @State private var banner: Banner?
    @State private var showsSessionOptions = false
    @State private var showsMIDIOptions = false
    @State private var showsLog = false
    @State private var showsMoreSteering = false
    @State private var didSeedSteering = false

    private struct Banner: Equatable {
        let severity: MereBanner.Severity
        let text: String

        static func == (lhs: Banner, rhs: Banner) -> Bool {
            lhs.text == rhs.text
        }
    }

    private static let contentWidth: CGFloat = 672
    private static let logLineHeight: CGFloat = 17.8
    private static let logHeight: CGFloat = 120

    init(initialDraft: StudioDraft) {
        _promptA = State(initialValue: initialDraft.prompt)
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

    // MARK: - Derived session state

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    private var isLive: Bool {
        guard let requestID else { return false }
        return controller.canSteerRealtimeMusic(requestID: requestID)
    }

    private var phase: StudioRealtimeJobBar.Phase {
        guard let item else { return .idle }
        if isLive { return .live }
        switch item.status {
        case .queued: return .queued
        case .running: return .live
        case .completed, .failed: return .ended(exitCode: item.exitCode)
        }
    }

    private var isPending: Bool {
        switch phase {
        case .queued, .live: return true
        case .idle, .ended: return false
        }
    }

    private var startedAt: Date? { item?.createdAt }

    private var sessionModel: String { item?.commandDraft?.model ?? model }

    private var renderedSeconds: TimeInterval? {
        guard let requestID else { return nil }
        return StudioRealtimeTransport.renderedSeconds(from: controller.progressByRequestID[requestID])
    }

    private var sessionDuration: Double {
        item?.commandDraft?.durationSeconds ?? duration
    }

    private var recordingURL: URL? {
        let path = item?.commandDraft?.outputPath ?? (capturesAudio ? outputPath : "")
        return path.isBlank ? nil : URL(fileURLWithPath: path)
    }

    private var logLines: [String] {
        guard let requestID, let startedAt else { return [] }
        return StudioRealtimeSessionLog.lines(controller.logs(for: requestID), startedAt: startedAt)
    }

    private var blendedPrompt: String {
        StudioRealtimeSteering.blendedPrompt(a: promptA, b: promptB, blend: blend)
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 14) {
                    if let banner {
                        MereBanner(severity: banner.severity, text: banner.text) {
                            self.banner = nil
                        }
                    }
                    transportPanel
                    waveformPanel
                    steeringPanel
                    logPanel
                }
                .frame(maxWidth: Self.contentWidth)
                .padding(EdgeInsets(top: 22, leading: 24, bottom: 20, trailing: 24))
                .frame(maxWidth: .infinity)
            }
            jobBar
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear(perform: attachToRunningSession)
        .task {
            midiInputs = await controller.loadMIDIInputs()
        }
        .task(id: "\(requestID?.uuidString ?? "none")-\(isLive)") {
            await pollPeaks()
        }
    }

    // MARK: - Transport

    private var transportPanel: some View {
        HStack(spacing: 14) {
            Button(action: toggleSession) {
                ZStack {
                    Circle().fill(MereRunTheme.accent)
                    TransportGlyph(stop: isPending)
                        .stroke(MereRunTheme.onAccent, style: StrokeStyle(lineWidth: 1.15, lineCap: .round, lineJoin: .round))
                        .frame(width: 16, height: 16)
                }
                .frame(width: 40, height: 40)
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(phase == .queued)
            .accessibilityLabel(isPending ? "Stop session" : "Start session")

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    statusBadge
                    Text(StudioRealtimeTransport.engineLabel(model: sessionModel))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
                transportClock
            }

            Spacer(minLength: 8)

            SessionSecondaryButton("Record to file") { showsSessionOptions.toggle() }
                .popover(isPresented: $showsSessionOptions, arrowEdge: .bottom) { sessionOptions }
            SessionSecondaryButton(midiInput.isEmpty ? "MIDI in: none" : "MIDI in: \(midiInput)") {
                showsMIDIOptions.toggle()
            }
            .popover(isPresented: $showsMIDIOptions, arrowEdge: .bottom) { midiOptions }
        }
        .padding(EdgeInsets(top: 14, leading: 18, bottom: 14, trailing: 18))
        .sessionPanel()
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch phase {
        case .idle:
            EmptyView()
        case .queued:
            SessionBadge(text: "QUEUED", color: MereRunTheme.yellow)
        case .live:
            SessionBadge(text: "LIVE", color: MereRunTheme.red)
        case .ended(let exitCode):
            if let exitCode, exitCode != 0 {
                SessionBadge(text: "FAILED", color: MereRunTheme.red)
            } else {
                SessionBadge(text: "ENDED", color: MereRunTheme.textMuted)
            }
        }
    }

    @ViewBuilder
    private var transportClock: some View {
        if phase == .live, let startedAt {
            TimelineView(.periodic(from: startedAt, by: 1)) { context in
                clockText(
                    StudioRealtimeTransport.statusLine(
                        wallElapsed: context.date.timeIntervalSince(startedAt),
                        rendered: renderedSeconds
                    )
                )
            }
        } else {
            clockText(staticClockLine)
        }
    }

    private var staticClockLine: String {
        switch phase {
        case .idle:
            return "00:00 · ready to start"
        case .queued:
            return "00:00 · queued behind the active job"
        case .live:
            return "00:00 · loading model"
        case .ended:
            let ran = renderedSeconds ?? item.map { $0.updatedAt.timeIntervalSince($0.createdAt) } ?? 0
            return "\(StudioRealtimeTransport.timestamp(ran)) · session ended"
        }
    }

    private func clockText(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 12.5, weight: .medium, design: .monospaced))
            .foregroundStyle(MereRunTheme.textPrimary)
            .lineLimit(1)
    }

    // MARK: - Waveform

    private var waveformPanel: some View {
        SessionWaveform(peaks: peaks, played: playedFraction)
            .frame(height: 60)
            .padding(EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16))
            .sessionPanel()
    }

    private var playedFraction: Double {
        switch phase {
        case .idle, .queued:
            return 0
        case .live:
            guard let startedAt, sessionDuration > 0 else { return 0 }
            let playback = StudioRealtimeTransport.playbackPosition(
                wallElapsed: Date().timeIntervalSince(startedAt),
                rendered: renderedSeconds
            )
            return min(1, playback / sessionDuration)
        case .ended:
            return peaks.isEmpty ? 0 : 1
        }
    }

    // MARK: - Steering

    private var steeringPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Steering")
                    .font(.system(size: 12.5, weight: .semibold))
                Spacer()
                Text("changes blend in over 4 bars")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
            SessionHairline()
            VStack(spacing: 16) {
                HStack(alignment: .bottom, spacing: 12) {
                    promptField("Prompt A", text: $promptA)
                    promptField("Prompt B", text: $promptB)
                }
                SessionSlider(
                    label: "Blend A ↔ B",
                    value: $blend,
                    range: 0...1,
                    display: StudioRealtimeSteering.formatBlend,
                    onCommit: sendPrompt
                )
                HStack(spacing: 16) {
                    SessionSlider(
                        label: "Temperature",
                        value: $temperature,
                        range: 0.1...2,
                        display: { String(format: "%.1f", $0) },
                        onCommit: { send("temp \(StudioRealtimeSteering.format(temperature))") }
                    )
                    SessionSlider(
                        label: "Top-k",
                        value: Binding(get: { Double(topK) }, set: { topK = Int($0.rounded()) }),
                        range: 0...512,
                        display: { String(Int($0.rounded())) },
                        onCommit: { send("topk \(topK)") }
                    )
                    SessionSlider(
                        label: "Guidance",
                        value: $guidance,
                        range: 0...10,
                        display: { String(format: "%.1f", $0) },
                        onCommit: { send("mc \(StudioRealtimeSteering.format(guidance))") }
                    )
                }
                moreSteering
            }
            .padding(EdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16))
        }
        .sessionPanel()
    }

    private func promptField(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            TextField("", text: text)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundStyle(MereRunTheme.textPrimary)
                .padding(.horizontal, 10)
                .frame(minHeight: 32)
                .background {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(MereRunTheme.surface)
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                        }
                }
                .onSubmit(sendPrompt)
                .accessibilityLabel(label)
        }
        .frame(maxWidth: .infinity)
    }

    /// The rest of the stdin protocol: style conditioning, the note and drum guidance, the
    /// drumless switch, a playable octave, and reset. Folded so the panel reads like the mockup.
    private var moreSteering: some View {
        VStack(alignment: .leading, spacing: 14) {
            Button {
                withAnimation(MereRunTheme.Motion.quick) { showsMoreSteering.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Text("More steering")
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8, weight: .bold))
                        .rotationEffect(.degrees(showsMoreSteering ? 90 : 0))
                }
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(showsMoreSteering ? "Hide more steering" : "Show more steering")

            if showsMoreSteering {
                Picker("Style conditioning", selection: $style) {
                    Text("Streaming").tag("streaming")
                    Text("Full").tag("full")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .onChange(of: style) { _, newValue in send("style \(newValue)") }

                HStack(spacing: 16) {
                    SessionSlider(
                        label: "Note guidance",
                        value: $noteGuidance,
                        range: 0...12,
                        display: { String(format: "%.1f", $0) },
                        onCommit: { send("notes \(StudioRealtimeSteering.format(noteGuidance))") }
                    )
                    SessionSlider(
                        label: "Drum guidance",
                        value: $drumGuidance,
                        range: 0...8,
                        display: { String(format: "%.1f", $0) },
                        onCommit: { send("drums \(StudioRealtimeSteering.format(drumGuidance))") }
                    )
                }

                Toggle("Drumless", isOn: $drumless)
                    .font(.system(size: 12, weight: .medium))
                    .onChange(of: drumless) { _, newValue in send("drumless \(newValue ? "on" : "off")") }

                VStack(alignment: .leading, spacing: 6) {
                    Text("Playable notes")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 8), spacing: 6) {
                        ForEach(Array(60...75), id: \.self) { note in
                            SessionSecondaryButton(Self.noteName(note)) { playNote(note) }
                                .disabled(!isLive)
                        }
                    }
                }

                HStack(spacing: 8) {
                    SessionSecondaryButton("Reset state") { send("reset") }
                        .disabled(!isLive)
                    SessionSecondaryButton("Apply all controls", action: applyControls)
                        .disabled(!isLive)
                }
            }
        }
    }

    // MARK: - Session log

    private var logPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Session log")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Copy", action: copyLog)
                    .buttonStyle(.plain)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .accessibilityLabel("Copy session log")
            }
            .padding(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
            SessionHairline()
            VStack(alignment: .leading, spacing: 0) {
                let visible = StudioRealtimeSessionLog.visibleTail(
                    logLines, height: Self.logHeight, lineHeight: Self.logLineHeight
                )
                if visible.isEmpty {
                    logLine(phase == .idle ? "Press play to start a session." : "Waiting for the first line…")
                }
                ForEach(Array(visible.enumerated()), id: \.offset) { _, line in
                    logLine(line)
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .frame(height: Self.logHeight, alignment: .top)
            .clipped()
            .padding(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Session log")
        }
        .sessionPanel()
    }

    private func logLine(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11.5, design: .monospaced))
            .foregroundStyle(MereRunTheme.textSecondary)
            .lineLimit(1)
            .frame(height: Self.logLineHeight, alignment: .leading)
    }

    // MARK: - Job bar

    private var jobBar: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(StudioRealtimeJobBar.dotColor(phase: phase))
                .frame(width: 8, height: 8)
            Text("Music · Realtime")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
            Text(StudioRealtimeJobBar.detail(phase: phase, startedAt: startedAt, model: sessionModel))
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer(minLength: 8)
            if case .ended = phase, let url = item?.outputURL {
                SessionSecondaryButton("Preview recording") {
                    QuickLookCoordinator.shared.preview(url)
                }
            }
            SessionSecondaryButton("Cancel", action: cancelSession)
                .disabled(!isLive)
            SessionSecondaryButton("Log") { showsLog.toggle() }
                .popover(isPresented: $showsLog, arrowEdge: .top) { logPopover }
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .frame(maxWidth: .infinity)
        .background(MereRunTheme.background)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.53)).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Job bar")
    }

    private var logPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Session log")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Copy", action: copyLog)
                    .controlSize(.small)
            }
            ScrollView {
                Text(logLines.isEmpty ? "No session output yet." : StudioRealtimeSessionLog.copyText(logLines))
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(14)
        .frame(width: 560, height: 340)
    }

    // MARK: - Popovers

    private var sessionOptions: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Session")
                .font(MereRunTheme.sectionFont)
            if isPending {
                Text("These options apply to the next session.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("Model")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                TextField("music-magenta-rt2-small", text: $model)
                    .mereField()
            }
            Stepper("Maximum duration \(Int(duration)) s", value: $duration, in: 5...3_600, step: 5)
                .font(MereRunTheme.captionFont)
            Toggle("Play through default output", isOn: $playsAudio)
                .font(MereRunTheme.captionFont)
            Toggle("Record to a WAV file", isOn: $capturesAudio)
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
            if !playsAudio && !capturesAudio {
                MereBanner(severity: .warning, text: "Enable playback or recording before starting.")
            }
        }
        .padding(16)
        .frame(width: 360)
    }

    private var midiOptions: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("MIDI input")
                .font(MereRunTheme.sectionFont)
            if isPending {
                Text("The input binds when a session starts; this applies to the next one.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
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
        .padding(16)
        .frame(width: 320)
    }

    // MARK: - Actions

    private func attachToRunningSession() {
        if requestID == nil {
            let running = controller.runningRequestID(for: .musicRealtime)
                ?? library.items.first {
                    $0.templateID == .musicRealtime && ($0.status == .running || $0.status == .queued)
                }?.id
            if let running {
                requestID = running
                if let draft = library.items.first(where: { $0.id == running })?.commandDraft {
                    if promptA.isBlank { promptA = draft.prompt }
                    model = draft.model.isBlank ? model : draft.model
                    duration = draft.durationSeconds
                    playsAudio = draft.musicPlay
                    capturesAudio = !draft.outputPath.isBlank
                    if capturesAudio { outputPath = draft.outputPath }
                    temperature = draft.musicTemperature
                    topK = draft.musicTopK
                    guidance = draft.musicCFGMusicCoCa
                    noteGuidance = draft.musicCFGNotes
                    drumGuidance = draft.musicCFGDrums
                    drumless = draft.musicDrumless
                    style = draft.musicStyleConditioning
                    midiInput = draft.musicMIDIInput
                    midiChannel = draft.musicMIDIChannel
                }
            }
        }
        if let steeringSeed, !didSeedSteering {
            didSeedSteering = true
            promptA = steeringSeed.promptA
            promptB = steeringSeed.promptB
            blend = steeringSeed.blend
        }
    }

    private func toggleSession() {
        if isPending {
            send("quit")
        } else {
            start()
        }
    }

    private func start() {
        guard let template = CommandCatalog.template(id: .musicRealtime) else {
            banner = Banner(severity: .error, text: "The realtime music command is unavailable in this build.")
            return
        }
        let prompt = blendedPrompt
        guard !prompt.isEmpty else {
            banner = Banner(severity: .error, text: "Describe the music in Prompt A before starting.")
            return
        }
        guard playsAudio || capturesAudio else {
            banner = Banner(severity: .error, text: "Enable playback or recording (Record to file) before starting.")
            return
        }
        var draft = template.defaultDraft()
        draft.prompt = prompt
        draft.model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        draft.durationSeconds = duration
        draft.musicPlay = playsAudio
        draft.outputPath = capturesAudio ? outputPath : ""
        draft.musicInteractive = true
        draft.musicStyleConditioning = style
        draft.musicTemperature = temperature
        draft.musicTopK = topK
        draft.musicCFGMusicCoCa = guidance
        draft.musicCFGNotes = noteGuidance
        draft.musicCFGDrums = drumGuidance
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
        peaks = []
        banner = nil
        if !controller.run(studio: request) {
            banner = Banner(severity: .error, text: "The session could not be started. Check the CLI path in Settings.")
        }
    }

    private func cancelSession() {
        guard let requestID else { return }
        if !controller.cancel(requestID: requestID) {
            banner = Banner(severity: .warning, text: "There is no running session to cancel.")
        }
    }

    private func sendPrompt() {
        guard isLive else { return }
        let prompt = blendedPrompt
        guard !prompt.isEmpty else { return }
        send("prompt \(prompt)")
    }

    private func applyControls() {
        for command in StudioRealtimeSteering.controlCommands(
            temperature: temperature,
            topK: topK,
            guidance: guidance,
            noteGuidance: noteGuidance,
            drumGuidance: drumGuidance,
            style: style,
            drumless: drumless
        ) {
            send(command)
        }
    }

    private func send(_ command: String) {
        guard let requestID, isLive else { return }
        if !controller.submitRealtimeMusicCommand(command, requestID: requestID) {
            banner = Banner(severity: .warning, text: "The realtime process is not ready for live control.")
        }
    }

    private func playNote(_ note: Int) {
        send("noteon \(note)")
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(280))
            send("noteoff \(note)")
        }
    }

    private static func noteName(_ midi: Int) -> String {
        let names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
        return "\(names[midi % 12])\(midi / 12 - 1)"
    }

    private func copyLog() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(StudioRealtimeSessionLog.copyText(logLines), forType: .string)
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

    /// Live audio is played by the CLI, not the app, so the waveform reads the recording the CLI
    /// is writing (a growing WAV) while the session runs, and the finished file afterwards.
    private func pollPeaks() async {
        while !Task.isCancelled {
            refreshPeaks()
            guard isLive else { return }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    private func refreshPeaks() {
        guard requestID != nil else {
            peaks = []
            return
        }
        let url: URL?
        if case .ended = phase {
            url = item?.outputURL ?? recordingURL
        } else {
            url = recordingURL
        }
        guard let url else { return }
        let loaded = isLive
            ? StudioWaveformLoader.growingWAVPeaks(url: url)
            : (StudioWaveformLoader.peaks(url: url) ?? StudioWaveformLoader.growingWAVPeaks(url: url))
        if let loaded {
            peaks = loaded
        }
    }
}

// MARK: - Pieces

private extension View {
    /// `panel()`: surface fill, hairline border, radius 10.
    func sessionPanel() -> some View {
        frame(maxWidth: .infinity)
            .background {
                RoundedRectangle(cornerRadius: 10)
                    .fill(MereRunTheme.surface)
                    .overlay {
                        RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                    }
            }
    }
}

private struct SessionHairline: View {
    var body: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.4))
            .frame(height: 1)
    }
}

/// `btnSecondary`: 26pt, raised fill, hairline border, 11.5pt medium.
private struct SessionSecondaryButton: View {
    let label: String
    let action: () -> Void
    @Environment(\.isEnabled) private var isEnabled

    init(_ label: String, action: @escaping () -> Void) {
        self.label = label
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(isEnabled ? MereRunTheme.textPrimary : MereRunTheme.textMuted)
                .lineLimit(1)
                .padding(.horizontal, 10)
                .frame(height: 26)
                .background {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(MereRunTheme.surfaceRaised)
                        .overlay {
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                        }
                }
                .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }
}

/// The LIVE pill: tinted fill, dot, 10.5pt bold tracked caps.
private struct SessionBadge: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text)
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.63)
        }
        .foregroundStyle(color)
        .padding(.horizontal, 7)
        .frame(height: 18)
        .background(Capsule().fill(color.opacity(0.13)))
        .accessibilityElement(children: .combine)
    }
}

/// Stroke-only stop square / play triangle on the 24-unit icon grid the mockup uses.
private struct TransportGlyph: Shape {
    let stop: Bool

    func path(in rect: CGRect) -> Path {
        let unit = rect.width / 24
        var path = Path()
        if stop {
            path.addRoundedRect(
                in: CGRect(x: rect.minX + 6 * unit, y: rect.minY + 6 * unit, width: 12 * unit, height: 12 * unit),
                cornerSize: CGSize(width: 2 * unit, height: 2 * unit)
            )
        } else {
            path.move(to: CGPoint(x: rect.minX + 7 * unit, y: rect.minY + 5 * unit))
            path.addLine(to: CGPoint(x: rect.minX + 19 * unit, y: rect.minY + 12 * unit))
            path.addLine(to: CGPoint(x: rect.minX + 7 * unit, y: rect.minY + 19 * unit))
            path.closeSubpath()
        }
        return path
    }
}

/// The session waveform: 4pt bars on a 7pt pitch, radius 2, played bars in accent and upcoming
/// bars in the border color. Peaks are resampled to however many bars fit the width.
private struct SessionWaveform: View {
    let peaks: [Float]
    let played: Double

    var body: some View {
        Canvas { context, size in
            let pitch: CGFloat = 7
            let barWidth: CGFloat = 4
            let count = max(1, Int((size.width + (pitch - barWidth)) / pitch))
            let playedBars = Int((played * Double(count)).rounded())
            for index in 0..<count {
                let amplitude: CGFloat
                if peaks.isEmpty {
                    amplitude = 0
                } else {
                    let source = min(peaks.count - 1, index * peaks.count / count)
                    amplitude = CGFloat(peaks[source])
                }
                let height = min(size.height, 8 + amplitude * (size.height - 12))
                let rect = CGRect(
                    x: CGFloat(index) * pitch,
                    y: (size.height - height) / 2,
                    width: barWidth,
                    height: height
                )
                let color = index < playedBars ? MereRunTheme.accent : MereRunTheme.border
                context.fill(Path(roundedRect: rect, cornerRadius: 2), with: .color(color))
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Session waveform")
        .accessibilityValue("\(Int((played * 100).rounded())) percent played")
    }
}

/// `slider()`: 4pt track, accent fill, 14pt surface thumb with hairline and a small shadow, mono
/// value on the right. Commits when the drag ends so a steer is one stdin line, not hundreds.
private struct SessionSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let display: (Double) -> String
    let onCommit: () -> Void

    private var fraction: CGFloat {
        let span = range.upperBound - range.lowerBound
        guard span > 0 else { return 0 }
        return CGFloat(min(1, max(0, (value - range.lowerBound) / span)))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer()
                Text(display(value))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
            }
            GeometryReader { proxy in
                let width = proxy.size.width
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(MereRunTheme.surfaceRaised)
                        .frame(height: 4)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(MereRunTheme.accent)
                        .frame(width: max(0, width * fraction), height: 4)
                    Circle()
                        .fill(MereRunTheme.surface)
                        .overlay { Circle().strokeBorder(MereRunTheme.border, lineWidth: 1) }
                        .mereShadow(radius: 1, y: 1)
                        .frame(width: 14, height: 14)
                        .offset(x: width * fraction - 7)
                }
                .frame(height: 16)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { drag in set(fractionAt: drag.location.x, width: width) }
                        .onEnded { drag in
                            set(fractionAt: drag.location.x, width: width)
                            onCommit()
                        }
                )
            }
            .frame(height: 16)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
        .accessibilityValue(display(value))
        .accessibilityAdjustableAction { direction in
            let step = (range.upperBound - range.lowerBound) / 20
            switch direction {
            case .increment: value = min(range.upperBound, value + step)
            case .decrement: value = max(range.lowerBound, value - step)
            @unknown default: break
            }
            onCommit()
        }
    }

    private func set(fractionAt x: CGFloat, width: CGFloat) {
        guard width > 0 else { return }
        let clamped = min(1, max(0, x / width))
        value = range.lowerBound + Double(clamped) * (range.upperBound - range.lowerBound)
    }
}
