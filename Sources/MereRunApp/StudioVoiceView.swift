import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

enum StudioVoiceTask: String, CaseIterable, Identifiable {
    case synthesize
    case transcribe
    case profiles

    var id: String { rawValue }

    var title: String {
        switch self {
        case .synthesize: "Create"
        case .transcribe: "Transcribe"
        case .profiles: "Voices"
        }
    }

    var symbol: String {
        switch self {
        case .synthesize: "waveform.badge.plus"
        case .transcribe: "text.bubble"
        case .profiles: "person.wave.2"
        }
    }
}

struct StudioVoiceProfileRecord: Codable, Identifiable, Equatable {
    let id: UUID
    let name: String
    let createdAt: Date
    let updatedAt: Date
    let transcript: String
    let language: String?
    let referenceAudioRelativePath: String
    let modelFingerprint: String?

    var referenceAudioURL: URL {
        StudioVoiceProfileStore.voicesDirectory
            .appendingPathComponent(referenceAudioRelativePath, isDirectory: false)
    }
}

enum StudioVoiceProfileStore {
    static var voicesDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("voices", isDirectory: true)
    }

    static var manifestURL: URL {
        voicesDirectory.appendingPathComponent("voice_profiles.json", isDirectory: false)
    }

    static func load() -> [StudioVoiceProfileRecord] {
        guard let data = try? Data(contentsOf: manifestURL),
              let profiles = try? JSONDecoder().decode([StudioVoiceProfileRecord].self, from: data) else {
            return []
        }
        return profiles.sorted {
            $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }
}

@MainActor
final class StudioVoiceRecorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var duration: Double = 0
    @Published private(set) var lastRecordingURL: URL?
    @Published private(set) var errorMessage: String?

    private var recorder: AVAudioRecorder?

    func start() async {
        errorMessage = nil
        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        guard granted else {
            errorMessage = "Microphone access was denied. Enable it in System Settings."
            return
        }

        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Music/MereRun/Recordings", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd-HHmmss"
            let url = directory.appendingPathComponent(
                "voice-reference-\(formatter.string(from: Date())).wav",
                isDirectory: false
            )
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 48_000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false
            ]
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.prepareToRecord()
            guard recorder.record() else {
                throw CocoaError(.fileWriteUnknown)
            }
            self.recorder = recorder
            lastRecordingURL = url
            duration = 0
            isRecording = true
        } catch {
            errorMessage = error.localizedDescription
            isRecording = false
        }
    }

    func stop() {
        recorder?.stop()
        refresh()
        recorder = nil
        isRecording = false
    }

    func refresh() {
        duration = recorder?.currentTime ?? duration
    }

}

struct StudioVoiceSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.dismiss) private var dismiss

    @StateObject private var recorder = StudioVoiceRecorder()
    @State private var task: StudioVoiceTask
    @State private var synthesisDraft: CommandDraft
    @State private var transcriptionDraft: CommandDraft
    @State private var profileDraft: CommandDraft
    @State private var profiles: [StudioVoiceProfileRecord] = []
    @State private var selectedProfileID: UUID?
    @State private var requestID: UUID?
    @State private var transcriptText = ""
    @State private var transcriptURL: URL?
    @State private var statusMessage: String?
    @State private var comparisonA: UUID?
    @State private var comparisonB: UUID?

    private let recorderTicker = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

    init(initialTask: StudioVoiceTask, initialDraft: StudioDraft) {
        _task = State(initialValue: initialTask)

        var synthesis = CommandCatalog.template(id: .speechSynthesize)?.defaultDraft() ?? CommandDraft()
        synthesis.prompt = initialDraft.prompt
        synthesis.secondaryText = initialDraft.secondaryText
        synthesis.model = initialDraft.model.localizedCaseInsensitiveContains("speech")
            ? initialDraft.model
            : synthesis.model
        synthesis.outputPath = Self.timestampedOutput(prefix: "voice", extension: "wav")
        _synthesisDraft = State(initialValue: synthesis)

        var transcription = CommandCatalog.template(id: .speechTranscribe)?.defaultDraft() ?? CommandDraft()
        transcription.inputPath = initialDraft.inputPath
        transcription.outputPath = Self.timestampedOutput(prefix: "transcript", extension: "txt")
        _transcriptionDraft = State(initialValue: transcription)

        let profile = CommandCatalog.template(id: .speechProfileCreate)?.defaultDraft() ?? CommandDraft()
        _profileDraft = State(initialValue: profile)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            HStack(spacing: 0) {
                taskRail
                    .frame(width: 170)
                Divider().overlay(MereRunTheme.border.opacity(0.6))
                configuration
                    .frame(width: 420)
                Divider().overlay(MereRunTheme.border.opacity(0.6))
                resultPane
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: 1_210, height: 780)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task { refreshProfiles() }
        .onReceive(recorderTicker) { _ in
            if recorder.isRecording {
                recorder.refresh()
            }
        }
        .onReceive(controller.runCompletions) { result in
            guard [
                CommandTemplateID.speechSynthesize,
                .speechTranscribe,
                .speechProfileCreate,
                .speechProfileDelete
            ].contains(result.templateID) else { return }
            if result.templateID == .speechTranscribe, result.exitCode == 0 {
                loadTranscriptFromCurrentRun()
            }
            if result.templateID == .speechProfileCreate || result.templateID == .speechProfileDelete {
                refreshProfiles()
            }
            refreshComparisons()
        }
        .onChange(of: recorder.lastRecordingURL) { _, url in
            guard let url else { return }
            profileDraft.inputPath = url.path
            synthesisDraft.refAudioPath = url.path
            transcriptionDraft.inputPath = url.path
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("Voice Studio")
                    .font(MereRunTheme.titleFont)
                Text("Record, clone, synthesize, compare, transcribe, and manage reusable voices")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            if recorder.isRecording {
                Label(
                    StudioTimeFormat.string(recorder.duration),
                    systemImage: "record.circle.fill"
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.red)
            }
            Button("Done") { dismiss() }
                .buttonStyle(.bordered)
        }
        .padding(18)
    }

    private var taskRail: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("WORKSPACE")
                .font(.system(size: 9.5, weight: .bold))
                .kerning(0.8)
                .foregroundStyle(MereRunTheme.textMuted)
                .padding(.horizontal, 12)
                .padding(.bottom, 3)
            ForEach(StudioVoiceTask.allCases) { item in
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
            Button {
                if recorder.isRecording {
                    recorder.stop()
                } else {
                    Task { await recorder.start() }
                }
            } label: {
                Label(
                    recorder.isRecording ? "Stop recording" : "Record reference",
                    systemImage: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(recorder.isRecording ? MereRunTheme.red : MereRunTheme.accent)
            if let error = recorder.errorMessage {
                Text(error)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                switch task {
                case .synthesize:
                    synthesisControls
                case .transcribe:
                    transcriptionControls
                case .profiles:
                    profileControls
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

    private var synthesisControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            sectionTitle("Script")
            TextEditor(text: $synthesisDraft.prompt)
                .font(MereRunTheme.bodyFont)
                .scrollContentBackground(.hidden)
                .padding(9)
                .frame(minHeight: 120)
                .merePanel()

            Picker("Voice source", selection: $synthesisDraft.voiceMode) {
                Text("Describe a style").tag("style")
                Text("Clone a voice").tag("clone")
            }
            .pickerStyle(.segmented)

            if synthesisDraft.voiceMode == "style" {
                labeledTextField(
                    "Voice direction",
                    placeholder: "Warm documentary narrator with measured pacing",
                    text: $synthesisDraft.secondaryText
                )
            } else {
                Picker("Saved voice", selection: $selectedProfileID) {
                    Text("Reference file only").tag(UUID?.none)
                    ForEach(profiles) { profile in
                        Text(profile.name).tag(Optional(profile.id))
                    }
                }
                .onChange(of: selectedProfileID) { _, id in
                    synthesisDraft.voiceProfile = id?.uuidString ?? ""
                }
                StudioPathField(
                    label: "Reference audio",
                    placeholder: "WAV, MP3, M4A…",
                    path: $synthesisDraft.refAudioPath,
                    allowedContentTypes: [.audio]
                )
                labeledTextField(
                    "Reference transcript",
                    placeholder: "Optional when using a saved profile",
                    text: $synthesisDraft.refText
                )
                labeledTextField(
                    "Save new profile as",
                    placeholder: "Optional profile name",
                    text: $synthesisDraft.saveProfileName
                )
            }

            Divider().overlay(MereRunTheme.border.opacity(0.5))
            sectionTitle("Generation")
            labeledTextField("Model", placeholder: "speech-tts-qwen3-nano", text: $synthesisDraft.model)
            HStack {
                labeledTextField("Language", placeholder: "auto", text: $synthesisDraft.language)
                VStack(alignment: .leading, spacing: 5) {
                    Text("Temperature \(synthesisDraft.temperature, specifier: "%.2f")")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Slider(value: $synthesisDraft.temperature, in: 0...1.5, step: 0.05)
                }
            }
            Toggle("Stream speech while generating", isOn: $synthesisDraft.stream)
            if synthesisDraft.stream {
                Stepper(
                    "Chunk tokens: \(synthesisDraft.speechStreamChunkTokens)",
                    value: $synthesisDraft.speechStreamChunkTokens,
                    in: 1...200
                )
            }
            StudioPathField(
                label: "Output WAV",
                placeholder: "Output WAV",
                path: $synthesisDraft.outputPath
            )
            runButton("Generate voice", symbol: "waveform.badge.plus", action: runSynthesis)
        }
    }

    private var transcriptionControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            sectionTitle("Source")
            StudioPathField(
                label: "Audio",
                placeholder: "Choose or record audio",
                path: $transcriptionDraft.inputPath,
                allowedContentTypes: [.audio]
            )
            if let url = recorder.lastRecordingURL {
                Button("Use latest recording") {
                    transcriptionDraft.inputPath = url.path
                }
                .buttonStyle(.bordered)
            }

            sectionTitle("Transcription")
            Picker("Backend", selection: $transcriptionDraft.backend) {
                Text("Automatic").tag("auto")
                Text("Parakeet").tag("parakeet")
                Text("Qwen").tag("qwen")
            }
            Picker("Task", selection: $transcriptionDraft.task) {
                Text("Transcribe").tag("transcribe")
                Text("Translate").tag("translate")
            }
            labeledTextField("Language", placeholder: "auto", text: $transcriptionDraft.language)
            Stepper(
                "Maximum tokens: \(transcriptionDraft.maxTokens)",
                value: $transcriptionDraft.maxTokens,
                in: 64...16_384,
                step: 64
            )
            Toggle("Streaming transcription", isOn: $transcriptionDraft.stream)
            if transcriptionDraft.stream {
                Stepper(
                    "Feed interval: \(transcriptionDraft.speechStreamChunkMS) ms",
                    value: $transcriptionDraft.speechStreamChunkMS,
                    in: 20...5_000,
                    step: 20
                )
                Stepper(
                    "Decode interval: \(transcriptionDraft.speechStreamDecodeMS) ms",
                    value: $transcriptionDraft.speechStreamDecodeMS,
                    in: 100...20_000,
                    step: 100
                )
            }
            Toggle("Include timestamps", isOn: $transcriptionDraft.timestamps)
            Toggle("Write JSONL events", isOn: $transcriptionDraft.speechJSONL)
                .onChange(of: transcriptionDraft.speechJSONL) { _, enabled in
                    transcriptionDraft.outputPath = Self.replacingExtension(
                        transcriptionDraft.outputPath,
                        with: enabled ? "jsonl" : "txt"
                    )
                }
            StudioPathField(
                label: "Transcript output",
                placeholder: "Transcript file",
                path: $transcriptionDraft.outputPath
            )
            runButton("Transcribe", symbol: "text.bubble", action: runTranscription)
        }
    }

    private var profileControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            sectionTitle("Create reusable voice")
            labeledTextField("Profile name", placeholder: "Narrator", text: $profileDraft.prompt)
            StudioPathField(
                label: "Reference audio",
                placeholder: "Choose or record a clean voice sample",
                path: $profileDraft.inputPath,
                allowedContentTypes: [.audio]
            )
            TextEditor(text: $profileDraft.secondaryText)
                .font(MereRunTheme.bodyFont)
                .scrollContentBackground(.hidden)
                .padding(9)
                .frame(minHeight: 90)
                .merePanel()
                .overlay(alignment: .topLeading) {
                    if profileDraft.secondaryText.isEmpty {
                        Text("Optional reference transcript—otherwise mere.run transcribes it")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .padding(14)
                            .allowsHitTesting(false)
                    }
                }
            labeledTextField("Language", placeholder: "auto", text: $profileDraft.language)
            runButton("Create profile", symbol: "person.badge.plus", action: createProfile)

            Divider().overlay(MereRunTheme.border.opacity(0.5))
            sectionTitle("Saved voices")
            if profiles.isEmpty {
                Text("No saved voices yet.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            } else {
                ForEach(profiles) { profile in
                    profileRow(profile)
                }
            }
        }
    }

    @ViewBuilder
    private var resultPane: some View {
        switch task {
        case .synthesize:
            synthesisResults
        case .transcribe:
            transcriptionResults
        case .profiles:
            profileDetail
        }
    }

    private var synthesisResults: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("A/B voice comparison")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Text("Recent renders stay in the Library")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }

            let runs = synthesisRuns
            if runs.isEmpty {
                StudioSpecialistResultView(requestID: requestID, preferredKinds: [.audio, .text])
            } else {
                HStack(spacing: 10) {
                    comparisonCard(label: "A", selectedID: $comparisonA, runs: runs)
                    comparisonCard(label: "B", selectedID: $comparisonB, runs: runs)
                }
            }
        }
        .padding(18)
    }

    private var transcriptionResults: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Transcript editor")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                if let transcriptURL {
                    Button("Save edits") { saveTranscript(to: transcriptURL) }
                        .buttonStyle(.borderedProminent)
                        .tint(MereRunTheme.accent)
                }
            }
            if !transcriptionDraft.inputPath.isBlank {
                StudioAudioPlayerView(url: URL(fileURLWithPath: transcriptionDraft.inputPath))
                    .frame(height: 150)
                    .merePanel()
            }
            if transcriptURL != nil || !transcriptText.isEmpty {
                TextEditor(text: $transcriptText)
                    .font(MereRunTheme.monoFont)
                    .scrollContentBackground(.hidden)
                    .padding(10)
                    .merePanel()
            } else {
                StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text, .audio])
            }
        }
        .padding(18)
    }

    private var profileDetail: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let profile = profiles.first(where: { $0.id == selectedProfileID }) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(profile.name)
                            .font(MereRunTheme.titleFont)
                        Text(profile.id.uuidString)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    Button(role: .destructive) {
                        deleteProfile(profile)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)
                }
                StudioAudioPlayerView(url: profile.referenceAudioURL)
                    .frame(height: 230)
                    .merePanel()
                VStack(alignment: .leading, spacing: 7) {
                    Text("Reference transcript")
                        .font(MereRunTheme.sectionFont)
                    Text(profile.transcript)
                        .font(MereRunTheme.bodyFont)
                        .textSelection(.enabled)
                    if let language = profile.language {
                        Label(language, systemImage: "globe")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
                .padding(14)
                .merePanel()
                Spacer()
            } else {
                ContentUnavailableView(
                    "Choose a saved voice",
                    systemImage: "person.wave.2",
                    description: Text("Preview its reference, inspect the transcript, or use it for cloning.")
                )
            }
        }
        .padding(18)
    }

    private func comparisonCard(
        label: String,
        selectedID: Binding<UUID?>,
        runs: [StudioLibraryItem]
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker(label, selection: selectedID) {
                ForEach(runs) { item in
                    Text(item.displayTitle).tag(Optional(item.id))
                }
            }
            if let id = selectedID.wrappedValue,
               let item = runs.first(where: { $0.id == id }),
               let url = item.allArtifactURLs.first(where: {
                   StudioOutputFileKind.classify($0) == .audio
               }) {
                StudioAudioPlayerView(url: url)
            } else {
                ContentUnavailableView("No render", systemImage: "waveform")
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .merePanel()
    }

    private func profileRow(_ profile: StudioVoiceProfileRecord) -> some View {
        Button {
            selectedProfileID = profile.id
        } label: {
            HStack {
                Image(systemName: "person.wave.2.fill")
                    .foregroundStyle(MereRunTheme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text(profile.name)
                        .font(.system(size: 12.5, weight: .semibold))
                    Text(profile.language ?? "Automatic language")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(9)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(selectedProfileID == profile.id ? MereRunTheme.accentSoft : MereRunTheme.surface)
            }
        }
        .buttonStyle(.plain)
    }

    private func runSynthesis() {
        guard !synthesisDraft.prompt.isBlank else {
            statusMessage = "Enter a script to synthesize."
            return
        }
        guard !synthesisDraft.outputPath.isBlank else {
            statusMessage = "Choose an output WAV."
            return
        }
        if synthesisDraft.voiceMode == "clone",
           synthesisDraft.voiceProfile.isBlank,
           synthesisDraft.refAudioPath.isBlank {
            statusMessage = "Choose a saved voice or reference audio for cloning."
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: .speechSynthesize,
            mode: .speak,
            draft: synthesisDraft,
            controller: controller,
            library: library
        )
        statusMessage = "Voice render submitted."
        synthesisDraft.outputPath = Self.timestampedOutput(prefix: "voice", extension: "wav")
        refreshComparisons()
    }

    private func runTranscription() {
        guard FileManager.default.fileExists(atPath: transcriptionDraft.inputPath) else {
            statusMessage = "Choose an audio file or make a recording first."
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: .speechTranscribe,
            mode: .listen,
            draft: transcriptionDraft,
            controller: controller,
            library: library
        )
        transcriptText = ""
        transcriptURL = nil
        statusMessage = "Transcription submitted."
    }

    private func createProfile() {
        guard !profileDraft.prompt.isBlank else {
            statusMessage = "Give the voice profile a name."
            return
        }
        guard FileManager.default.fileExists(atPath: profileDraft.inputPath) else {
            statusMessage = "Choose or record reference audio."
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: .speechProfileCreate,
            mode: .speak,
            draft: profileDraft,
            controller: controller,
            library: library
        )
        statusMessage = "Voice profile creation submitted."
    }

    private func deleteProfile(_ profile: StudioVoiceProfileRecord) {
        var draft = CommandCatalog.template(id: .speechProfileDelete)?.defaultDraft() ?? CommandDraft()
        draft.prompt = profile.id.uuidString
        requestID = StudioSpecialistRunner.submit(
            templateID: .speechProfileDelete,
            mode: .speak,
            draft: draft,
            controller: controller,
            library: library
        )
        selectedProfileID = nil
        statusMessage = "Deleting \(profile.name)…"
    }

    private func refreshProfiles() {
        profiles = StudioVoiceProfileStore.load()
        if selectedProfileID == nil {
            selectedProfileID = profiles.first?.id
        }
        if let id = selectedProfileID {
            synthesisDraft.voiceProfile = id.uuidString
        }
    }

    private var synthesisRuns: [StudioLibraryItem] {
        Array(
            library.items
                .filter {
                    $0.templateID == .speechSynthesize
                        && $0.allArtifactURLs.contains(where: {
                            StudioOutputFileKind.classify($0) == .audio
                        })
                }
                .prefix(8)
        )
    }

    private func refreshComparisons() {
        let runs = synthesisRuns
        if comparisonA == nil { comparisonA = runs.first?.id }
        if comparisonB == nil { comparisonB = runs.dropFirst().first?.id ?? runs.first?.id }
    }

    private func loadTranscriptFromCurrentRun() {
        guard let requestID,
              let item = library.items.first(where: { $0.id == requestID }) else { return }
        let url = item.allArtifactURLs.first(where: {
            ["txt", "jsonl", "json"].contains($0.pathExtension.lowercased())
        }) ?? (transcriptionDraft.outputPath.isBlank
            ? nil
            : URL(fileURLWithPath: transcriptionDraft.outputPath))
        guard let url, let text = try? String(contentsOf: url, encoding: .utf8) else { return }
        transcriptURL = url
        transcriptText = text
    }

    private func saveTranscript(to url: URL) {
        do {
            try transcriptText.write(to: url, atomically: true, encoding: .utf8)
            statusMessage = "Saved \(url.lastPathComponent)."
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(MereRunTheme.sectionFont)
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

    private func runButton(
        _ title: String,
        symbol: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: symbol)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(MereRunTheme.accent)
    }

    nonisolated private static func timestampedOutput(prefix: String, extension pathExtension: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Music/MereRun/Voice", isDirectory: true)
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
