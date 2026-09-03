import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

enum StudioVoiceTask: String, CaseIterable, Identifiable {
    case synthesize
    case transcribe
    case listen
    case diarize
    case profiles

    var id: String { rawValue }

    var title: String {
        switch self {
        case .synthesize: "Create"
        case .transcribe: "Transcribe"
        case .listen: "Listen Live"
        case .diarize: "Who Spoke"
        case .profiles: "Voices"
        }
    }

    var symbol: String {
        switch self {
        case .synthesize: "waveform.badge.plus"
        case .transcribe: "text.bubble"
        case .listen: "waveform.badge.mic"
        case .diarize: "person.2.wave.2"
        case .profiles: "person.wave.2"
        }
    }
}

struct StudioListenDevice: Equatable, Identifiable {
    let uid: String
    let name: String
    let isDefault: Bool

    var id: String { uid }

    static func parseList(_ output: String) -> [StudioListenDevice] {
        output.split(separator: "\n").compactMap { rawLine in
            let line = String(rawLine)
            guard let marker = line.first else { return nil }
            let fields = line.dropFirst()
                .trimmingCharacters(in: .whitespaces)
                .split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard fields.count == 2 else { return nil }
            let uid = String(fields[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let name = String(fields[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !uid.isEmpty, !name.isEmpty else { return nil }
            return StudioListenDevice(uid: uid, name: name, isDefault: marker == "*")
        }
    }
}

struct StudioLiveTranscriptAccumulator: Equatable {
    private struct Event: Decodable {
        let protocolVersion: Int
        let type: String
        let utteranceID: String?
        let revision: Int?
        let text: String?
        let message: String?

        enum CodingKeys: String, CodingKey {
            case protocolVersion = "protocol"
            case type
            case utteranceID = "utteranceId"
            case legacyUtteranceID = "utterance_id"
            case revision
            case text
            case message
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            protocolVersion = try values.decode(Int.self, forKey: .protocolVersion)
            type = try values.decode(String.self, forKey: .type)
            utteranceID = try values.decodeIfPresent(String.self, forKey: .utteranceID)
                ?? values.decodeIfPresent(String.self, forKey: .legacyUtteranceID)
            revision = try values.decodeIfPresent(Int.self, forKey: .revision)
            text = try values.decodeIfPresent(String.self, forKey: .text)
            message = try values.decodeIfPresent(String.self, forKey: .message)
        }
    }

    private var buffer = ""
    private var committedUtteranceIDs: Set<String> = []
    private var latestRevisions: [String: Int] = [:]
    private var committedSegments: [String] = []
    private(set) var partialText = ""
    private(set) var errorMessage: String?

    var committedText: String { committedSegments.joined(separator: "\n") }

    var displayText: String {
        [committedText, partialText]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    mutating func beginSession() {
        buffer = ""
        partialText = ""
        errorMessage = nil
        latestRevisions.removeAll(keepingCapacity: true)
    }

    mutating func clear() {
        self = StudioLiveTranscriptAccumulator()
    }

    mutating func receive(_ chunk: String) {
        buffer += chunk
        while let newline = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<newline])
            buffer.removeSubrange(...newline)
            receiveLine(line)
        }
    }

    private mutating func receiveLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let event = try? JSONDecoder().decode(Event.self, from: data),
              event.protocolVersion == 1 else { return }

        switch event.type {
        case "partial":
            guard let id = event.utteranceID,
                  !committedUtteranceIDs.contains(id),
                  let text = event.text else { return }
            let revision = event.revision ?? 0
            guard revision >= latestRevisions[id, default: -1] else { return }
            latestRevisions[id] = revision
            partialText = text
        case "commit":
            guard let id = event.utteranceID,
                  !committedUtteranceIDs.contains(id),
                  let text = event.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            committedUtteranceIDs.insert(id)
            committedSegments.append(text)
            partialText = ""
        case "error":
            errorMessage = event.message ?? "Live transcription failed."
        default:
            break
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

struct StudioVoiceView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    @StateObject private var recorder = StudioVoiceRecorder()
    /// Owned by the shell's task control. Voice hosts Clone and Voices; Audio hosts Who Spoke
    /// and Live. The rail mirrors the tasks of its host.
    @Binding var task: StudioVoiceTask
    let tasks: [StudioVoiceTask]
    @State private var synthesisDraft: CommandDraft
    @State private var transcriptionDraft: CommandDraft
    @State private var diarizationDraft: CommandDraft
    @State private var profileDraft: CommandDraft
    @State private var profiles: [StudioVoiceProfileRecord] = []
    @State private var selectedProfileID: UUID?
    @State private var requestID: UUID?
    @State private var transcriptText = ""
    @State private var transcriptURL: URL?
    @State private var statusMessage: String?
    @State private var comparisonA: UUID?
    @State private var comparisonB: UUID?
    @State private var listenDraft: CommandDraft
    @State private var listenTranscript = StudioLiveTranscriptAccumulator()
    @State private var listenDevices: [StudioListenDevice] = []
    @State private var listenCommandID: UUID?
    @State private var isListening = false

    private let recorderTicker = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

    init(task: Binding<StudioVoiceTask>, tasks: [StudioVoiceTask], initialDraft: StudioDraft) {
        _task = task
        self.tasks = tasks

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

        _listenDraft = State(
            initialValue: CommandCatalog.template(id: .speechListen)?.defaultDraft() ?? CommandDraft()
        )

        var diarization = CommandCatalog.template(id: .speechDiarize)?.defaultDraft() ?? CommandDraft()
        diarization.inputPath = initialDraft.inputPath
        diarization.outputPath = Self.timestampedOutput(prefix: "speakers", extension: "json")
        _diarizationDraft = State(initialValue: diarization)

        let profile = CommandCatalog.template(id: .speechProfileCreate)?.defaultDraft() ?? CommandDraft()
        _profileDraft = State(initialValue: profile)
    }

    var body: some View {
        HStack(spacing: 0) {
            taskRail
                .frame(width: 170)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            configuration
                .frame(minWidth: 300, idealWidth: 420, maxWidth: 420)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            resultPane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
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
                .speechDiarize,
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
            diarizationDraft.inputPath = url.path
        }
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
            if recorder.isRecording {
                Label(
                    StudioTimeFormat.string(recorder.duration),
                    systemImage: "record.circle.fill"
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.red)
            }
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
                case .listen:
                    listenControls
                case .diarize:
                    diarizationControls
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

    private var diarizationControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            sectionTitle("Conversation")
            StudioPathField(
                label: "Audio",
                placeholder: "Choose or record a meeting, interview, or conversation",
                path: $diarizationDraft.inputPath,
                allowedContentTypes: [.audio]
            )
            if let url = recorder.lastRecordingURL {
                Button("Use latest recording") {
                    diarizationDraft.inputPath = url.path
                }
                .buttonStyle(.bordered)
            }

            sectionTitle("Speaker detection")
            labeledTextField(
                "Model",
                placeholder: "speech-diarization-sortformer",
                text: $diarizationDraft.model
            )
            Picker(
                "Output",
                selection: Binding(
                    get: { diarizationDraft.speechDiarizationFormat ?? "json" },
                    set: { format in
                        diarizationDraft.speechDiarizationFormat = format
                        diarizationDraft.outputPath = Self.replacingExtension(
                            diarizationDraft.outputPath,
                            with: format
                        )
                    }
                )
            ) {
                Text("JSON timeline").tag("json")
                Text("RTTM").tag("rttm")
            }
            .pickerStyle(.segmented)

            VStack(alignment: .leading, spacing: 5) {
                Text(
                    "Activity threshold \(diarizationDraft.speechDiarizationThreshold ?? 0.5, specifier: "%.2f")"
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                Slider(
                    value: Binding(
                        get: { diarizationDraft.speechDiarizationThreshold ?? 0.5 },
                        set: { diarizationDraft.speechDiarizationThreshold = $0 }
                    ),
                    in: 0...1,
                    step: 0.01
                )
            }
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Minimum segment")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField(
                        "0.25",
                        value: Binding(
                            get: { diarizationDraft.speechDiarizationMinDuration ?? 0.25 },
                            set: { diarizationDraft.speechDiarizationMinDuration = max(0, $0) }
                        ),
                        format: .number.precision(.fractionLength(0...2))
                    )
                    .mereField()
                }
                VStack(alignment: .leading, spacing: 5) {
                    Text("Merge gap")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField(
                        "0.25",
                        value: Binding(
                            get: { diarizationDraft.speechDiarizationMergeGap ?? 0.25 },
                            set: { diarizationDraft.speechDiarizationMergeGap = max(0, $0) }
                        ),
                        format: .number.precision(.fractionLength(0...2))
                    )
                    .mereField()
                }
            }
            Text("Durations are measured in seconds. Adjacent segments from the same speaker can be merged.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            StudioPathField(
                label: "Speaker timeline",
                placeholder: "JSON or RTTM output",
                path: $diarizationDraft.outputPath
            )
            runButton("Identify speakers", symbol: "person.2.wave.2", action: runDiarization)
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
        case .listen:
            listenResults
        case .diarize:
            diarizationResults
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

    private var diarizationResults: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Speaker timeline")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Text("Sortformer · local on this Mac")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            if !diarizationDraft.inputPath.isBlank {
                StudioAudioPlayerView(url: URL(fileURLWithPath: diarizationDraft.inputPath))
                    .frame(height: 150)
                    .merePanel()
            }
            StudioSpecialistResultView(requestID: requestID, preferredKinds: [.text, .audio])
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

    private func runDiarization() {
        guard FileManager.default.fileExists(atPath: diarizationDraft.inputPath) else {
            statusMessage = "Choose a conversation recording first."
            return
        }
        guard !diarizationDraft.outputPath.isBlank else {
            statusMessage = "Choose a speaker timeline output file."
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: .speechDiarize,
            mode: .listen,
            draft: diarizationDraft,
            controller: controller,
            library: library
        )
        statusMessage = "Speaker identification submitted."
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

    // MARK: - Live transcription

    /// `speech listen --jsonl` streams versioned events until the process is interrupted,
    /// so this lane owns the child process directly instead of going through the run queue.
    private var listenControls: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            sectionTitle("Microphone")
            HStack(spacing: 8) {
                Picker("Input device", selection: $listenDraft.speechListenDevice) {
                    Text("System default").tag("")
                    ForEach(listenDevices) { device in
                        Text(device.isDefault ? "\(device.name) — default" : device.name)
                            .tag(device.uid)
                    }
                }
                .disabled(isListening)
                Button("Refresh") {
                    Task { await refreshListenDevices() }
                }
                .buttonStyle(.mereSecondary)
                .disabled(isListening)
                .help("List the microphones the CLI can capture from")
            }

            sectionTitle("Recognition")
            labeledTextField(
                "Language",
                placeholder: "auto",
                text: $listenDraft.language
            )
            labeledTextField(
                "Model",
                placeholder: "Managed ASR model id (optional)",
                text: $listenDraft.model
            )
            HStack {
                Stepper(
                    "Decode window \(listenDraft.speechListenDecodeMS) ms",
                    value: $listenDraft.speechListenDecodeMS,
                    in: 0...10_000,
                    step: 250
                )
                Stepper(
                    "Silence \(listenDraft.speechListenSilenceMS) ms",
                    value: $listenDraft.speechListenSilenceMS,
                    in: 0...5_000,
                    step: 100
                )
            }
            .disabled(isListening)
            Text("Leave a window at zero to use the runtime default.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)

            Button {
                if isListening {
                    stopListening()
                } else {
                    startListening()
                }
            } label: {
                Label(
                    isListening ? "Stop listening" : "Start listening",
                    systemImage: isListening ? "stop.fill" : "mic.fill"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(isListening ? MereRunTheme.red : MereRunTheme.accent)
        }
        .task {
            if listenDevices.isEmpty { await refreshListenDevices() }
        }
    }

    private var listenResults: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Live transcript")
                    .font(MereRunTheme.sectionFont)
                if isListening {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityHidden(true)
                    Text("Listening")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.green)
                        .accessibilityLabel("Currently listening to the microphone")
                }
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(listenTranscript.committedText, forType: .string)
                }
                .buttonStyle(.mereSecondary)
                .disabled(listenTranscript.committedText.isEmpty)
                .accessibilityLabel("Copy the live transcript")
                Button("Save…") { saveLiveTranscript() }
                    .buttonStyle(.mereSecondary)
                    .disabled(listenTranscript.committedText.isEmpty)
                Button("Clear") { listenTranscript.clear() }
                    .buttonStyle(.mereSecondary)
                    .disabled(listenTranscript.displayText.isEmpty || isListening)
            }

            if listenTranscript.displayText.isEmpty {
                ContentUnavailableView(
                    isListening ? "Waiting for speech" : "Not listening",
                    systemImage: "waveform.badge.mic",
                    description: Text(
                        isListening
                            ? "Partial transcripts appear here as the recognizer emits them."
                            : "Start listening to stream microphone audio through live Qwen ASR."
                    )
                )
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        if !listenTranscript.committedText.isEmpty {
                            Text(listenTranscript.committedText)
                        }
                        if !listenTranscript.partialText.isEmpty {
                            Text(listenTranscript.partialText)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .italic()
                                .accessibilityLabel("Partial transcript: \(listenTranscript.partialText)")
                        }
                    }
                    .font(MereRunTheme.bodyFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                }
                .merePanel()
            }
        }
        .padding(18)
    }

    private func refreshListenDevices() async {
        let result = await controller.utilityCommandResult(args: ["speech", "listen", "--list-devices"])
        guard result.exitCode == 0 else {
            statusMessage = "Could not list microphones."
            return
        }
        listenDevices = StudioListenDevice.parseList(result.stdout)
        if !listenDraft.speechListenDevice.isEmpty,
           !listenDevices.contains(where: { $0.uid == listenDraft.speechListenDevice }) {
            listenDraft.speechListenDevice = ""
        }
    }

    private func startListening() {
        guard let template = CommandCatalog.template(id: .speechListen) else {
            statusMessage = "Live transcription is unavailable."
            return
        }
        let commandID = UUID()
        listenCommandID = commandID
        isListening = true
        statusMessage = "Listening. Speak into the selected microphone."
        listenTranscript.beginSession()
        var draft = listenDraft
        draft.speechJSONL = true
        draft.quiet = true
        let args = template.arguments(from: draft)
        Task {
            let result = await controller.utilityCommandResult(
                args: args,
                commandID: commandID,
                onStandardOutput: { chunk in
                    listenTranscript.receive(chunk)
                }
            )
            isListening = false
            listenCommandID = nil
            if let errorMessage = listenTranscript.errorMessage {
                statusMessage = errorMessage
            } else if result.exitCode != 0, !listenTranscript.committedText.isEmpty {
                statusMessage = "Listening stopped."
            } else if result.exitCode != 0 {
                statusMessage = "Live transcription exited with code \(result.exitCode)."
            } else {
                statusMessage = "Listening stopped."
            }
        }
    }

    private func stopListening() {
        guard let commandID = listenCommandID else {
            isListening = false
            return
        }
        _ = controller.interruptUtilityCommand(commandID)
        statusMessage = "Stopping…"
    }

    private func saveLiveTranscript() {
        let suggested = URL(fileURLWithPath: Self.timestampedOutput(prefix: "live-transcript", extension: "txt"))
        guard let url = StudioSpecialistFiles.saveFile(
            title: "Save live transcript",
            suggestedName: suggested.lastPathComponent,
            allowedContentTypes: [.plainText]
        ) else { return }
        do {
            try listenTranscript.committedText.write(to: url, atomically: true, encoding: .utf8)
            statusMessage = "Saved the live transcript."
        } catch {
            statusMessage = "Could not save the transcript: \(error.localizedDescription)"
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
