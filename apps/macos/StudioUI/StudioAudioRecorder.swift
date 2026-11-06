import AVFoundation
import AppKit
import Combine
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
protocol StudioMicrophoneRecording: AnyObject {
    var currentTime: TimeInterval { get }
    func stop()
}

extension AVAudioRecorder: StudioMicrophoneRecording {}

/// Records the microphone to a 48 kHz mono WAV, filed in the domain's folder the way a run's
/// output is, so a reference voice or a conversation recorded here sits beside what it produces.
@MainActor
final class StudioAudioRecorder: @preconcurrency ObservableObject {
    private enum Phase {
        case idle
        case requestingPermission(UUID)
        case recording(any StudioMicrophoneRecording)
    }

    private struct State {
        var phase: Phase = .idle
        var duration: TimeInterval = 0
        var lastRecordingURL: URL?
        var errorMessage: String?
    }

    let objectWillChange = ObservableObjectPublisher()
    private var state = State()
    private let requestPermission: @MainActor () async -> Bool
    private let makeRecording: @MainActor (URL) throws -> any StudioMicrophoneRecording

    var isRecording: Bool {
        if case .recording = state.phase { return true }
        return false
    }

    var isStarting: Bool {
        if case .requestingPermission = state.phase { return true }
        return false
    }

    var duration: TimeInterval { state.duration }
    var lastRecordingURL: URL? { state.lastRecordingURL }
    var errorMessage: String? { state.errorMessage }

    convenience init() {
        self.init(requestPermission: { await AVCaptureDevice.requestAccess(for: .audio) }, makeRecording: { url in
            guard let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48_000, channels: 1, interleaved: true) else {
                throw CocoaError(.formatting)
            }
            let recorder = try AVAudioRecorder(url: url, format: format)
            recorder.prepareToRecord()
            guard recorder.record() else { throw CocoaError(.fileWriteUnknown) }
            return recorder
        })
    }

    init(
        requestPermission: @escaping @MainActor () async -> Bool,
        makeRecording: @escaping @MainActor (URL) throws -> any StudioMicrophoneRecording
    ) {
        self.requestPermission = requestPermission
        self.makeRecording = makeRecording
    }

    func cancelPendingStart() {
        guard isStarting else { return }
        state.phase = .idle
        objectWillChange.send()
    }

    func start(domain: StudioDomain, privateDirectory: URL? = nil) async {
        guard case .idle = state.phase, !Task.isCancelled else { return }
        let ticket = UUID()
        state = State(phase: .requestingPermission(ticket))
        objectWillChange.send()
        defer {
            if isRequestingPermission(ticket) {
                state.phase = .idle
                objectWillChange.send()
            }
        }
        guard isRequestingPermission(ticket), !Task.isCancelled else { return }
        let granted = await requestPermission()
        guard isRequestingPermission(ticket), !Task.isCancelled else { return }
        guard granted else {
            state.phase = .idle
            state.errorMessage = "Microphone access was denied. Enable it in System Settings."
            objectWillChange.send()
            return
        }

        let proposed = privateDirectory?.appendingPathComponent("Recordings", isDirectory: true)
            .appendingPathComponent("\(UUID()).wav") ?? StudioOutputLocation.specialistFile(
            domain: domain,
            name: "recording",
            fileExtension: "wav",
            now: StudioDisplayClock.now
        )
        do {
            let url: URL
            if privateDirectory != nil {
                try FileManager.default.createDirectory(at: proposed.deletingLastPathComponent(), withIntermediateDirectories: true)
                url = proposed
            } else {
                url = Self.recordingURL(proposed)
            }
            let recorder = try makeRecording(url)
            state = State(phase: .recording(recorder), lastRecordingURL: url)
        } catch {
            state.phase = .idle
            state.errorMessage = error.localizedDescription
        }
        objectWillChange.send()
    }

    private func isRequestingPermission(_ ticket: UUID) -> Bool {
        guard case .requestingPermission(let current) = state.phase else { return false }
        return current == ticket
    }

    /// The proposed path once its folder exists — or, when that folder cannot be made (an
    /// unplugged drive chosen in Settings), the same file name under App Outputs, so a
    /// recording never fails for want of somewhere to go.
    private static func recordingURL(_ proposed: URL) -> URL {
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(at: proposed.deletingLastPathComponent(), withIntermediateDirectories: true)
            return proposed
        } catch {
            let fallback = StudioOutputLocation.appOutputsRoot(fileManager: fileManager)
            try? fileManager.createDirectory(at: fallback, withIntermediateDirectories: true)
            return fallback.appendingPathComponent(proposed.lastPathComponent, isDirectory: false)
        }
    }

    /// Stops and returns the finished file.
    @discardableResult
    func stop() -> URL? {
        stopRecording()
        let url = state.lastRecordingURL
        objectWillChange.send()
        return url
    }

    private func stopRecording() {
        if case .recording(let recorder) = state.phase {
            state.duration = recorder.currentTime
            recorder.stop()
        }
        state.phase = .idle
    }

    /// Stops and removes the file: the recording was not wanted.
    func discard() {
        stopRecording()
        let url = state.lastRecordingURL
        if let url { try? FileManager.default.removeItem(at: url) }
        state = State()
        objectWillChange.send()
    }

    func refresh() {
        guard case .recording(let recorder) = state.phase else { return }
        state.duration = recorder.currentTime
        objectWillChange.send()
    }
}

/// The popover an audio slot's "Record…" opens: one big button that starts, then stops, the
/// recording, with the clock beside it. Stop hands the file to the slot and closes; Cancel
/// discards it.
struct StudioAudioRecordingPopover: View {
    @Environment(\.studioTaskSessions) private var sessions
    let domain: StudioDomain
    let onRecorded: (URL) -> Void

    @Environment(\.dismiss) private var dismiss
    @StateObject private var recorder: StudioAudioRecorder
    @State private var startTask: Task<Void, Never>?
    private let ticker = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

    init(domain: StudioDomain, recorder: StudioAudioRecorder = StudioAudioRecorder(), onRecorded: @escaping (URL) -> Void) {
        self.domain = domain
        self.onRecorded = onRecorded
        _recorder = StateObject(wrappedValue: recorder)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(recorder.isRecording ? "Recording" : "Record audio")
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textPrimary)
            HStack(spacing: 14) {
                Button {
                    if recorder.isRecording {
                        if let url = recorder.stop() {
                            onRecorded(url)
                        }
                        dismiss()
                    } else {
                        let directory = sessions?.privateOutputDirectory
                        startTask = Task { await recorder.start(domain: domain, privateDirectory: directory) }
                    }
                } label: {
                    ZStack {
                        Circle().fill(MereRunTheme.red)
                        if recorder.isRecording {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Color.white)
                                .frame(width: 16, height: 16)
                        } else {
                            Circle().fill(Color.white).frame(width: 18, height: 18)
                        }
                    }
                    .frame(width: 44, height: 44)
                }
                .buttonStyle(.plain)
                .disabled(recorder.isStarting)
                .help(recorder.isRecording ? "Stop and use the recording" : "Start recording")
                .accessibilityLabel(recorder.isRecording ? "Stop recording" : "Start recording")

                VStack(alignment: .leading, spacing: 3) {
                    Text(StudioTimeFormat.string(recorder.duration))
                        .font(.system(size: 22, weight: .medium, design: .monospaced))
                        .foregroundStyle(recorder.isRecording ? MereRunTheme.red : MereRunTheme.textPrimary)
                        .accessibilityLabel("Recorded \(StudioTimeFormat.string(recorder.duration))")
                    Text(recordingHint)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            if let error = recorder.errorMessage {
                Text(error)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Cancel") {
                    startTask?.cancel()
                    recorder.discard()
                    dismiss()
                }
                .buttonStyle(.mereSecondary)
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding(MereRunTheme.Spacing.md)
        .frame(width: 320)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onReceive(ticker) { _ in
            if recorder.isRecording { recorder.refresh() }
        }
        .onDisappear {
            startTask?.cancel()
            startTask = nil
            recorder.cancelPendingStart()
            if recorder.isRecording { recorder.discard() }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Record audio")
    }

    private var recordingHint: String {
        if recorder.isStarting { return "Waiting for microphone permission…" }
        return recorder.isRecording ? "Stop to use it." : "48 kHz mono WAV, saved with your \(domain.title) work."
    }
}

/// A "Record…" button that opens the recording popover and attaches the result to `slot`.
struct StudioAudioRecordButton<Draft: StudioAttachmentDraft>: View {
    let slot: StudioAttachmentSlot
    @Binding var draft: Draft
    let domain: StudioDomain

    @State private var isPresented = false

    var body: some View {
        Button("Record…") { isPresented = true }
            .buttonStyle(.mereSecondary)
            .help("Record \(slot.label.lowercased()) with the microphone")
            .popover(isPresented: $isPresented, arrowEdge: .bottom) {
                StudioAudioRecordingPopover(domain: domain) { url in
                    slot.attach([url], to: &draft)
                }
            }
    }
}

extension StudioAttachmentSlot {
    /// Whether the slot takes audio, and so offers "Record…".
    var canRecord: Bool {
        acceptedTypes.contains { UTType.audio.conforms(to: $0) }
    }
}
