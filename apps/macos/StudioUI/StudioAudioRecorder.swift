import AVFoundation
import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Records the microphone to a 48 kHz mono WAV, filed in the domain's folder the way a run's
/// output is, so a reference voice or a conversation recorded here sits beside what it produces.
@MainActor
final class StudioAudioRecorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var duration: Double = 0
    @Published private(set) var lastRecordingURL: URL?
    @Published private(set) var errorMessage: String?

    private var recorder: AVAudioRecorder?

    func start(domain: StudioDomain) async {
        errorMessage = nil
        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        guard granted else {
            errorMessage = "Microphone access was denied. Enable it in System Settings."
            return
        }

        let proposed = StudioOutputLocation.specialistFile(
            domain: domain,
            name: "recording",
            fileExtension: "wav",
            now: StudioDisplayClock.now
        )
        do {
            let url = Self.recordingURL(proposed)
            guard let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48_000, channels: 1, interleaved: true) else {
                throw CocoaError(.formatting)
            }
            let recorder = try AVAudioRecorder(url: url, format: format)
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
        recorder?.stop()
        refresh()
        recorder = nil
        isRecording = false
        return lastRecordingURL
    }

    /// Stops and removes the file: the recording was not wanted.
    func discard() {
        let url = stop()
        if let url { try? FileManager.default.removeItem(at: url) }
        lastRecordingURL = nil
        duration = 0
    }

    func refresh() {
        duration = recorder?.currentTime ?? duration
    }
}

/// The popover an audio slot's "Record…" opens: one big button that starts, then stops, the
/// recording, with the clock beside it. Stop hands the file to the slot and closes; Cancel
/// discards it.
struct StudioAudioRecordingPopover: View {
    let domain: StudioDomain
    let onRecorded: (URL) -> Void

    @Environment(\.dismiss) private var dismiss
    @StateObject private var recorder = StudioAudioRecorder()
    private let ticker = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

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
                        Task { await recorder.start(domain: domain) }
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
                .help(recorder.isRecording ? "Stop and use the recording" : "Start recording")
                .accessibilityLabel(recorder.isRecording ? "Stop recording" : "Start recording")

                VStack(alignment: .leading, spacing: 3) {
                    Text(StudioTimeFormat.string(recorder.duration))
                        .font(.system(size: 22, weight: .medium, design: .monospaced))
                        .foregroundStyle(recorder.isRecording ? MereRunTheme.red : MereRunTheme.textPrimary)
                        .accessibilityLabel("Recorded \(StudioTimeFormat.string(recorder.duration))")
                    Text(recorder.isRecording ? "Stop to use it." : "48 kHz mono WAV, saved with your \(domain.title) work.")
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
            if recorder.isRecording { recorder.discard() }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Record audio")
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
