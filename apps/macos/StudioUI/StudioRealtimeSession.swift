import Foundation
import StudioKit
import SwiftUI

/// Pure view-model logic for Music ▸ Realtime: the transport clock, the prompt blend, the
/// stdin steering commands, the session log, and the job-bar copy. Everything here is derived
/// from what the CLI reports (progress lines, log lines, exit state); nothing is invented.
enum StudioRealtimeTransport {
    /// Magenta RT2 renders 25 frames per second of audio (`MagentaRT2Resources.frameRate`).
    static let frameRate = 25.0
    /// Magenta RT2 output sample rate (`MagentaRT2Resources.sampleRate`).
    static let sampleRateHz = 48_000

    /// `04:12` — minutes always two digits so the mono line keeps a stable width.
    static func timestamp(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds > 0 else { return "00:00" }
        let total = Int(seconds.rounded(.down))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    /// Seconds of audio rendered so far, from the `Realtime frame N/T` progress the CLI writes
    /// (parsed into "Step N of T" by `StudioProgressParser`). Nil until the first frame.
    static func renderedSeconds(from progress: StudioRunProgress?) -> TimeInterval? {
        guard let progress, progress.label == "Realtime music", let detail = progress.detail else {
            return nil
        }
        let scanner = Scanner(string: detail)
        guard scanner.scanString("Step") != nil, let frames = scanner.scanInt() else { return nil }
        return Double(frames) / frameRate
    }

    /// Where playback is, given how long the session has run and how much audio exists. The
    /// CLI paces frames to wall-clock time, so playback can never be ahead of what is rendered.
    static func playbackPosition(wallElapsed: TimeInterval, rendered: TimeInterval?) -> TimeInterval {
        let elapsed = max(0, wallElapsed)
        guard let rendered else { return 0 }
        return min(elapsed, rendered)
    }

    /// `04:12 · 2.1 s ahead of playback`. Before the first frame the session is still loading.
    static func statusLine(wallElapsed: TimeInterval, rendered: TimeInterval?) -> String {
        guard let rendered else {
            return "\(timestamp(wallElapsed)) · loading model"
        }
        let playback = playbackPosition(wallElapsed: wallElapsed, rendered: rendered)
        let lead = rendered - playback
        return "\(timestamp(playback)) · \(String(format: "%.1f", max(0, lead))) s ahead of playback"
    }

    /// `Magenta RT2 · 48 kHz` for any Magenta RT2 checkpoint; other ids are shown verbatim.
    static func engineLabel(model: String) -> String {
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.localizedCaseInsensitiveContains("magenta-rt2") {
            return "Magenta RT2 · \(sampleRateHz / 1_000) kHz"
        }
        return trimmed
    }
}

/// Prompt A / Prompt B and the blend slider map onto the CLI's single `prompt <text>` steering
/// command. Magenta RT2 conditions on one text prompt, so the blend chooses which prompt leads:
/// the endpoints send one prompt alone, everything in between sends both with the dominant
/// prompt first. Embedding-level interpolation is a CLI feature; when it lands the same slider
/// drives it.
enum StudioRealtimeSteering {
    static func blendedPrompt(a: String, b: String, blend: Double) -> String {
        let promptA = a.trimmingCharacters(in: .whitespacesAndNewlines)
        let promptB = b.trimmingCharacters(in: .whitespacesAndNewlines)
        if promptB.isEmpty { return promptA }
        if promptA.isEmpty { return promptB }
        let clamped = min(1, max(0, blend))
        if clamped <= 0.05 { return promptA }
        if clamped >= 0.95 { return promptB }
        return clamped < 0.5 ? "\(promptA), \(promptB)" : "\(promptB), \(promptA)"
    }

    static func formatBlend(_ blend: Double) -> String {
        String(format: "%.2f", min(1, max(0, blend)))
    }

    /// The stdin lines for the slider row and the extended controls, in the CLI's own words
    /// (`MagentaRT2LiveControlQueue`). Guidance is the MusicCoCa classifier-free guidance.
    static func controlCommands(
        temperature: Double,
        topK: Int,
        guidance: Double,
        noteGuidance: Double,
        drumGuidance: Double,
        style: String,
        drumless: Bool
    ) -> [String] {
        [
            "style \(style)",
            "temp \(format(temperature))",
            "topk \(topK)",
            "mc \(format(guidance))",
            "notes \(format(noteGuidance))",
            "drums \(format(drumGuidance))",
            "drumless \(drumless ? "on" : "off")"
        ]
    }

    static func format(_ value: Double) -> String {
        let text = String(format: "%.2f", value)
        guard text.contains(".") else { return text }
        var trimmed = Substring(text)
        while trimmed.hasSuffix("0") { trimmed = trimmed.dropLast() }
        if trimmed.hasSuffix(".") { trimmed = trimmed.dropLast() }
        return String(trimmed)
    }
}

/// The session log is the CLI's own stderr plus the live-control lines the controller records,
/// stamped with the session clock.
enum StudioRealtimeSessionLog {
    static func lines(_ logs: [LogLine], startedAt: Date) -> [String] {
        logs.map { line in
            let stamp = StudioRealtimeTransport.timestamp(line.date.timeIntervalSince(startedAt))
            return "\(stamp)  \(line.text)"
        }
    }

    static func copyText(_ lines: [String]) -> String {
        lines.joined(separator: "\n")
    }

    /// The newest lines that fit a box of `height` points at `lineHeight` points per line.
    static func visibleTail(_ lines: [String], height: CGFloat, lineHeight: CGFloat) -> [String] {
        guard lineHeight > 0 else { return lines }
        let capacity = max(1, Int(height / lineHeight))
        return Array(lines.suffix(capacity))
    }
}

/// Copy and color for the job bar under the session.
enum StudioRealtimeJobBar {
    enum Phase: Equatable {
        case idle
        case queued
        case live
        case ended(exitCode: Int32?)
    }

    /// `rt2-medium` from `music-magenta-rt2-medium`; other ids keep their last dashed segments.
    static func residentName(model: String) -> String {
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "rt2-small" }
        for prefix in ["music-magenta-", "magenta-"] where trimmed.hasPrefix(prefix) {
            return String(trimmed.dropFirst(prefix.count))
        }
        return trimmed
    }

    static func detail(phase: Phase, startedAt: Date?, model: String, now: Date = Date()) -> String {
        let resident = "\(residentName(model: model)) resident"
        switch phase {
        case .idle:
            return "No session · press play to start · \(residentName(model: model))"
        case .queued:
            return "Queued behind the active job · \(residentName(model: model))"
        case .live:
            let started = startedAt.map { " · started \(startTimeFormatter.string(from: $0))" } ?? ""
            return "Session running\(started) · \(resident)"
        case .ended(let exitCode):
            let ran = startedAt.map { " · ran \(StudioRealtimeTransport.timestamp(now.timeIntervalSince($0)))" } ?? ""
            if let exitCode, exitCode != 0 {
                return "Session failed · exit \(exitCode)\(ran)"
            }
            return "Session ended\(ran) · \(residentName(model: model))"
        }
    }

    static func dotColor(phase: Phase) -> Color {
        switch phase {
        case .idle: return MereRunTheme.textMuted
        case .queued: return MereRunTheme.yellow
        case .live: return MereRunTheme.green
        case .ended(let exitCode):
            return exitCode == 0 || exitCode == nil ? MereRunTheme.textMuted : MereRunTheme.red
        }
    }

    private static let startTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "h:mm a"
        return formatter
    }()
}

/// Test seam for the steering panel: when set in the environment the fields start from these
/// values instead of empty, so the snapshot harness can show a mid-session steer without a
/// user typing. Production never sets it.
struct StudioRealtimeSteeringSeed: Equatable {
    var promptA: String
    var promptB: String
    var blend: Double
}

private struct StudioRealtimeSteeringSeedKey: EnvironmentKey {
    static let defaultValue: StudioRealtimeSteeringSeed? = nil
}

extension EnvironmentValues {
    var studioRealtimeSteeringSeed: StudioRealtimeSteeringSeed? {
        get { self[StudioRealtimeSteeringSeedKey.self] }
        set { self[StudioRealtimeSteeringSeedKey.self] = newValue }
    }
}
