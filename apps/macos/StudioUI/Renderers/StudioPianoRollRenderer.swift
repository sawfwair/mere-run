import AppKit
import StudioKit
import SwiftUI

/// Transcribed notes as the result panel's rows: the count, tracks, resolution, and tempo the
/// MIDI file declares, the notes on a piano roll, and Quick Look and Reveal for the file itself.
struct StudioPianoRollRenderer: View {
    let summary: StudioMIDISummary
    /// The MIDI file the notes were read from, when the run's artifacts name it.
    let midiURL: URL?

    private static let rollMinHeight: CGFloat = 320

    private var metrics: [(label: String, value: String)] {
        [
            ("Notes", "\(summary.notes.count)"),
            ("Tracks", "\(summary.trackCount)"),
            ("PPQ", "\(summary.ticksPerQuarter)"),
            ("Tempo", summary.tempoMicrosecondsPerQuarter.map {
                "\(Int((60_000_000.0 / Double($0)).rounded())) BPM"
            } ?? "—")
        ]
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                ForEach(metrics, id: \.label) { metric in
                    VStack(alignment: .leading, spacing: 3) {
                        Text(metric.label)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text(metric.value)
                            .font(.system(size: 14, weight: .bold, design: .rounded))
                            .foregroundStyle(MereRunTheme.textPrimary)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityElement(children: .combine)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            hairline
            StudioMIDIPianoRoll(summary: summary)
                .frame(minHeight: Self.rollMinHeight)
            hairline
            if let midiURL {
                HStack(spacing: 6) {
                    Text(midiURL.lastPathComponent)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 8)
                    Button("Quick Look") { QuickLookCoordinator.shared.preview(midiURL) }
                        .buttonStyle(.mereSecondary)
                        .help("Preview the MIDI file")
                    Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([midiURL]) }
                        .buttonStyle(.mereSecondary)
                        .help("Show the MIDI file in Finder")
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                hairline
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Transcribed notes")
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }
}

/// Every note as a bar: time left to right over the whole file, pitch bottom to top over the
/// range the file uses, hue by channel, opacity by velocity. The file's range and length are
/// read once; a dense file (a long orchestral transcription) is drawn per pixel column and pitch
/// row instead of per note, so the drawing costs the size of the roll, not the size of the file.
private struct StudioMIDIPianoRoll: View {
    let summary: StudioMIDISummary
    private let pitches: ClosedRange<Int>?
    private let totalTicks: Int

    /// Above this many notes the roll rasterizes into cells before drawing.
    private static let bucketingThreshold = 4_000

    init(summary: StudioMIDISummary) {
        self.summary = summary
        pitches = summary.pitchRange
        totalTicks = max(1, summary.totalTicks)
    }

    var body: some View {
        Canvas { context, size in
            guard !summary.notes.isEmpty, let pitches else {
                let text = context.resolve(
                    Text("No note events found")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                )
                context.draw(text, at: CGPoint(x: size.width / 2, y: size.height / 2))
                return
            }
            let pitchSpan = max(1, pitches.upperBound - pitches.lowerBound + 1)
            let rowHeight = max(2, size.height / CGFloat(pitchSpan))
            let ticksPerPoint = CGFloat(totalTicks) / max(1, size.width)
            if summary.notes.count > Self.bucketingThreshold {
                for run in bucketedRuns(pitches: pitches, ticksPerPoint: ticksPerPoint, width: size.width) {
                    let y = size.height - CGFloat(run.pitch - pitches.lowerBound + 1) * rowHeight
                    fill(&context, x: CGFloat(run.startColumn), width: CGFloat(run.endColumn - run.startColumn + 1),
                         y: y, rowHeight: rowHeight, channel: run.channel, velocity: run.velocity)
                }
            } else {
                for note in summary.notes {
                    let x = CGFloat(note.startTick) / ticksPerPoint
                    let width = max(2, CGFloat(note.durationTicks) / ticksPerPoint)
                    let y = size.height - CGFloat(note.pitch - pitches.lowerBound + 1) * rowHeight
                    fill(&context, x: x, width: width, y: y, rowHeight: rowHeight, channel: note.channel, velocity: note.velocity)
                }
            }
        }
        .background {
            LinearGradient(
                colors: [MereRunTheme.surfaceRaised, MereRunTheme.surface],
                startPoint: .top,
                endPoint: .bottom
            )
        }
        .accessibilityElement()
        .accessibilityLabel("Piano roll with \(summary.notes.count) notes")
    }

    private func fill(
        _ context: inout GraphicsContext, x: CGFloat, width: CGFloat, y: CGFloat, rowHeight: CGFloat,
        channel: Int, velocity: Int
    ) {
        context.fill(
            Path(roundedRect: CGRect(x: x, y: y, width: width, height: max(1.5, rowHeight - 1)), cornerRadius: 1.5),
            with: .color(Color(
                hue: Double(channel) / 16,
                saturation: 0.7,
                brightness: 0.92,
                opacity: 0.45 + 0.55 * Double(velocity) / 127
            ))
        )
    }

    /// One filled span per pitch row: the pixel columns the notes of that pitch cover, merged
    /// where they touch, carrying the loudest note's velocity and its channel.
    private struct Run {
        var pitch: Int
        var startColumn: Int
        var endColumn: Int
        var channel: Int
        var velocity: Int
    }

    private func bucketedRuns(pitches: ClosedRange<Int>, ticksPerPoint: CGFloat, width: CGFloat) -> [Run] {
        let columns = max(1, Int(width.rounded(.up)))
        // Per pitch row, per column: (velocity, channel) of the loudest note touching that cell.
        var cells: [Int: (velocity: Int, channel: Int)] = [:]
        for note in summary.notes {
            let first = min(columns - 1, Int(CGFloat(note.startTick) / ticksPerPoint))
            let last = min(columns - 1, max(first, Int(CGFloat(note.startTick + note.durationTicks) / ticksPerPoint)))
            let row = (note.pitch - pitches.lowerBound) * columns
            for column in first...last {
                let key = row + column
                if let existing = cells[key], existing.velocity >= note.velocity { continue }
                cells[key] = (note.velocity, note.channel)
            }
        }
        var runs: [Run] = []
        for key in cells.keys.sorted() {
            guard let cell = cells[key] else { continue }
            let pitch = pitches.lowerBound + key / columns
            let column = key % columns
            if var last = runs.last, last.pitch == pitch, last.endColumn + 1 == column, last.channel == cell.channel {
                last.endColumn = column
                last.velocity = max(last.velocity, cell.velocity)
                runs[runs.count - 1] = last
            } else {
                runs.append(Run(pitch: pitch, startColumn: column, endColumn: column, channel: cell.channel, velocity: cell.velocity))
            }
        }
        return runs
    }
}
