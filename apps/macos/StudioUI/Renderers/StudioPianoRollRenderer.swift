import AppKit
import StudioKit
import SwiftUI

/// Transcribed notes as the result panel's rows: the count, tracks, resolution, and tempo the
/// MIDI file declares, the notes on a piano roll, and Quick Look and Reveal for the file itself.
struct StudioPianoRollRenderer: View {
    let summary: StudioMIDISummary
    /// The MIDI file the notes were read from, when the run's artifacts name it.
    let midiURL: URL?

    private static let rollHeight: CGFloat = 240

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
                .frame(height: Self.rollHeight)
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
/// range the file uses, hue by channel, opacity by velocity.
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
