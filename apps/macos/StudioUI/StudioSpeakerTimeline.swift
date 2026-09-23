import StudioKit
import SwiftUI

/// Who spoke when: one lane per speaker over the recording's length, then every turn as a row in
/// the Analyze board's result panel — the rows Audio ▸ Transcribe already uses, so a diarized
/// conversation reads the way a transcript does.
struct StudioSpeakerTimelineView: View {
    let item: StudioLibraryItem
    let document: StudioDiarizationDocument
    let onSave: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StudioSpeakerLanes(document: document)
                .padding(12)
                .merePanel()
            StudioAnalyzeResultPanel(
                item: item,
                document: .diarization(document),
                detections: [],
                speechSegments: StudioAnalyzeDocument.diarization(document).speechSegments,
                outputText: nil,
                view: .timeline,
                nextActions: [.save("Save timeline…", .json)],
                onOpenTask: { _ in },
                onSave: { _ in onSave() }
            )
        }
    }
}

/// One row per speaker — the name, how long they held the floor — with their turns as bars on a
/// clock every lane shares, so turn-taking and overlap show at a glance.
struct StudioSpeakerLanes: View {
    let document: StudioDiarizationDocument

    private static let laneHeight: CGFloat = 18
    private static let labelWidth: CGFloat = 104

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(document.speakers) { speaker in
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(speaker.name)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(MereRunTheme.textPrimary)
                        Text("\(speaker.talkTimeDescription) · \(turns(speaker.turnCount))")
                            .font(.system(size: 10.5, weight: .medium))
                            .monospacedDigit()
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                    .frame(width: Self.labelWidth, alignment: .leading)
                    lane(for: speaker)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(speaker.name): \(turns(speaker.turnCount)), \(speaker.talkTimeDescription) speaking")
            }
            // The clock under the lanes: the start at the left edge of the track, the end at its
            // right edge, both under the lanes rather than the labels.
            HStack {
                Text("0:00")
                Spacer()
                Text(StudioTimeFormat.string(document.durationSeconds))
            }
            .padding(.leading, Self.labelWidth + 10)
            .font(.system(size: 10.5, weight: .medium, design: .monospaced))
            .foregroundStyle(MereRunTheme.textMuted)
            .accessibilityHidden(true)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Speaker lanes, \(document.summary)")
    }

    private func lane(for speaker: StudioDiarizationDocument.Speaker) -> some View {
        Canvas { context, size in
            let track = Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 4)
            context.fill(track, with: .color(MereRunTheme.surfaceRaised))
            guard document.durationSeconds > 0 else { return }
            let scale = size.width / document.durationSeconds
            for segment in document.segments where segment.speakerIndex == speaker.id {
                let x = segment.startSeconds * scale
                // A very short turn still gets a visible sliver.
                let width = max(2, (segment.endSeconds - segment.startSeconds) * scale)
                let bar = Path(
                    roundedRect: CGRect(x: x, y: 2, width: min(width, size.width - x), height: size.height - 4),
                    cornerRadius: 3
                )
                context.fill(bar, with: .color(Self.color(for: speaker.id)))
            }
        }
        .frame(height: Self.laneHeight)
    }

    private func turns(_ count: Int) -> String {
        count == 1 ? "1 turn" : "\(count) turns"
    }

    /// The theme's accent and green for the first two speakers, then hues stepped around the
    /// wheel from slate blue — never the theme's red, which means an error everywhere else.
    static func color(for index: Int) -> Color {
        let named = [MereRunTheme.accent, MereRunTheme.green]
        if index < named.count { return named[index] }
        return Color(hue: Double(index - named.count) * 0.17 + 0.6, saturation: 0.42, brightness: 0.66)
    }
}
