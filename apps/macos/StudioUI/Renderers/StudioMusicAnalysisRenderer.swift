import StudioKit
import SwiftUI

/// What ACE-Step understood about a piece, as the result panel's rows: tempo, key, meter,
/// language, and how much of the file it heard as tiles; the caption and the lyrics it heard as
/// prose; and — only when the run asked to keep them — the language model's whole reply and the
/// audio codes, folded away.
struct StudioMusicAnalysisRenderer: View {
    let analysis: StudioMusicAnalysisDocument
    @State private var showsRawReply = false
    @State private var showsAudioCodes = false

    private static let foldedMaxHeight: CGFloat = 220

    private var tiles: [(label: String, value: String)] {
        var tiles: [(String, String)] = []
        if let tempo = analysis.tempoDescription { tiles.append(("Tempo", tempo)) }
        if let key = analysis.metadata.keyscale, !key.isBlank { tiles.append(("Key", key)) }
        if let meter = analysis.metadata.timesignature, !meter.isBlank { tiles.append(("Meter", meter)) }
        if let language = analysis.languageDescription { tiles.append(("Language", language)) }
        tiles.append(("Analyzed", analysis.analyzedDescription))
        return tiles
    }

    private var tileRows: [[(label: String, value: String)]] {
        stride(from: 0, to: tiles.count, by: 3).map { Array(tiles[$0..<min($0 + 3, tiles.count)]) }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Tiles at their own width, three to a row, so five of them fit the result column;
            // a value longer than its share of the row truncates rather than pushing the row out.
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(tileRows.enumerated()), id: \.offset) { _, row in
                    HStack(alignment: .top, spacing: 22) {
                        ForEach(row, id: \.label) { tile in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(tile.label)
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(MereRunTheme.textMuted)
                                Text(tile.value)
                                    .font(.system(size: 14, weight: .bold, design: .rounded))
                                    .foregroundStyle(MereRunTheme.textPrimary)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                    .help(tile.value)
                            }
                            .accessibilityElement(children: .combine)
                        }
                        Spacer(minLength: 0)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            hairline

            if let caption = analysis.caption {
                prose("What it sounds like", caption)
            }
            if let lyrics = analysis.lyrics {
                prose("Lyrics", lyrics)
            }
            if analysis.caption == nil, analysis.lyrics == nil, analysis.metadata.bpm == nil {
                Text("The model found no tempo, key, caption, or lyrics in this recording.")
                    .font(.system(size: 12.5))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                hairline
            }
            if let reply = analysis.rawLMOutput, !reply.isBlank {
                disclosure("Model reply", isExpanded: $showsRawReply, text: reply)
            }
            if let codes = analysis.audioCodes, !codes.isBlank {
                disclosure("Audio codes", isExpanded: $showsAudioCodes, text: codes)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Music analysis")
    }

    private func prose(_ title: String, _ text: String) -> some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                    .kerning(0.55)
                    .textCase(.uppercase)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .accessibilityAddTraits(.isHeader)
                Text(text)
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            hairline
        }
    }

    private func disclosure(_ title: String, isExpanded: Binding<Bool>, text: String) -> some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Button {
                    withAnimation(MereRunTheme.Motion.quick) { isExpanded.wrappedValue.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: isExpanded.wrappedValue ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                        Text(title)
                            .font(.callout.weight(.medium))
                    }
                    .foregroundStyle(MereRunTheme.textSecondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(isExpanded.wrappedValue ? "Hide \(title.lowercased())" : "Show \(title.lowercased())")
                if isExpanded.wrappedValue {
                    ScrollView {
                        Text(text)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(MereRunTheme.textSecondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                    }
                    .frame(maxHeight: Self.foldedMaxHeight)
                    .background {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                            .fill(MereRunTheme.surfaceRaised.opacity(0.5))
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            hairline
        }
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }
}
