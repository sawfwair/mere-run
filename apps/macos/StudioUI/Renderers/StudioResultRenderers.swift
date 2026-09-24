import StudioKit
import SwiftUI

/// What a bespoke renderer draws: one case per (view, document) pairing the registry knows.
/// Typed rather than `AnyView` so the panel's switch is exhaustive and a new renderer is one
/// case plus one view.
enum StudioResultRendering: Equatable {
    /// A tensor file's header, for Earth's safetensors, `sfx ae encode`'s `.npy`, and the
    /// conditioning tensors `sfx condition text` exports.
    case tensor(StudioTensorHeader)
    /// The CLAP alignment gauge over `sfx clap score`'s printed result.
    case clap(StudioCLAPScore.Output)
    /// Video Foley's picture over the waveform it produced.
    case syncReview(video: URL, audio: URL)
    /// An audio result the input column cannot play for the run (`sfx ae decode` takes a
    /// latents file, not audio): the waveform player over the file's row.
    case audioOutput(URL)
}

/// The registry the Analyze result panel asks before drawing its own rows: given the view the
/// strip has selected and the decoded document, the bespoke rendering for that pair, or nil
/// when the panel's generic rows (detections, speech turns, text) already say it.
///
/// A page PR moves its renderer into this folder, adds a `StudioResultRendering` case, a match
/// here, and a branch in `StudioResultRendererView`; the panel, the canvas, and the schema never
/// change for it. Rendering is keyed by `(view, document)` rather than by task, so a document
/// drawn the same way for two tasks (a tensor header from Earth or from `sfx ae encode`) is
/// drawn once.
enum StudioResultRenderers {
    static func rendering(
        for view: StudioAnalyzeResultView,
        document: StudioAnalyzeDocument?,
        item: StudioLibraryItem
    ) -> StudioResultRendering? {
        switch (view, document) {
        case (.tensor, .tensor(let header)):
            return .tensor(header)
        case (.score, .clap(let output)):
            return .clap(output)
        case (.audio, nil):
            // The canvas plays an audio task's output in the input column; a run whose input
            // was not audio has nowhere else to be heard. Tasks on the shared workspace only,
            // like the panel's own output rows.
            guard item.templateID?.studioTask.usesTaskDraft == true,
                  item.inputURL.map({ StudioOutputFileKind.classify($0) }) != .audio,
                  let audio = item.allArtifactURLs.first(where: {
                      StudioOutputFileKind.classify($0) == .audio && FileManager.default.fileExists(atPath: $0.path)
                  }) else { return nil }
            return .audioOutput(audio)
        default:
            return nil
        }
    }

    /// The generation feed's counterpart: what a finished card draws in place of its output
    /// grid, keyed by what the run wrote rather than by task. Video Foley's WAV is reviewed
    /// against the clip it was made for; a tensor output (`sfx condition text`) shows its
    /// header, since a `.safetensors` file has no picture to tile. Nil leaves the grid to it.
    static func cardRendering(for item: StudioLibraryItem, files: [URL]) -> StudioResultRendering? {
        guard let output = item.outputURL, files.contains(output) else { return nil }
        if item.templateID == .sfxVideo, let video = item.inputURL, StudioOutputFileKind.classify(video) == .video,
           StudioOutputFileKind.classify(output) == .audio, FileManager.default.fileExists(atPath: video.path) {
            return .syncReview(video: video, audio: output)
        }
        if ["safetensors", "npy"].contains(output.pathExtension.lowercased()), let header = StudioTensorHeader.load(from: output) {
            return .tensor(header)
        }
        return nil
    }

    /// The files a card rendering already shows, so the card lists neither as a sidecar.
    static func renderedFiles(of rendering: StudioResultRendering, item: StudioLibraryItem) -> [URL] {
        switch rendering {
        case .tensor:
            return item.outputURL.map { [$0] } ?? []
        case .clap:
            return []
        case .syncReview(_, let audio), .audioOutput(let audio):
            return [audio]
        }
    }
}

/// A card rendering in the generation feed's chrome: rows (a tensor header) sit on the raised
/// panel the card gives a text preview; the sync review is its own tiles.
struct StudioCardRenderingView: View {
    let rendering: StudioResultRendering
    let item: StudioLibraryItem

    var body: some View {
        switch rendering {
        case .tensor, .clap, .audioOutput:
            StudioResultRendererView(rendering: rendering, item: item)
                .background(MereRunTheme.surfaceRaised.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        case .syncReview:
            StudioResultRendererView(rendering: rendering, item: item)
        }
    }
}

/// The rows a rendering contributes inside the result panel, under its header and above its
/// action row, or inside a generation card in place of the output grid.
struct StudioResultRendererView: View {
    let rendering: StudioResultRendering
    let item: StudioLibraryItem

    var body: some View {
        switch rendering {
        case .tensor(let header):
            StudioTensorInspector(header: header)
        case .clap(let output):
            StudioCLAPGauge(output: output)
        case .syncReview(let video, let audio):
            StudioSyncReviewTile(videoURL: video, audioURL: audio)
        case .audioOutput(let audio):
            StudioAudioOutputRows(url: audio)
        }
    }
}

/// The decoded audio as the result panel's rows: the player, then the file's name with Reveal.
private struct StudioAudioOutputRows: View {
    let url: URL

    var body: some View {
        VStack(spacing: 0) {
            StudioAudioPlayerView(url: url)
                .frame(height: 210)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            hairline
            HStack(spacing: 10) {
                Image(systemName: "waveform")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(width: 14)
                Text(url.lastPathComponent)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .help(url.path)
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    .buttonStyle(.mereSecondary)
                    .controlSize(.small)
                    .accessibilityLabel("Reveal \(url.lastPathComponent) in Finder")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .accessibilityElement(children: .contain)
            hairline
        }
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }
}
