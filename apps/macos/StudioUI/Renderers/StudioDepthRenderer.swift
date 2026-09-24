import AppKit
import StudioKit
import SwiftUI

// What a depth run wrote: the depth preview PNG beside its EXR and manifest for a still, the
// review clip and per-frame previews for a video. The canvas shows the review clip when there is
// one, else the preview, with a strip to step through several.

/// The depth previews and review clip of a run, in the Analyze input column.
struct StudioDepthPreviewView: View {
    let artifacts: StudioVisionRunArtifacts
    let maxHeight: CGFloat

    @State private var selection: URL?

    /// The files the strip switches between: the clip first, then every preview.
    private var choices: [URL] { artifacts.clips + artifacts.previews }

    private var active: URL? {
        if let selection, choices.contains(selection) { return selection }
        return choices.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let active {
                Group {
                    if StudioOutputFileKind.classify(active) == .video {
                        StudioVideoPlayerView(url: active)
                            .aspectRatio(16 / 9, contentMode: .fit)
                    } else {
                        StudioAsyncImagePreview(url: active, maxPixelSize: 2_000, contentMode: .fit, fallbackSystemImage: "photo")
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(maxHeight: choices.count > 1 ? maxHeight - 40 : maxHeight)
                .background(MereRunTheme.surfaceRaised)
                .mereMediaFrame()
                .accessibilityLabel("Depth \(StudioOutputFileKind.classify(active) == .video ? "review clip" : "preview") \(active.lastPathComponent)")
            }
            if choices.count > 1 {
                StudioResultFileStrip(files: choices, selection: Binding(get: { active }, set: { selection = $0 }))
            }
        }
    }
}

/// What the depth manifest says about the run: the picture's size, the inference size, the
/// checkpoint, and the decoded range.
struct StudioDepthManifestRows: View {
    let manifest: StudioDepthManifest

    var body: some View {
        VStack(spacing: 0) {
            StudioResultFactRow("Size", "\(manifest.width) × \(manifest.height) px")
            StudioResultFactRow("Inferred", "\(manifest.inferenceWidth) × \(manifest.inferenceHeight) px")
            StudioResultFactRow("Checkpoint", "\(manifest.checkpoint) · \(manifest.parameterization)")
            StudioResultFactRow("Range", String(format: "%.3f – %.3f · %@", manifest.depthStatistics.rawMinimum,
                                                manifest.depthStatistics.rawMaximum, manifest.semantics))
        }
    }
}

/// Which of a run's files the canvas shows: a scrolling row of segments named after the files,
/// the way the specialist result view offered them.
struct StudioResultFileStrip: View {
    let files: [URL]
    @Binding var selection: URL?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 2) {
                ForEach(files, id: \.self) { url in
                    MereSegment(title: url.lastPathComponent, isSelected: selection == url) {
                        selection = url
                    }
                    .help(url.path)
                }
            }
            .padding(2)
            .background {
                RoundedRectangle(cornerRadius: 7)
                    .fill(MereRunTheme.surfaceRaised)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Result files")
    }
}
