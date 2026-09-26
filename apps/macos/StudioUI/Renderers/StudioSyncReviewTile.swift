import StudioKit
import SwiftUI

/// Sound ▸ Video Foley's finished card: the picture the effect was made for over the waveform
/// it produced, so sync is judged by watching and listening together. The clip is the run's
/// input; the WAV is its output, drawn with the same full-width player the feed gives any audio.
struct StudioSyncReviewTile: View {
    let videoURL: URL
    let audioURL: URL

    private static let videoMaxHeight: CGFloat = 300

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioVideoPlayerView(url: videoURL)
                .aspectRatio(16 / 9, contentMode: .fit)
                .frame(maxWidth: .infinity, maxHeight: Self.videoMaxHeight)
                .background(MereRunTheme.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
                .help(videoURL.lastPathComponent)
                .accessibilityLabel("Picture \(videoURL.lastPathComponent)")
            StudioAudioPlayerView(url: audioURL)
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity)
                .background(MereRunTheme.surfaceRaised.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
                .studioFileDrag(audioURL)
                .accessibilityLabel("Generated foley \(audioURL.lastPathComponent)")
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Picture and generated foley")
    }
}
