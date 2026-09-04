import AVFoundation
import AVKit
import StudioKit
import SwiftUI

/// AVAudioPlayer-backed playback model. The view drives `refresh()` from a timeline timer,
/// keeping all timer/sendability concerns out of the model.
@MainActor
final class StudioAudioPlayer: ObservableObject {
    @Published private(set) var isPlaying = false
    @Published private(set) var duration: Double = 0
    @Published private(set) var isReady = false
    @Published var currentTime: Double = 0

    private var player: AVAudioPlayer?

    func load(url: URL) {
        player?.stop()
        guard let loaded = try? AVAudioPlayer(contentsOf: url) else {
            player = nil
            isReady = false
            duration = 0
            currentTime = 0
            isPlaying = false
            return
        }
        loaded.prepareToPlay()
        player = loaded
        duration = loaded.duration
        currentTime = 0
        isPlaying = false
        isReady = true
    }

    func togglePlay() {
        isPlaying ? pause() : play()
    }

    func play() {
        guard let player else { return }
        if currentTime >= duration { currentTime = 0; player.currentTime = 0 }
        player.play()
        isPlaying = true
    }

    func pause() {
        player?.pause()
        isPlaying = false
    }

    func seek(to time: Double) {
        guard let player else { return }
        let clamped = min(max(0, time), duration)
        player.currentTime = clamped
        currentTime = clamped
    }

    /// Called by the view's timeline tick to refresh the playhead and detect end-of-play.
    func refresh() {
        guard let player else { return }
        currentTime = player.currentTime
        if isPlaying && !player.isPlaying {
            isPlaying = false
            currentTime = 0
        }
    }

    func stop() {
        player?.stop()
        isPlaying = false
    }
}

struct StudioAudioPlayerView: View {
    let url: URL

    @StateObject private var player = StudioAudioPlayer()
    @State private var isScrubbing = false
    @State private var peaks: [Float]?
    @State private var peaksLoaded = false
    private let ticker = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    private var playedFraction: Double {
        guard player.duration > 0 else { return 0 }
        return min(1, max(0, player.currentTime / player.duration))
    }

    var body: some View {
        VStack(spacing: 18) {
            if player.isReady {
                waveformOrSlider
                    .frame(maxWidth: 520)

                HStack(spacing: 16) {
                    Text(StudioTimeFormat.string(player.currentTime))
                        .monospacedDigit()
                        .frame(minWidth: 44, alignment: .trailing)

                    Button {
                        player.togglePlay()
                    } label: {
                        Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                            .font(.system(size: 46))
                            .foregroundStyle(MereRunTheme.accent)
                            .contentTransition(.symbolEffect(.replace))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(player.isPlaying ? "Pause" : "Play")

                    Text(StudioTimeFormat.string(player.duration))
                        .monospacedDigit()
                        .frame(minWidth: 44, alignment: .leading)
                }
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            } else {
                Image(systemName: "waveform.slash")
                    .font(.system(size: 40, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Audio preview unavailable.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: url) {
            player.load(url: url)
            peaks = nil
            peaksLoaded = false
            let loaded = await Task.detached(priority: .userInitiated) {
                StudioWaveformLoader.peaks(url: url)
            }.value
            guard !Task.isCancelled else { return }
            peaks = loaded
            peaksLoaded = true
        }
        .onReceive(ticker) { _ in
            if !isScrubbing { player.refresh() }
        }
        .onDisappear { player.stop() }
    }

    /// The waveform is the preferred scrubber; the slider remains for files whose PCM peaks
    /// can't be decoded, and while peaks are still loading.
    @ViewBuilder
    private var waveformOrSlider: some View {
        if let peaks, !peaks.isEmpty {
            StudioWaveformView(peaks: peaks, progress: playedFraction) { fraction in
                player.seek(to: fraction * player.duration)
            }
            .frame(height: 72)
            .transition(.opacity)
        } else if peaksLoaded {
            Slider(
                value: $player.currentTime,
                in: 0...max(player.duration, 0.1),
                onEditingChanged: { editing in
                    isScrubbing = editing
                    if !editing { player.seek(to: player.currentTime) }
                }
            )
            .tint(MereRunTheme.accent)
        } else {
            // Quiet placeholder bars while peaks decode, so the layout doesn't jump.
            StudioWaveformView(peaks: StudioWaveformView.placeholderPeaks, progress: 0)
                .frame(height: 72)
                .opacity(0.35)
        }
    }
}

extension StudioWaveformView {
    /// A gentle, deterministic bar silhouette for the loading state.
    static let placeholderPeaks: [Float] = (0..<96).map { index in
        0.25 + 0.2 * abs(sin(Float(index) * 0.35))
    }
}

struct StudioVideoPlayerView: View {
    let url: URL

    @State private var player: AVPlayer?

    var body: some View {
        VideoPlayer(player: player)
            .task(id: url) {
                player?.pause()
                player = AVPlayer(url: url)
            }
            .onDisappear { player?.pause() }
    }
}
