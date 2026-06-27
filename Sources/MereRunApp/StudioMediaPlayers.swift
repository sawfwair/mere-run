import AVFoundation
import AVKit
import SwiftUI

/// Formats a duration in seconds as m:ss (or h:mm:ss) for transport labels.
enum StudioTimeFormat {
    static func string(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%d:%02d", minutes, secs)
    }
}

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
    private let ticker = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform")
                .font(.system(size: 48, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)

            if player.isReady {
                VStack(spacing: 8) {
                    Slider(
                        value: $player.currentTime,
                        in: 0...max(player.duration, 0.1),
                        onEditingChanged: { editing in
                            isScrubbing = editing
                            if !editing { player.seek(to: player.currentTime) }
                        }
                    )
                    .tint(MereRunTheme.accent)

                    HStack {
                        Text(StudioTimeFormat.string(player.currentTime))
                        Spacer()
                        Text(StudioTimeFormat.string(player.duration))
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                }
                .frame(maxWidth: 420)

                Button {
                    player.togglePlay()
                } label: {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(MereRunTheme.accent)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(player.isPlaying ? "Pause" : "Play")
            } else {
                Text("Audio preview unavailable.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: url) { player.load(url: url) }
        .onReceive(ticker) { _ in
            if !isScrubbing { player.refresh() }
        }
        .onDisappear { player.stop() }
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
