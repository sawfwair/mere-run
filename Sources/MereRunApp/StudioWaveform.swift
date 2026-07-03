import AVFoundation
import SwiftUI

enum StudioWaveformLoader {
    /// Downsamples an audio file into `barCount` peak amplitudes normalized to 0…1.
    /// Long files are sampled sparsely (peaks are visual, not analytic), so cost stays bounded.
    /// Returns nil when the file can't be decoded to float PCM — callers fall back to a slider.
    static func peaks(url: URL, barCount: Int = 96) -> [Float]? {
        guard barCount > 0, let file = try? AVAudioFile(forReading: url) else { return nil }
        let totalFrames = Int(file.length)
        guard totalFrames > 0 else { return nil }

        let format = file.processingFormat
        let channelCount = Int(format.channelCount)
        guard channelCount > 0 else { return nil }

        let framesPerBar = max(1, totalFrames / barCount)
        // Cap the per-bar sampling work; visually indistinguishable from exhaustive peaks.
        let sampleStride = max(1, framesPerBar / 128)
        var peaks = [Float](repeating: 0, count: barCount)

        let chunkCapacity: AVAudioFrameCount = 65_536
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunkCapacity) else {
            return nil
        }

        var frameOffset = 0
        while frameOffset < totalFrames {
            buffer.frameLength = 0
            do {
                try file.read(into: buffer)
            } catch {
                break
            }
            let frames = Int(buffer.frameLength)
            guard frames > 0, let channels = buffer.floatChannelData else { break }

            for frame in stride(from: 0, to: frames, by: sampleStride) {
                var amplitude: Float = 0
                for channel in 0..<channelCount {
                    amplitude = max(amplitude, abs(channels[channel][frame]))
                }
                let bar = min(barCount - 1, (frameOffset + frame) / framesPerBar)
                peaks[bar] = max(peaks[bar], amplitude)
            }
            frameOffset += frames
        }

        guard let maxPeak = peaks.max(), maxPeak > 0 else { return peaks }
        return peaks.map { $0 / maxPeak }
    }
}

/// The audio identity element: peak bars around a center line, played bars in accent,
/// click or drag anywhere to seek.
struct StudioWaveformView: View {
    let peaks: [Float]
    /// Played fraction, 0…1.
    let progress: Double
    var onSeek: ((Double) -> Void)?

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                guard !peaks.isEmpty else { return }
                let count = peaks.count
                let slot = size.width / CGFloat(count)
                let barWidth = max(1.5, slot * 0.6)
                let playedBoundary = progress * Double(count)

                for index in 0..<count {
                    let amplitude = CGFloat(peaks[index])
                    let height = max(2.5, amplitude * size.height * 0.92)
                    let rect = CGRect(
                        x: CGFloat(index) * slot + (slot - barWidth) / 2,
                        y: (size.height - height) / 2,
                        width: barWidth,
                        height: height
                    )
                    let played = Double(index) + 0.5 <= playedBoundary
                    let color = played
                        ? MereRunTheme.accent
                        : MereRunTheme.textMuted.opacity(0.35)
                    context.fill(
                        Path(roundedRect: rect, cornerRadius: barWidth / 2),
                        with: .color(color)
                    )
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in seek(at: value.location.x, width: proxy.size.width) }
                    .onEnded { value in seek(at: value.location.x, width: proxy.size.width) }
            )
        }
        .accessibilityElement()
        .accessibilityLabel("Waveform")
        .accessibilityValue("\(Int((progress * 100).rounded())) percent played")
        .accessibilityAdjustableAction { direction in
            let step = 0.05
            switch direction {
            case .increment: onSeek?(min(1, progress + step))
            case .decrement: onSeek?(max(0, progress - step))
            @unknown default: break
            }
        }
    }

    private func seek(at locationX: CGFloat, width: CGFloat) {
        guard let onSeek, width > 0 else { return }
        onSeek(min(1, max(0, locationX / width)))
    }
}
