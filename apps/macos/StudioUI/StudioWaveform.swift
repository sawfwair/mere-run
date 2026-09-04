import AVFoundation
import StudioKit
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

extension StudioWaveformLoader {
    /// Peaks for a WAV that is still being written. The realtime recorder (`StreamingWAVWriter`)
    /// patches the RIFF and data sizes only when it closes the file, so `AVAudioFile` reports zero
    /// frames mid-session. This reads the `fmt ` chunk itself and treats every byte after the
    /// `data` chunk header as samples, so a growing file yields a growing waveform. Supports the
    /// recorder's float32 PCM and plain 16-bit PCM; returns nil for anything else.
    static func growingWAVPeaks(url: URL, barCount: Int = 96) -> [Float]? {
        guard barCount > 0, let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            return nil
        }
        guard data.count >= 12,
              data[0..<4].elementsEqual("RIFF".utf8),
              data[8..<12].elementsEqual("WAVE".utf8) else {
            return nil
        }

        func uint16(_ offset: Int) -> Int {
            Int(data[offset]) | Int(data[offset + 1]) << 8
        }
        func uint32(_ offset: Int) -> Int {
            uint16(offset) | uint16(offset + 2) << 16
        }

        var offset = 12
        var formatCode = 0
        var channelCount = 0
        var bitsPerSample = 0
        var samplesStart: Int?
        while offset + 8 <= data.count {
            let chunkID = data[offset..<offset + 4]
            let chunkSize = uint32(offset + 4)
            let body = offset + 8
            if chunkID.elementsEqual("fmt ".utf8), body + 16 <= data.count {
                formatCode = uint16(body)
                channelCount = uint16(body + 2)
                bitsPerSample = uint16(body + 14)
            } else if chunkID.elementsEqual("data".utf8) {
                samplesStart = body
                break
            }
            offset = body + chunkSize + (chunkSize % 2)
        }

        guard let samplesStart, channelCount > 0 else { return nil }
        let bytesPerSample: Int
        switch (formatCode, bitsPerSample) {
        case (3, 32): bytesPerSample = 4
        case (1, 16): bytesPerSample = 2
        default: return nil
        }
        let frameBytes = channelCount * bytesPerSample
        let totalFrames = (data.count - samplesStart) / frameBytes
        guard totalFrames > 0 else { return [Float](repeating: 0, count: barCount) }

        let framesPerBar = max(1, totalFrames / barCount)
        let sampleStride = max(1, framesPerBar / 128)
        var peaks = [Float](repeating: 0, count: barCount)
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress?.advanced(by: samplesStart) else { return }
            for frame in stride(from: 0, to: totalFrames, by: sampleStride) {
                var amplitude: Float = 0
                for channel in 0..<channelCount {
                    let pointer = base.advanced(by: (frame * channelCount + channel) * bytesPerSample)
                    let value: Float
                    if bytesPerSample == 4 {
                        value = Float(bitPattern: UInt32(littleEndian: pointer.loadUnaligned(as: UInt32.self)))
                    } else {
                        value = Float(Int16(littleEndian: pointer.loadUnaligned(as: Int16.self))) / Float(Int16.max)
                    }
                    amplitude = max(amplitude, abs(value))
                }
                let bar = min(barCount - 1, frame / framesPerBar)
                peaks[bar] = max(peaks[bar], amplitude)
            }
        }
        guard let maxPeak = peaks.max(), maxPeak > 0 else { return peaks }
        return peaks.map { $0 / maxPeak }
    }
}
