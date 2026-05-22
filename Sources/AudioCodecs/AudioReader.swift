import Foundation
import MediaIO

/// Audio reader supporting WAV, M4A, MP3, and other formats via platform media backends.
public enum AudioReader {
    /// Read audio file and return mono 16kHz audio samples
    /// Supports WAV, M4A, MP3, CAF, AIFF, and other platform-supported formats.
    /// - Parameter url: URL to audio file
    /// - Returns: Audio samples in range [-1, 1] at 16kHz mono
    public static func readAudio(from url: URL) throws -> [Float] {
        let buffer = try readAudioBuffer(from: url, sampleRate: 16_000, channels: 1)
        return buffer.samples
    }

    /// Read audio and return a normalized floating-point buffer.
    public static func readAudioBuffer(
        from url: URL,
        sampleRate: Int = 16_000,
        channels: Int = 1
    ) throws -> MediaAudioBuffer {
        let decoded: MediaAudioBuffer
        do {
            decoded = try MediaAudioIO.decode(url, targetSampleRate: sampleRate, channels: channels)
        } catch {
            throw AudioReaderError.readFailed(error.localizedDescription)
        }

        if Self.isDebugEnabled {
            let stats = audioStats(decoded.samples)
            let message = String(
                format: "[ASR DEBUG] audio pre-norm rms=%.6f peak=%.6f\n",
                stats.rms, stats.peak
            )
            FileHandle.standardError.write(Data(message.utf8))
        }

        // Auto-gain very quiet audio so ASR doesn't return empty output.
        let samples = normalizeIfNeeded(decoded.samples)

        if Self.isDebugEnabled {
            let stats = audioStats(samples)
            let message = String(
                format: "[ASR DEBUG] audio post-norm rms=%.6f peak=%.6f\n",
                stats.rms, stats.peak
            )
            FileHandle.standardError.write(Data(message.utf8))
        }

        return MediaAudioBuffer(
            samples: samples,
            sampleRate: decoded.sampleRate,
            channelCount: decoded.channelCount,
            isInterleaved: decoded.isInterleaved
        )
    }

    /// Legacy WAV-only reader (kept for compatibility)
    public static func readWAV(from url: URL) throws -> [Float] {
        try readAudio(from: url)
    }

    /// Normalize extremely quiet audio to a target RMS level.
    private static func normalizeIfNeeded(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }

        var sumSquares: Float = 0
        for s in samples {
            sumSquares += s * s
        }
        let meanSquares = sumSquares / Float(samples.count)
        let rms = sqrt(meanSquares)

        // If RMS is below ~-40 dBFS, boost toward ~-20 dBFS.
        let minRms: Float = 0.01
        let targetRms: Float = 0.1
        guard rms > 0, rms < minRms else {
            return samples
        }

        let gain = min(targetRms / rms, 100.0)
        if gain <= 1 {
            return samples
        }

        var boosted = [Float](repeating: 0, count: samples.count)
        for i in 0..<samples.count {
            let scaled = samples[i] * gain
            boosted[i] = max(-1.0, min(1.0, scaled))
        }
        return boosted
    }

    private static var isDebugEnabled: Bool {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }

    private static func audioStats(_ samples: [Float]) -> (rms: Float, peak: Float) {
        guard !samples.isEmpty else { return (0, 0) }
        var sumSquares: Float = 0
        var peak: Float = 0
        for s in samples {
            sumSquares += s * s
            let absVal = abs(s)
            if absVal > peak { peak = absVal }
        }
        let meanSquares = sumSquares / Float(samples.count)
        return (sqrt(meanSquares), peak)
    }
}

public enum AudioReaderError: LocalizedError {
    case invalidFormat(String)
    case readFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let message):
            return "Invalid audio format: \(message)"
        case .readFailed(let message):
            return "Failed to read audio: \(message)"
        }
    }
}
