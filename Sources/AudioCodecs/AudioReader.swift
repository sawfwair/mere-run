import Foundation
import Accelerate
import AVFoundation

/// Audio reader supporting WAV, M4A, MP3, and other formats via AVFoundation
public enum AudioReader {
    /// Read audio file and return mono 16kHz audio samples
    /// Supports WAV, M4A, MP3, CAF, AIFF, and other AVFoundation-supported formats
    /// - Parameter url: URL to audio file
    /// - Returns: Audio samples in range [-1, 1] at 16kHz mono
    public static func readAudio(from url: URL) throws -> [Float] {
        // Use AVFoundation for broad format support
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioReaderError.readFailed("Failed to open audio file: \(error.localizedDescription)")
        }

        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard frameCount > 0 else {
            throw AudioReaderError.invalidFormat("Audio file is empty")
        }

        // Read into buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            throw AudioReaderError.readFailed("Failed to create audio buffer")
        }

        do {
            try file.read(into: buffer)
        } catch {
            throw AudioReaderError.readFailed("Failed to read audio data: \(error.localizedDescription)")
        }

        // Convert to mono 16kHz float using AVAudioConverter (more reliable than manual resample)
        var samples = try convertToMono16kFloat(buffer: buffer, sourceFormat: sourceFormat)

        if Self.isDebugEnabled {
            let stats = audioStats(samples)
            let message = String(
                format: "[ASR DEBUG] audio pre-norm rms=%.6f peak=%.6f\n",
                stats.rms, stats.peak
            )
            FileHandle.standardError.write(Data(message.utf8))
        }

        // Auto-gain very quiet audio so ASR doesn't return empty output.
        samples = normalizeIfNeeded(samples)

        if Self.isDebugEnabled {
            let stats = audioStats(samples)
            let message = String(
                format: "[ASR DEBUG] audio post-norm rms=%.6f peak=%.6f\n",
                stats.rms, stats.peak
            )
            FileHandle.standardError.write(Data(message.utf8))
        }

        return samples
    }

    /// Legacy WAV-only reader (kept for compatibility)
    public static func readWAV(from url: URL) throws -> [Float] {
        try readAudio(from: url)
    }

    /// Convert input buffer to mono 16kHz Float32 using AVAudioConverter.
    private static func convertToMono16kFloat(
        buffer: AVAudioPCMBuffer,
        sourceFormat: AVAudioFormat
    ) throws -> [Float] {
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        if sourceFormat.sampleRate == targetFormat.sampleRate,
           sourceFormat.channelCount == targetFormat.channelCount,
           sourceFormat.commonFormat == .pcmFormatFloat32 {
            if let floatData = buffer.floatChannelData {
                let frameLength = Int(buffer.frameLength)
                return Array(UnsafeBufferPointer(start: floatData[0], count: frameLength))
            }
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw AudioReaderError.invalidFormat("Failed to create audio converter")
        }

        let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
            throw AudioReaderError.readFailed("Failed to create output buffer")
        }

        var didConvert = false
        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if didConvert {
                outStatus.pointee = .noDataNow
                return nil
            }
            didConvert = true
            outStatus.pointee = .haveData
            return buffer
        }

        if let conversionError {
            throw AudioReaderError.readFailed("Audio conversion failed: \(conversionError.localizedDescription)")
        }

        guard let floatData = outputBuffer.floatChannelData else {
            throw AudioReaderError.invalidFormat("Converted audio missing float data")
        }

        let frameLength = Int(outputBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: floatData[0], count: frameLength))
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
