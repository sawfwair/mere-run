import Foundation

public enum MediaAudioIO {
    public typealias FloatSampleProvider = (_ sampleRange: Range<Int>) throws -> [Float]

    public static func decode(
        _ url: URL,
        targetSampleRate: Int,
        channels: Int
    ) throws -> MediaAudioBuffer {
        #if canImport(AVFoundation)
        try AppleMediaAudioIO.decode(url, targetSampleRate: targetSampleRate, channels: channels)
        #else
        try FFmpegMediaIO.decodeAudio(url, targetSampleRate: targetSampleRate, channels: channels)
        #endif
    }

    public static func probe(_ url: URL) throws -> MediaAudioMetadata {
        #if canImport(AVFoundation)
        try AppleMediaAudioIO.probe(url)
        #else
        try FFmpegMediaIO.probeAudio(url)
        #endif
    }

    /// Decodes only the requested source interval, then converts it to the
    /// requested sample rate and interleaved channel layout.
    public static func decodeSegment(
        _ url: URL,
        startTime: Double,
        duration: Double,
        targetSampleRate: Int,
        channels: Int
    ) throws -> MediaAudioBuffer {
        guard startTime.isFinite, duration.isFinite, startTime >= 0, duration > 0 else {
            throw MediaIOError.invalidAudioRange(startTime: startTime, duration: duration)
        }
        #if canImport(AVFoundation)
        return try AppleMediaAudioIO.decodeSegment(
            url,
            startTime: startTime,
            duration: duration,
            targetSampleRate: targetSampleRate,
            channels: channels
        )
        #else
        return try FFmpegMediaIO.decodeAudioSegment(
            url,
            startTime: startTime,
            duration: duration,
            targetSampleRate: targetSampleRate,
            channels: channels
        )
        #endif
    }

    public static func writeFloatWAV(
        samples: [Float],
        sampleRate: Int,
        channels: Int,
        to url: URL
    ) throws {
        try writeFloatWAV(
            sampleCount: samples.count,
            sampleRate: sampleRate,
            channels: channels,
            to: url
        ) { range in
            Array(samples[range])
        }
    }

    /// Writes a float WAV incrementally. The provider receives ascending,
    /// channel-aligned sample ranges so device-backed producers never need a
    /// whole-output host array or a second whole-file `Data` allocation.
    public static func writeFloatWAV(
        sampleCount: Int,
        sampleRate: Int,
        channels: Int,
        chunkSampleCount: Int = 65_536,
        to url: URL,
        samplesAt sampleProvider: FloatSampleProvider
    ) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let channelCount = max(1, channels)
        guard sampleCount >= 0, sampleCount.isMultiple(of: channelCount) else {
            throw MediaIOError.audioEncodeFailed(url, "Sample count must contain complete interleaved frames.")
        }
        let bitsPerSample = 32
        let byteRate = max(1, sampleRate) * channelCount * bitsPerSample / 8
        let blockAlign = channelCount * bitsPerSample / 8
        let (dataSize, overflow) = sampleCount.multipliedReportingOverflow(
            by: MemoryLayout<Float>.size
        )
        guard !overflow, dataSize <= Int(UInt32.max) - 36 else {
            throw MediaIOError.audioEncodeFailed(url, "Float WAV output exceeds the RIFF size limit.")
        }
        let riffSize = 36 + dataSize

        var header = Data()
        header.append(contentsOf: Array("RIFF".utf8))
        header.appendLE(UInt32(riffSize))
        header.append(contentsOf: Array("WAVE".utf8))
        header.append(contentsOf: Array("fmt ".utf8))
        header.appendLE(UInt32(16))
        header.appendLE(UInt16(3))
        header.appendLE(UInt16(channelCount))
        header.appendLE(UInt32(max(1, sampleRate)))
        header.appendLE(UInt32(byteRate))
        header.appendLE(UInt16(blockAlign))
        header.appendLE(UInt16(bitsPerSample))
        header.append(contentsOf: Array("data".utf8))
        header.appendLE(UInt32(dataSize))

        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        do {
            try handle.write(contentsOf: header)
            let requestedChunk = max(channelCount, chunkSampleCount)
            let alignedChunk = max(channelCount, (requestedChunk / channelCount) * channelCount)
            var lowerBound = 0
            while lowerBound < sampleCount {
                let upperBound = min(sampleCount, lowerBound + alignedChunk)
                let range = lowerBound..<upperBound
                let samples = try sampleProvider(range)
                guard samples.count == range.count else {
                    throw MediaIOError.invalidBufferSize(
                        expected: range.count * MemoryLayout<Float>.size,
                        actual: samples.count * MemoryLayout<Float>.size
                    )
                }
                let bytes = samples.withUnsafeBytes { Data($0) }
                try handle.write(contentsOf: bytes)
                lowerBound = upperBound
            }
            try handle.close()
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    public static func transcode(
        _ inputURL: URL,
        to outputURL: URL,
        format: String
    ) throws {
        try FFmpegMediaIO.transcodeAudio(inputURL, to: outputURL, format: format)
    }
}

extension Data {
    mutating func appendLE(_ value: UInt16) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }

    mutating func appendLE(_ value: UInt32) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
