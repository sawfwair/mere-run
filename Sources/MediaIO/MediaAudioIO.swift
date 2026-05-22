import Foundation

public enum MediaAudioIO {
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

    public static func writeFloatWAV(
        samples: [Float],
        sampleRate: Int,
        channels: Int,
        to url: URL
    ) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var data = Data()
        let channelCount = max(1, channels)
        let bitsPerSample = 32
        let byteRate = max(1, sampleRate) * channelCount * bitsPerSample / 8
        let blockAlign = channelCount * bitsPerSample / 8
        let dataSize = samples.count * MemoryLayout<Float>.size
        let riffSize = 36 + dataSize

        data.append(contentsOf: Array("RIFF".utf8))
        data.appendLE(UInt32(riffSize))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        data.appendLE(UInt32(16))
        data.appendLE(UInt16(3))
        data.appendLE(UInt16(channelCount))
        data.appendLE(UInt32(max(1, sampleRate)))
        data.appendLE(UInt32(byteRate))
        data.appendLE(UInt16(blockAlign))
        data.appendLE(UInt16(bitsPerSample))
        data.append(contentsOf: Array("data".utf8))
        data.appendLE(UInt32(dataSize))
        samples.withUnsafeBufferPointer { ptr in
            data.append(contentsOf: UnsafeRawBufferPointer(ptr))
        }
        try data.write(to: url)
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
