import Foundation

public final class StreamingWAVWriter {
    public enum Error: LocalizedError {
        case invalidFormat
        case appendAfterClose

        public var errorDescription: String? {
            switch self {
            case .invalidFormat:
                return "Failed to create audio format for streaming WAV output."
            case .appendAfterClose:
                return "Cannot append samples after the WAV stream has been closed."
            }
        }
    }

    private let fileHandle: FileHandle
    private let sampleRate: Int
    private let channels: Int
    private var sampleCount = 0
    private var didClose = false

    public init(outputURL: URL, sampleRate: Int, channels: Int = 1) throws {
        guard sampleRate > 0, channels > 0 else {
            throw Error.invalidFormat
        }

        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        self.fileHandle = try FileHandle(forWritingTo: outputURL)
        self.sampleRate = sampleRate
        self.channels = channels
        try writeHeader(dataByteCount: 0)
    }

    deinit {
        try? close()
    }

    public func append(samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        guard !didClose else {
            throw Error.appendAfterClose
        }

        var data = Data()
        data.reserveCapacity(samples.count * MemoryLayout<Float>.size)
        for sample in samples {
            data.append(contentsOf: sample.littleEndianBytes)
        }
        try fileHandle.write(contentsOf: data)
        sampleCount += samples.count
    }

    public func close() throws {
        guard !didClose else { return }
        didClose = true
        let dataByteCount = sampleCount * MemoryLayout<Float>.size
        try fileHandle.seek(toOffset: 0)
        try writeHeader(dataByteCount: dataByteCount)
        try fileHandle.close()
    }

    private func writeHeader(dataByteCount: Int) throws {
        let byteRate = sampleRate * channels * MemoryLayout<Float>.size
        let blockAlign = channels * MemoryLayout<Float>.size
        let chunkSize = 36 + dataByteCount

        var header = Data()
        header.append(contentsOf: Array("RIFF".utf8))
        header.appendLE(UInt32(chunkSize))
        header.append(contentsOf: Array("WAVE".utf8))
        header.append(contentsOf: Array("fmt ".utf8))
        header.appendLE(UInt32(16))
        header.appendLE(UInt16(3))
        header.appendLE(UInt16(channels))
        header.appendLE(UInt32(sampleRate))
        header.appendLE(UInt32(byteRate))
        header.appendLE(UInt16(blockAlign))
        header.appendLE(UInt16(MemoryLayout<Float>.size * 8))
        header.append(contentsOf: Array("data".utf8))
        header.appendLE(UInt32(dataByteCount))
        try fileHandle.write(contentsOf: header)
    }
}

private extension Float {
    var littleEndianBytes: [UInt8] {
        var bitPattern = self.bitPattern.littleEndian
        return withUnsafeBytes(of: &bitPattern) { Array($0) }
    }
}

private extension Data {
    mutating func appendLE(_ value: UInt16) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }

    mutating func appendLE(_ value: UInt32) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }
}
