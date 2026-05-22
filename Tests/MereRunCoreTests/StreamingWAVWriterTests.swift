import XCTest
@testable import MereRunCore
import AudioCodecs

final class StreamingWAVWriterTests: XCTestCase {
    func testWritesIncrementalFloatChunks() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("streaming-wav-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let outputURL = tempDir.appendingPathComponent("out.wav")
        do {
            let writer = try StreamingWAVWriter(outputURL: outputURL, sampleRate: 24_000)
            try writer.append(samples: [0.1, 0.2, 0.3])
            try writer.append(samples: [0.4, 0.5])
        }

        let data = try Data(contentsOf: outputURL)
        XCTAssertEqual(String(decoding: data[0..<4], as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: data[8..<12], as: UTF8.self), "WAVE")
        XCTAssertEqual(readUInt16LE(data, offset: 20), 3)
        XCTAssertEqual(readUInt16LE(data, offset: 22), 1)
        XCTAssertEqual(readUInt32LE(data, offset: 24), 24_000)
        XCTAssertEqual(readUInt32LE(data, offset: 40), 5 * UInt32(MemoryLayout<Float>.size))
    }

    private func readUInt16LE(_ data: Data, offset: Int) -> UInt16 {
        let bytes = [UInt8](data[offset..<(offset + 2)])
        return UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)
    }

    private func readUInt32LE(_ data: Data, offset: Int) -> UInt32 {
        let bytes = [UInt8](data[offset..<(offset + 4)])
        return UInt32(bytes[0])
            | (UInt32(bytes[1]) << 8)
            | (UInt32(bytes[2]) << 16)
            | (UInt32(bytes[3]) << 24)
    }
}
