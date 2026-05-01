import AVFoundation
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

        let file = try AVAudioFile(forReading: outputURL)
        XCTAssertEqual(Int(file.processingFormat.sampleRate), 24_000)
        XCTAssertEqual(Int(file.length), 5)
    }
}
