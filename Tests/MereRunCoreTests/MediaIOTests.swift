import AudioCodecs
import Foundation
import MediaIO
import XCTest

final class MediaIOTests: XCTestCase {
    func testImageResizeAndCenterCropUseStableRGBAStorage() throws {
        let image = try MediaImage(
            width: 2,
            height: 2,
            rgba8: [
                255, 0, 0, 255,
                0, 255, 0, 255,
                0, 0, 255, 255,
                255, 255, 255, 255,
            ]
        )

        let resized = try MediaImageIO.resized(image, width: 4, height: 4)
        XCTAssertEqual(resized.width, 4)
        XCTAssertEqual(resized.height, 4)
        XCTAssertEqual(resized.rgba8.count, 4 * 4 * 4)

        let cropped = try MediaImageIO.centerCropped(image, width: 1, height: 1)
        XCTAssertEqual(cropped.width, 1)
        XCTAssertEqual(cropped.height, 1)
        XCTAssertEqual(cropped.rgba8.count, 4)
    }

    func testFloatWAVWriterProducesPortableHeader() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try MediaAudioIO.writeFloatWAV(
            samples: [0.0, 0.5, -0.5, 1.0],
            sampleRate: 16_000,
            channels: 1,
            to: tempURL
        )

        let data = try Data(contentsOf: tempURL)
        XCTAssertEqual(String(decoding: data[0..<4], as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: data[8..<12], as: UTF8.self), "WAVE")
        XCTAssertEqual(readUInt16LE(data, offset: 20), 3)
        XCTAssertEqual(readUInt32LE(data, offset: 24), 16_000)
        XCTAssertEqual(readUInt32LE(data, offset: 40), 4 * UInt32(MemoryLayout<Float>.size))
    }

    func testAudioTranscodeWritesRequestedContainer() throws {
        guard isExecutableAvailable(MediaTool.ffmpegPath) else {
            throw XCTSkip("ffmpeg is not available")
        }
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-transcode-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let wavURL = tempDir.appendingPathComponent("input.wav")
        let mp3URL = tempDir.appendingPathComponent("output.mp3")
        let samples = (0..<4_000).map { index in
            sin(Float(index) * 0.05) * 0.2
        }
        try MediaAudioIO.writeFloatWAV(samples: samples, sampleRate: 16_000, channels: 1, to: wavURL)
        try MediaAudioIO.transcode(wavURL, to: mp3URL, format: "mp3")

        let data = try Data(contentsOf: mp3URL)
        XCTAssertGreaterThan(data.count, 0)
    }

    func testRealFFTPlanSupportsASRAndPowerOfTwoFrameSizes() throws {
        for size in [400, 512, 1024] {
            let plan = try RealFFTPlan(size: size)
            let samples = (0..<size).map { index in
                sin(Float(index) * 0.01)
            }
            let spectrum = plan.powerSpectrum(samples)
            XCTAssertEqual(spectrum.count, (size / 2) + 1)
            XCTAssertTrue(spectrum.allSatisfy { $0.isFinite && $0 >= 0 })
        }
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

    private func isExecutableAvailable(_ tool: String) -> Bool {
        if tool.contains("/") {
            return FileManager.default.isExecutableFile(atPath: tool)
        }
        let pathEntries = ProcessInfo.processInfo.environment["PATH"]?.split(separator: ":") ?? []
        return pathEntries.contains { entry in
            let candidate = URL(fileURLWithPath: String(entry), isDirectory: true)
                .appendingPathComponent(tool)
            return FileManager.default.isExecutableFile(atPath: candidate.path)
        }
    }
}
