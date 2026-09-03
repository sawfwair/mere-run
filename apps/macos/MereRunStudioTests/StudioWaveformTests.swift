import AVFoundation
import XCTest
@testable import MereRunApp

final class StudioWaveformTests: XCTestCase {
    /// Writes a mono float WAV whose first half is a loud sine and second half is silence,
    /// then asserts the downsampled peaks mirror that shape after normalization.
    func testPeaksReflectLoudThenSilentHalves() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("waveform-test-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let sampleRate = 44_100.0
        let frames: AVAudioFrameCount = 8_820
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1))
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        let channel = try XCTUnwrap(buffer.floatChannelData)[0]
        for frame in 0..<Int(frames) {
            if frame < Int(frames) / 2 {
                channel[frame] = 0.9 * sin(2 * .pi * 440 * Float(frame) / Float(sampleRate))
            } else {
                channel[frame] = 0
            }
        }
        buffer.frameLength = frames
        try file.write(from: buffer)
        // Finalize the WAV header; without closing, the file reads back as empty.
        file.close()

        let barCount = 10
        let peaks = try XCTUnwrap(StudioWaveformLoader.peaks(url: url, barCount: barCount))

        XCTAssertEqual(peaks.count, barCount)
        XCTAssertEqual(peaks.max(), 1.0)
        for bar in 0..<4 {
            XCTAssertGreaterThan(peaks[bar], 0.7, "bar \(bar) should carry the sine")
        }
        for bar in 6..<barCount {
            XCTAssertLessThan(peaks[bar], 0.05, "bar \(bar) should be silence")
        }
    }

    func testPeaksReturnsNilForMissingFile() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).wav")
        XCTAssertNil(StudioWaveformLoader.peaks(url: url))
    }

    func testPeaksRejectsZeroBarCount() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("waveform-zero-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1))
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 441))
        buffer.frameLength = 441
        try file.write(from: buffer)
        file.close()

        XCTAssertNil(StudioWaveformLoader.peaks(url: url, barCount: 0))
    }

    /// The realtime recorder leaves the RIFF/data sizes at zero until it closes the file, which
    /// makes `AVAudioFile` see no frames. The growing-file reader ignores the declared sizes.
    func testGrowingWAVPeaksReadPastAZeroLengthHeader() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("waveform-growing-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = Data()
        func appendLE32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func appendLE16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8))
        appendLE32(36)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        appendLE32(16)
        appendLE16(3) // IEEE float
        appendLE16(1)
        appendLE32(48_000)
        appendLE32(48_000 * 4)
        appendLE16(4)
        appendLE16(32)
        data.append(contentsOf: Array("data".utf8))
        appendLE32(0) // not yet patched by the writer
        let frames = 4_000
        for frame in 0..<frames {
            let value: Float = frame < frames / 2 ? 0.8 * sin(Float(frame) * 0.3) : 0
            withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)

        XCTAssertNil(StudioWaveformLoader.peaks(url: url), "AVAudioFile should see zero frames")
        let peaks = try XCTUnwrap(StudioWaveformLoader.growingWAVPeaks(url: url, barCount: 8))
        XCTAssertEqual(peaks.count, 8)
        XCTAssertEqual(peaks.max(), 1.0)
        for bar in 0..<3 {
            XCTAssertGreaterThan(peaks[bar], 0.7, "bar \(bar) should carry the sine")
        }
        for bar in 5..<8 {
            XCTAssertEqual(peaks[bar], 0, "bar \(bar) should be silence")
        }
    }

    func testGrowingWAVPeaksRejectsNonWAVData() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("waveform-notwav-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 7, count: 256).write(to: url)
        XCTAssertNil(StudioWaveformLoader.growingWAVPeaks(url: url))
    }
}
