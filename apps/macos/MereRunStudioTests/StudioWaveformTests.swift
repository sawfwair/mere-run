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
}
