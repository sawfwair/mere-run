import ArgumentParser
import Foundation
import MLX
import XCTest
@testable import MereRunCLI

final class AudioExportFadeTests: XCTestCase {
    func testOrdinaryStereoFadesPreserveSamplesAndChannelOrder() throws {
        let samples: [Float] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        XCTAssertEqual(
            try decoded(samples, channels: 2, fadeIn: 5),
            [0, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1]
        )
        XCTAssertEqual(
            try decoded(samples, channels: 2, fadeOut: 5),
            [1, -1, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 0, 0]
        )
        XCTAssertEqual(
            try decoded(samples, channels: 2, fadeIn: 5, fadeOut: 5),
            [0, 0, 0.1875, -0.1875, 0.25, -0.25, 0.1875, -0.1875, 0, 0]
        )
    }

    func testFractionalFadeFramesTruncateAndZeroFadesPreserveAudio() throws {
        let samples: [Float] = [1, 1, 1, 1, 1]
        XCTAssertEqual(try decoded(samples), samples)
        XCTAssertEqual(try decoded(samples, fadeIn: 0.9), samples)
        XCTAssertEqual(try decoded(samples, fadeIn: 3.9), [0, 0.5, 1, 1, 1])
        XCTAssertEqual(try decoded(samples, fadeOut: 3.9), [1, 1, 1, 0.5, 0])
    }

    func testPCMEncodingPreservesSignedSampleValues() throws {
        let samples: [Float] = [-1, -0.5, 0, 0.5, 1]
        for (format, expected) in [
            (ACEStepAudioFormat.pcm16, [Int32(-32_767), -16_383, 0, 16_383, 32_767]),
            (.pcm24, [-8_388_607, -4_194_303, 0, 4_194_303, 8_388_607]),
        ] {
            let data = try ACEStepWAVWriter.wavData(
                MLXArray(samples), sampleRate: 1_000,
                options: .init(format: format, normalization: .none,
                               fadeInMilliseconds: 0, fadeOutMilliseconds: 0, dither: false)
            )
            let width = Int(format.bitsPerSample / 8)
            let values = stride(from: 44, to: data.count, by: width).map { offset -> Int32 in
                var value: Int32 = 0
                for byte in 0..<width {
                    value |= Int32(data[offset + byte]) << (8 * byte)
                }
                let shift = 32 - 8 * width
                return (value << shift) >> shift
            }
            XCTAssertEqual(values, expected)
        }
    }

    func testHugeFiniteFadesBoundBeforeIntegerConversion() throws {
        let samples: [Float] = [1, 1, 1, 1, 1]
        for duration: Float in [1e20, .greatestFiniteMagnitude] {
            XCTAssertEqual(try decoded(samples, fadeIn: duration), [0, 0.25, 0.5, 0.75, 1])
            XCTAssertEqual(try decoded(samples, fadeOut: duration), [1, 0.75, 0.5, 0.25, 0])
            XCTAssertEqual(
                try decoded(samples, fadeIn: duration, fadeOut: duration),
                [0, 0.1875, 0.25, 0.1875, 0]
            )
        }
    }

    func testEmptyAndSingleFrameAudioWithHugeFades() throws {
        XCTAssertEqual(try decoded([], fadeIn: .greatestFiniteMagnitude), [])
        XCTAssertEqual(try decoded([], fadeOut: .greatestFiniteMagnitude), [])
        XCTAssertEqual(try decoded([0.5]), [0.5])
        XCTAssertEqual(try decoded([0.5], fadeIn: .greatestFiniteMagnitude), [0])
        XCTAssertEqual(try decoded([0.5], fadeOut: .greatestFiniteMagnitude), [0])
    }

    func testWriterRejectsInvalidFadeDurations() {
        for duration: Float in [-1, .nan, .infinity, -.infinity] {
            for options in [
                ACEStepAudioExportOptions(fadeInMilliseconds: duration),
                ACEStepAudioExportOptions(fadeOutMilliseconds: duration),
            ] {
                XCTAssertThrowsError(try ACEStepWAVWriter.wavData(
                    MLXArray([Float(0.5)]), sampleRate: 1_000, options: options
                )) { error in
                    guard case ACEStepWAVWriter.WriterError.invalidFade = error else {
                        return XCTFail("Expected invalid fade, got \(error)")
                    }
                }
            }
        }
    }

    func testWriterRejectsNonfinitePeakTargets() {
        for peak: Float in [.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try ACEStepWAVWriter.wavData(
                MLXArray([Float(0.5)]), sampleRate: 1_000,
                options: .init(targetPeakDB: peak)
            )) { error in
                guard case ACEStepWAVWriter.WriterError.invalidPeak = error else {
                    return XCTFail("Expected invalid peak, got \(error)")
                }
            }
        }
    }

    func testCLIRejectsInvalidExportSettingsBeforeOpeningModel() async throws {
        for flag in ["--fade-in-ms", "--fade-out-ms", "--target-peak-db"] {
            let values = flag == "--target-peak-db" ? ["nan", "inf", "-inf"] : ["nan", "inf", "-inf", "-1"]
            for value in values {
                let command = try MusicGenerate.parse([
                    "instrumental guitar", "--model", "/missing/export-fade-regression-model",
                    "\(flag)=\(value)",
                ])
                do {
                    try await command.run()
                    XCTFail("Expected rejection for \(flag)=\(value)")
                } catch let error as ValidationError {
                    XCTAssertEqual(
                        error.message,
                        flag == "--target-peak-db"
                            ? "--target-peak-db must be finite and <= 0"
                            : "Output fades must be finite and >= 0"
                    )
                }
            }
        }
    }

    private func decoded(
        _ samples: [Float], channels: Int = 1,
        fadeIn: Float = 0, fadeOut: Float = 0
    ) throws -> [Float] {
        let data = try ACEStepWAVWriter.wavData(
            MLXArray(samples, [1, samples.count / channels, channels]),
            sampleRate: 1_000,
            options: .init(format: .float32, normalization: .none,
                           fadeInMilliseconds: fadeIn, fadeOutMilliseconds: fadeOut, dither: false)
        )
        return stride(from: 44, to: data.count, by: 4).map { offset in
            let bits = (0..<4).reduce(UInt32.zero) { result, byte in
                result | (UInt32(data[offset + byte]) << (8 * byte))
            }
            return Float(bitPattern: bits)
        }
    }
}
