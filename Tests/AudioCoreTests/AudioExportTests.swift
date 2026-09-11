import Foundation
import XCTest
@testable import AudioCore

final class AudioExportTests: XCTestCase {
    func testCancelledExportPreservesDestinationAndRemovesTemporaryFiles() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("output.wav")
        let original = Data("existing output".utf8)
        try original.write(to: destination)
        let audio = try processed([], fadeIn: 0)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            XCTAssertThrowsError(try AudioWAVEncoder.write(audio, to: destination)) { error in
                XCTAssertTrue(error is CancellationError)
            }
            XCTAssertThrowsError(try AudioWAVEncoder.data(audio))
            XCTAssertThrowsError(try AudioExportProcessor.process(audio.waveform, plan: audio.plan))
        }
        try await task.value
        XCTAssertEqual(try Data(contentsOf: destination), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["output.wav"])
    }

    func testOneFrameResamplingKeepsExactFirstFrame() throws {
        let waveform = try AudioWaveform(interleaved: [.greatestFiniteMagnitude, -.greatestFiniteMagnitude], channels: 1, sampleRate: 2)
        XCTAssertEqual(try waveform.resampled(to: 1).samples, [.greatestFiniteMagnitude])
    }

    func testOrdinaryStereoFadesAndOverlaps() throws {
        let samples: [Float] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        XCTAssertEqual(try processed(samples, channels: 2, fadeIn: 5).waveform.samples,
                       [0, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1])
        XCTAssertEqual(try processed(samples, channels: 2, fadeOut: 5).waveform.samples,
                       [1, -1, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 0, 0])
        XCTAssertEqual(try processed(samples, channels: 2, fadeIn: 5, fadeOut: 5).waveform.samples,
                       [0, 0, 0.1875, -0.1875, 0.25, -0.25, 0.1875, -0.1875, 0, 0])
    }

    func testFractionalHugeAndEmptyFades() throws {
        let samples: [Float] = [1, 1, 1, 1, 1]
        XCTAssertEqual(try processed(samples, fadeIn: 0.9).waveform.samples, samples)
        XCTAssertEqual(try processed(samples, fadeIn: 3.9).waveform.samples, [0, 0.5, 1, 1, 1])
        for duration: Float in [1e20, .greatestFiniteMagnitude] {
            XCTAssertEqual(try processed(samples, fadeIn: duration).waveform.samples, [0, 0.25, 0.5, 0.75, 1])
            XCTAssertEqual(try processed(samples, fadeOut: duration).waveform.samples, [1, 0.75, 0.5, 0.25, 0])
            XCTAssertEqual(try processed([], fadeIn: duration, fadeOut: duration).waveform.samples, [])
            XCTAssertEqual(try processed([0.5], fadeIn: duration).waveform.samples, [0])
        }
    }

    func testInvalidPlansAndIncompleteFramesAreRejected() throws {
        for value: Float in [-1, .nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try AudioExportPlan(options: .init(fadeInMilliseconds: value)))
            XCTAssertThrowsError(try AudioExportPlan(options: .init(fadeOutMilliseconds: value)))
        }
        for value: Float in [1, .nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try AudioExportPlan(options: .init(targetPeakDB: value)))
        }
        XCTAssertThrowsError(try AudioWaveform(interleaved: [1], channels: 2, sampleRate: 48_000))
        XCTAssertThrowsError(try AudioWaveform(interleaved: [], channels: 0, sampleRate: 48_000))
        XCTAssertThrowsError(try AudioWaveform(interleaved: [], channels: 1, sampleRate: 0))
    }

    func testWireOverridesPreserveDistinctDefaults() throws {
        let overrides = try JSONDecoder().decode(AudioExportOverrides.self, from: Data(#"{"format":"pcm24","fade_out_ms":7}"#.utf8))
        let music = try overrides.resolve(defaults: .music)
        let reference = try overrides.resolve(defaults: .referencePCM16)
        XCTAssertEqual(music.options.normalization, .peak)
        XCTAssertEqual(music.options.fadeInMilliseconds, 5)
        XCTAssertTrue(music.options.dither)
        XCTAssertEqual(reference.options.format, .pcm24)
        XCTAssertEqual(reference.options.normalization, .none)
        XCTAssertEqual(reference.options.fadeInMilliseconds, 0)
        XCTAssertEqual(reference.options.fadeOutMilliseconds, 7)
        XCTAssertFalse(reference.options.dither)
    }

    func testPCMHeadersSignedSamplesAndOddChunkPadding() throws {
        for (format, values) in [(AudioWAVEncoding.pcm16, [Int32(-32_767), -16_383, 0, 16_383, 32_767]),
                                 (.pcm24, [-8_388_607, -4_194_303, 0, 4_194_303, 8_388_607])] {
            let plan = try AudioExportPlan(options: .init(format: format, normalization: .none,
                                                         fadeInMilliseconds: 0, fadeOutMilliseconds: 0, dither: false))
            let result = try AudioExportService.data(waveform([-1, -0.5, 0, 0.5, 1]), plan: plan)
            let data = result.data, width = Int(format.bitsPerSample / 8)
            XCTAssertEqual(read32(data, 4), UInt32(data.count - 8))
            XCTAssertEqual(read32(data, 40), UInt32(values.count * width))
            XCTAssertEqual(data.count % 2, 0)
            let actual = (0..<values.count).map { index -> Int32 in
                var value: Int32 = 0
                for byte in 0..<width { value |= Int32(data[44 + index * width + byte]) << (byte * 8) }
                let shift = 32 - 8 * width
                return (value << shift) >> shift
            }
            XCTAssertEqual(actual, values)
            if format == .pcm24 { XCTAssertEqual(data.last, 0) }
        }
    }

    func testFloatWAVContainsProcessedSamples() throws {
        let audio = try processed([0.25, -0.5, 0.75])
        let data = try AudioWAVEncoder.data(audio).data
        XCTAssertEqual(read32(data, 40), 12)
        XCTAssertEqual(data[20], 3)
        XCTAssertEqual(data[34], 32)
        XCTAssertEqual((0..<3).map { Float(bitPattern: read32(data, 44 + $0 * 4)) }, audio.waveform.samples)
    }

    func testStatisticsAndSubnormalNormalizationRemainFinite() throws {
        let result = try processed([.nan, .infinity, -.infinity, 2, -2, 0.5])
        XCTAssertEqual(result.waveform.samples, [0, 0, 0, 1, -1, 0.5])
        XCTAssertEqual(result.statistics.replacedNonfiniteSamples, 3)
        XCTAssertEqual(result.statistics.clippedSamples, 2)
        XCTAssertEqual(result.statistics.inputPeak, 2)
        XCTAssertEqual(result.statistics.outputPeak, 1)
        let normalized = try AudioExportProcessor.process(waveform([.leastNonzeroMagnitude, 0]), plan: AudioExportPlan(options: .init(
            targetPeakDB: 0, fadeInMilliseconds: 0, fadeOutMilliseconds: 0, dither: false)))
        XCTAssertEqual(normalized.waveform.samples, [1, 0])
        XCTAssertTrue(normalized.statistics.normalizationGain.isFinite)
        _ = try JSONEncoder().encode(normalized.statistics)
    }

    func testStereoResamplingPreservesFrameRoundingAndChannelOrder() throws {
        let input = try AudioWaveform(interleaved: [0, 10, 1, 11, 2, 12, 3, 13], channels: 2, sampleRate: 4)
        let output = try input.resampled(to: 2)
        XCTAssertEqual(output.samples, [0, 10, 2, 12])
        XCTAssertEqual(output.frameCount, 2)
        XCTAssertThrowsError(try input.resampled(to: 0))
        let extreme = try AudioWaveform(interleaved: [0, 1], channels: 1, sampleRate: 1)
        XCTAssertThrowsError(try extreme.resampled(to: Int(UInt32.max)))
    }

    func testWAVArithmeticRejectsOverflowWithoutLargeAllocations() {
        XCTAssertThrowsError(try AudioWAVLayout(sampleCount: Int.max, channels: 2, sampleRate: 48_000, encoding: .pcm24))
        XCTAssertThrowsError(try AudioWAVLayout(sampleCount: 1, channels: 8, sampleRate: Int.max, encoding: .float32))
        XCTAssertThrowsError(try AudioWAVLayout(sampleCount: Int(UInt32.max), channels: 1, sampleRate: 1, encoding: .pcm16))
    }

    func testChunkedFileEncodingMatchesMemoryAndReplacesAtomically() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let destination = folder.appendingPathComponent("audio.wav")
        for format in AudioWAVEncoding.allCases {
            let plan = try AudioExportPlan(options: .init(format: format))
            let audio = try AudioExportProcessor.process(waveform((0..<40_001).map { Float($0 % 100) / 100 }), plan: plan)
            let memory = try AudioWAVEncoder.data(audio)
            try Data("previous artifact".utf8).write(to: destination)
            let file = try AudioWAVEncoder.write(audio, to: destination)
            XCTAssertEqual(try Data(contentsOf: file.url), memory.data)
            XCTAssertEqual(file.byteCount, memory.data.count)
            XCTAssertEqual(file.statistics, memory.statistics)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), ["audio.wav"])
        }
    }

    private func waveform(_ samples: [Float], channels: Int = 1) throws -> AudioWaveform {
        try AudioWaveform(interleaved: samples, channels: channels, sampleRate: 1_000)
    }
    private func processed(_ samples: [Float], channels: Int = 1, fadeIn: Float = 0, fadeOut: Float = 0) throws -> ProcessedAudio {
        try AudioExportProcessor.process(waveform(samples, channels: channels), plan: AudioExportPlan(options: .init(
            format: .float32, normalization: .none, fadeInMilliseconds: fadeIn, fadeOutMilliseconds: fadeOut, dither: false)))
    }
    private func read32(_ data: Data, _ offset: Int) -> UInt32 {
        (0..<4).reduce(UInt32.zero) { $0 | (UInt32(data[offset + $1]) << ($1 * 8)) }
    }
}
