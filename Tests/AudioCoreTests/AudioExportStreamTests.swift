import Foundation
import XCTest
@testable import AudioCore

final class AudioExportStreamTests: XCTestCase {
    func testIncrementalEncodingMatchesWholeFileAcrossFormatsAndChunkBoundaries() throws {
        let directory = try directory()
        let samples: [Float] = [-1, -0.75, -0.5, 0, 0.25, 0.5, 1]
        for format in AudioWAVEncoding.allCases {
            let plan = try AudioExportPlan(options: AudioExportOptions(
                format: format, normalization: .none, targetPeakDB: 0,
                fadeInMilliseconds: 0, fadeOutMilliseconds: 0, dither: true
            ))
            let waveform = try AudioWaveform(interleaved: samples, channels: 1, sampleRate: 24_000)
            let expected = try AudioExportService.data(waveform, plan: plan)
            let output = directory.appendingPathComponent("\(format.rawValue).wav")
            let writer = try AudioExportStream(plan: plan, to: output, sampleRate: 24_000)
            try writer.append(samples: Array(samples[..<1]))
            try writer.append(samples: Array(samples[1..<4]))
            try writer.append(samples: Array(samples[4...]))
            let result = try writer.finish()
            XCTAssertEqual(try Data(contentsOf: output), expected.data)
            XCTAssertEqual(result.byteCount, expected.data.count)
            XCTAssertEqual(result.statistics.sampleCount, samples.count)
            XCTAssertEqual(result.statistics.outputRMS, expected.statistics.outputRMS, accuracy: 1e-12)
            XCTAssertThrowsError(try writer.finish())
            XCTAssertThrowsError(try writer.append(samples: [0]))
        }
    }

    func testFloatHeadroomAndNonfinitePolicyAreExplicit() throws {
        let directory = try directory()
        let output = directory.appendingPathComponent("float.wav")
        let plan = try AudioExportPlan(options: .speechStreaming, clipping: .preserveFloatHeadroom)
        let writer = try AudioExportStream(plan: plan, to: output, sampleRate: 24_000)
        try writer.append(samples: [2, -2, .nan, .infinity, 0.25])
        let result = try writer.finish()
        let data = try Data(contentsOf: output)
        XCTAssertEqual(data[20], 3)
        XCTAssertEqual(data[34], 32)
        XCTAssertEqual((0..<5).map { Float(bitPattern: read32(data, 44 + $0 * 4)) }, [2, -2, 0, 0, 0.25])
        XCTAssertEqual(result.statistics.replacedNonfiniteSamples, 2)
        XCTAssertEqual(result.statistics.clippedSamples, 0)
        XCTAssertEqual(result.statistics.outputPeak, 2)
        XCTAssertThrowsError(try AudioExportPlan(options: .referencePCM16, clipping: .preserveFloatHeadroom))
        let defaultFloat = try AudioExportService.data(
            AudioWaveform(interleaved: [2, -2], channels: 1, sampleRate: 24_000),
            plan: AudioExportPlan(options: .speechStreaming)
        )
        XCTAssertEqual(defaultFloat.statistics.clippedSamples, 2)
    }

    func testOnlyFinishedStreamsReplaceTheDestination() throws {
        let directory = try directory()
        let output = directory.appendingPathComponent("existing.wav")
        let original = Data("previous audio".utf8)
        try original.write(to: output)
        let plan = try AudioExportPlan(options: .speechStreaming)
        let discarded = try AudioExportStream(plan: plan, to: output, sampleRate: 24_000)
        try discarded.append(samples: [0.5])
        XCTAssertEqual(try Data(contentsOf: output), original)
        discarded.cancel()
        XCTAssertEqual(try Data(contentsOf: output), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["existing.wav"])
        let completed = try AudioExportStream(plan: plan, to: output, sampleRate: 24_000)
        try completed.append(samples: [0.5])
        _ = try completed.finish()
        XCTAssertEqual(try Data(contentsOf: output).count, 48)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["existing.wav"])
    }

    func testCancelledFinishPreservesExistingOutputAndRemovesTemporaryFile() async throws {
        let directory = try directory()
        let output = directory.appendingPathComponent("existing.wav")
        let original = Data("previous audio".utf8)
        try original.write(to: output)
        let task = Task {
            let writer = try AudioExportStream(plan: AudioExportPlan(options: .speechStreaming), to: output, sampleRate: 24_000)
            try writer.append(samples: [0.5])
            withUnsafeCurrentTask { $0?.cancel() }
            XCTAssertThrowsError(try writer.finish()) { XCTAssertTrue($0 is CancellationError) }
        }
        try await task.value
        XCTAssertEqual(try Data(contentsOf: output), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["existing.wav"])
    }

    func testInvalidChunkDiscardsTheStreamAndPreservesExistingOutput() throws {
        let directory = try directory()
        let output = directory.appendingPathComponent("existing.wav")
        let original = Data("previous audio".utf8)
        try original.write(to: output)
        let writer = try AudioExportStream(plan: AudioExportPlan(options: .speechStreaming),
                                           to: output, sampleRate: 24_000, channels: 2)
        try writer.append(samples: [0.25, -0.25])
        XCTAssertThrowsError(try writer.append(samples: [0.5]))
        XCTAssertThrowsError(try writer.finish())
        XCTAssertEqual(try Data(contentsOf: output), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["existing.wav"])
    }

    func testUnsupportedStreamProcessingAndInvalidFormatsDoNotCreateFiles() throws {
        let directory = try directory()
        let output = directory.appendingPathComponent("invalid.wav")
        XCTAssertThrowsError(try AudioExportStream(plan: AudioExportPlan(options: .music), to: output, sampleRate: 24_000))
        let plan = try AudioExportPlan(options: .speechStreaming)
        XCTAssertThrowsError(try AudioExportStream(plan: plan, to: output, sampleRate: Int.max))
        XCTAssertThrowsError(try AudioExportStream(plan: plan, to: output, sampleRate: 24_000, channels: 0))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func read32(_ data: Data, _ offset: Int) -> UInt32 {
        (0..<4).reduce(0) { $0 | UInt32(data[offset + $1]) << ($1 * 8) }
    }
}
