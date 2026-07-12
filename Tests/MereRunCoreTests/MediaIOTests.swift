import AudioCodecs
import Foundation
@testable import MediaIO
import XCTest

final class MediaIOTests: XCTestCase {
    private enum ExtractionRejection: Error, Equatable {
        case rejected
    }

    func testVideoFrameRateResolverRejectsNonFiniteValues() {
        for fps in [Double.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(
                try MediaVideoFrameRateResolver.resolve(fps, fallbackFPS: 30)
            ) { error in
                guard case .invalidVideoFrameRate(let actual) = error as? MediaIOError else {
                    return XCTFail("Unexpected error: \(error)")
                }
                if fps.isNaN {
                    XCTAssertTrue(actual.isNaN)
                } else {
                    XCTAssertEqual(actual, fps)
                }
            }
        }
    }

    func testVideoFrameRateResolverUsesExplicitNonPositiveFallback() throws {
        for fps in [0.0, -1.0, -Double.greatestFiniteMagnitude] {
            let resolved = try MediaVideoFrameRateResolver.resolve(fps, fallbackFPS: 30)
            XCTAssertEqual(resolved.framesPerSecond, 30)
            XCTAssertEqual(resolved.timeScale, 30)
        }

        XCTAssertThrowsError(try MediaVideoFrameRateResolver.resolve(0)) { error in
            guard case .invalidVideoFrameRate(let actual) = error as? MediaIOError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(actual, 0)
        }
    }

    func testVideoFrameRateResolverClampsSubOneRate() throws {
        let resolved = try MediaVideoFrameRateResolver.resolve(0.25)
        XCTAssertEqual(resolved.framesPerSecond, 1)
        XCTAssertEqual(resolved.timeScale, 1)
    }

    func testVideoFrameRateResolverGuardsInt32RoundingBoundary() throws {
        let maximum = Double(Int32.max)
        let maximumResolved = try MediaVideoFrameRateResolver.resolve(maximum)
        XCTAssertEqual(maximumResolved.framesPerSecond, maximum)
        XCTAssertEqual(maximumResolved.timeScale, Int32.max)

        let stillRepresentable = try MediaVideoFrameRateResolver.resolve(maximum + 0.49)
        XCTAssertEqual(stillRepresentable.timeScale, Int32.max)

        for fps in [maximum + 0.5, Double(Float(Int32.max))] {
            XCTAssertThrowsError(try MediaVideoFrameRateResolver.resolve(fps)) { error in
                guard case .invalidVideoFrameRate = error as? MediaIOError else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
        }
    }

    #if canImport(AVFoundation) && canImport(CoreGraphics)
    func testAppleVideoWriterRejectsInvalidFPSBeforeReadingFrames() {
        let missingFrameURL = URL(fileURLWithPath: "/definitely-missing/frame.png")
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("invalid-fps-\(UUID().uuidString)")
            .appendingPathExtension("mp4")

        for fps in [Double.nan, .infinity, Double(Float(Int32.max))] {
            XCTAssertThrowsError(
                try AppleMediaVideoIO.writeVideo(
                    frameURLs: [missingFrameURL],
                    fps: fps,
                    to: outputURL
                )
            ) { error in
                guard case .invalidVideoFrameRate = error as? MediaIOError else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
        }
    }
    #endif

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

    func testFloatWAVWriterRequestsChannelAlignedChunks() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-streamed-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let expected: [Float] = [0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 0.125]
        var requestedRanges: [Range<Int>] = []
        try MediaAudioIO.writeFloatWAV(
            sampleCount: expected.count,
            sampleRate: 24_000,
            channels: 2,
            chunkSampleCount: 5,
            to: tempURL
        ) { range in
            requestedRanges.append(range)
            return Array(expected[range])
        }

        XCTAssertEqual(requestedRanges, [0..<4, 4..<8, 8..<10])
        let data = try Data(contentsOf: tempURL)
        XCTAssertEqual(data.count, 44 + expected.count * MemoryLayout<Float>.size)
        let decoded = expected.indices.map { index in
            Float(bitPattern: readUInt32LE(data, offset: 44 + index * MemoryLayout<Float>.size))
        }
        XCTAssertEqual(decoded, expected)
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

    func testVideoExtractionValidatesDecodedBudgetBeforeWritingFrames() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-video-limits-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let videoURL = root.appendingPathComponent("input.mp4")
        let framesURL = root.appendingPathComponent("frames", isDirectory: true)
        try MediaVideoIO.writeMP4(
            rgb24: [UInt8](repeating: 127, count: 2 * 2 * 3 * 2),
            width: 2,
            height: 2,
            frameCount: 2,
            fps: 2,
            to: videoURL
        )

        XCTAssertThrowsError(
            try MediaVideoIO.extractFrames(
                from: videoURL,
                into: framesURL,
                endFrame: 1,
                decodeLimits: MediaVideoDecodeLimits(
                    maximumPixelCountPerFrame: 64,
                    maximumAggregatePixelCount: 128
                ),
                validateDecodedSequence: { width, height, frameCount in
                    XCTAssertEqual(width, 2)
                    XCTAssertEqual(height, 2)
                    XCTAssertEqual(frameCount, 2)
                    throw ExtractionRejection.rejected
                }
            )
        ) { error in
            guard let rejection = error as? ExtractionRejection else {
                return XCTFail("Unexpected FFmpeg admission error: \(error)")
            }
            XCTAssertEqual(rejection, .rejected)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: framesURL.path))
    }

    func testFFmpegExtractionRunsFrameAdmissionBeforeCreatingOutputDirectory() throws {
        guard isExecutableAvailable(MediaTool.ffmpegPath),
              isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffmpeg and ffprobe are required")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-ffmpeg-admission-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let videoURL = root.appendingPathComponent("input.mp4")
        let framesURL = root.appendingPathComponent("frames", isDirectory: true)
        try FFmpegMediaIO.writeMP4(
            rgb24: [UInt8](repeating: 127, count: 2 * 2 * 3 * 2),
            width: 2,
            height: 2,
            frameCount: 2,
            fps: 2,
            to: videoURL
        )
        var validationCount = 0

        XCTAssertThrowsError(
            try FFmpegMediaIO.extractFrames(
                from: videoURL,
                into: framesURL,
                endFrame: 1,
                decodeLimits: MediaVideoDecodeLimits(
                    maximumPixelCountPerFrame: 64,
                    maximumAggregatePixelCount: 128
                ),
                validateDecodedSequence: { width, height, frameCount in
                    XCTAssertEqual(width, 2)
                    XCTAssertEqual(height, 2)
                    XCTAssertEqual(frameCount, 2)
                    validationCount += 1
                    if validationCount == 1 {
                        return
                    }
                    throw ExtractionRejection.rejected
                }
            )
        ) { error in
            guard let rejection = error as? ExtractionRejection else {
                return XCTFail("Unexpected FFmpeg admission error: \(error)")
            }
            XCTAssertEqual(rejection, .rejected)
        }
        XCTAssertEqual(validationCount, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: framesURL.path))
    }

    func testFFmpegFrameAdmissionUsesAllProbedFrameDimensions() throws {
        let metadata = """
        [Parsed_showinfo_0] n: 0 pts: 0 s:1920x1080
        [Parsed_showinfo_0] n: 1 pts: 1 s:1280x2160
        [Parsed_showinfo_0] n: 2 pts: 2 s:4096x720
        """

        let plan = try FFmpegMediaIO.frameAdmissionPlan(
            fromFFmpegShowInfo: metadata,
            maximumFrameCount: 3
        )

        XCTAssertEqual(plan.frameCount, 3)
        XCTAssertEqual(plan.maximumWidth, 4096)
        XCTAssertEqual(plan.maximumHeight, 2160)
    }

    func testFFmpegFrameAdmissionAllowsAutorotatedOutput() throws {
        let metadata = "[Parsed_showinfo_0] n: 0 pts: 0 s:1920x1080"
        let plan = try FFmpegMediaIO.frameAdmissionPlan(
            fromFFmpegShowInfo: metadata,
            maximumFrameCount: 1
        )

        XCTAssertTrue(plan.admits(width: 1080, height: 1920))
        XCTAssertFalse(plan.admits(width: 1080, height: 1921))
    }

    func testFFmpegFrameAdmissionRejectsUnboundedProbeOutput() throws {
        let metadata = """
        [Parsed_showinfo_0] n: 0 pts: 0 s:2x2
        [Parsed_showinfo_0] n: 1 pts: 1 s:2x2
        """

        XCTAssertThrowsError(
            try FFmpegMediaIO.frameAdmissionPlan(
                fromFFmpegShowInfo: metadata,
                maximumFrameCount: 1
            )
        ) { error in
            guard case MediaIOError.videoOperationFailed(let message) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("2 frames"))
        }
    }

    func testFFmpegFrameAdmissionFailsClosedOnInvalidDimensions() throws {
        let metadata = "[Parsed_showinfo_0] n: 0 pts: 0 s:1920x0"

        XCTAssertThrowsError(
            try FFmpegMediaIO.frameAdmissionPlan(
                fromFFmpegShowInfo: metadata,
                maximumFrameCount: 1
            )
        ) { error in
            guard case MediaIOError.videoOperationFailed(let message) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("invalid decoded frame dimensions"))
        }
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
