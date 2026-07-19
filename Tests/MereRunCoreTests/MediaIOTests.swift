import AudioCodecs
import Foundation
@testable import MediaIO
import XCTest

#if canImport(CoreGraphics)
import CoreGraphics
#endif

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

    func testPNGRoundTripPreservesStraightAlphaBytes() throws {
        #if !canImport(CoreGraphics)
        guard isExecutableAvailable(MediaTool.ffmpegPath),
              isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffmpeg and ffprobe are required")
        }
        #endif
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-alpha-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Straight-alpha samples, including RGB under partial and zero alpha.
        // A premultiplied encode or decode step corrupts every non-opaque one.
        let image = try MediaImage(
            width: 2,
            height: 2,
            rgba8: [
                200, 200, 200, 128,
                200, 0, 0, 64,
                0, 200, 0, 255,
                50, 60, 70, 0,
            ]
        )
        let url = tempDir.appendingPathComponent("straight-alpha.png")
        try MediaImageIO.writePNG(image, to: url)

        let decoded = try MediaImageIO.decode(url)
        XCTAssertEqual(decoded.width, image.width)
        XCTAssertEqual(decoded.height, image.height)
        XCTAssertEqual(decoded.rgba8, image.rgba8)

        let decodedFromData = try MediaImageIO.decode(data: Data(contentsOf: url))
        XCTAssertEqual(decodedFromData.rgba8, image.rgba8)
    }

    #if canImport(CoreGraphics)
    func testCGImageDecodeUnpremultipliesNonDirectFormats() throws {
        // Premultiplied (100, 100, 100, 128) has no direct-copy path, so the
        // draw fallback must un-premultiply back to straight ≈ (199, 199, 199).
        let provider = try XCTUnwrap(CGDataProvider(data: Data([100, 100, 100, 128]) as CFData))
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let premultiplied = try XCTUnwrap(CGImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))

        let decoded = try AppleMediaImageIO.mediaImage(from: premultiplied)
        XCTAssertEqual(decoded.rgba8[3], 128)
        for channel in 0..<3 {
            XCTAssertEqual(Int(decoded.rgba8[channel]), 199, accuracy: 1)
        }
    }
    #endif

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

    func testAudioProbeAndSegmentDecodeStayWithinRequestedRange() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-segment-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let sampleRate = 8_000
        let samples = (0..<(sampleRate * 2)).flatMap { index -> [Float] in
            let value = sin(Float(index) * 2 * .pi * 220 / Float(sampleRate)) * 0.25
            return [value, -value]
        }
        try MediaAudioIO.writeFloatWAV(
            samples: samples,
            sampleRate: sampleRate,
            channels: 2,
            to: tempURL
        )

        let metadata = try MediaAudioIO.probe(tempURL)
        XCTAssertEqual(metadata.sampleRate, sampleRate)
        XCTAssertEqual(metadata.channelCount, 2)
        XCTAssertEqual(metadata.frameCount, Int64(sampleRate * 2))
        XCTAssertEqual(metadata.durationSeconds, 2, accuracy: 1e-6)

        let segment = try MediaAudioIO.decodeSegment(
            tempURL,
            startTime: 0.5,
            duration: 0.25,
            targetSampleRate: 16_000,
            channels: 2
        )
        XCTAssertEqual(segment.sampleRate, 16_000)
        XCTAssertEqual(segment.channelCount, 2)
        XCTAssertTrue(segment.isInterleaved)
        XCTAssertEqual(segment.samples.count / 2, 4_000, accuracy: 2)
        XCTAssertGreaterThan(segment.samples.map(abs).max() ?? 0, 0.1)
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

    func testVideoAudioMuxPreservesExactFrameCadenceAndDuration() throws {
        guard isExecutableAvailable(MediaTool.ffmpegPath),
              isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffmpeg and ffprobe are required")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mediaio-exact-mux-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let frameCount = 14
        let fps = 16
        let duration = Double(frameCount) / Double(fps)
        let silentURL = root.appendingPathComponent("silent.mp4")
        let audioURL = root.appendingPathComponent("audio.wav")
        let outputURL = root.appendingPathComponent("muxed.mp4")
        try MediaVideoIO.writeMP4(
            rgb24: [UInt8](repeating: 127, count: 32 * 32 * 3 * frameCount),
            width: 32,
            height: 32,
            frameCount: frameCount,
            fps: fps,
            to: silentURL
        )
        let audioFrameCount = Int((duration * 48_000).rounded())
        try MediaAudioIO.writeFloatWAV(
            samples: [Float](repeating: 0.1, count: audioFrameCount * 2),
            sampleRate: 48_000,
            channels: 2,
            to: audioURL
        )

        try MediaVideoIO.mux(
            videoURL: silentURL,
            audioURL: audioURL,
            outputURL: outputURL,
            audioBitRate: 192_000
        )

        let decoded = try MediaVideoIO.extractFrames(
            from: outputURL,
            into: root.appendingPathComponent("decoded", isDirectory: true)
        )
        let audio = try MediaAudioIO.probe(outputURL)
        XCTAssertEqual(decoded.frameURLs.count, frameCount)
        XCTAssertEqual(decoded.fps, Double(fps), accuracy: 0.001)
        XCTAssertEqual(audio.durationSeconds, duration, accuracy: 0.001)
        XCTAssertTrue(MediaVideoIO.hasAudioTrack(outputURL))
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

    func testRealFFTPlanNonPowerOfTwoImpulseHasUnitPowerInEveryBin() throws {
        let plan = try RealFFTPlan(size: 400)
        var samples = [Float](repeating: 0, count: 400)
        samples[0] = 1

        let spectrum = plan.powerSpectrum(samples)

        XCTAssertEqual(spectrum.count, 201)
        for power in spectrum {
            XCTAssertEqual(power, 1, accuracy: 0.000_01)
        }
    }

    func testRealFFTPlanNonPowerOfTwoMatchesScalarReferenceBins() throws {
        let size = 400
        let plan = try RealFFTPlan(size: size)
        let samples = (0..<size).map { index in
            (sinf(Float(index) * 0.037) * 0.7) + (cosf(Float(index) * 0.091) * 0.2)
        }
        let spectrum = plan.powerSpectrum(samples)

        for frequency in [0, 1, 7, 29, 113, 200] {
            var real = 0.0
            var imaginary = 0.0
            for sampleIndex in 0..<size {
                let angle = -2 * Double.pi * Double(frequency * sampleIndex) / Double(size)
                let value = Double(samples[sampleIndex])
                real += value * cos(angle)
                imaginary += value * sin(angle)
            }
            let expected = Float((real * real) + (imaginary * imaginary))
            XCTAssertEqual(
                spectrum[frequency],
                expected,
                accuracy: max(0.000_1, expected * 0.000_1),
                "frequency bin \(frequency)"
            )
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
