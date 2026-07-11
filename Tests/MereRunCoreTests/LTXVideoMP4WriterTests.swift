import MLX
@testable import MediaIO
@testable import MereRunCore
import XCTest

final class LTXVideoMP4WriterTests: XCTestCase {
    func testBGRAFrameConversionRunsPerFrameOnDevice() {
        let frames = MLXArray([
            UInt8(1), 2, 3,
            10, 20, 30,
            100, 110, 120,
            200, 210, 220,
        ]).reshaped(2, 1, 2, 3)

        let first = LTXVideoMP4Writer.bgraFrame(frames, at: 0)
        let second = LTXVideoMP4Writer.bgraFrame(frames, at: 1)
        MLX.eval(first, second)

        XCTAssertEqual(first.asArray(UInt8.self), [3, 2, 1, 255, 30, 20, 10, 255])
        XCTAssertEqual(second.asArray(UInt8.self), [120, 110, 100, 255, 220, 210, 200, 255])
    }

    func testPrepareAudioClampsStereoSamplesBeforeEncoding() throws {
        var raw = [Float](repeating: 0, count: 20)
        raw[0] = 2.0
        raw[1] = -2.0
        raw[10] = 0.5
        raw[11] = -0.5
        let audio = MLXArray(raw).reshaped(2, 10)

        let prepared = try LTXVideoMP4Writer.prepareAudio(audio)

        XCTAssertEqual(prepared.channels, 2)
        XCTAssertEqual(prepared.interleaved.count, 20)
        XCTAssertEqual(prepared.interleaved[0], 1.0, accuracy: 0.0001)
        XCTAssertEqual(prepared.interleaved[1], 0.5, accuracy: 0.0001)
        XCTAssertEqual(prepared.interleaved[2], -1.0, accuracy: 0.0001)
        XCTAssertEqual(prepared.interleaved[3], -0.5, accuracy: 0.0001)
    }

    func testPrepareAudioRejectsNonFiniteSamples() {
        let audio = MLXArray([Float.nan])

        XCTAssertThrowsError(try LTXVideoMP4Writer.prepareAudio(audio)) { error in
            guard case LTXVideoMP4Writer.WriterError.nonFiniteAudioSample = error else {
                return XCTFail("Expected nonFiniteAudioSample, got \(error)")
            }
        }
    }

    func testWriteMP4UsesExplicitAACBitrateWhenFFmpegIsAvailable() throws {
        guard isExecutableAvailable(MediaTool.ffmpegPath),
              isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffmpeg and ffprobe are not available")
        }

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-writer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let width = 16
        let height = 16
        let fps = 24
        let frameCount = 48
        let frames = MLXArray([UInt8](repeating: 24, count: frameCount * height * width * 3))
            .reshaped(frameCount, height, width, 3)

        let sampleRate = 24_000
        var audio = [Float](repeating: 0, count: sampleRate * 2 * 2)
        for sample in 0..<(sampleRate * 2) {
            let phase = Float(sample) / Float(sampleRate)
            audio[sample] = sin(phase * 440.0 * 2.0 * .pi) * 0.25
            audio[(sampleRate * 2) + sample] = sin(phase * 660.0 * 2.0 * .pi) * 0.25
        }

        let outputURL = tempDir.appendingPathComponent("clip.mp4")
        try LTXVideoMP4Writer.writeMP4(
            frames: frames,
            fps: fps,
            to: outputURL,
            audioWaveform: MLXArray(audio).reshaped(2, sampleRate * 2),
            audioSampleRate: sampleRate
        )

        let bitRate = try ffprobeAudioBitRate(outputURL)
        XCTAssertGreaterThan(bitRate, 120_000)
    }

    func testWriteMP4PreservesOddFrameCountWhenFFprobeIsAvailable() throws {
        guard isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffprobe is not available")
        }

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-writer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let width = 16
        let height = 16
        let fps = 8
        let frameCount = 9
        var bytes = [UInt8]()
        bytes.reserveCapacity(frameCount * width * height * 3)
        for frame in 0..<frameCount {
            bytes.append(contentsOf: [UInt8](repeating: UInt8(frame * 20), count: width * height * 3))
        }
        let frames = MLXArray(bytes).reshaped(frameCount, height, width, 3)

        let outputURL = tempDir.appendingPathComponent("odd-frames.mp4")
        try LTXVideoMP4Writer.writeMP4(frames: frames, fps: fps, to: outputURL)

        XCTAssertEqual(try ffprobeVideoFrameCount(outputURL), frameCount)
    }

    func testFFmpegFrameProviderStreamsInOrderWhenAvailable() throws {
        guard isExecutableAvailable(MediaTool.ffmpegPath),
              isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffmpeg and ffprobe are not available")
        }
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-streaming-writer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let width = 16
        let height = 16
        let frameCount = 5
        var requestedFrames: [Int] = []
        let outputURL = tempDir.appendingPathComponent("streamed.mp4")
        try FFmpegMediaIO.writeMP4(
            bgra32FrameAt: { frameIndex in
                requestedFrames.append(frameIndex)
                return [UInt8](repeating: UInt8(frameIndex * 30), count: width * height * 4)
            },
            width: width,
            height: height,
            frameCount: frameCount,
            fps: 8,
            to: outputURL
        )

        XCTAssertEqual(requestedFrames, Array(0..<frameCount))
        XCTAssertEqual(try ffprobeVideoFrameCount(outputURL), frameCount)
    }

    #if canImport(AVFoundation) && canImport(CoreGraphics)
    func testAppleMediaFrameProviderStreamsInOrder() throws {
        guard isExecutableAvailable(MediaTool.ffprobePath) else {
            throw XCTSkip("ffprobe is not available")
        }
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-apple-streaming-writer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let width = 16
        let height = 16
        let frameCount = 5
        var requestedFrames: [Int] = []
        let outputURL = tempDir.appendingPathComponent("streamed.mp4")
        try AppleMediaVideoIO.writeMP4(
            bgra32FrameAt: { frameIndex in
                requestedFrames.append(frameIndex)
                return [UInt8](repeating: UInt8(frameIndex * 30), count: width * height * 4)
            },
            width: width,
            height: height,
            frameCount: frameCount,
            fps: 8,
            to: outputURL
        )

        XCTAssertEqual(requestedFrames, Array(0..<frameCount))
        XCTAssertEqual(try ffprobeVideoFrameCount(outputURL), frameCount)
    }
    #endif

    private func ffprobeAudioBitRate(_ url: URL) throws -> Int {
        let result = try runTool(
            MediaTool.ffprobePath,
            arguments: [
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                url.path,
            ]
        )
        guard let bitRate = Int(result.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return 0
        }
        return bitRate
    }

    private func ffprobeVideoFrameCount(_ url: URL) throws -> Int {
        let result = try runTool(
            MediaTool.ffprobePath,
            arguments: [
                "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                url.path,
            ]
        )
        guard let frameCount = Int(result.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return 0
        }
        return frameCount
    }

    private func runTool(_ tool: String, arguments: [String]) throws -> String {
        let process = Process()
        if tool.contains("/") {
            process.executableURL = URL(fileURLWithPath: tool)
            process.arguments = arguments
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [tool] + arguments
        }
        let stdout = Pipe()
        process.standardOutput = stdout
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        return String(decoding: data, as: UTF8.self)
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
