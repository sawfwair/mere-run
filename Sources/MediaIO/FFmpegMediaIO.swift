import Foundation

enum FFmpegMediaIO {
    private struct AudioProbe: Decodable {
        struct Stream: Decodable {
            let sampleRate: String
            let channels: Int
            let duration: String?

            enum CodingKeys: String, CodingKey {
                case sampleRate = "sample_rate"
                case channels
                case duration
            }
        }

        struct Format: Decodable {
            let duration: String?
        }

        let streams: [Stream]
        let format: Format
    }

    struct FrameAdmissionPlan: Equatable, Sendable {
        let frameCount: Int
        let maximumWidth: Int
        let maximumHeight: Int
        let maximumLongSide: Int
        let maximumShortSide: Int
        let maximumPixelCount: Int

        func admits(width: Int, height: Int) -> Bool {
            guard width > 0, height > 0 else { return false }
            let (pixelCount, overflow) = width.multipliedReportingOverflow(by: height)
            guard !overflow else { return false }
            return max(width, height) <= maximumLongSide
                && min(width, height) <= maximumShortSide
                && pixelCount <= maximumPixelCount
        }
    }

    static func imageSize(of url: URL) throws -> (width: Int, height: Int) {
        let result = try FFmpegProcess.run(
            tool: MediaTool.ffprobePath,
            arguments: [
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                url.path
            ]
        )
        let text = String(decoding: result.stdout, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let pieces = text.split(separator: "x").compactMap { Int($0) }
        guard pieces.count == 2, pieces[0] > 0, pieces[1] > 0 else {
            throw MediaIOError.imageMetadataFailed(url)
        }
        return (pieces[0], pieces[1])
    }

    static func decodeImage(_ url: URL) throws -> MediaImage {
        let size = try imageSize(of: url)
        let result = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-i", url.path,
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "pipe:1"
            ]
        )
        let bytes = [UInt8](result.stdout)
        return try MediaImage(width: size.width, height: size.height, rgba8: bytes)
    }

    static func writePNG(_ image: MediaImage, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        _ = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", "\(image.width)x\(image.height)",
                "-i", "pipe:0",
                "-frames:v", "1",
                url.path
            ],
            stdin: Data(image.rgba8)
        )
    }

    static func decodeAudio(
        _ url: URL,
        targetSampleRate: Int,
        channels: Int
    ) throws -> MediaAudioBuffer {
        let sampleRate = max(1, targetSampleRate)
        let channelCount = max(1, channels)
        do {
            let result = try FFmpegProcess.run(
                tool: MediaTool.ffmpegPath,
                arguments: [
                    "-v", "error",
                    "-i", url.path,
                    "-ac", "\(channelCount)",
                    "-ar", "\(sampleRate)",
                    "-f", "f32le",
                    "pipe:1"
                ]
            )
            let samples = result.stdout.withUnsafeBytes { raw -> [Float] in
                let floats = raw.bindMemory(to: Float.self)
                return Array(floats)
            }
            return MediaAudioBuffer(
                samples: samples,
                sampleRate: sampleRate,
                channelCount: channelCount,
                isInterleaved: true
            )
        } catch let error as MediaIOError {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        } catch {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        }
    }

    static func probeAudio(_ url: URL) throws -> MediaAudioMetadata {
        do {
            let result = try FFmpegProcess.run(
                tool: MediaTool.ffprobePath,
                arguments: [
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=sample_rate,channels,duration:format=duration",
                    "-of", "json",
                    url.path
                ]
            )
            let probe = try JSONDecoder().decode(AudioProbe.self, from: result.stdout)
            guard let stream = probe.streams.first,
                  let sampleRate = Int(stream.sampleRate),
                  sampleRate > 0,
                  stream.channels > 0,
                  let duration = Double(stream.duration ?? probe.format.duration ?? ""),
                  duration.isFinite,
                  duration > 0 else {
                throw MediaIOError.audioDecodeFailed(url, "ffprobe returned incomplete audio metadata.")
            }
            return MediaAudioMetadata(
                sampleRate: sampleRate,
                channelCount: stream.channels,
                frameCount: Int64((duration * Double(sampleRate)).rounded()),
                durationSeconds: duration
            )
        } catch let error as MediaIOError {
            throw error
        } catch {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        }
    }

    static func decodeAudioSegment(
        _ url: URL,
        startTime: Double,
        duration: Double,
        targetSampleRate: Int,
        channels: Int
    ) throws -> MediaAudioBuffer {
        let sampleRate = max(1, targetSampleRate)
        let channelCount = max(1, channels)
        do {
            let result = try FFmpegProcess.run(
                tool: MediaTool.ffmpegPath,
                arguments: [
                    "-v", "error",
                    "-i", url.path,
                    "-ss", String(startTime),
                    "-t", String(duration),
                    "-ac", "\(channelCount)",
                    "-ar", "\(sampleRate)",
                    "-f", "f32le",
                    "pipe:1"
                ]
            )
            let samples = result.stdout.withUnsafeBytes { raw -> [Float] in
                Array(raw.bindMemory(to: Float.self))
            }
            guard !samples.isEmpty else {
                throw MediaIOError.audioDecodeFailed(url, "The requested audio interval is empty.")
            }
            return MediaAudioBuffer(
                samples: samples,
                sampleRate: sampleRate,
                channelCount: channelCount,
                isInterleaved: true
            )
        } catch let error as MediaIOError {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        } catch {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        }
    }

    static func transcodeAudio(
        _ inputURL: URL,
        to outputURL: URL,
        format: String
    ) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let normalized = format.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var arguments = [
            "-v", "error",
            "-y",
            "-i", inputURL.path,
            "-vn"
        ]
        switch normalized {
        case "mp3":
            arguments += ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"]
        case "opus":
            arguments += ["-c:a", "libopus", "-b:a", "96k", "-f", "opus"]
        case "aac":
            arguments += ["-c:a", "aac", "-b:a", "192k", "-f", "adts"]
        case "flac":
            arguments += ["-c:a", "flac", "-f", "flac"]
        default:
            throw MediaIOError.audioEncodeFailed(outputURL, "Unsupported audio format: \(format)")
        }
        arguments.append(outputURL.path)

        do {
            _ = try FFmpegProcess.run(tool: MediaTool.ffmpegPath, arguments: arguments)
        } catch let error as MediaIOError {
            throw MediaIOError.audioEncodeFailed(outputURL, error.localizedDescription)
        } catch {
            throw MediaIOError.audioEncodeFailed(outputURL, error.localizedDescription)
        }
    }

    static func writeMP4(
        rgb24: [UInt8],
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL
    ) throws {
        guard width > 0, height > 0, frameCount > 0, fps > 0 else {
            throw MediaIOError.videoOperationFailed("Invalid MP4 dimensions or frame rate.")
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        _ = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", "\(width)x\(height)",
                "-r", "\(fps)",
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                outputURL.path
            ],
            stdin: Data(rgb24)
        )
    }

    static func writeMP4(
        bgra32FrameAt frameProvider: MediaVideoIO.BGRAFrameProvider,
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL
    ) throws {
        guard width > 0, height > 0, frameCount > 0, fps > 0 else {
            throw MediaIOError.videoOperationFailed("Invalid MP4 dimensions or frame rate.")
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let frameStride = width * height * 4
        _ = try FFmpegProcess.runStreamingInput(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgra",
                "-s", "\(width)x\(height)",
                "-r", "\(fps)",
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                outputURL.path,
            ],
            writeInput: { handle in
                for frameIndex in 0..<frameCount {
                    let frame = try frameProvider(frameIndex)
                    guard frame.count == frameStride else {
                        throw MediaIOError.invalidBufferSize(expected: frameStride, actual: frame.count)
                    }
                    try handle.write(contentsOf: Data(frame))
                }
            }
        )
    }

    static func mux(
        videoURL: URL,
        audioURL: URL,
        outputURL: URL,
        audioBitRate: Int? = nil
    ) throws {
        let videoDuration = try videoDurationSeconds(videoURL)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var arguments = [
            "-v", "error",
            "-y",
            "-i", videoURL.path,
            "-i", audioURL.path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            // Preserve every generated video frame when decoded audio is a few
            // samples short. Without padding, `-shortest` can discard delayed
            // H.264 B-frames before their presentation timestamps are reached.
            "-af", "apad",
        ]
        if let audioBitRate {
            arguments += ["-b:a", "\(max(1, audioBitRate))"]
        }
        arguments += [
            "-t", String(videoDuration),
            outputURL.path,
        ]
        _ = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: arguments
        )
    }

    private static func videoDurationSeconds(_ url: URL) throws -> Double {
        let result = try FFmpegProcess.run(
            tool: MediaTool.ffprobePath,
            arguments: [
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                url.path,
            ]
        )
        let rawDuration = String(decoding: result.stdout, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let duration = Double(rawDuration), duration.isFinite, duration > 0 else {
            throw MediaIOError.videoOperationFailed("Could not determine video duration for muxing.")
        }
        return duration
    }

    static func extractFrames(
        from videoURL: URL,
        into directoryURL: URL,
        endFrame: Int?,
        decodeLimits: MediaVideoDecodeLimits?,
        validateDecodedSequence: ((Int, Int, Int) throws -> Void)?
    ) throws -> VideoFrameSequence {
        let sourceSize = try imageSize(of: videoURL)
        let requestedFrameCount: Int?
        if let endFrame {
            let (count, overflow) = endFrame.addingReportingOverflow(1)
            guard endFrame >= 0, !overflow else {
                throw MediaIOError.videoOperationFailed("Invalid end frame \(endFrame).")
            }
            requestedFrameCount = count
        } else {
            requestedFrameCount = nil
        }

        let admissionPlan: FrameAdmissionPlan?
        if let validateDecodedSequence {
            guard let requestedFrameCount else {
                throw MediaIOError.videoOperationFailed(
                    "A finite end frame is required when validating decoded video resources."
                )
            }
            guard let decodeLimits, decodeLimits.maximumPixelCountPerFrame > 0 else {
                throw MediaIOError.videoOperationFailed(
                    "Validated FFmpeg extraction requires a positive decoder pixel limit."
                )
            }
            guard decodeLimits.maximumAggregatePixelCount >= requestedFrameCount else {
                throw MediaIOError.videoOperationFailed(
                    "Validated FFmpeg extraction requires an aggregate pixel limit "
                        + "covering at least one pixel per requested frame."
                )
            }
            // Reject unsafe coded dimensions without opening the decoder. The
            // exact bounded pass below then proves aggregate and dynamic-frame
            // limits before any PNG is written.
            try validateDecodedSequence(
                sourceSize.width,
                sourceSize.height,
                requestedFrameCount
            )
            let plan = try decodeFrameAdmission(
                videoURL,
                maximumFrameCount: requestedFrameCount,
                decodeLimits: decodeLimits
            )
            try validateDecodedSequence(
                plan.maximumWidth,
                plan.maximumHeight,
                plan.frameCount
            )
            admissionPlan = plan
        } else {
            admissionPlan = nil
        }

        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let pattern = directoryURL.appendingPathComponent("frame_%05d.png")
        var arguments = [
            "-v", "error",
            "-y",
        ]
        if let decodeLimits {
            let frameLimit = admissionPlan?.frameCount ?? requestedFrameCount
            guard let frameLimit,
                  frameLimit > 0,
                  decodeLimits.maximumPixelCountPerFrame > 0,
                  decodeLimits.maximumAggregatePixelCount >= frameLimit else {
                throw MediaIOError.videoOperationFailed(
                    "FFmpeg decoder pixel and aggregate limits must be positive and bounded."
                )
            }
            let maximumDecodedPixels = min(
                decodeLimits.maximumPixelCountPerFrame,
                decodeLimits.maximumAggregatePixelCount / frameLimit
            )
            arguments += ["-max_pixels", "\(maximumDecodedPixels)"]
        }
        arguments += [
            "-i", videoURL.path,
            "-map", "0:v:0",
        ]
        if let frameCount = admissionPlan?.frameCount ?? requestedFrameCount {
            arguments += ["-frames:v", "\(frameCount)"]
        }
        arguments.append(pattern.path)
        _ = try FFmpegProcess.run(tool: MediaTool.ffmpegPath, arguments: arguments)

        let frameURLs = (try? FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension.lowercased() == "png" }.sorted { $0.path < $1.path } ?? []

        guard !frameURLs.isEmpty else {
            throw MediaIOError.videoOperationFailed("No frames extracted from \(videoURL.path).")
        }
        var firstFrameSize: (width: Int, height: Int)?
        var maximumWidth = 0
        var maximumHeight = 0
        for frameURL in frameURLs {
            let frameSize = try imageSize(of: frameURL)
            if firstFrameSize == nil {
                firstFrameSize = frameSize
            }
            maximumWidth = max(maximumWidth, frameSize.width)
            maximumHeight = max(maximumHeight, frameSize.height)
            if let admissionPlan, !admissionPlan.admits(width: frameSize.width, height: frameSize.height) {
                throw MediaIOError.videoOperationFailed(
                    "FFmpeg extracted a \(frameSize.width)x\(frameSize.height) frame outside "
                        + "the orientation-invariant admission bounds."
                )
            }
        }
        if let admissionPlan {
            guard frameURLs.count <= admissionPlan.frameCount else {
                throw MediaIOError.videoOperationFailed(
                    "FFmpeg extracted \(frameURLs.count) frames after admission allowed "
                        + "\(admissionPlan.frameCount)."
                )
            }
        }
        try validateDecodedSequence?(maximumWidth, maximumHeight, frameURLs.count)
        guard let firstFrameSize else {
            throw MediaIOError.videoOperationFailed("No frame dimensions were extracted.")
        }
        return VideoFrameSequence(
            frameURLs: frameURLs,
            fps: try videoFPS(videoURL),
            frameWidth: firstFrameSize.width,
            frameHeight: firstFrameSize.height
        )
    }

    static func frameAdmissionPlan(
        fromFFmpegShowInfo output: String,
        maximumFrameCount: Int
    ) throws -> FrameAdmissionPlan {
        guard maximumFrameCount > 0 else {
            throw MediaIOError.videoOperationFailed(
                "FFmpeg frame admission requires a positive frame limit."
            )
        }
        var dimensions: [(width: Int, height: Int)] = []
        for line in output.split(whereSeparator: \.isNewline) {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.contains("n:"),
                  let sizeField = fields.first(where: { $0.hasPrefix("s:") }) else {
                continue
            }
            let pieces = sizeField.dropFirst(2).split(separator: "x", maxSplits: 1)
            guard pieces.count == 2,
                  let width = Int(pieces[0]),
                  let height = Int(pieces[1]) else {
                throw MediaIOError.videoOperationFailed(
                    "FFmpeg showinfo returned malformed per-frame dimensions."
                )
            }
            dimensions.append((width, height))
        }
        guard !dimensions.isEmpty else {
            throw MediaIOError.videoOperationFailed(
                "FFmpeg showinfo did not return any decodable video frames."
            )
        }
        guard dimensions.count <= maximumFrameCount else {
            throw MediaIOError.videoOperationFailed(
                "FFmpeg showinfo returned \(dimensions.count) frames for a \(maximumFrameCount)-frame decode."
            )
        }

        var maximumWidth = 0
        var maximumHeight = 0
        var maximumLongSide = 0
        var maximumShortSide = 0
        var maximumPixelCount = 0
        for frame in dimensions {
            guard frame.width > 0, frame.height > 0 else {
                throw MediaIOError.videoOperationFailed(
                    "FFmpeg showinfo returned invalid decoded frame dimensions "
                        + "\(frame.width)x\(frame.height)."
                )
            }
            let (pixelCount, overflow) = frame.width.multipliedReportingOverflow(by: frame.height)
            guard !overflow else {
                throw MediaIOError.videoOperationFailed(
                    "FFmpeg showinfo frame dimensions overflow the pixel count."
                )
            }
            maximumWidth = max(maximumWidth, frame.width)
            maximumHeight = max(maximumHeight, frame.height)
            maximumLongSide = max(maximumLongSide, max(frame.width, frame.height))
            maximumShortSide = max(maximumShortSide, min(frame.width, frame.height))
            maximumPixelCount = max(maximumPixelCount, pixelCount)
        }
        return FrameAdmissionPlan(
            frameCount: dimensions.count,
            maximumWidth: maximumWidth,
            maximumHeight: maximumHeight,
            maximumLongSide: maximumLongSide,
            maximumShortSide: maximumShortSide,
            maximumPixelCount: maximumPixelCount
        )
    }

    private static func decodeFrameAdmission(
        _ url: URL,
        maximumFrameCount: Int,
        decodeLimits: MediaVideoDecodeLimits
    ) throws -> FrameAdmissionPlan {
        let maximumDecodedPixels = min(
            decodeLimits.maximumPixelCountPerFrame,
            decodeLimits.maximumAggregatePixelCount / maximumFrameCount
        )
        let result = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "info",
                "-max_pixels", "\(maximumDecodedPixels)",
                "-i", url.path,
                "-map", "0:v:0",
                "-vf", "showinfo",
                "-frames:v", "\(maximumFrameCount)",
                "-an", "-sn", "-dn",
                "-f", "null",
                "-",
            ]
        )
        return try frameAdmissionPlan(
            fromFFmpegShowInfo: result.stderr,
            maximumFrameCount: maximumFrameCount
        )
    }

    static func writeVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        guard let first = frameURLs.first else {
            throw MediaIOError.videoOperationFailed("No frames supplied for video writing.")
        }
        let tempDir = first.deletingLastPathComponent()
        let listURL = tempDir.appendingPathComponent("mererun-frames-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: listURL) }
        let resolvedFPS = try MediaVideoFrameRateResolver.resolve(fps)
        let duration = 1.0 / resolvedFPS.framesPerSecond
        let list = frameURLs.map { "file '\($0.path.replacingOccurrences(of: "'", with: "'\\''"))'\nduration \(duration)" }
            .joined(separator: "\n")
        try list.write(to: listURL, atomically: true, encoding: .utf8)
        _ = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", listURL.path,
                "-r", "\(resolvedFPS.timeScale)",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                outputURL.path
            ]
        )
    }

    static func writePaletteVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        guard let first = frameURLs.first else {
            throw MediaIOError.videoOperationFailed("No frames supplied for palette video writing.")
        }
        let tempDir = first.deletingLastPathComponent()
        let listURL = tempDir.appendingPathComponent("mererun-palette-frames-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: listURL) }
        let resolvedFPS = try MediaVideoFrameRateResolver.resolve(fps)
        let duration = 1.0 / resolvedFPS.framesPerSecond
        let list = frameURLs.map { "file '\($0.path.replacingOccurrences(of: "'", with: "'\\''"))'\nduration \(duration)" }
            .joined(separator: "\n")
        try list.write(to: listURL, atomically: true, encoding: .utf8)
        _ = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", listURL.path,
                "-r", "\(resolvedFPS.timeScale)",
                "-c:v", "prores_ks",
                "-profile:v", "4",
                "-pix_fmt", "yuva444p10le",
                outputURL.path
            ]
        )
    }

    static func hasAudioTrack(_ url: URL) -> Bool {
        guard let result = try? FFmpegProcess.run(
            tool: MediaTool.ffprobePath,
            arguments: [
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                url.path
            ]
        ) else {
            return false
        }
        let text = String(decoding: result.stdout, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return !text.isEmpty
    }

    private static func videoFPS(_ url: URL) throws -> Double {
        guard let result = try? FFmpegProcess.run(
            tool: MediaTool.ffprobePath,
            arguments: [
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                url.path
            ]
        ) else {
            return try MediaVideoFrameRateResolver.resolve(0, fallbackFPS: 30).framesPerSecond
        }
        let text = String(decoding: result.stdout, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = text.split(separator: "/").compactMap { Double($0) }
        if parts.count == 2, parts[1] != 0 {
            return try MediaVideoFrameRateResolver.resolve(
                parts[0] / parts[1],
                fallbackFPS: 30
            ).framesPerSecond
        }
        guard let candidateFPS = Double(text) else {
            return try MediaVideoFrameRateResolver.resolve(0, fallbackFPS: 30).framesPerSecond
        }
        return try MediaVideoFrameRateResolver.resolve(
            candidateFPS,
            fallbackFPS: 30
        ).framesPerSecond
    }

}
