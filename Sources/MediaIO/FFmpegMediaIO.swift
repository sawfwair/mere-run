import Foundation

enum FFmpegMediaIO {
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
                "-pix_fmt", "yuv420p",
                outputURL.path
            ],
            stdin: Data(rgb24)
        )
    }

    static func mux(videoURL: URL, audioURL: URL, outputURL: URL) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        _ = try FFmpegProcess.run(
            tool: MediaTool.ffmpegPath,
            arguments: [
                "-v", "error",
                "-y",
                "-i", videoURL.path,
                "-i", audioURL.path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                outputURL.path
            ]
        )
    }

    static func extractFrames(
        from videoURL: URL,
        into directoryURL: URL,
        endFrame: Int?
    ) throws -> VideoFrameSequence {
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let pattern = directoryURL.appendingPathComponent("frame_%05d.png")
        var arguments = [
            "-v", "error",
            "-y",
            "-i", videoURL.path
        ]
        if let endFrame {
            arguments += ["-frames:v", "\(max(0, endFrame) + 1)"]
        }
        arguments.append(pattern.path)
        _ = try FFmpegProcess.run(tool: MediaTool.ffmpegPath, arguments: arguments)

        let frameURLs = (try? FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension.lowercased() == "png" }.sorted { $0.path < $1.path } ?? []

        guard let first = frameURLs.first else {
            throw MediaIOError.videoOperationFailed("No frames extracted from \(videoURL.path).")
        }
        let size = try imageSize(of: first)
        return VideoFrameSequence(
            frameURLs: frameURLs,
            fps: videoFPS(videoURL) ?? 30.0,
            frameWidth: size.width,
            frameHeight: size.height
        )
    }

    static func writeVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        guard let first = frameURLs.first else {
            throw MediaIOError.videoOperationFailed("No frames supplied for video writing.")
        }
        let tempDir = first.deletingLastPathComponent()
        let listURL = tempDir.appendingPathComponent("mererun-frames-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: listURL) }
        let duration = 1.0 / max(fps, 1.0)
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
                "-r", "\(max(1, Int(fps.rounded())))",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
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

    private static func videoFPS(_ url: URL) -> Double? {
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
            return nil
        }
        let text = String(decoding: result.stdout, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = text.split(separator: "/").compactMap { Double($0) }
        if parts.count == 2, parts[1] != 0 {
            return parts[0] / parts[1]
        }
        return Double(text)
    }
}
