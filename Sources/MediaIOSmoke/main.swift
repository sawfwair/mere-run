import Foundation
import MediaIO

@main
struct MediaIOSmoke {
    static func main() throws {
        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-mediaio-smoke-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        try smokeImages(in: tempRoot)
        try smokeAudio(in: tempRoot)
        try smokeVideo(in: tempRoot)

        print("MediaIO smoke passed")
    }

    private static func smokeImages(in tempRoot: URL) throws {
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

        let pngURL = tempRoot.appendingPathComponent("image.png")
        try MediaImageIO.writePNG(image, to: pngURL)
        let size = try MediaImageIO.size(of: pngURL)
        try require(size.width == 2 && size.height == 2, "PNG size mismatch: \(size)")

        let decoded = try MediaImageIO.decode(pngURL)
        try require(decoded.width == 2 && decoded.height == 2, "PNG decode dimensions mismatch")
        try require(decoded.rgba8.count == 16, "PNG decode buffer size mismatch")

        let resized = try MediaImageIO.resized(decoded, width: 4, height: 4)
        try require(resized.rgba8.count == 4 * 4 * 4, "resize buffer size mismatch")

        let cropped = try MediaImageIO.centerCropped(decoded, width: 1, height: 1)
        try require(cropped.rgba8.count == 4, "crop buffer size mismatch")

        let tensor = MediaImageIO.rgbCHWFloat(decoded, normalizedToMinusOneToOne: true)
        try require(tensor.count == 12, "RGB CHW tensor size mismatch")
        try require(tensor.allSatisfy { $0.isFinite && $0 >= -1.0 && $0 <= 1.0 }, "RGB tensor range mismatch")
    }

    private static func smokeAudio(in tempRoot: URL) throws {
        let wavURL = tempRoot.appendingPathComponent("tone.wav")
        let sampleRate = 16_000
        let samples = (0..<sampleRate / 10).map { index -> Float in
            sin(Float(index) * 0.025)
        }
        try MediaAudioIO.writeFloatWAV(samples: samples, sampleRate: sampleRate, channels: 1, to: wavURL)

        let decoded = try MediaAudioIO.decode(wavURL, targetSampleRate: 8_000, channels: 1)
        try require(decoded.sampleRate == 8_000, "audio sample rate mismatch")
        try require(decoded.channelCount == 1, "audio channel count mismatch")
        try require(!decoded.samples.isEmpty, "audio decode produced no samples")
        try require(decoded.samples.allSatisfy(\.isFinite), "audio decode produced non-finite samples")
    }

    private static func smokeVideo(in tempRoot: URL) throws {
        let width = 4
        let height = 4
        let frameCount = 2
        let frameBytes = width * height * 3
        var rgb24 = [UInt8](repeating: 0, count: frameBytes * frameCount)
        for frame in 0..<frameCount {
            for pixel in 0..<(width * height) {
                let offset = (frame * frameBytes) + (pixel * 3)
                rgb24[offset] = frame == 0 ? 255 : 0
                rgb24[offset + 1] = frame == 0 ? 0 : 255
                rgb24[offset + 2] = UInt8((pixel * 11) % 255)
            }
        }

        let videoURL = tempRoot.appendingPathComponent("video.mp4")
        try MediaVideoIO.writeMP4(
            rgb24: rgb24,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: 2,
            to: videoURL
        )
        try require(FileManager.default.fileExists(atPath: videoURL.path), "video was not written")
        try require(!MediaVideoIO.hasAudioTrack(videoURL), "video-only MP4 unexpectedly has audio")

        let frameDirectory = tempRoot.appendingPathComponent("frames", isDirectory: true)
        let sequence = try MediaVideoIO.extractFrames(from: videoURL, into: frameDirectory, endFrame: 1)
        try require(!sequence.frameURLs.isEmpty, "frame extraction produced no frames")
        try require(sequence.frameWidth == width && sequence.frameHeight == height, "frame size mismatch")

        let rebuiltURL = tempRoot.appendingPathComponent("rebuilt.mp4")
        try MediaVideoIO.writeVideo(frameURLs: sequence.frameURLs, fps: sequence.fps, to: rebuiltURL)
        try require(FileManager.default.fileExists(atPath: rebuiltURL.path), "rebuilt video was not written")

        let audioURL = tempRoot.appendingPathComponent("mux-audio.wav")
        try MediaAudioIO.writeFloatWAV(
            samples: (0..<4_000).map { sin(Float($0) * 0.03) },
            sampleRate: 8_000,
            channels: 1,
            to: audioURL
        )
        let muxedURL = tempRoot.appendingPathComponent("muxed.mp4")
        try MediaVideoIO.mux(videoURL: rebuiltURL, audioURL: audioURL, outputURL: muxedURL)
        try require(MediaVideoIO.hasAudioTrack(muxedURL), "muxed MP4 has no audio track")
    }

    private static func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        if !condition() {
            throw MediaIOError.videoOperationFailed("MediaIO smoke failed: \(message)")
        }
    }
}
