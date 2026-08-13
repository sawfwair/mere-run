import Foundation

struct ResolvedMediaVideoFrameRate: Sendable, Equatable {
    let framesPerSecond: Double
    let frameDurationValue: Int64
    let timeScale: Int32
}

enum MediaVideoFrameRateResolver {
    static let minimumFramesPerSecond = 1.0

    /// Resolves metadata and caller-supplied frame rates before any integer
    /// conversion. Non-positive metadata may use an explicit positive fallback;
    /// non-finite values are always rejected instead of being silently hidden.
    static func resolve(
        _ candidateFPS: Double,
        fallbackFPS: Double? = nil
    ) throws -> ResolvedMediaVideoFrameRate {
        guard candidateFPS.isFinite else {
            throw MediaIOError.invalidVideoFrameRate(candidateFPS)
        }

        let selectedFPS: Double
        if candidateFPS > 0 {
            selectedFPS = candidateFPS
        } else if let fallbackFPS {
            guard fallbackFPS.isFinite, fallbackFPS > 0 else {
                throw MediaIOError.invalidVideoFrameRate(fallbackFPS)
            }
            selectedFPS = fallbackFPS
        } else {
            throw MediaIOError.invalidVideoFrameRate(candidateFPS)
        }

        let framesPerSecond = max(minimumFramesPerSecond, selectedFPS)
        let roundedFPS = framesPerSecond.rounded()
        guard roundedFPS >= 1,
              roundedFPS <= Double(Int32.max) else {
            throw MediaIOError.invalidVideoFrameRate(framesPerSecond)
        }

        let frameDurationValue: Int64
        let timeScale: Int32
        if abs(framesPerSecond - roundedFPS) < 1e-9 || framesPerSecond > 1_000_000 {
            frameDurationValue = 1
            timeScale = Int32(roundedFPS)
        } else if let ntscBase = [24, 30, 60, 120].first(where: {
            abs(framesPerSecond - Double($0) * 1_000 / 1_001) < 0.001
        }) {
            frameDurationValue = 1_001
            timeScale = Int32(ntscBase * 1_000)
        } else {
            let precision: Int32 = 1_000_000
            frameDurationValue = max(1, Int64((Double(precision) / framesPerSecond).rounded()))
            timeScale = precision
        }

        return ResolvedMediaVideoFrameRate(
            framesPerSecond: framesPerSecond,
            frameDurationValue: frameDurationValue,
            timeScale: timeScale
        )
    }
}

public struct MediaVideoDecodeLimits: Sendable, Equatable {
    public let maximumPixelCountPerFrame: Int
    public let maximumAggregatePixelCount: Int

    public init(
        maximumPixelCountPerFrame: Int,
        maximumAggregatePixelCount: Int
    ) {
        self.maximumPixelCountPerFrame = maximumPixelCountPerFrame
        self.maximumAggregatePixelCount = maximumAggregatePixelCount
    }
}

public enum MediaVideoIO {
    public typealias BGRAFrameProvider = (_ frameIndex: Int) throws -> [UInt8]

    public static func writeMP4(
        rgb24: [UInt8],
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL
    ) throws {
        #if canImport(AVFoundation)
        do {
            try FFmpegMediaIO.writeMP4(
                rgb24: rgb24,
                width: width,
                height: height,
                frameCount: frameCount,
                fps: fps,
                to: outputURL
            )
            return
        } catch MediaIOError.missingTool where ProcessInfo.processInfo.environment["MERERUN_FFMPEG"] == nil {
            // Fall back to the native writer on macOS machines without ffmpeg.
        }
        try AppleMediaVideoIO.writeMP4(
            rgb24: rgb24,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        )
        #else
        try FFmpegMediaIO.writeMP4(
            rgb24: rgb24,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        )
        #endif
    }

    /// Writes an MP4 from one BGRA frame at a time. The provider is called in
    /// ascending frame order and may reuse its backing storage after it
    /// returns. This keeps long generated videos out of one monolithic host
    /// buffer and lets producers overlap the next device transfer with encode.
    public static func writeMP4(
        bgra32FrameAt frameProvider: BGRAFrameProvider,
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL
    ) throws {
        try writeMP4(
            bgra32FrameAt: frameProvider,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: Double(fps),
            to: outputURL
        )
    }

    /// Double-precision variant used by model pipelines whose temporal RoPE
    /// and audio duration depend on fractional container rates such as 23.976.
    public static func writeMP4(
        bgra32FrameAt frameProvider: BGRAFrameProvider,
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Double,
        to outputURL: URL
    ) throws {
        #if canImport(AVFoundation)
        do {
            try FFmpegMediaIO.writeMP4(
                bgra32FrameAt: frameProvider,
                width: width,
                height: height,
                frameCount: frameCount,
                fps: fps,
                to: outputURL
            )
            return
        } catch MediaIOError.missingTool where ProcessInfo.processInfo.environment["MERERUN_FFMPEG"] == nil {
            // Preserve the existing backend policy: use native AVFoundation
            // only when ffmpeg was not explicitly configured and is absent.
        }
        try AppleMediaVideoIO.writeMP4(
            bgra32FrameAt: frameProvider,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        )
        #else
        try FFmpegMediaIO.writeMP4(
            bgra32FrameAt: frameProvider,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        )
        #endif
    }

    public static func mux(
        videoURL: URL,
        audioURL: URL,
        outputURL: URL,
        audioBitRate: Int? = nil
    ) throws {
        #if canImport(AVFoundation)
        if let audioBitRate {
            do {
                try FFmpegMediaIO.mux(
                    videoURL: videoURL,
                    audioURL: audioURL,
                    outputURL: outputURL,
                    audioBitRate: audioBitRate
                )
                return
            } catch MediaIOError.missingTool where ProcessInfo.processInfo.environment["MERERUN_FFMPEG"] == nil {
                // Fall back to the native muxer on macOS machines without ffmpeg.
            }
        }
        try AppleMediaVideoIO.mux(videoURL: videoURL, audioURL: audioURL, outputURL: outputURL)
        #else
        try FFmpegMediaIO.mux(
            videoURL: videoURL,
            audioURL: audioURL,
            outputURL: outputURL,
            audioBitRate: audioBitRate
        )
        #endif
    }

    public static func extractFrames(
        from videoURL: URL,
        into outputDirectoryURL: URL,
        endFrame: Int? = nil,
        decodeLimits: MediaVideoDecodeLimits? = nil,
        validateDecodedSequence: ((Int, Int, Int) throws -> Void)? = nil
    ) throws -> VideoFrameSequence {
        #if canImport(AVFoundation) && canImport(CoreGraphics)
        try AppleMediaVideoIO.extractFrames(
            from: videoURL,
            into: outputDirectoryURL,
            endFrame: endFrame,
            validateDecodedSequence: validateDecodedSequence
        )
        #else
        try FFmpegMediaIO.extractFrames(
            from: videoURL,
            into: outputDirectoryURL,
            endFrame: endFrame,
            decodeLimits: decodeLimits,
            validateDecodedSequence: validateDecodedSequence
        )
        #endif
    }

    public static func writeVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        #if canImport(AVFoundation) && canImport(CoreGraphics)
        try AppleMediaVideoIO.writeVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
        #else
        try FFmpegMediaIO.writeVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
        #endif
    }

    /// Writes a categorical color video without 4:2:0 chroma subsampling.
    /// SCAIL-2 masks use RGB channels as discrete labels, so ordinary H.264
    /// output is not an acceptable interchange format for these artifacts.
    public static func writePaletteVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        #if canImport(AVFoundation) && canImport(CoreGraphics)
        try AppleMediaVideoIO.writePaletteVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
        #else
        try FFmpegMediaIO.writePaletteVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
        #endif
    }

    public static func hasAudioTrack(_ url: URL) -> Bool {
        #if canImport(AVFoundation)
        AppleMediaVideoIO.hasAudioTrack(url)
        #else
        FFmpegMediaIO.hasAudioTrack(url)
        #endif
    }
}
