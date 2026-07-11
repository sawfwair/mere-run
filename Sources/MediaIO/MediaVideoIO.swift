import Foundation

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
        endFrame: Int? = nil
    ) throws -> VideoFrameSequence {
        #if canImport(AVFoundation) && canImport(CoreGraphics)
        try AppleMediaVideoIO.extractFrames(from: videoURL, into: outputDirectoryURL, endFrame: endFrame)
        #else
        try FFmpegMediaIO.extractFrames(from: videoURL, into: outputDirectoryURL, endFrame: endFrame)
        #endif
    }

    public static func writeVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        #if canImport(AVFoundation) && canImport(CoreGraphics)
        try AppleMediaVideoIO.writeVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
        #else
        try FFmpegMediaIO.writeVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
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
