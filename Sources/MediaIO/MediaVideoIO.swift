import Foundation

public enum MediaVideoIO {
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
