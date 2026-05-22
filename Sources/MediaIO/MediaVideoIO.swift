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

    public static func mux(videoURL: URL, audioURL: URL, outputURL: URL) throws {
        #if canImport(AVFoundation)
        try AppleMediaVideoIO.mux(videoURL: videoURL, audioURL: audioURL, outputURL: outputURL)
        #else
        try FFmpegMediaIO.mux(videoURL: videoURL, audioURL: audioURL, outputURL: outputURL)
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
