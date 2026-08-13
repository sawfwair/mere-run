import Foundation

/// An 8-bit RGBA raster whose `rgba8` samples use straight (un-premultiplied)
/// alpha, matching PNG storage and FFmpeg's `rgba` pixel format. Backends
/// convert premultiplied sources on decode and must declare straight alpha on
/// encode so semi-transparent RGB survives round trips byte-exactly.
public struct MediaImage: Sendable, Hashable {
    public let width: Int
    public let height: Int
    public var rgba8: [UInt8]

    public init(width: Int, height: Int, rgba8: [UInt8]) throws {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        guard rgba8.count == width * height * 4 else {
            throw MediaIOError.invalidBufferSize(expected: width * height * 4, actual: rgba8.count)
        }
        self.width = width
        self.height = height
        self.rgba8 = rgba8
    }
}

/// A scene-linear or log-encoded RGB raster with unbounded float samples.
/// Samples are interleaved RGB in row-major order; values above one are
/// intentionally preserved for OpenEXR/HDR workflows.
public struct MediaFloatImage: Sendable, Hashable {
    public let width: Int
    public let height: Int
    public var rgb: [Float]

    public init(width: Int, height: Int, rgb: [Float]) throws {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        guard rgb.count == width * height * 3 else {
            throw MediaIOError.invalidBufferSize(
                expected: width * height * 3 * MemoryLayout<Float>.size,
                actual: rgb.count * MemoryLayout<Float>.size
            )
        }
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

public struct MediaAudioBuffer: Sendable, Hashable {
    public let samples: [Float]
    public let sampleRate: Int
    public let channelCount: Int
    public let isInterleaved: Bool

    public init(
        samples: [Float],
        sampleRate: Int,
        channelCount: Int = 1,
        isInterleaved: Bool = true
    ) {
        self.samples = samples
        self.sampleRate = max(1, sampleRate)
        self.channelCount = max(1, channelCount)
        self.isInterleaved = isInterleaved
    }
}

public struct MediaAudioMetadata: Sendable, Hashable {
    public let sampleRate: Int
    public let channelCount: Int
    public let frameCount: Int64
    public let durationSeconds: Double

    public init(sampleRate: Int, channelCount: Int, frameCount: Int64, durationSeconds: Double) {
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.frameCount = frameCount
        self.durationSeconds = durationSeconds
    }
}

public struct VideoFrameSequence: Sendable, Hashable {
    public let frameURLs: [URL]
    public let fps: Double
    public let frameWidth: Int
    public let frameHeight: Int

    public init(frameURLs: [URL], fps: Double, frameWidth: Int, frameHeight: Int) {
        self.frameURLs = frameURLs
        self.fps = max(1, fps)
        self.frameWidth = max(0, frameWidth)
        self.frameHeight = max(0, frameHeight)
    }
}

public enum ImageResizeMode: Sendable, Hashable {
    case exact(width: Int, height: Int)
    case aspectFill(width: Int, height: Int)
}

public enum MediaIOError: LocalizedError, Sendable, Equatable {
    case invalidImageDimensions(width: Int, height: Int)
    case invalidBufferSize(expected: Int, actual: Int)
    case imageDecodeFailed(URL)
    case imageEncodeFailed(URL)
    case imageMetadataFailed(URL)
    case audioDecodeFailed(URL, String)
    case audioEncodeFailed(URL, String)
    case invalidAudioRange(startTime: Double, duration: Double)
    case invalidVideoFrameRate(Double)
    case videoOperationFailed(String)
    case missingTool(String)
    case processFailed(tool: String, status: Int32, stderr: String)
    case unsupportedPlatform(String)

    public var errorDescription: String? {
        switch self {
        case .invalidImageDimensions(let width, let height):
            return "Invalid media image dimensions \(width)x\(height)."
        case .invalidBufferSize(let expected, let actual):
            return "Invalid media buffer size: expected \(expected) bytes, got \(actual)."
        case .imageDecodeFailed(let url):
            return "Failed to decode image: \(url.path)"
        case .imageEncodeFailed(let url):
            return "Failed to encode image: \(url.path)"
        case .imageMetadataFailed(let url):
            return "Failed to read image metadata: \(url.path)"
        case .audioDecodeFailed(let url, let details):
            return "Failed to decode audio \(url.path): \(details)"
        case .audioEncodeFailed(let url, let details):
            return "Failed to encode audio \(url.path): \(details)"
        case .invalidAudioRange(let startTime, let duration):
            return "Invalid audio range: start time \(startTime) and duration \(duration) must be finite, with a nonnegative start and positive duration."
        case .invalidVideoFrameRate(let fps):
            return "Invalid video frame rate \(fps). Expected a finite, positive rate whose rounded time scale fits in Int32."
        case .videoOperationFailed(let details):
            return "Video operation failed: \(details)"
        case .missingTool(let tool):
            return "\(tool) was not found. Install ffmpeg or set MERERUN_\(tool.uppercased()) to the executable path."
        case .processFailed(let tool, let status, let stderr):
            let details = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "\(tool) exited with status \(status).\(details.isEmpty ? "" : " \(details)")"
        case .unsupportedPlatform(let message):
            return message
        }
    }
}

public enum MediaTool {
    public static var ffmpegPath: String {
        ProcessInfo.processInfo.environment["MERERUN_FFMPEG"] ?? "ffmpeg"
    }

    public static var ffprobePath: String {
        ProcessInfo.processInfo.environment["MERERUN_FFPROBE"] ?? "ffprobe"
    }
}
