import Foundation

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
