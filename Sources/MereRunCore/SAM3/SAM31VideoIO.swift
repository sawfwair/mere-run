import Foundation
import MediaIO

struct SAM31VideoAsset: Sendable {
    let frameURLs: [URL]
    let fps: Double
    let frameWidth: Int
    let frameHeight: Int
}

enum SAM31VideoIO {
    enum VideoIOError: LocalizedError, Sendable {
        case failedToExtractFrame(URL, Int)
        case failedToLoadFrame(URL)
        case failedToCreateWriter(URL)

        var errorDescription: String? {
            switch self {
            case .failedToExtractFrame(let url, let index):
                return "Failed to extract frame \(index) from \(url.path)."
            case .failedToLoadFrame(let url):
                return "Failed to load frame image: \(url.path)"
            case .failedToCreateWriter(let url):
                return "Failed to create MP4 writer for \(url.path)."
            }
        }
    }

    static func extractFrames(
        from videoURL: URL,
        into outputDirectoryURL: URL,
        endFrame: Int? = nil
    ) throws -> SAM31VideoAsset {
        do {
            let sequence = try MediaVideoIO.extractFrames(
                from: videoURL,
                into: outputDirectoryURL,
                endFrame: endFrame
            )
            return SAM31VideoAsset(
                frameURLs: sequence.frameURLs,
                fps: sequence.fps,
                frameWidth: sequence.frameWidth,
                frameHeight: sequence.frameHeight
            )
        } catch {
            throw VideoIOError.failedToExtractFrame(videoURL, endFrame ?? 0)
        }
    }

    static func writeVideo(
        frameURLs: [URL],
        fps: Double,
        to outputURL: URL
    ) throws {
        guard !frameURLs.isEmpty else {
            throw VideoIOError.failedToLoadFrame(outputURL)
        }
        do {
            try MediaVideoIO.writeVideo(frameURLs: frameURLs, fps: fps, to: outputURL)
        } catch {
            throw VideoIOError.failedToCreateWriter(outputURL)
        }
    }
}
