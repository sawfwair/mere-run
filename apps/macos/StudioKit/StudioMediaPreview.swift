import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

package enum StudioOutputFileKind: Equatable {
    case image
    case audio
    case video
    case text
    case model3D
    case other

    package static func classify(_ url: URL) -> StudioOutputFileKind {
        let pathExtension = url.pathExtension.lowercased()
        if knownImageExtensions.contains(pathExtension) {
            return .image
        }
        if knownAudioExtensions.contains(pathExtension) {
            return .audio
        }
        if knownVideoExtensions.contains(pathExtension) {
            return .video
        }
        if knownTextExtensions.contains(pathExtension) {
            return .text
        }
        if knownModel3DExtensions.contains(pathExtension) {
            return .model3D
        }

        guard let type = UTType(filenameExtension: pathExtension) else {
            return .other
        }
        if type.conforms(to: .image) {
            return .image
        }
        if type.conforms(to: .audio) {
            return .audio
        }
        if type.conforms(to: .movie) || type.conforms(to: .video) || type.conforms(to: .audiovisualContent) {
            return .video
        }
        if type.conforms(to: .text) || type.conforms(to: .sourceCode) {
            return .text
        }
        return .other
    }

    private static let knownImageExtensions: Set<String> = [
        "apng", "avif", "bmp", "gif", "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff", "webp"
    ]

    private static let knownAudioExtensions: Set<String> = [
        "aac", "aif", "aiff", "caf", "flac", "m4a", "mp3", "ogg", "opus", "wav"
    ]

    private static let knownVideoExtensions: Set<String> = [
        "m4v", "mov", "mp4", "webm"
    ]

    private static let knownTextExtensions: Set<String> = [
        "bash", "c", "cc", "conf", "cpp", "css", "csv", "h", "hpp", "htm", "html", "ini", "js",
        "json", "jsonl", "log", "m", "markdown", "md", "mm", "py", "rb", "sh", "swift", "toml",
        "ts", "tsx", "txt", "xml", "yaml", "yml", "zsh"
    ]

    private static let knownModel3DExtensions: Set<String> = [
        "3mf", "dae", "glb", "gltf", "obj", "ply", "stl", "usdz"
    ]
}

package struct StudioLoadedImage: @unchecked Sendable {
    package let image: NSImage

    package init(image: NSImage) {
        self.image = image
    }
}

package enum StudioImagePreviewLoader {
    package static func downsampledImage(from url: URL, maxPixelSize: CGFloat) -> StudioLoadedImage? {
        guard StudioOutputFileKind.classify(url) == .image else {
            return nil
        }

        let sourceOptions: [CFString: Any] = [
            kCGImageSourceShouldCache: false
        ]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions as CFDictionary) else {
            return nil
        }

        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, Int(maxPixelSize.rounded()))
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnailOptions as CFDictionary) else {
            return nil
        }

        let size = NSSize(width: cgImage.width, height: cgImage.height)
        return StudioLoadedImage(image: NSImage(cgImage: cgImage, size: size))
    }
}

package enum StudioTextPreviewReader {
    package static let maxPreviewBytes = 512 * 1024

    package static func previewText(from url: URL) -> String? {
        guard StudioOutputFileKind.classify(url) == .text,
              let handle = try? FileHandle(forReadingFrom: url) else {
            return nil
        }
        defer {
            try? handle.close()
        }

        guard let data = try? handle.read(upToCount: maxPreviewBytes + 1), !data.isEmpty else {
            return nil
        }

        let previewData = data.prefix(maxPreviewBytes)
        guard var text = String(data: Data(previewData), encoding: .utf8) else {
            return nil
        }

        if data.count > maxPreviewBytes {
            text += "\n\n[Preview truncated.]"
        }
        return text
    }
}

// MARK: - Durations
/// Formats a duration in seconds as m:ss (or h:mm:ss) for transport labels.
package enum StudioTimeFormat {
    package static func string(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%d:%02d", minutes, secs)
    }
}
