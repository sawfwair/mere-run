import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum StudioOutputFileKind: Equatable {
    case image
    case text
    case other

    static func classify(_ url: URL) -> StudioOutputFileKind {
        let pathExtension = url.pathExtension.lowercased()
        if knownImageExtensions.contains(pathExtension) {
            return .image
        }
        if knownTextExtensions.contains(pathExtension) {
            return .text
        }

        guard let type = UTType(filenameExtension: pathExtension) else {
            return .other
        }
        if type.conforms(to: .image) {
            return .image
        }
        if type.conforms(to: .text) || type.conforms(to: .sourceCode) {
            return .text
        }
        return .other
    }

    private static let knownImageExtensions: Set<String> = [
        "apng", "avif", "bmp", "gif", "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff", "webp"
    ]

    private static let knownTextExtensions: Set<String> = [
        "bash", "c", "cc", "conf", "cpp", "css", "csv", "h", "hpp", "htm", "html", "ini", "js",
        "json", "jsonl", "log", "m", "markdown", "md", "mm", "py", "rb", "sh", "swift", "toml",
        "ts", "tsx", "txt", "xml", "yaml", "yml", "zsh"
    ]
}

struct StudioLoadedImage: @unchecked Sendable {
    let image: NSImage
}

enum StudioImagePreviewLoader {
    static func downsampledImage(from url: URL, maxPixelSize: CGFloat) -> StudioLoadedImage? {
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

enum StudioTextPreviewReader {
    static let maxPreviewBytes = 512 * 1024

    static func previewText(from url: URL) -> String? {
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
