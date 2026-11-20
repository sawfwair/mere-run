import Foundation
#if canImport(PDFKit)
import PDFKit
import ImageIO
import UniformTypeIdentifiers
#endif

public struct MediaDocument: Sendable, Equatable {
    public let pages: [String]
    public var pageImages: [Data]? = nil
    public var renderedPixels: Int = 0
}

public enum MediaDocumentError: Error, LocalizedError {
    case oversized, unreadable, encrypted, unsupported
    public var errorDescription: String? {
        switch self {
        case .oversized: "Document exceeds the local limit: 20 MiB input, 100 pages, or 64 Mi pixels/64 MiB rendered pages."
        case .unreadable: "Document is unreadable or has no extractable text; use a vision-qualified document adapter."
        case .encrypted: "Encrypted PDF documents are not supported."
        case .unsupported: "PDF extraction requires macOS PDFKit."
        }
    }
}

/// Local document decoding; never fetches URLs or invokes client tools.
public enum MediaDocumentIO {
    public static let maximumBytes = 20 * 1024 * 1024
    public static let maximumPages = 100

    public static func text(_ data: Data) throws -> MediaDocument {
        guard data.count <= maximumBytes else { throw MediaDocumentError.oversized }
        guard let text = String(data: data, encoding: .utf8), !text.isEmpty else { throw MediaDocumentError.unreadable }
        return MediaDocument(pages: [text])
    }

    public static func pdf(_ data: Data, renderPages: Bool = false, maximumRenderedPixels: Int = 64 * 1024 * 1024) throws -> MediaDocument {
        guard data.count <= maximumBytes else { throw MediaDocumentError.oversized }
#if canImport(PDFKit)
        guard let pdf = PDFDocument(data: data) else { throw MediaDocumentError.unreadable }
        guard !pdf.isEncrypted else { throw MediaDocumentError.encrypted }
        guard pdf.pageCount > 0 else { throw MediaDocumentError.unreadable }
        guard pdf.pageCount <= maximumPages else { throw MediaDocumentError.oversized }
        let pages = try (0..<pdf.pageCount).map { index in
            guard let page = pdf.page(at: index) else { throw MediaDocumentError.unreadable }
            return page.string ?? ""
        }
        if renderPages {
            let rendered = try renderedPages(pdf, maximumPixels: min(maximumRenderedPixels, 64 * 1024 * 1024))
            return MediaDocument(pages: pages, pageImages: rendered.images, renderedPixels: rendered.pixels)
        }
        guard pages.contains(where: { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw MediaDocumentError.unreadable
        }
        return MediaDocument(pages: pages)
#else
        throw MediaDocumentError.unsupported
#endif
    }
#if canImport(PDFKit)
    /// Bounded page rasters preserve diagrams and image-only scans. The caller
    /// selects a vision adapter; text-only adapters never silently discard them.
    private static func renderedPages(_ pdf: PDFDocument, maximumPixels: Int) throws -> (images: [Data], pixels: Int) {
        var images: [Data] = []
        var pixels = 0
        var bytes = 0
        for index in 0..<pdf.pageCount {
            try Task.checkCancellation()
            guard let page = pdf.page(at: index) else { throw MediaDocumentError.unreadable }
            let bounds = page.bounds(for: .cropBox)
            guard bounds.width.isFinite, bounds.height.isFinite, bounds.width > 0, bounds.height > 0 else {
                throw MediaDocumentError.unreadable
            }
            let rotated = abs(page.rotation % 180) == 90
            let size = rotated ? CGSize(width: bounds.height, height: bounds.width) : bounds.size
            let scale = min(2, 1536 / max(size.width, size.height))
            let width = max(1, Int(ceil(size.width * scale)))
            let height = max(1, Int(ceil(size.height * scale)))
            pixels += width * height
            guard pixels <= maximumPixels else { throw MediaDocumentError.oversized }
            let data: Data = try autoreleasepool {
                guard let context = CGContext(data: nil, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw MediaDocumentError.unreadable }
                context.setFillColor(CGColor(gray: 1, alpha: 1))
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
                // PDFPage.draw honors crop origins and intrinsic rotation.
                context.scaleBy(x: CGFloat(width) / size.width, y: CGFloat(height) / size.height)
                page.draw(with: .cropBox, to: context)
                guard let image = context.makeImage() else { throw MediaDocumentError.unreadable }
                let output = NSMutableData()
                guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else {
                    throw MediaDocumentError.unreadable
                }
                CGImageDestinationAddImage(destination, image, nil)
                guard CGImageDestinationFinalize(destination) else { throw MediaDocumentError.unreadable }
                return output as Data
            }
            bytes += data.count
            guard bytes <= 64 * 1024 * 1024 else { throw MediaDocumentError.oversized }
            images.append(data)
        }
        return (images, pixels)
    }
#endif

}
