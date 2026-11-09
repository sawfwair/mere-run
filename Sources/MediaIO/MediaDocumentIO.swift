import Foundation
#if canImport(PDFKit)
import PDFKit
#endif

public struct MediaDocument: Sendable, Equatable {
    public let pages: [String]
}

public enum MediaDocumentError: Error, LocalizedError {
    case oversized, unreadable, encrypted, unsupported
    public var errorDescription: String? {
        switch self {
        case .oversized: "Document exceeds the 20 MiB or 100 page local limit."
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

    public static func pdf(_ data: Data) throws -> MediaDocument {
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
        guard pages.contains(where: { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw MediaDocumentError.unreadable
        }
        return MediaDocument(pages: pages)
#else
        throw MediaDocumentError.unsupported
#endif
    }
}
