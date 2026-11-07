#if canImport(PDFKit)
import AppKit
import CoreText
import PDFKit
import XCTest
import MediaIO
import MereRunCore
@testable import MereRunCLI

final class MessagesDocumentTests: XCTestCase {
    private func pdf(pages: Int = 1) throws -> Data {
        let output = NSMutableData()
        let consumer = try XCTUnwrap(CGDataConsumer(data: output))
        var bounds = CGRect(x: 0, y: 0, width: 300, height: 300)
        let context = try XCTUnwrap(CGContext(consumer: consumer, mediaBox: &bounds, nil))
        for index in 0..<pages {
            context.beginPDFPage(nil)
            context.textPosition = CGPoint(x: 20, y: 200)
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: "Local page \(index + 1)", attributes: [.font: NSFont.systemFont(ofSize: 12)]))
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()
        return output as Data
    }
    func testSystemPDFExtractionPreservesPages() throws {
        let document = try MediaDocumentIO.pdf(pdf(pages: 2))
        XCTAssertEqual(document.pages.count, 2)
        XCTAssertTrue(document.pages[0].contains("Local page 1"))
        XCTAssertTrue(document.pages[1].contains("Local page 2"))
    }
    func testEncryptedAndOversizedPageCountsFailExplicitly() throws {
        let source = try XCTUnwrap(PDFDocument(data: pdf()))
        let encrypted = try XCTUnwrap(source.dataRepresentation(options: [PDFDocumentWriteOption.ownerPasswordOption: "owner", PDFDocumentWriteOption.userPasswordOption: "user"]))
        XCTAssertThrowsError(try MediaDocumentIO.pdf(encrypted)) { error in
            XCTAssertEqual(error.localizedDescription, MediaDocumentError.encrypted.localizedDescription)
        }
        XCTAssertThrowsError(try MediaDocumentIO.pdf(pdf(pages: 101)))
    }
    func testCitationRequestsFailBeforeDocumentDecoding() throws {
        let request = try JSONDecoder().decode(MessagesRequest.self, from: Data(#"{"model":"fixture","max_tokens":16,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"invalid"},"citations":{"enabled":true}}]}]}"#.utf8))
        let input = try RuntimeChatRequest.messages(request, version: .original, counting: false, continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        XCTAssertThrowsError(try input.prepare(.init(modelID: MuseGlimmerResources.modelId, engine: .textChatMuseGlimmer, settings: .init(), profile: profile, contextSize: 32768, fallbackLoraPath: nil))) { error in
            XCTAssertTrue(error.localizedDescription.contains("document.citations"))
        }
    }
}
#endif
