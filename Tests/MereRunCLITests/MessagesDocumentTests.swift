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
    func testVisionPDFRenderingPreservesOrderAndRejectsTextOnlyScans() throws {
        let source = try pdf(pages: 2)
        let rendered = try MediaDocumentIO.pdf(source, renderPages: true)
        XCTAssertEqual(rendered.pageImages?.count, 2)
        XCTAssertNotEqual(rendered.pageImages?[0], rendered.pageImages?[1])
        for data in rendered.pageImages ?? [] {
            let image = try MediaImageIO.decode(data: data)
            XCTAssertGreaterThan(image.width, 0)
            XCTAssertLessThanOrEqual(max(image.width, image.height), 1536)
        }
        let scan = PDFDocument()
        let image = NSImage(size: NSSize(width: 80, height: 80), flipped: false) { rect in
            NSColor.red.setFill(); rect.fill(); return true
        }
        scan.insert(try XCTUnwrap(PDFPage(image: image)), at: 0)
        let data = try XCTUnwrap(scan.dataRepresentation())
        XCTAssertThrowsError(try MediaDocumentIO.pdf(data))
        let document = try MediaDocumentIO.pdf(data, renderPages: true)
        XCTAssertEqual(document.pages, [""])
        XCTAssertEqual(document.pageImages?.count, 1)
        let raster = try MediaImageIO.decode(data: XCTUnwrap(document.pageImages?.first))
        let center = ((raster.height / 2) * raster.width + raster.width / 2) * 4
        XCTAssertGreaterThan(raster.rgba8[center], 200)
        XCTAssertLessThan(raster.rgba8[center + 1], 30)
        XCTAssertLessThan(raster.rgba8[center + 2], 30)
    }

    func testRotatedCropRendersTheSelectedPageArea() throws {
        let output = NSMutableData()
        let consumer = try XCTUnwrap(CGDataConsumer(data: output))
        var bounds = CGRect(x: 0, y: 0, width: 300, height: 200)
        let context = try XCTUnwrap(CGContext(consumer: consumer, mediaBox: &bounds, nil))
        context.beginPDFPage(nil)
        context.setFillColor(NSColor.blue.cgColor); context.fill(bounds)
        let crop = CGRect(x: 40, y: 30, width: 160, height: 80)
        context.setFillColor(NSColor.red.cgColor); context.fill(crop)
        context.endPDFPage(); context.closePDF()
        let pdf = try XCTUnwrap(PDFDocument(data: output as Data))
        let page = try XCTUnwrap(pdf.page(at: 0))
        page.setBounds(crop, for: .cropBox); page.rotation = 90
        let rendered = try MediaDocumentIO.pdf(XCTUnwrap(pdf.dataRepresentation()), renderPages: true)
        let raster = try MediaImageIO.decode(data: XCTUnwrap(rendered.pageImages?.first))
        XCTAssertEqual(raster.width, 160); XCTAssertEqual(raster.height, 320)
        for (x, y) in [(20, 20), (80, 160), (140, 300)] {
            let offset = (y * raster.width + x) * 4
            XCTAssertGreaterThan(raster.rgba8[offset], 200)
            XCTAssertLessThan(raster.rgba8[offset + 2], 30)
        }
    }

    func testCanonicalVisionDocumentOrderAndSharedRenderingBudget() throws {
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        let configuration = RuntimeChatConfiguration(modelID: MuseGlimmerResources.modelId, engine: .textChatMuseGlimmer,
            settings: .init(), profile: profile, contextSize: 32768, fallbackLoraPath: nil)
        let source = try pdf(pages: 2).base64EncodedString()
        let wire = try JSONDecoder().decode(MessagesBlock.self, from: Data((#"{"type":"document","title":"Local report","context":"Read both pages","source":{"type":"base64","media_type":"application/pdf","data":""# + source + #""}}"#).utf8))
        let continuity = try MessagesContinuity(key: Data(repeating: 1, count: 32))
        let blocks = try MessagesAdapter.blocks([wire], configuration: configuration, continuity: continuity)
        guard case .document(let document) = blocks.first else { return XCTFail("missing canonical document") }
        XCTAssertEqual(document.title, "Local report")
        XCTAssertEqual(document.pageImages?.count, 2)
        XCTAssertTrue(document.pages[0].contains("Local page 1"))
        let budget = MessagesDocumentBudget()
        budget.remainingPixels = 1
        XCTAssertThrowsError(try MessagesAdapter.blocks([wire], configuration: configuration, continuity: continuity, documentBudget: budget))
        XCTAssertThrowsError(try MediaDocumentIO.pdf(pdf(), renderPages: true, maximumRenderedPixels: 1))
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
