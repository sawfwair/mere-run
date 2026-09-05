import XCTest
@testable import MereRunCore

final class IncrementalTokenTextDecoderTests: XCTestCase {
    func testBuffersTrailingReplacementScalarsUntilSequenceResolves() {
        var decoder = IncrementalTokenTextDecoder()

        XCTAssertEqual(decoder.append(decodedText: "can\u{FFFD}"), "can")
        XCTAssertEqual(decoder.append(decodedText: "can\u{FFFD}\u{FFFD}"), "")
        XCTAssertEqual(decoder.append(decodedText: "can‑"), "‑")
        XCTAssertEqual(decoder.append(decodedText: "can‑do"), "do")
    }

    func testTracksUTF8BytesAcrossExtendedGraphemeChanges() {
        var decoder = IncrementalTokenTextDecoder()

        XCTAssertEqual(decoder.append(decodedText: "e"), "e")
        XCTAssertEqual(decoder.append(decodedText: "e\u{0301}"), "\u{0301}")
    }
}
