import XCTest
@testable import MereRunCore
import AudioCore

final class ASRResultCodableTests: XCTestCase {
    func testASRResultEncodesAndDecodesAlignments() throws {
        let tokenAlignments = [
            ASRTokenAlignment(id: 0, text: "hello", startSeconds: 0.0, durationSeconds: 0.4, endSeconds: 0.4),
            ASRTokenAlignment(id: 1, text: "world", startSeconds: 0.4, durationSeconds: 0.5, endSeconds: 0.9),
        ]
        let sentenceAlignments = [
            ASRSentenceAlignment(
                text: "hello world",
                startSeconds: 0.0,
                durationSeconds: 0.9,
                endSeconds: 0.9,
                tokens: tokenAlignments
            )
        ]
        let result = ASRResult(
            text: "hello world",
            language: "en",
            duration: 1.0,
            tokenAlignments: tokenAlignments,
            sentenceAlignments: sentenceAlignments
        )

        let encoded = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(ASRResult.self, from: encoded)

        XCTAssertEqual(decoded.text, result.text)
        XCTAssertEqual(decoded.language, result.language)
        XCTAssertEqual(decoded.duration, result.duration)
        XCTAssertEqual(decoded.tokenAlignments, tokenAlignments)
        XCTAssertEqual(decoded.sentenceAlignments, sentenceAlignments)
    }

    func testASRResultDecodesLegacyPayloadWithoutAlignments() throws {
        let json = """
        {
          "text": "legacy payload",
          "language": "en",
          "duration": 2.5
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(ASRResult.self, from: json)
        XCTAssertEqual(decoded.text, "legacy payload")
        XCTAssertEqual(decoded.language, "en")
        XCTAssertEqual(decoded.duration, 2.5)
        XCTAssertNil(decoded.tokenAlignments)
        XCTAssertNil(decoded.sentenceAlignments)
    }
}
