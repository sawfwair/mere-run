import Foundation
import XCTest
@testable import MereRunCore

final class DeepseekV4FlashJSONRepairTests: XCTestCase {
    func testEscapesLiteralNewlinesInsideStrings() throws {
        let raw = Data("""
        {"choices":[{"message":{"content":"line one
        line two","reasoning_content":"thought one
        thought two"}}]}
        """.utf8)

        XCTAssertThrowsError(try JSONSerialization.jsonObject(with: raw))

        let repaired = DeepseekV4FlashJSONRepair.escapingControlCharactersInsideStrings(raw)
        let object = try JSONSerialization.jsonObject(with: repaired) as? [String: Any]
        let choices = object?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]

        XCTAssertEqual(message?["content"] as? String, "line one\nline two")
        XCTAssertEqual(message?["reasoning_content"] as? String, "thought one\nthought two")
    }

    func testLeavesWhitespaceOutsideStringsUntouched() throws {
        let raw = Data("""
        {
          "content": "already\\nescaped"
        }
        """.utf8)

        let repaired = DeepseekV4FlashJSONRepair.escapingControlCharactersInsideStrings(raw)

        XCTAssertEqual(String(data: repaired, encoding: .utf8), String(data: raw, encoding: .utf8))
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: repaired))
    }
}
