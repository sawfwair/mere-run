import XCTest
@testable import MereRunCore

final class ChatToolSearchTests: XCTestCase {
    private let tools = [
        ToolDefinition(name: "files_read", description: "Read file contents", parameters: [:]),
        ToolDefinition(name: "files_write", description: "Write file contents", parameters: [:]),
        ToolDefinition(name: "weather", description: "Weather for a city", parameters: [:])]
    func testRegexBM25LimitsAndDeterministicOrdering() throws {
        let search = ChatToolSearch(catalog: tools.reversed(), searchTools: ["regex": .regex, "bm25": .bm25])
        XCTAssertEqual(try search.search(name: "regex", input: ["pattern": .string("files_.*")]).map(\.name), ["files_read", "files_write"])
        XCTAssertEqual(try search.search(name: "bm25", input: ["query": .string("read file")]).first?.name, "files_read")
        XCTAssertEqual(try search.search(name: "regex", input: ["pattern": .string("files"), "limit": .number(1)]).count, 1)
        XCTAssertThrowsError(try search.search(name: "regex", input: ["pattern": .string("(")]))
        XCTAssertThrowsError(try search.search(name: "regex", input: ["pattern": .string(String(repeating: ".", count: 201))]))
        XCTAssertThrowsError(try search.search(name: "bm25", input: ["query": .string("x"), "limit": .number(0)]))
    }
}
