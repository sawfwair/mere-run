import XCTest
@testable import MereRunCLI

final class H3ImageBenchmarkCommandTests: XCTestCase {
    func testColdWarmBenchmarkConfiguration() throws {
        let command = try ModelBenchmarkH3Image.parse([
            "--model-root", "/tmp/h3-q8", "--prompt", "A camera", "--size", "4096",
            "--runs", "3", "--resident-bf16", "--attention-heads", "56",
            "--attention-queries", "4096", "--vae-tile-batch", "32",
            "--output", "/tmp/h3-benchmark-\(UUID().uuidString)", "--json"
        ])
        XCTAssertEqual(command.size, 4_096)
        XCTAssertTrue(command.residentBF16)
        XCTAssertEqual(command.attentionHeads, 56)
        XCTAssertEqual(command.attentionQueries, 4_096)
        XCTAssertEqual(command.vaeTileBatch, 32)
        XCTAssertNoThrow(try command.validate())
        var invalid = command
        invalid.attentionHeads = 0
        XCTAssertThrowsError(try invalid.validate())
        invalid = command
        invalid.attentionQueries = 0
        XCTAssertThrowsError(try invalid.validate())
        invalid = command
        invalid.vaeTileBatch = 0
        XCTAssertThrowsError(try invalid.validate())
    }
}
