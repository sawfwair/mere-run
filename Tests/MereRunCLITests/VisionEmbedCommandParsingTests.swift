import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class VisionEmbedCommandParsingTests: XCTestCase {
    func testParsesTextAndImageBatchOptions() throws {
        let command = try VisionEmbed.parse([
            "--text", "a white SUV", "a maroon pickup truck",
            "--image", "/tmp/one.png", "/tmp/two.png",
            "--instruction", "Retrieve similar vehicles",
            "--model", "/tmp/qwen-vl-embed",
            "--dimensions", "1024",
            "--max-tokens", "4096",
            "--min-pixels", "4096",
            "--max-pixels", "1048576",
            "--output", "/tmp/embeddings.json",
            "--pretty",
        ])

        XCTAssertEqual(command.text, ["a white SUV", "a maroon pickup truck"])
        XCTAssertEqual(command.image, ["/tmp/one.png", "/tmp/two.png"])
        XCTAssertEqual(command.instruction, "Retrieve similar vehicles")
        XCTAssertEqual(command.model, "/tmp/qwen-vl-embed")
        XCTAssertEqual(command.dimensions, 1_024)
        XCTAssertEqual(command.maxTokens, 4_096)
        XCTAssertEqual(command.minPixels, 4_096)
        XCTAssertEqual(command.maxPixels, 1_048_576)
        XCTAssertEqual(command.output, "/tmp/embeddings.json")
        XCTAssertTrue(command.pretty)
    }

    func testDefaultsMatchPublishedModelContract() throws {
        let command = try VisionEmbed.parse(["--text", "a white SUV"])

        XCTAssertNil(command.model)
        XCTAssertNil(command.dimensions)
        XCTAssertEqual(command.maxTokens, Qwen3VLEmbeddingCatalog.defaultMaxTokens)
        XCTAssertEqual(command.minPixels, Qwen3VLEmbeddingCatalog.defaultMinPixels)
        XCTAssertEqual(command.maxPixels, Qwen3VLEmbeddingCatalog.defaultMaxPixels)
        XCTAssertNoThrow(try command.validate())
    }

    func testRejectsConflictingJSONAndDirectInputs() {
        XCTAssertThrowsError(
            try VisionEmbed.parse([
                "--input-json", "/tmp/batch.json",
                "--text", "a white SUV",
            ])
        )
    }

    func testRejectsMissingOrInvalidInputs() throws {
        XCTAssertThrowsError(try VisionEmbed.parse([]).validate())
        XCTAssertThrowsError(
            try VisionEmbed.parse(["--text", "vehicle", "--dimensions", "2049"]).validate()
        )
        XCTAssertThrowsError(
            try VisionEmbed.parse(["--image", "/tmp/one.png", "--max-tokens", "0"]).validate()
        )
    }
}
