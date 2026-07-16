import XCTest
@testable import MereRunCLI

final class VisionFaceCommandTests: XCTestCase {
    func testParsesWarmBatchOptions() throws {
        let command = try VisionFaceBatch.parse([
            "--input-list", "/tmp/photos.txt",
            "--include-embeddings",
            "--model", "/tmp/buffalo_l",
            "--execution-provider", "cpu",
            "--score-threshold", "0.7",
            "--max-faces", "3",
            "--jsonl-output", "/tmp/faces.jsonl",
            "--fail-fast",
        ])

        XCTAssertEqual(command.inputList, "/tmp/photos.txt")
        XCTAssertTrue(command.includeEmbeddings)
        XCTAssertEqual(command.modelOptions.model, "/tmp/buffalo_l")
        XCTAssertEqual(command.modelOptions.executionProvider, "cpu")
        XCTAssertEqual(command.modelOptions.scoreThreshold, 0.7)
        XCTAssertEqual(command.maxFaces, 3)
        XCTAssertEqual(command.jsonlOutput, "/tmp/faces.jsonl")
        XCTAssertTrue(command.failFast)
    }

    func testBatchRequiresImagesOrInputList() throws {
        XCTAssertThrowsError(try VisionFaceBatch.parse([]))
    }

    func testRejectsInvalidExecutionProvider() throws {
        XCTAssertThrowsError(try VisionFaceDetect.parse([
            "/tmp/face.jpg", "--execution-provider", "cuda",
        ]))
    }
}
