import Foundation
import XCTest
@testable import MereRunCore

final class EmbeddingGemma2Tests: XCTestCase {
    func testFP16IsRejectedBeforeLoadingResources() async {
        do {
            _ = try await EmbeddingGemma2Model(resources: EmbeddingGemma2Resources(rootURL: URL(fileURLWithPath: "/missing")),
                                             dtype: .float16)
            XCTFail("FP16 must be rejected.")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("bfloat16 or float32"))
        }
    }

    func testOfficialTaskPrefixes() {
        XCTAssertEqual(EmbeddingGemma2Task.query.format("hello"), "task: search result | query: hello")
        XCTAssertEqual(EmbeddingGemma2Task.document.format("hello"), "title: none | text: hello")
        XCTAssertEqual(EmbeddingGemma2Task.document.format("hello", title: "File"), "title: File | text: hello")
        XCTAssertEqual(EmbeddingGemma2Task.codeRetrieval.format("sort"), "task: code retrieval | query: sort")
        XCTAssertEqual(EmbeddingGemma2Task.questionAnswering.format("Q"), "task: question answering | query: Q")
        XCTAssertEqual(EmbeddingGemma2Task.factChecking.format("F"), "task: fact checking | query: F")
        XCTAssertEqual(EmbeddingGemma2Task.classification.format("C"), "task: classification | query: C")
        XCTAssertEqual(EmbeddingGemma2Task.clustering.format("C"), "task: clustering | query: C")
        XCTAssertEqual(EmbeddingGemma2Task.similarity.format("S"), "task: sentence similarity | query: S")
        XCTAssertEqual(EmbeddingGemma2Task.raw.format("raw"), "raw")
    }

    func testResourceValidationSupportsSingleAndShardedWeights() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = EmbeddingGemma2Resources(rootURL: root)
        XCTAssertEqual(resources.validate().count, 4)
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"] {
            try Data().write(to: root.appending(path: file))
        }
        XCTAssertTrue(resources.validate().isEmpty)
        try FileManager.default.removeItem(at: resources.indexURL)
        try Data().write(to: resources.weightsURL)
        XCTAssertTrue(resources.validate().isEmpty)
    }
}
