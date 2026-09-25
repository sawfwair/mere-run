import Foundation
import XCTest
@testable import MereRunCore

final class GLiNERTests: MereRunCoreTestCase {
    func testManagedModelAdvertisesClassification() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: GLiNERCatalog.modelID))
        XCTAssertEqual(spec.category, .textClassify)
        XCTAssertEqual(spec.apiProfile?.task, .textClassifications)
        XCTAssertEqual(spec.upstreamRevision, GLiNERCatalog.revision)
        let manifest = MereRunModelManifest.template(for: .gliner25Decide)
        XCTAssertEqual(manifest.supports, [.textClassification, .textExtraction])
    }

    func testPinnedCheckpointTokenizationAndClassification() throws {
        guard let path = ProcessInfo.processInfo.environment["GLINER_CHECKPOINT"] else {
            throw XCTSkip("Set GLINER_CHECKPOINT to run pinned checkpoint parity.")
        }
        let operation = try GLiNERClassificationOperation(root: URL(fileURLWithPath: path), modelID: GLiNERCatalog.modelID)
        let request = GLiNERClassificationRequest(
            text: "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
            tasks: [
                GLiNERClassificationTask(name: "sentiment", labels: ["positive", "negative", "mixed", "neutral"]),
                GLiNERClassificationTask(name: "aspects", labels: ["battery", "keyboard", "screen", "camera", "price", "support"],
                                         multiLabel: true, threshold: 0.4)
            ])
        let encoded = try operation.encode(request)
        XCTAssertEqual(encoded.ids, [
            287, 128003, 9481, 287, 128007, 1453, 128007, 2330, 128007, 3230, 128007, 5678, 1263, 1263,
            128001, 287, 128003, 2592, 287, 128007, 2643, 128007, 6125, 128007, 1531, 128007, 1822,
            128007, 710, 128007, 523, 1263, 1263, 128002, 2643, 9745, 416, 2657, 366, 304, 262, 6125,
            263, 262, 1531, 281, 262, 410, 584, 286, 427, 277, 266, 4411, 323
        ])
        XCTAssertEqual(encoded.markers, [4, 6, 8, 10, 19, 21, 23, 25, 27, 29])
        let response = try operation.predict(request)
        XCTAssertEqual(response.heads["sentiment"]?.labels, ["positive"])
        XCTAssertEqual(response.heads["aspects"]?.labels, ["battery", "keyboard", "screen"])
        let referenceProbabilities: [String: [String: Double]] = [
            "sentiment": ["positive": 0.999493748, "negative": 0.000139468, "mixed": 0.000169864,
                          "neutral": 0.000196921],
            "aspects": ["battery": 0.761035372, "keyboard": 0.993998783, "screen": 0.984621025,
                        "camera": 0.000963633, "price": 0.001192867, "support": 0.003610026]
        ]
        for (head, probabilities) in referenceProbabilities {
            for (label, expected) in probabilities {
                XCTAssertEqual(response.heads[head]?.probabilities[label] ?? -1, expected, accuracy: 0.00001,
                               "\(head).\(label) differs from the pinned Python eval-mode reference")
            }
        }
    }

    func testPinnedCheckpointEntityExtraction() throws {
        guard let path = ProcessInfo.processInfo.environment["GLINER_CHECKPOINT"] else {
            throw XCTSkip("Set GLINER_CHECKPOINT to run pinned checkpoint parity.")
        }
        let operation = try GLiNERClassificationOperation(root: URL(fileURLWithPath: path), modelID: GLiNERCatalog.modelID)
        let request = GLiNERExtractionRequest(text: "Alice Smith joined Acme in Paris in 2024.", entities: [
            GLiNERExtractionTerm(name: "person"), GLiNERExtractionTerm(name: "organization"),
            GLiNERExtractionTerm(name: "location")
        ])
        let plan = try operation.prepare(request)
        XCTAssertEqual(plan.inputTokens, 23)
        XCTAssertEqual(plan.wordCount, 9)
        let result = try operation.predict(request)
        XCTAssertEqual(result.entities["person"]?.first?.text, "Alice Smith")
        XCTAssertEqual(result.entities["organization"]?.first?.text, "Acme")
        XCTAssertEqual(result.entities["location"]?.first?.text, "Paris")
        XCTAssertEqual(result.entities["person"]?.first?.confidence ?? -1, 0.999427199, accuracy: 0.00001)
        XCTAssertEqual(result.entities["organization"]?.first?.confidence ?? -1, 0.999817550, accuracy: 0.00001)
        XCTAssertEqual(result.entities["location"]?.first?.confidence ?? -1, 0.999998808, accuracy: 0.00001)
        XCTAssertEqual(result.entities["person"]?.first?.start, 0)
        XCTAssertEqual(result.entities["person"]?.first?.end, 11)
    }

    func testPinnedCheckpointRelationsAndStructure() throws {
        guard let path = ProcessInfo.processInfo.environment["GLINER_CHECKPOINT"] else {
            throw XCTSkip("Set GLINER_CHECKPOINT to run pinned checkpoint parity.")
        }
        let operation = try GLiNERClassificationOperation(root: URL(fileURLWithPath: path), modelID: GLiNERCatalog.modelID)
        let text = "Alice Smith joined Acme in Paris in 2024."
        let relations = try operation.predict(GLiNERExtractionRequest(text: text, relations: [
            GLiNERExtractionTerm(name: "works_for"), GLiNERExtractionTerm(name: "located_in")
        ]))
        XCTAssertEqual(relations.relations["works_for"]?.first?.head.text, "Alice Smith")
        XCTAssertEqual(relations.relations["works_for"]?.first?.tail.text, "Acme")
        XCTAssertEqual(relations.relations["located_in"]?.first?.head.text, "Acme")
        XCTAssertEqual(relations.relations["located_in"]?.first?.tail.text, "Paris")
        let structure = try operation.predict(GLiNERExtractionRequest(text: text, structures: [
            GLiNERExtractionStructure(name: "employment", fields: [
                GLiNERExtractionField(name: "person"), GLiNERExtractionField(name: "organization"),
                GLiNERExtractionField(name: "location")
            ])
        ]))
        XCTAssertEqual(structure.structures["employment"]?.first?["person"]?.first?.text, "Alice Smith")
        XCTAssertEqual(structure.structures["employment"]?.first?["organization"]?.first?.text, "Acme")
        XCTAssertEqual(structure.structures["employment"]?.first?["location"]?.first?.text, "Paris")
    }

    func testLongExtractionRemapsOffsetsAndBatchKeepsRequestOrder() throws {
        guard let path = ProcessInfo.processInfo.environment["GLINER_CHECKPOINT"] else {
            throw XCTSkip("Set GLINER_CHECKPOINT to run pinned checkpoint parity.")
        }
        let operation = try GLiNERClassificationOperation(root: URL(fileURLWithPath: path), modelID: GLiNERCatalog.modelID)
        let sentence = "Alice Smith joined Acme in Paris in 2024."
        let request = GLiNERExtractionRequest(text: sentence + " " + sentence,
                                              entities: [GLiNERExtractionTerm(name: "person")])
        let plans = try operation.prepareLong(request, chunkSize: 9, chunkOverlap: 0)
        XCTAssertEqual(plans.count, 2)
        let response = try operation.predictLong(request, chunkSize: 9, chunkOverlap: 0)
        XCTAssertEqual(response.entities["person"]?.map(\.start), [0, 42])
        let batch = try operation.predictBatch([request, GLiNERExtractionRequest(text: sentence,
            entities: [GLiNERExtractionTerm(name: "person")])])
        XCTAssertEqual(batch.count, 2)
        XCTAssertEqual(batch[1].entities["person"]?.first?.start, 0)
    }

    func testPinnedCheckpointJointSchemaSharesOneEncoderPass() throws {
        guard let path = ProcessInfo.processInfo.environment["GLINER_CHECKPOINT"] else {
            throw XCTSkip("Set GLINER_CHECKPOINT to run pinned checkpoint parity.")
        }
        let operation = try GLiNERClassificationOperation(root: URL(fileURLWithPath: path), modelID: GLiNERCatalog.modelID)
        let response = try operation.predict(GLiNERExtractionRequest(
            text: "Alice Smith joined Acme in Paris in 2024.",
            entities: [GLiNERExtractionTerm(name: "person"), GLiNERExtractionTerm(name: "organization")],
            relations: [GLiNERExtractionTerm(name: "works_for")],
            classifications: [GLiNERClassificationTask(name: "tone", labels: ["positive", "negative"])]))
        XCTAssertEqual(response.entities["person"]?.first?.text, "Alice Smith")
        XCTAssertEqual(response.entities["organization"]?.first?.text, "Acme")
        XCTAssertEqual(response.relations["works_for"]?.first?.tail.text, "Acme")
        XCTAssertEqual(response.classifications["tone"]?.labels, ["positive"])
        XCTAssertEqual(response.classifications["tone"]?.probabilities["positive"] ?? -1,
                       0.678021550, accuracy: 0.00001)
    }
}
