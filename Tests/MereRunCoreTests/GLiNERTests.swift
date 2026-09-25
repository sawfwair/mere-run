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
        XCTAssertEqual(manifest.supports, [.textClassification])
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
}
