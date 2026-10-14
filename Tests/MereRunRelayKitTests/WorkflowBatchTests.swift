import Foundation
import XCTest
@testable import MereRunRelayKit

final class WorkflowBatchTests: XCTestCase {
    private func graph() throws -> WorkflowGraphDocument {
        try JSONDecoder().decode(WorkflowGraphDocument.self, from: Data(#"""
        {"schema_version":1,"kind":"mere.run/workflow-graph","name":"batch-fixture",
         "inputs":{"prompt":{"type":"string"}},
         "nodes":[{"id":"generate","kind":"image.generate","arguments":{
         "prompt":{"$ref":"inputs.prompt"},"model":"image-klein-9b"}}],
         "outputs":{"image":{"$ref":"nodes.generate.outputs.image"}}}
        """#.utf8))
    }

    func testReplayKeepsWholeGraphsJobIdentitiesSeedsAndExactDocumentBytes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let rows = [WorkflowBatchInput(key: "first", inputs: .init(values: ["prompt": .string("one")])),
                    WorkflowBatchInput(key: "second", inputs: .init(values: ["prompt": .string("two")]))]
        let first = try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root, environment: .portable)
        let paths = first.items.map { root.appendingPathComponent($0.bundleDirectory).appendingPathComponent("graph.json") }
        let bytes = try paths.map { try Data(contentsOf: $0) }
        let replay = try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root, environment: .portable)
        XCTAssertEqual(first, replay)
        XCTAssertEqual(Set(first.items.map(\.jobID)).count, 2)
        XCTAssertEqual(bytes, try paths.map { try Data(contentsOf: $0) })
        for item in first.items {
            let bundle = try verifiedPortableWorkflowBundle(at: root.appendingPathComponent(item.bundleDirectory), pluginNodes: [])
            XCTAssertEqual(bundle.graph.nodes.count, 1)
            XCTAssertEqual(bundle.job.jobID, item.jobID)
            XCTAssertEqual(bundle.graph.nodes[0].arguments["seed"], .integer(item.seed))
        }
    }

    func testChangedRowsOrAdmissionPolicyCannotReuseTheBatchDirectory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let rows = [WorkflowBatchInput(key: "first", inputs: .init(values: ["prompt": .string("one")]))]
        _ = try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root, environment: .portable)
        XCTAssertThrowsError(try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root,
                                                                      environment: .portable, maxAttempts: 2))
        let changed = [WorkflowBatchInput(key: "first", inputs: .init(values: ["prompt": .string("changed")]))]
        XCTAssertThrowsError(try WorkflowBatchMaterializer.materialize(graph: graph(), items: changed, destination: root, environment: .portable))
    }

    func testInterruptedPreparationPreservesOriginalDirectoryAndUsesPinnedSeed() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let rows = [WorkflowBatchInput(key: "first", inputs: .init(values: ["prompt": .string("one")]))]
        let first = try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root, environment: .portable)
        let directory = root.appendingPathComponent(first.items[0].bundleDirectory)
        let graphBytes = try Data(contentsOf: directory.appendingPathComponent("graph.json"))
        try FileManager.default.removeItem(at: directory.appendingPathComponent("job.json"))
        let replay = try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root, environment: .portable)
        XCTAssertEqual(first, replay)
        XCTAssertEqual(graphBytes, try Data(contentsOf: directory.appendingPathComponent("graph.json")))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.deletingLastPathComponent().path)
            .contains(where: { $0.hasPrefix("0.incomplete-") }))
    }

    func testDuplicateKeysInvalidKeysAndBoundsFailBeforeCreatingAnyBundle() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let row = WorkflowBatchInput(key: "row", inputs: .init(values: ["prompt": .string("one")]))
        for rows in [[], [row, row], [WorkflowBatchInput(key: "../bad", inputs: row.inputs)]] {
            XCTAssertThrowsError(try WorkflowBatchMaterializer.materialize(graph: graph(), items: rows, destination: root, environment: .portable))
        }
        XCTAssertThrowsError(try WorkflowBatchMaterializer.materialize(graph: graph(), items: [row], destination: root, environment: .portable, maxInFlight: 21))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    func testBatchRequestPreserves64BitSeedsInCanonicalBundleDocuments() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = try WorkflowBundleMaterializer(graph: graph(), suppliedInputs: .init(values: ["prompt": .string("seed")]),
                                                   destination: root, environment: .portable, seed: { Int64.max }).materialize()
        let request = try RelayGraphCreateRequest(bundleDirectory: root)
        XCTAssertEqual(request.bundleDocuments["graph.json"], try Data(contentsOf: root.appendingPathComponent("graph.json")))
        XCTAssertTrue(String(decoding: try WorkflowBundleCodec.encoder().encode(request), as: UTF8.self).contains(String(Int64.max)))
        XCTAssertEqual(bundle.graph.nodes[0].arguments["seed"], .integer(Int64.max))
    }
}
