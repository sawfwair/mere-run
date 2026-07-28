import Foundation
import XCTest
@testable import MereRunApp

final class StudioFirstClassWorkspaceTests: XCTestCase {
    func testEmbeddingTemplateMapsOneLineToEachCLITextArgument() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .textEmbed))
        var draft = template.defaultDraft()
        draft.prompt = "query one\nquery two"
        draft.outputPath = "/tmp/vectors.json"

        let arguments = template.arguments(from: draft)
        XCTAssertEqual(Array(arguments.prefix(4)), ["text", "embed", "query one", "query two"])
        XCTAssertTrue(arguments.contains("/tmp/vectors.json"))
    }

    func testEmbeddingDocumentDecodesVectorsAndCosineSimilarity() throws {
        let data = Data(
            """
            {
              "model": "text-embed-qwen3-0.6b",
              "data": [
                {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                {"index": 1, "embedding": [0.8, 0.6, 0.0]}
              ],
              "usage": {"prompt_tokens": 7, "total_tokens": 7}
            }
            """.utf8
        )

        let document = try XCTUnwrap(StudioEmbeddingDocument.decode(data))
        XCTAssertEqual(document.model, "text-embed-qwen3-0.6b")
        XCTAssertEqual(document.promptTokens, 7)
        XCTAssertEqual(document.vectors.count, 2)
        XCTAssertEqual(document.vectors[0].norm, 1, accuracy: 0.0001)
        XCTAssertEqual(
            document.cosineSimilarity(document.vectors[0], document.vectors[1]),
            0.8,
            accuracy: 0.0001
        )
    }

    func testAnonymizationDocumentDecodesProtectedTextAndSpans() throws {
        let data = Data(
            """
            {
              "model": "text-anonymize-privacy-filter",
              "data": [{
                "text": "Email alice@example.com",
                "anonymized_text": "Email [EMAIL]",
                "token_count": 4,
                "spans": [{
                  "label": "EMAIL",
                  "text": "alice@example.com",
                  "startToken": 1,
                  "endToken": 3
                }]
              }]
            }
            """.utf8
        )

        let document = try XCTUnwrap(StudioAnonymizationDocument.decode(data))
        XCTAssertEqual(document.results.first?.anonymizedText, "Email [EMAIL]")
        XCTAssertEqual(document.results.first?.spans.first?.label, "EMAIL")
        XCTAssertEqual(document.results.first?.spans.first?.startToken, 1)
    }

    func testDatasetDiscoveryDocumentDecodesCandidateDiagnostics() throws {
        let data = Data(
            """
            {
              "summary": "Found two candidates.",
              "result": {
                "scanned_directory_count": 8,
                "candidates": [{
                  "id": "portraits",
                  "name": "Portraits",
                  "path": "/tmp/portraits",
                  "status": "warning",
                  "trainable": true,
                  "image_count": 12,
                  "caption_count": 11,
                  "usable_pair_count": 10,
                  "diagnostics": [{"title": "Missing captions", "message": "Two images need captions."}]
                }]
              },
              "diagnostics": []
            }
            """.utf8
        )

        let document = try XCTUnwrap(StudioDatasetDiscoveryDocument.decode(data))
        XCTAssertEqual(document.scannedDirectories, 8)
        XCTAssertEqual(document.candidates.first?.usablePairs, 10)
        XCTAssertEqual(document.candidates.first?.problems, ["Missing captions: Two images need captions."])
    }

    func testRunPlanDocumentExtractsMaterializedPaths() throws {
        let data = Data(
            """
            {
              "status": "ok",
              "summary": "Materialized run.",
              "result": {
                "run_directory": "/tmp/run",
                "plan_path": "/tmp/run/plan.json",
                "nested": {"events_path": "/tmp/run/events.jsonl"}
              },
              "diagnostics": [{"title": "Output relocated", "message": "Output is durable."}]
            }
            """.utf8
        )

        let document = try XCTUnwrap(StudioRunPlanDocument.decode(data))
        XCTAssertEqual(document.status, "ok")
        XCTAssertTrue(document.paths.contains { $0.path == "/tmp/run/plan.json" })
        XCTAssertTrue(document.paths.contains { $0.path == "/tmp/run/events.jsonl" })
        XCTAssertEqual(document.diagnostics, ["Output relocated: Output is durable."])
    }

    func testNPYMetadataReadsShapeAndDescriptor() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("studio-npy-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 8, 32), }"
        let remainder = (16 - ((10 + header.utf8.count + 1) % 16)) % 16
        header += String(repeating: " ", count: remainder) + "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        data.append(UInt8(header.utf8.count & 0xff))
        data.append(UInt8((header.utf8.count >> 8) & 0xff))
        data.append(Data(header.utf8))
        data.append(Data(repeating: 0, count: 16))
        let url = root.appendingPathComponent("latents.npy")
        try data.write(to: url)

        let metadata = try XCTUnwrap(StudioNPYMetadata.load(from: url))
        XCTAssertEqual(metadata.version, "1.0")
        XCTAssertEqual(metadata.descriptor, "<f4")
        XCTAssertEqual(metadata.shape, "(1, 8, 32)")
        XCTAssertFalse(metadata.fortranOrder)
    }

    func testTrainingDatasetInspectorFindsImageCaptionPairs() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("studio-training-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: root.appendingPathComponent("frame.png"))
        try Data("a subject in warm light".utf8).write(to: root.appendingPathComponent("frame.txt"))

        let snapshot = StudioTrainingDatasetSnapshot.inspect(kind: .image, path: root.path)
        XCTAssertEqual(snapshot.totalRecords, 1)
        XCTAssertEqual(snapshot.usableRecords, 1)
        XCTAssertEqual(snapshot.previews.first?.detail, "a subject in warm light")
    }

    func testMIDISummaryReadsNoteOnAndOff() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("studio-midi-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let data = Data([
            0x4d, 0x54, 0x68, 0x64,
            0x00, 0x00, 0x00, 0x06,
            0x00, 0x00,
            0x00, 0x01,
            0x01, 0xe0,
            0x4d, 0x54, 0x72, 0x6b,
            0x00, 0x00, 0x00, 0x0d,
            0x00, 0x90, 0x3c, 0x64,
            0x83, 0x60, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ])
        let url = root.appendingPathComponent("note.mid")
        try data.write(to: url)

        let summary = try XCTUnwrap(StudioMIDISummary.load(from: url))
        XCTAssertEqual(summary.trackCount, 1)
        XCTAssertEqual(summary.ticksPerQuarter, 480)
        XCTAssertEqual(summary.notes.count, 1)
        XCTAssertEqual(summary.notes.first?.pitch, 60)
        XCTAssertEqual(summary.notes.first?.durationTicks, 480)
    }
}
