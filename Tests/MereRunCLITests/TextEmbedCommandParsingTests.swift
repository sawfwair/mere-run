import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class TextEmbedCommandParsingTests: XCTestCase {
    func testDefaultQwenContractIsPreserved() throws {
        let command = try TextEmbed.parse(["hello", "world"])
        XCTAssertEqual(command.texts, ["hello", "world"])
        XCTAssertNil(command.model)
        XCTAssertEqual(command.task, "raw")
        XCTAssertNil(command.dimensions)
    }

    func testEmbeddingGemmaOptions() throws {
        let command = try TextEmbed.parse(["swift sort", "--model", EmbeddingGemma2Catalog.modelID,
                                          "--task", "document", "--title", "sort.swift", "--dimensions", "256",
                                          "--max-tokens", "512", "--pretty"])
        XCTAssertEqual(command.task, "document")
        XCTAssertEqual(command.title, "sort.swift")
        XCTAssertEqual(command.dimensions, 256)
        XCTAssertEqual(command.maxTokens, 512)
        XCTAssertTrue(command.pretty)
    }

    func testInvalidInputFailsBeforeRuntimeLoading() {
        for args in [[], ["hello", "--max-tokens", "0"], ["hello", "--task", "unknown"],
                     ["hello", "--title", "Title"], ["hello", "--dimensions", "256"],
                     ["hello", "--model", EmbeddingGemma2Catalog.modelID, "--dimensions", "64"],
                     ["hello", "--model", EmbeddingGemma2Catalog.modelID, "--max-tokens", "1"]] {
            XCTAssertThrowsError(try TextEmbed.parse(args), "Expected rejection for \(args)")
        }
    }

    func testMediaOnlyAndOrderedJSONParsing() throws {
        let model = ["--model", EmbeddingGemma2Catalog.modelID]
        let direct = try TextEmbed.parse(["--image", "one.png", "two.jpg", "--audio", "sound.wav", "--video", "clip.mp4"] + model)
        XCTAssertTrue(direct.texts.isEmpty)
        XCTAssertEqual(direct.image, ["one.png", "two.jpg"])
        XCTAssertEqual(direct.audio, ["sound.wav"])
        XCTAssertEqual(direct.video, ["clip.mp4"])
        XCTAssertEqual(try TextEmbed.parse(["--input-json", "-"] + model).inputJSON, "-")
        for args in [["--image", "one.png"], ["--input-json", "inputs.json"],
                     ["--image", "https://example.com/a.png"] + model,
                     ["hello", "--input-json", "inputs.json"] + model] {
            XCTAssertThrowsError(try TextEmbed.parse(args))
        }
    }

    func testManagedModelPinAndScope() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: EmbeddingGemma2Catalog.modelID))
        XCTAssertEqual(spec.category, .textEmbed)
        XCTAssertEqual(spec.validationKind, .embeddingGemma2)
        XCTAssertEqual(spec.upstreamRepoId, "google/embeddinggemma-2")
        XCTAssertEqual(spec.upstreamRevision, EmbeddingGemma2Catalog.revision)
        XCTAssertEqual(spec.defaultCLICommands, ["text embed"])
        XCTAssertNotNil(InstalledModelSmokePlans.plan(for: spec, installedIDs: [spec.id]))
        let manifest = MereRunModelManifest.template(for: .embeddingGemma2)
        XCTAssertEqual(manifest.engine, .embeddingGemma2)
        XCTAssertEqual(manifest.family, .embed)
        XCTAssertEqual(manifest.supports, [.textEmbedding, .multimodalEmbedding])
        XCTAssertEqual(manifest.upstreamRepoId, EmbeddingGemma2Catalog.repository)
        XCTAssertFalse(spec.isAPISidecarRuntimeModel)
    }

    func testLocalModelDispatchReadsTypedConfig() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = root.appending(path: "config.json")
        try Data(#"{"model_type":"embedding_gemma2"}"#.utf8).write(to: config)
        XCTAssertTrue(try TextEmbed.isEmbeddingGemma2(root: root))
        try Data(#"{"model_type":"qwen3"}"#.utf8).write(to: config)
        XCTAssertFalse(try TextEmbed.isEmbeddingGemma2(root: root))
        try Data(#"{"model_type":"gemma4"}"#.utf8).write(to: config)
        XCTAssertThrowsError(try TextEmbed.isEmbeddingGemma2(root: root))
    }
}
