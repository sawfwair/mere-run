import XCTest
import MereRunCore
@testable import MereRunCLI

final class PPLXEmbedV2CommandTests: XCTestCase {
    func testLateQueryAcceptsNativeTokenVectorOptions() throws {
        let command = try TextEmbed.parse(["find it", "--model", PPLXEmbedV2Catalog.lateSmallID, "--task", "query", "--dimensions", "128"])
        try command.validate()
        XCTAssertEqual(command.texts, ["find it"])
        XCTAssertThrowsError(try TextEmbed.parse(["find it", "--model", PPLXEmbedV2Catalog.lateSmallID, "--dimensions", "256"]).validate())
    }

    func testContextualJSONAcceptsTruncationAndNormalizationOptions() throws {
        let command = try TextEmbed.parse(["--chunks-json", "chunks.json", "--model", PPLXEmbedV2Catalog.contextID,
                                          "--task", "document", "--dimensions", "1024", "--normalize"])
        try command.validate()
        XCTAssertTrue(command.normalize)
        XCTAssertThrowsError(try TextEmbed.parse(["text", "--chunks-json", "chunks.json", "--model", PPLXEmbedV2Catalog.contextID]).validate())
    }

    func testPackedManagedModelsUseTheirOriginalInputFamilies() throws {
        try TextEmbed.parse(["--image", "image.png", "--model", PPLXEmbedV2Catalog.lateQuantizedID]).validate()
        try TextEmbed.parse(["--chunks-json", "chunks.json", "--model", PPLXEmbedV2Catalog.contextQuantizedID,
                             "--dimensions", "1024", "--normalize"]).validate()
        XCTAssertThrowsError(try TextEmbed.parse(["--image", "image.png", "--model", PPLXEmbedV2Catalog.contextQuantizedID]).validate())
        XCTAssertThrowsError(try TextEmbed.parse(["text", "--model", PPLXEmbedV2Catalog.lateQuantizedID, "--normalize"]).validate())
    }

    func testImagesRequireSeparateLateDocumentBatches() throws {
        try TextEmbed.parse(["--image", "image.png", "--model", PPLXEmbedV2Catalog.lateSmallID]).validate()
        for id in [PPLXEmbedV2Catalog.lateSmallID, PPLXEmbedV2Catalog.contextID] {
            XCTAssertThrowsError(try TextEmbed.parse(["text", "--image", "image.png", "--model", id]).validate())
            XCTAssertThrowsError(try TextEmbed.parse(["--image", "image.png", "--model", id, "--task", "query"]).validate())
        }
    }

    func testInstalledSmokeHashesOnlyVectorsAndRejectsMalformedRows() throws {
        let payload = #"{"model":"a","representation":"late-interaction","dimensions":2,"normalized":true,"data":[{"index":0,"embeddings":[[1,0],[0,1]],"tokenCount":2}]}"#
        let hash = try GateRunner.pplxEmbeddingVectorHash(Data(payload.utf8))
        let changedMetadata = payload.replacingOccurrences(of: "\"model\":\"a\"", with: "\"model\":\"b\"")
        XCTAssertEqual(hash, try GateRunner.pplxEmbeddingVectorHash(Data(changedMetadata.utf8)))
        XCTAssertThrowsError(try GateRunner.pplxEmbeddingVectorHash(Data(payload.replacingOccurrences(of: "\"dimensions\":2", with: "\"dimensions\":3").utf8)))
        XCTAssertThrowsError(try GateRunner.pplxEmbeddingVectorHash(Data(payload.replacingOccurrences(of: "[[1,0],[0,1]]", with: "[]").utf8)))
    }

    func testExistingModelsRejectPPLXOnlyOptions() throws {
        XCTAssertThrowsError(try TextEmbed.parse(["text", "--normalize"]).validate())
        XCTAssertThrowsError(try TextEmbed.parse(["text", "--model", EmbeddingGemma2Catalog.modelID, "--dimensions", "2048"]).validate())
        XCTAssertThrowsError(try TextEmbed.parse(["--chunks-json", "chunks.json", "--model", EmbeddingGemma2Catalog.modelID]).validate())
    }
}
