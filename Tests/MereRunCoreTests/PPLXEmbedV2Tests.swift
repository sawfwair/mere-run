import Foundation
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class PPLXEmbedV2Tests: XCTestCase {
    func testManagedModelsUsePinnedArtifactsAndExplicitInstall() throws {
        for id in PPLXEmbedV2Catalog.modelIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
            XCTAssertEqual(spec.validationKind, .pplxEmbedV2)
            XCTAssertEqual(spec.upstreamRevision, PPLXEmbedV2Catalog.revision(id))
            XCTAssertEqual(spec.upstreamRevision?.count, 40)
            XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
            XCTAssertEqual(spec.apiAvailability, .cliOnly)
            let patterns = PPLXEmbedV2Catalog.hubFallback(id).patterns
            XCTAssertTrue(patterns.contains("*.safetensors") || patterns.contains("*"),
                          "Contextual heads must be downloaded alongside indexed model shards.")
        }
    }

    func testPackedCatalogEntriesRetainDistinctPrecisionAndOutputContracts() throws {
        for (id, precision) in [(ModelResolver.ModelID.pplxEmbedV2LateLargeMixed4Bit, MereRunModelManifest.Precision.int4),
                                (.pplxEmbedV2Context8Bit, .int8)] {
            let manifest = MereRunModelManifest.template(for: id)
            XCTAssertEqual(manifest.precision, precision)
            XCTAssertEqual(manifest.engine, .pplxEmbedV2)
            let supports = try XCTUnwrap(manifest.supports)
            XCTAssertEqual(supports.contains(.multimodalEmbedding), !PPLXEmbedV2Catalog.isContextual(id.rawValue))
            XCTAssertTrue(PPLXEmbedV2Catalog.repository(id.rawValue).hasPrefix("Sawfwair/"))
            let descriptor = try XCTUnwrap(ManagedModelCapabilityCatalog.descriptor(for: id.rawValue))
            XCTAssertEqual(descriptor.minimumUnifiedMemoryGB, 24)
            XCTAssertEqual(descriptor.recommendedUnifiedMemoryGB, 36)
        }
    }

    func testReal9BTokenizerNFCOffsetsWhenCheckpointIsAvailable() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_PPLX_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_PPLX_TOKENIZER_ROOT to a downloaded 9B checkpoint.")
        }
        struct Query: Decodable { let text: String; let ids: [Int] }
        struct Cases: Decodable {
            let chunks: [String], documentIDs: [Int], spans: [[Int]], queries: [Query]
            enum CodingKeys: String, CodingKey { case chunks, documentIDs = "document_ids", spans, queries }
        }
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/PPLXEmbedV2")
        let root = URL(fileURLWithPath: path)
        let installedConfig = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        let name = installedConfig.isContextual ? "real-context-tokenizer-nfc-cases.json" : "real-tokenizer-nfc-cases.json"
        let cases = try JSONDecoder().decode(Cases.self, from: Data(contentsOf: fixtures.appending(path: name)))
        let config = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: fixtures.appending(path: "context-config.json")))
        let tokenizer = try PPLXEmbedV2Tokenizer.load(root: URL(fileURLWithPath: path), contextual: true)
        let actual = try tokenizer.contextual(cases.chunks, task: .document, config: config, limit: 1024)
        XCTAssertEqual(actual.ids, cases.documentIDs)
        XCTAssertEqual(actual.spans.map { [$0.lowerBound, $0.upperBound] }, cases.spans)
        for query in cases.queries {
            XCTAssertEqual(try tokenizer.contextual([query.text], task: .query, config: config, limit: 1024).ids, query.ids)
        }
    }

    func testMaxSimUsesSumOfPerQueryTokenMaximaIncludingNegativeScores() throws {
        XCTAssertEqual(try PPLXEmbedV2Result.maxSim(query: [[1, 0], [0, 1]], document: [[1, 0], [0, 1]]), 2)
        XCTAssertEqual(try PPLXEmbedV2Result.maxSim(query: [[1, 0], [0, 1]], document: [[-1, -1]]), -2)
        XCTAssertThrowsError(try PPLXEmbedV2Result.maxSim(query: [], document: [[1]]))
        XCTAssertThrowsError(try PPLXEmbedV2Result.maxSim(query: [[1]], document: [[1, 2]]))
        XCTAssertThrowsError(try PPLXEmbedV2Result.maxSim(query: [[.nan]], document: [[1]]))
    }

    func testChunkSpansExcludeEmptyChunksAndIncludeOverlappingBoundaryTokens() {
        let offsets = [0..<3, 3..<6, 6..<9, 9..<12]
        XCTAssertEqual(PPLXEmbedV2Tokenizer.overlappingTokenSpans(offsets: offsets, chunks: [2..<7, 7..<7, 10..<12]),
                       [0..<3, 0..<0, 3..<4])
        // UTF-8 bytes are counted per byte-level scalar, even if a character spans tokens.
        XCTAssertEqual(PPLXEmbedV2Tokenizer.byteOffsets(tokens: ["a", "Ã", "©"]), [0..<1, 1..<2, 2..<3])
    }

    func testRejectsCausalOrMalformedConfigurationBeforeLoading() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let data = try Data(contentsOf: root.appending(path: "Fixtures/PPLXEmbedV2/reference.json"))
        struct Fixture: Decodable { let config: PPLXEmbedV2Config }
        _ = try JSONDecoder().decode(Fixture.self, from: data)
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertThrowsError(try JSONDecoder().decode(Fixture.self, from: Data(text.replacingOccurrences(of: "\"is_causal\": false", with: "\"is_causal\": true").utf8)))
        XCTAssertThrowsError(try JSONDecoder().decode(Fixture.self, from: Data(text.replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": 0").utf8)))
    }
}

final class PPLXEmbedV2ImagePositionTests: MLXTestCase {
    func testImagePositionsIncludePrefixAndVisionMarkers() {
        let positions = PPLXEmbedV2Model.imagePositions(grid: (1, 4, 6), mergeSize: 2)
        MLX.eval(positions)
        XCTAssertEqual(positions.shape, [3, 1, 9])
        XCTAssertEqual(positions.asArray(Int32.self), [0, 1, 2, 2, 2, 2, 2, 2, 5,
                                                     0, 1, 2, 2, 2, 3, 3, 3, 5,
                                                     0, 1, 2, 3, 4, 2, 3, 4, 5])
    }
}
