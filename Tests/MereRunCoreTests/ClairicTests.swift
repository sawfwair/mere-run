import Foundation
import MediaIO
import MereRunQwenModel
import XCTest
@testable import MereRunCore

final class ClairicTests: XCTestCase {
    func testPublishedShardNamesMatchCatalogDownloadsAndCheckpointValidation() throws {
        let index = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: index) }
        let releases = [
            (ClairicCatalog.flashModelID, "model.safetensors-*.safetensors", "model.safetensors-00004-of-00004.safetensors"),
            (ClairicCatalog.modelID, "model-*.safetensors", "model-00018-of-00018.safetensors"),
        ]
        for (model, pattern, shard) in releases {
            XCTAssertTrue(ClairicCatalog.fallback(model).patterns.contains(pattern))
            let indexData = try JSONEncoder().encode(["weight_map": ["lm_head.weight": shard]])
            try indexData.write(to: index)
            XCTAssertEqual(try ClairicCatalog.shardFilenames(at: index), [shard])
        }
    }

    func testJSONGrammarTracksSlotTerminatorsAndNumbers() throws {
        let grammar = ClairicGrammar(.nullable(.number))
        for valid in ["null", "-2.31e+39", "0", " 2.0\t"] {
            XCTAssertTrue(try grammar.accepts(XCTUnwrap(grammar.advance(grammar.start, bytes: Array(valid.utf8)))))
        }
        for invalid in ["00", "nullify", "2. 1", "01", "--1"] {
            XCTAssertNil(grammar.advance(grammar.start, bytes: Array(invalid.utf8)), invalid)
        }
        for partial in ["nu", "1.", "-", "3e-"] {
            XCTAssertFalse(try grammar.accepts(XCTUnwrap(grammar.advance(grammar.start, bytes: Array(partial.utf8)))))
        }
    }

    func testGrammarAllowsTokensCrossingObjectAndUnicodeBoundaries() throws {
        let grammar = ClairicGrammar(.object([("\"label\"", .string), ("\"n\"", .nullable(.integer))]))
        let prefix = Array(#"{"label":"café "#.utf8)
        let pieces: [[UInt8]] = [prefix + [0xF0, 0x9F], [0xA7, 0xBE] + Array(#"","n":null}"#.utf8)]
        let first = try XCTUnwrap(grammar.advance(grammar.start, bytes: pieces[0]))
        XCTAssertFalse(grammar.accepts(first))
        XCTAssertNil(grammar.advance(first, bytes: [0x20]))
        let last = try XCTUnwrap(grammar.advance(first, bytes: pieces[1]))
        XCTAssertTrue(grammar.accepts(last))
        XCTAssertNil(grammar.advance(last, bytes: [0x22]))
    }

    func testTypedOptionsValidateAndRejectUnsupportedReadouts() throws {
        let request = try decode(#"{"state":{"code":"001"},"fields":{"ok":{"type":"boolean","nullable":false},"n":{"type":"number","decimals":2,"minimum":0}}}"#)
        try request.validate()
        XCTAssertEqual(request.fields["n"]?.normalized(.number(10.126)), .number(10.13))
        XCTAssertEqual(request.fields["n"]?.normalized(.number(1.005)), .number(1.01))
        let signed = ClairicField(type: .number, decimals: 2)
        XCTAssertEqual(signed.normalized(.number(-1.005)), .number(-1.01))
        XCTAssertNil(request.fields["n"]?.normalized(.number(-1)))
        XCTAssertNil(request.fields["ok"]?.normalized(.null))
        let unsupported = try decode(#"{"state":"123","fields":{"date":{"type":"date"}}}"#)
        XCTAssertThrowsError(try unsupported.validate())
    }

#if !os(iOS)
    func testTransparentImageIsCompositedOnWhiteBeforeNormalization() throws {
        var rgba = [UInt8](repeating: 0, count: 32 * 32 * 4)
        rgba[7] = 128
        rgba[11] = 255
        let image = try MediaImage(width: 32, height: 32, rgba8: rgba)
        let png = try MediaImageIO.pngData(from: image).base64EncodedString()
        let request = try decode("""
            {"images":["data:image/png;base64,\(png)"],"fields":{"vendor":{"type":"text"}}}
            """)
        let processor = ClairicConfiguration.Processor(
            size: .init(longest_edge: 1_024, shortest_edge: 1_024),
            patch_size: 16, temporal_patch_size: 2, merge_size: 2,
            image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5])
        let item = try XCTUnwrap(ClairicMedia.prepare(request, processor: processor).first)

        for channel in 0..<3 {
            let start = channel * 32 * 32
            XCTAssertEqual(item.pixels[start], 1, accuracy: 0.000001)
            XCTAssertEqual(item.pixels[start + 1], -1.0 / 255, accuracy: 0.000001)
            XCTAssertEqual(item.pixels[start + 2], -1, accuracy: 0.000001)
        }
    }

    func testPromptMatchesPublishedOrdinarySlotFormat() throws {
        let request = try decode(#"{"state":"Northwind Supply","fields":{"vendor":{"type":"text","instructions":"The invoice vendor."}}}"#)
        let slot = try XCTUnwrap(ClairicPrompt.slots(for: request, compactLists: true).first)
        XCTAssertEqual(slot.instruction, """
            Extract the requested value from the supplied input, following the field instructions and scope. Return exactly one JSON scalar matching the field schema: quote and escape strings, write numbers without digit-group separators, and use null only when the value is unavailable and null is allowed. Zero is a present number. Preserve repeated source words and list entries. End immediately after the value.
            Field: vendor
            Schema: {"type":["string","null"]}
            Instructions: The invoice vendor.
            """)
    }

    func testListsPreserveSlotOrderingAndObjectSchema() throws {
        let request = try decode(#"{"fields":{"lines":{"type":"list","max_items":2,"items":{"type":"object","properties":{"sku":{"type":"text"},"price":{"type":"number"}}}}}}"#)
        try request.validate()
        let slots = try ClairicPrompt.slots(for: request, compactLists: true)
        XCTAssertEqual(slots.map(\.index), [1, 2])
        XCTAssertTrue(slots[1].instruction.contains("Field: lines.item2"))
        XCTAssertTrue(slots[1].instruction.contains(#""required":["price","sku"]"#))
    }

    func testPrintedTemporalFieldsUseTheOrdinarySlotAndNormalizeWithoutInferredValues() throws {
        let request = try decode(#"{"fields":{"issued":{"type":"date","source_only":true}}}"#)
        try request.validate()
        let slot = try XCTUnwrap(ClairicPrompt.slots(for: request, compactLists: true).first)
        XCTAssertEqual(slot.spec.type, .text)
        XCTAssertEqual(slot.temporal, .date)
        XCTAssertEqual(ClairicPrintedTemporal.normalize(.string("2026-11-18"), as: .date), .string("2026-11-18"))
        XCTAssertEqual(ClairicPrintedTemporal.normalize(.string("Nov 18, 2026 at 8:15 PM"), as: .datetime),
                       .string("2026-11-18T20:15:00"))
        XCTAssertNil(ClairicPrintedTemporal.normalize(.string("11/12/2026"), as: .date))
        XCTAssertNil(ClairicPrintedTemporal.normalize(.string("2026-02-29"), as: .date))
    }

    func testGroundedSpansUseScalarOffsetsAndRequireUniqueMatches() {
        XCTAssertEqual(ClairicDecodeRow.uniqueSpan(of: .string("Café"), in: "🧾 Café"),
                       .array([.number(2), .number(6)]))
        XCTAssertNil(ClairicDecodeRow.uniqueSpan(of: .number(42.5), in: "$42.50 and $42.5"))
        XCTAssertEqual(ClairicDecodeRow.uniqueSpan(of: .number(1_234.5), in: "€1,234.50"),
                       .array([.number(1), .number(9)]))
    }
#endif

    private func decode(_ source: String) throws -> ClairicRequest {
        try JSONDecoder().decode(ClairicRequest.self, from: Data(source.utf8))
    }
}
