import Foundation
import MLX
import MLXNN
import MediaIO
import MereRunQwenModel
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class PPLXEmbedV2LoadingTests: MLXTestCase {
    struct TokenCase: Decodable { let text: String, task: String; let ids: [Int] }
    struct Cases: Decodable {
        let late: [TokenCase], chunks: [String], documentIDs: [Int], spans: [[Int]], queryText: String, queryIDs: [Int]
        enum CodingKeys: String, CodingKey { case late, chunks, spans, documentIDs = "document_ids", queryText = "query_text", queryIDs = "query_ids" }
    }
    struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
    struct Fixture: Decodable { let weights: [String: Tensor] }
    let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/PPLXEmbedV2")

    func withCheckpoint(contextual: Bool = false, sharded: Bool = false, missing: String? = nil,
                        operation: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appending(path: "1_Dense"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appending(path: "2_MultiVectorMask"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try FileManager.default.copyItem(at: fixtures.appending(path: name), to: root.appending(path: name))
        }
        try FileManager.default.copyItem(at: fixtures.appending(path: contextual ? "context-config.json" : "config.json"), to: root.appending(path: "config.json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtures.appending(path: "reference.json")))
        var arrays = fixture.weights.mapValues { MLXArray($0.values, $0.shape) }
        if let missing { arrays.removeValue(forKey: missing) }
        if contextual { arrays["contextual_projection.weight"] = MLXArray.ones([2048, 8]) * MLXArray(Float(0.003)) }
        if sharded {
            let sorted = arrays.sorted { $0.key < $1.key }
            var weightMap: [String: String] = [:]
            for (index, shard) in [Array(sorted.prefix(sorted.count / 2)), Array(sorted.suffix(sorted.count - sorted.count / 2))].enumerated() {
                let name = "model-\(index).safetensors"
                try MLX.save(arrays: Dictionary(uniqueKeysWithValues: shard.map { ($0.key, $0.value) }), url: root.appending(path: name))
                for key in shard.map(\.key) { weightMap[key] = name }
            }
            try JSONEncoder().encode(HFSafetensorsIndex(metadata: nil, weightMap: weightMap)).write(to: root.appending(path: "model.safetensors.index.json"))
        } else { try MLX.save(arrays: arrays, url: root.appending(path: "model.safetensors")) }
        try MLX.save(arrays: ["linear.weight": MLXArray.ones([128, 8])], url: root.appending(path: "1_Dense/model.safetensors"))
        try Data(#"{"in_features":8,"out_features":128,"bias":false}"#.utf8).write(to: root.appending(path: "1_Dense/config.json"))
        try Data(#"{"skiplist_words":["!",",","."],"skiplist_tasks":["document"],"keep_only_token_ids":null}"#.utf8)
            .write(to: root.appending(path: "2_MultiVectorMask/config.json"))
        try operation(root)
    }

    func testTokenizerMatchesIndependentUnicodeSpecialTokenAndChunkFixtures() throws {
        let cases = try JSONDecoder().decode(Cases.self, from: Data(contentsOf: fixtures.appending(path: "tokenizer-cases.json")))
        try withCheckpoint { root in
            let tokenizer = try PPLXEmbedV2Tokenizer.load(root: root, contextual: false)
            for row in cases.late {
                XCTAssertEqual(tokenizer.late(row.text, task: row.task == "query" ? .query : .document, limit: 1024).ids, row.ids)
            }
            let config = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: fixtures.appending(path: "context-config.json")))
            let sequence = try tokenizer.contextual(cases.chunks, task: .document, config: config, limit: 1024)
            XCTAssertEqual(sequence.ids, cases.documentIDs)
            XCTAssertEqual(sequence.spans.map { [$0.lowerBound, $0.upperBound] }, cases.spans)
            XCTAssertEqual(try tokenizer.contextual([cases.queryText], task: .query, config: config, limit: 1024).ids, cases.queryIDs)
            XCTAssertEqual(tokenizer.late("many words", task: .query, limit: 1).ids, [tokenizer.queryID])
        }
    }

    func testContextKeepsNonSpecialAddedSeparatorsWhileSplittingSpecialMarkers() throws {
        try withCheckpoint { root in
            let path = root.appending(path: "tokenizer.json")
            var json = try String(contentsOf: path, encoding: .utf8)
            let separator = try XCTUnwrap(json.range(of: "\"content\": \"<|chunk_sep|>\""))
            let special = try XCTUnwrap(json.range(of: "\"special\": true", range: separator.upperBound..<json.endIndex))
            json.replaceSubrange(special, with: "\"special\": false")
            try json.write(to: path, atomically: true, encoding: .utf8)
            let tokenizer = try PPLXEmbedV2Tokenizer.load(root: root, contextual: true)
            let config = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: fixtures.appending(path: "context-config.json")))
            let document = try tokenizer.contextual(["first", "second"], task: .document, config: config, limit: 1024)
            XCTAssertEqual(document.ids.filter { $0 == 3 }.count, 1)
            for span in document.spans { XCTAssertFalse(document.ids[span].contains(3)) }
            let query = try tokenizer.contextual(["<|chunk_sep|> [Q] "], task: .query, config: config, limit: 1024)
            XCTAssertTrue(query.ids.contains(3))
            XCTAssertEqual(query.ids.filter { $0 == tokenizer.queryID }.count, 1)
        }
    }

    func testNFCNormalizationPreservesUnicodeChunkBoundariesAndEmptyChunks() throws {
        try withCheckpoint { root in
            let path = root.appending(path: "tokenizer.json")
            let original = try String(contentsOf: path, encoding: .utf8)
            try original.replacingOccurrences(of: "\"normalizer\": null", with: "\"normalizer\": {\"type\": \"NFC\"}").write(to: path, atomically: true, encoding: .utf8)
            let tokenizer = try PPLXEmbedV2Tokenizer.load(root: root, contextual: true)
            XCTAssertTrue(tokenizer.normalizesNFC)
            let config = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: Data(contentsOf: fixtures.appending(path: "context-config.json")))
            let decomposed = ["cafe\u{0301}", "", "\u{1100}\u{1161}", "na\u{0308}ive 👩🏽‍💻"]
            let composed = decomposed.map(\.precomposedStringWithCanonicalMapping)
            let actual = try tokenizer.contextual(decomposed, task: .document, config: config, limit: 1024)
            let expected = try tokenizer.contextual(composed, task: .document, config: config, limit: 1024)
            XCTAssertEqual(actual.ids, expected.ids)
            XCTAssertEqual(actual.spans, expected.spans)
            XCTAssertEqual(actual.spans[1], 0..<0)
            XCTAssertEqual(tokenizer.late(decomposed[0], task: .query, limit: 1024).ids,
                           tokenizer.late(composed[0], task: .query, limit: 1024).ids)
            try original.replacingOccurrences(of: "\"normalizer\": null", with: "\"normalizer\": {\"type\": \"NFKC\"}").write(to: path, atomically: true, encoding: .utf8)
            XCTAssertThrowsError(try PPLXEmbedV2Tokenizer.load(root: root, contextual: true))
        }
    }

    func testLateLoadsFP32AndMasksDocumentPunctuationButRetainsQueryPunctuation() throws {
        try withCheckpoint { root in
            let model = try PPLXEmbedV2Model(resources: .init(rootURL: root))
            let document = try model.embed(texts: ["Hello!", ""], task: .document)
            let query = try model.embed(texts: ["Hello!"], task: .query)
            XCTAssertEqual(query.data[0].embeddings.count, document.data[0].embeddings.count + 1)
            XCTAssertEqual(document.data[1].embeddings.count, 1) // Document marker survives.
            XCTAssertTrue(document.normalized)
            XCTAssertEqual(document.dimensions, 128)
            XCTAssertTrue(model.encoder.parameters().flattened().allSatisfy { $0.0.hasSuffix(".qkNormWeightBF16") || $0.1.dtype == .float32 })
            let alone = try model.embed(texts: ["Hello!"], task: .document)
            XCTAssertEqual(alone.data[0].embeddings, document.data[0].embeddings)
            XCTAssertThrowsError(try model.embed(documents: [["a", "b"]]))
            XCTAssertThrowsError(try model.embed(texts: ["a"], dimensions: 256))
        }
    }

    func testContextualShardsLoadAndReturnOneInt8VectorPerChunkWithoutTruncation() throws {
        try withCheckpoint(contextual: true, sharded: true) { root in
            let model = try PPLXEmbedV2Model(resources: .init(rootURL: root))
            let output = try model.embed(documents: [["Hello", "", "world"]])
            XCTAssertEqual(output.data[0].embeddings.count, 3)
            XCTAssertEqual(output.data[0].embeddings[0].count, 2048)
            XCTAssertEqual(output.data[0].embeddings[1], Array(repeating: 0, count: 2048))
            XCTAssertTrue(output.data[0].embeddings.flatMap { $0 }.allSatisfy { $0 == $0.rounded() && abs($0) <= 127 })
            XCTAssertFalse(output.normalized)
            let truncated = try model.embed(texts: ["Hello"], dimensions: 1024, normalize: true)
            XCTAssertTrue(truncated.normalized)
            XCTAssertEqual(truncated.data[0].embeddings[0].count, 1024)
            XCTAssertThrowsError(try model.embed(documents: [["a", "b"]], task: .query))
            XCTAssertThrowsError(try model.embed(texts: ["too long"], maxTokens: 1))
            XCTAssertThrowsError(try model.embed(documents: [[]]))
        }
    }

    func testImageDocumentsUseNativeTowerAndRejectPartialTruncationOrMissingWeights() throws {
        try withCheckpoint { root in
            let processor = #"{"image_processor":{"patch_size":2,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,"resample":3,"size":{"shortest_edge":64,"longest_edge":64}}}"#
            try Data(processor.utf8).write(to: root.appending(path: "processor_config.json"))
            let imageURL = root.appending(path: "image.png")
            let image = try MediaImage(width: 8, height: 8, rgba8: Array(repeating: [UInt8(80), 120, 160, 255], count: 64).flatMap { $0 })
            try MediaImageIO.writePNG(image, to: imageURL)
            let model = try PPLXEmbedV2Model(resources: .init(rootURL: root))
            let output = try model.embed(images: [imageURL])
            XCTAssertEqual(output.data[0].tokenCount, 7)
            XCTAssertEqual(output.dimensions, 128)
            XCTAssertTrue(output.data[0].embeddings.flatMap { $0 }.allSatisfy(\.isFinite))
            XCTAssertThrowsError(try model.embed(images: [imageURL], maxTokens: 6))
        }
        try withCheckpoint(missing: "visual.pos_embed.weight") { root in
            let processor = #"{"image_processor":{"patch_size":2,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,"resample":3,"size":{"shortest_edge":64,"longest_edge":64}}}"#
            try Data(processor.utf8).write(to: root.appending(path: "processor_config.json"))
            let imageURL = root.appending(path: "image.png")
            try MediaImageIO.writePNG(MediaImage(width: 8, height: 8, rgba8: Array(repeating: UInt8(80), count: 256)), to: imageURL)
            let model = try PPLXEmbedV2Model(resources: .init(rootURL: root))
            XCTAssertThrowsError(try model.embed(images: [imageURL]))
        }
    }

    func testMissingTextWeightsAreRejectedInSingleAndShardedCheckpoints() throws {
        for sharded in [false, true] {
            try withCheckpoint(sharded: sharded, missing: "language_model.norm.weight") { root in
                XCTAssertThrowsError(try PPLXEmbedV2Model(resources: .init(rootURL: root)))
            }
        }
    }

    func testMixedWeightsLoadThroughSingleAndShardedPublicResources() throws {
        for contextual in [false, true] {
            try withCheckpoint(contextual: contextual, sharded: contextual) { root in
                var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: root.appending(path: "config.json"))) as? [String: Any])
                var text = try XCTUnwrap(json["text_config"] as? [String: Any])
                text["hidden_size"] = 64; text["intermediate_size"] = 128
                json["text_config"] = text
                let module = "embed_tokens"
                json["quantization"] = ["bits": 4, "group_size": 32, "mode": "affine",
                    "modules": [module: ["bits": 8, "group_size": 32, "mode": "affine"]]]
                let data = try JSONSerialization.data(withJSONObject: json)
                try data.write(to: root.appending(path: "config.json"))
                let config = try JSONDecoder().decode(PPLXEmbedV2Config.self, from: data)
                let encoder = PPLXEmbedV2Encoder(config: config.backbone)
                var arrays = Dictionary(uniqueKeysWithValues: encoder.parameters().flattened().compactMap { key, value -> (String, MLXArray)? in
                    guard encoder.checkpointParameterNames.contains(key) else { return nil }
                    return ("language_model." + key, key.hasSuffix(".conv1d.weight") ? value.transposed(0, 2, 1) : value)
                })
                let key = "language_model.embed_tokens"
                let (weight, scales, biases) = MLX.quantized(arrays[key + ".weight"]!, groupSize: 32, bits: 8)
                arrays[key + ".weight"] = weight; arrays[key + ".scales"] = scales; arrays[key + ".biases"] = biases
                if contextual { arrays["contextual_projection.weight"] = MLXArray.ones([2048, 64]) * MLXArray(Float(0.003)) }
                let name = contextual ? "model-quantized.safetensors" : "model.safetensors"
                try MLX.save(arrays: arrays, url: root.appending(path: name))
                if contextual {
                    try JSONEncoder().encode(HFSafetensorsIndex(metadata: nil, weightMap: arrays.mapValues { _ in name }))
                        .write(to: root.appending(path: "model.safetensors.index.json"))
                }
                try MLX.save(arrays: ["linear.weight": MLXArray.ones([128, 64])], url: root.appending(path: "1_Dense/model.safetensors"))
                try Data(#"{"in_features":64,"out_features":128,"bias":false}"#.utf8).write(to: root.appending(path: "1_Dense/config.json"))
                let model = try PPLXEmbedV2Model(resources: .init(rootURL: root))
                let result = try model.embed(texts: ["Hello!"])
                XCTAssertEqual(result.dimensions, contextual ? 2048 : 128)
                XCTAssertTrue(result.data[0].embeddings.flatMap { $0 }.allSatisfy(\.isFinite))
            }
        }
    }
}
