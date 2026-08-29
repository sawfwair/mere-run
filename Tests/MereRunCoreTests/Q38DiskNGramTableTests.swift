import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38DiskNGramTableTests: MereRunCoreTestCase {
    func testInstalledTableUsesNoPersistentMLXAllocation() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS"] == "1",
                          "Installed checkpoint test is opt-in.")
        let rootPath = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        let root = URL(fileURLWithPath: rootPath)
        let configuration = try JSONDecoder().decode(Q35Config.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let embedding = Q38NGramEmbedding(config: configuration, pleLayerIndex: 0)
        let text = configuration.textConfig
        let heads = (text.ngramSize - 1) * text.headsPerNgram
        let before = Memory.activeMemory
        let table = try Q38DiskNGramTable(
            indexURL: root.appendingPathComponent("model.safetensors.index.json"),
            base: "language_model.model.layers.\(text.pleLayerIds[0] - 1).ple.ple_embedding.ngram_embedding",
            shardCount: text.splitNgramParts, dimensions: text.pleEmbeddingDimensions / heads,
            minimumRowCount: embedding.minimumRowCount
        )
        XCTAssertEqual(Memory.activeMemory, before, "Mapping a table must not upload its tensors")
        embedding.installDiskTable(table)
        let start = Date()
        let output = embedding(MLXArray([Int32(14), 72, 109]).reshaped(1, 3), cache: nil)
        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 3, text.pleEmbeddingDimensions])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertLessThan(Memory.activeMemory - before, 1_048_576)
        print("[q38-disk-table] bytes=\(table.tableByteCount) mlx_delta=\(Memory.activeMemory - before) lookup_seconds=\(Date().timeIntervalSince(start))")
    }

    func testDiskEmbeddingPreservesBatchedHashAndIncrementalHistory() throws {
        let fixture = try Fixture(rows: [17, 19])
        defer { fixture.remove() }
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","text_config":{
          "model_type":"qwen4_exp_text","hidden_size":8,"num_hidden_layers":1,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
          "layer_types":["linear_attention"],
          "linear_num_value_heads":1,"linear_num_key_heads":1,
          "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":2,
          "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,
          "intermediate_size":8,"max_position_embeddings":4096,
          "rms_norm_eps":0.000001,"vocab_size":32,"eos_token_id":31,
          "ple_embed_dim":640,"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":5,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5}
        }}
        """#.utf8))
        let resident = Q38NGramEmbedding(config: config, pleLayerIndex: 0)
        let disk = Q38NGramEmbedding(config: config, pleLayerIndex: 0)
        resident.installShards(fixture.embeddings)
        disk.installDiskTable(try fixture.load(minimumRows: disk.minimumRowCount))
        let ids = MLXArray([Int32(1), 2, 31, 3, 4, 5, 6, 7, 8, 31]).reshaped(2, 5)
        let residentCache = Q35LinearCache()
        let diskCache = Q35LinearCache()
        for bounds in [0..<3, 3..<5] {
            let input = ids[0..., bounds]
            XCTAssertEqual(disk(input, cache: diskCache).asArray(Float.self),
                           resident(input, cache: residentCache).asArray(Float.self))
            XCTAssertEqual(diskCache.pleTokenContext?.asArray(Int32.self), residentCache.pleTokenContext?.asArray(Int32.self))
        }
    }

    func testShardedGatherIsBitExactWithResidentEmbedding() throws {
        for dtype in [DType.bfloat16, .float16, .float32] {
            for bits in [2, 4, 8] {
                let fixture = try Fixture(dtype: dtype, bits: bits)
                defer { fixture.remove() }
                let table = try fixture.load()
                XCTAssertEqual(table.rowCount, 8)
                XCTAssertEqual(table.bits, bits)
                XCTAssertEqual(table.groupSize, 32)
                XCTAssertEqual(table.dtype, dtype)
                XCTAssertEqual(table.tableByteCount, 8 * (160 * bits / 8 + 10 * dtype.size))
                let resident = MLX.concatenated([
                    fixture.embeddings[0](MLXArray(0..<Int32(3))),
                    fixture.embeddings[1](MLXArray(0..<Int32(5))),
                ], axis: 0)
                MLX.eval(resident)
                // Exercise shard boundaries, original row ordering, repeats,
                // and decode, verification, and prefill-shaped gathers.
                for count in [1, 16, 64, 4_096] {
                    let ids = (0..<count).map { Int32(($0 * 7 + 3) % 8) }
                    let actual = table.lookup(ids)
                    let expected = resident.take(MLXArray(ids), axis: 0)
                    XCTAssertEqual(actual.dtype, expected.dtype)
                    XCTAssertTrue(actual.asArray(Float.self) == expected.asArray(Float.self),
                                  "Disk lookup changed \(dtype) Q\(bits), \(count) rows")
                }
                XCTAssertEqual(table.lookup([]).shape, [0, 160])
            }
        }
    }

    func testMappingRemainsValidAfterCheckpointFileUnlink() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let table = try fixture.load()
        let expected = table.lookup([0, 2, 3, 7]).asArray(Float.self)
        // An open mapping owns the inode; removing the model's path must not
        // leave a dangling file descriptor or silently return different rows.
        try FileManager.default.removeItem(at: fixture.root.appendingPathComponent("weights.safetensors"))
        XCTAssertEqual(table.lookup([0, 2, 3, 7]).asArray(Float.self), expected)
    }

    func testMissingBiasIsRejectedAtLoad() throws {
        let fixture = try Fixture(omitBias: true)
        defer { fixture.remove() }
        XCTAssertThrowsError(try fixture.load())
    }

    func testMismatchedRowsAndTooSmallVocabularyAreRejectedAtLoad() throws {
        let fixture = try Fixture(mismatchedRows: true)
        defer { fixture.remove() }
        XCTAssertThrowsError(try fixture.load())
        let valid = try Fixture()
        defer { valid.remove() }
        XCTAssertThrowsError(try valid.load(minimumRows: 9))
        XCTAssertThrowsError(try valid.load(dimensions: 159))
    }

    private struct Fixture {
        let root: URL
        let embeddings: [PreQuantizedEmbedding]
        private static let base = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding"

        init(dtype: DType = .bfloat16, bits: Int = 4, omitBias: Bool = false, mismatchedRows: Bool = false,
             rows: [Int] = [3, 5]) throws {
            MLXRandom.seed(104)
            root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            var weights: [String: MLXArray] = [:]
            var parameters: [String: MLXArray] = [:]
            var weightMap: [String: String] = [:]
            var embeddings: [PreQuantizedEmbedding] = []
            for (part, rows) in rows.enumerated() {
                let prefix = "\(Self.base).shard_\(part)"
                let (weight, scales, biases) = MLX.quantized(
                    MLXRandom.normal([rows, 160]).asType(dtype), groupSize: 32, bits: bits
                )
                embeddings.append(PreQuantizedEmbedding(
                    weight: weight, scales: scales, biases: biases, groupSize: 32, bits: bits
                ))
                weights["\(prefix).weight"] = weight
                parameters["\(prefix).scales"] = mismatchedRows ? scales[0..<1] : scales
                weightMap["\(prefix).weight"] = "weights.safetensors"
                weightMap["\(prefix).scales"] = "parameters.safetensors"
                if !omitBias {
                    parameters["\(prefix).biases"] = biases
                    weightMap["\(prefix).biases"] = "parameters.safetensors"
                }
            }
            self.embeddings = embeddings
            try MLX.save(arrays: weights, url: root.appendingPathComponent("weights.safetensors"))
            try MLX.save(arrays: parameters, url: root.appendingPathComponent("parameters.safetensors"))
            try JSONEncoder().encode(Index(weightMap: weightMap))
                .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        }

        func load(minimumRows: Int = 8, dimensions: Int = 160) throws -> Q38DiskNGramTable {
            try Q38DiskNGramTable(
                indexURL: root.appendingPathComponent("model.safetensors.index.json"),
                base: Self.base, shardCount: 2, dimensions: dimensions, minimumRowCount: minimumRows
            )
        }

        func remove() { try? FileManager.default.removeItem(at: root) }

        private struct Index: Encodable {
            let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
        }
    }
}
