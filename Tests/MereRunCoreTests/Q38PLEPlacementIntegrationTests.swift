import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Q38PLEPlacementIntegrationTests: MereRunCoreTestCase {
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
        let base = "language_model.model.layers.\(text.pleLayerIds[0] - 1).ple.ple_embedding.ngram_embedding"
        let table: Q38DiskNGramTable
        if let placement = try Q38PLEPlacement.resolve(rootURL: root) {
            table = try Q38DiskNGramTable(
                indexURL: placement.indexURL,
                base: base,
                shardCount: text.splitNgramParts,
                dimensions: text.pleEmbeddingDimensions / heads,
                minimumRowCount: embedding.minimumRowCount
            )
            print("[q38-disk-table] packaged_ple_internal_cache=\(placement.usedInternalCache)")
        } else {
            table = try Q38DiskNGramTable(
                indexURL: root.appendingPathComponent("model.safetensors.index.json"),
                base: base, shardCount: text.splitNgramParts,
                dimensions: text.pleEmbeddingDimensions / heads,
                minimumRowCount: embedding.minimumRowCount
            )
        }
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

    func testPlacementManifestUsesTheOriginalTableOnTheInternalVolume() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let manifestURL = try fixture.writePlacementManifest()
        let resolution = try XCTUnwrap(Q38PLEPlacement.resolve(
            rootURL: fixture.root,
            cacheBase: fixture.root.appendingPathComponent("cache")
        ))
        XCTAssertFalse(resolution.usedInternalCache)
        let placed = try Q38DiskNGramTable(
            indexURL: resolution.indexURL,
            base: Fixture.base,
            shardCount: 2,
            dimensions: 160,
            minimumRowCount: 8
        )
        let sharded = try fixture.load()
        XCTAssertEqual(placed.tableByteCount, sharded.tableByteCount)
        for count in [1, 16, 64, 4_096] {
            let ids = (0..<count).map { Int32(($0 * 7 + 3) % 8) }
            XCTAssertEqual(
                placed.lookup(ids).asArray(Float.self),
                sharded.lookup(ids).asArray(Float.self)
            )
        }
        XCTAssertEqual(manifestURL.lastPathComponent, Q38PLEPlacement.manifestFilename)
    }

    private struct Fixture {
        let root: URL
        let embeddings: [PreQuantizedEmbedding]
        static let base = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding"

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

        func writePlacementManifest() throws -> URL {
            let weights = root.appendingPathComponent("weights.safetensors")
            let parameters = root.appendingPathComponent("parameters.safetensors")
            let manifest = #"""
            {
              "version": 1,
              "format": "mere-run-q38-ple-safetensors-placement-v1",
              "artifact_id": "q38-ple-placement-test",
              "index": "model.safetensors.index.json",
              "preferred_placement": "internal_cache",
              "files": [
                {"path": "weights.safetensors", "byte_count": \#(try fileSize(weights)),
                 "sha256": "\#(String(repeating: "0", count: 64))"},
                {"path": "parameters.safetensors", "byte_count": \#(try fileSize(parameters)),
                 "sha256": "\#(String(repeating: "0", count: 64))"}
              ]
            }
            """#
            let manifestURL = root.appendingPathComponent(Q38PLEPlacement.manifestFilename)
            try Data(manifest.utf8).write(to: manifestURL)
            return manifestURL
        }

        private func fileSize(_ url: URL) throws -> Int {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            return try XCTUnwrap((attributes[.size] as? NSNumber)?.intValue)
        }

        func remove() { try? FileManager.default.removeItem(at: root) }

        private struct Index: Encodable {
            let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
        }
    }
}
