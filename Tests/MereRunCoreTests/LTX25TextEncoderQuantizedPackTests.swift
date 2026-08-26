import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LTX25TextEncoderQuantizedPackTests: XCTestCase {
    func testRealOptimizedPackLoadsAndEncodes() async throws {
        guard let rootPath = ProcessInfo.processInfo.environment[
            "MERERUN_LTX25_TEXT_ENCODER_ROOT"
        ] else {
            throw XCTSkip(
                "Set MERERUN_LTX25_TEXT_ENCODER_ROOT for real Q4 text-encoder coverage."
            )
        }
        let resources = LTX25Resources(rootURL: URL(fileURLWithPath: rootPath))
        _ = try XCTUnwrap(
            LTX25TextEncoderQuantizedPack.optimizedIndexURLIfValid(resources: resources)
        )
        let encoder = LTXGemmaTextEncoder()
        try await encoder.load(modelRoot: resources.rootURL, dtype: .bfloat16)
        let encoding = try await encoder.encode(
            prompt: "A camera glides forward through a quiet forest.",
            maxLength: 32
        )
        if let audio = encoding.audioEmbeddings {
            MLX.eval(encoding.videoEmbeddings, audio)
            XCTAssertGreaterThan(audio.size, 0)
        } else {
            XCTFail("Expected LTX 2.5 audio embeddings.")
        }
        XCTAssertGreaterThan(encoding.videoEmbeddings.size, 0)
        await encoder.unload()
        Memory.clearCache()
    }

    func testOptimizerBuildsSourceBoundQ4LanguagePack() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let result = try LTX25TextEncoderQuantizedPack.optimize(
            resources: fixture.resources
        )
        XCTAssertEqual(result.sourceTensorCount, 3)
        XCTAssertEqual(result.quantizedTensorCount, 2)
        XCTAssertEqual(result.shardCount, 3)
        XCTAssertLessThan(result.packedBytes, result.sourceBytes)
        XCTAssertEqual(
            LTX25TextEncoderQuantizedPack.optimizedIndexURLIfValid(
                resources: fixture.resources
            ),
            result.indexURL
        )

        let arrays = try loadPackArrays(resources: fixture.resources)
        XCTAssertNotNil(arrays["embed_tokens.weight"])
        XCTAssertNotNil(arrays["embed_tokens.scales"])
        XCTAssertNotNil(arrays["embed_tokens.biases"])
        XCTAssertNotNil(arrays["layers.0.mlp.up_proj.weight"])
        XCTAssertNotNil(arrays["layers.0.mlp.up_proj.scales"])
        XCTAssertNotNil(arrays["norm.weight"])
        XCTAssertNil(arrays["text_embedding_projection.aggregate_embed.weight"])

        let restored = MLX.dequantized(
            try XCTUnwrap(arrays["layers.0.mlp.up_proj.weight"]),
            scales: try XCTUnwrap(arrays["layers.0.mlp.up_proj.scales"]),
            biases: arrays["layers.0.mlp.up_proj.biases"],
            groupSize: LTX25TextEncoderQuantizedPack.groupSize,
            bits: LTX25TextEncoderQuantizedPack.bits,
            mode: .affine,
            dtype: .float32
        )
        let error = MLX.mean(
            MLX.abs(restored - fixture.layerWeight.asType(.float32))
        ).item(Float.self)
        XCTAssertLessThan(error, 0.05)
    }

    func testOptimizerRequiresExplicitReplacement() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        _ = try LTX25TextEncoderQuantizedPack.optimize(resources: fixture.resources)
        XCTAssertThrowsError(
            try LTX25TextEncoderQuantizedPack.optimize(resources: fixture.resources)
        ) { error in
            guard case LTX25TextEncoderQuantizedPackError.outputExists = error else {
                return XCTFail("Expected outputExists, got \(error)")
            }
        }
        let replacement = try LTX25TextEncoderQuantizedPack.optimize(
            resources: fixture.resources,
            replacing: true
        )
        XCTAssertEqual(
            LTX25TextEncoderQuantizedPack.optimizedIndexURLIfValid(
                resources: fixture.resources
            ),
            replacement.indexURL
        )
    }

    private func makeFixture() throws -> (
        root: URL,
        resources: LTX25Resources,
        layerWeight: MLXArray
    ) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        let resources = LTX25Resources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.textEncoderURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let embeddingValues: [Float] = (0..<(32 * 64)).map { index in
            Float(index % 29) / 14.0 - 1.0
        }
        let embedding = MLXArray(embeddingValues)
            .reshaped(32, 64)
            .asType(.bfloat16)
        let layerValues: [Float] = (0..<(64 * 64)).map { index in
            Float(index % 37) / 18.0 - 1.0
        }
        let layerWeight = MLXArray(layerValues)
            .reshaped(64, 64)
            .asType(.bfloat16)
        try MLX.save(
            arrays: [
                "model.embed_tokens.weight": embedding,
                "model.layers.0.mlp.up_proj.weight": layerWeight,
                "model.norm.weight": MLX.ones([64], dtype: .bfloat16),
                "text_embedding_projection.aggregate_embed.weight": MLX.ones(
                    [64, 64],
                    dtype: .bfloat16
                ),
            ],
            url: resources.textEncoderURL
        )
        return (root, resources, layerWeight)
    }

    private func loadPackArrays(resources: LTX25Resources) throws -> [String: MLXArray] {
        let indexData = try Data(
            contentsOf: LTX25TextEncoderQuantizedPack.indexURL(resources: resources)
        )
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: indexData)
        let directory = LTX25TextEncoderQuantizedPack.outputDirectoryURL(
            resources: resources
        )
        var arrays: [String: MLXArray] = [:]
        for filename in index.shardFilenames {
            let shard = directory.appendingPathComponent(filename, isDirectory: false)
            arrays.merge(try MLX.loadArrays(url: shard)) { _, latest in latest }
        }
        return arrays
    }
}
