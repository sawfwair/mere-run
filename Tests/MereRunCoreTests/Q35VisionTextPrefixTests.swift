import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q35VisionTextPrefixTests: MereRunCoreTestCase {
    func testReuseEndsBeforeFirstImageAndRequiresOrdinaryPositionsOnEveryAxis() {
        let tokens = [1, 2, 63, 63, 3, 63]
        let ordinary = Array(0..<tokens.count).map(Int32.init)
        let positions = MLXArray(ordinary + ordinary + ordinary, [3, 1, tokens.count])
        XCTAssertEqual(Q35Generator.visionTextPrefixTokenCount(
            promptTokens: tokens, imageTokenId: 63, positionIds: positions
        ), 2)
        for axis in 0..<3 {
            var changed = ordinary + ordinary + ordinary
            changed[axis * tokens.count + 1] = 7
            XCTAssertEqual(Q35Generator.visionTextPrefixTokenCount(
                promptTokens: tokens, imageTokenId: 63,
                positionIds: MLXArray(changed, [3, 1, tokens.count])
            ), 0)
        }
        for prompt in [[63, 1], [1, 2], []] {
            XCTAssertEqual(Q35Generator.visionTextPrefixTokenCount(
                promptTokens: prompt, imageTokenId: 63, positionIds: positions
            ), 0)
        }
        XCTAssertEqual(Q35Generator.visionTextPrefixTokenCount(
            promptTokens: tokens, imageTokenId: 63, positionIds: nil
        ), 0)
    }

    func testImageSuffixParityAndSnapshotIsolation() async throws {
        try await Q35CompiledOperations.withNewDefaultStream(scoped: true) {
            try await Self.qualifySuffix(Q35Generator(prefixKVCacheEnabled: true))
        }
    }

    private static func qualifySuffix(_ generator: isolated Q35Generator) async throws {
        MLXRandom.seed(7429)
        let model = Q35Model(config: try configuration())
        let text = (0..<19).map { $0 % 30 + 1 }
        let tokens = text + [63, 63, 63, 63, 7, 8, 9]
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        var axes = Array(repeating: Array(0..<text.count), count: 3)
        axes[0] += [19, 19, 19, 19, 21, 22, 23]
        axes[1] += [19, 19, 20, 20, 21, 22, 23]
        axes[2] += [19, 20, 19, 20, 21, 22, 23]
        let positions = MLXArray(axes.flatMap { $0.map(Int32.init) }, [3, 1, tokens.count])
        _ = try await generator.chunkedPrefill(
            model: model, promptTokens: text, cache: caches(),
            modelPath: "fixture", progressHandler: nil
        )
        // A longer text-only cache with the same placeholder IDs must not cross
        // the image boundary even if all token IDs match.
        _ = try await generator.chunkedPrefill(
            model: model, promptTokens: tokens, cache: caches(),
            modelPath: "fixture", progressHandler: nil
        )
        var imageLogits: [MLXArray] = []
        let firstImage = MLXRandom.normal([1, 4, 32]) * 0.3
        let secondImage = MLXRandom.normal([1, 4, 32]) * -0.8
        for replacement in [firstImage, secondImage, firstImage] {
            let embedded = model.embeddings(for: input)
            let embeddings = MLX.concatenated([
                embedded[0..., 0..<text.count, 0...], replacement,
                embedded[0..., (text.count + 4)..., 0...],
            ], axis: 1)
            let expected = try await generator.chunkedPrefillEmbeddings(
                model: model, inputIds: input, inputEmbeddings: embeddings,
                cache: caches(), positionIds: positions, progressHandler: nil
            )
            let prefix = try await generator.prefillVisionTextPrefix(
                model: model, promptTokens: tokens, imageTokenId: 63,
                positionIds: positions, modelPath: "fixture", cacheMode: .default,
                cache: caches(), progressHandler: nil
            )
            XCTAssertEqual(prefix.tokenCount, text.count)
            let actual = try await generator.chunkedPrefillEmbeddings(
                model: model, inputIds: input, inputEmbeddings: embeddings,
                cache: prefix.caches, positionIds: positions,
                startIndex: prefix.tokenCount, progressHandler: nil
            )
            XCTAssertLessThan((actual.logits - expected.logits).abs().max().item(Float.self), 0.0001)
            XCTAssertEqual(argMax(actual.logits, axis: -1).asArray(Int32.self),
                           argMax(expected.logits, axis: -1).asArray(Int32.self))
            imageLogits.append(actual.logits)
            let snapshot = try XCTUnwrap(generator.prefixKVCacheSeed(
                modelPath: "fixture", promptTokens: text, cacheMode: .default
            ))
            guard case .full(let attention)? = snapshot.caches[1] else {
                return XCTFail("Missing attention cache")
            }
            XCTAssertEqual(attention.offset, text.count, "Image continuation must not mutate the text snapshot")
        }
        XCTAssertGreaterThan((imageLogits[0] - imageLogits[1]).abs().max().item(Float.self), 0.0001)
        XCTAssertEqual((imageLogits[0] - imageLogits[2]).abs().max().item(Float.self), 0)
        XCTAssertNil(generator.prefixKVCacheSeed(
            modelPath: "other-model", promptTokens: text, cacheMode: .default
        ))
        for mode in [RuntimeKVCacheMode.affine4, .affine8] {
            XCTAssertNil(generator.prefixKVCacheSeed(modelPath: "fixture", promptTokens: text, cacheMode: mode))
        }
        XCTAssertNil(generator.prefixKVCacheSeed(
            modelPath: "fixture", promptTokens: [42] + Array(text.dropFirst()), cacheMode: .default
        ))
    }

    func testFirstImageRequestCreatesOnlyTextSnapshotAndDisabledCacheSkipsReuse() async throws {
        for enabled in [true, false] {
            try await Q35CompiledOperations.withNewDefaultStream(scoped: true) {
                try await Self.qualifyColdPrefix(Q35Generator(prefixKVCacheEnabled: enabled), enabled: enabled)
            }
        }
    }

    private static func qualifyColdPrefix(_ generator: isolated Q35Generator, enabled: Bool) async throws {
        MLXRandom.seed(422)
        let model = Q35Model(config: try configuration())
        let tokens = [1, 2, 3, 63, 63, 4]
        let axis = Array(0..<tokens.count).map(Int32.init)
        let prefix = try await generator.prefillVisionTextPrefix(
            model: model, promptTokens: tokens, imageTokenId: 63,
            positionIds: MLXArray(axis + axis + axis, [3, 1, tokens.count]),
            modelPath: "fixture", cacheMode: .default, cache: caches(), progressHandler: nil
        )
        XCTAssertEqual(prefix.tokenCount, enabled ? 3 : 0)
        let snapshot = generator.prefixKVCacheSeed(modelPath: "fixture", promptTokens: tokens, cacheMode: .default)
        XCTAssertEqual(snapshot?.tokenCount, enabled ? 3 : nil)
        XCTAssertEqual(generator.prefixKVCacheStats().storedTokens, enabled ? 3 : 0)
    }

    private static func caches() -> [Q35LayerCache?] {
        [.linear(Q35LinearCache()), .full(KVCacheSimple())]
    }

    private static func configuration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen3_5","text_config":{
          "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
          "layer_types":["linear_attention","full_attention"],
          "linear_num_key_heads":1,"linear_num_value_heads":2,
          "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
          "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,
          "eos_token_id":62,"vocab_size":64,"max_position_embeddings":4096,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1,
            "mrope_interleaved":true,"mrope_section":[2,1,1]}}}
        """#.utf8))
    }
}
