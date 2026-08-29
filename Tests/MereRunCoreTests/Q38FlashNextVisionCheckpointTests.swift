#if os(macOS)
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import MLX
import MLXNN
import UniformTypeIdentifiers
import XCTest
@testable import MereRunCore

final class Q38FlashNextVisionCheckpointTests: MereRunCoreTestCase {
    func testVisionRotaryCoefficientsRetainFloat32Precision() {
        let angles: [Float] = [0.123456, 1.23456, 2.34567, 3.45678]
        let rotary = QwenVisionTower.prepareQwen3Rotary(rotaryPosEmb: MLXArray(angles, [1, 4]))
        XCTAssertEqual(rotary.cos.dtype, .float32)
        XCTAssertEqual(rotary.sin.dtype, .float32)
        for (index, angle) in (angles + angles).enumerated() {
            XCTAssertEqual(rotary.cos[0, index].item(Float.self), cos(angle), accuracy: 0.000001)
            XCTAssertEqual(rotary.sin[0, index].item(Float.self), sin(angle), accuracy: 0.000001)
        }
    }

    func testVisionPatchBiasLoadsFromFlashNextCheckpointKey() throws {
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","text_config":{
          "model_type":"qwen4_exp_text","hidden_size":8,"intermediate_size":8,
          "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
          "layer_types":["linear_attention"],"linear_num_value_heads":1,"linear_num_key_heads":1,
          "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":2,
          "max_position_embeddings":131072,"vocab_size":32,"rms_norm_eps":0.000001,
          "attention_bias":false,"attention_dropout":0,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5}},
          "vision_config":{"depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
          "in_channels":3,"out_hidden_size":8,"patch_size":2,"spatial_merge_size":2,
          "temporal_patch_size":2,"num_position_embeddings":16,"hidden_act":"gelu_pytorch_tanh"}}
        """#.utf8))
        let tower = Q35VisionTower(config: config)
        let updates = Q35VisionTower.mapVisionWeight(
            "vision_tower.patch_embed.proj.bias", MLXArray.ones([8])
        )
        try tower.update(parameters: ModuleParameters.unflattened(updates), verify: [.noUnusedKeys, .shapeMismatch])
        let bias = try XCTUnwrap(tower.parameters().flattened().first { $0.0 == "visionTower.patch_embed.proj.bias" })
        XCTAssertEqual(bias.1.asArray(Float.self), [Float](repeating: 1, count: 8))
    }

    func testInstalledVisionShapesOCRAndImageOrder() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS=1 for installed vision qualification.")
        }
        let root = try XCTUnwrap(env["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        let directory = try XCTUnwrap(env["MERERUN_TEST_Q38_FLASH_NEXT_VISION_FIXTURES"])
        try validateVisionWeights(root: root)
        Memory.clearCache()
        let fixtures = try Self.makeFixtures(directory: URL(fileURLWithPath: directory, isDirectory: true))
        let generator = Q35Generator(
            modelId: Q35Resources.q38FlashNextMixedModelId,
            prefixKVCacheEnabled: true, continuousBatchingEnabled: false
        )
        do {
            try await qualifyVision(generator, root: root, fixtures: fixtures)
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func validateVisionWeights(root: String) throws {
        let directory = URL(fileURLWithPath: root, isDirectory: true)
        let config = try JSONDecoder().decode(
            Q35Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json"))
        )
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        )
        let expected = Set(index.weightMap.keys.flatMap {
            Q35VisionTower.mapVisionWeight($0, MLXArray.zeros([1])).map(\.0)
        })
        let tower = Q35VisionTower(config: config)
        try tower.loadWeights(from: Q35Resources(rootURL: directory))
        let parameters = tower.parameters().flattened()
        let actual = Set(parameters.map(\.0))
        XCTAssertEqual(expected, actual, "Vision checkpoint/runtime keys must match without dropped or random tensors")
        MLX.eval(parameters.map(\.1))
        print("[q38-vision-weights] checkpoint_tensors=\(expected.count) runtime_tensors=\(actual.count)")
    }

    private func qualifyVision(_ generator: Q35Generator, root: String, fixtures: [URL]) async throws {
        let shapeQuestion = "Describe the two shapes from left to right, giving each shape's color and shape."
        let cases: [(String, [ChatMessage], [[String]])] = [
            ("scene-a", [.init(role: .user, content: shapeQuestion, imageUrl: fixtures[0].path)],
             [["red", "circle"], ["blue", "square"]]),
            ("scene-b", [.init(role: .user, content: shapeQuestion, imageUrl: fixtures[1].path)],
             [["green", "square"], ["yellow", "circle"]]),
            ("ocr", [.init(role: .user, content: "Read the printed text exactly.", imageUrl: fixtures[2].path)],
             [["orbit"], ["7429"]]),
            ("image-order", [
                .init(role: .user, content: "First image.", imageUrl: fixtures[0].path),
                .init(role: .user, content: "Second image. Name the color of the square in the first image, "
                    + "then the color of the square in the second image.", imageUrl: fixtures[1].path),
            ], [["blue"], ["green"]]),
        ]
        for (label, messages, groups) in cases {
            let response = try await generator.chat(
                ChatRequest(messages: messages, maxTokens: 128, temperature: 0, topP: 1,
                            showThinking: false, maxContextTokens: 8_192),
                modelPath: root, progressHandler: nil
            )
            print("[q38-vision] case=\(label) prompt_tokens=\(response.promptTokens ?? 0) "
                + "output_tokens=\(response.tokensGenerated) text=\(String(reflecting: response.response))")
            let output = response.response.lowercased()
            var remainder = output[...]
            for words in groups {
                // A shape/color pair may be prose or JSON with either key
                // order. Require complete pairs in image order, not one format.
                var end = remainder.startIndex
                for word in words {
                    if let range = remainder.range(of: word) {
                        end = max(end, range.upperBound)
                    } else {
                        XCTFail("\(label): missing/incorrect order for \(word): \(output)")
                    }
                }
                remainder = remainder[end...]
            }
            XCTAssertEqual(response.acceleration?.route, "final-target-pipelined")
        }
        let cache = await generator.prefixKVCacheStats()
        XCTAssertEqual(cache.entries, 0, "Vision prompts must not populate text-only prefix checkpoints")
    }

    private static func makeFixtures(directory: URL) throws -> [URL] {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return try (0..<3).map { scene in
            let context = try XCTUnwrap(CGContext(
                data: nil, width: 512, height: 384, bitsPerComponent: 8, bytesPerRow: 512 * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ))
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 512, height: 384))
            if scene == 2 {
                let attributes: [NSAttributedString.Key: Any] = [
                    NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Menlo-Bold" as CFString, 60, nil),
                    NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0, alpha: 1),
                ]
                let line = CTLineCreateWithAttributedString(NSAttributedString(string: "ORBIT 7429", attributes: attributes))
                context.textPosition = CGPoint(x: 65, y: 170)
                CTLineDraw(line, context)
            } else {
                context.setFillColor(scene == 0
                    ? CGColor(red: 0.9, green: 0.05, blue: 0.05, alpha: 1)
                    : CGColor(red: 0.05, green: 0.65, blue: 0.15, alpha: 1))
                let left = CGRect(x: 55, y: 117, width: 150, height: 150)
                if scene == 0 { context.fillEllipse(in: left) } else { context.fill(left) }
                context.setFillColor(scene == 0
                    ? CGColor(red: 0.05, green: 0.15, blue: 0.9, alpha: 1)
                    : CGColor(red: 1, green: 0.85, blue: 0, alpha: 1))
                let right = CGRect(x: 307, y: 117, width: 150, height: 150)
                if scene == 0 { context.fill(right) } else { context.fillEllipse(in: right) }
            }
            let url = directory.appendingPathComponent("fixture-\(scene).png")
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            return url
        }
    }
}
#endif
