import Foundation
import XCTest
@testable import MereRunCore

final class Flux2DevConfigTests: XCTestCase {
    func testTransformerConfigUsesDiffusersGuidanceDefaultWhenFieldIsAbsent() throws {
        let data = Data("""
        {
          "attention_head_dim": 128,
          "axes_dims_rope": [32, 32, 32, 32],
          "eps": 0.000001,
          "in_channels": 128,
          "joint_attention_dim": 15360,
          "mlp_ratio": 3,
          "num_attention_heads": 48,
          "num_layers": 8,
          "num_single_layers": 48,
          "patch_size": 1,
          "rope_theta": 2000,
          "timestep_guidance_channels": 256
        }
        """.utf8)

        let config = try JSONDecoder().decode(Flux2TransformerConfig.self, from: data)
        XCTAssertNil(config.guidanceEmbeds)
        XCTAssertTrue(config.resolvedGuidanceEmbeds)
        XCTAssertEqual(config.hiddenSize, 6_144)
    }

    func testNestedMistralConfigCarriesArchitectureAndQuantization() throws {
        let data = Data("""
        {
          "model_type": "mistral3",
          "quantization": {"group_size": 64, "bits": 4},
          "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 32768,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 0.00001,
            "rope_theta": 1000000000,
            "vocab_size": 131072,
            "max_position_embeddings": 131072
          }
        }
        """.utf8)

        let config = try JSONDecoder().decode(Flux2TextEncoderConfig.self, from: data)
        XCTAssertEqual(config.architecture, .mistral3)
        XCTAssertEqual(config.hiddenSize, 5_120)
        XCTAssertEqual(config.numHiddenLayers, 40)
        XCTAssertEqual(config.quantizationConfig?.bits, 4)
        XCTAssertEqual(config.quantizationConfig?.groupSize, 64)
    }

    func testMistralWeightMapperDropsUnusedMultimodalHeads() {
        XCTAssertEqual(
            Flux2KleinGenerator.mapTextEncoderWeightKey(
                "language_model.model.layers.0.self_attn.q_proj.weight"
            ),
            "encoder.layers.0.self_attn.q_proj.weight"
        )
        XCTAssertNil(
            Flux2KleinGenerator.mapTextEncoderWeightKey("language_model.lm_head.weight")
        )
        XCTAssertNil(Flux2KleinGenerator.mapTextEncoderWeightKey("vision_tower.patch_conv.weight"))
        XCTAssertNil(
            Flux2KleinGenerator.mapTextEncoderWeightKey("multi_modal_projector.linear_1.weight")
        )
    }

    func testCheckpointFileURLPreservesSafetensorsExtensionForContentAddressedSymlink() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let component = root.appendingPathComponent("vae", isDirectory: true)
        let blob = root.appendingPathComponent("content-addressed-blob")
        try FileManager.default.createDirectory(at: component, withIntermediateDirectories: true)
        try Data().write(to: blob)
        defer { try? FileManager.default.removeItem(at: root) }

        let alias = component.appendingPathComponent("diffusion_pytorch_model.safetensors")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: blob)

        let loaderURL = Flux2KleinGenerator.checkpointFileURL(
            in: component,
            filename: "diffusion_pytorch_model.safetensors"
        )
        XCTAssertEqual(loaderURL.pathExtension, "safetensors")
        XCTAssertEqual(loaderURL.resolvingSymlinksInPath().pathExtension, "")
    }
}
