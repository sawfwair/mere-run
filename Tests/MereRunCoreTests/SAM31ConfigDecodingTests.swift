import Foundation
import XCTest
@testable import MereRunCore

final class SAM31ConfigDecodingTests: XCTestCase {
    func testDecodesRepresentativeSAM31Config() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let configURL = temp.appendingPathComponent("config.json")
        let json = """
        {
          "model_type": "sam3.1_video",
          "low_res_mask_size": 288,
          "det_nms_thresh": 0.15,
          "score_threshold_detection": 0.42,
          "detector_config": {
            "model_type": "sam3.1",
            "vision_config": {
              "model_type": "sam3_vision_model",
              "fpn_hidden_size": 256,
              "fpn_kernel_size": 2,
              "fpn_stride": 2,
              "scale_factors": [4.0, 2.0, 1.0],
              "num_feature_levels": 3,
              "layer_norm_eps": 0.000001,
              "backbone_config": {
                "model_type": "sam3_vit_model",
                "hidden_size": 1024,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "intermediate_size": 4736,
                "image_size": 1008,
                "patch_size": 14,
                "num_channels": 3,
                "window_size": 24,
                "global_attn_indexes": [7, 15, 23, 31],
                "qkv_bias": true,
                "rope_theta": 10000,
                "pretrain_image_size": 336,
                "layer_norm_eps": 0.000001
              }
            },
            "text_config": {
              "model_type": "clip_text_model",
              "hidden_size": 1024,
              "num_hidden_layers": 24,
              "num_attention_heads": 16,
              "intermediate_size": 4096,
              "vocab_size": 49408,
              "max_position_embeddings": 32,
              "projection_dim": 512,
              "layer_norm_eps": 0.00001,
              "pad_token_id": 1
            },
            "detr_encoder_config": {
              "model_type": "sam3_detr_encoder",
              "hidden_size": 256,
              "num_layers": 6,
              "num_attention_heads": 8,
              "intermediate_size": 2048,
              "dropout": 0.1,
              "layer_norm_eps": 0.000001
            },
            "detr_decoder_config": {
              "model_type": "sam3_detr_decoder",
              "hidden_size": 256,
              "num_layers": 6,
              "num_attention_heads": 8,
              "num_queries": 200,
              "intermediate_size": 2048,
              "dropout": 0.1,
              "layer_norm_eps": 0.000001
            },
            "mask_decoder_config": {
              "model_type": "sam3_mask_decoder",
              "hidden_size": 256,
              "num_attention_heads": 8,
              "num_upsampling_stages": 3,
              "layer_norm_eps": 0.000001
            }
          }
        }
        """
        try Data(json.utf8).write(to: configURL)

        let config = try SAM31ModelConfig.load(from: configURL)
        XCTAssertEqual(config.modelType, "sam3.1_video")
        XCTAssertEqual(config.detNMSThresh, 0.15, accuracy: 0.0001)
        XCTAssertEqual(config.scoreThresholdDetection, 0.42, accuracy: 0.0001)
        XCTAssertEqual(config.detectorConfig.visionConfig.backboneConfig.imageSize, 1008)
        XCTAssertEqual(config.detectorConfig.detrDecoderConfig.numQueries, 200)
        XCTAssertEqual(config.detectorConfig.textConfig.maxPositionEmbeddings, 32)
    }
}
