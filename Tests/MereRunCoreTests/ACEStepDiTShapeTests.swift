import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class ACEStepDiTShapeTests: MereRunCoreTestCase {

    func testForwardShapesWithPadding() {
        let config = ACEStepConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            numHiddenLayers: 2,
            numAttentionHeads: 8,
            numKeyValueHeads: 4,
            encoderHiddenSize: 48,
            encoderIntermediateSize: 192,
            encoderNumAttentionHeads: 6,
            encoderNumKeyValueHeads: 3,
            headDim: 8,
            maxPositionEmbeddings: 1024,
            useSlidingWindow: true,
            slidingWindow: 16,
            layerTypes: ["sliding_attention", "full_attention"],
            audioAcousticHiddenDim: 4,
            inChannels: 12,
            patchSize: 2
        )

        let model = ACEStepDiT(config: config)

        let B = 2
        let T = 11  // odd -> exercise pad/unpad logic
        let audioDim = config.audioAcousticHiddenDim
        let contextDim = config.inChannels - audioDim

        let hidden = MLXRandom.normal([B, T, audioDim]).asType(.float32)
        let context = MLXRandom.normal([B, T, contextDim]).asType(.float32)
        let encoder = MLXRandom.normal([B, 7, config.encoderHiddenSize]).asType(.float32)

        let t = MLXArray(Array(repeating: Float(0.5), count: B))
        let out = model(
            hiddenStates: hidden,
            timestep: t,
            timestepR: t,
            encoderHiddenStates: encoder,
            encoderAttentionMask: nil,
            contextLatents: context
        )

        XCTAssertEqual(out.dim(0), B)
        XCTAssertEqual(out.dim(1), T)
        XCTAssertEqual(out.dim(2), audioDim)
    }

    func testDecodesSeparateXLConditionEncoderDimensions() throws {
        let json = """
        {
          "hidden_size": 2560,
          "intermediate_size": 9728,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "encoder_hidden_size": 2048,
          "encoder_intermediate_size": 6144,
          "encoder_num_attention_heads": 16,
          "encoder_num_key_value_heads": 8,
          "head_dim": 128
        }
        """

        let config = try JSONDecoder().decode(ACEStepConfig.self, from: Data(json.utf8))

        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.encoderHiddenSize, 2048)
        XCTAssertEqual(config.conditionEncoderConfig.hiddenSize, 2048)
        XCTAssertEqual(config.conditionEncoderConfig.numAttentionHeads, 16)
        XCTAssertEqual(config.conditionEncoderConfig.intermediateSize, 6144)
    }
}
