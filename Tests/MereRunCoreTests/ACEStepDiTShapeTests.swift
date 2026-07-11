import MLX
import MLXFast
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

    func testNativeGroupedQueryAttentionMatchesExpandedReference() {
        let config = ACEStepConfig(
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 8,
            audioAcousticHiddenDim: 4,
            inChannels: 12
        )
        MLXRandom.seed(410)
        let attention = ACEStepAttention(config: config)
        let hidden = MLXRandom.normal([2, 5, config.hiddenSize]).asType(.float32)
        let encoder = MLXRandom.normal([2, 7, config.hiddenSize]).asType(.float32)

        let actual = attention(
            hidden,
            mask: .none,
            encoderHiddenStates: encoder
        )

        let batch = hidden.dim(0)
        let queryLength = hidden.dim(1)
        let keyLength = encoder.dim(1)
        let queries = attention.qNorm(
            attention.qProj(hidden)
                .reshaped(batch, queryLength, config.numAttentionHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        var keys = attention.kNorm(
            attention.kProj(encoder)
                .reshaped(batch, keyLength, config.numKeyValueHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        var values = attention.vProj(encoder)
            .reshaped(batch, keyLength, config.numKeyValueHeads, config.headDim)
            .transposed(0, 2, 1, 3)
        let repeats = config.numAttentionHeads / config.numKeyValueHeads
        keys = MLX.repeated(keys, count: repeats, axis: 1)
        values = MLX.repeated(values, count: repeats, axis: 1)
        let expanded = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values.asType(.float32),
            scale: Float(config.headDim).squareRoot().reciprocal,
            mask: .none
        ).asType(queries.dtype)
        let expected = attention.oProj(
            expanded.transposed(0, 2, 1, 3).reshaped(batch, queryLength, -1)
        )

        MLX.eval(actual, expected)
        let maxDifference = MLX.max(MLX.abs(actual - expected)).item(Float.self)
        XCTAssertLessThan(maxDifference, 1e-5)
    }
}
