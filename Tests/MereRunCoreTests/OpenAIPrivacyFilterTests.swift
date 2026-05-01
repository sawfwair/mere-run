import MLX
import XCTest
@testable import MereRunCore

final class OpenAIPrivacyFilterTests: MereRunCoreTestCase {
    func testConfigDecodesStringLabelKeys() throws {
        let json = """
        {
          "model_type": "openai_privacy_filter",
          "vocab_size": 64,
          "hidden_size": 32,
          "id2label": {
            "0": "O",
            "1": "B-private_email"
          },
          "label2id": {
            "O": 0,
            "B-private_email": 1
          }
        }
        """

        let config = try JSONDecoder().decode(OpenAIPrivacyFilterConfig.self, from: Data(json.utf8))
        XCTAssertEqual(config.modelType, "openai_privacy_filter")
        XCTAssertEqual(config.vocabSize, 64)
        XCTAssertEqual(config.hiddenSize, 32)
        XCTAssertEqual(config.id2label[1], "B-private_email")
        XCTAssertEqual(config.label2id["B-private_email"], 1)
    }

    func testModelForwardShapes() throws {
        let config = OpenAIPrivacyFilterConfig(
            vocabSize: 64,
            hiddenSize: 32,
            intermediateSize: 32,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 8,
            slidingWindow: 16,
            maxPositionEmbeddings: 128,
            numLocalExperts: 4,
            numExpertsPerTok: 2
        )
        let model = OpenAIPrivacyFilterModel(config: config)
        let inputs = MLXArray([Int32(0), 1, 2, 3, 4], [1, 5])
        let mask = MLX.ones([1, 5], dtype: .int32)

        let outputs = model(inputIds: inputs, attentionMask: mask)
        MLX.eval(outputs.lastHiddenState, outputs.logits)

        XCTAssertEqual(outputs.lastHiddenState.shape, [1, 5, config.hiddenSize])
        XCTAssertEqual(outputs.logits.shape, [1, 5, config.numLabels])
    }

    func testWeightSanitizerSplitsMoEGateUpProjection() {
        let value = MLXArray(Array(repeating: Float32(1), count: 2 * 3 * 8), [2, 3, 8])
        let updates = OpenAIPrivacyFilterModel.sanitizeWeight(
            key: "model.layers.0.mlp.experts.gate_up_proj",
            value: value
        )

        XCTAssertEqual(updates.map(\.0), [
            "model.layers.0.mlp.experts.gate_proj.weight",
            "model.layers.0.mlp.experts.up_proj.weight",
        ])
        XCTAssertEqual(updates[0].1.shape, [2, 4, 3])
        XCTAssertEqual(updates[1].1.shape, [2, 4, 3])
    }

    func testSlidingWindowMaskUsesRequestedDType() {
        let attentionMask = MLXArray([Int32(1), 1, 1, 0], [1, 4])
        let mask = OpenAIPrivacyFilterBackbone.bidirectionalSlidingWindowMask(
            attentionMask: attentionMask,
            seqLen: 4,
            window: 2,
            dtype: .bfloat16
        )

        XCTAssertEqual(mask.dtype, .bfloat16)
        XCTAssertEqual(mask.shape, [1, 1, 4, 4])
    }

    func testViterbiDecoderRepairsInvalidBIOESPath() throws {
        let labelInfo = try OpenAIPrivacyFilterLabelInfo(classNames: [
            "O",
            "B-private_person",
            "I-private_person",
            "E-private_person",
            "S-private_person",
        ])
        let decoder = OpenAIPrivacyFilterViterbiDecoder(
            labelInfo: labelInfo,
            biases: .zero
        )

        let logProbs: [Float] = [
            0, 9, 10, 0, 0,
            0, 0, 0, 10, 0,
        ]

        let decoded = decoder.decode(logProbs: logProbs, sequenceLength: 2, classCount: 5)
        XCTAssertEqual(decoded, [1, 3], "Decoder should choose a valid B->E path instead of starting with I.")
    }

    func testByteLevelOffsetsMatchDecodedText() {
        let decoded = OpenAIPrivacyFilterByteLevelCodec.decodeTextWithOffsets(
            tokenStrings: ["Hello", "\u{0120}world"]
        )

        XCTAssertEqual(decoded.text, "Hello world")
        XCTAssertEqual(decoded.charStarts, [0, 5])
        XCTAssertEqual(decoded.charEnds, [5, 11])
    }

    func testByteLevelOffsetsPreferOriginalTextWhenTokensArePrefix() {
        let ranges = OpenAIPrivacyFilterByteLevelCodec.charRanges(
            for: ["Hello", "\u{0120}world"],
            in: "Hello world!"
        )

        XCTAssertEqual(ranges?.charStarts, [0, 5])
        XCTAssertEqual(ranges?.charEnds, [5, 11])
    }
}
