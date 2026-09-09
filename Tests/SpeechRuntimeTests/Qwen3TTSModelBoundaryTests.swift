import MereRunMLXTestSupport
import MLX
import XCTest
@testable import AudioQwen3TTSModel

final class Qwen3TTSModelBoundaryTests: MLXTestCase {
    func testTalkerCachedDecodeMatchesFullSequenceProjection() {
        let config = Qwen3TTSTalkerConfig(
            codePredictorConfig: Qwen3TTSTalkerCodePredictorConfig(
                vocabSize: 32, hiddenSize: 8, intermediateSize: 16, numHiddenLayers: 1,
                numAttentionHeads: 4, numKeyValueHeads: 2, headDim: 2, numCodeGroups: 2
            ),
            vocabSize: 32, hiddenSize: 8, intermediateSize: 16, numHiddenLayers: 1,
            numAttentionHeads: 4, numKeyValueHeads: 2, headDim: 2,
            ropeScaling: nil, numCodeGroups: 2, textHiddenSize: 8, textVocabSize: 32
        )
        let talker = Qwen3TTSTalkerForConditionalGeneration(config: config)
        let input = MLXArray((0..<32).map { Float($0) / 32 }, [1, 4, 8])
        let cache = talker.makeCache()
        let prefix = talker(input[0..., 0..<3, 0...], cache: cache)
        eval(prefix.logits, prefix.hidden)
        let next = talker(input[0..., 3..<4, 0...], cache: cache)
        let full = talker(input)
        eval(next.logits, next.hidden, full.logits, full.hidden)

        XCTAssertEqual(cache[0].offset, 4)
        XCTAssertEqual(full.logits.shape, [1, 1, 32])
        XCTAssertLessThan(MLX.max(MLX.abs(next.logits - full.logits)).item(Float.self), 1e-5)
        XCTAssertLessThan(MLX.max(MLX.abs(next.hidden - full.hidden)).item(Float.self), 1e-5)
    }

    func testSpeechTokenizerSanitizesFusedAttentionAndConvolutionLayout() throws {
        let config = Qwen3TTSTokenizerConfig(encoderConfig: Qwen3TTSTokenizerEncoderConfig())
        let prefix = "encoder.encoder_transformer.layers.0.self_attn."
        let q = MLX.ones([8, 8])
        let k = MLX.ones([4, 8]) * 2
        let v = MLX.ones([4, 8]) * 3
        let conv = MLXArray((0..<24).map(Float.init), [2, 3, 4])
        let sanitized = Qwen3TTSSpeechTokenizer.sanitize([
            prefix + "q_proj.weight": q,
            prefix + "k_proj.weight": k,
            prefix + "v_proj.weight": v,
            "encoder.encoder.layers.0.weight": conv
        ], config: config)
        let fused = try XCTUnwrap(sanitized["encoder_model.encoder_transformer.transformer.layers.0.self_attn.in_proj.weight"])
        let convolution = try XCTUnwrap(sanitized["encoder_model.encoder.init_conv1d.conv.weight"])
        eval(fused, convolution)
        XCTAssertEqual(fused.shape, [16, 8])
        XCTAssertEqual(fused[0..<8].asArray(Float.self), Array(repeating: 1, count: 64))
        XCTAssertEqual(fused[8..<12].asArray(Float.self), Array(repeating: 2, count: 32))
        XCTAssertEqual(fused[12..<16].asArray(Float.self), Array(repeating: 3, count: 32))
        XCTAssertEqual(convolution.shape, [2, 4, 3])
        XCTAssertEqual(convolution.asArray(Float.self), conv.transposed(0, 2, 1).asArray(Float.self))
    }

    func testSpeechDecoderChunkingPreservesAudioWhenContextCoversPrefix() {
        let config = Qwen3TTSTokenizerDecoderConfig(
            latentDim: 8, codebookDim: 8, codebookSize: 16, decoderDim: 8,
            hiddenSize: 8, intermediateSize: 16, maxPositionEmbeddings: 64,
            headDim: 2, numAttentionHeads: 4, numHiddenLayers: 1, numKeyValueHeads: 2,
            numQuantizers: 2, numSemanticQuantizers: 1, semanticCodebookSize: 16,
            upsampleRates: [2], upsamplingRatios: [2], vectorQuantizationHiddenDimension: 8
        )
        let decoder = Qwen3TTSSpeechTokenizerDecoder(config: config)
        let codes = MLXArray([Int32(1), 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7], [1, 2, 6])
        let full = decoder(codes)
        let chunked = decoder.chunkedDecode(codes: codes, chunkSize: 2, leftContextSize: 6)
        eval(full, chunked)
        XCTAssertEqual(full.shape, [1, 1, 24])
        XCTAssertEqual(chunked.shape, full.shape)
        XCTAssertTrue(chunked.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertLessThan(MLX.max(MLX.abs(full - chunked)).item(Float.self), 1e-5)
    }
}
