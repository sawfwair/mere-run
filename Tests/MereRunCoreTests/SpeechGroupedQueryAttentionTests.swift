import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
@testable import AudioSTT
@testable import AudioTTS
@testable import MereRunCore

private final class RecordingSpeechKVCache: KVCache {
    private(set) var offset = 0
    private(set) var keyHeadCounts: [Int] = []
    private(set) var valueHeadCounts: [Int] = []

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        keyHeadCounts.append(keys.dim(1))
        valueHeadCounts.append(values.dim(1))
        offset += keys.dim(2)
        return (keys, values)
    }

    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n == 1 ? .none : .causal
    }

    func fork() -> KVCache {
        let copy = RecordingSpeechKVCache()
        copy.offset = offset
        copy.keyHeadCounts = keyHeadCounts
        copy.valueHeadCounts = valueHeadCounts
        return copy
    }
}

final class SpeechGroupedQueryAttentionTests: XCTestCase {
    private let hidden = MLXArray((0..<24).map { Float($0) / 24 }, [1, 3, 8])
    private let positionEmbeddings = (
        cos: MLX.ones([1, 3, 2], dtype: .float32),
        sin: MLX.zeros([1, 3, 2], dtype: .float32)
    )

    func testASRStoresCompactGroupedQueryHeads() {
        let config = Qwen3ASRDecoderConfig(
            vocabSize: 32,
            hiddenSize: 8,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            intermediateSize: 16,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            headDim: 2
        )
        let attention = Qwen3ASRDecoderAttention(config: config)
        let cache = RecordingSpeechKVCache()

        let output = attention(hidden, mask: .causal, cache: cache)
        MLX.eval(output)

        XCTAssertEqual(output.shape, [1, 3, 8])
        XCTAssertEqual(cache.keyHeadCounts, [2])
        XCTAssertEqual(cache.valueHeadCounts, [2])
    }

    func testTTSModelsStoreCompactGroupedQueryHeads() {
        let genericConfig = Qwen3TTSModelConfiguration(
            vocabSize: 32,
            textVocabSize: 32,
            hiddenSize: 8,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            intermediateSize: 16,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            headDim: 2
        )
        let genericCache = RecordingSpeechKVCache()
        let generic = Qwen3TTSModel(configuration: genericConfig)
        let genericOutput = generic.forwardText(
            inputIds: MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3),
            cache: [genericCache]
        )

        let predictorConfig = Qwen3TTSTalkerCodePredictorConfig(
            vocabSize: 32,
            hiddenSize: 8,
            intermediateSize: 16,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 2,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            numCodeGroups: 2
        )
        let talkerConfig = Qwen3TTSTalkerConfig(
            codePredictorConfig: predictorConfig,
            vocabSize: 32,
            hiddenSize: 8,
            intermediateSize: 16,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 2,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            ropeScaling: nil,
            numCodeGroups: 2,
            textHiddenSize: 8,
            textVocabSize: 32
        )
        let talkerCache = RecordingSpeechKVCache()
        let talkerOutput = Qwen3TTSTalkerModel(config: talkerConfig)(
            hidden,
            cache: [talkerCache]
        )

        let predictorCache = RecordingSpeechKVCache()
        let predictorOutput = Qwen3TTSCodePredictorModel(
            config: predictorConfig,
            talkerHiddenSize: 8
        )(
            hidden,
            cache: [predictorCache]
        )
        MLX.eval(genericOutput, talkerOutput, predictorOutput)

        XCTAssertEqual(genericOutput.shape, [1, 3, 32])
        XCTAssertEqual(talkerOutput.shape, [1, 3, 8])
        XCTAssertEqual(predictorOutput.shape, [1, 3, 8])
        XCTAssertEqual(genericCache.keyHeadCounts, [2])
        XCTAssertEqual(talkerCache.keyHeadCounts, [2])
        XCTAssertEqual(predictorCache.keyHeadCounts, [2])
    }

    func testSpeechTokenizerStoresCompactGroupedQueryHeads() {
        let encoderConfig = Qwen3TTSTokenizerEncoderConfig(
            headDim: 2,
            hiddenSize: 8,
            intermediateSize: 16,
            maxPositionEmbeddings: 64,
            numAttentionHeads: 4,
            numHiddenLayers: 1,
            numKeyValueHeads: 2
        )
        let encoderCache = RecordingSpeechKVCache()
        let encoderOutput = MimiAttention(config: encoderConfig)(
            hidden,
            cache: encoderCache,
            mask: .causal
        )

        let decoderConfig = Qwen3TTSTokenizerDecoderConfig(
            hiddenSize: 8,
            intermediateSize: 16,
            maxPositionEmbeddings: 64,
            headDim: 2,
            numAttentionHeads: 4,
            numHiddenLayers: 1,
            numKeyValueHeads: 2
        )
        let decoderCache = RecordingSpeechKVCache()
        let decoderOutput = DecoderAttention(config: decoderConfig, layerIdx: 0)(
            hidden,
            positionEmbeddings: positionEmbeddings,
            mask: .causal,
            cache: decoderCache
        )
        MLX.eval(encoderOutput, decoderOutput)

        XCTAssertEqual(encoderOutput.shape, [1, 3, 8])
        XCTAssertEqual(decoderOutput.shape, [1, 3, 8])
        XCTAssertEqual(encoderCache.keyHeadCounts, [2])
        XCTAssertEqual(decoderCache.keyHeadCounts, [2])
    }

    func testASRNativeBFloat16AttentionStaysCloseToLegacyFloat32Path() {
        let config = Qwen3ASRDecoderConfig(
            vocabSize: 64,
            hiddenSize: 32,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            intermediateSize: 64,
            maxPositionEmbeddings: 128,
            ropeTheta: 10_000,
            headDim: 8
        )
        MLXRandom.seed(913)
        let attention = Qwen3ASRDecoderAttention(config: config)
        attention.update(parameters: attention.parameters().mapValues { $0.asType(.bfloat16) })
        let input = MLXRandom.normal([2, 17, config.hiddenSize]).asType(.bfloat16)

        let actual = attention(input, mask: .causal, cache: nil)
        let expected = legacyASROutput(attention: attention, config: config, input: input)

        MLX.eval(actual, expected)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertLessThan(maxAbsoluteDifference(actual, expected), 0.02)
    }

    func testTTSNativeBFloat16AttentionStaysCloseToLegacyFloat32Path() {
        let config = Qwen3TTSModelConfiguration(
            vocabSize: 64,
            textVocabSize: 64,
            hiddenSize: 32,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            intermediateSize: 64,
            maxPositionEmbeddings: 128,
            ropeTheta: 10_000,
            headDim: 8
        )
        MLXRandom.seed(914)
        let attention = Qwen3TTSAttention(configuration: config)
        attention.update(parameters: attention.parameters().mapValues { $0.asType(.bfloat16) })
        let input = MLXRandom.normal([2, 17, config.hiddenSize]).asType(.bfloat16)
        let rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: 1.0
        )

        let actual = attention(input, mask: .causal, cache: nil, rope: rope)
        let expected = legacyTTSOutput(
            attention: attention,
            config: config,
            input: input,
            rope: rope
        )

        MLX.eval(actual, expected)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertLessThan(maxAbsoluteDifference(actual, expected), 0.02)
    }

    private func legacyASROutput(
        attention: Qwen3ASRDecoderAttention,
        config: Qwen3ASRDecoderConfig,
        input: MLXArray
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        var queries = attention.qNorm(
            attention.qProj(input).reshaped(batch, sequence, config.numAttentionHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        var keys = attention.kNorm(
            attention.kProj(input).reshaped(batch, sequence, config.numKeyValueHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        let values = attention.vProj(input)
            .reshaped(batch, sequence, config.numKeyValueHeads, config.headDim)
            .transposed(0, 2, 1, 3)
        let rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: 1.0
        )
        queries = rope(queries.asType(.bfloat16), offset: 0)
        keys = rope(keys.asType(.bfloat16), offset: 0)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values.asType(.float32),
            scale: pow(Float(config.headDim), -0.5),
            mask: .causal
        ).asType(queries.dtype)
        return attention.oProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, -1))
    }

    private func legacyTTSOutput(
        attention: Qwen3TTSAttention,
        config: Qwen3TTSModelConfiguration,
        input: MLXArray,
        rope: RoPE
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        var queries = attention.qNorm(
            attention.qProj(input).reshaped(batch, sequence, config.numAttentionHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        var keys = attention.kNorm(
            attention.kProj(input).reshaped(batch, sequence, config.numKeyValueHeads, config.headDim)
        ).transposed(0, 2, 1, 3)
        let values = attention.vProj(input)
            .reshaped(batch, sequence, config.numKeyValueHeads, config.headDim)
            .transposed(0, 2, 1, 3)
        queries = rope(queries.asType(.bfloat16), offset: 0)
        keys = rope(keys.asType(.bfloat16), offset: 0)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values.asType(.float32),
            scale: pow(Float(config.headDim), -0.5),
            mask: .causal
        ).asType(queries.dtype)
        return attention.oProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, -1))
    }

    private func maxAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        MLX.max(MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
    }
}
