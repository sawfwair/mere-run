import MLX
import MLXFast
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
}
