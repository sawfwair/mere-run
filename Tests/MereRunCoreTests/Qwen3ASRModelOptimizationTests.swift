import MLX
import XCTest
@testable import AudioSTT

final class Qwen3ASRModelOptimizationTests: XCTestCase {
    func testPrefillProjectsOnlyTheFinalPosition() {
        let textConfig = Qwen3ASRDecoderConfig(
            vocabSize: 32,
            hiddenSize: 8,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            intermediateSize: 16,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            rmsNormEps: 1e-6,
            headDim: 2,
            tieWordEmbeddings: true
        )
        let audioConfig = Qwen3ASRAudioEncoderConfig(
            dModel: 8,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            ffnDim: 16,
            maxSourcePositions: 32,
            numMelBins: 128,
            outputDim: 8,
            downsampleHiddenSize: 8,
            nWindow: 4,
            nWindowInfer: 8,
            convChunkSize: 8
        )
        let config = Qwen3ASRModelConfig(
            audioConfig: audioConfig,
            textConfig: textConfig,
            audioTokenId: 29,
            audioStartTokenId: 30,
            audioEndTokenId: 31
        )
        let thinker = Qwen3ASRThinker(config: config)
        let inputIds = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped(1, 4)
        let fullCache = thinker.makeCache()
        let finalCache = thinker.makeCache()

        let full = thinker(inputIds: inputIds, cache: fullCache)
        let final = thinker(
            inputIds: inputIds,
            cache: finalCache,
            lastPositionOnly: true
        )
        eval(full, final)

        XCTAssertEqual(full.shape, [1, 4, 32])
        XCTAssertEqual(final.shape, [1, 1, 32])
        XCTAssertEqual(fullCache[0].offset, 4)
        XCTAssertEqual(finalCache[0].offset, 4)

        let expected = full[0, 3].asType(.float32)
        let actual = final[0, 0].asType(.float32)
        let maxDifference = MLX.max(MLX.abs(expected - actual)).item(Float.self)
        XCTAssertLessThan(maxDifference, 1e-5)
    }
}
