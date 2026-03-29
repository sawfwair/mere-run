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
        let encoder = MLXRandom.normal([B, 7, config.hiddenSize]).asType(.float32)

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
}
