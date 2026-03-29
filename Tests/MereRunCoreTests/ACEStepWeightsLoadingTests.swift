import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class ACEStepWeightsLoadingTests: MereRunCoreTestCase {

    func testLoadTurboDecoderWeights() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/acestep-v15-turbo to run this test.")
        }

        let resources = ACEStepResources(rootURL: URL(fileURLWithPath: root))
        let missing = resources.validate()
        if !missing.isEmpty {
            let list = missing.map { $0.path }.joined(separator: "\n")
            throw XCTSkip("ACE-Step turbo checkpoint incomplete:\n\(list)")
        }

        let config = try ACEStepCheckpointLoader.loadConfig(resources: resources)
        let decoder = try ACEStepCheckpointLoader.loadDecoder(resources: resources)

        let B = 1
        let T = 7
        let audioDim = config.audioAcousticHiddenDim
        let contextDim = config.inChannels - audioDim

        let hidden = MLXRandom.normal([B, T, audioDim]).asType(.bfloat16)
        let context = MLXRandom.normal([B, T, contextDim]).asType(.bfloat16)
        let encoderHidden = MLXRandom.normal([B, 13, config.hiddenSize]).asType(.bfloat16)

        let t = MLXArray(Array(repeating: Float(0.5), count: B))
        let out = decoder(
            hiddenStates: hidden,
            timestep: t,
            timestepR: t,
            encoderHiddenStates: encoderHidden,
            encoderAttentionMask: nil,
            contextLatents: context
        )

        XCTAssertEqual(out.shape[0], B)
        XCTAssertEqual(out.shape[1], T)
        XCTAssertEqual(out.shape[2], audioDim)

        let maxAbs = MLX.max(MLX.abs(out.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }
}
