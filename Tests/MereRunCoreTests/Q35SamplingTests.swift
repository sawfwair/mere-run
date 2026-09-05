import MLX
import XCTest
@testable import MereRunCore

final class Q35SamplingTests: MereRunCoreTestCase {
    func testRequestSeedsReplayAcrossStreamsAndInterveningRandomWork() async {
        defer { MLXRandom.seed(0) }
        let first = await samples(seed: 7)
        let differentSeed = await samples(seed: 42)
        MLXRandom.seed(991)
        MLX.eval(MLXRandom.uniform(0 ..< 1, [128]))
        let replay = await samples(seed: 7)
        XCTAssertEqual(first, replay)
        XCTAssertNotEqual(first, differentSeed)
        XCTAssertTrue(first.allSatisfy { (0..<8).contains($0) })
    }

    private func samples(seed: UInt64) async -> [Int] {
        await Q35CompiledOperations.withNewDefaultStream(scoped: true) {
            await Q35Sampling.withRequestState(seed: seed) {
                let logits = MLXArray([Float(1), 1, 1, 1, -20])
                let config = GenerationConfig(temperature: 0.7, topK: 4, topP: 0.9, minP: 0.05)
                var result: [Int] = []
                XCTAssertFalse(Q35Sampling.acceptsDraft(probability: 0))
                XCTAssertTrue(Q35Sampling.acceptsDraft(probability: 1))
                for _ in 0..<32 {
                    let token = sampleToken(logits: logits, config: config, previousTokens: [])
                    let accepted = Q35Sampling.acceptsDraft(probability: 0.5)
                    result.append(token + (accepted ? 4 : 0))
                    await Task.yield()
                }
                return result
            }
        }
    }
}
