import MLX
import XCTest
@testable import MereRunCore

final class Q35SamplingTests: MereRunCoreTestCase {
    func testRequestControlsReachSamplerWithoutEnablingNeutralPenalties() {
        var request = ChatRequest(
            messages: [], temperature: 0.6, topP: 0.95, topK: 20, minP: 0,
            presencePenalty: 0, frequencyPenalty: 0, repetitionPenalty: 1
        )
        let coding = Q35Sampling.generationConfig(for: request, promptTokenCount: 128)
        XCTAssertEqual(coding.temperature, 0.6)
        XCTAssertEqual(coding.topP, 0.95)
        XCTAssertEqual(coding.topK, 20)
        XCTAssertEqual(coding.minP, 0)
        XCTAssertFalse(coding.hasActivePenalties)
        XCTAssertNil(coding.repetitionPenalty)
        request.presencePenalty = 1.5
        request.frequencyPenalty = 0.25
        request.repetitionPenalty = 1.1
        let general = Q35Sampling.generationConfig(for: request, promptTokenCount: 128)
        XCTAssertTrue(general.hasActivePenalties)
        XCTAssertEqual(general.presencePenalty, 1.5)
        XCTAssertEqual(general.frequencyPenalty, 0.25)
        XCTAssertEqual(general.repetitionPenalty, 1.1)
        XCTAssertEqual(general.penaltyPromptTokenCount, 128)
        XCTAssertEqual(general.repetitionContextSize, Int.max)
    }

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
