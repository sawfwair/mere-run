import Foundation
import XCTest
@testable import MereRunCore

final class LTXInferenceTimingsTests: XCTestCase {
    func testLoadTimingsCodableRoundTrip() throws {
        let timings = LTXLoadTimings(
            textEncoderSeconds: 1,
            transformerSeconds: 2,
            videoDecoderSeconds: 3,
            upsamplerSeconds: 4,
            audioDecoderSeconds: 5,
            loraAdapterSeconds: 6,
            totalSeconds: 21
        )

        let decoded = try JSONDecoder().decode(
            LTXLoadTimings.self,
            from: JSONEncoder().encode(timings)
        )

        XCTAssertEqual(decoded, timings)
    }

    func testGenerationTimingsDefaultToZero() {
        let timings = LTXGenerationTimings()

        XCTAssertEqual(timings.textEncodingSeconds, 0)
        XCTAssertEqual(timings.promptCacheHits, 0)
        XCTAssertEqual(timings.promptCacheMisses, 0)
        XCTAssertEqual(timings.guidanceProjectionCacheBuildSeconds, 0)
        XCTAssertEqual(timings.guidanceProjectionCacheBuilds, 0)
        XCTAssertEqual(timings.guidanceProjectionCacheReuses, 0)
        XCTAssertEqual(timings.guidanceProjectionCacheFallbacks, 0)
        XCTAssertEqual(timings.teaCacheDecisionSeconds, 0)
        XCTAssertEqual(timings.teaCacheComputedBlockStacks, 0)
        XCTAssertEqual(timings.teaCacheReusedBlockStacks, 0)
        XCTAssertEqual(timings.preparationSeconds, 0)
        XCTAssertEqual(timings.stage1DenoiseSeconds, 0)
        XCTAssertEqual(timings.loraFusionSeconds, 0)
        XCTAssertEqual(timings.upsampleSeconds, 0)
        XCTAssertEqual(timings.stage2DenoiseSeconds, 0)
        XCTAssertEqual(timings.videoDecodeSeconds, 0)
        XCTAssertEqual(timings.audioDecodeSeconds, 0)
        XCTAssertEqual(timings.totalSeconds, 0)
    }
}
