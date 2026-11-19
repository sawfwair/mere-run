#if os(macOS)
import MereRunMLXTestSupport
import MereRunTensor
import MLX
import MLXRandom
import XCTest
@testable import MereRunQwenModel

final class Q35PrefillGDNRecurrenceTests: MLXTestCase {
    func testEightColumnOutputsAndContinuationStateMatch() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "GDN prefill requires Metal.")
        MLXRandom.seed(1471)
        let logDecay = MLXRandom.normal([48]), bias = MLXRandom.normal([48])
        var state = MLXRandom.normal([1, 48, 128, 128]).asType(.float32)
        for rows in [1024, 526] {
            let q = (MLXRandom.normal([1, rows, 16, 128]) * 0.08).asType(.bfloat16)
            let k = (MLXRandom.normal(q.shape) * 0.08).asType(.bfloat16)
            let v = MLXRandom.normal([1, rows, 48, 128]).asType(.bfloat16)
            let a = MLXRandom.normal([1, rows, 48]).asType(.bfloat16)
            let b = MLXRandom.normal(a.shape).asType(.bfloat16)
            func run(wide: Bool) throws -> (MLXArray, MLXArray) {
                try QuantizedPrefillContext.$mode.withValue(.gdnRepaired) {
                    try Q35PrefillGDNRecurrence.$enabled.withValue(wide) {
                        XCTAssertEqual(Q35PrefillGDNRecurrence.eligible(q: q, k: k, v: v, keyHeads: 16, valueHeads: 48), wide)
                        return try XCTUnwrap(q35GatedDeltaUpdateMetal(q: q, k: k, v: v, a: a, b: b, aLog: logDecay,
                            dtBias: bias, state: state, numKeyHeads: 16, numValueHeads: 48, valueHeadDim: 128))
                    }
                }
            }
            let reference = try run(wide: false)
            let wide = try run(wide: true)
            XCTAssertEqual((reference.0 .!= wide.0).sum().item(Int.self), 0)
            XCTAssertEqual((reference.1 .!= wide.1).sum().item(Int.self), 0)
            state = wide.1
        }
    }
}
#endif
