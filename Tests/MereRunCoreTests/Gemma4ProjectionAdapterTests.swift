import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import MereRunCore
@testable import MereRunTensor

final class Gemma4ProjectionAdapterTests: MereRunCoreTestCase {
    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func testFusedProjectionMatchesSeparateProjections() {
        let input = 256
        let outputs = [192, 96, 96]

        MLXRandom.seed(19)
        let projections = outputs.map { out -> QuantizedLinear in
            let weight = MLXRandom.normal([out, input]).asType(.float16)
            return QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: 4)
        }

        guard let fused = FusedQuantizedProjection.fuse(projections) else {
            XCTFail("expected fusion to succeed for uniform QuantizedLinear projections")
            return
        }
        XCTAssertTrue(fused.matches(projections))

        // Gemma and Q35 retain this concatenated projection outside the module
        // tree. Replacing even one source with its production LoRA wrapper must
        // invalidate the retained layout and must not silently drop the delta by
        // building a new fusion over the wrapper.
        let adapted: [Linear?] = [
            LoRAQuantizedLinear(base: projections[0], rank: 2),
            projections[1],
            projections[2],
        ]
        XCTAssertFalse(fused.matches(adapted))
        XCTAssertNil(FusedQuantizedProjection.fuse(adapted))

        let x = MLXRandom.normal([2, 1, input]).asType(.float16)
        let fusedParts = fused.callSplit(x)
        XCTAssertEqual(fusedParts.count, projections.count)
        for (part, projection) in zip(fusedParts, projections) {
            let expected = projection(x)
            XCTAssertEqual(part.shape, expected.shape)
            XCTAssertLessThan(maxAbsDifference(part, expected), 1e-4)
        }
    }
}
