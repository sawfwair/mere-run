import Foundation
import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest

/// Opt-in native reference export; no checkpoint or full model is loaded.
final class Gemma4ExpertActivationProfileTests: MLXTestCase {
    func testExportNativeActivationRounding() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let output = environment["MERERUN_PROFILE_SOV_ACTIVATION_OUTPUT"] else {
            throw XCTSkip("Set an activation reference output path for this diagnostic.")
        }
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Requires Metal.")
        let bits = MLXArray(0..<65536).asType(.uint16)
        let values = bits.view(dtype: .bfloat16)
        let finite = which(isFinite(values), values, MLXArray.zeros(values.shape, dtype: .bfloat16))
        let exhaustive = geluApproximate(finite)
        let pair = (sin(MLXArray(0..<(128 * 1408)).asType(.float32) * 0.017) * 5)
            .reshaped(128, 1, 1408).asType(.bfloat16)
        let gate = pair[.ellipsis, ..<704], up = pair[.ellipsis, 704...]
        let native = geluApproximate(gate) * up
        eval(finite, exhaustive, native)
        XCTAssertEqual(exhaustive.dtype, .bfloat16)
        XCTAssertEqual(native.dtype, .bfloat16)
        try MLX.save(arrays: ["exhaustive_input": finite, "exhaustive_gelu": exhaustive,
                              "strided_gate": gate, "strided_up": up, "native_glu": native],
                     url: URL(fileURLWithPath: output))
    }
}
