import MLX
import XCTest
@testable import MereRunCore

final class ACEStepDCWTests: MereRunCoreTestCase {

    func testHaarRoundTripsOddLengthLatents() {
        let latents = MLXArray([Float(1), 3, 5], [1, 3, 1]).asType(.float32)

        let (low, high) = ACEStepDCW.haarDWT1D(latents)
        let reconstructed = ACEStepDCW.haarIDWT1D(low: low, high: high, targetFrames: 3).asType(.float32)

        XCTAssertEqual(reconstructed.shape, latents.shape)
        let maxDiff = MLX.max(MLX.abs(reconstructed - latents)).item(Float.self)
        XCTAssertEqual(maxDiff, 0, accuracy: 1e-5)
    }

    func testDoubleModeMatchesNativeHaarSchedule() {
        let xNext = MLXArray([Float(1), 3, 5], [1, 3, 1]).asType(.float32)
        let denoised = MLXArray.zeros([1, 3, 1], dtype: .float32)

        let corrected = ACEStepDCW.applyHaar(
            xNext: xNext,
            denoised: denoised,
            tCurr: 0.5,
            enabled: true,
            mode: .double,
            scaler: 0.1,
            highScaler: 0.2
        ).asType(.float32)

        XCTAssertEqual(corrected[0, 0, 0].item(Float.self), 1.0, accuracy: 1e-4)
        XCTAssertEqual(corrected[0, 1, 0].item(Float.self), 3.2, accuracy: 1e-4)
        XCTAssertEqual(corrected[0, 2, 0].item(Float.self), 5.375, accuracy: 1e-4)
    }

    func testDisabledDCWIsIdentity() {
        let xNext = MLXArray([Float(1), 3, 5], [1, 3, 1]).asType(.float32)
        let denoised = MLXArray.zeros([1, 3, 1], dtype: .float32)

        let corrected = ACEStepDCW.applyHaar(
            xNext: xNext,
            denoised: denoised,
            tCurr: 0.5,
            enabled: false,
            mode: .double,
            scaler: 0.1,
            highScaler: 0.2
        ).asType(.float32)

        let maxDiff = MLX.max(MLX.abs(corrected - xNext)).item(Float.self)
        XCTAssertEqual(maxDiff, 0, accuracy: 1e-5)
    }
}
