import Foundation
import MLX
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
@testable import MereRunLTXModel
import XCTest

final class LTXSamplerSupportModelTests: MLXTestCase {
    func testRes2sPhiUsesStableLimitAtZero() {
        XCTAssertEqual(LTXRes2s.phi(order: 1, negativeStep: 0), 1, accuracy: 1e-12)
        XCTAssertEqual(LTXRes2s.phi(order: 2, negativeStep: 0), 0.5, accuracy: 1e-12)
    }

    func testRes2sCoefficientsMatchOfficialFormula() {
        let coefficients = LTXRes2s.coefficients(step: log(2))
        XCTAssertEqual(coefficients.a21, 0.42255559429217385, accuracy: 1e-12)
        XCTAssertEqual(coefficients.b1, -0.08267358032783734, accuracy: 1e-12)
        XCTAssertEqual(coefficients.b2, 0.804021100772319, accuracy: 1e-12)
    }

    func testEulerStepLandsOnDenoisedAtTerminalSigma() {
        let sample = MLXArray([Float(3), -1])
        let denoised = MLXArray([Float(1), 2])
        let next = ltxEulerStep(sample: sample, denoised: denoised, sigma: 0.5, nextSigma: 0)
        XCTAssertEqual(next.asArray(Float.self), [1, 2])
    }

    func testOfficialHQSamplerConfiguration() {
        XCTAssertEqual(LTXSamplerConfiguration.hq.mode, .res2s)
        XCTAssertEqual(LTXSamplerConfiguration.hq.eta, 0.5)
        XCTAssertEqual(LTXSamplerConfiguration.hq.noiseSeedOffset, -1)
        XCTAssertEqual(LTXSamplerConfiguration.hq.substepNoiseSeedOffset, 9_999)
        XCTAssertTrue(LTXSamplerConfiguration.hq.res2sBongMath)
    }
}
