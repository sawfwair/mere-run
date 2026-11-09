import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunTensor

final class QuantizedVerificationBackendTests: MLXTestCase {
    func testReferenceRequestScopeRestoresWithoutMutatingSharedProjection() {
        let layer = projection()
        layer.verificationBackend = .m5Matrix
        XCTAssertEqual(layer.effectiveVerificationBackend, .m5Matrix)
        QuantizedVerificationContext.$suppressM5.withValue(true) {
            XCTAssertEqual(layer.effectiveVerificationBackend, .reference)
            XCTAssertEqual(layer.verificationBackend, .m5Matrix)
            QuantizedVerificationContext.$suppressM5.withValue(false) {
                XCTAssertEqual(layer.effectiveVerificationBackend, .m5Matrix)
            }
            XCTAssertEqual(layer.effectiveVerificationBackend, .reference)
        }
        XCTAssertEqual(layer.effectiveVerificationBackend, .m5Matrix)
    }

    func testConcurrentRequestScopesKeepIndependentPolicies() async {
        async let plain = QuantizedVerificationContext.$suppressM5.withValue(true) {
            await Task.yield()
            return QuantizedVerificationContext.suppressM5
        }
        async let accelerated = QuantizedVerificationContext.$suppressM5.withValue(false) {
            await Task.yield()
            return QuantizedVerificationContext.suppressM5
        }
        let values = await (plain, accelerated)
        XCTAssertTrue(values.0)
        XCTAssertFalse(values.1)
        XCTAssertFalse(QuantizedVerificationContext.suppressM5)
    }

    func testFusionPreservesAndInvalidatesBackendPolicy() throws {
        let first = projection()
        let second = projection()
        let reference = try XCTUnwrap(FusedQuantizedProjection.fuse([first, second]))
        XCTAssertTrue(reference.matches([first, second]))
        first.verificationBackend = .m5Matrix
        XCTAssertFalse(reference.matches([first, second]))
        XCTAssertNil(FusedQuantizedProjection.fuse([first, second]), "Mixed arithmetic must not fuse")
        second.verificationBackend = .m5Matrix
        let matrix = try XCTUnwrap(FusedQuantizedProjection.fuse([first, second]))
        XCTAssertTrue(matrix.matches([first, second]))
        second.verificationBackend = .reference
        XCTAssertFalse(matrix.matches([first, second]))
    }

    func testOnlyQualifiedShapesSelectMatrixPolicy() {
        XCTAssertNil(QuantizedVerificationBackend.reference.matrixSplits(inputSize: 5_120, outputSize: 248_320))
        XCTAssertEqual(QuantizedVerificationBackend.m5Matrix.matrixSplits(inputSize: 5_120, outputSize: 248_320), 1)
        XCTAssertNil(QuantizedVerificationBackend.m5Matrix.matrixSplits(inputSize: 2_048, outputSize: 2_048))
    }

    private func projection() -> PortableQuantizedLinear {
        PortableQuantizedLinear(
            weight: MLXArray.zeros([64, 64], dtype: .uint32), bias: nil,
            scales: MLXArray.ones([64, 8], dtype: .bfloat16),
            biases: MLXArray.zeros([64, 8], dtype: .bfloat16),
            groupSize: 64, bits: 4, mode: .affine
        )
    }
}
