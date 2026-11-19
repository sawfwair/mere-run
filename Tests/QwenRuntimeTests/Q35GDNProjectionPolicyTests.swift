import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

final class Q35GDNProjectionPolicyTests: MLXTestCase {
    func testFusedBackendOptInAndReferenceScope() throws {
        func projection(_ rows: Int) -> PortableQuantizedLinear {
            let p = PortableQuantizedLinear(weight: MLXArray.zeros([rows, 640], dtype: .uint32), bias: nil,
                scales: MLXArray.ones([rows, 80], dtype: .bfloat16),
                biases: MLXArray.zeros([rows, 80], dtype: .bfloat16), groupSize: 64, bits: 4, mode: .affine)
            p.verificationBackend = .m5Matrix
            return p
        }
        let qkv = projection(10_240), z = projection(6_144)
        let reference = try Q35GDNProjectionPolicy.$matrixEnabled.withValue(false) {
            try XCTUnwrap(q35FusedPortableQuantizedLinear(qkv, z))
        }
        XCTAssertEqual(reference.verificationBackend, .reference)
        let matrix = try Q35GDNProjectionPolicy.$matrixEnabled.withValue(true) {
            try XCTUnwrap(q35FusedPortableQuantizedLinear(qkv, z))
        }
        XCTAssertEqual(matrix.verificationBackend, .m5Matrix)
        QuantizedVerificationContext.$suppressM5.withValue(true) {
            XCTAssertEqual(matrix.effectiveVerificationBackend, .reference)
        }
        XCTAssertEqual(matrix.effectiveVerificationBackend, .m5Matrix)
        Q35GDNProjectionPolicy.$matrixEnabled.withValue(true) {
            z.verificationBackend = .reference
            XCTAssertEqual(Q35GDNProjectionPolicy.backend(qkv, z), .reference)
            let b = projection(48), a = projection(48)
            XCTAssertEqual(Q35GDNProjectionPolicy.backend(b, a), .reference)
        }
        XCTAssertEqual((matrix.weight .!= reference.weight).sum().item(Int.self), 0)
        XCTAssertEqual((matrix.scales .!= reference.scales).sum().item(Int.self), 0)
    }
}
