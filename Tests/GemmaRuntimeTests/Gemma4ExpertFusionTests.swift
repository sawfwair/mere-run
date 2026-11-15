import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import MereRunGemmaModel

final class Gemma4ExpertFusionTests: MLXTestCase {
    func testFusedProjectionMatchesNativeBF16StagesAcrossRoutesAndTails() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        for width in [64, 128, 256, 704, 2816] {
            let outputs = width == 64 ? 1408 : 32
            let weights = (sin(MLXArray(0..<(5 * outputs * width)).asType(.float32) * 0.017) * 0.07)
                .reshaped(5, outputs, width).asType(.bfloat16)
            let projection = Gemma4SwitchLinear(packing: weights, groupSize: 64)
            let implementation = try Gemma4MetalExperts()
            implementation.fusedActivation = true
            projection.metalVerification = implementation
            for values in [[Int32(0)], [0, 0, 0, 1, 1, 3, 4, 4, 4], Array(repeating: Int32(2), count: 17)] {
                let indices = MLXArray(values)
                let input = cos(MLXArray(0..<(values.count * width)).asType(.float32) * 0.073)
                    .reshaped(values.count, 1, width).asType(.bfloat16)
                let pair = projection.applyFlat(input, indices: indices, sortedIndices: true)
                let expected = geluApproximate(pair[.ellipsis, ..<(outputs / 2)]) * pair[.ellipsis, (outputs / 2)...]
                let actual = try XCTUnwrap(projection.fusedGateUpActivation(input, indices: indices, sortedIndices: true))
                eval(expected, actual)
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertEqual(actual.shape, [values.count, 1, outputs / 2])
                XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self), expected.view(dtype: .uint16).asArray(UInt16.self),
                    "K=\(width), routes=\(values)")
            }
        }
    }

    func testFusionEligibilityAndCurrentParameters() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        let weights = sin(MLXArray(0..<(5 * 32 * 64)).asType(.float32) * 0.027).reshaped(5, 32, 64).asType(.bfloat16)
        let projection = Gemma4SwitchLinear(packing: weights, groupSize: 64)
        let implementation = try Gemma4MetalExperts()
        projection.metalVerification = implementation
        let input = MLXArray.ones([2, 1, 64], dtype: .bfloat16), indices = MLXArray([Int32(0), 0])
        XCTAssertNil(projection.fusedGateUpActivation(input, indices: indices, sortedIndices: true))
        let unfusedIdentity = implementation.compilationIdentity
        implementation.fusedActivation = true
        XCTAssertFalse(unfusedIdentity === implementation.compilationIdentity)
        XCTAssertNil(projection.fusedGateUpActivation(input, indices: indices, sortedIndices: false))
        let dense = MLXArray.ones([256, 1, 64], dtype: .bfloat16)
        XCTAssertNil(projection.fusedGateUpActivation(dense, indices: MLXArray.zeros([256], dtype: .int32), sortedIndices: true))
        let before = try XCTUnwrap(projection.fusedGateUpActivation(input, indices: indices, sortedIndices: true))
        eval(before)
        XCTAssertGreaterThan(abs(before).max().item(Float.self), 0)
        try projection.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXArray.zeros(projection.weight.shape, dtype: .uint32)),
            ("scales", MLXArray.zeros(try XCTUnwrap(projection.scales).shape, dtype: .bfloat16)),
            ("biases", MLXArray.zeros(try XCTUnwrap(projection.biases).shape, dtype: .bfloat16))
        ]), verify: .all)
        let after = try XCTUnwrap(projection.fusedGateUpActivation(input, indices: indices, sortedIndices: true))
        XCTAssertEqual(abs(after).max().item(Float.self), 0)
        let biased = Gemma4SwitchLinear(inputDims: 64, outputDims: 32, numExperts: 5,
            groupSize: 64, bits: 8, mode: .affine, bias: true)
        biased.metalVerification = implementation
        XCTAssertNil(biased.fusedGateUpActivation(input, indices: indices, sortedIndices: true))
        implementation.fusedActivation = false
        XCTAssertTrue(unfusedIdentity === implementation.compilationIdentity)
    }
}
