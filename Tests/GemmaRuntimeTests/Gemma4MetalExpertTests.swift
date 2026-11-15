import XCTest
import MLX
import MLXNN
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class Gemma4MetalExpertTests: MLXTestCase {
    func testAutomaticReductionRequiresQualifiedDeviceAndArchitecture() {
        XCTAssertTrue(Gemma4MetalExperts.supportsAutomaticReduction(deviceName: "Apple M5 Max", architecture: "applegpu_g17s"))
        for name in [nil, "", "Apple M5", "Apple M5 Pro", "Apple M4 Max", "Apple M6 Max"] as [String?] {
            XCTAssertFalse(Gemma4MetalExperts.supportsAutomaticReduction(deviceName: name, architecture: "applegpu_g17s"))
        }
        for architecture in [nil, "", "Unknown", "applegpu_g16s", "applegpu_g18s"] as [String?] {
            XCTAssertFalse(Gemma4MetalExperts.supportsAutomaticReduction(deviceName: "Apple M5 Max", architecture: architecture))
        }
        Device.withDefaultDevice(.cpu) {
            XCTAssertFalse(Gemma4MetalExperts.automaticDistributedReduction)
        }
    }

    func testAutomaticProjectionMatchesExplicitReference() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        let kernel = try Gemma4MetalExperts()
        XCTAssertEqual(kernel.distributedReduction, Gemma4MetalExperts.automaticDistributedReduction)
        for width in [704, 2816] {
            let weights = sin(MLXArray(0..<(5 * 32 * width)).asType(.float32) * 0.027)
                .reshaped(5, 32, width).asType(.bfloat16)
            let packed = quantized(weights, groupSize: 64, bits: 8)
            let indices = MLXArray([Int32(0), 0, 0, 1, 3, 3, 4])
            let input = cos(MLXArray(0..<(7 * width)).asType(.float32) * 0.073).reshaped(7, 1, width).asType(.bfloat16)
            kernel.distributedReduction = Gemma4MetalExperts.automaticDistributedReduction
            let automatic = kernel(input, weight: packed.wq, scales: packed.scales, biases: try XCTUnwrap(packed.biases), indices: indices)
            kernel.distributedReduction = false
            let reference = kernel(input, weight: packed.wq, scales: packed.scales, biases: try XCTUnwrap(packed.biases), indices: indices)
            eval(automatic, reference)
            XCTAssertEqual(automatic.view(dtype: .uint16).asArray(UInt16.self), reference.view(dtype: .uint16).asArray(UInt16.self))
        }
    }

    func testReusedRoutesMatchQ8AcrossTailsAndExpertBoundaries() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        let kernel = try Gemma4MetalExperts()
        for width in [64, 128, 256, 704, 2816] {
            let outputs = width == 64 ? 1408 : (width == 2816 ? 16 : (width == 704 ? 32 : 24))
            let original = (sin(MLXArray(0..<(5 * outputs * width)).asType(.float32) * 0.017) * 0.07)
                .reshaped(5, outputs, width).asType(.bfloat16)
            let packed = quantized(original, groupSize: 64, bits: 8)
            for values in [[Int32(0)], [0, 0, 0, 1, 1, 3, 4, 4, 4], Array(repeating: Int32(2), count: 17)] {
                let indices = MLXArray(values)
                let x = cos(MLXArray(0..<(values.count * width)).asType(.float32) * 0.073)
                    .reshaped(values.count, 1, width).asType(.bfloat16)
                let expected = gatherQuantizedMM(x, packed.wq, scales: packed.scales, biases: packed.biases,
                    rhsIndices: indices, transpose: true, groupSize: 64, bits: 8, sortedIndices: true)
                eval(expected)
                for byteLoads in [false, true] {
                    kernel.byteLoads = byteLoads
                    for layout in [false, true] {
                        kernel.optimizedLayout = layout
                        for distributed in [false, true] {
                            kernel.distributedReduction = distributed
                            let actual = kernel(x, weight: packed.wq, scales: packed.scales, biases: try XCTUnwrap(packed.biases), indices: indices)
                            eval(actual)
                            XCTAssertEqual(actual.shape, [values.count, 1, outputs])
                            XCTAssertEqual(actual.dtype, .bfloat16)
                            XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self), expected.view(dtype: .uint16).asArray(UInt16.self),
                                "K=\(width), routes=\(values), byteLoads=\(byteLoads), layout=\(layout), distributed=\(distributed)")
                        }
                    }
                }
            }
        }
    }

    func testVerificationKernelUsesCurrentParametersAndPrefillFallback() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        let weight = sin(MLXArray(0..<(128 * 16 * 128)).asType(.float32) * 0.031).reshaped(128, 16, 128).asType(.bfloat16)
        let projection = Gemma4SwitchLinear(packing: weight, groupSize: 64)
        let x = cos(MLXArray(0..<(128 * 128)).asType(.float32) * 0.029).reshaped(128, 1, 128).asType(.bfloat16)
        let indices = MLXArray((0..<128).map { Int32($0 / 32) })
        let reference = projection.applyFlat(x, indices: indices, sortedIndices: true)
        eval(reference)
        let prefill = concatenated([x, x, x, x], axis: 0)
        let prefillRoutes = MLXArray((0..<512).map { Int32($0 / 4) })
        let prefillReference = projection.applyFlat(prefill, indices: prefillRoutes, sortedIndices: true)
        eval(prefillReference)
        projection.metalVerification = try Gemma4MetalExperts()
        let implementation = try XCTUnwrap(projection.metalVerification)
        implementation.distributedReduction = false
        let compiledIdentity = implementation.compilationIdentity
        implementation.optimizedLayout = false
        XCTAssertFalse(compiledIdentity === implementation.compilationIdentity,
            "A compiled MoE graph must retrace after a kernel specialization changes.")
        let referenceIdentity = implementation.compilationIdentity
        implementation.optimizedLayout = false
        XCTAssertTrue(referenceIdentity === implementation.compilationIdentity)
        implementation.byteLoads = false
        XCTAssertFalse(compiledIdentity === implementation.compilationIdentity,
            "Changing both options before a compiled call must select a distinct specialization.")
        implementation.byteLoads = true
        implementation.optimizedLayout = true
        XCTAssertTrue(compiledIdentity === implementation.compilationIdentity)
        implementation.distributedReduction = true
        XCTAssertFalse(compiledIdentity === implementation.compilationIdentity)
        let actual = projection.applyFlat(x, indices: indices, sortedIndices: true)
        XCTAssertEqual(abs(actual.asType(.float32) - reference.asType(.float32)).max().item(Float.self), 0)
        let prefillActual = projection.applyFlat(prefill, indices: prefillRoutes, sortedIndices: true)
        XCTAssertEqual(abs(prefillActual.asType(.float32) - prefillReference.asType(.float32)).max().item(Float.self), 0)
        try projection.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXArray.zeros(projection.weight.shape, dtype: .uint32)),
            ("scales", MLXArray.zeros(try XCTUnwrap(projection.scales).shape, dtype: .bfloat16)),
            ("biases", MLXArray.zeros(try XCTUnwrap(projection.biases).shape, dtype: .bfloat16))
        ]), verify: .all)
        XCTAssertEqual(abs(projection.applyFlat(x, indices: indices, sortedIndices: true)).max().item(Float.self), 0)
        // Large route batches retain the existing matrix-kernel path.
        let changedPrefill = concatenated([x, x], axis: 0)
        let routes = concatenated([indices, indices + 4], axis: 0)
        XCTAssertEqual(projection.applyFlat(changedPrefill, indices: routes, sortedIndices: true).shape, [256, 1, 16])
    }

    func testDenseRoutingRetainsMatrixArithmetic() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        let weight = sin(MLXArray(0..<(4 * 16 * 128)).asType(.float32) * 0.031).reshaped(4, 16, 128).asType(.bfloat16)
        let projection = Gemma4SwitchLinear(packing: weight, groupSize: 64)
        let x = cos(MLXArray(0..<(128 * 128)).asType(.float32) * 0.029).reshaped(128, 1, 128).asType(.bfloat16)
        let indices = MLXArray((0..<128).map { Int32($0 / 32) })
        let reference = projection.applyFlat(x, indices: indices, sortedIndices: true)
        eval(reference)
        projection.metalVerification = try Gemma4MetalExperts()
        let actual = projection.applyFlat(x, indices: indices, sortedIndices: true)
        XCTAssertEqual(abs(actual.asType(.float32) - reference.asType(.float32)).max().item(Float.self), 0)
    }
}
