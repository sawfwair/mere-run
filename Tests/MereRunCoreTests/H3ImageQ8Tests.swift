import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class H3ImageQ8Tests: MLXTestCase {
    func testAttentionDefaultsAreScopedToQualifiedQ8Architecture() {
        let qualified = MiniMaxH3ImageGenerator.defaultAttentionSchedule(nativeQ8: true, architecture: "applegpu_g17s")
        XCTAssertEqual(qualified.heads, 28)
        XCTAssertEqual(qualified.queries, 1_024)
        for (native, architecture) in [(false, "applegpu_g17s"), (true, "applegpu_g16s"), (true, "Unknown")] {
            let schedule = MiniMaxH3ImageGenerator.defaultAttentionSchedule(nativeQ8: native, architecture: architecture)
            XCTAssertEqual(schedule.heads, 8)
            XCTAssertEqual(schedule.queries, 640)
        }
    }

    private final class Projections: Module {
        @ModuleInfo var projection: Linear
        @ModuleInfo var retained: Linear
        init(weight: MLXArray) {
            _projection.wrappedValue = Linear(weight: weight)
            _retained.wrappedValue = Linear(weight: MLXArray.ones([8, 64]))
        }
    }

    func testQ8PackingLoadingAndFP32Retention() throws {
        let weight = MLX.sin(MLXArray(0..<(96 * 64)).asType(.float32) * 0.019).reshaped(96, 64).asType(.bfloat16)
        let model = Projections(weight: weight)
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let packed = MiniMaxH3ImageOptimizer.quantizeArrays(arrays,
            linearKeys: ["projection.weight", "retained.weight"])
        XCTAssertEqual(packed["projection.weight"]?.dtype, .uint32)
        XCTAssertEqual(packed["projection.weight"]?.shape, [96, 16])
        XCTAssertEqual(packed["retained.weight"]?.dtype, .float32)
        XCTAssertNil(packed["retained.scales"])
        try MiniMaxH3ImageMLXFormat.apply(packed, to: model)
        XCTAssertTrue(model.projection is QuantizedLinear)
        let codes = try XCTUnwrap(packed["projection.weight"])
        let unpacked = MLX.bitwiseAnd(codes.expandedDimensions(axis: -1) >> MLXArray([UInt32(0), 8, 16, 24]), 255)
            .reshaped(96, 64).asType(.float32)
        let dense = unpacked * (try XCTUnwrap(packed["projection.scales"])).asType(.float32)
            + (try XCTUnwrap(packed["projection.biases"])).asType(.float32)
        // Reconstruct the affine integers directly in FP32, without rounding
        // dequantized individual weights through BF16 before the reference dot product.
        let input = MLXArray.ones([1, 3, 64], dtype: .float32)
        let error = MLX.max(MLX.abs(model.projection(input).asType(.float32) - MLX.matmul(input, dense.T).asType(.float32))).item(Float.self)
        XCTAssertLessThan(error, 1e-4)
        XCTAssertLessThan(MLX.max(MLX.abs(dense - weight.asType(.float32))).item(Float.self), 0.02)
        var invalid = packed
        invalid["projection.scales"] = MLXArray.zeros([96, 2], dtype: .bfloat16)
        XCTAssertThrowsError(try MiniMaxH3ImageMLXFormat.apply(invalid, to: Projections(weight: weight)))
    }

    func testQ8PreflightRequiresDeclaredGroupSize() {
        var manifest = MereRunModelManifest.template(for: .miniMaxH3Image)
        manifest.precision = .int8
        manifest.quantization = .init(bits: 8, groupSize: 64, scheme: MiniMaxH3ImageOptimizer.format)
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: URL(fileURLWithPath: "/tmp/q8.png"))
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        manifest.quantization?.groupSize = 128
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "model_quantization_unsupported" })
    }

    func testVisionProjectionWithUnalignedInputWidthRetainsBF16() throws {
        let weight = MLXArray.ones([16, 4_304], dtype: .bfloat16)
        let packed = MiniMaxH3ImageOptimizer.quantizeArrays(["vision.weight": weight], linearKeys: ["vision.weight"])
        XCTAssertEqual(try XCTUnwrap(packed["vision.weight"]).dtype, .bfloat16)
        XCTAssertNil(packed["vision.scales"])
    }

    func testShardedQ8UpdatesPreserveOtherArrayLayers() throws {
        final class Layers: Module {
            @ModuleInfo var layers: [Linear]
            override init() {
                _layers.wrappedValue = (0..<3).map { _ in Linear(weight: MLXArray.ones([64, 64], dtype: .bfloat16)) }
            }
        }
        let model = Layers()
        for layer in [2, 0, 1] {
            let key = "layers.\(layer).weight"
            let packed = MiniMaxH3ImageOptimizer.quantizeArrays([key: MLXArray.ones([64, 64], dtype: .bfloat16)], linearKeys: [key])
            try MiniMaxH3ImageMLXFormat.apply(packed, to: model)
            XCTAssertTrue(model.layers[layer] is QuantizedLinear)
            XCTAssertEqual(model.layers.count, 3)
        }
        XCTAssertTrue(model.layers.allSatisfy { $0 is QuantizedLinear })
    }
}
