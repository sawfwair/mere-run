import Foundation
import MLX
import MLXFast
import MLXNN
import XCTest
@testable import MereRunCore

/// Operation-level diagnostics use frozen inputs so upstream drift cannot accumulate.
final class FourDAnyoneOperationTests: MereRunCoreTestCase {
    func testBF16ActivationsMatchIndependentPrimitiveReference() throws {
        let tensors = try MLX.loadArrays(url: FourDAnyoneFixture.url("activations", extension: "safetensors"))
        FourDAnyoneFixture.assertClose(
            FourDAnyoneActivation.gelu(tensors["input"]!), tensors["gelu"]!, tolerance: 2e-5
        )
        FourDAnyoneFixture.assertClose(
            FourDAnyoneActivation.silu(tensors["input"]!), tensors["silu"]!, tolerance: 2e-5
        )
    }

    func testFrozenBF16OperationsWhenExplicitlyEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["MERERUN_4DANYONE_OPERATIONS_ROOT"] else {
            throw XCTSkip("Set MERERUN_4DANYONE_OPERATIONS_ROOT to the frozen operation cases and reference.")
        }
        let root = URL(fileURLWithPath: path)
        let cases = try MLX.loadArrays(url: root.appendingPathComponent("operations-cases.safetensors"))
        let device = environment["MERERUN_4DANYONE_REFERENCE_DEVICE"] ?? "cpu"
        let expected = try MLX.loadArrays(
            url: root.appendingPathComponent("operations-reference-\(device).safetensors")
        )
        let activation = cases["activation.input"]!
        let linear = Linear(3_072, 3_072)
        try linear.update(parameters: ModuleParameters.unflattened([
            "weight": cases["linear.weight"]!, "bias": cases["linear.bias"]!,
        ]), verify: .noUnusedKeys)
        let rms = FourDAnyoneRMSNorm(3_072, epsilon: 1e-6)
        try rms.update(parameters: ModuleParameters.unflattened([
            "weight": cases["rms.weight"]!,
        ]), verify: .noUnusedKeys)
        let normInput = cases["norm.input"]!
        let outputs = [
            "gelu.native": MLXNN.geluApproximate(activation),
            "gelu.promoted": FourDAnyoneActivation.gelu(activation),
            "silu.native": MLXNN.silu(activation),
            "silu.promoted": FourDAnyoneActivation.silu(activation),
            "linear.native": linear(cases["linear.input"]!),
            "rms.native": rms(normInput),
            "layer.bf16.native": MLXFast.layerNorm(
                normInput, weight: cases["layer.weight"]!, bias: cases["layer.bias"]!, eps: 1e-6
            ),
            "layer.fp32.native": MLXFast.layerNorm(
                normInput.asType(.float32), weight: cases["layer.weight"]!.asType(.float32),
                bias: cases["layer.bias"]!.asType(.float32), eps: 1e-6
            ),
            "attention.native": MLXFast.scaledDotProductAttention(
                queries: cases["attention.query"]!, keys: cases["attention.key"]!,
                values: cases["attention.value"]!, scale: 1 / Float(128).squareRoot(), mask: .none
            ),
        ]
        var metrics: [Metric] = []
        for name in outputs.keys.sorted() {
            let reference = name.replacingOccurrences(of: ".native", with: "")
                .replacingOccurrences(of: ".promoted", with: "")
            let a = outputs[name]!.asType(.float32)
            let b = expected[reference]!.asType(.float32)
            XCTAssertEqual(a.shape, b.shape)
            XCTAssertTrue(MLX.all(MLX.isFinite(a)).item(Bool.self))
            let difference = a - b
            let relative = MLX.sqrt(MLX.mean(difference * difference) / MLX.mean(b * b)).item(Float.self)
            metrics.append(Metric(
                name: name, normalizedRMSE: relative, maximumError: MLX.max(MLX.abs(difference)).item(Float.self),
                exactFraction: MLX.mean((a .== b).asType(.float32)).item(Float.self)
            ))
        }
        try MLX.save(arrays: outputs, url: root.appendingPathComponent("native-operations.safetensors"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metrics).write(to: root.appendingPathComponent("native-operations-vs-\(device).json"))
    }

    private struct Metric: Codable {
        let name: String
        let normalizedRMSE: Float
        let maximumError: Float
        let exactFraction: Float
    }
}
