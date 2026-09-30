import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Bounded precision experiments retain the frozen inputs and original error limit.
final class FourDAnyonePrecisionTests: MereRunCoreTestCase {
    func testConditioningPrecisionWhenExplicitlyEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_4DANYONE_PRECISION_DIAGNOSTICS"] == "1",
              let path = environment["MERERUN_4DANYONE_VALIDATION_ROOT"] else {
            throw XCTSkip("Enable MERERUN_4DANYONE_PRECISION_DIAGNOSTICS with the trained validation root.")
        }
        let root = URL(fileURLWithPath: path)
        let trace = try MLX.loadArrays(url: root.appendingPathComponent("reference/transformer-reference.safetensors"))
        let model = try FourDAnyoneModelLoader.loadTransformer(
            from: root.appendingPathComponent("assets/model.safetensors"), computePrecision: .model
        )
        var metrics: [Metric] = []
        for mode in ["baseline", "float32_time", "float32_conditioning"] {
            if mode == "float32_time" {
                try promote(model.timeEmbedding)
                try promote(model.timeProjection)
            } else if mode == "float32_conditioning" {
                try promote(model.textEmbedding)
                try promote(model.head)
            }
            for packed in [false, true] {
                let name = packed ? "packed" : "direct"
                let input = FourDAnyoneTransformerInput(
                    latents: trace["latents"]!, sources: trace["sources"]![0..<(packed ? 5 : 1)],
                    poseFeatures: trace["pose.features"]!,
                    nullPoseFeatures: MLX.repeated(trace["pose.null"]!, count: packed ? 2 : 1, axis: 0),
                    promptContext: trace["context"]!, timestep: 625
                )
                let started = Date()
                let output = try model(input).asType(.float32)
                eval(output)
                let reference = trace[name + ".bf16_input_control"]!.asType(.float32)
                let difference = output - reference
                let error = MLX.sqrt(MLX.mean(difference * difference) / MLX.mean(reference * reference))
                    .item(Float.self)
                XCTAssertTrue(error.isFinite)
                metrics.append(Metric(
                    mode: mode, source: name, normalizedRMSE: error,
                    seconds: Date().timeIntervalSince(started), passesOriginalLimit: error <= 0.03
                ))
                try MLX.save(array: output, url: root.appendingPathComponent("precision-\(mode)-\(name).npy"))
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(metrics).write(to: root.appendingPathComponent("conditioning-precision.json"))
        }
    }

    private func promote(_ module: Module) throws {
        try module.update(parameters: ModuleParameters.unflattened(
            module.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        ), verify: .noUnusedKeys)
        eval(module.parameters().flattened().map(\.1))
    }

    private struct Metric: Codable {
        let mode: String
        let source: String
        let normalizedRMSE: Float
        let seconds: Double
        let passesOriginalLimit: Bool
    }
}
