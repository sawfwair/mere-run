import Foundation
import MereRunAdmission
import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Bonsai2PerformanceTests: MereRunCoreTestCase {
    func testInstalledProjectionProfile() async throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_BONSAI2_PROFILE"] else {
            throw XCTSkip("Set MERERUN_BONSAI2_PROFILE to a checkpoint for the opt-in GPU profile")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The projection profile requires MERERUN_TEST_MLX_DEVICE=gpu")
        }
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(label: "Bonsai packed projection profile", resourceClass: .standard))
        defer { lease.release() }
        let config = try JSONDecoder().decode(Q35PrismConfiguration.self,
            from: Data(contentsOf: URL(fileURLWithPath: path).appendingPathComponent("config.json")))
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path).appendingPathComponent("model.safetensors"))
        let head = config.modules.contains { $0.path == "lm_head" } ? "lm_head" : "model.embed_tokens"
        for name in ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj", head] {
            let block = try XCTUnwrap(config.modules.first { $0.path == name }).block
            let prefix = "language_model." + name
            let weight = try XCTUnwrap(arrays[prefix + ".weight"])
            let scales = try XCTUnwrap(arrays[prefix + ".scales"])
            let biases = try XCTUnwrap(arrays[prefix + ".biases"])
            let signs = try XCTUnwrap(arrays[prefix + ".signs"])
            let width = weight.dim(1) * 16
            let input = MLXArray((0..<width).map { Float($0 % 19 - 9) / 16 })
                .reshaped(1, 1, width).asType(.float16)
            let rotated = Q35PrismTransform.apply(input, block: block, signs: signs)
            var low = MLXArray.zeros(weight.shape, dtype: .uint32)
            var high = MLXArray.zeros(weight.shape, dtype: .uint32)
            for index in 0..<8 {
                low = low | (((weight >> (index * 2)) & 3) << (index * 4))
                high = high | (((weight >> ((index + 8) * 2)) & 3) << (index * 4))
            }
            let expanded = MLX.stacked([low, high], axis: -1).reshaped(weight.dim(0), -1)
            MLX.eval(weight, scales, biases, rotated, expanded)
            let packedOp = { MLX.quantizedMM(rotated, weight, scales: scales, biases: biases,
                                            transpose: true, groupSize: 128, bits: 2) }
            let expandedOp = { MLX.quantizedMM(rotated, expanded, scales: scales, biases: biases,
                                              transpose: true, groupSize: 128, bits: 4) }
            let difference = MLX.max(MLX.abs(packedOp() - expandedOp())).item(Float.self)
            print("BONSAI_PROFILE \(name) expansion_max_error=\(difference)")
            for trial in 0..<3 {
                let operations: [(String, () -> MLXArray)] = [
                    ("reference_transform", { Q35PrismTransform.reference(input, block: block, signs: signs) }),
                    ("transform", { Q35PrismTransform.apply(input, block: block, signs: signs) }),
                    ("packed2", packedOp), ("expanded4", expandedOp),
                ]
                for (label, operation) in trial.isMultiple(of: 2) ? operations : operations.reversed() {
                    for _ in 0..<5 { MLX.eval(operation()) }
                    let start = Date()
                    for _ in 0..<50 { MLX.eval(operation()) }
                    print("BONSAI_PROFILE \(name) \(label) trial=\(trial) ms=\(Date().timeIntervalSince(start) * 20)")
                }
            }
        }
    }
}
