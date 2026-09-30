import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class FourDAnyoneTrajectoryTests: MereRunCoreTestCase {
    func testTrainedBaseTrajectoryWhenExplicitlyEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_4DANYONE_TRAJECTORY"] == "1",
              let path = environment["MERERUN_4DANYONE_VALIDATION_ROOT"] else {
            throw XCTSkip("Enable MERERUN_4DANYONE_TRAJECTORY with the trained validation root.")
        }
        let root = URL(fileURLWithPath: path)
        let trace = try MLX.loadArrays(url: root.appendingPathComponent("reference/transformer-reference.safetensors"))
        let reference = try MLX.loadArrays(url: root.appendingPathComponent("trajectory/reference.safetensors"))
        Memory.peakMemory = 0
        let model = try FourDAnyoneModelLoader.loadTransformer(
            from: root.appendingPathComponent("assets/model.safetensors")
        )
        let generator = FourDAnyoneGenerator(transformer: model)
        let started = Date()
        var completedSteps = 0
        let actual = try generator.generate(
            initialLatents: reference["initial"]!,
            conditioning: FourDAnyonePreparedConditioning(
                sources: trace["sources"]![0..<1], poseFeatures: trace["pose.features"]!,
                nullPoseFeatures: trace["pose.null"]!, promptContext: trace["context"]!
            ),
            plan: FourDAnyoneViewPlan(viewsPerLayer: 4),
            progress: { completedSteps = $0.step }
        )
        let expected = reference["step24"]!
        XCTAssertEqual(completedSteps, 24)
        XCTAssertEqual(actual.dtype, .float32)
        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertTrue(model.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        let difference = actual - expected
        let error = MLX.sqrt(MLX.mean(difference * difference) / MLX.mean(expected * expected)).item(Float.self)
        XCTAssertTrue(error.isFinite)
        XCTAssertLessThanOrEqual(error, 1e-4)
        let receipt = Receipt(
            completedSteps: completedSteps, normalizedRMSE: error, seconds: Date().timeIntervalSince(started),
            peakAllocatedBytes: Memory.peakMemory
        )
        try MLX.save(array: actual, url: root.appendingPathComponent("trajectory/native-step24.npy"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(to: root.appendingPathComponent("trajectory/native.json"))
    }

    private struct Receipt: Codable {
        let completedSteps: Int
        let normalizedRMSE: Float
        let seconds: Double
        let peakAllocatedBytes: Int
    }
}
