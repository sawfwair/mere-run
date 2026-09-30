import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class FourDAnyoneProductionSmokeTests: MereRunCoreTestCase {
    func testReleasedResolutionForwardWhenExplicitlyEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_4DANYONE_PRODUCTION_SMOKE"] == "1",
              let path = environment["MERERUN_4DANYONE_VALIDATION_ROOT"] else {
            throw XCTSkip("Enable MERERUN_4DANYONE_PRODUCTION_SMOKE with the trained validation assets.")
        }
        XCTAssertEqual(environment["MERERUN_TEST_MLX_DEVICE"], "gpu")
        let root = URL(fileURLWithPath: path)
        let checkpoint = root.appendingPathComponent("assets/model.safetensors")
        Memory.peakMemory = 0
        let started = Date()
        let nullPose = try fullNullPose(checkpoint: checkpoint)
        let poseSeconds = Date().timeIntervalSince(started)
        try MLX.save(array: nullPose, url: root.appendingPathComponent("production-null-pose-float32-compute.npy"))
        Memory.clearCache()
        let modelStarted = Date()
        let model = try FourDAnyoneModelLoader.loadTransformer(from: checkpoint)
        XCTAssertEqual(model.computePrecision, .float32)
        XCTAssertTrue(model.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        let loadSeconds = Date().timeIntervalSince(modelStarted)
        let trace = try MLX.loadArrays(url: root.appendingPathComponent("reference/transformer-reference.safetensors"))
        let context = try FourDAnyoneModelLoader.loadPromptContext(
            from: root.appendingPathComponent("assets/prompt_context.safetensors")
        )
        // Tiled diagnostic source latents and null target poses exercise the
        // production graph. This is not canonical conditioning or a quality run.
        var source = trace["sources"]![0..<1].asType(.bfloat16)
        for (axis, count) in [(2, 31), (3, 80), (4, 44)] {
            let indices = MLXArray((0..<count).map { Int32($0 % source.dim(axis)) })
            source = MLX.take(source, indices, axis: axis)
        }
        let noise = MLXRandom.normal([4, 48, 31, 80, 44], key: MLXRandom.key(4162)).asType(.bfloat16)
        let input = FourDAnyoneTransformerInput(
            latents: noise, sources: source, poseFeatures: MLX.repeated(nullPose, count: 4, axis: 0),
            nullPoseFeatures: nullPose, promptContext: context, timestep: 1_000
        )
        var blockChecks = 0
        let forwardStarted = Date()
        let output = try model(input, checkCancellation: {
            try Task.checkCancellation()
            let elapsed = Date().timeIntervalSince(forwardStarted)
            guard elapsed < 900 else { throw SmokeError.timeLimit }
            let completedBlocks = max(blockChecks - 1, 0)
            print("4DAnyone production check \(completedBlocks)/30 elapsed=\(elapsed)s")
            let progress = Progress(completedBlocks: completedBlocks, seconds: elapsed, peakBytes: Memory.peakMemory)
            try JSONEncoder().encode(progress)
                .write(to: root.appendingPathComponent("production-float32-compute-progress.json"))
            blockChecks += 1
        })
        eval(output)
        let forwardSeconds = Date().timeIntervalSince(forwardStarted)
        try JSONEncoder().encode(Progress(completedBlocks: 30, seconds: forwardSeconds, peakBytes: Memory.peakMemory))
            .write(to: root.appendingPathComponent("production-float32-compute-progress.json"))
        XCTAssertEqual(output.shape, [4, 48, 31, 80, 44])
        XCTAssertEqual(output.dtype, .float32)
        XCTAssertTrue(model.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        XCTAssertTrue(MLX.all(MLX.isFinite(output)).item(Bool.self))
        let deviation = MLX.std(output.asType(.float32)).item(Float.self)
        XCTAssertGreaterThan(deviation, 0)
        try MLX.save(array: output, url: root.appendingPathComponent("production-prediction-float32-compute.npy"))
        let receipt = Receipt(
            scope: "One trained FP32-compute forward with BF16 storage and diagnostic conditioning; no video-quality claim.",
            poseSeconds: poseSeconds, loadSeconds: loadSeconds, forwardSeconds: forwardSeconds,
            peakAllocatedBytes: Memory.peakMemory, outputShape: output.shape, outputStandardDeviation: deviation,
            computePrecision: model.computePrecision.rawValue, storageDtype: "bfloat16"
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(to: root.appendingPathComponent("production-float32-compute.json"))
    }

    private func fullNullPose(checkpoint: URL) throws -> MLXArray {
        let encoder = try FourDAnyoneModelLoader.loadPoseEncoder(from: checkpoint)
        let video = MLX.full([1, 3, 121, 1_280, 704], values: MLXArray(Float(-1)), dtype: .bfloat16)
        let encoded = try encoder(video)
        eval(encoded)
        XCTAssertEqual(encoded.shape, [1, 3_072, 31, 40, 22])
        XCTAssertTrue(MLX.all(MLX.isFinite(encoded)).item(Bool.self))
        return encoded
    }

    private enum SmokeError: Error { case timeLimit }

    private struct Progress: Codable {
        let completedBlocks: Int
        let seconds: Double
        let peakBytes: Int
    }

    private struct Receipt: Codable {
        let scope: String
        let poseSeconds: Double
        let loadSeconds: Double
        let forwardSeconds: Double
        let peakAllocatedBytes: Int
        let outputShape: [Int]
        let outputStandardDeviation: Float
        let computePrecision: String
        let storageDtype: String
    }
}
