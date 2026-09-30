import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

/// A bounded, resident first-block profile; it does not run a generation trajectory.
final class FourDAnyoneProfilingTests: MereRunCoreTestCase {
    func testProductionBlockProfileWhenExplicitlyEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_4DANYONE_PROFILE"] == "1",
              let path = environment["MERERUN_4DANYONE_VALIDATION_ROOT"] else {
            throw XCTSkip("Enable MERERUN_4DANYONE_PROFILE with the trained validation root.")
        }
        XCTAssertEqual(environment["MERERUN_TEST_MLX_DEVICE"], "gpu")
        let root = URL(fileURLWithPath: path)
        let model = try FourDAnyoneModelLoader.loadTransformer(
            from: root.appendingPathComponent("assets/model.safetensors"), computePrecision: .model
        )
        let nullPose = try MLX.loadArray(url: root.appendingPathComponent("production-null-pose.npy"))
        let grid = Wan2GridSize(frames: 31, height: 40, width: 22)
        let context = model.textEmbedding(try FourDAnyoneModelLoader.loadPromptContext(
            from: root.appendingPathComponent("assets/prompt_context.safetensors")
        ))
        let time = model.embeddedTime(timestep: 1_000, targets: 4, packed: 1)
        let modulation = model.timeProjection(time).reshaped(5, 6, 3_072)
        let latents = MLXRandom.normal([5, 48, 31, 80, 44], key: MLXRandom.key(4196)).asType(.bfloat16)
        let hidden = model.patchify(latents) + MLX.repeated(nullPose, count: 5, axis: 0)
            .transposed(0, 2, 3, 4, 1).reshaped(5, grid.sequenceLength, 3_072)
        let spatial = FourDAnyoneRoPE.prepare(grid: grid, headDimension: 128)
        let multiview = FourDAnyoneRoPE.prepare(
            grid: Wan2GridSize(frames: 5, height: 40, width: 22), headDimension: 128
        )
        eval(hidden, context, modulation)
        var baseline: MLXArray?
        var runs: [Run] = []
        for chunk in [512, 1_024] {
            for repetition in 0..<2 {
                Memory.peakMemory = 0
                let started = Date()
                var previous = started
                var stages: [Stage] = []
                let output = model.blocks[0](
                    hidden, context: context, time: modulation, grid: grid,
                    spatialRoPE: spatial, multiviewRoPE: multiview, maximumQueryTokens: chunk,
                    observe: { name, value in
                        eval(value)
                        let now = Date()
                        stages.append(Stage(name: name, seconds: now.timeIntervalSince(previous)))
                        previous = now
                    }
                )
                eval(output)
                let elapsed = Date().timeIntervalSince(started)
                XCTAssertTrue(MLX.all(MLX.isFinite(output)).item(Bool.self))
                let expected = baseline ?? output
                let difference = output.asType(.float32) - expected.asType(.float32)
                let error = MLX.sqrt(MLX.mean(difference * difference)
                    / MLX.mean(expected.asType(.float32) * expected.asType(.float32))).item(Float.self)
                XCTAssertLessThanOrEqual(error, 1e-3, "Query chunk changed the block result")
                if baseline == nil { baseline = output }
                runs.append(Run(
                    queryChunk: chunk, repetition: repetition, seconds: elapsed,
                    peakAllocatedBytes: Memory.peakMemory, normalizedError: error, stages: stages
                ))
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(runs).write(to: root.appendingPathComponent("production-block-profile.json"))
            }
        }
    }

    private struct Stage: Codable {
        let name: String
        let seconds: Double
    }

    private struct Run: Codable {
        let queryChunk: Int
        let repetition: Int
        let seconds: Double
        let peakAllocatedBytes: Int
        let normalizedError: Float
        let stages: [Stage]
    }
}
