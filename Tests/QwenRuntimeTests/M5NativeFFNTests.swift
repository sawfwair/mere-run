import Foundation
import MLX
import MLXFast
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor
#if os(macOS)
import Metal

final class M5NativeFFNTests: MLXTestCase {
    func testResidualNormalizationFeedsFFNInOneEncoder() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try requireM5()
        MLXRandom.seed(934)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
            let norm = try M5NativeRMSNorm(device: device), program = try M5NativeFFN(device: device)
            let fixture = try Fixture(device: device), workspace = try M5NativeFFN.Workspace(device: device)
            let scale = MLXRandom.uniform(0.5..<1.5, [5_120]).asType(.bfloat16)
            let scaleBuffer = try XCTUnwrap(scale.asMTLBuffer(device: device))
            let normalized = try XCTUnwrap(device.makeBuffer(length: 16 * 5_120 * 2, options: .storageModePrivate))
            let output = try XCTUnwrap(device.makeBuffer(length: 16 * 5_120 * 2, options: .storageModeShared))
            for rows in 1...16 {
                let input = MLXRandom.uniform(-1..<1, [1, rows, 5_120]).asType(.bfloat16)
                let residual = MLXRandom.uniform(-1..<1, [1, rows, 5_120]).asType(.bfloat16)
                let expectedCombined = input + residual
                let expected = try fixture.reference(MLXFast.rmsNorm(expectedCombined, weight: scale, eps: 1e-6))
                MLX.eval(expected, expectedCombined)
                let inputBuffer = try XCTUnwrap(input.asMTLBuffer(device: device))
                let residualBuffer = try XCTUnwrap(residual.asMTLBuffer(device: device))
                let command = try XCTUnwrap(queue.makeCommandBuffer())
                let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                norm.encode(into: encoder, input: inputBuffer, residual: residualBuffer, weight: scaleBuffer,
                    combined: inputBuffer, normalized: normalized, sums: workspace.inputSums, rows: rows, epsilon: 1e-6)
                program.encode(into: encoder, input: normalized, output: output, weights: fixture.native,
                    workspace: workspace, rows: rows, inputSums: .provided)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertNil(command.error)
                XCTAssertEqual(delta(read(inputBuffer, rows: rows), expectedCombined), 0)
                XCTAssertEqual(delta(read(output, rows: rows), expected), 0, "rows=\(rows)")
            }
        }
    }

    func testEveryVerificationWidthMatchesCompleteMLXFFN() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try requireM5()
        MLXRandom.seed(930)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
            let program = try M5NativeFFN(device: device), fixture = try Fixture(device: device)
            let workspace = try M5NativeFFN.Workspace(device: device)
            for rows in 1...16 {
                let x = MLXRandom.uniform(-1..<1, [1, rows, 5_120]).asType(.bfloat16)
                let expected = try fixture.reference(x); MLX.eval(expected)
                let input = try XCTUnwrap(x.asMTLBuffer(device: device))
                let output = try XCTUnwrap(device.makeBuffer(length: 16 * 5_120 * 2, options: .storageModeShared))
                for split in [false, true] {
                    _ = try submit(program, queue: queue, input: input, output: output, fixture: fixture,
                        workspace: workspace, rows: rows, split: split)
                    XCTAssertEqual(delta(read(output, rows: rows), expected), 0, "rows=\(rows), split=\(split)")
                }
            }
        }
    }

    func testConcurrentScratchReusePreservesSeparateLiveOutputs() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try requireM5()
        MLXRandom.seed(931)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
            let program = try M5NativeFFN(device: device), fixture = try Fixture(device: device)
            let workspace = try M5NativeFFN.Workspace(device: device, untracked: true)
            let inputs = [8, 16].map { MLXRandom.uniform(-1..<1, [1, $0, 5_120]).asType(.bfloat16) }
            let expected = try inputs.map { try fixture.reference($0) }; MLX.eval(expected)
            let buffers = try inputs.map { try XCTUnwrap($0.asMTLBuffer(device: device)) }
            let outputs = try inputs.map { _ in
                try XCTUnwrap(device.makeBuffer(length: 16 * 5_120 * 2, options: .storageModeShared))
            }
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            let hazards = M5NativeHazards()
            for index in inputs.indices {
                program.encode(into: encoder, input: buffers[index], output: outputs[index], weights: fixture.native,
                    workspace: workspace, rows: inputs[index].dim(1), hazards: hazards)
            }
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertNil(command.error)
            for index in inputs.indices { XCTAssertEqual(delta(read(outputs[index], rows: inputs[index].dim(1)), expected[index]), 0) }
        }
    }

    func testDistinctFFNSubmissionCosts() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try requireM5()
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_NATIVE_FFN"] == "1", "Opt-in native FFN benchmark.")
        let url = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_NATIVE_FFN_OUTPUT"]))
        try Data("[]".utf8).write(to: url, options: .withoutOverwriting)
        MLXRandom.seed(932)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
            let program = try M5NativeFFN(device: device), workspace = try M5NativeFFN.Workspace(device: device)
            let fixtures = try (0..<8).map { _ in try Fixture(device: device) }
            var receipt: [Sample] = []
            for rows in env["MERERUN_TEST_NATIVE_FFN_REVERSE"] == "1" ? [16, 8] : [8, 16] {
                let inputs = fixtures.map { _ in MLXRandom.uniform(-1..<1, [1, rows, 5_120]).asType(.bfloat16) }
                MLX.eval(inputs)
                let buffers = try inputs.map { try XCTUnwrap($0.asMTLBuffer(device: device)) }
                let outputs = try inputs.map { _ in
                    try XCTUnwrap(device.makeBuffer(length: 16 * 5_120 * 2, options: .storageModeShared))
                }
                let expected = try zip(fixtures, inputs).map { try $0.reference($1) }; MLX.eval(expected)
                var maximum: Float = 0, wall: [String: [Double]] = [:], gpu: [String: [Double]] = [:]
                let modes = ["mlx", "native-chain", "native-one", "native-five"]
                for trial in 0..<16 {
                    let offset = trial % modes.count
                    let order = Array(modes[offset...]) + Array(modes[..<offset])
                    for mode in order {
                        let start = DispatchTime.now().uptimeNanoseconds
                        var gpuSeconds = 0.0
                        if mode == "mlx" {
                            let result = try zip(fixtures, inputs).map { try $0.reference($1) }; MLX.eval(result)
                        } else {
                            gpuSeconds = try submitMany(program, queue: queue, inputs: buffers, outputs: outputs,
                                fixtures: fixtures, workspace: workspace, rows: rows, mode: mode)
                        }
                        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(fixtures.count)
                        if trial >= 4 {
                            wall[mode, default: []].append(milliseconds)
                            if mode != "mlx" { gpu[mode, default: []].append(gpuSeconds * 1_000 / Double(fixtures.count)) }
                        }
                        if mode != "mlx", trial < 4 {
                            for index in fixtures.indices { maximum = max(maximum, delta(read(outputs[index], rows: rows), expected[index])) }
                        }
                    }
                }
                for index in fixtures.indices { maximum = max(maximum, delta(read(outputs[index], rows: rows), expected[index])) }
                XCTAssertEqual(maximum, 0)
                receipt.append(Sample(rows: rows, distinctWeights: fixtures.count, maximumDifference: maximum,
                    wallMillisecondsPerFFN: wall, nativeCommandBufferSpanMillisecondsPerFFN: gpu))
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(receipt).write(to: url, options: .atomic)
            }
        }
    }

    private struct Sample: Encodable {
        let rows: Int
        let distinctWeights: Int
        let maximumDifference: Float
        let wallMillisecondsPerFFN: [String: [Double]]
        let nativeCommandBufferSpanMillisecondsPerFFN: [String: [Double]]
    }

    @available(macOS 26.0, *)
    private struct Fixture {
        let gate: M5TiledAffineQ4.Weights
        let up: M5TiledAffineQ4.Weights
        let down: M5TiledAffineQ4.Weights
        let native: M5NativeFFN.Weights

        init(device: any MTLDevice) throws {
            func weight(_ k: Int, _ n: Int) throws -> M5TiledAffineQ4.Weights {
                try XCTUnwrap(M5TiledAffineQ4.Weights(weight: MLXRandom.randInt(0..<UInt32.max, [n, k / 8]),
                    scales: MLXRandom.uniform(0.001..<0.01, [n, k / 64]).asType(.bfloat16),
                    biases: MLXRandom.uniform(-0.1..<0.1, [n, k / 64]).asType(.bfloat16)))
            }
            gate = try weight(5_120, 17_408); up = try weight(5_120, 17_408); down = try weight(17_408, 5_120)
            native = try M5NativeFFN.Weights(gate: gate, up: up, down: down, device: device)
        }

        func reference(_ x: MLXArray) throws -> MLXArray {
            let activated = try XCTUnwrap(M5TiledAffineQ4.swiglu(x, gate: gate, up: up))
            return try XCTUnwrap(M5TiledAffineQ4.matmul(activated, weights: down))
        }
    }

    @available(macOS 26.0, *)
    private func submitMany(_ program: M5NativeFFN, queue: any MTLCommandQueue,
                            inputs: [any MTLBuffer], outputs: [any MTLBuffer], fixtures: [Fixture],
                            workspace: M5NativeFFN.Workspace, rows: Int, mode: String) throws -> Double {
        var commands: [any MTLCommandBuffer] = []
        if mode == "native-chain" {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            for index in fixtures.indices {
                program.encode(into: encoder, input: inputs[index], output: outputs[index],
                    weights: fixtures[index].native, workspace: workspace, rows: rows)
            }
            encoder.endEncoding(); command.commit(); commands.append(command)
        } else {
            for index in fixtures.indices {
                for stage in mode == "native-five" ? M5NativeFFN.Stage.allCases.map(Optional.some) : [nil] {
                    let command = try XCTUnwrap(queue.makeCommandBuffer())
                    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                    if let stage {
                        program.encode(stage, into: encoder, input: inputs[index], output: outputs[index],
                            weights: fixtures[index].native, workspace: workspace, rows: rows)
                    } else {
                        program.encode(into: encoder, input: inputs[index], output: outputs[index],
                            weights: fixtures[index].native, workspace: workspace, rows: rows)
                    }
                    encoder.endEncoding(); command.commit(); commands.append(command)
                }
            }
        }
        commands.last!.waitUntilCompleted()
        for command in commands { if let error = command.error { throw error } }
        return commands.reduce(0) { $0 + $1.gpuEndTime - $1.gpuStartTime }
    }

    @available(macOS 26.0, *)
    private func submit(_ program: M5NativeFFN, queue: any MTLCommandQueue, input: any MTLBuffer,
                        output: any MTLBuffer, fixture: Fixture, workspace: M5NativeFFN.Workspace,
                        rows: Int, split: Bool) throws -> Double {
        var commands: [any MTLCommandBuffer] = []
        for stage in split ? M5NativeFFN.Stage.allCases.map(Optional.some) : [nil] {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            if let stage {
                program.encode(stage, into: encoder, input: input, output: output,
                    weights: fixture.native, workspace: workspace, rows: rows)
            } else {
                program.encode(into: encoder, input: input, output: output,
                    weights: fixture.native, workspace: workspace, rows: rows)
            }
            encoder.endEncoding(); command.commit(); commands.append(command)
        }
        commands.last!.waitUntilCompleted()
        for command in commands { if let error = command.error { throw error } }
        return commands.reduce(0) { $0 + $1.gpuEndTime - $1.gpuStartTime }
    }

    private func read(_ buffer: any MTLBuffer, rows: Int) -> MLXArray {
        MLXArray(rawPointer: buffer.contents(), [1, rows, 5_120], dtype: .bfloat16) { _ = buffer }
    }

    private func delta(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        (lhs.asType(.float32) - rhs.asType(.float32)).abs().max().item(Float.self)
    }

    private func requireM5() throws {
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "M5 Metal tensor support required.")
        Device.setDefault(device: .gpu)
    }
}
#endif
