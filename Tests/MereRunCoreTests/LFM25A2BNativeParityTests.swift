import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunAdmission
import XCTest
@testable import MereRunCore

/// Opt-in numerical diagnostics keep checkpoint weights and shared admission unchanged.
final class LFM25A2BNativeParityTests: MereRunCoreTestCase {
    private let compiledSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { gate, up in
        (gate * MLX.sigmoid(gate)) * up
    }

    private struct Suite: Decodable {
        let cases: [Row]
    }

    private struct Row: Decodable {
        let id: String
        let tokens: [Int]
        let scoreStart: Int

        enum CodingKeys: String, CodingKey {
            case id, tokens
            case scoreStart = "score_start"
        }
    }

    private struct Progress: Encodable {
        let mode: String
        let caseID: String
        let layer: Int
        let elapsedSeconds: TimeInterval
        let peakMLXBytes: Int
    }

    private struct RoutingIntervention: Decodable {
        let caseID: String
        let layer: Int
        let tokenIndex: Int
        let originalExpert: Int32
        let replacementExpert: Int32
    }

    private enum Mode: String {
        case baseline
        case positivePartition = "positive-partition"
        case positiveSorted = "positive-sorted"
        case denseExpertReference = "dense-expert-reference"
        case boundaryCounterfactual = "boundary-counterfactual"
        case fp32Parameters = "fp32-parameters"
        case compiledSwiGLU = "compiled-swiglu"
        case upstreamReference = "upstream-reference"
    }

    func testInstalledLayerwiseNumericalDiagnostics() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_LFM25_A2B_QUALIFICATION_ROOT"],
              let suitePath = env["MERERUN_LFM25_A2B_QUALIFICATION_SUITE"],
              let outputPath = env["MERERUN_LFM25_A2B_PARITY_DIAGNOSTIC_OUTPUT"],
              let modeNames = env["MERERUN_LFM25_A2B_PARITY_DIAGNOSTIC_MODES"] else {
            throw XCTSkip("Provide an installed checkpoint, frozen diagnostic suite, modes, and output folder.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Trained-checkpoint diagnostics require the GPU test device.")
        }
        let modes = try modeNames.split(separator: ",").map { try XCTUnwrap(Mode(rawValue: String($0))) }
        let root = URL(fileURLWithPath: rootPath)
        let output = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let config = try JSONDecoder().decode(LFM2Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        let suite = try JSONDecoder().decode(Suite.self, from: Data(contentsOf: URL(fileURLWithPath: suitePath)))
        XCTAssertTrue(config.isA2B24B)
        XCTAssertTrue(config.useFP32Activations)
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appending(path: "admission"),
            hostSnapshot: {
                MachineInferenceCoordinator.currentHostSnapshot(
                    diskURL: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent()
                )
            }
        )
        let lease = try await coordinator.acquire(.init(label: "LFM2.5 native numerical diagnostics", resourceClass: .standard))
        defer { lease.release() }
        if modes == [.upstreamReference] {
            let pythonPath = try XCTUnwrap(env["MERERUN_LFM25_A2B_REFERENCE_PYTHON"])
            let scriptPath = try XCTUnwrap(env["MERERUN_LFM25_A2B_REFERENCE_SCRIPT"])
            let process = Process()
            process.executableURL = URL(fileURLWithPath: pythonPath)
            process.arguments = [scriptPath, "--model", rootPath, "--suite", suitePath, "--output", outputPath]
            var childEnvironment = env
            childEnvironment["MERERUN_LFM25_A2B_REFERENCE_LEASE_HELD"] = "1"
            process.environment = childEnvironment
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0, "Independent local upstream reference failed.")
            return
        }
        let model = LFM2Model(config: config)
        try HFSafetensorsWeightsLoader.applyQuantizedWeights(
            indexURL: root.appending(path: "model.safetensors.index.json"), to: model,
            groupSize: try XCTUnwrap(config.quantization).groupSize,
            bits: try XCTUnwrap(config.quantization).bits, mapper: LFM2Resources.mapWeight(key:value:)
        )
        let intervention = try env["MERERUN_LFM25_A2B_PARITY_ROUTING_INTERVENTION"].map {
            try JSONDecoder().decode(RoutingIntervention.self, from: Data(contentsOf: URL(fileURLWithPath: $0)))
        }
        let batch = env["MERERUN_LFM25_A2B_PARITY_MATCH_REFERENCE_BATCH"] == "1"
        let started = ProcessInfo.processInfo.systemUptime
        MLX.Memory.peakMemory = 0
        for mode in modes {
            if mode == .fp32Parameters {
                let promoted = model.parameters().flattened().map { name, value in
                    (name, value.dtype == .float16 || value.dtype == .bfloat16 ? value.asType(.float32) : value)
                }
                try model.update(parameters: ModuleParameters.unflattened(promoted), verify: .all)
                MLX.eval(model.parameters().flattened().map { $0.1 })
                MLX.Memory.clearCache()
            }
            for row in batch ? Array(suite.cases.prefix(1)) : suite.cases {
                let folder = output.appending(path: mode.rawValue).appending(path: row.id)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let batchRows = batch ? suite.cases : [row]
                let width = batchRows.map { $0.tokens.count - 1 }.max() ?? 0
                let tokens = batchRows.flatMap { item in
                    item.tokens.dropLast().map(Int32.init) + Array(repeating: Int32(0), count: width - item.tokens.count + 1)
                }
                let ids = MLXArray(tokens, [batchRows.count, width])
                var hidden = model.embeddings(for: ids).asType(.float32)
                for (index, layer) in model.model.layers.enumerated() {
                    let pressure = RuntimeMemoryPressurePolicy.default.pressure(for: .current())
                    guard pressure != .critical else {
                        XCTFail("Numerical diagnostics reached critical memory pressure.")
                        return
                    }
                    let activeIntervention = mode == .boundaryCounterfactual && intervention?.caseID == row.id ? intervention : nil
                    let result = try diagnosticLayer(
                        hidden, layer: layer, index: index, config: config, mode: mode,
                        intervention: activeIntervention
                    )
                    hidden = try XCTUnwrap(result["output"])
                    MLX.eval(Array(result.values))
                    let saved = batch ? result.mapValues { $0[0..<1] } : result
                    try MLX.save(arrays: saved, url: folder.appending(path: "layer-\(index).safetensors"))
                    let progress = Progress(
                        mode: mode.rawValue, caseID: row.id, layer: index,
                        elapsedSeconds: ProcessInfo.processInfo.systemUptime - started,
                        peakMLXBytes: MLX.Memory.peakMemory
                    )
                    try JSONEncoder().encode(progress).write(to: output.appending(path: "progress.json"), options: .atomic)
                    MLX.Memory.clearCache()
                }
                let normalized = model.model.embeddingNorm(hidden)
                let logits = model.logits(from: normalized)[0, (row.scoreStart - 1)..., 0...].asType(.float32)
                MLX.eval(logits)
                try MLX.save(arrays: ["logits": logits], url: folder.appending(path: "logits.safetensors"))
                if mode == .positivePartition {
                    let actual = model.logits(from: model.model(ids, cache: nil))[0, (row.scoreStart - 1)..., 0...]
                    XCTAssertEqual(MLX.abs(actual - logits).max().item(Float.self), 0, row.id)
                }
                MLX.Memory.clearCache()
            }
        }
    }

    private func diagnosticLayer(
        _ input: MLXArray,
        layer: LFM2DecoderLayer,
        index: Int,
        config: LFM2Config,
        mode: Mode,
        intervention: RoutingIntervention?
    ) throws -> [String: MLXArray] {
        let operatorInput = layer.operatorNorm(input)
        let operatorOutput: MLXArray
        if layer.isAttentionLayer {
            operatorOutput = try XCTUnwrap(layer.selfAttention)(operatorInput, mask: .causal, cache: nil)
        } else {
            operatorOutput = try XCTUnwrap(layer.conv)(operatorInput, cache: nil)
        }
        let residual = input + operatorOutput
        let ffnInput = layer.ffnNorm(residual)
        var arrays = ["input": input, "operator_input": operatorInput, "operator_output": operatorOutput, "ffn_input": ffnInput]
        let ffnOutput: MLXArray
        if index < config.numDenseLayers {
            if mode == .compiledSwiGLU {
                let gate = try XCTUnwrap(layer.feedForward.gateProj)(ffnInput)
                let up = try XCTUnwrap(layer.feedForward.upProj)(ffnInput)
                ffnOutput = try XCTUnwrap(layer.feedForward.downProj)(compiledSwiglu(gate, up))
            } else {
                ffnOutput = layer.feedForward(ffnInput)
            }
        } else {
            let moe = layer.feedForward
            let gateLogits = try XCTUnwrap(moe.gate)(ffnInput)
            let probabilities = sigmoid(gateLogits)
            let selection = probabilities.asType(.float32) + (try XCTUnwrap(moe.expertBias))
            let topK = config.numExpertsPerTok
            var indices: MLXArray
            switch mode {
            case .baseline:
                indices = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
            case .positivePartition, .positiveSorted, .denseExpertReference, .boundaryCounterfactual, .fp32Parameters, .compiledSwiGLU, .upstreamReference:
                indices = argPartition(selection, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
            }
            if let intervention, intervention.layer == index {
                var selected = indices.asArray(Int32.self)
                let range = (intervention.tokenIndex * topK)..<((intervention.tokenIndex + 1) * topK)
                let slot = try XCTUnwrap(range.first { selected[$0] == intervention.originalExpert })
                XCTAssertFalse(range.contains { selected[$0] == intervention.replacementExpert })
                selected[slot] = intervention.replacementExpert
                indices = MLXArray(selected, indices.shape)
            }
            var scores = takeAlong(probabilities, indices, axis: -1)
            if config.normTopKProb {
                scores = scores / (scores.sum(axis: -1, keepDims: true) + MLXArray(1e-6))
            }
            scores = (scores * config.routedScalingFactor).asType(ffnInput.dtype)
            let mlp = try XCTUnwrap(moe.switchMLP)
            let switched: MLXArray
            if mode == .positiveSorted || mode == .denseExpertReference || mode == .fp32Parameters || mode == .compiledSwiGLU {
                switched = try sortedExperts(
                    ffnInput, indices: indices, mlp: mlp, dense: mode == .denseExpertReference,
                    compiled: mode == .compiledSwiGLU
                )
            } else {
                switched = mlp(ffnInput, indices: indices)
            }
            ffnOutput = (switched * MLX.expandedDimensions(scores, axis: scores.ndim)).sum(axis: -2)
            arrays.merge([
                "gate_logits": gateLogits, "probabilities": probabilities, "selection": selection,
                "indices": indices, "scores": scores, "ffn_output": ffnOutput,
            ]) { _, value in value }
        }
        arrays["output"] = residual + ffnOutput
        return arrays
    }
    private func sortedExperts(
        _ input: MLXArray, indices: MLXArray, mlp: LFM2SwitchGLU, dense: Bool, compiled: Bool
    ) throws -> MLXArray {
        let topK = indices.dim(2)
        let flattenedIndices = indices.reshaped([-1])
        let order = argSort(flattenedIndices)
        let inverse = argSort(order)
        let rows = MLX.floorDivide(order, MLXArray(topK)).asType(.int32)
        let sortedInput = take(input.reshaped([-1, input.dim(-1)]), rows, axis: 0).expandedDimensions(axis: 1)
        let sortedIndices = take(flattenedIndices, order, axis: 0)
        let up = try sortedProjection(sortedInput, projection: mlp.upProj, indices: sortedIndices, dense: dense)
        let gate = try sortedProjection(sortedInput, projection: mlp.gateProj, indices: sortedIndices, dense: dense)
        let activated = compiled ? compiledSwiglu(gate, up) : MLXNN.silu(gate) * up
        let down = try sortedProjection(activated, projection: mlp.downProj, indices: sortedIndices, dense: dense)
        return take(down, inverse, axis: 0).reshaped([input.dim(0), input.dim(1), topK, down.dim(-1)])
    }

    private func sortedProjection(
        _ input: MLXArray, projection: LFM2SwitchLinear, indices: MLXArray, dense: Bool
    ) throws -> MLXArray {
        let scales = try XCTUnwrap(projection.scales)
        let biases = try XCTUnwrap(projection.biases)
        let output: MLXArray
        if dense {
            // Materialize just one expert bank at a time; the full expanded model would not fit.
            let weight = MLX.dequantized(
                projection.weight, scales: scales.asType(.float32), biases: biases.asType(.float32),
                groupSize: projection.groupSize, bits: projection.bits, mode: .affine
            )
            output = gatherMM(input, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: true)
        } else {
            output = MLX.gatherQuantizedMM(
                input, projection.weight, scales: scales, biases: biases, rhsIndices: indices,
                transpose: true, groupSize: projection.groupSize, bits: projection.bits, sortedIndices: true
            )
        }
        MLX.eval(output)
        MLX.Memory.clearCache()
        return output
    }

}
