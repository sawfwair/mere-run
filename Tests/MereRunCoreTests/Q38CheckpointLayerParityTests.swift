import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

/// Loads one decoder block at a time, never the full checkpoint or PLE table.
/// Learned weights complement the synthetic fixtures when rare rounding
/// differences only surface after many speculative verification rounds.
final class Q38CheckpointLayerParityTests: MereRunCoreTestCase {
    func testInstalledDecoderVerificationAndRollbackMatchSerial() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS"] == "1",
                          "Installed decoder parity requires the checkpoint opt-in.")
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu,
                          "Installed decoder parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        let root = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"]))
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self, from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json"))
        )
        for layerIndex in [0, 1, 3, 23, 47] {
            try qualifyLayer(layerIndex, config: config, index: index, root: root)
            Stream.gpu.synchronize()
            Memory.clearCache()
        }
    }

    private func qualifyLayer(_ layerIndex: Int, config: Q35Config, index: HFSafetensorsIndex, root: URL) throws {
        MLXRandom.seed(UInt64(126 + layerIndex))
        let layer = Q35DecoderLayer(config: config, layerIndex: layerIndex, includesPLE: false)
        try loadLayer(layer, layerIndex: layerIndex, config: config, index: index, root: root)
        let text = config.textConfig
        let width = text.hiddenSize * text.hyperConnectionCount
        let base = makeCache(layer: layer, config: config)
        var committed = base.fork()
        let reference = base.fork()
        for round in 0..<16 {
            let input = MLXRandom.normal([1, 4, width]).asType(.bfloat16)
            let referenceBefore = reference.fork()
            let candidate = committed.fork()
            let actual = layer(input, fullMask: .causal, cache: candidate, targetVerify: true)
            MLX.eval(actual)
            let kept = round % 4 + 1
            let serial = (0..<kept).map { row in
                let output = layer(input[0..., row..<(row + 1), 0...], fullMask: .none, cache: reference)
                MLX.eval(output)
                return output
            }
            XCTAssertTrue(candidate.restoreVerificationPrefix(totalTokens: 4, tokenCount: kept))
            let label = "layer=\(layerIndex) round=\(round) kept=\(kept)"
            let expected = MLX.concatenated(serial, axis: 1)
            assertExact(actual[0..., 0..<kept, 0...], expected, label)
            if layer.layerType == .full,
               (actual[0..., 0..<kept, 0...] - expected).abs().max().item(Float.self) != 0 {
                try traceFullAttentionStages(
                    layer, input: input, candidate: committed.fork(), reference: referenceBefore,
                    kept: kept, label: label
                )
            }
            if case .linear(let restored) = candidate, case .linear(let expected) = reference {
                assertExact(try XCTUnwrap(restored.convState), try XCTUnwrap(expected.convState), label + " convolution")
                assertExact(try XCTUnwrap(restored.recurrentState), try XCTUnwrap(expected.recurrentState), label + " recurrent")
            }
            committed = candidate
        }
        print("[q38-layer-parity] layer=\(layerIndex) type=\(layer.layerType.rawValue) rounds=16")
    }

    private func traceFullAttentionStages(
        _ layer: Q35DecoderLayer, input: MLXArray, candidate: Q35LayerCache,
        reference: Q35LayerCache, kept: Int, label: String
    ) throws {
        let attention = try XCTUnwrap(layer.attentionHyperConnection)
        let mlp = try XCTUnwrap(layer.mlpHyperConnection)
        guard case .full(let candidateCache) = candidate,
              case .full(let referenceCache) = reference else { return XCTFail("Expected full-attention caches") }
        let mixed = attention.mix(input)
        let attended = layer.selfAttention(mixed.mixed, mask: .causal, cache: candidateCache, targetVerify: true)
        let residual = attention.inject(
            blockOutput: attended, residual: mixed.residual, injectionWeights: mixed.injectionWeights
        )
        let mlpMixed = mlp.mix(residual)
        let output = layer.mlp(mlpMixed.mixed)
        let actual = [mixed.mixed, mixed.injectionWeights, attended, residual,
                      mlpMixed.mixed, mlpMixed.injectionWeights, output]
        MLX.eval(actual)
        var serial = Array(repeating: [MLXArray](), count: actual.count)
        for row in 0..<kept {
            let piece = input[0..., row..<(row + 1), 0...]
            let mixed = attention.mix(piece)
            let attended = layer.selfAttention(mixed.mixed, mask: .none, cache: referenceCache)
            let residual = attention.inject(
                blockOutput: attended, residual: mixed.residual, injectionWeights: mixed.injectionWeights
            )
            let mlpMixed = mlp.mix(residual)
            let output = layer.mlp(mlpMixed.mixed)
            let values = [mixed.mixed, mixed.injectionWeights, attended, residual,
                          mlpMixed.mixed, mlpMixed.injectionWeights, output]
            MLX.eval(values)
            for index in values.indices { serial[index].append(values[index]) }
        }
        let names = ["attention-mix", "attention-injection", "attention-output", "attention-residual",
                     "mlp-mix", "mlp-injection", "mlp-output"]
        for index in actual.indices {
            let expected = MLX.concatenated(serial[index], axis: 1)
            let difference = (actual[index][0..., 0..<kept, 0...].asType(.float32) - expected.asType(.float32)).abs()
            print("[q38-layer-stage] \(label) stage=\(names[index]) max_error=\(difference.max().item(Float.self))")
        }
    }

    private func makeCache(layer: Q35DecoderLayer, config: Q35Config) -> Q35LayerCache {
        if layer.layerType == .linear { return .linear(Q35LinearCache()) }
        let text = config.textConfig
        let count = Int(ProcessInfo.processInfo.environment["MERERUN_TEST_Q38_LAYER_PREFIX_TOKENS"] ?? "64740")!
        precondition((0...129_703).contains(count))
        let cache = Q38QSACache()
        let keys = MLXRandom.normal([1, text.numKeyValueHeads, count, text.headDim]).asType(.bfloat16)
        let values = MLXRandom.normal(keys.shape).asType(.bfloat16)
        let indexKeys = MLXRandom.normal([1, text.indexerKVHeadCount, count, text.indexerHeadDimension]).asType(.bfloat16)
        let positions = Q38QSAIndexer.positionRows(batch: 1, count: count, offsets: [0], positionIds: nil)
        _ = cache.update(keys: keys, values: values)
        _ = cache.updateIndexer(keys: indexKeys, positions: positions)
        MLX.eval(keys, values, indexKeys, positions)
        return .full(cache)
    }

    private func loadLayer(
        _ layer: Q35DecoderLayer, layerIndex: Int, config: Q35Config, index: HFSafetensorsIndex, root: URL
    ) throws {
        let prefix = "language_model.model.layers.\(layerIndex)."
        let keys = Set(index.weightMap.keys.filter { $0.hasPrefix(prefix) && !$0.hasPrefix(prefix + "ple.") })
        XCTAssertFalse(keys.isEmpty)
        let files = Set(keys.compactMap { index.weightMap[$0] })
        var weights: [String: MLXArray] = [:]
        for file in files.sorted() {
            let arrays = try SafetensorsStreamingLoader.loadArrays(url: root.appendingPathComponent(file), where: keys.contains)
            for (key, value) in arrays {
                let local = String(key.dropFirst(prefix.count))
                weights[local] = local.hasSuffix("conv1d.weight")
                    ? Q35Generator.normalizedLinearAttentionConv1DWeight(value) : value
            }
        }
        XCTAssertEqual(weights.count, keys.count)
        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            weights, to: layer, groupSize: config.quantization?.groupSize ?? 64,
            bits: config.quantization?.bits ?? 4
        )
        XCTAssertTrue(Set(weights.keys).isSubset(of: Set(layer.parameters().flattened().map(\.0))))
        MLX.eval(layer.parameters().flattened().map(\.1))
        _ = layer.mlp.prepareFusedSwitchGLU()
    }

    private func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        let difference = (actual.asType(.float32) - expected.asType(.float32)).abs()
        XCTAssertEqual(difference.max().item(Float.self), 0, "\(label) changed serial arithmetic")
    }
}
