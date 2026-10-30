import Foundation
import MLX
import MLXFast
import XCTest
import MereRunMLXTestSupport
@testable import MereRunUnlimitedOCRModel

/// Opt-in differential diagnostics. The upstream tensors are generated externally.
final class UnlimitedOCRParityTests: MLXTestCase {
    func testReleasedCheckpointIntermediatesAgainstUpstream() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_TEST_UNLIMITED_OCR_TRACE"] else {
            throw XCTSkip("Set MERERUN_TEST_UNLIMITED_OCR_TRACE to an upstream tensor capture.")
        }
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
        let root = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_ROOT"]))
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var tensors: [String: MLXArray] = [:]
        for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] {
            tensors.merge(try MLX.loadArrays(url: root.appendingPathComponent(shard))) { first, _ in first }
        }
        var metrics: [String: Metric] = [:]
        func compare(_ name: String, _ actual: MLXArray) {
            let key = String(name.split(separator: "/").last ?? Substring(name)).replacingOccurrences(of: "isolated.", with: "")
            guard var expected = reference[key] else { return }
            if name == "decoder.norm" { expected = expected[(-1)...] }
            XCTAssertEqual(actual.shape, expected.shape, name)
            let delta = actual.asType(.float32) - expected.asType(.float32)
            let metric = Metric(maximum: abs(delta).max().item(Float.self),
                                rmse: sqrt(mean(square(delta))).item(Float.self),
                                referenceRMS: sqrt(mean(square(expected.asType(.float32)))).item(Float.self),
                                unequalFraction: mean((actual .!= expected).asType(.float32)).item(Float.self))
            metrics[name] = metric
            FileHandle.standardError.write(Data("[parity] \(name): max=\(metric.maximum), rmse=\(metric.rmse), unequal=\(metric.unequalFraction)\n".utf8))
        }
        let weights = UnlimitedOCRWeights(tensors: tensors)
        let vision = UnlimitedOCRVision(weights: weights)
        compare("probe.sam_pos_40", UnlimitedOCRMath.resizeGrid(weights.tensor("model.sam_model.pos_embed"), height: 40, width: 40))
        compare("probe.clip_pos_10", UnlimitedOCRMath.resizeGrid(
            weights.tensor("model.vision_model.embeddings.position_embedding.weight")[1...].reshaped(1, 16, 16, 1024), height: 10, width: 10))
        _ = vision.encode(try XCTUnwrap(reference["pixels"]), trace: compare)
        for layer in 0..<12 {
            let input = layer == 0
                ? try XCTUnwrap(reference["sam.patch"]) + weights.tensor("model.sam_model.pos_embed")
                : try XCTUnwrap(reference["sam.block.\(layer - 1)"])
            compare("isolated.sam.block.\(layer)", vision.samBlock(input, layer: layer))
        }
        for layer in 0..<24 {
            let input = try XCTUnwrap(reference[layer == 0 ? "clip.pre" : "clip.block.\(layer - 1)"])
            compare("isolated.clip.block.\(layer)", vision.clipBlock(input, layer: layer))
        }
        // Same-input probes distinguish operation errors from propagated vision drift.
        compare("isolated.sam.first.gelu", UnlimitedOCRMath.gelu(try XCTUnwrap(reference["sam.first.up"])))
        let clipUp = try XCTUnwrap(reference["clip.first.up"])
        compare("isolated.clip.first.quick_gelu", UnlimitedOCRMath.quickGelu(clipUp))
        let normalized = try XCTUnwrap(reference["decoder.first.norm1"])
        compare("isolated.decoder.first.norm1", weights.rms(try XCTUnwrap(reference["embeddings"]), "model.layers.0.input_layernorm"))
        compare("isolated.decoder.first.norm2", weights.rms(
            (try XCTUnwrap(reference["embeddings"])) + (try XCTUnwrap(reference["decoder.first.attention"])),
            "model.layers.0.post_attention_layernorm"))
        for projection in ["q_proj", "k_proj", "v_proj"] {
            compare("isolated.decoder.first.\(projection)", weights.linear(normalized, "model.layers.0.self_attn.\(projection)"))
        }
        compare("isolated.decoder.first.gated", UnlimitedOCRMath.silu(try XCTUnwrap(reference["decoder.first.gate"]))
                * (try XCTUnwrap(reference["decoder.first.up"])))
        let config = try UnlimitedOCRConfiguration.load(Data(contentsOf: root.appendingPathComponent("config.json")))
        let decoder = UnlimitedOCRDecoder(config: config, tensors: tensors)
        let ids = try XCTUnwrap(reference["prompt_ids"]).asArray(Int.self)
        _ = decoder.forward(try XCTUnwrap(reference["embeddings"]), state: try UnlimitedOCRPageState(tokens: ids),
                            cache: (0..<12).map { _ in UnlimitedOCRCache() }, trace: compare)
        let state = try UnlimitedOCRPageState(tokens: ids)
        if let q = reference["probe.attention.q"], let k = reference["probe.attention.k"], let v = reference["probe.attention.v"] {
            let positions = MLXArray(0..<64)
            let causal = positions.expandedDimensions(axis: -1) .>= positions
            let mask = which(causal, MLXArray(Float(0)), MLXArray(-Float.infinity))
            let result = MLXFast.scaledDotProductAttention(
                queries: q.asType(.float32).expandedDimensions(axis: 0), keys: k.asType(.float32).expandedDimensions(axis: 0),
                values: v.asType(.float32).expandedDimensions(axis: 0), scale: 1 / sqrt(Float(128)), mask: mask
            ).asType(q.dtype).squeezed(axis: 0)
            for device in ["cpu", "mps"] {
                for dtype in ["torch.bfloat16", "torch.float32"] { compare("probe.attention.\(device)_\(dtype)", result) }
            }
        }
        for projection in ["q", "k"] where reference["decoder.first.rope_\(projection)"] != nil {
            let projected = try XCTUnwrap(reference["decoder.first.\(projection)_proj"]).reshaped(-1, 10, 128).transposed(1, 0, 2)
            compare("isolated.decoder.first.rope_\(projection)", UnlimitedOCRMath.rope(projected,
                tokens: MLXArray(state.tokenPositions), pages: MLXArray(state.pagePositions)))
        }
        for layer in 0..<12 {
            let input = try XCTUnwrap(reference[layer == 0 ? "embeddings" : "decoder.block.\(layer - 1)"])
            compare("isolated.decoder.block.\(layer)", decoder.block(input, layer: layer,
                tok: MLXArray(state.tokenPositions), page: MLXArray(state.pagePositions),
                state: state, offset: 0, cache: UnlimitedOCRCache()))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let output = env["MERERUN_TEST_UNLIMITED_OCR_PARITY_OUTPUT"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: path).deletingPathExtension().appendingPathExtension("native.json")
        try encoder.encode(metrics).write(to: output)
    }

    private struct Metric: Encodable {
        let maximum: Float
        let rmse: Float
        let referenceRMS: Float
        let unequalFraction: Float
    }
}
