import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LTXDiffusionVideoDecoderTests: MereRunCoreTestCase {
    func testOfficialFusedQKVMapsToNativeSplitProjections() {
        let weight = MLX.zeros([768, 256], dtype: .float32)
        let mappedWeight = mapLTXDiffusionVideoDecoderWeight(
            key: "decoder.diff_blocks.0.attn.qkv.weight",
            value: weight,
            dtype: .bfloat16
        )
        XCTAssertEqual(mappedWeight.map(\.0), [
            "diff_blocks.0.attn.to_q.weight",
            "diff_blocks.0.attn.to_k.weight",
            "diff_blocks.0.attn.to_v.weight",
        ])
        XCTAssertTrue(mappedWeight.allSatisfy { $0.1.shape == [256, 256] })
        XCTAssertTrue(mappedWeight.allSatisfy { $0.1.dtype == .bfloat16 })

        let bias = MLX.zeros([768], dtype: .float32)
        let mappedBias = mapLTXDiffusionVideoDecoderWeight(
            key: "decoder.diff_blocks.0.attn.qkv.bias",
            value: bias,
            dtype: .bfloat16
        )
        XCTAssertEqual(mappedBias.map(\.0), [
            "diff_blocks.0.attn.to_q.bias",
            "diff_blocks.0.attn.to_k.bias",
            "diff_blocks.0.attn.to_v.bias",
        ])
        XCTAssertTrue(mappedBias.allSatisfy { $0.1.shape == [256] })
    }

    func testOfficialNestedStageAndTimestepKeysMapToNativeModules() {
        let stage = mapLTXDiffusionVideoDecoderWeight(
            key: "decoder.det_stages.3.1.mlp.w_down.weight",
            value: MLX.zeros([512, 2_048]),
            dtype: .float32
        )
        XCTAssertEqual(stage.first?.0, "det_stages.3.blocks.1.mlp.w_down.weight")

        let first = mapLTXDiffusionVideoDecoderWeight(
            key: "decoder.t_embedder.mlp.0.weight",
            value: MLX.zeros([384, 256]),
            dtype: .float32
        )
        let second = mapLTXDiffusionVideoDecoderWeight(
            key: "decoder.t_embedder.mlp.2.weight",
            value: MLX.zeros([384, 384]),
            dtype: .float32
        )
        XCTAssertEqual(first.first?.0, "t_embedder.linear_1.weight")
        XCTAssertEqual(second.first?.0, "t_embedder.linear_2.weight")
        XCTAssertTrue(
            mapLTXDiffusionVideoDecoderWeight(
                key: "decoder.type_emb",
                value: MLX.zeros([128]),
                dtype: .float32
            ).isEmpty
        )
    }

    func testNativeParameterLayoutMatchesOfficialDecoderInventory() {
        let parameters = Dictionary(
            uniqueKeysWithValues: LTXDiffusionVideoDecoder().parameters().flattened()
        )
        XCTAssertEqual(parameters.count, 407)
        XCTAssertEqual(parameters["conv_in.weight"]?.shape, [2_048, 128])
        XCTAssertEqual(parameters["det_stages.0.blocks.0.attn.to_q.weight"]?.shape, [2_048, 2_048])
        XCTAssertEqual(parameters["upsamples.3.proj.weight"]?.shape, [2_048, 512])
        XCTAssertEqual(parameters["diff_blocks.7.scale_shift_table"]?.shape, [7, 256])
        XCTAssertEqual(parameters["shared_adaln.proj.weight"]?.shape, [1_792, 384])
        XCTAssertEqual(parameters["conv_out.weight"]?.shape, [48, 256])
    }

    func testNattenShiftedBoundaryWindowsMatchOfficialSemantics() {
        let bounds = ltxDiffVAEWindowBounds(length: 8, kernel: 5)
        XCTAssertEqual(bounds.starts, [0, 0, 0, 1, 2, 3, 3, 3])
        XCTAssertEqual(bounds.ends, [5, 5, 5, 6, 7, 8, 8, 8])
    }

    func testInstalledOfficialCheckpointMetadataCoversEveryNativeParameter() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_LTX25_DIFFVAE_WEIGHTS"] else {
            throw XCTSkip("Set MERERUN_LTX25_DIFFVAE_WEIGHTS for installed-checkpoint coverage.")
        }
        let metadata = try SafetensorsStreamingLoader.metadata(
            url: URL(fileURLWithPath: path).standardizedFileURL
        )
        var mapped: [String: [Int]] = [:]
        for (key, tensor) in metadata where key.hasPrefix("decoder.") {
            for target in ltxDiffusionVideoDecoderWeightTargets(key: key, shape: tensor.shape) {
                mapped[target.name] = target.shape
            }
        }
        var native = Dictionary(
            uniqueKeysWithValues: LTXDiffusionVideoDecoder().parameters().flattened().map {
                ($0.0, $0.1.shape)
            }
        )
        native.removeValue(forKey: "latentsMean")
        native.removeValue(forKey: "latentsStd")
        XCTAssertEqual(mapped, native)
    }
}
