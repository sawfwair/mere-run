import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class H3ImageRealCheckpointTests: MLXTestCase {
    func testReal4KTransformerBlockProbe() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_IMAGE_CHECKPOINT"],
              let output = environment["MERERUN_H3_IMAGE_PROBE_OUTPUT"] else {
            throw XCTSkip("Requires the accepted H3-Image checkpoint and a probe output directory.")
        }
        let resources = MiniMaxH3ImageResources(rootURL: URL(fileURLWithPath: root))
        let model = MiniMaxH3ImageTransformer(configuration: .init(layerCount: 1))
        for shard in try resources.index("transformer").shardFilenames {
            let raw = try MLX.loadArrays(url: resources.rootURL.appending(path: "transformer/\(shard)"))
            let mapped = raw.flatMap { key, value -> [(String, MLXArray)] in
                if key.hasPrefix("blocks."), !key.hasPrefix("blocks.0.") { return [] }
                if key == "rope.inv_freq" || key.hasPrefix("final_layer.video_out.") { return [] }
                return MiniMaxH3ModelLoader.releasedBF16TransformerWeight(key: key, value: value, omitCachedAdaLNWeights: false)
            }
            try model.update(parameters: ModuleParameters.unflattened(mapped), verify: [.shapeMismatch, .noUnusedKeys])
        }
        let heads = try MLX.loadArrays(url: resources.rootURL.appending(path: "pdd_fused_heads.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened([
            ("final_layer.pdd_weight", try XCTUnwrap(heads["video_out.weight"])),
            ("final_layer.pdd_bias", try XCTUnwrap(heads["video_out.bias"]))
        ]), verify: [.shapeMismatch, .noUnusedKeys])
        let states = MLXRandom.normal([1, 3, 5_120], key: MLXRandom.key(71)).asType(.bfloat16)
        let rows = MLXRandom.normal([1, 16_384, 96], key: MLXRandom.key(72))
        let text = model.prepareText(states)
        let layout = MiniMaxH3ImageLayout(textTags: [1, 1, 1], referenceGrids: [], height: 256, width: 256)
        let prediction = try model(imageRows: rows, text: text, layout: layout, step: 0)
        MLX.eval(prediction)
        let directory = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(arrays: ["prediction": prediction], url: directory.appending(path: "native-4k-block-probe.safetensors"))
        XCTAssertTrue(MLX.all(MLX.isFinite(prediction)).item(Bool.self))
    }

    func testRealTransformerProbe() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_IMAGE_CHECKPOINT"],
              let output = environment["MERERUN_H3_IMAGE_PROBE_OUTPUT"] else {
            throw XCTSkip("Requires the accepted H3-Image checkpoint and a probe output directory.")
        }
        let resources = MiniMaxH3ImageResources(rootURL: URL(fileURLWithPath: root))
        let model = try MiniMaxH3ImageLoader.transformer(resources)
        let edge = environment["MERERUN_H3_IMAGE_PROBE_SIZE"] == "4096" ? 256 : 4
        let states = MLXRandom.normal([1, 3, 5_120], key: MLXRandom.key(71)).asType(.bfloat16)
        let rows = MLXRandom.normal([1, edge / 2 * (edge / 2), 96], key: MLXRandom.key(72))
        let text = model.prepareText(states)
        let layout = MiniMaxH3ImageLayout(textTags: [1, 1, 1], referenceGrids: [], height: edge, width: edge)
        let prediction = try model(imageRows: rows, text: text, layout: layout, step: 0)
        MLX.eval(prediction)
        let directory = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(arrays: ["states": states, "rows": rows, "text": text, "prediction": prediction],
                     url: directory.appending(path: edge == 256 ? "native-transformer-4096-probe.safetensors" : "native-transformer-probe.safetensors"))
        XCTAssertTrue(MLX.all(MLX.isFinite(prediction)).item(Bool.self))
    }

    func testRealVAEDecodeProbe() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_IMAGE_CHECKPOINT"],
              let output = environment["MERERUN_H3_IMAGE_PROBE_OUTPUT"] else {
            throw XCTSkip("Requires the accepted H3-Image checkpoint and a probe output directory.")
        }
        let resources = MiniMaxH3ImageResources(rootURL: URL(fileURLWithPath: root))
        let vae = try MiniMaxH3ImageLoader.vae(resources)
        let latents = MLX.sin(MLXArray(0..<(24 * 16 * 16)).asType(.float32) * 0.017).reshaped(1, 24, 16, 16)
        let decoded = vae.decode(latents)
        MLX.eval(decoded)
        let directory = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(arrays: ["latents": latents, "rgb": decoded], url: directory.appending(path: "native-vae-probe.safetensors"))
        XCTAssertTrue(MLX.all(MLX.isFinite(decoded)).item(Bool.self))
    }
}
