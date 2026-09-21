import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Opt-in original-checkpoint evidence; no downloads happen inside this test.
final class QwenImage21QualificationTests: MereRunCoreTestCase {
    private struct TokenCase: Decodable {
        let text: String
        let ids: [Int]
    }

    func testCheckpointTokenizerMatchesReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["QWEN21_MODEL_ROOT"],
              let outputPath = environment["QWEN21_QUALIFICATION_ROOT"] else {
            throw XCTSkip("Set QWEN21_MODEL_ROOT and QWEN21_QUALIFICATION_ROOT for checkpoint tokenizer parity.")
        }
        let cases = try JSONDecoder().decode([TokenCase].self, from: Data(contentsOf:
            URL(fileURLWithPath: outputPath).appending(path: "tokenizer-reference.json")))
        let tokenizer = try QwenTokenizer.load(from: URL(fileURLWithPath: modelPath).appending(path: "processor"),
                                              maxLengthOverride: 262_144)
        for item in cases { XCTAssertEqual(tokenizer.encodeText(item.text), item.ids, item.text) }
    }

    func testExportTrainedCheckpointComponents() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["QWEN21_MODEL_ROOT"],
              let outputPath = environment["QWEN21_QUALIFICATION_ROOT"] else {
            throw XCTSkip("Set QWEN21_MODEL_ROOT and QWEN21_QUALIFICATION_ROOT for trained checkpoint qualification.")
        }
        let resources = QwenImage21Resources(rootURL: URL(fileURLWithPath: modelPath))
        let transformerFloat32 = environment["QWEN21_TRANSFORMER_FP32"] == "1"
        let targetGrid = Int(environment["QWEN21_TARGET_GRID"] ?? "2")!
        let targetCount = targetGrid * targetGrid
        let output = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var tensors: [String: MLXArray] = [:]
        let pixels = sin(MLXArray(0..<(32 * 32 * 4)).asType(.float32) * 0.013)
            .reshaped(1, 32, 32, 4).asType(.bfloat16)
        let vision = pixels[0..., 0..., 0..., ..<3].transposed(0, 3, 1, 2).asType(.float32)
        tensors["rgba"] = pixels
        tensors["vision"] = vision
        let prompt = "A red ceramic teapot on a wooden table."
        var layouts: [String: QwenImage21Layout] = [:]
        do {
            let conditioner = try QwenImage21Conditioner(resources: resources)
            for mode in ["text", "image"] {
                let images: [QwenImage21ImageIO.Reference] = mode == "text" ? [] : [
                    .init(rgba: pixels, vision: vision, width: 32, height: 32)
                ]
                let (hidden, layout) = try conditioner.encode(prompt: prompt, images: images, targetHeight: targetGrid, targetWidth: targetGrid)
                tensors[mode + "_conditioning"] = hidden
                layouts[mode] = layout
                let prefix = mode == "text" ? "" : "<image1><|vision_start|><|image_pad|><|vision_end|>"
                let presentation = "<|im_start|>system\n" + QwenImage21Conditioner.systemPrompt
                    + "<|im_end|>\n<|im_start|>user\n" + prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n"
                let ids = conditioner.tokenizer.encodeText(presentation)
                tensors[mode + "_ids"] = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
                tensors[mode + "_mask"] = MLXArray(ids.dropFirst(conditioner.dropCount).map {
                    $0 == conditioner.tokenizer.imageTokenId
                } + Array(repeating: true, count: targetCount / 4)).reshaped(1, -1)
                tensors["drop_count"] = MLXArray(Int32(conditioner.dropCount))
                tensors["target_grid"] = MLXArray(Int32(targetGrid))
            }
        }
        Memory.clearCache()
        do {
            let config = try resources.decode("vae/config.json", as: QwenImage21VAEConfig.self)
            let vae = try QwenImage21VAE(config: config, arrays: resources.arrays("vae", stem: "diffusion_pytorch_model"))
            tensors["encoded"] = try vae.encode(pixels)
            let latent = cos(MLXArray(0..<(4 * config.zDim)).asType(.float32) * 0.07)
                .reshaped(1, 2, 2, config.zDim).asType(.bfloat16)
            tensors["decode_latent"] = latent
            tensors["decoded"] = try vae.decode(latent)
            eval(tensors)
        }
        Memory.clearCache()
        do {
            let config = try resources.decode("transformer/config.json", as: QwenImage21TransformerConfig.self)
            let dtype: DType = transformerFloat32 ? .float32 : .bfloat16
            let arrays = try resources.arrays("transformer", stem: "diffusion_pytorch_model").mapValues { $0.asType(dtype) }
            let transformer = try QwenImage21Transformer(config: config, arrays: arrays)
            for mode in ["text", "image"] {
                let count = targetCount + (mode == "text" ? 0 : 4)
                let latent = sin(MLXArray(0..<(count * config.inChannels)).asType(.float32) * 0.03)
                    .reshaped(1, count, config.inChannels).asType(.bfloat16).asType(dtype)
                let hidden = try XCTUnwrap(tensors[mode + "_conditioning"]).asType(dtype)
                let layout = try XCTUnwrap(layouts[mode])
                let cache = QwenImage21PrefixCache(layerCount: config.numLayers)
                tensors[mode + "_latents"] = latent
                tensors[mode + "_velocity"] = try transformer(latents: latent, text: hidden, timestep: 0.75, layout: layout, cache: cache)
                let changed = concatenated([latent[0..., ..<(count - targetCount), 0...], latent[0..., (count - targetCount)..., 0...] + 0.125], axis: 1)
                let cached = try transformer(latents: changed, text: hidden, timestep: 0.4, layout: layout, cache: cache)
                let uncached = try transformer(latents: changed, text: hidden, timestep: 0.4, layout: layout)
                tensors[mode + "_changed"] = changed
                tensors[mode + "_cached"] = cached
                tensors[mode + "_uncached"] = uncached
                XCTAssertLessThan(max(abs(cached.asType(.float32) - uncached.asType(.float32))).item(Float.self), 0.05)
                eval(tensors)
            }
        }
        for (name, tensor) in tensors where tensor.dtype.isFloatingPoint {
            XCTAssertTrue(all(isFinite(tensor)).item(Bool.self), name)
        }
        let stem = transformerFloat32 ? "native-transformer-fp32" : "native-components"
        let filename = stem + (targetGrid == 2 ? "" : "-grid\(targetGrid)") + ".safetensors"
        try save(arrays: tensors, url: output.appending(path: filename))
        Memory.clearCache()
    }
}
