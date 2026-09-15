import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class MarigoldV2TransformerDiagnosticTests: MereRunCoreTestCase {
    func testInstalledTransformerQuantizationWhenRequested() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_MARIGOLD_ROOT"],
              let fixture = environment["MERERUN_TEST_MARIGOLD_VAE_FIXTURE"],
              let output = environment["MERERUN_TEST_MARIGOLD_TRANSFORMER_OUTPUT"],
              let quantization = environment["MERERUN_TEST_MARIGOLD_QUANTIZATION"] else {
            throw XCTSkip("Set the Marigold model, shared VAE fixture, transformer output, and quantization variables.")
        }
        let resources = MarigoldV2Resources(rootURL: URL(fileURLWithPath: root))
        let configs = try MarigoldV2ModelConfigs.load(from: resources)
        let transformer = MMDiT(config: configs.transformer)
        Self.log("loading base")
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.transformerWeightsIndexURL, to: transformer, dtype: .bfloat16,
            verify: [.shapeMismatch], mapper: QwenImageEditGenerator.transformerWeightMapper(config: configs.transformer)
        )
        MLX.eval(transformer)
        Memory.clearCache()
        Self.log("quantizing \(quantization)")
        if quantization == "affine" || quantization == "affine-skip" {
            MLXNN.quantize(model: transformer, groupSize: 64, bits: 4) { path, module in
                if quantization == "affine-skip" && path == "transformer_blocks.0.adaLN_modulation.linear" {
                    return false
                }
                return (module as? Linear).map { $0.shape.1 % 64 == 0 } ?? true
            }
        } else if quantization == "nf4" {
            for (path, module) in transformer.leafModules().flattened() {
                guard let linear = module as? Linear,
                      path != "transformer_blocks.0.adaLN_modulation.linear" else { continue }
                let weight = Self.nf4EffectiveWeights(linear.weight)
                MLX.eval(weight)
                try linear.update(parameters: ModuleParameters.unflattened(["weight": weight]), verify: [.shapeMismatch])
                Memory.clearCache()
            }
        } else {
            XCTAssertEqual(quantization, "none")
        }
        Self.log("installing adapters")
        try MarigoldV2LoRAAdapter.install(url: resources.trainablesURL, into: transformer)
        MLX.eval(transformer)
        let prompt = try MarigoldV2PromptEmbeddingLoader.load(
            embedsURL: resources.promptEmbedsURL, maskURL: resources.promptMaskURL
        )
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let latents = try XCTUnwrap(arrays["normalized"]).asType(.bfloat16)
        let packed = QwenImageEditLatentCreator.packLatents(latents)
        Self.log("running transformer")
        let prediction = transformer(
            hiddenStates: packed, timestep: MarigoldV2Generator.inferenceTimestep(),
            contextEmbeds: prompt.embeddings, contextMask: prompt.mask,
            imageShapes: [(temporal: 1, height: latents.dim(2) / 2, width: latents.dim(3) / 2)],
            outputTokenCount: packed.dim(1)
        )
        let velocity = QwenImageEditLatentCreator.unpackLatents(
            prediction, height: latents.dim(2), width: latents.dim(3), channels: latents.dim(1)
        )
        let stepped = latents - velocity.asType(latents.dtype)
        MLX.eval(prediction, stepped)
        Self.log("decoding")
        let vae = QwenImageEditVAE(config: configs.vae)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL, to: vae.underlyingVAE, dtype: .bfloat16,
            verify: [.shapeMismatch], mapper: QwenImageEditVAE.weightMapper
        )
        try MarigoldV2VAEDecoder.applyIfPresent(url: resources.trainablesURL, to: vae)
        let pixels = vae.decodeGenerated(stepped)
        MLX.eval(pixels)
        let outputURL = URL(fileURLWithPath: output)
        try MLX.save(arrays: [
            "latents": latents.asType(.float32), "packed": packed.asType(.float32),
            "prediction": prediction.asType(.float32), "stepped": stepped.asType(.float32),
            "pixels": pixels.asType(.float32)
        ], url: outputURL)
        let depth = pixels.mean(axis: 1, keepDims: true).asType(.float32)
        let preview = 1 - (depth - depth.min()) / (depth.max() - depth.min())
        try QwenImageIO.saveImage(
            array: MLX.repeated(preview, count: 3, axis: 1),
            to: outputURL.deletingPathExtension().appendingPathExtension("png")
        )
        Self.log("saved \(output)")
        XCTAssertTrue(MLX.all(MLX.isFinite(pixels)).item(Bool.self))
    }

    // Diagnostic effective weights only; this does not implement packed NF4 inference.
    private static func nf4EffectiveWeights(_ weight: MLXArray) -> MLXArray {
        let values: [Float] = [
            -1, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1
        ]
        let blocks = weight.asType(.float32).reshaped(-1, 64)
        let scales = MLX.abs(blocks).max(axis: 1, keepDims: true)
        let normalized = blocks / MLX.maximum(scales, MLXArray(Float.leastNormalMagnitude))
        var levels = MLX.full(blocks.shape, values: values[0])
        for index in 1..<values.count {
            let boundary = (values[index - 1] + values[index]) / 2
            levels = MLX.where(normalized .> boundary, MLXArray(values[index]), levels)
        }
        return (levels * scales).asType(weight.dtype).reshaped(weight.shape)
    }

    private static func log(_ text: String) {
        FileHandle.standardError.write(Data("marigold_transformer \(text)\n".utf8))
    }
}
