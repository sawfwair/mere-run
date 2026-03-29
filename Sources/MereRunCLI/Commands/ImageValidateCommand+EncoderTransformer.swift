import Foundation
import MLX
import MLXRandom
import MereRunCore

/// Owns the model-component checks that sit between the VAE and the full
/// pipeline. Keeping encoder and transformer validation together makes it
/// easier to inspect the prompt-conditioning path in one place.
extension ImageValidate {
    func runEncoderTests(modelURL: URL, outputDir: URL) async throws {
        print("\n== Text encoder validation ==")
        print("▶ Test: Text encoder output shape and embedding quality")
        print("  Purpose: Validate tokenization and text encoding")

        let testPrompt = "a red cube on a white background"
        print("  Prompt: \"\(testPrompt)\"")

        let resources = ZImageTurboResources(rootURL: modelURL)
        let configs = try ZImageTurboModelConfigs.load(from: resources)
        print("  Model path: \(modelURL.path)")
        print("  Config: vocabSize=\(configs.textEncoder.vocabSize), hiddenSize=\(configs.textEncoder.hiddenSize), numLayers=\(configs.textEncoder.numHiddenLayers)")

        let tokenizer = try QwenTokenizer.load(
            from: modelURL.appendingPathComponent("tokenizer"),
            maxLengthOverride: configs.textEncoder.maxPositionEmbeddings
        )

        let maxLength = 512
        let batch = tokenizer.encode(prompts: [testPrompt], maxLength: maxLength)
        MLX.eval(batch.inputIds)
        print("  Tokens: \(batch.inputIds.dim(1)) tokens (padded to \(maxLength))")

        let textEncoder = try loadValidationTextEncoder(resources: resources, configs: configs)
        let (embeddings, mask) = textEncoder(inputIds: batch.inputIds, attentionMask: batch.attentionMask)
        MLX.eval(embeddings)

        print("  Embeddings shape: \(embeddings.shape)")
        let mean = MLX.mean(embeddings).item(Float.self)
        let std = MLX.std(embeddings).item(Float.self)
        let minVal = MLX.min(embeddings).item(Float.self)
        let maxVal = MLX.max(embeddings).item(Float.self)

        print("  Embedding stats: mean=\(String(format: "%.4f", mean)), std=\(String(format: "%.4f", std))")
        print("                   min=\(String(format: "%.4f", minVal)), max=\(String(format: "%.4f", maxVal))")
        print("  Mask shape: \(mask.shape)")

        if Swift.abs(mean) < 10 && std > 0.01 && std < 100 {
            print("  ✓ Embedding values: REASONABLE")
        } else {
            print("  ⚠ Embedding values: May need investigation")
        }

        if saveReference {
            let embPath = outputDir.appendingPathComponent("reference_embeddings.safetensors")
            try MLX.save(arrays: ["embeddings": embeddings, "mask": mask], url: embPath)
            print("  Saved embeddings: \(embPath.lastPathComponent)")
        }

        Memory.clearCache()
    }

    func runTransformerTests(modelURL: URL, outputDir: URL) async throws {
        print("\n== Transformer validation ==")
        print("▶ Test: Denoising loop with intermediate latent inspection")
        print("  Purpose: Validate transformer produces correct noise predictions")

        let testPrompt = "a red cube"
        let seed: UInt64 = 42
        let width = 256
        let height = 256
        let steps = 4

        print("  Prompt: \"\(testPrompt)\"")
        print("  Seed: \(seed)")
        print("  Size: \(width)x\(height)")
        print("  Steps: \(steps)")

        let resources = ZImageTurboResources(rootURL: modelURL)
        let configs = try ZImageTurboModelConfigs.load(from: resources)
        print("  Loading components...")

        let tokenizer = try QwenTokenizer.load(
            from: modelURL.appendingPathComponent("tokenizer"),
            maxLengthOverride: configs.textEncoder.maxPositionEmbeddings
        )
        let textEncoder = try loadValidationTextEncoder(resources: resources, configs: configs)

        let transformer = ZImageTransformer2DModel(configuration: configs.transformer)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.transformerWeightsIndexURL,
            to: transformer,
            dtype: .bfloat16,
            verify: .none
        )
        MLX.eval(transformer)
        print("  Components loaded")

        let batch = tokenizer.encode(prompts: [testPrompt], maxLength: 512)
        let promptEmbedsList = textEncoder.encodeForZImage(
            inputIds: batch.inputIds,
            attentionMask: batch.attentionMask
        )
        let promptEmbeds = promptEmbedsList[0].expandedDimensions(axis: 0)
        MLX.eval(promptEmbeds)

        let inferenceConfig = ZImageTurboInferenceConfig(
            width: width,
            height: height,
            numInferenceSteps: steps,
            imageStrength: nil
        )
        let schedulerConfig = configs.scheduler
        let resolvedSigmaShift = schedulerConfig.useDynamicShifting ? nil : schedulerConfig.shift
        let scheduler = ZImageTurboLinearScheduler(
            config: inferenceConfig,
            requiresSigmaShift: schedulerConfig.useDynamicShifting,
            sigmaShift: resolvedSigmaShift
        )

        MLXRandom.seed(seed)
        var latents = MLXRandom.normal([1, 16, height / 8, width / 8]).asType(.float32)
        print("\n  Running denoising loop:")
        print("  Sigmas: \(scheduler.sigmas.asArray(Float.self).prefix(steps + 1).map { String(format: "%.4f", $0) }.joined(separator: " → "))")

        var intermediateLatents: [MLXArray] = [latents]
        for stepIndex in 0..<steps {
            let sigma = scheduler.sigmas[stepIndex]
            let tInput = (MLXArray([Float(1.0)]) - sigma.asType(.float32)).asType(.float32)
            let predicted = transformer.forward(
                latents: latents.asType(.bfloat16),
                timestep: tInput,
                promptEmbeds: promptEmbeds
            )
            let noisePred = (-predicted).asType(.float32)
            latents = scheduler.step(noise: noisePred, timestep: stepIndex, latents: latents)
            MLX.eval(latents)
            intermediateLatents.append(latents)

            let latentMean = MLX.mean(latents).item(Float.self)
            let latentStd = MLX.std(latents).item(Float.self)
            print("  Step \(stepIndex + 1)/\(steps): mean=\(String(format: "%+.4f", latentMean)), std=\(String(format: "%.4f", latentStd))")
        }

        let initialMean = MLX.mean(intermediateLatents[0]).item(Float.self)
        let finalMean = MLX.mean(intermediateLatents.last!).item(Float.self)
        let latentDrift = Swift.abs(finalMean - initialMean)

        print("\n  Latent drift (initial → final): \(String(format: "%.4f", latentDrift))")
        if latentDrift > 0.01 {
            print("  ✓ Latents evolved: Transformer is producing noise predictions")
        } else {
            print("  ⚠ Latents static: Transformer may not be predicting correctly")
        }

        if saveReference {
            var arrays: [String: MLXArray] = [:]
            for (i, lat) in intermediateLatents.enumerated() {
                arrays["step_\(i)"] = lat
            }
            let latPath = outputDir.appendingPathComponent("reference_intermediate_latents.safetensors")
            try MLX.save(arrays: arrays, url: latPath)
            print("  Saved intermediate latents: \(latPath.lastPathComponent)")
        }

        transformer.clearCache()
        Memory.clearCache()
    }

    private func loadValidationTextEncoder(
        resources: ZImageTurboResources,
        configs: ZImageTurboModelConfigs
    ) throws -> QwenTextEncoder {
        print("  Loading text encoder...")
        let encoderConfig = QwenTextEncoderConfiguration(
            vocabSize: configs.textEncoder.vocabSize,
            hiddenSize: configs.textEncoder.hiddenSize,
            numHiddenLayers: configs.textEncoder.numHiddenLayers,
            numAttentionHeads: configs.textEncoder.numAttentionHeads,
            numKeyValueHeads: configs.textEncoder.numKeyValueHeads,
            intermediateSize: configs.textEncoder.intermediateSize,
            ropeTheta: configs.textEncoder.ropeTheta,
            maxPositionEmbeddings: configs.textEncoder.maxPositionEmbeddings,
            rmsNormEps: configs.textEncoder.rmsNormEps,
            headDim: configs.textEncoder.headDim
        )
        let textEncoder = QwenTextEncoder(configuration: encoderConfig)

        print("  Weight index: \(resources.textEncoderWeightsIndexURL.path)")
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.textEncoderWeightsIndexURL,
            to: textEncoder,
            dtype: .bfloat16,
            verify: .none,
            mapper: { key, value in
                if key.hasPrefix("model.") {
                    let remainder = String(key.dropFirst("model.".count))
                    return [("encoder.\(remainder)", value)]
                }
                return [("encoder.\(key)", value)]
            }
        )
        print("  Weights loaded, evaluating...")
        MLX.eval(textEncoder)
        print("  Text encoder ready")
        return textEncoder
    }
}
