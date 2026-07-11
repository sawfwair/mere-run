import Foundation
import MediaIO
import MLX
import MLXRandom
import MLXNN

extension Flux2KleinGenerator {

    // MARK: - Generation

    func generate(
        _ request: GenerationRequest,
        modelPath: String,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        let timingEnabled = {
            guard let raw = ProcessInfo.processInfo.environment["MERERUN_FLUX2_TIMING"]?.lowercased() else { return false }
            return raw == "1" || raw == "true" || raw == "yes"
        }()

        // Load models if needed (also reload if only chat models were loaded - vae is nil)
        if loadedModelPath != modelPath || vae == nil {
            try await loadModels(from: modelPath, progressHandler: progressHandler)
        }

        try await applyLoRAIfNeeded(request.lora, progressHandler: progressHandler)

        guard let transformer = transformer,
              let textEncoder = textEncoder,
              let tokenizer = tokenizer,
              let vae = vae,
              let bnMean = bnRunningMean,
              let bnVar = bnRunningVar else {
            throw Flux2Error.modelsNotLoaded
        }

        let debugLog: ((String) -> Void)? = {
            guard ProcessInfo.processInfo.environment["MERERUN_FLUX2_DEBUG"] == "1" else { return nil }
            return { message in
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }
        }()

        // 1. Encode prompt (and negative prompt for CFG)
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
        let (promptEmbeds, _) = try encodePrompt(
            prompt: request.prompt,
            tokenizer: tokenizer,
            textEncoder: textEncoder,
            debugLog: debugLog
        )

        guard let variant = loadedManifest?.variant else {
            throw Flux2Error.invalidManifest("Missing manifest.variant")
        }
        let isDistilled = variant == .distilled

        // CFG behavior differs between distilled and base models:
        // - Distilled: CFG only if user explicitly provides negative prompt (mflux behavior)
        // - Base: CFG required when guidance > 1.0, uses empty string as unconditional
        let useCFG: Bool
        if isDistilled {
            let hasNegativePrompt = request.negativePrompt.map {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            } ?? false
            useCFG = request.guidanceScale > 1.0 && hasNegativePrompt
        } else {
            // Base model: always use CFG when guidance > 1.0 (diffusers behavior)
            useCFG = request.guidanceScale > 1.0
        }

        let negativePromptEmbeds: MLXArray?
        if useCFG {
            // Use negative prompt if provided, otherwise empty string for unconditional
            let negPrompt = request.negativePrompt ?? ""
            let (negEmbeds, _) = try encodePrompt(
                prompt: negPrompt,
                tokenizer: tokenizer,
                textEncoder: textEncoder,
                debugLog: nil
            )
            negativePromptEmbeds = negEmbeds
        } else {
            negativePromptEmbeds = nil
        }
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))

        let negTxtIds: MLXArray? = {
            guard useCFG, let negEmbeds = negativePromptEmbeds else { return nil }
            return Flux2PosEmbed.prepareTextIds(seqLen: negEmbeds.shape[1], numAxes: 4)
        }()

        // 2. Prepare latent dimensions
        let seed = request.seed ?? UInt64.random(in: 0..<UInt64.max)
        let vaeScaleFactor = vae.configuration.vaeScaleFactor
        let latentHeight = request.height / vaeScaleFactor
        let latentWidth = request.width / vaeScaleFactor

        // FLUX.2 patchifies latents: 32 channels * 2 * 2 patch = 128 channels for transformer
        let patchedHeight = latentHeight / 2
        let patchedWidth = latentWidth / 2
        let seqLen = patchedHeight * patchedWidth

        // 2a. Encode reference images (if any)
        var referenceLatents: [MLXArray] = []
        let referenceImages = Array(request.referenceImages.prefix(4))
        let numRefs = referenceImages.count
        let refStrength = Float(request.referenceStrength)
        for (i, refURL) in referenceImages.enumerated() {
            progressHandler?(GenerationProgress(stage: .encodingReferenceImages, stepIndex: i + 1, totalSteps: numRefs))
            var refLatent = try encodeReferenceImage(
                refURL,
                vae: vae,
                width: request.width,
                height: request.height,
                patchedHeight: patchedHeight,
                patchedWidth: patchedWidth,
                bnMean: bnMean,
                bnVar: bnVar
            )

            // Add noise based on referenceStrength (0 = clean, 1 = pure noise)
            if refStrength > 0 {
                let refNoise = MLXRandom.normal(refLatent.shape, key: MLXRandom.key(seed &+ UInt64(i + 1))).asType(refLatent.dtype)
                // Blend: (1-strength)*clean + strength*noise
                refLatent = (1.0 - refStrength) * refLatent + refStrength * refNoise
            }

            referenceLatents.append(refLatent)
        }
        if numRefs > 0 {
            progressHandler?(GenerationProgress(stage: .encodingReferenceImages, stepIndex: numRefs, totalSteps: numRefs))
        }

        // 2b. Create noise latent for generated image
        // CRITICAL: Create noise in 4D spatial layout first, THEN pack to 3D
        // This preserves spatial correlation needed for coherent image structure
        let randomKey = MLXRandom.key(seed)
        let noise4D = MLXRandom.normal([1, 128, patchedHeight, patchedWidth], key: randomKey).asType(.bfloat16)
        // Pack: (B, C, H, W) -> (B, H*W, C)
        let noiseLatent = noise4D.reshaped(1, 128, seqLen).transposed(0, 2, 1)

        // 2c. Combine reference latents with noise latent for multi-reference editing
        // Layout: [ref1, ref2, ..., noise] in sequence dimension
        var latents: MLXArray
        if referenceLatents.isEmpty {
            latents = noiseLatent
        } else {
            let allLatents = referenceLatents + [noiseLatent]
            latents = MLX.concatenated(allLatents, axis: 1)  // [1, (numRefs+1)*seqLen, 128]
        }

        // 3. Prepare position IDs (2D: [seq_len, num_axes])
        let txtIds = Flux2PosEmbed.prepareTextIds(seqLen: promptEmbeds.shape[1], numAxes: 4)

        // For Klein Edit: each image gets unique t_coord in RoPE
        // Zero ordering: [ref1, ref2, ..., generated]
        // t_coords:      [10,   20,   ..., 0       ]
        let totalImages = numRefs + 1
        var tCoords = (0..<numRefs).map { 10 + 10 * $0 }
        tCoords.append(0)  // Generated image gets t_coord=0

        let imgIds = Flux2PosEmbed.prepareMultiImageIds(
            imageCount: totalImages,
            height: patchedHeight,
            width: patchedWidth,
            tCoords: tCoords
        )
        let machine = MereRunMachineProfile.current
        let useBatchedCFG = useCFG && negativePromptEmbeds.map { negativeEmbeds in
            let negativeIds = negTxtIds ?? Flux2PosEmbed.prepareTextIds(
                seqLen: negativeEmbeds.shape[1],
                numAxes: 4
            )
            return DiffusionCFGExecution.canPair(negativeEmbeds, promptEmbeds)
                && negativeIds.shape == txtIds.shape
                && DiffusionCFGExecution.shouldBatch(
                    mode: DiffusionCFGExecutionMode.current(
                        modelEnvironmentKey: "MERERUN_FLUX2_BATCHED_CFG"
                    ),
                    width: request.width,
                    height: request.height,
                    physicalMemoryBytes: machine.physicalMemoryBytes,
                    activeMemoryBytes: Memory.activeMemory,
                    cacheMemoryBytes: Memory.cacheMemory,
                    isUnifiedMemory: machine.isAppleSiliconMac,
                    baseReserveBytes: 6 * DiffusionCFGExecution.gibibyte,
                    activationBytesPerPixel: 4_096 * totalImages
                )
        } ?? false

        // 4. Setup scheduler (FLUX.2 Klein specific)
        // isDistilled is driven by manifest.variant (no heuristics).
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: request.steps,
            numTrainTimesteps: 1000,
            imageSeqLen: seqLen,
            isDistilled: isDistilled,
            sigmaShift: request.sigmaShift
        )

        // Note: FLUX.2 Klein does NOT scale initial latents (unlike SD3/etc)

        if let debugLog {
            debugLog("=== Scheduler Debug ===")
            debugLog("Sigmas: \((0...request.steps).map { scheduler.sigmas[$0].item(Float.self) })")
            debugLog("Initial latents (unscaled): mean=\(latents.mean().item(Float.self)), std=\(sqrt((latents * latents).mean().item(Float.self)))")
        }

        // 5. Denoising loop
        let denoiseStart = timingEnabled ? CFAbsoluteTimeGetCurrent() : 0
        for step in 0..<request.steps {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: step, totalSteps: request.steps))

            // mflux passes raw timestep (sigma * 1000), transformer handles scaling conditionally
            let timestepTensor = scheduler.timestep(at: step).expandedDimensions(axis: 0)

            // Run transformer (with CFG if enabled)
            let noisePred: MLXArray
            if useCFG, let negEmbeds = negativePromptEmbeds {
                if useBatchedCFG {
                    let predictions = transformerForward(
                        transformer,
                        hiddenStates: DiffusionCFGExecution.duplicateBatch(latents),
                        encoderHiddenStates: DiffusionCFGExecution.paired(negEmbeds, promptEmbeds),
                        timestep: DiffusionCFGExecution.duplicateBatch(timestepTensor),
                        imgIds: imgIds,
                        txtIds: txtIds
                    )
                    noisePred = DiffusionCFGExecution.combinePredictions(
                        predictions,
                        guidanceScale: Float(request.guidanceScale)
                    )
                } else {
                    // Unconditional pass (FLUX.2 Klein doesn't use guidance parameter)
                    let uncondNoise = transformerForward(
                        transformer,
                        hiddenStates: latents,
                        encoderHiddenStates: negEmbeds,
                        timestep: timestepTensor,
                        imgIds: imgIds,
                        txtIds: negTxtIds ?? Flux2PosEmbed.prepareTextIds(seqLen: negEmbeds.shape[1], numAxes: 4)
                    )

                    // Conditional pass (FLUX.2 Klein doesn't use guidance parameter)
                    let condNoise = transformerForward(
                        transformer,
                        hiddenStates: latents,
                        encoderHiddenStates: promptEmbeds,
                        timestep: timestepTensor,
                        imgIds: imgIds,
                        txtIds: txtIds
                    )

                    // CFG: output = uncond + guidance_scale * (cond - uncond)
                    let g = MLXArray(Float(request.guidanceScale))
                    noisePred = uncondNoise + g * (condNoise - uncondNoise)
                }
            } else {
                // No CFG - single pass (FLUX.2 Klein doesn't use guidance parameter)
                noisePred = transformerForward(
                    transformer,
                    hiddenStates: latents,
                    encoderHiddenStates: promptEmbeds,
                    timestep: timestepTensor,
                    imgIds: imgIds,
                    txtIds: txtIds
                )
            }

            if let debugLog {
                // Debug-only: the sigma readback lived outside this guard and
                // forced a GPU sync every denoise step in production runs.
                let sigmaValue = scheduler.sigma(at: step).item(Float.self)
                MLX.eval(noisePred)
                let npMean = noisePred.mean().item(Float.self)
                let npStd = sqrt((noisePred * noisePred).mean().item(Float.self))
                let npMin = noisePred.min().item(Float.self)
                let npMax = noisePred.max().item(Float.self)
                let timestepVal = timestepTensor.item(Float.self)
                debugLog("Step \(step): sigma=\(sigmaValue), timestep=\(timestepVal)")
                debugLog("  noise_pred: mean=\(npMean), std=\(npStd), min=\(npMin), max=\(npMax)")
            }

            // Euler step: only update the noise portion (last seqLen tokens)
            // Reference latents stay fixed - they're conditioning context
            if numRefs > 0 {
                // Extract only noise prediction for generated image (last seqLen tokens)
                let noiseOnlyPred = noisePred[0..., (numRefs * seqLen)..., 0...]
                // Extract current noise latent
                let currentNoiseLatent = latents[0..., (numRefs * seqLen)..., 0...]
                // Update only noise latent
                let updatedNoiseLatent = scheduler.step(modelOutput: noiseOnlyPred, timestepIndex: step, sample: currentNoiseLatent)
                // Recompose: keep reference latents fixed, update noise latent
                let refLatentsPart = latents[0..., 0..<(numRefs * seqLen), 0...]
                latents = MLX.concatenated([refLatentsPart, updatedNoiseLatent], axis: 1)
            } else {
                latents = scheduler.step(modelOutput: noisePred, timestepIndex: step, sample: latents)
            }
            if !Self.compileEnabled {
                MLX.eval(latents)
            }

            if let debugLog {
                let latMean = latents.mean().item(Float.self)
                let latStd = sqrt((latents * latents).mean().item(Float.self))
                debugLog("  latents: mean=\(latMean), std=\(latStd)")
            }
        }

        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: request.steps, totalSteps: request.steps))
        if timingEnabled {
            let denoiseEnd = CFAbsoluteTimeGetCurrent()
            let total = denoiseEnd - denoiseStart
            let perStepMs = (total / Double(max(1, request.steps))) * 1000.0
            let message = String(format: "[Flux2KleinGenerator] denoise_time_s=%.3f steps=%d step_ms=%.3f\n", total, request.steps, perStepMs)
            FileHandle.standardError.write(Data(message.utf8))
        }

        // 6. Extract only the noise latent (generated image) for decoding
        // If multi-reference, we take the last seqLen tokens
        let finalNoiseLatent: MLXArray
        if numRefs > 0 {
            finalNoiseLatent = latents[0..., (numRefs * seqLen)..., 0...]
        } else {
            finalNoiseLatent = latents
        }

        // 6a. Reshape to packed format: [batch, seq, 128] -> [batch, 128, height, width]
        let packedLatents = finalNoiseLatent
            .reshaped([1, patchedHeight, patchedWidth, 128])
            .transposed(0, 3, 1, 2)

        // 7. Apply BatchNorm inverse transform (denormalization)
        // mflux: latents = packed * sqrt(var + eps) + mean
        // mflux Flux2BatchNormStats uses eps = 0.0001 (1e-4), NOT PyTorch default 1e-5
        let denormalizedLatents = Flux2KleinBatchNorm.denormalizePackedLatents(
            packedLatents,
            mean: bnMean,
            variance: bnVar
        )

        if let debugLog {
            let bnStd = Flux2KleinBatchNorm.std(mean: bnMean, variance: bnVar)
            debugLog("=== BatchNorm Denormalization ===")
            debugLog("BN mean range: [\(bnMean.min().item(Float.self)), \(bnMean.max().item(Float.self))]")
            debugLog("BN std range: [\(bnStd.min().item(Float.self)), \(bnStd.max().item(Float.self))]")
            debugLog("Packed latents: mean=\(packedLatents.mean().item(Float.self)), std=\(sqrt((packedLatents * packedLatents).mean().item(Float.self)))")
            debugLog("Denormalized: mean=\(denormalizedLatents.mean().item(Float.self)), std=\(sqrt((denormalizedLatents * denormalizedLatents).mean().item(Float.self)))")
        }

        // 8. Unpatchify latents: [batch, 128, H, W] -> [batch, 32, H*2, W*2]
        let unpatchedLatents = unpatchifyPackedLatents(
            denormalizedLatents,
            height: patchedHeight,
            width: patchedWidth
        )

        if let debugLog {
            debugLog("Unpatchified: shape=\(unpatchedLatents.shape), mean=\(unpatchedLatents.mean().item(Float.self)), std=\(sqrt((unpatchedLatents * unpatchedLatents).mean().item(Float.self)))")
        }

        // 9. Decode with VAE
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))
        let decoded = vae.decode(unpatchedLatents)
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 1, totalSteps: 1))

        if let debugLog {
            let (decodedImg, _) = decoded
            debugLog("Decoded: shape=\(decodedImg.shape), mean=\(decodedImg.mean().item(Float.self)), std=\(sqrt((decodedImg * decodedImg).mean().item(Float.self))), min=\(decodedImg.min().item(Float.self)), max=\(decodedImg.max().item(Float.self))")
        }

        // 10. Save image with quality enhancement
        let outputURL = request.outputURL

        let outDir = outputURL.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: outDir.path) else {
            throw Flux2Error.invalidOutputDirectory(outputURL)
        }

        let (decodedImage, _) = decoded
        var image = QwenImageIO.denormalizeFromDecoder(decodedImage)

        // Enhance contrast and detail (subtle boost for better faces)
        // Apply mild contrast curve: 0.5 + (x - 0.5) * 1.1
        let contrastFactor: Float = 1.1
        image = 0.5 + (image - 0.5) * contrastFactor

        image = MLX.clip(image, min: 0, max: 1)
        try saveImage(image, to: outputURL)

        return GenerationResult(outputURL: outputURL, seed: seed)
    }


    // MARK: - Prompt Encoding

    private func encodePrompt(
        prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder,
        debugLog: ((String) -> Void)?
    ) throws -> (MLXArray, MLXArray) {
        // Tokenize using FLUX.2 chat template (matching mflux with enable_thinking=False)
        // Template (from `chat_template.jinja`): <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
        //
        // Important: we must NOT use `QwenTokenizer.encode(...)` here because it applies the
        // Z-Image system prefix/suffix (image-editing instructions), which breaks FLUX.2 prompts.
        let promptWithTemplate = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let encoded = tokenizer.encodePlain(prompts: [promptWithTemplate], maxLength: 512)

        if let debugLog {
            let tokens = encoded.inputIds[0].asArray(Int32.self)
            debugLog("Tokenized \(tokens.count) tokens: \(tokens.prefix(20))...")
        }

        // Get hidden states - diffusers uses indices 9, 18, 27 directly
        // These are the outputs of layers 8, 17, 26 (0-indexed from transformers convention)
        let result = textEncoder.forwardWithHiddenStates(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let hiddenStates = result.hiddenStates, hiddenStates.count >= 28 else {
            throw Flux2Error.insufficientHiddenStates
        }

        // Use direct indices 9, 18, 27 matching diffusers exactly.
        // No RMSNorm: diffusers uses raw hidden states.
        if let debugLog {
            debugLog("=== Hidden States Debug ===")
            debugLog("Total hidden states: \(hiddenStates.count)")
        }

        let h1 = hiddenStates[9]   // hidden_states[9]
        let h2 = hiddenStates[18]  // hidden_states[18]
        let h3 = hiddenStates[27]  // hidden_states[27]

        if let debugLog {
            debugLog("h1 (idx 9): shape=\(h1.shape), mean=\(h1.mean().item(Float.self)), std=\(sqrt((h1 * h1).mean().item(Float.self)))")
            debugLog("h2 (idx 18): shape=\(h2.shape), mean=\(h2.mean().item(Float.self)), std=\(sqrt((h2 * h2).mean().item(Float.self)))")
            debugLog("h3 (idx 27): shape=\(h3.shape), mean=\(h3.mean().item(Float.self)), std=\(sqrt((h3 * h3).mean().item(Float.self)))")
        }

        // Concatenate along feature dimension: [batch, seq, 2560*3 = 7680]
        let promptEmbeds = concatenated([h1, h2, h3], axis: -1)

        if let debugLog {
            let mean = promptEmbeds.mean().item(Float.self)
            let std = sqrt((promptEmbeds * promptEmbeds).mean().item(Float.self))
            debugLog("promptEmbeds: shape=\(promptEmbeds.shape), mean=\(mean), std=\(std)")
            debugLog("=== Expected (diffusers): mean=0.0593, std=29.47 ===")
        }

        // Pooled embedding: mean pool the last hidden state
        let lastHidden = result.lastHiddenState
        let pooledEmbeds = lastHidden.mean(axis: 1)  // [batch, hidden_size]

        return (promptEmbeds, pooledEmbeds)
    }

    // MARK: - Latent Utilities

    /// Unpatchify packed latents from [batch, 128, H, W] to [batch, 32, H*2, W*2]
    /// This matches mflux's _unpatchify_latents
    private func unpatchifyPackedLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        // Input: [batch, 128, height, width]
        // Output: [batch, 32, height*2, width*2]
        let batch = latents.shape[0]
        let numChannels = latents.shape[1]  // 128

        // Reshape: [batch, 128, H, W] -> [batch, 32, 2, 2, H, W]
        var x = latents.reshaped([batch, numChannels / 4, 2, 2, height, width])

        // Transpose to interleave the 2x2 patches: [batch, 32, H, 2, W, 2]
        x = x.transposed(0, 1, 4, 2, 5, 3)

        // Reshape to final: [batch, 32, H*2, W*2]
        x = x.reshaped([batch, numChannels / 4, height * 2, width * 2])

        return x
    }

    // MARK: - Reference Image Encoding

    /// Encode a reference image to patchified latent space for multi-reference editing
    /// Returns latent of shape [1, seqLen, 128] ready for concatenation
    private func encodeReferenceImage(
        _ url: URL,
        vae: AutoencoderKL,
        width: Int,
        height: Int,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Flux2Error.referenceImageNotFound(url)
        }

        let image: MediaImage
        do {
            image = try MediaImageIO.decode(url)
        } catch {
            throw Flux2Error.referenceImageDecodeFailed(url)
        }

        // 1. Load and resize image to target dimensions
        let resizedArray = try QwenImageIO.resizedPixelArray(
            from: image,
            width: width,
            height: height,
            addBatchDimension: true,
            dtype: .float32
        )

        // 2. Normalize for VAE encoder
        let normalized = QwenImageIO.normalizeForEncoder(resizedArray)

        // 3. Encode through VAE
        let encoded = vae.encode(normalized)  // [1, latentChannels*2, H/8, W/8] (mean+var)

        // 4. Extract mean (clean latent) - first half of channels
        let latentChannels = vae.configuration.latentChannels  // 32 for FLUX
        let mean = encoded[0..., 0..<latentChannels, 0..., 0...]  // [1, 32, H/8, W/8]

        // 5. Apply VAE scale/shift (FLUX.2 Klein uses 1.0/0.0)
        let cleanLatent = (mean - MLXArray(vae.configuration.shiftFactor)) * MLXArray(vae.configuration.scalingFactor)

        // 6. Patchify FIRST: [1, 32, H/8, W/8] -> [1, 128, H/16, W/16]
        // This converts to 128-channel packed format before BatchNorm
        let patchified = patchifyLatents(cleanLatent, height: patchedHeight * 2, width: patchedWidth * 2)

        // 7. Apply BatchNorm normalization (inverse of what we do for decode)
        // Decode does: packed * bnStd + bnMean
        // Encode should do: (packed - bnMean) / bnStd
        // mflux Flux2BatchNormStats uses eps = 0.0001 (1e-4)
        let normalizedPacked = Flux2KleinBatchNorm.normalizePackedLatents(
            patchified,
            mean: bnMean,
            variance: bnVar
        )

        // 8. Reshape to sequence format: [1, 128, H/16, W/16] -> [1, seqLen, 128]
        let seqLatent = normalizedPacked
            .transposed(0, 2, 3, 1)  // [1, H/16, W/16, 128]
            .reshaped([1, patchedHeight * patchedWidth, 128])
            .asType(.bfloat16)

        return seqLatent
    }

    /// Patchify latents from [batch, 32, H, W] to [batch, 128, H/2, W/2]
    /// This is the inverse of unpatchifyPackedLatents
    private func patchifyLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        // Input: [batch, 32, height, width]
        // Output: [batch, 128, height/2, width/2]
        let batch = latents.shape[0]
        let numChannels = latents.shape[1]  // 32

        // Reshape: [batch, 32, H, W] -> [batch, 32, H/2, 2, W/2, 2]
        var x = latents.reshaped([batch, numChannels, height / 2, 2, width / 2, 2])

        // Transpose to group patches: [batch, 32, 2, 2, H/2, W/2]
        x = x.transposed(0, 1, 3, 5, 2, 4)

        // Reshape to packed: [batch, 128, H/2, W/2]
        x = x.reshaped([batch, numChannels * 4, height / 2, width / 2])

        return x
    }

    // MARK: - Image Saving

    private func saveImage(_ tensor: MLXArray, to url: URL) throws {
        // Use QwenImageIO to save the decoded image
        try QwenImageIO.saveImage(array: tensor, to: url)
    }

}
