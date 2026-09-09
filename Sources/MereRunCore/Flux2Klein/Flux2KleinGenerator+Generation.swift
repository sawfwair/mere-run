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

        try await applyLoRAIfNeeded(request.loras, progressHandler: progressHandler)

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
        let usesEmbeddedGuidance = transformer.config.guidanceEmbeds
        if let sigmas = request.sigmas {
            try Flux2EulerScheduler.validateCustomSigmas(sigmas, expectedSteps: request.steps)
        }

        // CFG behavior differs between distilled and base models:
        // - Distilled: CFG only if user explicitly provides negative prompt (mflux behavior)
        // - Base: CFG required when guidance > 1.0, uses empty string as unconditional
        let useCFG: Bool
        if usesEmbeddedGuidance {
            useCFG = false
        } else if isDistilled {
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
        let seed = ImageGenerationSeed.resolve(request.seed, prompt: request.prompt, backend: .flux2Klein)
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
            usesEmpiricalMu: usesEmbeddedGuidance,
            sigmaShift: request.sigmaShift,
            customSigmas: request.sigmas
        )

        // Note: FLUX.2 Klein does NOT scale initial latents (unlike SD3/etc)

        if let debugLog {
            debugLog("=== Scheduler Debug ===")
            debugLog("Sigmas: \((0...request.steps).map { scheduler.sigmas[$0].item(Float.self) })")
            debugLog("Initial latents (unscaled): mean=\(latents.mean().item(Float.self)), std=\(sqrt((latents * latents).mean().item(Float.self)))")
        }

        latents = try denoise(
            latents: latents, transformer: transformer, scheduler: scheduler,
            steps: request.steps, guidanceScale: Float(request.guidanceScale),
            promptEmbeds: promptEmbeds, negativePromptEmbeds: negativePromptEmbeds,
            txtIds: txtIds, negTxtIds: negTxtIds, imgIds: imgIds,
            numRefs: numRefs, seqLen: seqLen, useCFG: useCFG,
            useBatchedCFG: useBatchedCFG, usesEmbeddedGuidance: usesEmbeddedGuidance,
            timingEnabled: timingEnabled, debugLog: debugLog, progressHandler: progressHandler
        )

        try decodeAndSave(
            latents: latents, vae: vae, numRefs: numRefs, seqLen: seqLen,
            patchedHeight: patchedHeight, patchedWidth: patchedWidth,
            bnMean: bnMean, bnVar: bnVar, outputURL: request.outputURL,
            debugLog: debugLog, progressHandler: progressHandler
        )

        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }


}
