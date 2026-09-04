import Foundation

// MARK: - Image templates

extension CommandCatalog {
    package static let imageTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .imageGenerate,
            category: .image,
            title: "Generate or edit",
            subtitle: "Text, image, multi-reference, structured prompt, and LoRA",
            systemImage: "photo",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a ceramic coffee mug in soft morning light",
            defaultModel: "image-zimage-nano"
        ),
        CommandTemplate(
            id: .imageTrainLoRA,
            category: .image,
            title: "Train LoRA",
            subtitle: "Krea 2 and FLUX.2 Klein recipes, previews, and dashboards",
            systemImage: "slider.horizontal.3",
            inputKind: .directory,
            outputKind: .file("safetensors"),
            defaultModel: "image-krea2-raw"
        ),
        CommandTemplate(
            id: .imageValidate,
            category: .image,
            title: "Validate image stack",
            subtitle: "Run deterministic image runtime checks",
            systemImage: "checkmark.seal",
            outputKind: .directory
        ),
        CommandTemplate(
            id: .imageDatasetDiscover,
            category: .image,
            title: "Discover datasets",
            subtitle: "Find trainable image-caption folders",
            systemImage: "folder.badge.questionmark",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .imageRunPlan,
            category: .image,
            title: "Run workflow plan",
            subtitle: "Preflight, materialize, or execute a saved image plan",
            systemImage: "list.bullet.clipboard",
            inputKind: .file([.json])
        ),
        CommandTemplate(
            id: .imageVisualizeRun,
            category: .image,
            title: "Training dashboard",
            subtitle: "Open a durable LoRA run viewer",
            systemImage: "chart.xyaxis.line",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .imageReconstruct3D,
            category: .image,
            title: "TripoSR 3D",
            subtitle: "Reconstruct a colored mesh from one image",
            systemImage: "cube.transparent",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "image-3d-triposr"
        ),
        CommandTemplate(
            id: .imageReconstruct3DTrellis2,
            category: .image,
            title: "TRELLIS.2 PBR 3D",
            subtitle: "Build a 512-resolution PBR O-Voxel asset",
            systemImage: "cube.fill",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "image-3d-trellis2-4b"
        ),
        CommandTemplate(
            id: .imageReconstruct3DMultiview,
            category: .image,
            title: "InstantMesh multiview",
            subtitle: "Reconstruct from four or six ordered views",
            systemImage: "square.3.layers.3d",
            outputKind: .directory,
            defaultModel: "image-3d-instantmesh-base"
        )
    ]
}

// MARK: - Image arguments

extension CommandArguments {
    package static func imageGenerate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageGenerate
        var args = ArgumentBuilder(F.self)
        args.option(F.prompt, draft.prompt)
        args.option(F.output, draft.outputPath)
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        args.option(F.steps, String(draft.steps))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.secondaryText.isBlank { args.option(F.negativePrompt, draft.secondaryText) }
        if draft.cfgScale != 1.0 { args.option(F.cfg, format(draft.cfgScale)) }
        if draft.sigmaShift > 0 { args.option(F.sigmaShift, format(draft.sigmaShift)) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if !draft.inputPath.isBlank {
            args.option(F.input, draft.inputPath)
            args.option(F.strength, format(draft.strength))
        }
        if let mask = draft.imageMaskPath, !mask.isBlank { args.option(F.mask, mask) }
        if let outpaint = draft.imageOutpaint, !outpaint.isBlank { args.option(F.outpaint, outpaint) }
        if let feather = draft.imageMaskFeather {
            args.optionUnlessDefault(F.maskFeather, String(feather))
        }
        args.repeated(F.refImage, pathList(draft.referenceImagePaths))
        if draft.keepOriginalAspect { args.flag(F.keepOriginalAspect) }
        if draft.inputPath.isBlank, !draft.referenceImagePaths.isBlank, draft.strength != 0 {
            args.option(F.strength, format(draft.strength))
        }
        args.optionUnlessDefault(F.maxSequenceLength, String(draft.maxSequenceLength))
        if draft.structuredPrompt {
            args.flag(F.structuredPrompt)
            if !draft.structuredPromptModel.isBlank {
                args.option(F.structuredPromptModel, draft.structuredPromptModel)
            }
            if !draft.structuredPromptModelRoot.isBlank {
                args.option(F.structuredPromptModelRoot, draft.structuredPromptModelRoot)
            }
            args.option(F.structuredPromptMaxTokens, String(draft.structuredPromptMaxTokens))
            if !draft.structuredPromptOutputPath.isBlank {
                args.option(F.structuredPromptOutput, draft.structuredPromptOutputPath)
            }
        }
        if !draft.loraPath.isBlank {
            args.option(F.lora, draft.loraPath)
            args.option(F.loraScale, format(draft.loraScale))
        }
        if draft.kreaConditioningMultiplier > 0 {
            args.option(F.kreaConditioningMultiplier, format(draft.kreaConditioningMultiplier))
        }
        if !draft.kreaConditioningLayerWeights.isBlank {
            args.option(F.kreaConditioningLayerWeights, draft.kreaConditioningLayerWeights)
        }
        if !draft.kreaBaseQuantizationBits.isBlank {
            args.option(F.kreaBaseQuantizationBits, draft.kreaBaseQuantizationBits)
        }
        if draft.preflight { args.flag(F.preflight) }
        if draft.preflight, draft.json { args.flag(F.json) }
        if draft.progressJSON { args.flag(F.progressJSON) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func imageTrainLoRA(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageTrainLoRA
        var args = ArgumentBuilder(F.self)
        args.option(F.output, draft.outputPath)
        if !draft.inputPath.isBlank { args.option(F.data, draft.inputPath) }
        if !draft.trainingRecipe.isBlank { args.option(F.recipe, draft.trainingRecipe) }
        let emitsRecipeOverrides = draft.trainingRecipe.isBlank || draft.overrideTrainingRecipe
        if emitsRecipeOverrides {
            args.option(F.width, String(draft.width))
            args.option(F.height, String(draft.height))
            args.option(F.trainingSteps, String(draft.steps))
            if !draft.model.isBlank { args.option(F.model, draft.model) }
            args.option(F.learningRate, format(draft.learningRate))
            args.option(F.rank, String(draft.rank))
            if draft.alpha > 0 { args.option(F.alpha, format(draft.alpha)) }
            if draft.captionDropout > 0 {
                args.option(F.captionDropout, format(draft.captionDropout))
            }
        }
        args.option(F.batchSize, String(draft.batchSize))
        args.option(F.maxTextLength, String(draft.maxSequenceLength))
        args.option(F.schedulerSteps, String(draft.schedulerSteps))
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.trainingLite { args.flag(F.lite) }
        if !draft.baseQuantizationBits.isBlank {
            args.option(F.baseQuantizationBits, draft.baseQuantizationBits)
        }
        if draft.excludePreviewImages { args.flag(F.excludePreviewImages) }
        if emitsRecipeOverrides, draft.checkpointInterval > 0 {
            args.option(F.checkpointInterval, String(draft.checkpointInterval))
        }
        if let resumePath = draft.trainingResumePath, !resumePath.isBlank {
            args.option(F.resumeFrom, resumePath)
        }
        if emitsRecipeOverrides, draft.maxResolution > 0 {
            args.option(F.maxResolution, String(draft.maxResolution))
        }
        if draft.progressive { args.flag(F.progressive) }
        if emitsRecipeOverrides, draft.lowRAM { args.flag(F.lowRam) }
        if emitsRecipeOverrides, draft.disableCompile { args.flag(F.noCompile) }
        if draft.gradientCheckpointing { args.flag(F.gradientCheckpointing) }
        if draft.benchmarkSteps > 0 { args.option(F.benchmarkSteps, String(draft.benchmarkSteps)) }
        if draft.benchmarkSteps > 0 {
            args.option(F.benchmarkWarmupSteps, String(draft.benchmarkWarmupSteps))
        }
        if draft.sampleInterval > 0 {
            args.option(F.sampleInterval, String(draft.sampleInterval))
            if !draft.samplePrompt.isBlank { args.option(F.samplePrompt, draft.samplePrompt) }
            if !draft.sampleModel.isBlank { args.option(F.sampleModel, draft.sampleModel) }
            args.option(F.sampleSteps, String(draft.sampleSteps))
            args.option(F.sampleCfg, format(draft.sampleCFG))
            args.option(F.sampleLoRAScale, format(draft.sampleLoRAScale))
            if !draft.sampleSeed.isBlank { args.option(F.sampleSeed, draft.sampleSeed) }
        }
        if draft.visualize {
            args.flag(F.visualize)
            args.option(F.visualizePort, String(draft.visualizePort))
        }
        if draft.preflight { args.flag(F.preflight) }
        if draft.preflight, draft.json { args.flag(F.json) }
        if !draft.loraTargetRanks.isBlank { args.option(F.loraTargetRanks, draft.loraTargetRanks) }
        if !draft.loraRankPreset.isBlank { args.option(F.loraRankPreset, draft.loraRankPreset) }
        if emitsRecipeOverrides, !draft.loraTargetPreset.isBlank {
            args.option(F.loraTargetPreset, draft.loraTargetPreset)
        }
        if !draft.loraTargetMode.isBlank { args.option(F.loraTargetMode, draft.loraTargetMode) }
        if !draft.timestepSampling.isBlank { args.option(F.timestepSampling, draft.timestepSampling) }
        if !draft.timestepLossWeighting.isBlank {
            args.option(F.timestepLossWeighting, draft.timestepLossWeighting)
        }
        if !draft.lossWeighting.isBlank { args.option(F.lossWeighting, draft.lossWeighting) }
        if draft.timestepLow > 0 { args.option(F.timestepLow, String(draft.timestepLow)) }
        if draft.timestepHigh > 0 { args.option(F.timestepHigh, String(draft.timestepHigh)) }
        if emitsRecipeOverrides, draft.lrWarmupSteps > 0 {
            args.option(F.lrWarmupSteps, String(draft.lrWarmupSteps))
        }
        if draft.disableCosineScheduler { args.flag(F.noCosineScheduler) }
        if emitsRecipeOverrides, draft.lrMinFactor > 0 {
            args.option(F.lrMinFactor, format(draft.lrMinFactor))
        }
        if draft.adamWeightDecay > 0 {
            args.option(F.adamWeightDecay, format(draft.adamWeightDecay))
        }
        if draft.syntheticSamples > 0 { args.option(F.syntheticSamples, String(draft.syntheticSamples)) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func imageValidate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageValidate
        var args = ArgumentBuilder(F.self)
        args.option(F.test, draft.backend)
        args.option(F.family, draft.variant)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.force { args.flag(F.saveReference) }
        if draft.all { args.flag(F.compare) }
        if !draft.referenceDirectoryPath.isBlank {
            args.option(F.referenceDir, draft.referenceDirectoryPath)
        }
        return args.arguments
    }

    package static func imageDatasetDiscover(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageDatasetDiscover
        var args = ArgumentBuilder(F.self)
        args.option(F.root, draft.inputPath)
        args.option(F.maxDepth, String(draft.maxDepth))
        args.option(F.minUsablePairs, String(draft.minUsablePairs))
        if !draft.trainingOutputRoot.isBlank {
            args.option(F.trainingOutputRoot, draft.trainingOutputRoot)
        }
        if !draft.trainingModel.isBlank { args.option(F.trainingModel, draft.trainingModel) }
        if !draft.trainingRecipe.isBlank { args.option(F.trainingRecipe, draft.trainingRecipe) }
        if draft.excludePreviewImages { args.flag(F.excludePreviewImages) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func imageRunPlan(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageRunPlan
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if draft.preflight { args.flag(F.preflight) }
        if !draft.materializePath.isBlank { args.option(F.materialize, draft.materializePath) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func imageVisualizeRun(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageVisualizeRun
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.option(F.port, String(draft.port))
        return args.arguments
    }

    package static func imageReconstruct3D(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageReconstruct3D
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.resolution, String(draft.reconstructionResolution))
        args.option(F.densityThreshold, format(draft.densityThreshold))
        args.option(F.foregroundRatio, format(draft.foregroundRatio))
        if draft.alreadyFramed { args.flag(F.alreadyFramed) }
        if draft.noVertexColors { args.flag(F.noVertexColors) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func imageReconstruct3DTrellis2(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageReconstruct3DTrellis2
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if let textureSeed = draft.trellisTextureSeed, !textureSeed.isBlank {
            args.option(F.textureSeed, textureSeed)
        }
        args.option(F.maxTokens, String(draft.maxTokens))
        if draft.alreadyFramed { args.flag(F.alreadyFramed) }
        if draft.trellisNoRemesh == true { args.flag(F.noRemesh) }
        if let remeshBand = draft.trellisRemeshBand {
            args.option(F.remeshBand, format(remeshBand))
        }
        if let sealRadius = draft.trellisSealRadius {
            args.option(F.sealRadius, String(sealRadius))
        }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func imageReconstruct3DMultiview(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ImageReconstruct3DMultiview
        var args = ArgumentBuilder(F.self)
        args.repeated(F.view, pathList(draft.referenceImagePaths))
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.camerasPath.isBlank { args.option(F.cameras, draft.camerasPath) }
        args.option(F.resolution, String(draft.reconstructionResolution))
        if draft.noVertexColors { args.flag(F.noVertexColors) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }
}
