import Foundation

// MARK: - Vision templates

extension CommandCatalog {
    static let visionTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .visionInspect,
            category: .vision,
            title: "Inspect image",
            subtitle: "Ask a VLM about an image",
            systemImage: "eye",
            promptLabel: "Question",
            inputKind: .image,
            defaultPrompt: "Describe this image."
        ),
        CommandTemplate(
            id: .visionEmbed,
            category: .vision,
            title: "Embed text + image",
            subtitle: "Create shared retrieval vectors",
            systemImage: "point.3.connected.trianglepath.dotted",
            promptLabel: "Text (optional)",
            secondaryLabel: "Retrieval instruction",
            inputKind: .image,
            outputKind: .file("json"),
            defaultSecondaryText: "Represent the item for retrieval.",
            defaultModel: "vision-embed-qwen3-vl-2b"
        ),
        CommandTemplate(
            id: .visionCaption,
            category: .vision,
            title: "Caption",
            subtitle: "Write LoRA-friendly captions",
            systemImage: "text.bubble",
            promptLabel: "Instruction",
            inputKind: .image,
            outputKind: .directory,
            defaultPrompt: "Write a short, concrete caption describing the image for LoRA training. Avoid fluff."
        ),
        CommandTemplate(
            id: .visionOCR,
            category: .vision,
            title: "OCR",
            subtitle: "Extract text from images",
            systemImage: "doc.text.viewfinder",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-ocr-lighton"
        ),
        CommandTemplate(
            id: .visionGround,
            category: .vision,
            title: "Ground objects",
            subtitle: "Find prompted objects in an image",
            systemImage: "scope",
            promptLabel: "Query",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a person",
            defaultModel: "vision-ground-falcon-perception"
        ),
        CommandTemplate(
            id: .visionSegment,
            category: .vision,
            title: "Segment",
            subtitle: "Segment prompted objects",
            systemImage: "square.dashed",
            promptLabel: "Prompt",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionTrack,
            category: .vision,
            title: "Track video",
            subtitle: "Track prompted objects through a clip",
            systemImage: "point.topleft.down.curvedto.point.bottomright.up",
            promptLabel: "Prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionTrackLive,
            category: .vision,
            title: "Track camera",
            subtitle: "Capture then track from a camera",
            systemImage: "video.badge.waveform",
            promptLabel: "Prompt",
            outputKind: .file("mp4"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionFaceDetect,
            category: .vision,
            title: "Detect faces",
            subtitle: "Buffalo-L boxes, landmarks, and optional embeddings",
            systemImage: "face.dashed",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionFaceEmbed,
            category: .vision,
            title: "Embed face",
            subtitle: "Create a normalized ArcFace identity vector",
            systemImage: "person.crop.square",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionFaceCompare,
            category: .vision,
            title: "Compare faces",
            subtitle: "Cosine similarity between two selected faces",
            systemImage: "person.2",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionFaceBatch,
            category: .vision,
            title: "Batch face analysis",
            subtitle: "Warm-session detection and embeddings to JSONL",
            systemImage: "person.3.sequence",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionPose,
            category: .vision,
            title: "Pose landmarks",
            subtitle: "Native body, hand, and face landmarks",
            systemImage: "figure.stand",
            inputKind: .image
        ),
        CommandTemplate(
            id: .visionFlow,
            category: .vision,
            title: "Optical flow",
            subtitle: "Dense motion between two equal-size images",
            systemImage: "arrow.triangle.2.circlepath",
            inputKind: .image,
            outputKind: .file("flo")
        ),
        CommandTemplate(
            id: .visionDepthVideo,
            category: .vision,
            title: "Video depth",
            subtitle: "Temporally consistent native VDA-S depth",
            systemImage: "square.3.layers.3d",
            inputKind: .video,
            outputKind: .directory,
            defaultModel: "vision-depth-vda-small"
        ),
        CommandTemplate(
            id: .visionGeometry,
            category: .vision,
            title: "Metric geometry",
            subtitle: "MoGe-2 depth, normals, camera, and point cloud",
            systemImage: "rotate.3d",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-geometry-moge2-small"
        ),
        CommandTemplate(
            id: .visionGeometryMultiview,
            category: .vision,
            title: "Multi-view geometry",
            subtitle: "DA3 cameras, confidence, and colored point cloud",
            systemImage: "view.3d",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-geometry-da3-small"
        )
    ]
}

// MARK: - Vision arguments

extension CommandArguments {
    static func visionInspect(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionInspect
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.option(F.prompt, draft.prompt)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.maxTokens, String(draft.maxTokens))
        args.option(F.temperature, format(draft.temperature))
        args.option(F.topP, format(draft.topP))
        return args.arguments
    }

    static func visionEmbed(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionEmbed
        var args = ArgumentBuilder(F.self)
        if !draft.prompt.isBlank { args.option(F.text, draft.prompt) }
        if !draft.inputPath.isBlank { args.option(F.image, draft.inputPath) }
        if !draft.secondaryText.isBlank { args.option(F.instruction, draft.secondaryText) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.maxTokens, String(draft.maxTokens))
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        return args.arguments
    }

    static func visionCaption(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionCaption
        var args = ArgumentBuilder(F.self)
        if !draft.inputPath.isBlank { args.value(draft.inputPath) }
        args.values(pathList(draft.visionAdditionalInputs))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.outputPath.isBlank { args.option(F.outputDir, draft.outputPath) }
        if !draft.prompt.isBlank { args.option(F.prompt, draft.prompt) }
        if !draft.visionPromptFile.isBlank { args.option(F.promptFile, draft.visionPromptFile) }
        args.repeated(F.focus, lineList(draft.visionFocus))
        if !draft.visionTriggerToken.isBlank {
            args.option(F.triggerToken, draft.visionTriggerToken)
        }
        args.option(F.maxTokens, String(draft.maxTokens))
        args.option(F.temperature, format(draft.temperature))
        args.option(F.topP, format(draft.topP))
        return args.arguments
    }

    static func visionOCR(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionOCR
        var args = ArgumentBuilder(F.self)
        if !draft.inputPath.isBlank { args.value(draft.inputPath) }
        args.values(pathList(draft.visionAdditionalInputs))
        args.option(F.backend, draft.backend)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.outputPath.isBlank { args.option(F.outputDir, draft.outputPath) }
        args.option(F.maxTokens, String(draft.maxTokens))
        args.option(F.temperature, format(draft.temperature))
        if draft.all { args.flag(F.compare) }
        if !draft.visionGLMOCRCLI.isBlank { args.option(F.glmocrCli, draft.visionGLMOCRCLI) }
        if !draft.visionGLMConfig.isBlank { args.option(F.glmConfig, draft.visionGLMConfig) }
        args.option(F.infinityRuntime, draft.visionInfinityRuntime)
        args.option(F.infinityParserCli, draft.visionInfinityParserCLI)
        args.option(F.infinityModel, draft.visionInfinityModel)
        args.option(F.infinityBackend, draft.visionInfinityBackend)
        args.option(F.infinityAPIURL, draft.visionInfinityAPIURL)
        args.option(F.infinityAPIKey, draft.visionInfinityAPIKey)
        args.option(F.infinityTask, draft.visionInfinityTask)
        args.option(F.infinityOutputFormat, draft.visionInfinityOutputFormat)
        args.option(F.infinityBatchSize, String(draft.visionInfinityBatchSize))
        args.option(F.infinityMinPixels, String(draft.visionInfinityMinPixels))
        args.option(F.infinityMaxPixels, String(draft.visionInfinityMaxPixels))
        if !draft.visionInfinityPrompt.isBlank {
            args.option(F.infinityPrompt, draft.visionInfinityPrompt)
        }
        if !draft.visionInfinityModelCacheDirectory.isBlank {
            args.option(F.infinityModelCacheDir, draft.visionInfinityModelCacheDirectory)
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func visionGround(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionGround
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.repeated(F.query, lineList(draft.prompt))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        if !draft.visionMaskOutputDirectory.isBlank {
            args.option(F.maskOutputDir, draft.visionMaskOutputDirectory)
        }
        return args.arguments
    }

    static func visionSegment(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionSegment
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.repeated(F.prompt, lineList(draft.prompt))
        args.repeated(F.box, lineList(draft.visionBoxPrompts))
        args.repeated(F.point, lineList(draft.visionPointPrompts))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        if !draft.visionMaskOutputDirectory.isBlank {
            args.option(F.maskOutputDir, draft.visionMaskOutputDirectory)
        }
        args.option(F.threshold, format(draft.visionThreshold))
        args.option(F.resolution, String(draft.visionResolution))
        if draft.force { args.flag(F.showBoxes) }
        if draft.visionMultimask { args.flag(F.multimask) }
        return args.arguments
    }

    static func visionTrack(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionTrack
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.repeated(F.prompt, lineList(draft.prompt))
        args.repeated(F.box, lineList(draft.visionBoxPrompts))
        args.repeated(F.point, lineList(draft.visionPointPrompts))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        if !draft.visionMaskOutputDirectory.isBlank {
            args.option(F.maskOutputDir, draft.visionMaskOutputDirectory)
        }
        args.option(F.initFrame, String(draft.visionInitFrame))
        args.option(F.threshold, format(draft.visionThreshold))
        args.option(F.resolution, String(draft.visionResolution))
        if !draft.visionEndFrame.isBlank { args.option(F.endFrame, draft.visionEndFrame) }
        if draft.force { args.flag(F.showBoxes) }
        if draft.visionShowLabels { args.flag(F.showLabels) }
        if draft.preflight {
            args.flag(F.preflight)
            if draft.json { args.flag(F.json) }
        }
        return args.arguments
    }

    static func visionTrackLive(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionTrackLive
        var args = ArgumentBuilder(F.self)
        args.repeated(F.prompt, lineList(draft.prompt))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        args.option(F.camera, String(draft.visionCamera))
        args.option(F.durationSeconds, format(draft.durationSeconds))
        args.option(F.initFrame, String(draft.visionInitFrame))
        args.option(F.seedSearchFrames, String(draft.visionSeedSearchFrames))
        args.option(F.threshold, format(draft.visionThreshold))
        args.option(F.resolution, String(draft.visionResolution))
        if draft.force { args.flag(F.showBoxes) }
        if draft.visionShowLabels { args.flag(F.showLabels) }
        return args.arguments
    }

    static func visionFaceDetect(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionFaceDetect
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        appendFaceOptions(to: &args, draft: draft)
        if draft.visionMaxFaces > 0 { args.option(F.maxFaces, String(draft.visionMaxFaces)) }
        if draft.visionIncludeEmbeddings { args.flag(F.includeEmbeddings) }
        return args.arguments
    }

    static func visionFaceEmbed(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionFaceEmbed
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        appendFaceOptions(to: &args, draft: draft)
        if !draft.visionFaceIndex.isBlank { args.option(F.faceIndex, draft.visionFaceIndex) }
        return args.arguments
    }

    static func visionFaceCompare(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionFaceCompare
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.value(draft.visionSecondInputPath)
        appendFaceOptions(to: &args, draft: draft)
        if !draft.visionReferenceFaceIndex.isBlank {
            args.option(F.referenceFaceIndex, draft.visionReferenceFaceIndex)
        }
        if !draft.visionCandidateFaceIndex.isBlank {
            args.option(F.candidateFaceIndex, draft.visionCandidateFaceIndex)
        }
        return args.arguments
    }

    static func visionFaceBatch(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionFaceBatch
        var args = ArgumentBuilder(F.self)
        if !draft.inputPath.isBlank { args.value(draft.inputPath) }
        args.values(pathList(draft.visionAdditionalInputs))
        if !draft.visionInputList.isBlank { args.option(F.inputList, draft.visionInputList) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.scoreThreshold, format(draft.visionFaceScoreThreshold))
        args.option(F.executionProvider, draft.visionExecutionProvider)
        if draft.visionMaxFaces > 0 { args.option(F.maxFaces, String(draft.visionMaxFaces)) }
        if draft.visionIncludeEmbeddings { args.flag(F.includeEmbeddings) }
        if !draft.visionJSONLOutput.isBlank {
            args.option(F.jsonlOutput, draft.visionJSONLOutput)
        }
        if draft.visionFailFast { args.flag(F.failFast) }
        return args.arguments
    }

    static func visionPose(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionPose
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        if !draft.visionPoseBody { args.flag(F.noBody) }
        if !draft.visionPoseHands { args.flag(F.noHands) }
        if !draft.visionPoseFace { args.flag(F.noFace) }
        args.option(F.maxHands, String(draft.visionMaxHands))
        args.option(F.minimumConfidence, format(draft.visionMinimumConfidence))
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func visionFlow(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionFlow
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.value(draft.visionSecondInputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        args.option(F.accuracy, draft.visionFlowAccuracy)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func visionDepthVideo(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionDepthVideo
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.inputSize, String(draft.visionInputSize))
        args.option(F.maxFrames, String(draft.visionMaxFrames))
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func visionGeometry(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionGeometry
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.resolutionLevel, String(draft.visionResolutionLevel))
        if draft.visionTokenCount > 0 {
            args.option(F.tokenCount, String(draft.visionTokenCount))
        }
        if draft.visionMaxPoints > 0 {
            args.option(F.maxPoints, String(draft.visionMaxPoints))
        }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func visionGeometryMultiview(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionGeometryMultiview
        var args = ArgumentBuilder(F.self)
        if !draft.inputPath.isBlank { args.value(draft.inputPath) }
        args.values(pathList(draft.visionAdditionalInputs))
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.camerasPath.isBlank { args.option(F.cameras, draft.camerasPath) }
        args.option(F.processResolution, String(draft.visionProcessResolution))
        args.option(F.referenceView, draft.visionReferenceView)
        args.option(F.confidencePercentile, format(draft.visionConfidencePercentile))
        if draft.visionMaxPoints > 0 {
            args.option(F.maxPoints, String(draft.visionMaxPoints))
        }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }
}
