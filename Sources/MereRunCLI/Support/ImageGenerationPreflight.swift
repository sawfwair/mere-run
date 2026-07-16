import Foundation
import MereRunCore

struct ImageGenerationPreflightInput {
    let prompt: String
    let negativePrompt: String?
    let outputURL: URL
    let width: Int
    let height: Int
    let steps: Int?
    let seed: UInt64?
    let model: String?
    let input: String?
    let referenceImages: [String]
    let keepOriginalAspect: Bool
    let strength: Double?
    let cfgScale: Double?
    let sigmaShift: Double?
    let maxSequenceLength: Int
    let structuredPrompt: Bool
    let structuredPromptModel: String
    let structuredPromptModelRoot: String?
    let structuredPromptMaxTokens: Int
    let structuredPromptOutput: String?
    let lora: String?
    let loraScale: Double
    let kreaConditioningMultiplier: Double?
    let kreaConditioningLayerWeights: String?
    let kreaBaseQuantizationBits: Int?
    let generationArgv: [String]
    let cwd: String
}

struct ImageGenerationPreflightRequest: Codable, Equatable {
    let prompt: String
    let negativePrompt: String?
    let output: String
    let model: String
    let width: Int
    let height: Int
    let steps: Int?
    let seed: UInt64?
    let input: String?
    let referenceImages: [String]
    let keepOriginalAspect: Bool
    let strength: Double?
    let cfgScale: Double?
    let sigmaShift: Double?
    let maxSequenceLength: Int
    let structuredPrompt: Bool
    let structuredPromptModel: String
    let structuredPromptModelRoot: String?
    let structuredPromptMaxTokens: Int
    let structuredPromptOutput: String?
    let lora: String?
    let loraScale: Double
    let kreaConditioningMultiplier: Double?
    let kreaConditioningLayerWeights: String?
    let kreaBaseQuantizationBits: Int?

    enum CodingKeys: String, CodingKey {
        case prompt
        case negativePrompt = "negative_prompt"
        case output
        case model
        case width
        case height
        case steps
        case seed
        case input
        case referenceImages = "reference_images"
        case keepOriginalAspect = "keep_original_aspect"
        case strength
        case cfgScale = "cfg_scale"
        case sigmaShift = "sigma_shift"
        case maxSequenceLength = "max_sequence_length"
        case structuredPrompt = "structured_prompt"
        case structuredPromptModel = "structured_prompt_model"
        case structuredPromptModelRoot = "structured_prompt_model_root"
        case structuredPromptMaxTokens = "structured_prompt_max_tokens"
        case structuredPromptOutput = "structured_prompt_output"
        case lora
        case loraScale = "lora_scale"
        case kreaConditioningMultiplier = "krea_conditioning_multiplier"
        case kreaConditioningLayerWeights = "krea_conditioning_layer_weights"
        case kreaBaseQuantizationBits = "krea_base_quantization_bits"
    }
}

struct ImageGenerationPreflightResult: Codable, Equatable {
    let model: ImageGenerationModelPreflightSummary
    let output: ImageGenerationOutputPreflightSummary
    let inputs: ImageGenerationInputPreflightSummary
    let lora: ImageGenerationLoRAPreflightSummary?
    let structuredPrompt: ImageGenerationStructuredPromptPreflightSummary
    let plan: ImageGenerationPlanPreflightSummary
    let runPlan: ImageGenerationRunPlan

    enum CodingKeys: String, CodingKey {
        case model
        case output
        case inputs
        case lora
        case structuredPrompt = "structured_prompt"
        case plan
        case runPlan = "run_plan"
    }
}

struct ImageGenerationModelPreflightSummary: Codable, Equatable {
    let requested: String
    let kind: String
    let installed: Bool
    let path: String?
    let id: String?
    let family: String?
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case requested
        case kind
        case installed
        case path
        case id
        case family
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
    }
}

struct ImageGenerationOutputPreflightSummary: Codable, Equatable {
    let path: String
    let parentDirectory: String
    let parentExists: Bool
    let parentWillBeCreated: Bool
    let exists: Bool
    let expectedExtension: String
    let extensionValid: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case parentDirectory = "parent_directory"
        case parentExists = "parent_exists"
        case parentWillBeCreated = "parent_will_be_created"
        case exists
        case expectedExtension = "expected_extension"
        case extensionValid = "extension_valid"
    }
}

struct ImageGenerationInputPreflightSummary: Codable, Equatable {
    let inputImage: ImageGenerationPathPreflightSummary?
    let referenceImages: [ImageGenerationPathPreflightSummary]
    let missingCount: Int

    enum CodingKeys: String, CodingKey {
        case inputImage = "input_image"
        case referenceImages = "reference_images"
        case missingCount = "missing_count"
    }
}

struct ImageGenerationPathPreflightSummary: Codable, Equatable {
    let requested: String
    let path: String
    let exists: Bool
    let isDirectory: Bool

    enum CodingKeys: String, CodingKey {
        case requested
        case path
        case exists
        case isDirectory = "is_directory"
    }
}

struct ImageGenerationLoRAPreflightSummary: Codable, Equatable {
    let requested: String
    let path: String
    let exists: Bool
    let isDirectory: Bool
    let scale: Double

    enum CodingKeys: String, CodingKey {
        case requested
        case path
        case exists
        case isDirectory = "is_directory"
        case scale
    }
}

struct ImageGenerationStructuredPromptPreflightSummary: Codable, Equatable {
    let enabled: Bool
    let model: String
    let backend: String
    let modelRoot: ImageGenerationPathPreflightSummary?
    let maxTokens: Int
    let output: ImageGenerationOutputPreflightSummary?
    let fallbackAvailable: Bool

    enum CodingKeys: String, CodingKey {
        case enabled
        case model
        case backend
        case modelRoot = "model_root"
        case maxTokens = "max_tokens"
        case output
        case fallbackAvailable = "fallback_available"
    }
}

struct ImageGenerationPlanPreflightSummary: Codable, Equatable {
    let family: String?
    let width: Int
    let height: Int
    let requestedSteps: Int?
    let effectiveSteps: Int?
    let requestedCFGScale: Double?
    let effectiveCFGScale: Double?
    let requestedSigmaShift: Double?
    let effectiveSigmaShift: Double?
    let maxSequenceLength: Int
    let effectiveMaxSequenceLength: Int
    let inputMode: String

    enum CodingKeys: String, CodingKey {
        case family
        case width
        case height
        case requestedSteps = "requested_steps"
        case effectiveSteps = "effective_steps"
        case requestedCFGScale = "requested_cfg_scale"
        case effectiveCFGScale = "effective_cfg_scale"
        case requestedSigmaShift = "requested_sigma_shift"
        case effectiveSigmaShift = "effective_sigma_shift"
        case maxSequenceLength = "max_sequence_length"
        case effectiveMaxSequenceLength = "effective_max_sequence_length"
        case inputMode = "input_mode"
    }
}

typealias ImageGenerationPreflightEnvelope = StructuredRunEnvelope<
    ImageGenerationPreflightRequest,
    ImageGenerationPreflightResult
>

struct ImageGenerationPreflightAnalyzer {
    let input: ImageGenerationPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: ImageGenerationPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    func envelope() -> ImageGenerationPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let createdAt = now()
        validatePrompt(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let output = outputSummary(
            for: input.outputURL,
            expectedExtension: "png",
            unexpectedExtensionDiagnosticID: "output_extension_unusual",
            unexpectedExtensionTitle: "Output extension is not PNG",
            unexpectedExtensionMessage: "Image generation writes PNG data; use a .png output path for clarity.",
            diagnostics: &diagnostics
        )
        let inputs = inputSummary(diagnostics: &diagnostics)
        let lora = loraSummary(diagnostics: &diagnostics)
        let structuredPrompt = structuredPromptSummary(diagnostics: &diagnostics)
        let plan = planSummary(model: model)
        let runPlan = runPlan(resolved: plan, createdAt: createdAt)
        let status = StructuredRunOutput.status(for: diagnostics)
        let actions = actions(status: status, model: model, output: output, inputs: inputs, lora: lora)

        return ImageGenerationPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["image", "generate"],
            mode: .preflight,
            status: status,
            createdAt: createdAt,
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(),
            result: ImageGenerationPreflightResult(
                model: model,
                output: output,
                inputs: inputs,
                lora: lora,
                structuredPrompt: structuredPrompt,
                plan: plan,
                runPlan: runPlan
            ),
            diagnostics: diagnostics,
            actions: actions
        )
    }

    private func request() -> ImageGenerationPreflightRequest {
        ImageGenerationPreflightRequest(
            prompt: input.prompt,
            negativePrompt: input.negativePrompt,
            output: input.outputURL.path,
            model: requestedModel,
            width: input.width,
            height: input.height,
            steps: input.steps,
            seed: input.seed,
            input: input.input,
            referenceImages: input.referenceImages,
            keepOriginalAspect: input.keepOriginalAspect,
            strength: input.strength,
            cfgScale: input.cfgScale,
            sigmaShift: input.sigmaShift,
            maxSequenceLength: input.maxSequenceLength,
            structuredPrompt: input.structuredPrompt,
            structuredPromptModel: input.structuredPromptModel,
            structuredPromptModelRoot: input.structuredPromptModelRoot,
            structuredPromptMaxTokens: input.structuredPromptMaxTokens,
            structuredPromptOutput: input.structuredPromptOutput,
            lora: input.lora,
            loraScale: input.loraScale,
            kreaConditioningMultiplier: input.kreaConditioningMultiplier,
            kreaConditioningLayerWeights: input.kreaConditioningLayerWeights,
            kreaBaseQuantizationBits: input.kreaBaseQuantizationBits
        )
    }

    private var requestedModel: String {
        input.model ?? ImageGenerate.defaultManagedModelID.rawValue
    }

    private func validatePrompt(diagnostics: inout [PreflightDiagnostic]) {
        if input.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "prompt_empty",
                    severity: .blocker,
                    title: "Prompt is empty",
                    message: "--prompt must include non-whitespace text."
                )
            )
        }
    }

    private func modelSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> ImageGenerationModelPreflightSummary {
        let requested = requestedModel
        let localURL = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: localURL.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "model_path_not_directory",
                        severity: .blocker,
                        title: "Model path is not a directory",
                        message: "Model path is not a directory: \(localURL.path)",
                        locations: [.init(kind: "file", path: localURL.path)]
                    )
                )
                return modelResult(requested: requested, kind: "local_path", installed: false, path: localURL.path)
            }
            let manifest = loadManifest(at: localURL, diagnostics: &diagnostics)
            return modelResult(
                requested: requested,
                kind: "local_path",
                installed: manifest != nil,
                path: localURL.path,
                id: manifest?.id,
                family: manifest?.family?.rawValue
            )
        }

        if let modelID = ModelResolver.ModelID(rawValue: requested) {
            let spec = ManagedModelCatalog.spec(for: modelID.rawValue)
            if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
                let manifest = loadManifest(at: resolution.rootURL, diagnostics: &diagnostics)
                return modelResult(
                    requested: requested,
                    kind: "managed_model",
                    installed: manifest != nil,
                    path: resolution.rootURL.path,
                    id: manifest?.id,
                    family: manifest?.family?.rawValue,
                    upstreamRepoID: spec?.upstreamRepoId,
                    estimatedDownloadBytes: spec?.estimatedDownloadBytes
                )
            }

            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_missing",
                    severity: .blocker,
                    title: "Model missing",
                    message: "Model \(modelID.rawValue) is not installed. Pull it before generation.",
                    suggestedActionIDs: ["pull-model"]
                )
            )
            return modelResult(
                requested: requested,
                kind: "managed_model",
                installed: false,
                upstreamRepoID: spec?.upstreamRepoId,
                estimatedDownloadBytes: spec?.estimatedDownloadBytes
            )
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: "model_unknown",
                severity: .blocker,
                title: "Unknown model",
                message: "Model path not found and not a known model id: \(requested)."
            )
        )
        return modelResult(requested: requested, kind: "unknown", installed: false)
    }

    private func outputSummary(
        for outputURL: URL,
        expectedExtension: String,
        unexpectedExtensionDiagnosticID: String,
        unexpectedExtensionTitle: String,
        unexpectedExtensionMessage: String,
        diagnostics: inout [PreflightDiagnostic]
    ) -> ImageGenerationOutputPreflightSummary {
        let parent = outputURL.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        let outputExists = fileManager.fileExists(atPath: outputURL.path)
        let extensionValid = outputURL.pathExtension.lowercased() == expectedExtension

        if parentExists, !parentIsDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_parent_not_directory",
                    severity: .blocker,
                    title: "Output parent is not a directory",
                    message: "Output parent is not a directory: \(parent.path)",
                    locations: [.init(kind: "file", path: parent.path)]
                )
            )
        }
        if outputExists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_exists",
                    severity: .warning,
                    title: "Output exists",
                    message: "Output already exists and may be overwritten: \(outputURL.path)",
                    locations: [.init(kind: "file", path: outputURL.path)]
                )
            )
        }
        if !extensionValid {
            diagnostics.append(
                PreflightDiagnostic(
                    id: unexpectedExtensionDiagnosticID,
                    severity: .warning,
                    title: unexpectedExtensionTitle,
                    message: unexpectedExtensionMessage,
                    locations: [.init(kind: "file", path: outputURL.path)]
                )
            )
        }

        return ImageGenerationOutputPreflightSummary(
            path: outputURL.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: outputExists,
            expectedExtension: expectedExtension,
            extensionValid: extensionValid
        )
    }

    private func inputSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> ImageGenerationInputPreflightSummary {
        let inputImage = input.input.map { pathSummary(requested: $0) }
        let referenceImages = input.referenceImages.map { pathSummary(requested: $0) }
        let allInputs = [inputImage].compactMap { $0 } + referenceImages

        for item in allInputs {
            if !item.exists {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "input_image_missing",
                        severity: .blocker,
                        title: "Input image missing",
                        message: "Input image not found: \(item.path)",
                        locations: [.init(kind: "file", path: item.path)]
                    )
                )
            } else if item.isDirectory {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "input_image_is_directory",
                        severity: .blocker,
                        title: "Input image is a directory",
                        message: "Input image path is a directory: \(item.path)",
                        locations: [.init(kind: "directory", path: item.path)]
                    )
                )
            }
        }

        return ImageGenerationInputPreflightSummary(
            inputImage: inputImage,
            referenceImages: referenceImages,
            missingCount: allInputs.filter { !$0.exists }.count
        )
    }

    private func loraSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> ImageGenerationLoRAPreflightSummary? {
        guard let lora = input.lora else { return nil }
        let summary = pathSummary(requested: lora)
        if !summary.exists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "lora_missing",
                    severity: .blocker,
                    title: "LoRA missing",
                    message: "LoRA file not found: \(summary.path)",
                    locations: [.init(kind: "file", path: summary.path)]
                )
            )
        } else if summary.isDirectory {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "lora_is_directory",
                    severity: .blocker,
                    title: "LoRA path is a directory",
                    message: "LoRA path is a directory: \(summary.path)",
                    locations: [.init(kind: "directory", path: summary.path)]
                )
            )
        }
        return ImageGenerationLoRAPreflightSummary(
            requested: lora,
            path: summary.path,
            exists: summary.exists,
            isDirectory: summary.isDirectory,
            scale: input.loraScale
        )
    }

    private func structuredPromptSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> ImageGenerationStructuredPromptPreflightSummary {
        let modelRoot = input.structuredPromptModelRoot.map { pathSummary(requested: $0) }
        let output = input.structuredPromptOutput.map {
            outputSummary(
                for: URL(fileURLWithPath: $0).standardizedFileURL,
                expectedExtension: "json",
                unexpectedExtensionDiagnosticID: "structured_prompt_output_extension_unusual",
                unexpectedExtensionTitle: "Structured prompt output extension is not JSON",
                unexpectedExtensionMessage: "Structured prompt output writes JSON; use a .json path for clarity.",
                diagnostics: &diagnostics
            )
        }
        if input.structuredPrompt, input.structuredPromptMaxTokens < 1 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "structured_prompt_max_tokens_invalid",
                    severity: .blocker,
                    title: "Structured prompt token limit is invalid",
                    message: "--structured-prompt-max-tokens must be >= 1."
                )
            )
        }
        if input.structuredPrompt, let modelRoot {
            if !modelRoot.exists {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "structured_prompt_model_root_missing",
                        severity: .blocker,
                        title: "Structured prompt model root missing",
                        message: "Structured prompt model root not found: \(modelRoot.path)",
                        locations: [.init(kind: "directory", path: modelRoot.path)]
                    )
                )
            } else if !modelRoot.isDirectory {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "structured_prompt_model_root_not_directory",
                        severity: .blocker,
                        title: "Structured prompt model root is not a directory",
                        message: "Structured prompt model root is not a directory: \(modelRoot.path)",
                        locations: [.init(kind: "file", path: modelRoot.path)]
                    )
                )
            }
        }
        return ImageGenerationStructuredPromptPreflightSummary(
            enabled: input.structuredPrompt,
            model: input.structuredPromptModel,
            backend: StructuredImagePromptAdapter.backendDescription(
                for: input.structuredPromptModel,
                includeDefaultDevice: false
            ),
            modelRoot: modelRoot,
            maxTokens: input.structuredPromptMaxTokens,
            output: output,
            fallbackAvailable: true
        )
    }

    private func planSummary(
        model: ImageGenerationModelPreflightSummary
    ) -> ImageGenerationPlanPreflightSummary {
        let family = model.family.flatMap(MereRunModelManifest.Family.init(rawValue:))
        let effectiveSteps = effectiveSteps(for: family, modelPath: model.path)
        let effectiveCFG = effectiveCFGScale(for: family, modelPath: model.path)
        let effectiveSigma = input.sigmaShift ?? manifestDefaultSigmaShift(modelPath: model.path)
        let effectiveMaxSequenceLength = input.structuredPrompt
            ? max(input.maxSequenceLength, StructuredImagePromptAdapter.recommendedImagePromptTokens)
            : input.maxSequenceLength

        return ImageGenerationPlanPreflightSummary(
            family: model.family,
            width: input.width,
            height: input.height,
            requestedSteps: input.steps,
            effectiveSteps: effectiveSteps,
            requestedCFGScale: input.cfgScale,
            effectiveCFGScale: effectiveCFG,
            requestedSigmaShift: input.sigmaShift,
            effectiveSigmaShift: effectiveSigma,
            maxSequenceLength: input.maxSequenceLength,
            effectiveMaxSequenceLength: effectiveMaxSequenceLength,
            inputMode: inputMode(family: family)
        )
    }

    private func runPlan(
        resolved: ImageGenerationPlanPreflightSummary,
        createdAt: Date
    ) -> ImageGenerationRunPlan {
        ImageGenerationRunPlan(
            schemaVersion: ImageGenerationRunPlan.currentSchemaVersion,
            kind: ImageGenerationRunPlan.kind,
            command: ["image", "generate"],
            createdAt: createdAt,
            cwd: input.cwd,
            arguments: ImageGenerationRunPlanArguments(
                prompt: input.prompt,
                negativePrompt: input.negativePrompt,
                output: input.outputURL.path,
                model: requestedModel,
                width: input.width,
                height: input.height,
                steps: input.steps,
                seed: input.seed,
                input: input.input,
                referenceImages: input.referenceImages,
                keepOriginalAspect: input.keepOriginalAspect,
                strength: input.strength,
                cfgScale: input.cfgScale,
                sigmaShift: input.sigmaShift,
                maxSequenceLength: input.maxSequenceLength,
                structuredPrompt: input.structuredPrompt,
                structuredPromptModel: input.structuredPromptModel,
                structuredPromptModelRoot: input.structuredPromptModelRoot,
                structuredPromptMaxTokens: input.structuredPromptMaxTokens,
                structuredPromptOutput: input.structuredPromptOutput,
                lora: input.lora,
                loraScale: input.loraScale,
                kreaConditioningMultiplier: input.kreaConditioningMultiplier,
                kreaConditioningLayerWeights: input.kreaConditioningLayerWeights,
                kreaBaseQuantizationBits: input.kreaBaseQuantizationBits,
                quiet: input.generationArgv.contains("--quiet")
            ),
            resolved: resolved
        )
    }

    private func actions(
        status: StructuredRunStatus,
        model: ImageGenerationModelPreflightSummary,
        output: ImageGenerationOutputPreflightSummary,
        inputs: ImageGenerationInputPreflightSummary,
        lora: ImageGenerationLoRAPreflightSummary?
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        let blocked = status == .blocked
        actions.append(
            DeclarativeAction(
                id: "start-generation",
                label: "Start generation",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.generationArgv,
                    cwd: input.cwd,
                    commandPath: ["image", "generate"]
                ),
                requires: ["preflight.passed"]
            )
        )

        if model.kind == "managed_model", !model.installed {
            actions.append(
                DeclarativeAction(
                    id: "pull-model",
                    label: "Pull model",
                    kind: .command,
                    style: .secondary,
                    command: DeclarativeCommand(
                        argv: ["mere.run", "model", "pull", model.requested],
                        cwd: input.cwd,
                        commandPath: ["model", "pull"]
                    )
                )
            )
        }

        actions.append(
            DeclarativeAction(
                id: "open-output-directory",
                label: "Open output directory",
                kind: .openDirectory,
                style: .link,
                enabled: output.parentExists,
                disabledReason: output.parentExists ? nil : "Output directory will be created when generation starts.",
                path: output.parentDirectory
            )
        )

        for action in inputActions(inputs: inputs) {
            actions.append(action)
        }

        if let lora {
            actions.append(
                DeclarativeAction(
                    id: "reveal-lora",
                    label: "Reveal LoRA",
                    kind: .revealFile,
                    style: .link,
                    enabled: lora.exists && !lora.isDirectory,
                    path: lora.path
                )
            )
        }

        return actions
    }

    private func inputActions(inputs: ImageGenerationInputPreflightSummary) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        if let inputImage = inputs.inputImage {
            actions.append(
                DeclarativeAction(
                    id: "open-input-image",
                    label: "Open input image",
                    kind: .openFile,
                    style: .link,
                    enabled: inputImage.exists && !inputImage.isDirectory,
                    path: inputImage.path
                )
            )
        }
        for (index, reference) in inputs.referenceImages.enumerated() {
            actions.append(
                DeclarativeAction(
                    id: "open-reference-image-\(index + 1)",
                    label: "Open reference image \(index + 1)",
                    kind: .openFile,
                    style: .link,
                    enabled: reference.exists && !reference.isDirectory,
                    path: reference.path
                )
            )
        }
        return actions
    }

    private func summary(status: StructuredRunStatus, diagnostics: [PreflightDiagnostic]) -> String {
        let blockerCount = diagnostics.filter { $0.severity == .blocker }.count
        let warningCount = diagnostics.filter { $0.severity == .warning }.count
        switch status {
        case .blocked:
            return "\(blockerCount) blocker(s), \(warningCount) warning(s), generation blocked."
        case .warning:
            return "\(warningCount) warning(s), ready to generate."
        default:
            return "Ready to generate."
        }
    }

    private func pathSummary(requested: String) -> ImageGenerationPathPreflightSummary {
        let url = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return ImageGenerationPathPreflightSummary(
            requested: requested,
            path: url.path,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue
        )
    }

    private func modelResult(
        requested: String,
        kind: String,
        installed: Bool,
        path: String? = nil,
        id: String? = nil,
        family: String? = nil,
        upstreamRepoID: String? = nil,
        estimatedDownloadBytes: Int64? = nil
    ) -> ImageGenerationModelPreflightSummary {
        ImageGenerationModelPreflightSummary(
            requested: requested,
            kind: kind,
            installed: installed,
            path: path,
            id: id,
            family: family,
            upstreamRepoID: upstreamRepoID,
            estimatedDownloadBytes: estimatedDownloadBytes
        )
    }

    private func loadManifest(
        at modelRoot: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> MereRunModelManifest? {
        do {
            let manifest = try MereRunModelManifest.loadRequired(from: modelRoot, fileManager: fileManager)
            if !Self.supportedImageFamilies.contains(manifest.family) {
                let family = manifest.family?.rawValue ?? "unknown"
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "model_family_unsupported",
                        severity: .blocker,
                        title: "Unsupported model family",
                        message: "Unsupported image generation model family: \(family).",
                        locations: [.init(kind: "directory", path: modelRoot.path)]
                    )
                )
            }
            return manifest
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_manifest_unreadable",
                    severity: .blocker,
                    title: "Model manifest unreadable",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: modelRoot.path)]
                )
            )
            return nil
        }
    }

    private func effectiveSteps(for family: MereRunModelManifest.Family?, modelPath: String?) -> Int? {
        if let steps = input.steps { return steps }
        guard let family else { return nil }
        if Self.familyUsesManifestDefaults(family) {
            return manifestDefaults(modelPath: modelPath)?.steps ?? 4
        }
        return 4
    }

    private func effectiveCFGScale(for family: MereRunModelManifest.Family?, modelPath: String?) -> Double? {
        if let cfgScale = input.cfgScale { return cfgScale }
        guard let family else { return nil }
        if Self.familyUsesManifestDefaults(family) {
            return manifestDefaults(modelPath: modelPath)?.cfg ?? 1.0
        }
        return 1.0
    }

    private func manifestDefaultSigmaShift(modelPath: String?) -> Double? {
        guard let modelPath else { return nil }
        return manifestDefaults(modelPath: modelPath)?.sigmaShift
    }

    private func manifestDefaults(modelPath: String?) -> MereRunModelManifest.Defaults? {
        guard let modelPath else { return nil }
        return try? MereRunModelManifest
            .loadRequired(from: URL(fileURLWithPath: modelPath), fileManager: fileManager)
            .defaults
    }

    private func inputMode(family: MereRunModelManifest.Family?) -> String {
        let hasInput = input.input != nil
        let hasReferences = !input.referenceImages.isEmpty
        guard hasInput || hasReferences else { return "text_to_image" }
        if family == .klein {
            return "reference_image"
        }
        if hasInput && hasReferences {
            return "image_to_image_with_references"
        }
        if hasInput {
            return "image_to_image"
        }
        return "reference_image"
    }

    private static func familyUsesManifestDefaults(_ family: MereRunModelManifest.Family) -> Bool {
        family == .hidream || family == .krea || family == .ideogram
    }

    private static let supportedImageFamilies: Set<MereRunModelManifest.Family?> = [
        .klein,
        .zimage,
        .hidream,
        .krea,
        .ideogram,
    ]
}
