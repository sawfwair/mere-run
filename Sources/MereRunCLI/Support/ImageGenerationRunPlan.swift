import ArgumentParser
import Foundation

struct ImageGenerationRunPlan: Codable, Equatable {
    static let currentSchemaVersion = 1
    static let kind = "image.generate"

    let schemaVersion: Int
    let kind: String
    let command: [String]
    let createdAt: Date
    let cwd: String
    let arguments: ImageGenerationRunPlanArguments
    let resolved: ImageGenerationPlanPreflightSummary

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case command
        case createdAt = "created_at"
        case cwd
        case arguments
        case resolved
    }

    func validateExecutable() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw ValidationError("Unsupported image generate plan schema_version \(schemaVersion).")
        }
        guard kind == Self.kind else {
            throw ValidationError("Unsupported image run plan kind '\(kind)'.")
        }
        guard command == ["image", "generate"] else {
            throw ValidationError("Unsupported image run plan command: \(command.joined(separator: " ")).")
        }
    }

    static func decode(from url: URL) throws -> ImageGenerationRunPlan {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let plan = try decoder.decode(ImageGenerationRunPlan.self, from: data)
        try plan.validateExecutable()
        return plan
    }

    func relocatingOutput(to output: String) -> ImageGenerationRunPlan {
        relocatingOutputs(output: output, structuredPromptOutput: arguments.structuredPromptOutput)
    }

    func relocatingOutputs(
        output: String,
        structuredPromptOutput: String?
    ) -> ImageGenerationRunPlan {
        ImageGenerationRunPlan(
            schemaVersion: schemaVersion,
            kind: kind,
            command: command,
            createdAt: createdAt,
            cwd: cwd,
            arguments: arguments.relocatingOutputs(
                output: output,
                structuredPromptOutput: structuredPromptOutput
            ),
            resolved: resolved
        )
    }
}

struct ImageGenerationRunPlanArguments: Codable, Equatable {
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
    let sigmaList: String?
    let maxSequenceLength: Int
    let structuredPrompt: Bool
    let structuredPromptModel: String
    let structuredPromptModelRoot: String?
    let structuredPromptMaxTokens: Int
    let structuredPromptOutput: String?
    let lora: String?
    let loras: [String]?
    let loraScale: Double
    let kreaConditioningMultiplier: Double?
    let kreaConditioningLayerWeights: String?
    let kreaBaseQuantizationBits: Int?
    let quiet: Bool

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
        case sigmaList = "sigmas"
        case maxSequenceLength = "max_sequence_length"
        case structuredPrompt = "structured_prompt"
        case structuredPromptModel = "structured_prompt_model"
        case structuredPromptModelRoot = "structured_prompt_model_root"
        case structuredPromptMaxTokens = "structured_prompt_max_tokens"
        case structuredPromptOutput = "structured_prompt_output"
        case lora
        case loras
        case loraScale = "lora_scale"
        case kreaConditioningMultiplier = "krea_conditioning_multiplier"
        case kreaConditioningLayerWeights = "krea_conditioning_layer_weights"
        case kreaBaseQuantizationBits = "krea_base_quantization_bits"
        case quiet
    }

    func executableArgv() -> [String] {
        ["mere.run", "image", "generate"] + generateArguments()
    }

    func relocatingOutput(to output: String) -> ImageGenerationRunPlanArguments {
        relocatingOutputs(output: output, structuredPromptOutput: structuredPromptOutput)
    }

    func relocatingOutputs(
        output: String,
        structuredPromptOutput: String?
    ) -> ImageGenerationRunPlanArguments {
        ImageGenerationRunPlanArguments(
            prompt: prompt,
            negativePrompt: negativePrompt,
            output: output,
            model: model,
            width: width,
            height: height,
            steps: steps,
            seed: seed,
            input: input,
            referenceImages: referenceImages,
            keepOriginalAspect: keepOriginalAspect,
            strength: strength,
            cfgScale: cfgScale,
            sigmaShift: sigmaShift,
            sigmaList: sigmaList,
            maxSequenceLength: maxSequenceLength,
            structuredPrompt: structuredPrompt,
            structuredPromptModel: structuredPromptModel,
            structuredPromptModelRoot: structuredPromptModelRoot,
            structuredPromptMaxTokens: structuredPromptMaxTokens,
            structuredPromptOutput: structuredPromptOutput,
            lora: lora,
            loras: loras,
            loraScale: loraScale,
            kreaConditioningMultiplier: kreaConditioningMultiplier,
            kreaConditioningLayerWeights: kreaConditioningLayerWeights,
            kreaBaseQuantizationBits: kreaBaseQuantizationBits,
            quiet: quiet
        )
    }

    func generateArguments() -> [String] {
        var args = [
            "--prompt", prompt,
            "--output", output,
            "--model", model,
            "--width", String(width),
            "--height", String(height),
        ]
        appendOption("--negative-prompt", negativePrompt, to: &args)
        appendOption("--cfg", cfgScale, to: &args)
        appendOption("--sigma-shift", sigmaShift, to: &args)
        appendOption("--sigmas", sigmaList, to: &args)
        appendOption("--steps", steps, to: &args)
        appendOption("--seed", seed, to: &args)
        appendOption("--input", input, to: &args)
        for referenceImage in referenceImages {
            args += ["--ref-image", referenceImage]
        }
        appendBoolFlag("--keep-original-aspect", when: keepOriginalAspect, to: &args)
        appendOption("--strength", strength, to: &args)
        args += ["--max-sequence-length", String(maxSequenceLength)]
        appendBoolFlag("--structured-prompt", when: structuredPrompt, to: &args)
        args += ["--structured-prompt-model", structuredPromptModel]
        appendOption("--structured-prompt-model-root", structuredPromptModelRoot, to: &args)
        args += ["--structured-prompt-max-tokens", String(structuredPromptMaxTokens)]
        appendOption("--structured-prompt-output", structuredPromptOutput, to: &args)
        let adapterArguments = loras ?? lora.map { [$0] } ?? []
        for adapter in adapterArguments {
            args += ["--lora", adapter]
        }
        args += ["--lora-scale", String(loraScale)]
        appendOption("--krea-conditioning-multiplier", kreaConditioningMultiplier, to: &args)
        appendOption("--krea-conditioning-layer-weights", kreaConditioningLayerWeights, to: &args)
        appendOption("--krea-base-quantization-bits", kreaBaseQuantizationBits, to: &args)
        appendBoolFlag("--quiet", when: quiet, to: &args)
        return args
    }

    private func appendBoolFlag(_ flag: String, when condition: Bool, to args: inout [String]) {
        if condition {
            args.append(flag)
        }
    }

    private func appendOption<T>(_ flag: String, _ value: T?, to args: inout [String]) {
        if let value {
            args += [flag, String(describing: value)]
        }
    }
}

struct ImageGenerationRunMaterializationRequest: Codable, Equatable {
    let planFile: String
    let runDirectory: String

    enum CodingKeys: String, CodingKey {
        case planFile = "plan_file"
        case runDirectory = "run_directory"
    }
}

struct ImageGenerationRunMaterializationResult: Codable, Equatable {
    let runDirectory: String
    let planPath: String
    let actionsPath: String
    let runManifestPath: String
    let eventsPath: String
    let outputPath: String
    let originalOutputPath: String
    let structuredPromptOutputPath: String?

    enum CodingKeys: String, CodingKey {
        case runDirectory = "run_directory"
        case planPath = "plan_path"
        case actionsPath = "actions_path"
        case runManifestPath = "run_manifest_path"
        case eventsPath = "events_path"
        case outputPath = "output_path"
        case originalOutputPath = "original_output_path"
        case structuredPromptOutputPath = "structured_prompt_output_path"
    }
}

typealias ImageGenerationRunMaterializationEnvelope = StructuredRunEnvelope<
    ImageGenerationRunMaterializationRequest,
    ImageGenerationRunMaterializationResult
>
