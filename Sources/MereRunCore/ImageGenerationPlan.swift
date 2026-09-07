import Foundation

/// Model-dependent sampling values. An observational plan for a missing model
/// leaves unknown defaults unset instead of inventing a runnable configuration.
public struct ImageGenerationSampling: Sendable, Equatable {
    public let steps: Int?
    public let guidanceScale: Double?
    public let sigmaShift: Float?
    public let sigmas: [Float]?

    public static func resolve(
        _ options: ImageGenerationOptions,
        manifest: MereRunModelManifest?,
        policy: ImageGenerationPolicy = .manifestDefaults
    ) -> Self {
        let turbo = options.loras.contains {
            ManagedAdapterCatalog.spec(for: $0.reference)?.id == ManagedAdapterCatalog.flux2DevTurboEightStepID
        }
        let sigmas = options.sigmas ?? (turbo ? Flux2DevTurboRecipe.sigmas : nil)
        let usesManifest: Bool
        switch manifest?.family {
        case .klein: usesManifest = policy.kleinUsesManifestDefaults
        case .hidream, .senseNova, .krea, .ideogram, .flux1: usesManifest = true
        case .qwen: usesManifest = manifest?.engine == .qwenImageEdit
        default: usesManifest = false
        }
        let defaultSteps = manifest == nil ? nil
            : (usesManifest ? manifest?.defaults?.steps : nil) ?? policy.fallbackSteps
        let defaultCFG = manifest == nil ? nil
            : (usesManifest ? manifest?.defaults?.cfg : nil) ?? policy.fallbackGuidanceScale
        return Self(
            steps: options.steps ?? sigmas?.count ?? defaultSteps,
            guidanceScale: options.guidanceScale ?? (turbo ? Flux2DevTurboRecipe.guidanceScale : nil) ?? defaultCFG,
            sigmaShift: options.sigmaShift ?? manifest?.defaults?.sigmaShift.map(Float.init),
            sigmas: sigmas
        )
    }

    public static func parseSigmas(_ raw: String?) throws -> [Float]? {
        guard let raw else { return nil }
        var values = try raw.split(separator: ",", omittingEmptySubsequences: false).map { item in
            let trimmed = item.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = Float(trimmed), value.isFinite else {
                throw ImageGenerationIssue("sigma_schedule_invalid", "--sigmas must contain finite comma-separated values.")
            }
            return value
        }
        if values.last == 0 { values.removeLast() }
        guard !values.isEmpty else {
            throw ImageGenerationIssue("sigma_schedule_invalid", "--sigmas must include at least one non-terminal value.")
        }
        do {
            try Flux2EulerScheduler.validateCustomSigmas(values, expectedSteps: values.count)
        } catch {
            throw ImageGenerationIssue("sigma_schedule_invalid", error.localizedDescription)
        }
        return values
    }
}

public struct ImageGenerationConditioning: Sendable, Equatable {
    public let inputImage: URL?
    public let referenceImages: [URL]
    public let strength: Double
    public let referenceStrength: Double

    public static func resolve(
        family: MereRunModelManifest.Family?,
        inputImage: URL?,
        referenceImages: [URL],
        strength: Double?,
        policy: ImageGenerationPolicy = .manifestDefaults
    ) -> Self {
        guard family == .klein, policy.kleinInputAsReference else {
            return Self(
                inputImage: inputImage, referenceImages: referenceImages,
                strength: strength ?? 0.75, referenceStrength: 0
            )
        }
        return Self(
            inputImage: nil,
            referenceImages: inputImage.map { [$0] + referenceImages } ?? referenceImages,
            strength: strength ?? 0.75,
            referenceStrength: strength ?? (inputImage == nil ? 0 : 0.75)
        )
    }

    public static func inputMode(
        family: MereRunModelManifest.Family?, inputImage: URL?, referenceImages: [URL]
    ) -> String {
        guard inputImage != nil || !referenceImages.isEmpty else { return "text_to_image" }
        if family == .klein { return "reference_image" }
        if inputImage != nil, !referenceImages.isEmpty { return "image_to_image_with_references" }
        return inputImage == nil ? "reference_image" : "image_to_image"
    }
}

/// A validated operation, independent of command parsing, HTTP, and runtime residency.
/// Resolving a plan reads local metadata and checks paths; it never downloads or loads weights.
public struct ImageGenerationPlan: Sendable {
    public let options: ImageGenerationOptions
    public let modelRoot: URL
    public let manifest: MereRunModelManifest
    public let backend: ImageGenerationBackend
    public let policy: ImageGenerationPolicy
    public let request: GenerationRequest

    public static func resolve(
        _ options: ImageGenerationOptions,
        modelRoot: URL,
        manifest: MereRunModelManifest,
        policy: ImageGenerationPolicy = .manifestDefaults,
        fileManager: FileManager = .default
    ) throws -> Self {
        if let issue = issues(options, manifest: manifest, policy: policy).first { throw issue }
        let backend = try ImageGenerationBackend(manifest: manifest)
        try validateFiles(options, fileManager: fileManager)
        let loras = try resolveLoRAs(options.loras, baseModelID: manifest.id, fileManager: fileManager)
        let sampling = ImageGenerationSampling.resolve(options, manifest: manifest, policy: policy)
        let conditioning = ImageGenerationConditioning.resolve(
            family: manifest.family, inputImage: options.inputImage,
            referenceImages: options.referenceImages, strength: options.strength, policy: policy
        )
        // A present, supported manifest always resolves both values.
        guard let steps = sampling.steps, let guidance = sampling.guidanceScale else {
            throw ImageGenerationIssue("model_defaults_missing", "Image model defaults could not be resolved.")
        }
        return Self(
            options: options, modelRoot: modelRoot.standardizedFileURL, manifest: manifest,
            backend: backend, policy: policy,
            request: GenerationRequest(
                prompt: options.prompt, negativePrompt: options.negativePrompt,
                referenceImages: conditioning.referenceImages, referenceStrength: conditioning.referenceStrength,
                width: options.width, height: options.height, steps: steps, guidanceScale: guidance,
                seed: options.seed, outputURL: options.outputURL, model: modelRoot.standardizedFileURL.path,
                maxSequenceLength: options.maxSequenceLength, loras: loras,
                inputImage: conditioning.inputImage, strength: conditioning.strength,
                keepOriginalAspect: options.keepOriginalAspect, sigmaShift: sampling.sigmaShift,
                sigmas: sampling.sigmas, kreaConditioningRebalance: options.kreaConditioningRebalance,
                kreaBaseQuantizationBits: options.kreaBaseQuantizationBits
            )
        )
    }

    /// Diagnostics are shared with preflight, including requests whose model is missing.
    public static func issues(
        _ options: ImageGenerationOptions,
        manifest: MereRunModelManifest? = nil,
        policy: ImageGenerationPolicy = .manifestDefaults
    ) -> [ImageGenerationIssue] {
        var issues: [ImageGenerationIssue] = []
        func reject(_ condition: Bool, _ code: String, _ message: String) {
            if condition { issues.append(ImageGenerationIssue(code, message)) }
        }
        reject(options.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "prompt_empty", "Prompt is empty.")
        reject(options.width <= 0 || options.height <= 0, "dimensions_invalid", "--width/--height must be > 0")
        reject(options.steps.map { $0 <= 0 } == true, "steps_invalid", "--steps must be >= 1")
        reject(options.guidanceScale?.isFinite == false, "cfg_invalid", "--cfg must be finite")
        reject(options.sigmaShift?.isFinite == false, "sigma_shift_invalid", "--sigma-shift must be finite")
        reject(options.strength.map { !(0...1).contains($0) } == true, "strength_invalid", "--strength must be between 0.0 and 1.0")
        reject(options.maxSequenceLength <= 0, "sequence_length_invalid", "--max-sequence-length must be > 0")
        reject(options.maskFeather < 0, "mask_feather_invalid", "--mask-feather must be >= 0")
        reject(
            (options.mask != nil || options.outpaint != nil) && options.inputImage == nil,
            "image_edit_input_missing", "--mask and --outpaint require --input"
        )
        reject(
            options.kreaBaseQuantizationBits.map { $0 != 4 && $0 != 8 } == true,
            "krea_quantization_invalid", "--krea-base-quantization-bits must be 4 or 8"
        )
        reject(options.loras.contains { !$0.scale.isFinite }, "lora_scale_invalid", "--lora scale must be finite")
        reject(
            options.sigmas != nil && options.sigmaShift != nil,
            "sigma_options_conflict", "--sigmas cannot be combined with --sigma-shift"
        )
        if let rebalance = options.kreaConditioningRebalance {
            reject(
                !rebalance.multiplier.isFinite || rebalance.layerWeights.contains { !$0.isFinite },
                "krea_conditioning_invalid", "Krea conditioning values must be finite."
            )
        }
        if options.width > 0, options.height > 0 {
            reject(options.width > Int.max / options.height / 4, "dimensions_invalid", "Image dimensions are too large.")
        }
        if let padding = options.outpaint {
            let edges = [padding.top, padding.right, padding.bottom, padding.left]
            reject(edges.contains { $0 < 0 } || !edges.contains { $0 > 0 }, "image_edit_invalid", "Outpaint padding is invalid.")
            // Compare each edge before subtracting to avoid overflow in untrusted dimensions.
            let fitsWidth = padding.left >= 0 && padding.left < options.width
                && padding.right >= 0 && padding.right < options.width - padding.left
            let fitsHeight = padding.top >= 0 && padding.top < options.height
                && padding.bottom >= 0 && padding.bottom < options.height - padding.top
            reject(!fitsWidth || !fitsHeight, "image_edit_invalid", "Outpaint padding must leave a positive source region.")
        }
        let sampling = ImageGenerationSampling.resolve(options, manifest: manifest, policy: policy)
        if let sigmas = sampling.sigmas {
            do {
                try Flux2EulerScheduler.validateCustomSigmas(sigmas, expectedSteps: sigmas.count)
            } catch {
                issues.append(ImageGenerationIssue("sigma_schedule_invalid", error.localizedDescription))
            }
            reject(
                sampling.steps != sigmas.count, "sigma_step_count_mismatch",
                "--steps must equal the number of non-terminal --sigmas values (\(sigmas.count))."
            )
        }
        reject(sampling.steps.map { $0 <= 0 } == true, "model_steps_invalid", "Effective image steps must be positive.")
        reject(sampling.guidanceScale?.isFinite == false, "model_cfg_invalid", "Effective image guidance must be finite.")
        reject(sampling.sigmaShift?.isFinite == false, "model_sigma_shift_invalid", "Effective sigma shift must be finite.")
        guard let manifest else { return issues }
        do { _ = try ImageGenerationBackend(manifest: manifest) } catch let issue as ImageGenerationIssue { issues.append(issue) } catch {
            issues.append(ImageGenerationIssue("model_family_unsupported", error.localizedDescription))
        }
        reject(
            options.loras.count > 1 && manifest.family != .klein && manifest.family != .flux1,
            "lora_stack_model_unsupported", "Stacked image LoRAs are supported only for FLUX.1 and FLUX.2 models."
        )
        reject(options.sigmas != nil && manifest.family != .klein, "sigma_model_unsupported", "--sigmas is supported only for FLUX.2 models.")
        reject(
            options.kreaBaseQuantizationBits != nil && manifest.family != .krea,
            "krea_quantization_unsupported", "--krea-base-quantization-bits is only supported for Krea 2 generation"
        )
        if [.flux1, .krea, .ideogram].contains(manifest.family) {
            reject(options.inputImage != nil, "input_mode_unsupported", "\(manifest.id) does not support image-to-image generation.")
            reject(!options.referenceImages.isEmpty, "references_unsupported", "\(manifest.id) does not support reference images.")
        }
        reject(
            manifest.family == .flux1 && options.negativePrompt?.isEmpty == false,
            "negative_prompt_unsupported", "FLUX.1-dev native generation does not support negative prompts."
        )
        reject(
            manifest.family == .zimage && !options.referenceImages.isEmpty,
            "references_unsupported", "ZImage does not support reference images; use an input image for image-to-image generation."
        )
        reject(
            !options.loras.isEmpty && ![.flux1, .klein, .zimage, .krea].contains(manifest.family),
            "lora_unsupported", "\(manifest.id) does not support image LoRA adapters."
        )
        reject(
            manifest.family == .senseNova && (!options.width.isMultiple(of: 32) || !options.height.isMultiple(of: 32)),
            "dimensions_invalid", "SenseNova image dimensions must be multiples of 32."
        )
        if manifest.engine == .qwenImageEdit {
            let count = Set(([options.inputImage].compactMap { $0 } + options.referenceImages).map { $0.standardizedFileURL }).count
            reject(count == 0, "edit_input_missing", "Qwen-Image-Edit requires at least one input or reference image.")
            reject(count > QwenImageEditConditioningPlan.maximumReferenceCount, "reference_count_invalid", "Qwen-Image-Edit supports up to 3 ordered images.")
        }
        return issues
    }

    public static func resolveLoRAs(
        _ arguments: [ImageLoRAReference], baseModelID: String, fileManager: FileManager = .default
    ) throws -> [LoRA] {
        var resolvedPaths = Set<String>()
        return try arguments.map { argument in
            guard argument.scale.isFinite else { throw ImageGenerationIssue("lora_scale_invalid", "LoRA scale must be finite.") }
            let resolved = try ManagedAdapterArgumentResolver.resolve(
                argument.reference, baseModelID: baseModelID, fileManager: fileManager
            ) ?? argument.reference
            let url = URL(fileURLWithPath: resolved).standardizedFileURL
            try validateFile(url, label: "LoRA file", fileManager: fileManager)
            guard resolvedPaths.insert(url.resolvingSymlinksInPath().path).inserted else {
                throw ImageGenerationIssue("lora_duplicate", "Duplicate LoRA adapter: \(url.path)")
            }
            return .local(path: url.path, scale: argument.scale)
        }
    }

    static func validateFiles(_ options: ImageGenerationOptions, fileManager: FileManager) throws {
        if let url = options.inputImage { try validateFile(url, label: "Input image", fileManager: fileManager) }
        if let url = options.mask { try validateFile(url, label: "Mask image", fileManager: fileManager) }
        for url in options.referenceImages { try validateFile(url, label: "Reference image", fileManager: fileManager) }
    }

    private static func validateFile(_ url: URL, label: String, fileManager: FileManager) throws {
        var directory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &directory) else {
            throw ImageGenerationIssue("input_missing", "\(label) not found: \(url.path)")
        }
        guard !directory.boolValue else {
            throw ImageGenerationIssue("input_not_file", "\(label) is a directory: \(url.path)")
        }
    }
}
