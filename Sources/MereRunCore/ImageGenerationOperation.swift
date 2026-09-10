import Foundation

public struct ImageGenerationOutcome: Sendable {
    public let id: UUID
    public let result: GenerationResult
    public let modelID: String
    public let backend: ImageGenerationBackend
    /// Includes the actual seed and durable input paths, never temporary edit files.
    public let effectiveRequest: GenerationRequest
}

public enum ImageGenerationEvent: Sendable {
    case started(UUID)
    case progress(UUID, GenerationProgress)
    case succeeded(ImageGenerationOutcome)
    case failed(UUID, ImageGenerationIssue)
    case cancelled(UUID)
}

/// Owns the image operation's preparation, execution, and terminal result. The
/// executor owns admission and runtime residency; this layer never acquires a
/// second machine permit or unloads a runtime borrowed from a resident pool.
public enum ImageGenerationOperation {
    public typealias Executor = @Sendable (
        ImageGenerationBackend, GenerationRequest, (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult

    public static func execute(
        _ plan: ImageGenerationPlan,
        id: UUID = UUID(),
        recording: ImageRunSession? = nil,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil,
        eventHandler: (@Sendable (ImageGenerationEvent) -> Void)? = nil,
        executor: Executor = generate
    ) async throws -> ImageGenerationOutcome {
        let id = recording?.id ?? id
        eventHandler?(.started(id))
        let progress: (@Sendable (GenerationProgress) -> Void)?
        if progressHandler != nil || eventHandler != nil {
            progress = { value in
                progressHandler?(value)
                eventHandler?(.progress(id, value))
            }
        } else {
            progress = nil
        }
        do {
            let effectivePlan = try recording?.prepare(plan) ?? plan
            let outcome = try await perform(effectivePlan, id: id, progressHandler: progress, executor: executor)
            try recording?.succeed(outcome)
            eventHandler?(.succeeded(outcome))
            return outcome
        } catch {
            if error is CancellationError || Task.isCancelled {
                try recording?.fail(CancellationError())
                eventHandler?(.cancelled(id))
                throw CancellationError()
            }
            try recording?.fail(error)
            let issue = error as? ImageGenerationIssue
                ?? ImageGenerationIssue("generation_failed", error.localizedDescription)
            eventHandler?(.failed(id, issue))
            throw error
        }
    }

    private static func perform(
        _ plan: ImageGenerationPlan,
        id: UUID,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?,
        executor: Executor
    ) async throws -> ImageGenerationOutcome {
        try Task.checkCancellation()
        // A preflight plan is observational. Recheck mutable inputs and adapters
        // at execution; the owning runtime checks checkpoint integrity on load.
        try ImageGenerationPlan.validateFiles(plan.options, fileManager: .default)
        var request = plan.request
        request.loras = try ImageGenerationPlan.resolveLoRAs(plan.options.loras, baseModelID: plan.manifest.id)
        try FileManager.default.createDirectory(at: request.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let preparation: ImageEditPreparation?
        if plan.options.mask != nil || plan.options.outpaint != nil {
            guard let input = plan.options.inputImage else {
                throw ImageGenerationIssue("edit_input_missing", "Mask and outpaint require an input image.")
            }
            preparation = try ImageEditPreparation.make(
                inputURL: input, maskURL: plan.options.mask, outpaint: plan.options.outpaint,
                width: request.width, height: request.height, featherPixels: plan.options.maskFeather
            )
        } else {
            preparation = nil
        }
        defer { preparation?.cleanup() }
        if let preparation {
            let conditioning = ImageGenerationConditioning.resolve(
                family: plan.manifest.family, inputImage: preparation.generationInputURL,
                referenceImages: plan.options.referenceImages, strength: plan.options.strength, policy: plan.policy
            )
            request.inputImage = conditioning.inputImage
            request.referenceImages = conditioning.referenceImages
            request.strength = conditioning.strength
            request.referenceStrength = conditioning.referenceStrength
        }
        try Task.checkCancellation()
        let result = try await executor(plan.backend, request, progressHandler)
        try Task.checkCancellation()
        try preparation?.finish(generatedURL: result.outputURL)
        try Task.checkCancellation()
        var effective = plan.request
        effective.loras = request.loras
        effective.seed = result.seed
        return ImageGenerationOutcome(id: id, result: result, modelID: plan.manifest.id, backend: plan.backend, effectiveRequest: effective)
    }

    public static func generate(
        backend: ImageGenerationBackend,
        request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        switch backend {
        case .flux1:
            let generator = Flux1Generator()
            return try await withCleanup {
                try await generator.generate(request, progressHandler: progressHandler)
            } cleanup: { await generator.unload() }
        case .flux2Klein:
            let generator = Flux2KleinGenerator()
            return try await withCleanup {
                try await generator.generate(request, progressHandler: progressHandler)
            } cleanup: { await generator.unload() }
        case .zImageTurbo:
            let generator = ZImageTurboGenerator()
            return try await withCleanup {
                try await generator.generate(request, progressHandler: progressHandler)
            } cleanup: { await generator.unload() }
        case .hiDreamO1:
            let generator = HiDreamO1Generator()
            defer { generator.unload() }
            return try await generator.generate(request, progressHandler: progressHandler)
        case .senseNovaU15:
            let generator = SenseNovaU15Generator()
            defer { generator.unload() }
            return try await generator.generate(request, progressHandler: progressHandler)
        case .krea2:
            let generator = Krea2Generator()
            defer { generator.unload() }
            return try await generator.generate(request, progressHandler: progressHandler)
        case .ideogram4:
            let generator = Ideogram4Generator()
            defer { generator.unload() }
            return try await generator.generate(request, progressHandler: progressHandler)
        case .qwenImageEdit:
            let generator = QwenImageEditGenerator()
            return try await withCleanup {
                try await generator.generate(request, progressHandler: progressHandler)
            } cleanup: { await generator.clearCache() }
        }
    }

    private static func withCleanup(
        operation: () async throws -> GenerationResult,
        cleanup: () async -> Void
    ) async throws -> GenerationResult {
        do {
            let result = try await operation()
            await cleanup()
            return result
        } catch {
            await cleanup()
            throw error
        }
    }
}
