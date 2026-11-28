import Foundation
import MereRunCore

public enum Cosmos3SuperImageGeneratorError: LocalizedError, Sendable {
    case missingModelRoot
    case incompleteCheckpoint([String])

    public var errorDescription: String? {
        switch self {
        case .missingModelRoot:
            return "Cosmos3-Super image generation requires a resolved model root."
        case .incompleteCheckpoint(let issues):
            return "Cosmos3-Super text-to-image checkpoint is incomplete: \(issues.joined(separator: ", "))."
        }
    }
}

/// Runs the distilled Cosmos3-Super text-to-image checkpoint behind the image generation
/// contract, on the same runtime `video cosmos3 --mode text-to-image` uses. The checkpoint
/// supplies its four-step sigma schedule and runs at guidance 1, so a negative prompt has no
/// effect and is not passed on.
public final class Cosmos3SuperImageGenerator: Cosmos3SuperUnloadableImageGenerator, @unchecked Sendable {
    private let generator = Cosmos3EdgeGenerator()

    public init() {}

    public func unload() {
        generator.unload()
    }

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        guard let model = request.model else {
            throw Cosmos3SuperImageGeneratorError.missingModelRoot
        }
        let resources = Cosmos3Resources(rootURL: URL(fileURLWithPath: model).standardizedFileURL)
        let missing = resources.validateSuperTextToImage()
        guard missing.isEmpty else {
            throw Cosmos3SuperImageGeneratorError.incompleteCheckpoint(missing.map(\.path))
        }
        let issues = try resources.superTextToImageValidationIssues()
        guard issues.isEmpty else {
            throw Cosmos3SuperImageGeneratorError.incompleteCheckpoint(issues)
        }
        let options = try Cosmos3GenerationOptions(
            prompt: request.prompt,
            outputURL: request.outputURL,
            width: request.width,
            height: request.height,
            numFrames: 1,
            steps: request.steps,
            guidanceScale: Float(request.guidanceScale),
            seed: ImageGenerationSeed.resolve(request.seed, prompt: request.prompt, backend: .cosmos3Super)
        )
        let result = try await generator.generate(
            options: options,
            resources: resources,
            progressHandler: progressHandler
        )
        return GenerationResult(outputURL: request.outputURL, seed: result.seed)
    }
}
