import Foundation

/// Validated controls shared by reconstruction adapters and the native generator.
public struct TripoSRGenerationSettings: Equatable, Sendable {
    public static let defaultResolution = 256
    public static let defaultForegroundRatio: Float = 0.85
    public let extractionResolution: Int
    public let densityThreshold: Float
    public let foregroundRatio: Float
    public let alreadyFramed: Bool
    public let includesVertexColors: Bool

    public var foregroundPolicy: TripoSRForegroundPolicy {
        alreadyFramed ? .alreadyFramed : .automaticTransparentAlpha(foregroundRatio: foregroundRatio)
    }

    public init(
        extractionResolution: Int = defaultResolution,
        densityThreshold: Float = TripoSRConfiguration.production.densityThreshold,
        foregroundRatio: Float = defaultForegroundRatio,
        alreadyFramed: Bool = false,
        includesVertexColors: Bool = true
    ) throws {
        guard (2...512).contains(extractionResolution) else {
            throw TripoSRGeneratorError.invalidExtractionResolution(extractionResolution)
        }
        guard densityThreshold.isFinite else {
            throw TripoSRGeneratorError.invalidDensityThreshold(densityThreshold)
        }
        guard foregroundRatio.isFinite, foregroundRatio > 0, foregroundRatio <= 1 else {
            throw TripoSRPreprocessingError.invalidForegroundRatio(foregroundRatio)
        }
        self.extractionResolution = extractionResolution
        self.densityThreshold = densityThreshold
        self.foregroundRatio = foregroundRatio
        self.alreadyFramed = alreadyFramed
        self.includesVertexColors = includesVertexColors
    }
}

public struct TripoSRGenerationRequest: Equatable, Sendable {
    public let imageURL: URL
    public let outputDirectory: URL
    public let model: String?
    public let settings: TripoSRGenerationSettings

    public init(imageURL: URL, outputDirectory: URL, model: String? = nil, settings: TripoSRGenerationSettings) {
        self.imageURL = imageURL.standardizedFileURL
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.model = model
        self.settings = settings
    }
}

/// Owns per-request execution and awaited cleanup within the caller's admission.
public enum TripoSRGenerationOperation {
    public typealias ProgressHandler = @Sendable (TripoSRProgress) -> Void

    public struct Runtime: Sendable {
        public let generate: @Sendable (TripoSRGenerationRequest, ProgressHandler?) async throws -> TripoSRRunResult
        public let unload: @Sendable () async -> Void

        public init(
            generate: @escaping @Sendable (TripoSRGenerationRequest, ProgressHandler?) async throws -> TripoSRRunResult,
            unload: @escaping @Sendable () async -> Void
        ) {
            self.generate = generate
            self.unload = unload
        }
    }

    /// Inspects current headers without creating output or resolving a checkpoint.
    public static func prepare(_ request: TripoSRGenerationRequest) throws -> VFXImageInputDimensions {
        try Task.checkCancellation()
        guard FileManager.default.fileExists(atPath: request.imageURL.path) else {
            throw TripoSRGeneratorError.inputImageNotFound(request.imageURL.path)
        }
        return try VFXImageInputValidator.inspectAndValidate([request.imageURL])[0]
    }

    public static func execute(
        _ request: TripoSRGenerationRequest,
        prepareRuntime: @Sendable () throws -> Void = {},
        progress: ProgressHandler? = nil,
        makeRuntime: @Sendable () -> Runtime = nativeRuntime
    ) async throws -> TripoSRRunResult {
        _ = try prepare(request)
        try prepareRuntime()
        try Task.checkCancellation()
        let runtime = makeRuntime()
        let result: TripoSRRunResult
        do {
            result = try await runtime.generate(request, progress)
            try Task.checkCancellation()
        } catch {
            await runtime.unload()
            throw error
        }
        await runtime.unload()
        try Task.checkCancellation()
        return result
    }

    public static func nativeRuntime() -> Runtime {
        let generator = TripoSRGenerator()
        return Runtime(
            generate: { request, progress in
                try await generator.generate(
                    imageURL: request.imageURL, outputDirectory: request.outputDirectory,
                    model: request.model, foregroundPolicy: request.settings.foregroundPolicy,
                    extractionResolution: request.settings.extractionResolution,
                    densityThreshold: request.settings.densityThreshold,
                    includeVertexColors: request.settings.includesVertexColors, progress: progress
                )
            },
            unload: { await generator.unload() }
        )
    }
}
