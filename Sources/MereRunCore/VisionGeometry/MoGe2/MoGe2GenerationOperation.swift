import Foundation
import MediaIO

public enum MoGe2GenerationError: Error, Equatable, LocalizedError, Sendable {
    case invalidResolutionLevel(Int)
    case invalidTokenCount(Int)
    case invalidMaximumPointCount(Int)
    case inputNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResolutionLevel: "Resolution level must be between 0 and 9."
        case .invalidTokenCount: "Token count must be between 1 and 3600."
        case .invalidMaximumPointCount: "Maximum point count must be positive."
        case .inputNotFound(let path): "Input image not found: \(path)"
        }
    }
}

/// Validates caller values before constructing the compatibility configuration.
public struct MoGe2GenerationSettings: Equatable, Sendable {
    public static let defaultResolutionLevel = MoGe2InferenceConfiguration.defaultResolutionLevel
    public let configuration: MoGe2InferenceConfiguration

    public init(
        resolutionLevel: Int = defaultResolutionLevel,
        tokenCount: Int? = nil,
        maximumPointCount: Int? = nil
    ) throws {
        guard (0...9).contains(resolutionLevel) else {
            throw MoGe2GenerationError.invalidResolutionLevel(resolutionLevel)
        }
        if let tokenCount,
           !(MoGe2TokenGrid.minimumTokenCount...MoGe2TokenGrid.maximumTokenCount).contains(tokenCount) {
            throw MoGe2GenerationError.invalidTokenCount(tokenCount)
        }
        if let maximumPointCount, maximumPointCount <= 0 {
            throw MoGe2GenerationError.invalidMaximumPointCount(maximumPointCount)
        }
        configuration = MoGe2InferenceConfiguration(
            resolutionLevel: resolutionLevel, tokenCount: tokenCount, maximumPointCount: maximumPointCount
        )
    }
}

/// Carries the same settings from CLI or API translation into native execution.
public struct MoGe2GenerationRequest: Equatable, Sendable {
    public static let defaultModelID = ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue
    public let imageURL: URL
    public let outputDirectory: URL
    public let model: String?
    public let settings: MoGe2GenerationSettings

    public init(imageURL: URL, outputDirectory: URL, model: String? = nil, settings: MoGe2GenerationSettings) {
        self.imageURL = imageURL.standardizedFileURL
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.model = model
        self.settings = settings
    }
}

/// Describes an observational plan without loading weights or creating output.
public struct MoGe2GenerationPlan: Equatable, Sendable {
    public let request: MoGe2GenerationRequest
    public let dimensions: VFXImageInputDimensions
    public let tokenGrid: MoGe2TokenGrid

    public init(request: MoGe2GenerationRequest, imageWidth: Int, imageHeight: Int) throws {
        self.request = request
        dimensions = try VFXImageInputValidator.validate(
            width: imageWidth, height: imageHeight, path: request.imageURL.path
        )
        tokenGrid = try MoGe2TokenGrid.resolve(
            imageWidth: imageWidth, imageHeight: imageHeight,
            requestedTokenCount: request.settings.configuration.effectiveTokenCount
        )
    }
}

/// Owns one request's runtime and awaited release. Admission stays with the caller.
public enum MoGe2GenerationOperation {
    public typealias ProgressHandler = @Sendable (String) -> Void

    public struct Runtime: Sendable {
        public let generate: @Sendable (MoGe2GenerationRequest, ProgressHandler?) async throws -> MoGe2RunResult
        public let unload: @Sendable () async -> Void

        public init(
            generate: @escaping @Sendable (MoGe2GenerationRequest, ProgressHandler?) async throws -> MoGe2RunResult,
            unload: @escaping @Sendable () async -> Void
        ) {
            self.generate = generate
            self.unload = unload
        }
    }

    public static func prepare(_ request: MoGe2GenerationRequest) throws -> MoGe2GenerationPlan {
        try Task.checkCancellation()
        guard FileManager.default.fileExists(atPath: request.imageURL.path) else {
            throw MoGe2GenerationError.inputNotFound(request.imageURL.path)
        }
        let size = try MediaImageIO.size(of: request.imageURL)
        return try MoGe2GenerationPlan(request: request, imageWidth: size.width, imageHeight: size.height)
    }

    public static func execute(
        _ request: MoGe2GenerationRequest,
        prepareRuntime: @Sendable () throws -> Void = {},
        progress: ProgressHandler? = nil,
        makeRuntime: @Sendable () -> Runtime = nativeRuntime
    ) async throws -> MoGe2RunResult {
        // Recheck mutable input headers even when the caller previously planned.
        _ = try prepare(request)
        try prepareRuntime()
        try Task.checkCancellation()
        let runtime = makeRuntime()
        let result: MoGe2RunResult
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
        let generator = MoGe2Generator()
        return Runtime(
            generate: { request, progress in
                try await generator.generate(
                    imageURL: request.imageURL, outputDirectory: request.outputDirectory,
                    model: request.model, configuration: request.settings.configuration, progress: progress
                )
            },
            unload: { await generator.unload() }
        )
    }
}
