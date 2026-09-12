import Foundation

/// Validates extraction controls and supplied camera rows without loading weights.
public struct InstantMeshGenerationSettings: Equatable, Sendable {
    public let extractionResolution: Int
    public let includesVertexColors: Bool
    public let cameras: [[Float]]?

    public init(
        extractionResolution: Int = InstantMeshConfiguration.production.gridResolution,
        includesVertexColors: Bool = true,
        cameras: [[Float]]? = nil
    ) throws {
        guard (2...256).contains(extractionResolution) else {
            throw InstantMeshGeneratorError.invalidExtractionResolution(extractionResolution)
        }
        if let cameras {
            for (index, camera) in cameras.enumerated()
            where camera.count != 16 || !camera.allSatisfy(\.isFinite) {
                throw InstantMeshPreprocessingError.invalidCamera(index: index)
            }
        }
        self.extractionResolution = extractionResolution
        self.includesVertexColors = includesVertexColors
        self.cameras = cameras
    }

    public static func validateViewCount(_ count: Int) throws {
        guard count == 4 || count == 6 else {
            throw InstantMeshGeneratorError.invalidViewCount(count)
        }
    }

    public func validate(viewCount: Int) throws {
        try Self.validateViewCount(viewCount)
        if let cameras, cameras.count != viewCount {
            throw InstantMeshPreprocessingError.cameraCountMismatch(expected: viewCount, actual: cameras.count)
        }
    }
}

public struct InstantMeshGenerationRequest: Equatable, Sendable {
    public let viewURLs: [URL]
    public let outputDirectory: URL
    public let model: String?
    public let settings: InstantMeshGenerationSettings

    public init(viewURLs: [URL], outputDirectory: URL, model: String? = nil, settings: InstantMeshGenerationSettings) {
        self.viewURLs = viewURLs.map(\.standardizedFileURL)
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.model = model
        self.settings = settings
    }
}

/// Owns per-request execution and awaited cleanup within the caller's admission.
public enum InstantMeshGenerationOperation {
    public typealias ProgressHandler = @Sendable (InstantMeshProgress) -> Void

    public struct Runtime: Sendable {
        public let generate: @Sendable (InstantMeshGenerationRequest, ProgressHandler?) async throws -> InstantMeshRunResult
        public let unload: @Sendable () async -> Void

        public init(
            generate: @escaping @Sendable (InstantMeshGenerationRequest, ProgressHandler?) async throws -> InstantMeshRunResult,
            unload: @escaping @Sendable () async -> Void
        ) {
            self.generate = generate
            self.unload = unload
        }
    }

    /// Inspects current headers without creating output or resolving a checkpoint.
    public static func prepare(_ request: InstantMeshGenerationRequest) throws -> [VFXImageInputDimensions] {
        try Task.checkCancellation()
        try request.settings.validate(viewCount: request.viewURLs.count)
        for url in request.viewURLs where !FileManager.default.fileExists(atPath: url.path) {
            throw InstantMeshGeneratorError.inputViewNotFound(url.path)
        }
        return try VFXImageInputValidator.inspectAndValidate(request.viewURLs)
    }

    public static func execute(
        _ request: InstantMeshGenerationRequest,
        prepareRuntime: @Sendable () throws -> Void = {},
        progress: ProgressHandler? = nil,
        makeRuntime: @Sendable () -> Runtime = nativeRuntime
    ) async throws -> InstantMeshRunResult {
        _ = try prepare(request)
        try prepareRuntime()
        try Task.checkCancellation()
        let runtime = makeRuntime()
        let result: InstantMeshRunResult
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
        let generator = InstantMeshGenerator()
        return Runtime(
            generate: { request, progress in
                try await generator.generate(
                    viewURLs: request.viewURLs, outputDirectory: request.outputDirectory,
                    model: request.model, cameras: request.settings.cameras,
                    extractionResolution: request.settings.extractionResolution,
                    includeVertexColors: request.settings.includesVertexColors, progress: progress
                )
            },
            unload: { await generator.unload() }
        )
    }
}
