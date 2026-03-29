import Foundation
import MLX

// MARK: - Psi3 Chat Generator

/// Actor-based chat generator wrapping GLM-4.7 Flash with automatic R2 download.
/// Implements the ChatGenerator protocol for standardized chat access.
public actor Psi3ChatGenerator: ChatGenerator {

    // MARK: - Private State

    private var model: GLM47Flash?
    private var loadedModelPath: String?
    private let modelId: String

    public init(modelId: String = Psi3ChatResources.defaultModelId) {
        self.modelId = modelId
    }

    // MARK: - ChatGenerator Protocol

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let rootURL = try await resolveModelRoot(modelPath: nil, progressHandler: progressHandler)

        if loadedModelPath != rootURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Psi model..."))
            model = try GLM47Flash(modelRoot: rootURL)
            loadedModelPath = rootURL.path
        }

        guard let model else {
            throw Psi3ChatError.modelNotLoaded
        }

        progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
        let result = try model.generateWithStats(
            messages: request.messages,
            maxNewTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP)
        )

        return ChatResponse(
            response: result.response,
            tokensGenerated: result.tokensGenerated
        )
    }

    /// Chat with an explicit model path (skips auto-download).
    public func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != rootURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Psi model..."))
            model = try GLM47Flash(modelRoot: rootURL)
            loadedModelPath = rootURL.path
        }

        guard let model else {
            throw Psi3ChatError.modelNotLoaded
        }

        progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
        let result = try model.generateWithStats(
            messages: request.messages,
            maxNewTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP)
        )

        return ChatResponse(
            response: result.response,
            tokensGenerated: result.tokensGenerated
        )
    }

    /// Pre-load the model without generating.
    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        if loadedModelPath != rootURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Psi model..."))
            model = try GLM47Flash(modelRoot: rootURL)
            loadedModelPath = rootURL.path
        }
    }

    /// Unload the model from memory.
    public func unload() {
        model = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    // MARK: - Model Resolution

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        do {
            return try await PretrainedModelLoader.fromPretrainedArchive(
                modelPath: modelPath,
                modelId: modelId,
                defaultModelIds: [Psi3ChatResources.defaultModelId],
                storageId: Psi3ChatResources.defaultModelId,
                archiveKey: Psi3ChatResources.r2ArchiveKey,
                archiveSize: Psi3ChatResources.r2ArchiveSize,
                validate: { root, fileManager in
                    Psi3ChatResources(rootURL: root).validate(fileManager: fileManager)
                },
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(
                            ChatProgress(stage: .loadingModel, message: "Downloading model... \(percent)%")
                        )
                    case .extracting:
                        progressHandler?(
                            ChatProgress(stage: .loadingModel, message: "Extracting model...")
                        )
                    }
                }
            )
        } catch let error as PretrainedModelLoader.LoadError {
            throw mapModelLoaderError(error)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> Psi3ChatError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        case .extractionFailed:
            return .extractionFailed
        }
    }
}
