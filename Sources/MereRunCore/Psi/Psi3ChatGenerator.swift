import Foundation
import MLX

// MARK: - Psi3 Chat Generator

/// Actor-based chat generator wrapping GLM-4.7 Flash with automatic Hugging Face download.
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
            topP: Float(request.topP),
            minP: Float(request.minP)
        )

        return ChatResponse(
            generatedText: result.response,
            tokensGenerated: result.tokensGenerated,
            showThinking: request.showThinking
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
            topP: Float(request.topP),
            minP: Float(request.minP)
        )

        return ChatResponse(
            generatedText: result.response,
            tokensGenerated: result.tokensGenerated,
            showThinking: request.showThinking
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
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: Psi3ChatResources.defaultModelId,
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
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw Psi3ChatError.downloadFailed(error.localizedDescription)
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
        }
    }
}
