#if canImport(llama)
import Foundation

/// Actor-based code generation using GGUF models via llama.cpp.
/// Implements the ChatGenerator protocol for standardized chat access.
public actor CodeGenGenerator: ChatGenerator {

    // MARK: - Private State

    private var llamaContext: LlamaContext?
    private var loadedModelPath: String?
    private let modelId: String

    public init(modelId: String = CodeGenResources.defaultModelId) {
        self.modelId = modelId
    }

    // MARK: - ChatGenerator Protocol

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let modelURL = try await resolveModelRoot(modelPath: nil, progressHandler: progressHandler)
            .resolvingSymlinksInPath()

        if loadedModelPath != modelURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen3-Coder..."))
            llamaContext = try await LlamaContext.createContext(
                path: modelURL.path,
                contextSize: 32768,
                temperature: Float(request.temperature),
                topP: Float(request.topP)
            )
            loadedModelPath = modelURL.path
        }

        guard let ctx = llamaContext else {
            throw CodeGenError.modelNotLoaded
        }

        // Configure sampling parameters for this request
        ctx.configureSampler(
            temperature: Float(request.temperature),
            topP: Float(request.topP)
        )

        // Set max tokens
        ctx.setMaxTokens(request.maxTokens)

        // Format prompt with Qwen3 chat template
        let prompt = formatPrompt(request.messages)

        progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
        try ctx.completionInit(text: prompt)

        var response = ""
        while !ctx.isDone {
            let token = try ctx.completionLoop()
            response += token
            progressHandler?(ChatProgress(stage: .generating, message: token))
        }

        let tokensDecoded = ctx.getTokensDecoded()
        return ChatResponse(
            response: response,
            tokensGenerated: tokensDecoded
        )
    }

    /// Chat with an explicit model path (skips auto-download).
    public func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let modelURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            .resolvingSymlinksInPath()

        if loadedModelPath != modelURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen3-Coder..."))
            llamaContext = try await LlamaContext.createContext(
                path: modelURL.path,
                contextSize: 32768,
                temperature: Float(request.temperature),
                topP: Float(request.topP)
            )
            loadedModelPath = modelURL.path
        }

        guard let ctx = llamaContext else {
            throw CodeGenError.modelNotLoaded
        }

        // Configure sampling parameters for this request
        ctx.configureSampler(
            temperature: Float(request.temperature),
            topP: Float(request.topP)
        )

        // Set max tokens
        ctx.setMaxTokens(request.maxTokens)

        // Format prompt with Qwen3 chat template
        let prompt = formatPrompt(request.messages)

        progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
        try ctx.completionInit(text: prompt)

        var response = ""
        while !ctx.isDone {
            let token = try ctx.completionLoop()
            response += token
            progressHandler?(ChatProgress(stage: .generating, message: token))
        }

        let tokensDecoded = ctx.getTokensDecoded()
        return ChatResponse(
            response: response,
            tokensGenerated: tokensDecoded
        )
    }

    /// Pre-load the model without generating.
    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        let modelURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            .resolvingSymlinksInPath()
        if loadedModelPath != modelURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen3-Coder..."))
            llamaContext = try await LlamaContext.createContext(path: modelURL.path)
            loadedModelPath = modelURL.path
        }
    }

    /// Unload the model from memory.
    public func unload() {
        llamaContext = nil
        loadedModelPath = nil
    }

    // MARK: - Model Resolution

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolution = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: CodeGenResources.defaultModelId,
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
            return resolution.url
        } catch let error as PretrainedModelLoader.LoadError {
            throw mapModelLoaderError(error)
        } catch let error as ManagedModelResolver.ResolverError {
            throw CodeGenError.downloadFailed(error.localizedDescription)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> CodeGenError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    // MARK: - Chat Template

    /// Format messages using Qwen3 chat template.
    private func formatPrompt(_ messages: [ChatMessage]) -> String {
        var prompt = ""

        for message in messages {
            switch message.role {
            case .system:
                prompt += "<|im_start|>system\n\(message.content)<|im_end|>\n"
            case .user:
                prompt += "<|im_start|>user\n\(message.content)<|im_end|>\n"
            case .assistant:
                prompt += "<|im_start|>assistant\n\(message.content)<|im_end|>\n"
            case .tool:
                prompt += "<|im_start|>tool\n\(message.content)<|im_end|>\n"
            }
        }

        // Add assistant start tag for generation
        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}
#endif
