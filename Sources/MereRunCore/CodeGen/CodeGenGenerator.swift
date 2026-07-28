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

        #if os(Linux)
        if let llamaCLI = LlamaCLIProcess.discover() {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Starting isolated llama.cpp runtime..."))
            return try await llamaCLI.chat(
                request: request,
                modelPath: modelURL.path,
                progressHandler: progressHandler
            )
        }
        #endif

        if loadedModelPath != modelURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading \(CodeGenResources.displayName(for: modelId))..."))
            llamaContext = try await LlamaContext.createContext(
                path: modelURL.path,
                contextSize: CodeGenResources.contextSize(for: modelId),
                temperature: Float(request.temperature),
                topP: Float(request.topP),
                minP: Float(request.minP)
            )
            loadedModelPath = modelURL.path
        }

        guard let ctx = llamaContext else {
            throw CodeGenError.modelNotLoaded
        }

        // Configure sampling parameters for this request
        ctx.configureSampler(
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            minP: Float(request.minP)
        )

        // Set max tokens
        ctx.setMaxTokens(request.maxTokens)

        let prompt = ctx.chatPrompt(for: request.messages)

        return try Self.generateResponse(
            request: request,
            context: ctx,
            prompt: prompt,
            progressHandler: progressHandler
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

        #if os(Linux)
        if let llamaCLI = LlamaCLIProcess.discover() {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Starting isolated llama.cpp runtime..."))
            return try await llamaCLI.chat(
                request: request,
                modelPath: modelURL.path,
                progressHandler: progressHandler
            )
        }
        #endif

        if loadedModelPath != modelURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading \(CodeGenResources.displayName(for: modelId))..."))
            llamaContext = try await LlamaContext.createContext(
                path: modelURL.path,
                contextSize: CodeGenResources.contextSize(for: modelId),
                temperature: Float(request.temperature),
                topP: Float(request.topP),
                minP: Float(request.minP)
            )
            loadedModelPath = modelURL.path
        }

        guard let ctx = llamaContext else {
            throw CodeGenError.modelNotLoaded
        }

        // Configure sampling parameters for this request
        ctx.configureSampler(
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            minP: Float(request.minP)
        )

        // Set max tokens
        ctx.setMaxTokens(request.maxTokens)

        let prompt = ctx.chatPrompt(for: request.messages)

        return try Self.generateResponse(
            request: request,
            context: ctx,
            prompt: prompt,
            progressHandler: progressHandler
        )
    }

    private static func generateResponse(
        request: ChatRequest,
        context ctx: LlamaContext,
        prompt: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> ChatResponse {
        let stopSequences = TextGenerationStopSequences.merged(
            TextGenerationStopSequences.defaultRenderedChatStops + request.stopSequences
        )
        progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
        try ctx.completionInit(text: prompt)

        var response = ""
        var finishReason: ChatFinishReason?
        while !ctx.isDone {
            let token = try ctx.completionLoop()
            response += token
            let trimmed = TextGenerationStopSequences.trimming(response, sequences: stopSequences)
            if trimmed.matchedSequence != nil {
                response = trimmed.text
                finishReason = .stopSequence
                break
            }
            progressHandler?(ChatProgress(stage: .generating, message: token))
        }

        let tokensDecoded = ctx.getTokensDecoded()
        if finishReason == nil {
            finishReason = tokensDecoded >= request.maxTokens ? .length : .stop
        }
        return ChatResponse(
            generatedText: TextGenerationStopSequences.trimming(response, sequences: stopSequences).text,
            tokensGenerated: tokensDecoded,
            showThinking: request.showThinking,
            finishReason: finishReason
        )
    }

    /// Pre-load the model without generating.
    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        let modelURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            .resolvingSymlinksInPath()
        #if os(Linux)
        if LlamaCLIProcess.discover() != nil {
            loadedModelPath = modelURL.path
            llamaContext = nil
            return
        }
        #endif
        if loadedModelPath != modelURL.path {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading \(CodeGenResources.displayName(for: modelId))..."))
            llamaContext = try await LlamaContext.createContext(
                path: modelURL.path,
                contextSize: CodeGenResources.contextSize(for: modelId)
            )
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

}
#endif
