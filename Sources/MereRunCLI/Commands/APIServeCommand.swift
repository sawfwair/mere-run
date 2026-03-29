import ArgumentParser
import Foundation
import Hummingbird
import NIOCore
import MereRunCore

struct APIServe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an OpenAI-compatible API server for local text-code or text-chat models.",
        discussion: """
        Runs an HTTP server that exposes an OpenAI-compatible API for the selected engine.
        Compatible with any OpenAI client (VS Code extensions, Continue, Cursor, etc.).

        Endpoints:
          GET  /health              - Health check
          GET  /v1/models           - List available models
          POST /v1/chat/completions - Chat completions (streaming supported)

        Example:
          # Start with the default local code model
          mere.run api serve

          # Start with a specific GGUF model
          mere.run api serve -m ~/models/Qwen3-Coder-Next-Q4_K_M.gguf

          # Start a Q35 text-chat server
          mere.run api serve --engine text-chat-q35

          # Start a Q35 text-chat server with an explicit nano model root
          mere.run api serve --engine text-chat-q35 -m ~/Library/Application\\ Support/MereRun/models/text-chat-q35-nano

          # Custom host/port
          mere.run api serve --host 0.0.0.0 --port 11434

          # Test with curl
          curl http://localhost:8080/v1/chat/completions \\
            -H "Content-Type: application/json" \\
            -d '{"model":"text-code-qwen3","messages":[{"role":"user","content":"Hello!"}]}'
        """
    )

    @Option(name: [.short, .long], help: "Port to listen on.")
    var port: Int = 8080

    @Option(name: [.long], help: "Host to bind to.")
    var host: String = "127.0.0.1"

    @Option(name: [.customShort("m"), .long, .customLong("model-path")], help: "Model path. For --engine text-code, pass a GGUF file. For --engine text-chat-klein, pass a Klein-root text chat model. For --engine text-chat-q35, pass a Q35 text chat model root.")
    var model: String?

    @Option(name: [.long], help: "Serving engine: text-code (default), text-chat-klein, or text-chat-q35.")
    var engine: APIEngine = .textCode

    @Option(name: [.long], help: "Default LoRA adapter path for all requests.")
    var lora: String?

    @Option(name: [.long], help: "Context size (default: 32768).")
    var contextSize: Int = 32768

    func run() async throws {
        let resolvedModelPath = try resolveModelPath()
        let server = try await CodeGenServer(
            modelPath: resolvedModelPath,
            fallbackLoraPath: lora,
            engine: engine,
            contextSize: contextSize
        )
        try await server.run(host: host, port: port)
    }

    private func resolveModelPath() throws -> String? {
        switch engine {
        case .textCode:
            return model
        case .textChatKlein:
            if let explicit = model {
                return explicit
            }
            // Prefer standalone MeBot Instruct model
            if let mebotPath = MeBotModelCatalog.resolveModelPath() {
                return mebotPath
            }
            // Fall back to Klein model
            if let resolved = ModelResolver().resolveIfPresent(.mebot) {
                return resolved.rootURL.path
            }
            throw ValidationError("Model 'text-chat-mebot' is not installed. Download image-klein-nano or image-klein-max first.")
        case .textChatQ35:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.q35) {
                return resolved.rootURL.path
            }
            if let resolved = ModelResolver().resolveIfPresent(.q35Nano) {
                return resolved.rootURL.path
            }
            // Allow Q35Generator to auto-download from R2 when model path is omitted.
            return nil
        }
    }
}

enum APIEngine: String, ExpressibleByArgument {
    case textCode = "text-code"
    case textChatKlein = "text-chat-klein"
    case textChatQ35 = "text-chat-q35"
}

struct APIHealthStatus: Codable, Equatable, Sendable {
    let status: String
}

enum APIServerContract {
    static func healthStatus() -> APIHealthStatus {
        APIHealthStatus(status: "ok")
    }

    static func modelsResponse(modelId: String, createdAt: Date = Date()) -> OpenAIModelsResponse {
        OpenAIModelsResponse(
            object: "list",
            data: [
                OpenAIModel(
                    id: modelId,
                    object: "model",
                    created: Int(createdAt.timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            ]
        )
    }
}

// MARK: - Server Implementation

actor CodeGenServer {
    private let engine: APIEngine
    private let llamaGenerator: CodeGenGenerator?
    private let mlxGenerator: Flux2KleinGenerator?
    private let q35Generator: Q35Generator?
    private let modelPath: String?
    private let fallbackLoraPath: String?
    private let modelId: String
    private let contextSize: Int
    private let useStandaloneModel: Bool

    init(
        modelPath: String?,
        fallbackLoraPath: String?,
        engine: APIEngine,
        contextSize: Int = 32768
    ) async throws {
        self.contextSize = contextSize
        self.engine = engine
        self.fallbackLoraPath = fallbackLoraPath
        self.modelPath = modelPath
        self.modelId = modelPath.map { URL(fileURLWithPath: $0).lastPathComponent }
            ?? {
                switch engine {
                case .textChatKlein:
                    return ModelResolver.ModelID.mebot.rawValue
                case .textChatQ35:
                    return ModelResolver.ModelID.q35.rawValue
                case .textCode:
                    return CodeGenResources.defaultModelId
                }
            }()

        switch engine {
        case .textCode:
            let generator = CodeGenGenerator(modelId: modelPath ?? CodeGenResources.defaultModelId)
            self.llamaGenerator = generator
            self.mlxGenerator = nil
            self.q35Generator = nil
            self.useStandaloneModel = false

            // Pre-load model
            try await generator.prepare(modelPath: modelPath) { progress in
                fputs("[\(progress.stage.rawValue)] \(progress.message ?? "")\n", stderr)
            }
        case .textChatKlein:
            self.llamaGenerator = nil
            self.mlxGenerator = Flux2KleinGenerator()
            self.q35Generator = nil
            // Check if the resolved path is a standalone MeBot Instruct model
            self.useStandaloneModel = MeBotModelCatalog.resolveModelPath() != nil
                && modelPath == MeBotModelCatalog.resolveModelPath()
        case .textChatQ35:
            let generator = Q35Generator(modelId: Q35Resources.defaultModelId)
            self.llamaGenerator = nil
            self.mlxGenerator = nil
            self.q35Generator = generator
            self.useStandaloneModel = false

            try await generator.prepare(modelPath: modelPath) { progress in
                fputs("[\(progress.stage.rawValue)] \(progress.message ?? "")\n", stderr)
            }
        }
    }

    func run(host: String, port: Int) async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        print("Starting server at http://\(host):\(port)")
        print("OpenAI-compatible endpoint: http://\(host):\(port)/v1/chat/completions")
        print("Press Ctrl+C to stop.")

        try await app.runService()
    }

    nonisolated func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        // Health check
        router.get("/health") { _, _ in
            let data = try JSONEncoder().encode(APIServerContract.healthStatus())
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(data: data))
            )
        }

        // List models
        router.get("/v1/models") { [self] _, _ in
            return try await self.handleModels()
        }

        // Chat completions
        router.post("/v1/chat/completions") { [self] request, _ in
            return try await self.handleChatCompletions(request)
        }

        return router
    }

    private func handleModels() async throws -> Response {
        let models = APIServerContract.modelsResponse(modelId: modelId)

        let data = try JSONEncoder().encode(models)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func handleChatCompletions(_ request: Request) async throws -> Response {
        // Decode request body
        let body = try await request.body.collect(upTo: 10 * 1024 * 1024) // 10MB limit
        let openaiRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: body)

        // Convert to ChatRequest
        let messages = openaiRequest.messages.map { msg in
            ChatMessage(
                role: ChatMessage.Role(rawValue: msg.role) ?? .user,
                content: msg.content
            )
        }

        let effectiveLoraPath = openaiRequest.lora ?? fallbackLoraPath
        let lora = effectiveLoraPath.map { LoRA.local(path: $0, scale: 1.0) }

        let chatRequest = ChatRequest(
            messages: messages,
            maxTokens: openaiRequest.max_tokens ?? 2048,
            temperature: openaiRequest.temperature ?? 1.0,
            topP: openaiRequest.top_p ?? 0.95,
            lora: lora
        )

        if openaiRequest.stream == true {
            return try await handleStreamingChat(chatRequest)
        } else {
            return try await handleNonStreamingChat(chatRequest)
        }
    }

    private func handleNonStreamingChat(_ request: ChatRequest) async throws -> Response {
        let result = try await generateChat(request, progressHandler: nil)

        let response = OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: [
                OpenAIChatChoice(
                    index: 0,
                    message: OpenAIChatMessage(role: "assistant", content: result.response),
                    finish_reason: "stop"
                )
            ],
            usage: OpenAIUsage(
                prompt_tokens: 0,
                completion_tokens: result.tokensGenerated,
                total_tokens: result.tokensGenerated
            )
        )

        let data = try JSONEncoder().encode(response)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func handleStreamingChat(_ request: ChatRequest) async throws -> Response {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let encoder = JSONEncoder()

        // Create async stream for SSE
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()

        // Start generation in a detached task
        Task { [self, modelId] in
            do {
                _ = try await self.generateChat(request) { progress in
                    if progress.stage == .generating, let token = progress.message {
                        let chunk = OpenAIChatResponse(
                            id: id,
                            object: "chat.completion.chunk",
                            created: Int(Date().timeIntervalSince1970),
                            model: modelId,
                            choices: [
                                OpenAIChatChoice(
                                    index: 0,
                                    delta: OpenAIChatDelta(content: token),
                                    finish_reason: nil
                                )
                            ]
                        )
                        if let data = try? encoder.encode(chunk),
                           let json = String(data: data, encoding: .utf8) {
                            let line = "data: \(json)\n\n"
                            continuation.yield(ByteBuffer(string: line))
                        }
                    }
                }

                // Final chunk with finish_reason
                let finalChunk = OpenAIChatResponse(
                    id: id,
                    object: "chat.completion.chunk",
                    created: Int(Date().timeIntervalSince1970),
                    model: modelId,
                    choices: [
                        OpenAIChatChoice(
                            index: 0,
                            delta: OpenAIChatDelta(),
                            finish_reason: "stop"
                        )
                    ]
                )
                if let data = try? encoder.encode(finalChunk),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                }

                continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
                continuation.finish()
            } catch {
                // Send error in SSE format
                let errorJson = "{\"error\":{\"message\":\"\(error.localizedDescription)\",\"type\":\"server_error\"}}"
                continuation.yield(ByteBuffer(string: "data: \(errorJson)\n\n"))
                continuation.finish()
            }
        }

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .init("Cache-Control")!: "no-cache",
                .connection: "keep-alive"
            ],
            body: .init(asyncSequence: stream)
        )
    }

    private func generateChat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        switch engine {
        case .textCode:
            guard let generator = llamaGenerator else {
                throw CodeGenError.modelNotLoaded
            }
            return try await generator.chat(request, progressHandler: progressHandler)
        case .textChatKlein:
            guard let generator = mlxGenerator else {
                throw Flux2Error.modelsNotLoaded
            }
            guard let modelPath else {
                throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
            }
            if useStandaloneModel {
                return try await generator.chatStandalone(request, modelPath: modelPath, progressHandler: progressHandler)
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatQ35:
            guard let generator = q35Generator else {
                throw Q35Error.modelNotLoaded
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        }
    }
}
