import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import AudioCore
import AudioSTT
import AudioTTS
import MediaIO
import MereRunCore

struct APIServe: AsyncParsableCommand {
    @Option(name: [.customLong("image-run-records")], help: "Keep durable image run directories under this root.")
    var imageRunRecords: String?

    static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an OpenAI-compatible API server for local chat and embedding models.",
        discussion: """
        Runs an HTTP server that exposes an OpenAI-compatible API for the selected local runtime.
        Compatible with any OpenAI client (VS Code extensions, Continue, Cursor, etc.).

        Endpoints:
          GET  /health              - Health check
          GET  /v1/models           - List available models
          POST /v1/chat/completions - Chat completions (streaming supported)
          POST /v1/embeddings       - Native Qwen3 text embeddings
          POST /v1/images/generations - Native image generation
          POST /v1/images/edits       - Native image editing
          POST /v1/videos/generations - Native video generation (loopback artifact route)
          POST /v1/vision/geometry    - Native metric image geometry
          POST /v1/vision/geometry/multiview - Native DA3 multi-view geometry and cameras
          POST /v1/vision/image-to-3d - Native TripoSR object mesh reconstruction
          POST /v1/vision/image-to-3d-multiview - Native InstantMesh reconstruction from uploaded views
          POST /v1/vision/depth-video - Native temporally consistent video depth
          POST /v1/audio/speech      - Native text to speech
          POST /v1/audio/transcriptions - Native speech to text

        Example:
          # Start with the default local code model
          mere.run api serve

          # Start with a specific GGUF model
          mere.run api serve -m ~/models/Qwen3-Coder-Next-Q4_K_M.gguf

          # Start a Gemma 4 text-chat server
          mere.run api serve --engine text-chat-gemma4

          # Start a DiffusionGemma block-diffusion text server
          mere.run api serve --engine text-chat-diffusiongemma

          # Start a Qwen3.6 text-chat server with an explicit model root
          mere.run api serve --engine text-chat-q36 -m ~/Models/text-chat-q36-nano

          # Start an LFM2.5 text-chat server
          mere.run api serve --engine text-chat-lfm2

          # Start a Laguna XS 2.1 server
          mere.run api serve --engine text-chat-laguna --model text-chat-laguna-xs-2-1

          # Start the DeepSeek V4 Flash OpenAI-compatible server
          mere.run api serve --engine text-chat-deepseek-v4-flash

          # Start the native Muse Glimmer multimodal agent server
          mere.run api serve --engine text-chat-muse-glimmer

          # Custom host/port (non-loopback binds require an API key)
          export MERERUN_API_KEY=change-me
          mere.run api serve --host 0.0.0.0 --port 11434 --api-key "$MERERUN_API_KEY"

          # Check server health and the served model from another terminal
          mere.run status --port 11434

          # Test chat with curl (request.json contains an OpenAI chat payload)
          curl http://localhost:8080/v1/chat/completions \\
            -H "Content-Type: application/json" \\
            --data @request.json

          # Test embeddings with curl
          curl http://localhost:8080/v1/embeddings \\
            -H "Content-Type: application/json" \\
            --data '{"model":"text-embed-qwen3-0.6b","input":"hello"}'

          # Generate an image as base64 PNG JSON
          curl http://localhost:8080/v1/images/generations \\
            -H "Content-Type: application/json" \\
            --data '{"model":"image-zimage-nano","prompt":"a tiny workstation in morning light","size":"1024x1024"}'

          # Edit an image as base64 PNG JSON
          curl http://localhost:8080/v1/images/edits \\
            -F model=image-zimage-nano \\
            -F prompt='make the workstation dusk-lit' \\
            -F image=@input.png

          # Generate synchronized LTX-2.5 video and audio
          curl http://localhost:8080/v1/videos/generations \\
            -H "Content-Type: application/json" \\
            --data '{"prompt":"waves break under moonlight","model":"video-ltx25-distilled-bf16","size":"768x512","num_frames":65,"output_mode":"audio-video"}'

          # Generate speech
          curl http://localhost:8080/v1/audio/speech \\
            -H "Content-Type: application/json" \\
            --output speech.wav \\
            --data '{"model":"speech-tts-qwen3-nano","input":"mere.run is online","voice":"nova","response_format":"wav"}'

          # Recover metric depth, normals, camera, and a point cloud
          curl http://localhost:8080/v1/vision/geometry \
            -F model=vision-geometry-moge2-small \
            -F resolution_level=9 \
            -F image=@frame.png

          # Solve relative multi-view geometry, confidence, and cameras
          curl http://localhost:8080/v1/vision/geometry/multiview \
            -F model=vision-geometry-da3-small \
            -F process_resolution=504 \
            -F 'image[]=@view-0.png' \
            -F 'image[]=@view-1.png'

          # Reconstruct a normalized colored object mesh from uploaded image bytes
          curl http://localhost:8080/v1/vision/image-to-3d \
            -F model=image-3d-triposr \
            -F resolution=256 \
            -F image=@object.png

          # Reconstruct from exactly four or six uploaded, user-supplied views
          curl http://localhost:8080/v1/vision/image-to-3d-multiview \
            -F model=image-3d-instantmesh-base \
            -F resolution=128 \
            -F 'image[]=@view-0.png' \
            -F 'image[]=@view-1.png' \
            -F 'image[]=@view-2.png' \
            -F 'image[]=@view-3.png'

          # Recover temporally consistent relative depth from uploaded video bytes
          curl http://localhost:8080/v1/vision/depth-video \
            -F model=vision-depth-vda-small \
            -F input_size=518 \
            -F video=@shot.mp4

          # Transcribe audio
          curl http://localhost:8080/v1/audio/transcriptions \\
            -F model=speech-asr-parakeet \\
            -F file=@speech.wav
        """
    )

    @Option(name: [.short, .long], help: "Port to listen on.")
    var port: Int = 8080

    @Option(name: [.long], help: "Host to bind to.")
    var host: String = "127.0.0.1"

    @Option(name: [.customShort("m"), .long, .customLong("model-path")], help: "Model path. For --engine text-code, pass a GGUF file. For --engine text-chat-klein, pass a Klein-root text chat model. For --engine text-chat-gemma4, pass a Gemma 4 model root or repo ID. For --engine text-chat-diffusiongemma, pass text-chat-diffusiongemma-26b-optiq-4bit or its installed MLX root. For --engine text-chat-laguna, pass text-chat-laguna-s-2-1, text-chat-laguna-xs-2-1, or an installed Laguna MLX root. For --engine text-chat-q36, pass a Qwen3.6 text chat model root. For --engine text-chat-lfm2, pass an LFM2 MLX model root or repo ID. For --engine text-chat-deepseek-v4-flash, pass a DS4 GGUF file or managed model root. For --engine text-chat-muse-glimmer, pass vision-chat-muse-glimmer-30b or an installed Muse Glimmer MLX root. For --engine text-chat-nemotron-omni, pass omni-chat-nemotron3-nano-30b-a3b-bf16 or its installed wrapper root.")
    var model: String?

    @Option(name: [.long], help: "Serving engine: text-chat-q36 (default; serves text-chat-q36-nano), text-code, text-chat-klein, text-chat-gemma4, text-chat-diffusiongemma, text-chat-laguna, text-chat-lfm2, text-chat-deepseek-v4-flash, text-chat-muse-glimmer, text-chat-nemotron-h, or text-chat-nemotron-omni.")
    var engine: APIEngine = .textChatQ36

    @Option(name: [.long], help: "Default cataloged adapter id or local LoRA path for all requests.")
    var lora: String?

    @Option(name: [.long], help: "Bearer token required by API endpoints. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "Global OpenAI-compatible inference request limit per rolling minute.")
    var rateLimitPerMinute: Int = 60

    @Option(
        name: [.long],
        help: """
        Maximum local inference requests admitted at once. Defaults to 1; values above 1 automatically enable \
        supported Gemma4, Qwen-family, and LFM2 decode batching unless overridden by environment.
        """
    )
    var maxActiveRequests: Int = 1

    @Option(name: [.long], help: "Runtime memory guard tier: off, safe, balanced, aggressive, or custom.")
    var memoryGuard: RuntimeMemoryGuardTier = .default

    @Option(name: [.long], help: "Custom memory guard ceiling in GiB. Requires --memory-guard custom.")
    var memoryGuardCustomCeilingGB: Double?

    @Option(name: [.long], help: "Context size (default: 32768).")
    var contextSize: Int = 32768

    @Flag(
        name: [.customLong("warmup")],
        inversion: .prefixedNo,
        help: "Warm supported default-model inference graphs before the server becomes healthy."
    )
    var warmup: Bool = true

    @Option(name: [.long], help: "Quantize the Gemma4 KV cache to this many bits. Supports integer widths for uniform/polar and integer/.5 widths for turboquant.")
    var kvBits: Double?

    @Option(name: [.long], help: "Gemma4 KV cache quantization backend: uniform, polar, or turboquant.")
    var kvQuantScheme: String?

    @Option(name: [.long], help: "Gemma4 KV cache quantization group size.")
    var kvGroupSize: Int?

    @Option(name: [.long], help: "Gemma4 token offset at which KV cache quantization begins.")
    var quantizedKVStart: Int?

    @Flag(name: [.customLong("preflight")], help: "Inspect server configuration without starting the API server.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    func run() async throws {
        if preflight {
            try runPreflight()
            return
        }
        PiAgentIntegration.startAgentParentExitMonitorIfConfigured()
        guard !json else {
            throw ValidationError("--json is only supported with --preflight for api serve.")
        }

        let resolvedAPIKey = resolveAPIKey()
        try validateServerSecurity(apiKey: resolvedAPIKey)
        let resolvedModelPath = try resolveModelPath()
        let defaultModelID = defaultRuntimeModelID(modelPath: resolvedModelPath)
        let resolvedLoraPath = try resolveLoraPath(modelPath: resolvedModelPath)
        let gemma4KVCacheQuantization = try resolveGemma4KVCacheQuantization()
        let memoryPressurePolicy = try resolveMemoryPressurePolicy()
        try await withMachineInferenceAdmission(
            using: .shared,
            request: CLIInferenceAdmissionClassifier.apiServerRequest(engine: engine, modelID: defaultModelID),
            onWait: { snapshot in
                CLIStderr.write(
                    "API server queued by machine admission "
                        + "(\(snapshot.activePermits)/\(snapshot.capacityPermits) permits active, "
                        + "\(snapshot.queued.count) queued).\n"
                )
            }
        ) {
            CLIStderr.write("\(PiAgentIntegration.serverAdmissionMarker)\n")
            let server = try await CodeGenServer(
                defaultModelID: defaultModelID,
                modelPath: resolvedModelPath,
                fallbackLoraPath: resolvedLoraPath,
                apiKey: resolvedAPIKey,
                rateLimitPerMinute: rateLimitPerMinute,
                maxActiveRequests: maxActiveRequests,
                engine: engine,
                contextSize: contextSize,
                gemma4KVCacheQuantization: gemma4KVCacheQuantization,
                memoryPressurePolicy: memoryPressurePolicy,
                imageRunRecords: imageRunRecords.map { URL(fileURLWithPath: $0).standardizedFileURL },
                warmupDefaultModel: warmup
            )
            try await server.run(host: host, port: port)
        }
    }

    func defaultRuntimeModelID(modelPath: String?) -> String {
        if let requested = model?.trimmingCharacters(in: .whitespacesAndNewlines),
           let spec = ManagedModelCatalog.spec(for: requested),
           spec.defaultRuntimeServingEngine == engine.runtimeServingEngine {
            return spec.id
        }
        if let modelPath, model != nil {
            return URL(fileURLWithPath: modelPath).lastPathComponent
        }
        switch engine {
        case .textChatKlein:
            return ModelResolver.ModelID.mebot.rawValue
        case .textChatGemma4:
            return ModelResolver.ModelID.gemma4.rawValue
        case .textChatDiffusionGemma:
            return DiffusionGemmaResources.modelID
        case .textChatLaguna:
            return LagunaResources.modelID
        case .textChatQ36, .textChatQ35:
            return ModelResolver.ModelID.q36Nano.rawValue
        case .textChatLFM2:
            return LFM2Resources.defaultModelId
        case .textCode:
            return CodeGenResources.defaultModelId
        case .textChatDeepseekV4Flash:
            return DeepseekV4FlashResources.defaultModelId
        case .textChatMuseGlimmer:
            return MuseGlimmerResources.modelId
        case .textChatNemotronH:
            return NemotronHResources.modelID
        case .textChatNemotronOmni:
            return NemotronOmniResources.modelID
        }
    }

    func resolveLoraPath(
        modelPath: String?,
        fileManager: FileManager = .default
    ) throws -> String? {
        try ManagedAdapterArgumentResolver.resolve(
            lora,
            baseModelID: defaultRuntimeModelID(modelPath: modelPath),
            fileManager: fileManager
        )
    }

    func resolveModelPath() throws -> String? {
        switch engine {
        case .textCode:
            return model
        case .textChatKlein:
            if let explicit = model {
                return explicit
            }
            if let mebotPath = MeBotModelCatalog.resolveModelPath() {
                return mebotPath
            }
            throw ValidationError("Model 'text-chat-mebot' is not installed.")
        case .textChatGemma4:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.gemma4) {
                return resolved.rootURL.path
            }
            return nil
        case .textChatDiffusionGemma:
            if let explicit = model {
                return explicit
            }
            return ManagedModelResolver.resolveInstalledModel(
                id: DiffusionGemmaResources.modelID
            )?.path
        case .textChatLaguna:
            if let explicit = model {
                if LagunaResources.isManagedIdentifier(explicit) {
                    let requestedID = LagunaResources.managedModelID(for: explicit)
                        ?? LagunaResources.modelID
                    guard let installed = ManagedModelResolver.resolveInstalledModel(
                        id: requestedID
                    ) else {
                        throw ValidationError(
                            "Model '\(requestedID)' is not installed. Run "
                                + "'mere.run model pull \(requestedID)' first."
                        )
                    }
                    return installed.path
                }
                return explicit
            }
            guard let installed = ManagedModelResolver.resolveInstalledModel(
                id: LagunaResources.modelID
            ) else {
                throw ValidationError(
                    "Model '\(LagunaResources.modelID)' is not installed. Run "
                        + "'mere.run model pull \(LagunaResources.modelID)' first."
                )
            }
            return installed.path
        case .textChatQ36, .textChatQ35:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.q36Nano) {
                return resolved.rootURL.path
            }
            // Allow the Qwen-family generator to auto-download from Hugging Face when model path is omitted.
            return nil
        case .textChatLFM2:
            if let explicit = model {
                return explicit
            }
            if let resolved = ModelResolver().resolveIfPresent(.lfm25A1B8Bit) {
                return resolved.rootURL.path
            }
            // Allow LFM2Generator to auto-download from Hugging Face when model path is omitted.
            return nil
        case .textChatDeepseekV4Flash:
            // DeepseekV4FlashGenerator resolves and (if needed) downloads its own GGUF.
            // Honor an explicit --model path if provided.
            return model
        case .textChatMuseGlimmer:
            if let explicit = model {
                return explicit
            }
            return ManagedModelResolver.resolveInstalledModel(id: MuseGlimmerResources.modelId)?.path
        case .textChatNemotronH:
            if let explicit = model {
                if NemotronHResources.handles(modelSpec: explicit) {
                    guard let installed = ManagedModelResolver.resolveInstalledModel(
                        id: NemotronHResources.modelID
                    ) else {
                        throw ValidationError(
                            "Model '\(NemotronHResources.modelID)' is not installed. Run "
                                + "'mere.run model pull \(NemotronHResources.modelID)' first."
                        )
                    }
                    return installed.path
                }
                return explicit
            }
            return ManagedModelResolver.resolveInstalledModel(id: NemotronHResources.modelID)?.path
        case .textChatNemotronOmni:
            if let explicit = model {
                if explicit == NemotronOmniResources.modelID {
                    guard let installed = ManagedModelResolver.resolveInstalledModel(
                        id: NemotronOmniResources.modelID
                    ) else {
                        throw ValidationError(
                            "Model '\(NemotronOmniResources.modelID)' is not installed. Run "
                                + "'mere.run model pull \(NemotronOmniResources.modelID) "
                                + "--accept-model-license' first."
                        )
                    }
                    return installed.path
                }
                return explicit
            }
            return ManagedModelResolver.resolveInstalledModel(id: NemotronOmniResources.modelID)?.path
        }
    }

    func resolveGemma4KVCacheQuantization() throws -> Gemma4KVCacheQuantization {
        let scheme = try resolveGemma4KVQuantizationScheme()
        return Gemma4KVCacheQuantization(
            bits: resolvedGemma4KVBits,
            scheme: scheme,
            groupSize: kvGroupSize ?? Gemma4Resources.defaultKVGroupSize,
            quantizedStart: resolvedGemma4QuantizedKVStart
        )
    }

    private var resolvedGemma4KVBits: Double? {
        kvBits
    }

    private func resolveGemma4KVQuantizationScheme() throws -> Gemma4KVQuantizationScheme {
        let raw = kvQuantScheme
            ?? Gemma4Resources.defaultKVQuantizationScheme.rawValue
        guard let scheme = Gemma4KVQuantizationScheme(
            rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        ) else {
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform', 'polar', or 'turboquant'.")
        }
        return scheme
    }

    private var resolvedGemma4QuantizedKVStart: Int {
        quantizedKVStart ?? Gemma4Resources.defaultQuantizedKVStart
    }

    func resolveAPIKey() -> String? {
        if let apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !apiKey.isEmpty {
            return apiKey
        }
        if let apiKey = ProcessInfo.processInfo.environment[Self.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        return nil
    }

    func validateServerSecurity(apiKey: String?) throws {
        guard rateLimitPerMinute > 0 else {
            throw ValidationError("--rate-limit-per-minute must be greater than zero.")
        }
        guard maxActiveRequests > 0 else {
            throw ValidationError("--max-active-requests must be greater than zero.")
        }
        if let memoryGuardCustomCeilingGB {
            guard memoryGuard == .custom else {
                throw ValidationError("--memory-guard-custom-ceiling-gb requires --memory-guard custom.")
            }
            guard memoryGuardCustomCeilingGB.isFinite, memoryGuardCustomCeilingGB > 0 else {
                throw ValidationError("--memory-guard-custom-ceiling-gb must be greater than zero.")
            }
        }
        guard (1...Int(Int32.max)).contains(contextSize) else {
            throw ValidationError("--context-size must be between 1 and \(Int(Int32.max)).")
        }
        guard Self.isLoopbackHost(host) || apiKey != nil else {
            throw ValidationError("Binding to non-loopback hosts requires --api-key or MERERUN_API_KEY.")
        }
    }

    func resolveMemoryPressurePolicy() throws -> RuntimeMemoryPressurePolicy {
        let gib = Double(1024 * 1024 * 1024)
        let customCeilingBytes = memoryGuardCustomCeilingGB.map { UInt64(($0 * gib).rounded(.down)) }
        if memoryGuard == .custom, customCeilingBytes == nil {
            throw ValidationError("--memory-guard custom requires --memory-guard-custom-ceiling-gb.")
        }
        return RuntimeMemoryPressurePolicy(
            tier: memoryGuard,
            customCeilingBytes: customCeilingBytes
        )
    }

    static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "127.0.0.1" || normalized == "localhost" || normalized == "::1"
    }

    func makePreflightEnvelope(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        now: @escaping () -> Date = Date.init
    ) -> APIServePreflightEnvelope {
        APIServePreflightAnalyzer(
            command: self,
            fileManager: fileManager,
            environment: environment,
            now: now
        ).envelope()
    }

    private func runPreflight() throws {
        let envelope = makePreflightEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            for diagnostic in envelope.diagnostics {
                print("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }
}

extension RuntimeMemoryGuardTier: ExpressibleByArgument {}

enum APIEngine: String, ExpressibleByArgument {
    case textCode = "text-code"
    case textChatKlein = "text-chat-klein"
    case textChatGemma4 = "text-chat-gemma4"
    case textChatDiffusionGemma = "text-chat-diffusiongemma"
    case textChatLaguna = "text-chat-laguna"
    case textChatQ36 = "text-chat-q36"
    case textChatQ35 = "text-chat-q35"
    case textChatLFM2 = "text-chat-lfm2"
    case textChatDeepseekV4Flash = "text-chat-deepseek-v4-flash"
    case textChatMuseGlimmer = "text-chat-muse-glimmer"
    case textChatNemotronH = "text-chat-nemotron-h"
    case textChatNemotronOmni = "text-chat-nemotron-omni"

    var runtimeServingEngine: RuntimeServingEngine {
        switch self {
        case .textCode:
            return .textCode
        case .textChatKlein:
            return .textChatKlein
        case .textChatGemma4:
            return .textChatGemma4
        case .textChatDiffusionGemma:
            return .textChatDiffusionGemma
        case .textChatLaguna:
            return .textChatLaguna
        case .textChatQ36:
            return .textChatQ36
        case .textChatQ35:
            return .textChatQ36
        case .textChatLFM2:
            return .textChatLFM2
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash
        case .textChatMuseGlimmer:
            return .textChatMuseGlimmer
        case .textChatNemotronH:
            return .textChatNemotronH
        case .textChatNemotronOmni:
            return .textChatNemotronOmni
        }
    }

    var openAICompatibility: APIEngineCapabilities {
        runtimeServingEngine.openAICompatibility
    }
}
