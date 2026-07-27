import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Hummingbird
import NIOCore
import AudioCore
import AudioSTT
import AudioTTS
import MediaIO
import MereRunCore

struct APIServe: AsyncParsableCommand {
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

          # Start a Qwen3.6 text-chat server with an explicit model root
          mere.run api serve --engine text-chat-q36 -m ~/Models/text-chat-q36-nano

          # Start an LFM2.5 text-chat server
          mere.run api serve --engine text-chat-lfm2

          # Start the DeepSeek V4 Flash OpenAI-compatible server
          mere.run api serve --engine text-chat-deepseek-v4-flash

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

    @Option(name: [.customShort("m"), .long, .customLong("model-path")], help: "Model path. For --engine text-code, pass a GGUF file. For --engine text-chat-klein, pass a Klein-root text chat model. For --engine text-chat-gemma4, pass a Gemma 4 model root or repo ID. For --engine text-chat-q36, pass a Qwen3.6 text chat model root. For --engine text-chat-lfm2, pass an LFM2 MLX model root or repo ID. For --engine text-chat-deepseek-v4-flash, pass a DS4 GGUF file or managed model root.")
    var model: String?

    @Option(name: [.long], help: "Serving engine: text-chat-q36 (default; serves text-chat-q36-nano), text-code, text-chat-klein, text-chat-gemma4, text-chat-lfm2, or text-chat-deepseek-v4-flash.")
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
            memoryPressurePolicy: memoryPressurePolicy
        )
        try await server.run(host: host, port: port)
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
        case .textChatQ36, .textChatQ35:
            return ModelResolver.ModelID.q36Nano.rawValue
        case .textChatLFM2:
            return LFM2Resources.defaultModelId
        case .textCode:
            return CodeGenResources.defaultModelId
        case .textChatDeepseekV4Flash:
            return DeepseekV4FlashResources.defaultModelId
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

    private var requestedGemma4ModelSpec: String {
        model?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? Gemma4Resources.defaultModelId
    }

    private var usesGemma4TurboKVDefaults: Bool {
        Gemma4Resources.usesTurboDefaults(modelSpec: requestedGemma4ModelSpec)
            && Gemma4Resources.supportsDefaultTurboKVQuantization
    }

    private var resolvedGemma4KVBits: Double? {
        kvBits ?? (usesGemma4TurboKVDefaults ? Gemma4Resources.defaultTurboKVBits : nil)
    }

    private func resolveGemma4KVQuantizationScheme() throws -> Gemma4KVQuantizationScheme {
        let raw = kvQuantScheme
            ?? (usesGemma4TurboKVDefaults
                ? Gemma4Resources.defaultTurboKVQuantizationScheme.rawValue
                : Gemma4Resources.defaultKVQuantizationScheme.rawValue)
        guard let scheme = Gemma4KVQuantizationScheme(
            rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        ) else {
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform', 'polar', or 'turboquant'.")
        }
        return scheme
    }

    private var resolvedGemma4QuantizedKVStart: Int {
        quantizedKVStart ?? (usesGemma4TurboKVDefaults
            ? Gemma4Resources.defaultTurboQuantizedKVStart
            : Gemma4Resources.defaultQuantizedKVStart)
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
    case textChatQ36 = "text-chat-q36"
    case textChatQ35 = "text-chat-q35"
    case textChatLFM2 = "text-chat-lfm2"
    case textChatDeepseekV4Flash = "text-chat-deepseek-v4-flash"

    var runtimeServingEngine: RuntimeServingEngine {
        switch self {
        case .textCode:
            return .textCode
        case .textChatKlein:
            return .textChatKlein
        case .textChatGemma4:
            return .textChatGemma4
        case .textChatQ36:
            return .textChatQ36
        case .textChatQ35:
            return .textChatQ36
        case .textChatLFM2:
            return .textChatLFM2
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash
        }
    }

    var openAICompatibility: APIEngineCapabilities {
        runtimeServingEngine.openAICompatibility
    }
}

struct APIEngineCapabilities: Equatable, Sendable {
    var supportsRawProxy: Bool = false
    var supportsTools: Bool = false
    var supportsToolChoice: Bool = false
    var supportsDeveloperRole: Bool = true
    var supportsStructuredOutputs: Bool = false
    var supportsReasoningEffort: Bool = false
    var supportsMaxCompletionTokens: Bool = true
    var supportsUsageInStreaming: Bool = true
    var supportsVisionContentParts: Bool = false
    var supportsStrictMode: Bool = false
    var supportsStopSequences: Bool = false
    var supportsSeed: Bool = false
    var supportsPenalties: Bool = false
    var supportsLogprobs: Bool = false
    var supportsProviderThinkingControls: Bool = false

    static let localText = APIEngineCapabilities()

    static let localTextWithStopSequences = APIEngineCapabilities(
        supportsStopSequences: true
    )

    static let localTextWithStructuredJSON = APIEngineCapabilities(
        supportsStructuredOutputs: true
    )

    static let localTextWithTools = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true
    )

    static let localTextWithToolsAndStructuredJSON = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsStructuredOutputs: true
    )

    static let localTextWithToolsAndVision = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsVisionContentParts: true
    )

    static let localTextWithToolsVisionAndStructuredJSON = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsStructuredOutputs: true,
        supportsVisionContentParts: true,
        supportsStrictMode: false
    )

    static let rawProxy = APIEngineCapabilities(
        supportsRawProxy: true,
        supportsTools: true,
        supportsToolChoice: true,
        supportsDeveloperRole: true,
        supportsStructuredOutputs: true,
        supportsReasoningEffort: true,
        supportsMaxCompletionTokens: true,
        supportsUsageInStreaming: true,
        supportsVisionContentParts: true,
        supportsStrictMode: false,
        supportsStopSequences: true,
        supportsSeed: true,
        supportsPenalties: true,
        supportsLogprobs: true,
        supportsProviderThinkingControls: true
    )
}

struct APIHealthStatus: Codable, Equatable, Sendable {
    let status: String
}

struct APIGeometryArtifactResponse: Codable, Equatable, Sendable {
    let kind: GeometryArtifactKind
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIGeometryTimingResponse: Codable, Equatable, Sendable {
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessSeconds: Double

    enum CodingKeys: String, CodingKey {
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
        case postprocessSeconds = "postprocess_seconds"
    }
}

struct APIGeometryResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let width: Int
    let height: Int
    let units: GeometryValueUnits
    let coordinateSystem: GeometryCoordinateSystem
    let camera: GeometryCameraManifest
    let depthStatistics: GeometryDepthStatistics
    let focal: Double
    let shift: Double
    let metricScale: Double
    let tokenCount: Int
    let manifestURL: String
    let artifacts: [APIGeometryArtifactResponse]
    let timing: APIGeometryTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case width
        case height
        case units
        case coordinateSystem = "coordinate_system"
        case camera
        case depthStatistics = "depth_statistics"
        case focal
        case shift
        case metricScale = "metric_scale"
        case tokenCount = "token_count"
        case manifestURL = "manifest_url"
        case artifacts
        case timing
    }
}

struct APIDepthVideoArtifactResponse: Codable, Equatable, Sendable {
    let kind: String
    let frameIndex: Int?
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case frameIndex = "frame_index"
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIDepthVideoTimingResponse: Codable, Equatable, Sendable {
    let checkpointVerificationSeconds: Double
    let frameExtractionSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let exportSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case checkpointVerificationSeconds = "checkpoint_verification_seconds"
        case frameExtractionSeconds = "frame_extraction_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
        case exportSeconds = "export_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct APIDepthVideoResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let semantics: DepthSemantics
    let checkpointFormat: VideoDepthAnythingCheckpointFormat
    let checkpointSHA256: String
    let width: Int
    let height: Int
    let fps: Double
    let frameCount: Int
    let windowCount: Int
    let temporalWindowLength: Int
    let temporalOverlap: Int
    let hasConfidence: Bool
    let hasCameraIntrinsics: Bool
    let hasCameraExtrinsics: Bool
    let hasPointCloud: Bool
    let manifest: APIDepthVideoArtifactResponse
    let review: APIDepthVideoArtifactResponse
    let artifacts: [APIDepthVideoArtifactResponse]
    let timing: APIDepthVideoTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case semantics
        case checkpointFormat = "checkpoint_format"
        case checkpointSHA256 = "checkpoint_sha256"
        case width
        case height
        case fps
        case frameCount = "frame_count"
        case windowCount = "window_count"
        case temporalWindowLength = "temporal_window_length"
        case temporalOverlap = "temporal_overlap"
        case hasConfidence = "has_confidence"
        case hasCameraIntrinsics = "has_camera_intrinsics"
        case hasCameraExtrinsics = "has_camera_extrinsics"
        case hasPointCloud = "has_point_cloud"
        case manifest
        case review
        case artifacts
        case timing
    }
}

struct APIMultiViewGeometryCheckpointResponse: Codable, Equatable, Sendable {
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let weightsByteCount: Int64
    let weightsSHA256: String
    let configurationByteCount: Int64
    let configurationSHA256: String

    enum CodingKeys: String, CodingKey {
        case repository
        case revision
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case license
        case weightsByteCount = "weights_byte_count"
        case weightsSHA256 = "weights_sha256"
        case configurationByteCount = "configuration_byte_count"
        case configurationSHA256 = "configuration_sha256"
    }
}

struct APIMultiViewGeometryArtifactResponse: Codable, Equatable, Sendable {
    let kind: String
    let viewIndex: Int?
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case viewIndex = "view_index"
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIMultiViewGeometryCameraResponse: Codable, Equatable, Sendable {
    let viewIndex: Int
    let width: Int
    let height: Int
    let intrinsics: GeometryCameraIntrinsics
    let extrinsics: GeometryCameraExtrinsics
    let selectedPointCount: Int

    enum CodingKeys: String, CodingKey {
        case viewIndex = "view_index"
        case width
        case height
        case intrinsics
        case extrinsics
        case selectedPointCount = "selected_point_count"
    }
}

struct APIMultiViewGeometryTimingResponse: Codable, Equatable, Sendable {
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessingSeconds: Double
    let exportSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case checkpointVerificationSeconds = "checkpoint_verification_seconds"
        case decodingSeconds = "decoding_seconds"
        case preprocessingSeconds = "preprocessing_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
        case postprocessingSeconds = "postprocessing_seconds"
        case exportSeconds = "export_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct APIMultiViewGeometryResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let checkpoint: APIMultiViewGeometryCheckpointResponse
    let units: GeometryValueUnits
    let coordinateSystem: GeometryCoordinateSystem
    let poseConditioned: Bool
    let cameraSemantics: DepthAnything3CameraSemantics
    let cameraScaleAlignment: String
    let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    let depthScaleDivisor: Float
    let processResolution: Int
    let confidencePercentile: Double
    let confidenceThreshold: Float
    let viewCount: Int
    let cameraCount: Int
    let pointCount: Int
    let pointCloudRepresentation: String
    let containsMesh: Bool
    let containsGaussianParameters: Bool
    let threeDGaussianHandoff: Geometry3DGSHandoffManifest
    let cameras: [APIMultiViewGeometryCameraResponse]
    let manifest: APIMultiViewGeometryArtifactResponse
    let artifacts: [APIMultiViewGeometryArtifactResponse]
    let timing: APIMultiViewGeometryTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case checkpoint
        case units
        case coordinateSystem = "coordinate_system"
        case poseConditioned = "pose_conditioned"
        case cameraSemantics = "camera_semantics"
        case cameraScaleAlignment = "camera_scale_alignment"
        case referenceViewStrategy = "reference_view_strategy"
        case depthScaleDivisor = "depth_scale_divisor"
        case processResolution = "process_resolution"
        case confidencePercentile = "confidence_percentile"
        case confidenceThreshold = "confidence_threshold"
        case viewCount = "view_count"
        case cameraCount = "camera_count"
        case pointCount = "point_count"
        case pointCloudRepresentation = "point_cloud_representation"
        case containsMesh = "contains_mesh"
        case containsGaussianParameters = "contains_gaussian_parameters"
        case threeDGaussianHandoff = "three_d_gaussian_handoff"
        case cameras
        case manifest
        case artifacts
        case timing
    }
}

struct APIImageTo3DCheckpointResponse: Codable, Equatable, Sendable {
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let format: TripoSRCheckpointFormat
    let weightsByteCount: Int64
    let weightsSHA256: String
    let sourceSHA256: String
    let configurationSHA256: String

    enum CodingKeys: String, CodingKey {
        case repository
        case revision
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case license
        case format
        case weightsByteCount = "weights_byte_count"
        case weightsSHA256 = "weights_sha256"
        case sourceSHA256 = "source_sha256"
        case configurationSHA256 = "configuration_sha256"
    }
}

struct APIImageTo3DArtifactResponse: Codable, Equatable, Sendable {
    let kind: String
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIImageTo3DTimingResponse: Codable, Equatable, Sendable {
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let sceneEncodingSeconds: Double
    let meshExtractionSeconds: Double
    let exportSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case checkpointVerificationSeconds = "checkpoint_verification_seconds"
        case decodingSeconds = "decoding_seconds"
        case preprocessingSeconds = "preprocessing_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case sceneEncodingSeconds = "scene_encoding_seconds"
        case meshExtractionSeconds = "mesh_extraction_seconds"
        case exportSeconds = "export_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct APIImageTo3DResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let checkpoint: APIImageTo3DCheckpointResponse
    let sourceWidth: Int
    let sourceHeight: Int
    let preparedWidth: Int
    let preparedHeight: Int
    let foregroundPolicy: String
    let foregroundRatio: Float?
    let croppedTransparentForeground: Bool
    let extractionResolution: Int
    let densityThreshold: Float
    let includesVertexColors: Bool
    let meshExtractionAlgorithm: String
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let vertexCount: Int
    let triangleCount: Int
    let bounds: MeshBounds
    let manifest: APIImageTo3DArtifactResponse
    let meshManifest: APIImageTo3DArtifactResponse
    let artifacts: [APIImageTo3DArtifactResponse]
    let timing: APIImageTo3DTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case checkpoint
        case sourceWidth = "source_width"
        case sourceHeight = "source_height"
        case preparedWidth = "prepared_width"
        case preparedHeight = "prepared_height"
        case foregroundPolicy = "foreground_policy"
        case foregroundRatio = "foreground_ratio"
        case croppedTransparentForeground = "cropped_transparent_foreground"
        case extractionResolution = "extraction_resolution"
        case densityThreshold = "density_threshold"
        case includesVertexColors = "includes_vertex_colors"
        case meshExtractionAlgorithm = "mesh_extraction_algorithm"
        case coordinateSystem = "coordinate_system"
        case units
        case inferredUnseenGeometry = "inferred_unseen_geometry"
        case vertexCount = "vertex_count"
        case triangleCount = "triangle_count"
        case bounds
        case manifest
        case meshManifest = "mesh_manifest"
        case artifacts
        case timing
    }
}

enum APIServerContract {
    static let defaultMaxTokens = 2048
    static let maxEmbeddingInputCount = 256
    static let maxEmbeddingInputUTF8Bytes = 2 * 1_024 * 1_024
    static let maxImageInferenceSteps = 100
    static let maxSpeechPromptUTF8Bytes = 32 * 1_024
    static let maxTranscriptionTokens = 4_096
    static let defaultImageModelID = ModelResolver.ModelID.zetaNano.rawValue
    static let defaultSpeechModelID = Qwen3TTSResources.defaultModelId
    static let defaultTranscriptionModelID = ParakeetResources.defaultModelId
    static let defaultGeometryModelID = ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue
    static let defaultMultiViewGeometryModelID = ModelResolver.ModelID.visionGeometryDA3Small.rawValue
    static let defaultImageTo3DModelID = ModelResolver.ModelID.image3DTripoSR.rawValue
    static let defaultDepthVideoModelID = ModelResolver.ModelID.visionDepthVDASmall.rawValue
    static let geometryRoutePath = "/v1/vision/geometry"
    static let geometryRouterPath = RouterPath(geometryRoutePath)
    static let multiViewGeometryRoutePath = "/v1/vision/geometry/multiview"
    static let multiViewGeometryRouterPath = RouterPath(multiViewGeometryRoutePath)
    static let maximumMultiViewGeometryUploadByteCount = 512 * 1024 * 1024
    static let imageTo3DRoutePath = "/v1/vision/image-to-3d"
    static let imageTo3DRouterPath = RouterPath(imageTo3DRoutePath)
    static let maximumImageTo3DUploadByteCount = 100 * 1024 * 1024
    static let depthVideoRoutePath = "/v1/vision/depth-video"
    static let depthVideoRouterPath = RouterPath(depthVideoRoutePath)
    static let maximumDepthVideoUploadByteCount = 512 * 1024 * 1024

    static func decodeImageGenerationRequest(from data: Data) throws -> OpenAIImageGenerationRequest {
        try decodeJSONRequest(OpenAIImageGenerationRequest.self, from: data)
    }

    static func decodeSpeechRequest(from data: Data) throws -> OpenAIAudioSpeechRequest {
        try decodeJSONRequest(OpenAIAudioSpeechRequest.self, from: data)
    }

    struct ImageGenerationPlan: Equatable, Sendable {
        let modelID: String
        let prompt: String
        let width: Int
        let height: Int
        let responseFormat: String
        let seed: UInt64?
        let negativePrompt: String?
        let steps: Int?
        let guidanceScale: Double?
        let inputImage: URL?
        let additionalInputImages: [URL]
        let maskImage: URL?
        let strength: Double?

        init(
            modelID: String,
            prompt: String,
            width: Int,
            height: Int,
            responseFormat: String,
            seed: UInt64?,
            negativePrompt: String?,
            steps: Int?,
            guidanceScale: Double?,
            inputImage: URL?,
            additionalInputImages: [URL] = [],
            maskImage: URL? = nil,
            strength: Double?
        ) {
            self.modelID = modelID
            self.prompt = prompt
            self.width = width
            self.height = height
            self.responseFormat = responseFormat
            self.seed = seed
            self.negativePrompt = negativePrompt
            self.steps = steps
            self.guidanceScale = guidanceScale
            self.inputImage = inputImage
            self.additionalInputImages = additionalInputImages
            self.maskImage = maskImage
            self.strength = strength
        }
    }

    struct SpeechPlan: Equatable, Sendable {
        let modelID: String
        let input: String
        let voiceDescription: String
        let responseFormat: String
        let speed: Float
        let temperature: Float
    }

    struct TranscriptionPlan: Equatable, Sendable {
        let modelID: String
        let language: String?
        let responseFormat: String
        let task: ASRTask
        let maxTokens: Int
    }

    struct GeometryPlan: Equatable, Sendable {
        let modelID: String
        let resolutionLevel: Int
        let tokenCount: Int?
        let maximumPointCount: Int?

        var configuration: MoGe2InferenceConfiguration {
            MoGe2InferenceConfiguration(
                resolutionLevel: resolutionLevel,
                tokenCount: tokenCount,
                maximumPointCount: maximumPointCount
            )
        }
    }

    struct MultiViewGeometryPlan: Equatable, Sendable {
        let modelID: String
        let processResolution: Int
        let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
        let confidencePercentile: Double
        let maximumPointCount: Int
        let knownCameras: [DepthAnything3KnownCamera]?

        var poseConditioned: Bool { knownCameras != nil }
    }

    struct ImageTo3DPlan: Equatable, Sendable {
        let modelID: String
        let extractionResolution: Int
        let densityThreshold: Float
        let foregroundRatio: Float
        let alreadyFramed: Bool
        let includesVertexColors: Bool

        var foregroundPolicy: TripoSRForegroundPolicy {
            alreadyFramed
                ? .alreadyFramed
                : .automaticTransparentAlpha(foregroundRatio: foregroundRatio)
        }
    }

    struct DepthVideoPlan: Equatable, Sendable {
        let modelID: String
        let inputSize: Int
        let maximumFrameCount: Int
    }

    static func healthStatus() -> APIHealthStatus {
        APIHealthStatus(status: "ok")
    }

    static func modelsResponse(modelId: String, createdAt: Date = Date()) -> OpenAIModelsResponse {
        modelsResponse(modelIds: [modelId], createdAt: createdAt)
    }

    static func modelsResponse(modelIds: [String], createdAt: Date = Date()) -> OpenAIModelsResponse {
        OpenAIModelsResponse(
            object: "list",
            data: modelIds.map {
                OpenAIModel(
                    id: $0,
                    object: "model",
                    created: Int(createdAt.timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            }
        )
    }

    static func embeddingTexts(from request: OpenAIEmbeddingRequest) throws -> [String] {
        guard !request.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw APIRequestValidationError.invalidField("model", "must not be empty")
        }
        if let encodingFormat = request.encoding_format?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
           !encodingFormat.isEmpty,
           encodingFormat != "float" {
            throw APIRequestValidationError.invalidField(
                "encoding_format",
                "only float embeddings are supported"
            )
        }
        if request.dimensions != nil {
            throw APIRequestValidationError.invalidField(
                "dimensions",
                "dimension overrides are not supported by this embedding model"
            )
        }

        let texts = request.input.texts
        guard !texts.isEmpty else {
            throw APIRequestValidationError.invalidField("input", "must contain at least one text")
        }
        guard texts.count <= maxEmbeddingInputCount else {
            throw APIRequestValidationError.invalidField(
                "input",
                "must contain at most \(maxEmbeddingInputCount) texts"
            )
        }
        var totalUTF8Bytes = 0
        for text in texts {
            let textBytes = text.utf8.count
            guard textBytes <= maxEmbeddingInputUTF8Bytes - totalUTF8Bytes else {
                throw APIRequestValidationError.invalidField(
                    "input",
                    "UTF-8 content must total at most \(maxEmbeddingInputUTF8Bytes) bytes"
                )
            }
            totalUTF8Bytes += textBytes
        }
        return texts
    }

    static func embeddingResponse(
        modelId: String,
        embeddings: [[Float]],
        tokenCounts: [Int]
    ) -> OpenAIEmbeddingResponse {
        let promptTokens = tokenCounts.reduce(0, +)
        return OpenAIEmbeddingResponse(
            model: modelId,
            data: embeddings.enumerated().map { index, vector in
                OpenAIEmbeddingDatum(index: index, embedding: vector)
            },
            usage: OpenAIEmbeddingUsage(
                prompt_tokens: promptTokens,
                total_tokens: promptTokens
            )
        )
    }

    static func companionModelIDs(
        fileManager: FileManager = .default,
        installedModelIDs: Set<String>? = nil,
        includeLoopbackArtifactModels: Bool = true
    ) -> [String] {
        let categories: Set<ManagedModelCategory> = [
            .image, .image3D, .speechTTS, .speechASR, .textEmbed, .visionGeometry, .visionDepth,
        ]
        let ids = ManagedModelCatalog.allSpecs
            .filter { categories.contains($0.category) }
            .filter {
                includeLoopbackArtifactModels
                    || !APIVFXArtifactRoutePolicy.modelIDs.contains($0.id)
            }
            .filter { isCompanionModelInstalled($0, fileManager: fileManager, installedModelIDs: installedModelIDs) }
            .map(\.id)
        var uniqueIDs = Set(ids)
        if isQwenImageEditInstalled(fileManager: fileManager, installedModelIDs: installedModelIDs) {
            uniqueIDs.insert(QwenImageEditRepository.modelId)
        }
        return Array(uniqueIDs).sorted()
    }

    static func geometryPlan(from form: MultipartFormData) throws -> GeometryPlan {
        let modelID = normalizedModelID(form.field("model"), defaultID: defaultGeometryModelID)
        guard modelID == defaultGeometryModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only \(defaultGeometryModelID) is supported"
            )
        }
        let resolutionLevel: Int
        if let raw = normalizedOptional(form.field("resolution_level")) {
            guard let value = Int(raw), (0...9).contains(value) else {
                throw APIRequestValidationError.invalidField(
                    "resolution_level",
                    "must be an integer between 0 and 9"
                )
            }
            resolutionLevel = value
        } else {
            resolutionLevel = 9
        }
        let tokenCount = try optionalPositiveIntField(form.field("token_count"), field: "token_count")
        if let tokenCount,
           (tokenCount < MoGe2InferenceConfiguration.minimumTokenCount
            || tokenCount > MoGe2InferenceConfiguration.maximumTokenCount) {
            throw APIRequestValidationError.invalidField(
                "token_count",
                "must be an integer between 1 and 3600"
            )
        }
        return GeometryPlan(
            modelID: modelID,
            resolutionLevel: resolutionLevel,
            tokenCount: tokenCount,
            maximumPointCount: try optionalPositiveIntField(form.field("max_points"), field: "max_points")
        )
    }

    static func geometryResponse(
        from result: MoGe2RunResult,
        createdAt: Date = Date()
    ) throws -> APIGeometryResponse {
        let manifest = result.export.manifest
        let root = URL(fileURLWithPath: manifest.outputDirectory, isDirectory: true)
        var artifacts = manifest.artifacts.map { artifact in
            APIGeometryArtifactResponse(
                kind: artifact.kind,
                url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                mediaType: artifact.mediaType,
                byteCount: artifact.byteCount,
                sha256: artifact.sha256
            )
        }
        let manifestURL = result.export.manifestURL
        let attributes = try FileManager.default.attributesOfItem(atPath: manifestURL.path)
        artifacts.append(
            APIGeometryArtifactResponse(
                kind: .manifest,
                url: manifestURL.absoluteString,
                mediaType: "application/json",
                byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                sha256: try ModelArtifactPin.fileSHA256(manifestURL)
            )
        )
        artifacts.sort { lhs, rhs in
            if lhs.kind.rawValue == rhs.kind.rawValue { return lhs.url < rhs.url }
            return lhs.kind.rawValue < rhs.kind.rawValue
        }
        return APIGeometryResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.geometry",
            status: "completed",
            model: manifest.model.modelID,
            width: manifest.width,
            height: manifest.height,
            units: manifest.units,
            coordinateSystem: manifest.coordinateSystem,
            camera: manifest.camera,
            depthStatistics: manifest.depthStatistics,
            focal: result.focalShift.focal,
            shift: result.focalShift.shift,
            metricScale: result.metricScale,
            tokenCount: result.tokenCount,
            manifestURL: manifestURL.absoluteString,
            artifacts: artifacts,
            timing: APIGeometryTimingResponse(
                modelLoadSeconds: result.modelLoadSeconds,
                inferenceSeconds: result.inferenceSeconds,
                postprocessSeconds: result.postprocessSeconds
            )
        )
    }

    static func multiViewGeometryPlan(from form: MultipartFormData) throws -> MultiViewGeometryPlan {
        let allowedTextFields: Set<String> = [
            "model",
            "process_resolution",
            "reference_view",
            "confidence_percentile",
            "max_points",
            "cameras",
        ]
        let allowedFileFields: Set<String> = ["image", "image[]", "cameras"]
        for part in form.parts {
            if part.filename != nil {
                guard allowedFileFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "unsupported file part; only uploaded image/image[] and cameras JSON files are accepted"
                    )
                }
            } else {
                guard allowedTextFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "unsupported field; client input, output, model, and camera filesystem paths are not accepted"
                    )
                }
                guard String(data: part.body, encoding: .utf8) != nil else {
                    throw APIRequestValidationError.invalidField(part.name, "must contain valid UTF-8 text")
                }
            }
        }
        for field in allowedTextFields
        where form.parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }

        // Multipart order is the view order. Do not regroup image and image[]
        // aliases because cameras are indexed against this exact sequence.
        let imageUploads = form.parts.filter {
            $0.filename != nil && ($0.name == "image" || $0.name == "image[]")
        }
        guard !imageUploads.isEmpty, imageUploads.allSatisfy({ !$0.body.isEmpty }) else {
            throw APIRequestValidationError.invalidField(
                "image",
                "one or more non-empty uploaded image/image[] files are required"
            )
        }
        for image in imageUploads {
            if let rawContentType = image.contentType {
                let contentType = rawContentType
                    .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                    .first?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased() ?? ""
                guard contentType.hasPrefix("image/") || contentType == "application/octet-stream" else {
                    throw APIRequestValidationError.invalidField(
                        image.name,
                        "uploaded parts must have an image content type"
                    )
                }
            }
        }

        let modelID = normalizedModelID(
            form.field("model"),
            defaultID: defaultMultiViewGeometryModelID
        ).lowercased()
        guard modelID == defaultMultiViewGeometryModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only the managed model id \(defaultMultiViewGeometryModelID) is supported"
            )
        }

        let processResolution = try optionalPositiveIntField(
            form.field("process_resolution"),
            field: "process_resolution"
        ) ?? 504
        do {
            try DepthAnything3Limits.validateRequest(
                viewCount: imageUploads.count,
                processResolution: processResolution
            )
        } catch let error as DepthAnything3LimitError {
            let field: String
            switch error {
            case .processResolutionOutOfRange:
                field = "process_resolution"
            case .viewCountOutOfRange, .processedPixelBudgetExceeded,
                 .invalidSourceDimensions, .sourcePixelBudgetExceeded,
                 .totalSourcePixelBudgetExceeded, .encodedByteBudgetExceeded,
                 .totalEncodedByteBudgetExceeded:
                field = "image"
            }
            throw APIRequestValidationError.invalidField(field, error.localizedDescription)
        }
        let maximumPointCount = try optionalPositiveIntField(
            form.field("max_points"),
            field: "max_points"
        ) ?? 1_000_000

        let confidencePercentile: Double
        if let raw = normalizedOptional(form.field("confidence_percentile")) {
            guard let value = Double(raw), value.isFinite, (0...100).contains(value) else {
                throw APIRequestValidationError.invalidField(
                    "confidence_percentile",
                    "must be a finite number between 0 and 100"
                )
            }
            confidencePercentile = value
        } else {
            confidencePercentile = 40
        }

        let referenceViewRaw = normalizedOptional(form.field("reference_view"))?.lowercased()
            ?? DepthAnything3ReferenceViewStrategy.saddleBalanced.rawValue
        guard let referenceViewStrategy = DepthAnything3ReferenceViewStrategy(
            rawValue: referenceViewRaw
        ) else {
            throw APIRequestValidationError.invalidField(
                "reference_view",
                "must be one of \(DepthAnything3ReferenceViewStrategy.allCases.map(\.rawValue).joined(separator: ", "))"
            )
        }

        let cameraFileParts = form.files(named: "cameras")
        let cameraTextParts = form.parts.filter { $0.name == "cameras" && $0.filename == nil }
        guard cameraFileParts.count <= 1 else {
            throw APIRequestValidationError.invalidField("cameras", "must be supplied at most once")
        }
        guard cameraFileParts.isEmpty || cameraTextParts.isEmpty else {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "supply either an uploaded JSON document or an inline JSON document, not both"
            )
        }
        if let cameraFile = cameraFileParts.first {
            guard !cameraFile.body.isEmpty else {
                throw APIRequestValidationError.invalidField("cameras", "uploaded JSON document must not be empty")
            }
            if let rawContentType = cameraFile.contentType {
                let contentType = rawContentType
                    .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                    .first?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased() ?? ""
                guard contentType == "application/json"
                    || contentType.hasSuffix("+json")
                    || contentType == "application/octet-stream"
                    || contentType == "text/json" else {
                    throw APIRequestValidationError.invalidField(
                        "cameras",
                        "uploaded camera document must have a JSON content type"
                    )
                }
            }
        }
        let cameraData = cameraFileParts.first?.body ?? cameraTextParts.first?.body
        let knownCameras = try cameraData.map {
            try decodeMultiViewCameraDocument($0, expectedCount: imageUploads.count)
        }

        return MultiViewGeometryPlan(
            modelID: modelID,
            processResolution: processResolution,
            referenceViewStrategy: referenceViewStrategy,
            confidencePercentile: confidencePercentile,
            maximumPointCount: maximumPointCount,
            knownCameras: knownCameras
        )
    }

    static func multiViewGeometryResponse(
        from result: DepthAnything3RunResult,
        export: MultiViewGeometryExportResult,
        exportSeconds: Double,
        createdAt: Date = Date()
    ) throws -> APIMultiViewGeometryResponse {
        let scene = export.manifest
        let root = URL(fileURLWithPath: scene.outputDirectory, isDirectory: true)
        let manifest = try multiViewGeometryFileArtifact(
            kind: "manifest",
            viewIndex: nil,
            url: export.manifestURL,
            mediaType: "application/json"
        )
        let artifacts = scene.artifacts.map { artifact in
            APIMultiViewGeometryArtifactResponse(
                kind: artifact.kind.rawValue,
                viewIndex: artifact.viewIndex,
                url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                mediaType: artifact.mediaType,
                byteCount: artifact.byteCount,
                sha256: artifact.sha256
            )
        }.sorted { lhs, rhs in
            if lhs.viewIndex != rhs.viewIndex {
                return (lhs.viewIndex ?? Int.max) < (rhs.viewIndex ?? Int.max)
            }
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.url < rhs.url
        }
        let cameras = try scene.views.map { view in
            guard let extrinsics = view.camera.extrinsics else {
                throw APIRequestValidationError.invalidField(
                    "result",
                    "multi-view camera \(view.index) is missing extrinsics"
                )
            }
            return APIMultiViewGeometryCameraResponse(
                viewIndex: view.index,
                width: view.width,
                height: view.height,
                intrinsics: view.camera.intrinsics,
                extrinsics: extrinsics,
                selectedPointCount: view.selectedPointCount
            )
        }.sorted { $0.viewIndex < $1.viewIndex }
        let totalSeconds = result.checkpointVerificationSeconds
            + result.decodingSeconds
            + result.preprocessingSeconds
            + result.modelLoadSeconds
            + result.inferenceSeconds
            + result.postprocessingSeconds
            + exportSeconds
        return APIMultiViewGeometryResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.geometry.multiview",
            status: "completed",
            model: result.checkpoint.modelID,
            checkpoint: APIMultiViewGeometryCheckpointResponse(
                repository: result.checkpoint.repository,
                revision: result.checkpoint.revision,
                sourceRepository: result.checkpoint.sourceRepository,
                sourceRevision: result.checkpoint.sourceRevision,
                license: result.checkpoint.license,
                weightsByteCount: result.checkpoint.weightsByteCount,
                weightsSHA256: result.checkpoint.weightsSHA256,
                configurationByteCount: result.checkpoint.configurationByteCount,
                configurationSHA256: result.checkpoint.configurationSHA256
            ),
            units: scene.units,
            coordinateSystem: scene.coordinateSystem,
            poseConditioned: scene.poseConditioned,
            cameraSemantics: result.cameraSemantics,
            cameraScaleAlignment: result.cameraScaleAlignment,
            referenceViewStrategy: result.referenceViewStrategy,
            depthScaleDivisor: result.depthScaleDivisor,
            processResolution: result.processResolution,
            confidencePercentile: scene.confidencePercentile,
            confidenceThreshold: scene.confidenceThreshold,
            viewCount: scene.views.count,
            cameraCount: cameras.count,
            pointCount: scene.pointCount,
            pointCloudRepresentation: scene.pointCloudRepresentation,
            containsMesh: false,
            containsGaussianParameters: scene.threeDGaussianHandoff.containsGaussianParameters,
            threeDGaussianHandoff: scene.threeDGaussianHandoff,
            cameras: cameras,
            manifest: manifest,
            artifacts: artifacts,
            timing: APIMultiViewGeometryTimingResponse(
                checkpointVerificationSeconds: result.checkpointVerificationSeconds,
                decodingSeconds: result.decodingSeconds,
                preprocessingSeconds: result.preprocessingSeconds,
                modelLoadSeconds: result.modelLoadSeconds,
                inferenceSeconds: result.inferenceSeconds,
                postprocessingSeconds: result.postprocessingSeconds,
                exportSeconds: exportSeconds,
                totalSeconds: totalSeconds
            )
        )
    }

    private static func decodeMultiViewCameraDocument(
        _ data: Data,
        expectedCount: Int
    ) throws -> [DepthAnything3KnownCamera] {
        let document: DepthAnything3CameraDocument
        do {
            document = try JSONDecoder().decode(DepthAnything3CameraDocument.self, from: data)
        } catch {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "must be a valid schemaVersion 1 camera JSON document"
            )
        }
        guard document.schemaVersion == 1 else {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "unsupported schemaVersion \(document.schemaVersion); expected 1"
            )
        }
        guard document.cameras.count == expectedCount else {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "contains \(document.cameras.count) cameras for \(expectedCount) uploaded images"
            )
        }
        for (index, camera) in document.cameras.enumerated() {
            do {
                try DepthAnything3CameraValidation.validate(camera, index: index)
            } catch {
                throw APIRequestValidationError.invalidField(
                    "cameras",
                    error.localizedDescription
                )
            }
        }
        return document.cameras
    }

    private static func multiViewGeometryFileArtifact(
        kind: String,
        viewIndex: Int?,
        url: URL,
        mediaType: String
    ) throws -> APIMultiViewGeometryArtifactResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return APIMultiViewGeometryArtifactResponse(
            kind: kind,
            viewIndex: viewIndex,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    static func imageTo3DPlan(from form: MultipartFormData) throws -> ImageTo3DPlan {
        let allowedTextFields: Set<String> = [
            "model",
            "resolution",
            "density_threshold",
            "foreground_ratio",
            "already_framed",
            "vertex_colors",
        ]
        for part in form.parts {
            if part.filename != nil {
                guard part.name == "image" else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "only one uploaded image file is accepted"
                    )
                }
            } else {
                guard allowedTextFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "unsupported field; client input, output, and checkpoint paths are not accepted"
                    )
                }
                guard String(data: part.body, encoding: .utf8) != nil else {
                    throw APIRequestValidationError.invalidField(part.name, "must contain valid UTF-8 text")
                }
            }
        }
        for field in allowedTextFields
        where form.parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }

        let uploads = form.files(named: "image")
        guard uploads.count == 1, let upload = uploads.first, !upload.body.isEmpty else {
            throw APIRequestValidationError.invalidField(
                "image",
                "exactly one non-empty uploaded image file is required"
            )
        }
        if let rawContentType = upload.contentType {
            let contentType = rawContentType
                .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                .first?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() ?? ""
            guard contentType.hasPrefix("image/") || contentType == "application/octet-stream" else {
                throw APIRequestValidationError.invalidField(
                    "image",
                    "uploaded part must have an image content type"
                )
            }
        }

        let modelID = normalizedModelID(
            form.field("model"),
            defaultID: defaultImageTo3DModelID
        ).lowercased()
        guard modelID == defaultImageTo3DModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only the managed model id \(defaultImageTo3DModelID) is supported"
            )
        }

        let extractionResolution = try optionalPositiveIntField(
            form.field("resolution"),
            field: "resolution"
        ) ?? 256
        guard (2...512).contains(extractionResolution) else {
            throw APIRequestValidationError.invalidField(
                "resolution",
                "must be an integer between 2 and 512"
            )
        }

        let densityThreshold: Float
        if let raw = normalizedOptional(form.field("density_threshold")) {
            guard let value = Float(raw), value.isFinite else {
                throw APIRequestValidationError.invalidField(
                    "density_threshold",
                    "must be a finite number"
                )
            }
            densityThreshold = value
        } else {
            densityThreshold = TripoSRConfiguration.production.densityThreshold
        }

        let foregroundRatio: Float
        if let raw = normalizedOptional(form.field("foreground_ratio")) {
            guard let value = Float(raw), value.isFinite, value > 0, value <= 1 else {
                throw APIRequestValidationError.invalidField(
                    "foreground_ratio",
                    "must be greater than 0 and at most 1"
                )
            }
            foregroundRatio = value
        } else {
            foregroundRatio = 0.85
        }

        return ImageTo3DPlan(
            modelID: modelID,
            extractionResolution: extractionResolution,
            densityThreshold: densityThreshold,
            foregroundRatio: foregroundRatio,
            alreadyFramed: try multipartBoolean(
                form.field("already_framed"),
                field: "already_framed",
                defaultValue: false
            ),
            includesVertexColors: try multipartBoolean(
                form.field("vertex_colors"),
                field: "vertex_colors",
                defaultValue: true
            )
        )
    }

    static func imageTo3DResponse(
        from result: TripoSRRunResult,
        createdAt: Date = Date()
    ) throws -> APIImageTo3DResponse {
        let mesh = result.export.manifest
        let root = URL(fileURLWithPath: mesh.outputDirectory, isDirectory: true)
        let artifacts = result.runManifest.manifest.artifacts.map { artifact in
            APIImageTo3DArtifactResponse(
                kind: artifact.kind,
                url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                mediaType: artifact.mediaType,
                byteCount: artifact.byteCount,
                sha256: artifact.sha256
            )
        }.sorted { lhs, rhs in
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.url < rhs.url
        }
        let manifest = try imageTo3DFileArtifact(
            kind: "manifest",
            url: result.runManifest.manifestURL,
            mediaType: "application/json"
        )
        let meshManifest = try imageTo3DFileArtifact(
            kind: "mesh-manifest",
            url: result.export.manifestURL,
            mediaType: "application/json"
        )
        let totalSeconds = result.checkpointVerificationSeconds
            + result.decodingSeconds
            + result.preprocessingSeconds
            + result.modelLoadSeconds
            + result.sceneEncodingSeconds
            + result.meshExtractionSeconds
            + result.exportSeconds
        return APIImageTo3DResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.image-to-3d",
            status: "completed",
            model: result.checkpoint.modelID,
            checkpoint: APIImageTo3DCheckpointResponse(
                repository: result.checkpoint.repository,
                revision: result.checkpoint.revision,
                sourceRepository: result.checkpoint.sourceRepository,
                sourceRevision: result.checkpoint.sourceRevision,
                license: result.checkpoint.license,
                format: result.checkpoint.format,
                weightsByteCount: result.checkpoint.weightsByteCount,
                weightsSHA256: result.checkpoint.weightsSHA256,
                sourceSHA256: result.checkpoint.sourceSHA256,
                configurationSHA256: result.checkpoint.configurationSHA256
            ),
            sourceWidth: result.sourceWidth,
            sourceHeight: result.sourceHeight,
            preparedWidth: result.preparedWidth,
            preparedHeight: result.preparedHeight,
            foregroundPolicy: result.foregroundPolicy,
            foregroundRatio: result.foregroundRatio,
            croppedTransparentForeground: result.croppedTransparentForeground,
            extractionResolution: result.extractionResolution,
            densityThreshold: result.densityThreshold,
            includesVertexColors: result.includesVertexColors,
            meshExtractionAlgorithm: TripoSRRunManifestExporter.extractionAlgorithm,
            coordinateSystem: mesh.coordinateSystem,
            units: mesh.units,
            inferredUnseenGeometry: mesh.inferredUnseenGeometry,
            vertexCount: mesh.vertexCount,
            triangleCount: mesh.triangleCount,
            bounds: mesh.bounds,
            manifest: manifest,
            meshManifest: meshManifest,
            artifacts: artifacts,
            timing: APIImageTo3DTimingResponse(
                checkpointVerificationSeconds: result.checkpointVerificationSeconds,
                decodingSeconds: result.decodingSeconds,
                preprocessingSeconds: result.preprocessingSeconds,
                modelLoadSeconds: result.modelLoadSeconds,
                sceneEncodingSeconds: result.sceneEncodingSeconds,
                meshExtractionSeconds: result.meshExtractionSeconds,
                exportSeconds: result.exportSeconds,
                totalSeconds: totalSeconds
            )
        )
    }

    private static func imageTo3DFileArtifact(
        kind: String,
        url: URL,
        mediaType: String
    ) throws -> APIImageTo3DArtifactResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return APIImageTo3DArtifactResponse(
            kind: kind,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private static func multipartBoolean(
        _ raw: String?,
        field: String,
        defaultValue: Bool
    ) throws -> Bool {
        guard let value = normalizedOptional(raw)?.lowercased() else { return defaultValue }
        switch value {
        case "true", "1": return true
        case "false", "0": return false
        default:
            throw APIRequestValidationError.invalidField(
                field,
                "must be true, false, 1, or 0"
            )
        }
    }

    static func depthVideoPlan(from form: MultipartFormData) throws -> DepthVideoPlan {
        let allowedFields: Set<String> = ["model", "input_size", "max_frames"]
        for part in form.parts {
            if part.filename != nil {
                guard part.name == "video" else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "only a single uploaded 'video' file is accepted"
                    )
                }
            } else if !allowedFields.contains(part.name) {
                throw APIRequestValidationError.invalidField(
                    part.name,
                    "unsupported field; client filesystem paths are not accepted"
                )
            }
        }
        for field in allowedFields where form.parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }

        let uploads = form.files(named: "video")
        guard uploads.count == 1, let upload = uploads.first, !upload.body.isEmpty else {
            throw APIRequestValidationError.invalidField(
                "video",
                "exactly one non-empty uploaded video file is required"
            )
        }
        if let contentType = upload.contentType?.lowercased(),
           !contentType.hasPrefix("video/"),
           contentType != "application/octet-stream" {
            throw APIRequestValidationError.invalidField(
                "video",
                "uploaded part must have a video content type"
            )
        }

        let modelID = normalizedModelID(
            form.field("model"),
            defaultID: defaultDepthVideoModelID
        ).lowercased()
        let supportedModelIDs = Set(VideoDepthAnythingVariant.allCases.map(\.modelID))
        guard supportedModelIDs.contains(modelID) else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only \(supportedModelIDs.sorted().joined(separator: ", ")) are supported"
            )
        }

        let inputSize = try optionalPositiveIntField(
                form.field("input_size"),
                field: "input_size"
            ) ?? VideoDepthAnythingLimits.defaultInputSize
        let maximumFrameCount = try optionalPositiveIntField(
                form.field("max_frames"),
                field: "max_frames"
            ) ?? VideoDepthAnythingLimits.defaultMaximumFrameCount
        do {
            _ = try VideoDepthAnythingLimits.validateRequest(
                inputSize: inputSize,
                maximumFrameCount: maximumFrameCount
            )
        } catch let error as VideoDepthAnythingLimitError {
            let field: String
            switch error {
            case .inputSizeOutOfRange:
                field = "input_size"
            default:
                field = "max_frames"
            }
            throw APIRequestValidationError.invalidField(field, error.localizedDescription)
        } catch {
            throw APIRequestValidationError.invalidField("video", error.localizedDescription)
        }
        return DepthVideoPlan(
            modelID: modelID,
            inputSize: inputSize,
            maximumFrameCount: maximumFrameCount
        )
    }

    static func depthVideoResponse(
        from result: VideoDepthAnythingRunResult,
        createdAt: Date = Date()
    ) throws -> APIDepthVideoResponse {
        let sequence = result.export.manifest
        let root = URL(fileURLWithPath: sequence.outputDirectory, isDirectory: true)
        let manifest = try depthVideoFileArtifact(
            kind: GeometryArtifactKind.manifest.rawValue,
            frameIndex: nil,
            url: result.export.manifestURL,
            mediaType: "application/json"
        )
        let review = APIDepthVideoArtifactResponse(
            kind: result.reviewVideo.kind,
            frameIndex: nil,
            url: root.appendingPathComponent(result.reviewVideo.relativePath).absoluteString,
            mediaType: result.reviewVideo.mediaType,
            byteCount: result.reviewVideo.byteCount,
            sha256: result.reviewVideo.sha256
        )
        let artifacts = sequence.frames.flatMap { frame in
            frame.artifacts.map { artifact in
                APIDepthVideoArtifactResponse(
                    kind: artifact.kind.rawValue,
                    frameIndex: frame.index,
                    url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                    mediaType: artifact.mediaType,
                    byteCount: artifact.byteCount,
                    sha256: artifact.sha256
                )
            }
        }.sorted { lhs, rhs in
            if lhs.frameIndex != rhs.frameIndex {
                return (lhs.frameIndex ?? -1) < (rhs.frameIndex ?? -1)
            }
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.url < rhs.url
        }
        let timing = APIDepthVideoTimingResponse(
            checkpointVerificationSeconds: result.checkpointVerificationSeconds,
            frameExtractionSeconds: result.frameExtractionSeconds,
            modelLoadSeconds: result.modelLoadSeconds,
            inferenceSeconds: result.inferenceSeconds,
            exportSeconds: result.exportSeconds,
            totalSeconds: result.checkpointVerificationSeconds
                + result.frameExtractionSeconds
                + result.modelLoadSeconds
                + result.inferenceSeconds
                + result.exportSeconds
        )
        return APIDepthVideoResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.depth-video",
            status: "completed",
            model: sequence.model.modelID,
            semantics: sequence.semantics,
            checkpointFormat: result.checkpoint.format,
            checkpointSHA256: result.checkpoint.weightsSHA256,
            width: sequence.width,
            height: sequence.height,
            fps: sequence.fps,
            frameCount: sequence.frameCount,
            windowCount: result.windowCount,
            temporalWindowLength: sequence.temporalWindowLength,
            temporalOverlap: sequence.temporalOverlap,
            hasConfidence: sequence.frames.contains { $0.confidencePath != nil },
            hasCameraIntrinsics: sequence.frames.contains { $0.intrinsics != nil },
            hasCameraExtrinsics: false,
            hasPointCloud: false,
            manifest: manifest,
            review: review,
            artifacts: artifacts,
            timing: timing
        )
    }

    private static func depthVideoFileArtifact(
        kind: String,
        frameIndex: Int?,
        url: URL,
        mediaType: String
    ) throws -> APIDepthVideoArtifactResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return APIDepthVideoArtifactResponse(
            kind: kind,
            frameIndex: frameIndex,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    static func imageGenerationPlan(
        from request: OpenAIImageGenerationRequest
    ) throws -> ImageGenerationPlan {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        let steps = try imageInferenceSteps(request.steps)
        if let guidanceScale = request.guidance_scale,
           (!guidanceScale.isFinite || guidanceScale < 0) {
            throw APIRequestValidationError.invalidField("guidance_scale", "must be a positive number")
        }
        let size = try imageSize(from: request.size)
        let responseFormat = try imageResponseFormat(request.response_format)
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(request.model),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: responseFormat,
            seed: request.seed,
            negativePrompt: normalizedOptional(request.negative_prompt),
            steps: steps,
            guidanceScale: request.guidance_scale,
            inputImage: nil,
            strength: nil
        )
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURL: URL
    ) throws -> ImageGenerationPlan {
        try imageEditPlan(from: form, inputImageURLs: [inputImageURL], maskImageURL: nil)
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURLs: [URL],
        maskImageURL: URL?
    ) throws -> ImageGenerationPlan {
        guard let inputImageURL = inputImageURLs.first else {
            throw APIRequestValidationError.invalidField("image", "image file is required")
        }
        let prompt = normalizedOptional(form.field("prompt")) ?? ""
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let rawN = normalizedOptional(form.field("n")) {
            guard let n = Int(rawN), n == 1 else {
                throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
            }
        }
        let steps = try imageInferenceSteps(
            optionalPositiveIntField(form.field("steps"), field: "steps")
        )
        let guidanceScale = try optionalPositiveDoubleField(form.field("guidance_scale"), field: "guidance_scale")
        let strength = try optionalUnitDoubleField(form.field("strength"), field: "strength")
        let size = try imageSize(from: form.field("size"))
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(form.field("model")),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: try imageResponseFormat(form.field("response_format")),
            seed: try optionalUInt64Field(form.field("seed")),
            negativePrompt: normalizedOptional(form.field("negative_prompt")),
            steps: steps,
            guidanceScale: guidanceScale,
            inputImage: inputImageURL,
            additionalInputImages: Array(inputImageURLs.dropFirst()),
            maskImage: maskImageURL,
            strength: strength
        )
    }

    static func imageResponse(
        outputURL: URL,
        plan: ImageGenerationPlan,
        createdAt: Date = Date()
    ) throws -> OpenAIImageGenerationResponse {
        let datum: OpenAIImageGenerationData
        switch plan.responseFormat {
        case "url":
            datum = OpenAIImageGenerationData(url: outputURL.absoluteString, revised_prompt: plan.prompt)
        case "b64_json":
            let data = try Data(contentsOf: outputURL)
            datum = OpenAIImageGenerationData(
                b64_json: data.base64EncodedString(),
                revised_prompt: plan.prompt
            )
        default:
            throw APIRequestValidationError.invalidField("response_format", "unsupported response format")
        }
        return OpenAIImageGenerationResponse(
            created: Int(createdAt.timeIntervalSince1970),
            data: [datum]
        )
    }

    static func speechPlan(from request: OpenAIAudioSpeechRequest) throws -> SpeechPlan {
        let input = request.input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else {
            throw APIRequestValidationError.invalidField("input", "must not be empty")
        }
        let responseFormat = try speechResponseFormat(request.response_format)
        let speed = try speechSpeed(request.speed)
        let temperature = request.temperature ?? 0.6
        guard temperature.isFinite, (0...2).contains(temperature) else {
            throw APIRequestValidationError.invalidField("temperature", "must be between 0 and 2")
        }
        let voiceDescription = voiceDescription(for: request.voice, instructions: request.instructions)
        var promptUTF8Bytes = 0
        for component in [input, voiceDescription] {
            let componentBytes = component.utf8.count
            guard componentBytes <= maxSpeechPromptUTF8Bytes - promptUTF8Bytes else {
                throw APIRequestValidationError.invalidField(
                    "input",
                    "input and voice instructions must total at most \(maxSpeechPromptUTF8Bytes) UTF-8 bytes"
                )
            }
            promptUTF8Bytes += componentBytes
        }
        return SpeechPlan(
            modelID: normalizedSpeechModelID(request.model),
            input: input,
            voiceDescription: voiceDescription,
            responseFormat: responseFormat,
            speed: speed,
            temperature: temperature
        )
    }

    static func transcriptionPlan(from form: MultipartFormData) throws -> TranscriptionPlan {
        let modelID = normalizedTranscriptionModelID(form.field("model"))
        return TranscriptionPlan(
            modelID: modelID,
            language: normalizedOptional(form.field("language")),
            responseFormat: try transcriptionResponseFormat(form.field("response_format")),
            task: try transcriptionTask(form.field("task")),
            maxTokens: try transcriptionMaxTokens(form.field("max_tokens"))
        )
    }

    static func transcriptionResponse(
        from result: ASRResult,
        verbose: Bool
    ) -> OpenAIAudioTranscriptionResponse {
        OpenAIAudioTranscriptionResponse(
            text: result.text,
            language: result.language,
            duration: verbose ? result.duration : nil,
            segments: verbose ? transcriptionSegments(from: result) : nil
        )
    }

    static func transcriptionSubtitle(from result: ASRResult, format: String) -> String {
        let segments = transcriptionSegments(from: result) ?? [
            OpenAIAudioTranscriptionSegment(
                id: 0,
                start: 0,
                end: max(result.duration, 0.001),
                text: result.text
            ),
        ]
        switch format {
        case "srt":
            return segments.enumerated()
                .map { index, segment in
                    let start = subtitleTimestamp(segment.start, separator: ",")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ",")
                    return "\(index + 1)\n\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n") + "\n"
        default:
            let body = segments
                .map { segment in
                    let start = subtitleTimestamp(segment.start, separator: ".")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ".")
                    return "\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n")
            return body.isEmpty ? "WEBVTT\n" : "WEBVTT\n\n\(body)\n"
        }
    }

    static func chatRequest(
        from openaiRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        contextSize: Int,
        capabilities: APIEngineCapabilities = .localText,
        servedModelID: String? = nil
    ) throws -> ChatRequest {
        guard !openaiRequest.messages.isEmpty else {
            throw APIRequestValidationError.invalidField("messages", "must contain at least one message")
        }

        try validateTopLevelOptions(openaiRequest, capabilities: capabilities)

        if let requestLora = openaiRequest.lora?.trimmingCharacters(in: .whitespacesAndNewlines),
           !requestLora.isEmpty {
            throw APIRequestValidationError.invalidField(
                "lora",
                "per-request LoRA paths are not supported; start the server with --lora instead"
            )
        }

        let maxTokens = try validateMaxTokens(
            maxTokens: openaiRequest.max_tokens,
            maxCompletionTokens: openaiRequest.max_completion_tokens,
            contextSize: contextSize,
            capabilities: capabilities
        )
        let temperature = try validateTemperature(openaiRequest.temperature)
        let topP = try validateTopP(openaiRequest.top_p)
        let minP = try validateMinP(openaiRequest.min_p)
        let tools = try toolDefinitions(from: openaiRequest, capabilities: capabilities)
        let requiresJSON = try requiresJSONResponseFormat(
            openaiRequest.response_format,
            capabilities: capabilities
        )

        let messages = try openaiRequest.messages.map { msg in
            try chatMessage(from: msg, capabilities: capabilities)
        }

        let lora: LoRA?
        if let loraPath = fallbackLoraPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !loraPath.isEmpty {
            lora = LoRA.local(path: loraPath, scale: 1.0)
        } else {
            lora = nil
        }

        // R1-style lanes degenerate without reasoning; their published top_k
        // applies only when the client did not set explicit sampling.
        let laneModelID = servedModelID ?? ""
        let recommendedSampling = Q35Resources.recommendedSampling(forModelId: laneModelID)
        let usesExplicitSampling = openaiRequest.temperature != nil
            || openaiRequest.top_p != nil
            || openaiRequest.min_p != nil

        return ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: openaiRequest.temperature == nil
                ? recommendedSampling?.temperature ?? temperature
                : temperature,
            topP: openaiRequest.top_p == nil
                ? recommendedSampling?.topP ?? topP
                : topP,
            topK: usesExplicitSampling ? nil : recommendedSampling?.topK,
            minP: minP,
            showThinking: requiresJSON ? false : Q35Resources.thinkingDefault(forModelId: laneModelID),
            lora: lora,
            requiresJSON: requiresJSON,
            tools: tools,
            stopSequences: openaiRequest.stop?.values ?? [],
            maxContextTokens: contextSize
        )
    }

    static func includeUsageInStreaming(
        _ openaiRequest: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard openaiRequest.stream_options?.include_usage == true else {
            return false
        }
        guard capabilities.supportsUsageInStreaming else {
            throw APIRequestValidationError.invalidField(
                "stream_options.include_usage",
                "this engine cannot emit usage chunks while streaming"
            )
        }
        return true
    }

    private static func chatMessage(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> ChatMessage {
        let normalizedRole = msg.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let role: ChatMessage.Role
        switch normalizedRole {
        case "system":
            role = .system
        case "developer":
            guard capabilities.supportsDeveloperRole else {
                throw APIRequestValidationError.invalidField(
                    "messages.role",
                    "developer messages are not supported by this engine"
                )
            }
            role = .system
        case "user":
            role = .user
        case "assistant":
            role = .assistant
        case "tool":
            role = .tool
        default:
            throw APIRequestValidationError.invalidField(
                "messages.role",
                "unsupported role '\(msg.role)'"
            )
        }

        let imageURL = try firstImageURL(from: msg, capabilities: capabilities)
        let content = renderMessageContent(msg)
        let toolCalls = try chatMessageToolCalls(from: msg)
        return ChatMessage(
            role: role,
            content: content,
            imageUrl: imageURL,
            reasoningContent: msg.reasoning_content,
            name: msg.name,
            toolCallID: msg.tool_call_id,
            toolCalls: toolCalls
        )
    }

    private static func chatMessageToolCalls(
        from message: OpenAIChatMessage
    ) throws -> [ChatMessageToolCall]? {
        guard message.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "assistant",
              let openAIToolCalls = message.tool_calls,
              !openAIToolCalls.isEmpty else {
            return nil
        }

        return try openAIToolCalls.compactMap { toolCall in
            guard toolCall.type == "function", let function = toolCall.function else {
                return nil
            }
            guard let data = function.arguments.data(using: .utf8) else {
                throw APIRequestValidationError.invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a UTF-8 JSON object"
                )
            }
            let arguments: [String: OpenAIJSONValue]
            do {
                arguments = try JSONDecoder().decode([String: OpenAIJSONValue].self, from: data)
            } catch {
                throw APIRequestValidationError.invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a JSON object"
                )
            }
            return ChatMessageToolCall(
                id: toolCall.id,
                name: function.name,
                arguments: arguments
            )
        }
    }

    private static func firstImageURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.imageURLs.isEmpty else { return nil }
        guard capabilities.supportsVisionContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "image content parts are not supported by this engine"
            )
        }
        guard msg.imageURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one image content part is currently supported"
            )
        }
        return msg.imageURLs.first
    }

    private static func renderMessageContent(_ msg: OpenAIChatMessage) -> String {
        guard msg.role.lowercased() == "assistant",
              let toolCalls = msg.tool_calls,
              !toolCalls.isEmpty else {
            return msg.content
        }
        let renderedCalls = toolCalls.compactMap { call -> String? in
            guard call.type == "function", let function = call.function else { return nil }
            return "<|tool_call>call:\(function.name)\(function.arguments)<tool_call|>"
        }
        guard !renderedCalls.isEmpty else {
            return msg.content
        }
        return ([msg.content] + renderedCalls)
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func validateTopLevelOptions(
        _ request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws {
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        if request.store == true {
            throw APIRequestValidationError.invalidField("store", "stored chat completions are not supported")
        }
        if let modalities = request.modalities,
           modalities.contains(where: { $0 != "text" }) {
            throw APIRequestValidationError.invalidField("modalities", "only text output is supported")
        }
        if request.audio != nil {
            throw APIRequestValidationError.invalidField("audio", "audio output is not supported by /v1/chat/completions")
        }
        if request.prediction != nil {
            throw APIRequestValidationError.invalidField("prediction", "predicted outputs are not supported")
        }
        if let stop = request.stop, !stop.values.isEmpty, !capabilities.supportsStopSequences {
            throw APIRequestValidationError.invalidField("stop", "stop sequences are not supported by this engine")
        }
        if request.seed != nil, !capabilities.supportsSeed {
            throw APIRequestValidationError.invalidField("seed", "deterministic seeds are not supported by this engine")
        }
        if let penalty = request.presence_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("presence_penalty", "presence penalties are not supported by this engine")
        }
        if let penalty = request.frequency_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("frequency_penalty", "frequency penalties are not supported by this engine")
        }
        if request.logprobs == true, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("logprobs", "token log probabilities are not supported by this engine")
        }
        if request.top_logprobs != nil, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("top_logprobs", "token log probabilities are not supported by this engine")
        }
        if request.reasoning_effort != nil, !capabilities.supportsReasoningEffort {
            throw APIRequestValidationError.invalidField("reasoning_effort", "reasoning effort is not supported by this engine")
        }
        if request.think != nil || request.thinking != nil {
            guard capabilities.supportsProviderThinkingControls else {
                throw APIRequestValidationError.invalidField(
                    "thinking",
                    "provider thinking controls are not supported by this engine"
                )
            }
        }
    }

    private static func toolDefinitions(
        from request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> [ToolDefinition]? {
        guard let tools = request.tools, !tools.isEmpty else {
            return nil
        }
        guard capabilities.supportsTools else {
            throw APIRequestValidationError.invalidField("tools", "tools are not supported by this engine")
        }

        let convertedTools = try tools.map { try toolDefinition(from: $0) }
        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("required")?:
            return convertedTools
        case .mode("none")?:
            return nil
        case .function(let name)?:
            guard let selected = convertedTools.first(where: { $0.name == name }) else {
                throw APIRequestValidationError.invalidField(
                    "tool_choice",
                    "requested tool '\(name)' is not present in tools"
                )
            }
            return [selected]
        case .custom?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported object shape")
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }
    }

    private static func toolDefinition(from tool: OpenAIChatTool) throws -> ToolDefinition {
        guard tool.type == "function", let function = tool.function else {
            throw APIRequestValidationError.invalidField("tools", "only function tools are supported")
        }

        let schema = function.parameters?.objectValue ?? [:]
        let properties = schema["properties"]?.objectValue ?? [:]
        var converted: [String: ToolParameterProperty] = [:]
        for (name, rawProperty) in properties {
            guard let property = rawProperty.objectValue else { continue }
            let type = property["type"]?.stringValue ?? "string"
            let description = property["description"]?.stringValue ?? ""
            converted[name] = ToolParameterProperty(type: type, description: description)
        }
        let required = schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? []

        return ToolDefinition(
            name: function.name,
            description: function.description ?? "",
            parameters: converted,
            required: required
        )
    }

    private static func requiresJSONResponseFormat(
        _ responseFormat: OpenAIResponseFormat?,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard let responseFormat else { return false }
        switch responseFormat.type {
        case "text":
            return false
        case "json_object":
            guard capabilities.supportsStructuredOutputs else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "JSON mode is not supported by this engine"
                )
            }
            return true
        case "json_schema":
            guard capabilities.supportsStrictMode else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "strict JSON schema outputs are not supported by this engine"
                )
            }
            return true
        default:
            throw APIRequestValidationError.invalidField(
                "response_format",
                "unsupported response format '\(responseFormat.type)'"
            )
        }
    }

    private static func validateMaxTokens(
        maxTokens: Int?,
        maxCompletionTokens: Int?,
        contextSize: Int,
        capabilities: APIEngineCapabilities
    ) throws -> Int {
        if maxCompletionTokens != nil, !capabilities.supportsMaxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "this engine does not support max_completion_tokens"
            )
        }
        if let maxTokens, let maxCompletionTokens, maxTokens != maxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "must match max_tokens when both are provided"
            )
        }
        return try validateMaxTokens(maxCompletionTokens ?? maxTokens, contextSize: contextSize)
    }

    static func acceptsJSONContentType(_ rawValue: String?) -> Bool {
        guard let mediaType = rawValue?
            .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
            .first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            !mediaType.isEmpty
        else {
            return false
        }
        return mediaType == "application/json" || mediaType.hasSuffix("+json")
    }

    static func multipartBoundary(from rawValue: String?) -> String? {
        guard let pieces = rawValue?.split(separator: ";", omittingEmptySubsequences: true),
              let mediaType = pieces.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              mediaType == "multipart/form-data" else {
            return nil
        }
        for piece in pieces.dropFirst() {
            let pair = piece.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2,
                  pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "boundary" else {
                continue
            }
            var boundary = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if boundary.hasPrefix("\""), boundary.hasSuffix("\""), boundary.count >= 2 {
                boundary.removeFirst()
                boundary.removeLast()
            }
            return boundary.isEmpty ? nil : String(boundary)
        }
        return nil
    }

    static func isStreamingStatusMessage(_ message: String) -> Bool {
        switch message {
        case "Generating...", "Generating response", "Retrying generation", "DS4 chat completion":
            return true
        default:
            return false
        }
    }

    private static func validateMaxTokens(_ rawValue: Int?, contextSize: Int) throws -> Int {
        let value = rawValue ?? defaultMaxTokens
        let upperBound = min(contextSize, Int(Int32.max))
        guard upperBound > 0 else {
            throw APIRequestValidationError.invalidField("context_size", "must be greater than zero")
        }
        guard (1...upperBound).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "max_tokens",
                "must be between 1 and \(upperBound)"
            )
        }
        return value
    }

    private static func validateTemperature(_ rawValue: Double?) throws -> Double {
        let value = rawValue ?? 1.0
        guard value.isFinite, (0...2).contains(value) else {
            throw APIRequestValidationError.invalidField("temperature", "must be between 0 and 2")
        }
        return value
    }

    private static func validateTopP(_ rawValue: Double?) throws -> Double {
        let value = rawValue ?? 0.95
        guard value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField("top_p", "must be between 0 and 1")
        }
        return value
    }

    private static func validateMinP(_ rawValue: Double?) throws -> Double {
        let value = rawValue ?? 0
        guard value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField("min_p", "must be between 0 and 1")
        }
        return value
    }

    private static func imageSize(from rawValue: String?) throws -> (width: Int, height: Int) {
        guard let rawValue = normalizedOptional(rawValue), rawValue.lowercased() != "auto" else {
            return (1024, 1024)
        }
        let parts = rawValue.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]),
              width > 0,
              height > 0 else {
            throw APIRequestValidationError.invalidField("size", "expected WIDTHxHEIGHT, for example 1024x1024")
        }

        let maxDimension = 4_096
        guard width <= maxDimension, height <= maxDimension else {
            throw APIRequestValidationError.invalidField(
                "size",
                "width and height must each be at most \(maxDimension) pixels"
            )
        }

        // Validate the pixel product with division so attacker-controlled
        // dimensions can never overflow an Int before the request reaches a
        // tensor allocation.
        let maxPixels = 4_194_304
        guard width <= maxPixels / height else {
            throw APIRequestValidationError.invalidField(
                "size",
                "total image area must be at most \(maxPixels) pixels"
            )
        }

        let imageAlignment = 16
        guard width >= imageAlignment,
              height >= imageAlignment,
              width % imageAlignment == 0,
              height % imageAlignment == 0 else {
            throw APIRequestValidationError.invalidField(
                "size",
                "width and height must each be at least \(imageAlignment) pixels and divisible by \(imageAlignment)"
            )
        }
        return (width, height)
    }

    private static func imageResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "b64_json"
        guard value == "b64_json" || value == "url" else {
            throw APIRequestValidationError.invalidField("response_format", "expected b64_json or url")
        }
        return value
    }

    private static func speechResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "wav"
        guard ["wav", "mp3", "opus", "aac", "flac"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected wav, mp3, opus, aac, or flac"
            )
        }
        return value
    }

    static func speechContentType(for responseFormat: String) -> String {
        switch responseFormat {
        case "mp3":
            return "audio/mpeg"
        case "opus":
            return "audio/ogg"
        case "aac":
            return "audio/aac"
        case "flac":
            return "audio/flac"
        default:
            return "audio/wav"
        }
    }

    private static func speechSpeed(_ rawValue: Double?) throws -> Float {
        let value = rawValue ?? 1.0
        guard value.isFinite, (0.25...4.0).contains(value) else {
            throw APIRequestValidationError.invalidField("speed", "must be between 0.25 and 4.0")
        }
        return Float(value)
    }

    private static func transcriptionResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "json"
        guard ["json", "text", "verbose_json", "srt", "vtt"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected json, text, verbose_json, srt, or vtt"
            )
        }
        return value
    }

    private static func transcriptionTask(_ rawValue: String?) throws -> ASRTask {
        let value = normalizedOptional(rawValue)?.lowercased() ?? ASRTask.transcribe.rawValue
        guard let task = ASRTask(rawValue: value) else {
            throw APIRequestValidationError.invalidField("task", "expected transcribe or translate")
        }
        return task
    }

    private static func transcriptionMaxTokens(_ rawValue: String?) throws -> Int {
        guard let rawValue = normalizedOptional(rawValue) else {
            return 448
        }
        guard let value = Int(rawValue), (1...maxTranscriptionTokens).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "max_tokens",
                "must be between 1 and \(maxTranscriptionTokens)"
            )
        }
        return value
    }

    private static func imageInferenceSteps(_ value: Int?) throws -> Int? {
        guard let value else { return nil }
        guard (1...maxImageInferenceSteps).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "steps",
                "must be between 1 and \(maxImageInferenceSteps)"
            )
        }
        return value
    }

    private static func optionalUInt64Field(_ rawValue: String?) throws -> UInt64? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = UInt64(rawValue) else {
            throw APIRequestValidationError.invalidField("seed", "must be an unsigned integer")
        }
        return value
    }

    private static func optionalPositiveIntField(_ rawValue: String?, field: String) throws -> Int? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Int(rawValue), value > 0 else {
            throw APIRequestValidationError.invalidField(field, "must be greater than zero")
        }
        return value
    }

    private static func optionalPositiveDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, value >= 0 else {
            throw APIRequestValidationError.invalidField(field, "must be a positive number")
        }
        return value
    }

    private static func optionalUnitDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField(field, "must be between 0 and 1")
        }
        return value
    }

    private static func normalizedModelID(_ rawValue: String?, defaultID: String) -> String {
        normalizedOptional(rawValue) ?? defaultID
    }

    private static func normalizedImageModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultImageModelID)
        switch modelID.lowercased() {
        case "gpt-image-1", "dall-e-3", "dall-e-2":
            return defaultImageModelID
        default:
            return modelID
        }
    }

    private static func normalizedSpeechModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultSpeechModelID)
        switch modelID.lowercased() {
        case "tts-1", "tts-1-hd", "gpt-4o-mini-tts":
            return defaultSpeechModelID
        default:
            return modelID
        }
    }

    private static func normalizedTranscriptionModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultTranscriptionModelID)
        switch modelID.lowercased() {
        case "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe":
            return defaultTranscriptionModelID
        default:
            return modelID
        }
    }

    private static func normalizedOptional(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func isCompanionModelInstalled(
        _ spec: ManagedModelSpec,
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(spec.id)
        }
        return spec.managedRuntimeURL(fileManager: fileManager) != nil
    }

    private static func isQwenImageEditInstalled(
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(QwenImageEditRepository.modelId)
        }
        return QwenImageEditRepository.resolveInstalledModelRoot(fileManager: fileManager) != nil
    }

    private static func voiceDescription(for rawVoice: String?, instructions: String?) -> String {
        let instructionText = normalizedOptional(instructions)
        let voice = normalizedOptional(rawVoice)?.lowercased() ?? "nova"
        let base: String
        switch voice {
        case "alloy":
            base = "A balanced, natural voice with clear pronunciation"
        case "ash":
            base = "A calm, low voice with a steady delivery"
        case "ballad":
            base = "A warm, expressive voice with a storytelling cadence"
        case "coral":
            base = "A bright, friendly voice with gentle energy"
        case "echo":
            base = "A clear male voice with an even, conversational tone"
        case "fable":
            base = "A warm narrative voice with a measured pace"
        case "nova":
            base = "A calm female voice with clear pronunciation"
        case "onyx":
            base = "A deep, confident voice with crisp articulation"
        case "sage":
            base = "A thoughtful, composed voice with soft emphasis"
        case "shimmer":
            base = "A bright, gentle voice with smooth pronunciation"
        default:
            base = rawVoice?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? "A calm female voice with clear pronunciation"
        }
        if let instructionText {
            return "\(base). \(instructionText)"
        }
        return base
    }

    private static func transcriptionSegments(
        from result: ASRResult
    ) -> [OpenAIAudioTranscriptionSegment]? {
        if let sentences = result.sentenceAlignments, !sentences.isEmpty {
            return sentences.enumerated().map { index, sentence in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: sentence.startSeconds,
                    end: sentence.endSeconds,
                    text: sentence.text
                )
            }
        }
        if let tokens = result.tokenAlignments, !tokens.isEmpty {
            return tokens.enumerated().map { index, token in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: token.startSeconds,
                    end: token.endSeconds,
                    text: token.text
                )
            }
        }
        return nil
    }

    private static func subtitleTimestamp(_ seconds: Double, separator: String) -> String {
        let milliseconds = max(0, Int((seconds * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secs = (milliseconds % 60_000) / 1_000
        let millis = milliseconds % 1_000
        return String(format: "%02d:%02d:%02d%@%03d", hours, minutes, secs, separator, millis)
    }

    private static func decodeJSONRequest<Request: Decodable>(
        _ type: Request.Type,
        from data: Data
    ) throws -> Request {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw APIRequestValidationError.invalidPayload
        }
    }
}

enum APIRequestValidationError: LocalizedError, Equatable {
    case invalidPayload
    case invalidField(String, String)

    var errorDescription: String? {
        switch self {
        case .invalidPayload:
            return "Invalid request payload."
        case .invalidField(let field, let reason):
            return "Invalid '\(field)': \(reason)."
        }
    }
}

struct MultipartFormData: Equatable, Sendable {
    struct Part: Equatable, Sendable {
        let name: String
        let filename: String?
        let contentType: String?
        let body: Data
    }

    enum ParseError: LocalizedError, Equatable {
        case missingBoundary
        case malformedBody
        case missingName

        var errorDescription: String? {
            switch self {
            case .missingBoundary:
                return "Missing multipart boundary."
            case .malformedBody:
                return "Malformed multipart body."
            case .missingName:
                return "Multipart part is missing a form-data name."
            }
        }
    }

    let parts: [Part]

    static func parse(body: Data, boundary: String?) throws -> MultipartFormData {
        guard let boundary, !boundary.isEmpty else {
            throw ParseError.missingBoundary
        }
        let marker = Data("--\(boundary)".utf8)
        guard !marker.isEmpty,
              let firstMarker = body.range(of: marker, options: [], in: body.startIndex..<body.endIndex) else {
            throw ParseError.malformedBody
        }

        var parts: [Part] = []
        var markerRange = firstMarker
        while true {
            let afterMarker = markerRange.upperBound
            if body.hasBytes(Data("--".utf8), at: afterMarker) {
                break
            }
            let partStart = body.indexAfterLineBreak(at: afterMarker)
            guard let nextMarker = body.range(of: marker, options: [], in: partStart..<body.endIndex) else {
                throw ParseError.malformedBody
            }
            let partEnd = body.indexTrimmingLineBreak(before: nextMarker.lowerBound)
            if partStart < partEnd {
                parts.append(try parsePart(Data(body[partStart..<partEnd])))
            }
            markerRange = nextMarker
        }

        return MultipartFormData(parts: parts)
    }

    func field(_ name: String) -> String? {
        guard let part = parts.first(where: { $0.name == name && $0.filename == nil }) else {
            return nil
        }
        return String(data: part.body, encoding: .utf8)
    }

    func file(named name: String) -> Part? {
        parts.first { $0.name == name && $0.filename != nil }
    }

    func files(named name: String) -> [Part] {
        parts.filter { $0.name == name && $0.filename != nil }
    }

    private static func parsePart(_ data: Data) throws -> Part {
        let separator = Data("\r\n\r\n".utf8)
        let fallbackSeparator = Data("\n\n".utf8)
        let separatorRange = data.range(of: separator, options: [], in: data.startIndex..<data.endIndex)
            ?? data.range(of: fallbackSeparator, options: [], in: data.startIndex..<data.endIndex)
        guard let separatorRange,
              let headerText = String(data: data[data.startIndex..<separatorRange.lowerBound], encoding: .utf8) else {
            throw ParseError.malformedBody
        }
        let body = Data(data[separatorRange.upperBound..<data.endIndex])
        let headers = parseHeaders(headerText)
        guard let disposition = headers["content-disposition"] else {
            throw ParseError.missingName
        }
        let params = parseDispositionParameters(disposition)
        guard let name = params["name"], !name.isEmpty else {
            throw ParseError.missingName
        }
        return Part(
            name: name,
            filename: params["filename"],
            contentType: headers["content-type"],
            body: body
        )
    }

    private static func parseHeaders(_ text: String) -> [String: String] {
        var headers: [String: String] = [:]
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[name] = value
        }
        return headers
    }

    private static func parseDispositionParameters(_ value: String) -> [String: String] {
        var params: [String: String] = [:]
        for part in value.split(separator: ";", omittingEmptySubsequences: false).dropFirst() {
            let pair = part.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            var rawValue = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if rawValue.hasPrefix("\""), rawValue.hasSuffix("\""), rawValue.count >= 2 {
                rawValue.removeFirst()
                rawValue.removeLast()
            }
            params[key] = rawValue
        }
        return params
    }
}

private extension Data {
    func hasBytes(_ bytes: Data, at index: Data.Index) -> Bool {
        guard index >= startIndex, index + bytes.count <= endIndex else {
            return false
        }
        return self[index..<(index + bytes.count)].elementsEqual(bytes)
    }

    func indexAfterLineBreak(at index: Data.Index) -> Data.Index {
        if hasBytes(Data("\r\n".utf8), at: index) {
            return index + 2
        }
        if hasBytes(Data("\n".utf8), at: index) {
            return index + 1
        }
        return index
    }

    func indexTrimmingLineBreak(before index: Data.Index) -> Data.Index {
        if index >= 2, self[(index - 2)..<index].elementsEqual(Data("\r\n".utf8)) {
            return index - 2
        }
        if index >= 1, self[(index - 1)..<index].elementsEqual(Data("\n".utf8)) {
            return index - 1
        }
        return index
    }
}

struct APIServerRequestContext: RequestContext, RemoteAddressRequestContext {
    var coreContext: CoreRequestContextStorage
    let remoteAddress: SocketAddress?

    init(source: Source) {
        self.coreContext = CoreRequestContextStorage(source: source)
        self.remoteAddress = source.channel.remoteAddress
    }
}

enum APIVFXArtifactRoutePolicy {
    /// Successful responses expose server-local `file:` URLs for this long.
    static let outputTTLSeconds: UInt64 = 60 * 60
    static let denialMessage = "VFX artifact routes are loopback-only because responses contain server-local file URLs; authenticated remote clients cannot use this route."

    static var routePaths: Set<String> {
        [
            APIServerContract.geometryRoutePath,
            APIServerContract.multiViewGeometryRoutePath,
            APIServerContract.imageTo3DRoutePath,
            APIServerContract.instantMeshRoutePath,
            APIServerContract.depthVideoRoutePath,
        ]
    }

    static var modelIDs: Set<String> {
        [
            APIServerContract.defaultGeometryModelID,
            APIServerContract.defaultMultiViewGeometryModelID,
            APIServerContract.defaultImageTo3DModelID,
            APIServerContract.defaultInstantMeshModelID,
            ModelResolver.ModelID.visionDepthVDASmall.rawValue,
            ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
        ]
    }

    /// Trust only the connected peer socket. Proxy headers are intentionally
    /// ignored because this server has no trusted-proxy configuration.
    static func allows(remoteAddress: SocketAddress?) -> Bool {
        guard let remoteAddress else { return false }
        switch remoteAddress {
        case .v4:
            guard let address = remoteAddress.ipAddress else { return false }
            return address.split(separator: ".", omittingEmptySubsequences: false).first == "127"
        case .v6:
            guard let address = remoteAddress.ipAddress?.lowercased() else { return false }
            return address == "::1" || address.hasPrefix("::ffff:127.")
        case .unixDomainSocket:
            return true
        }
    }
}

enum APIVFXClientErrorPolicy {
    static func status(for error: Error) -> HTTPResponse.Status? {
        switch error {
        case is MoGe2TokenGridError,
             is VideoDepthAnythingLimitError,
             is DepthAnything3LimitError,
             is MediaIOError,
             is VFXImageInputValidationError,
             is DepthAnything3CameraValidationError,
             is DepthAnything3CameraConditioningError,
             is DepthAnything3PreprocessingError,
             is VideoDepthAnythingPreprocessingError,
             is VideoDepthAnythingWindowingError,
             is MultiViewGeometryExportConfigurationError,
             is TripoSRPreprocessingError,
             is InstantMeshPreprocessingError:
            return .badRequest
        default:
            return nil
        }
    }
}

struct APIArtifactDirectoryCleanupScheduler: Sendable {
    static let productionDelayNanoseconds = APIVFXArtifactRoutePolicy.outputTTLSeconds * 1_000_000_000

    let delayNanoseconds: UInt64
    private let sleeper: @Sendable (UInt64) async throws -> Void
    private let removeDirectory: @Sendable (URL) -> Void

    init(
        delayNanoseconds: UInt64 = productionDelayNanoseconds,
        sleeper: @escaping @Sendable (UInt64) async throws -> Void = {
            try await Task.sleep(nanoseconds: $0)
        },
        removeDirectory: @escaping @Sendable (URL) -> Void = {
            try? FileManager.default.removeItem(at: $0)
        }
    ) {
        self.delayNanoseconds = delayNanoseconds
        self.sleeper = sleeper
        self.removeDirectory = removeDirectory
    }

    func scheduleCleanup(of directory: URL) {
        let delayNanoseconds = delayNanoseconds
        let sleeper = sleeper
        let removeDirectory = removeDirectory
        Task.detached(priority: .utility) {
            do {
                try await sleeper(delayNanoseconds)
            } catch {
                return
            }
            removeDirectory(directory)
        }
    }
}

func withVFXRequestAdmission<T: Sendable>(
    using admission: RuntimeRequestAdmission,
    isolation _: isolated (any Actor)? = #isolation,
    operation: () async throws -> T
) async throws -> T {
    let lease = try await admission.acquire()
    do {
        let result = try await operation()
        await lease.release()
        return result
    } catch {
        await lease.release()
        throw error
    }
}

// MARK: - Server Implementation

actor CodeGenServer {
    private let apiKey: String?
    private let fallbackLoraPath: String?
    private let defaultModelID: String
    private let contextSize: Int
    private let requestLimiter: APIRateLimiter
    private let requestAdmission: RuntimeRequestAdmission
    private let pool: RuntimeModelPool
    private let sidecarPool: APISidecarModelPool
    private let artifactCleanupScheduler: APIArtifactDirectoryCleanupScheduler

    init(
        defaultModelID: String,
        modelPath: String?,
        fallbackLoraPath: String?,
        apiKey: String?,
        rateLimitPerMinute: Int,
        maxActiveRequests: Int = 1,
        engine: APIEngine,
        contextSize: Int = 32768,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default,
        artifactCleanupScheduler: APIArtifactDirectoryCleanupScheduler = APIArtifactDirectoryCleanupScheduler()
    ) async throws {
        self.apiKey = apiKey
        self.contextSize = contextSize
        self.fallbackLoraPath = fallbackLoraPath
        self.defaultModelID = defaultModelID
        self.requestLimiter = APIRateLimiter(limitPerMinute: rateLimitPerMinute)
        self.artifactCleanupScheduler = artifactCleanupScheduler
        // Continuous batching follows the request-concurrency setting: any
        // --max-active-requests above 1 enables it for the engines that
        // support it, with the per-engine env switches as explicit overrides.
        // (The previous double-gate — env AND flag — shipped a serve that
        // silently never batched: /runtime/status showed batchedDecodeSteps=0
        // under concurrent load until the env was discovered.)
        let batching = RuntimeContinuousBatchingConfiguration(
            maxActiveRequests: maxActiveRequests
        )
        let settingsURL = RuntimeModelSettingsStore.defaultURL()
        let runtimePool = RuntimeModelPool(
            defaultModelID: defaultModelID,
            defaultEngine: engine.runtimeServingEngine,
            startupModelPath: modelPath,
            settingsStore: RuntimeModelSettingsStore(url: settingsURL),
            gemma4KVCacheQuantization: gemma4KVCacheQuantization,
            gemma4ContinuousBatchingEnabled: batching.gemma4,
            q35ContinuousBatchingEnabled: batching.q35,
            lfm2ContinuousBatchingEnabled: batching.lfm2,
            memoryPressurePolicy: memoryPressurePolicy
        )
        self.pool = runtimePool
        self.sidecarPool = APISidecarModelPool(
            settingsURL: settingsURL,
            memoryPressurePolicy: memoryPressurePolicy,
            relieveTextModelPressure: {
                _ = await runtimePool.relieveMemoryPressure(preserveDefault: false)
            },
            releaseOneIdleTextModelForLoad: {
                await runtimePool.releaseOneIdleModelForSidecarLoad() != nil
            }
        )
        self.requestAdmission = RuntimeRequestAdmission(
            maxActiveRequests: maxActiveRequests,
            pressureProvider: {
                await runtimePool.currentMemoryPressure()
            }
        )
        try await pool.preloadDefault()
    }

    func run(host: String, port: Int) async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        print("Starting server at http://\(host):\(port)")
        print("OpenAI-compatible base URL: http://\(host):\(port)/v1")
        print("Chat endpoint: http://\(host):\(port)/v1/chat/completions")
        print("Embeddings endpoint: http://\(host):\(port)/v1/embeddings")
        print("Images endpoint: http://\(host):\(port)/v1/images/generations")
        print("Image edits endpoint: http://\(host):\(port)/v1/images/edits")
        print("VFX artifact endpoints are loopback-only; local file URLs expire after one hour.")
        print("Geometry endpoint: http://\(host):\(port)/v1/vision/geometry")
        print("Multi-view geometry endpoint: http://\(host):\(port)\(APIServerContract.multiViewGeometryRoutePath)")
        print("Image-to-3D endpoint: http://\(host):\(port)\(APIServerContract.imageTo3DRoutePath)")
        print("Multi-view image-to-3D endpoint: http://\(host):\(port)\(APIServerContract.instantMeshRoutePath)")
        print("Video depth endpoint: http://\(host):\(port)\(APIServerContract.depthVideoRoutePath)")
        print("Speech endpoint: http://\(host):\(port)/v1/audio/speech")
        print("Transcriptions endpoint: http://\(host):\(port)/v1/audio/transcriptions")
        print("Press Ctrl+C to stop.")

        try await app.runService()
    }

    nonisolated func buildRouter() -> Router<APIServerRequestContext> {
        let router = Router(context: APIServerRequestContext.self)

        // Health check
        router.get("/health") { _, _ in
            let data = try JSONEncoder().encode(APIServerContract.healthStatus())
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(bytes: data))
            )
        }

        // List models
        router.get("/v1/models") { [self] request, context in
            return try await self.handleModels(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        // Chat completions
        router.post("/v1/chat/completions") { [self] request, _ in
            return try await self.handleChatCompletions(request)
        }

        // Embeddings
        router.post("/v1/embeddings") { [self] request, _ in
            return try await self.handleEmbeddings(request)
        }

        router.post("/v1/images/generations") { [self] request, _ in
            return try await self.handleImageGenerations(request)
        }

        router.post("/v1/images/edits") { [self] request, _ in
            return try await self.handleImageEdits(request)
        }

        router.post(APIServerContract.geometryRouterPath) { [self] request, context in
            return try await self.handleVisionGeometry(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.multiViewGeometryRouterPath) { [self] request, context in
            return try await self.handleVisionMultiViewGeometry(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.imageTo3DRouterPath) { [self] request, context in
            return try await self.handleVisionImageTo3D(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.instantMeshRouterPath) { [self] request, context in
            return try await self.handleVisionInstantMesh(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.depthVideoRouterPath) { [self] request, context in
            return try await self.handleVisionDepthVideo(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post("/v1/audio/speech") { [self] request, _ in
            return try await self.handleAudioSpeech(request)
        }

        router.post("/v1/audio/transcriptions") { [self] request, _ in
            return try await self.handleAudioTranscriptions(request)
        }

        router.get("/runtime/status") { [self] request, _ in
            return try await self.handleRuntimeStatus(request)
        }

        router.post("/runtime/models/:id/load") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeLoad(request, id: id)
        }

        router.post("/runtime/models/:id/unload") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeUnload(request, id: id)
        }

        router.get("/runtime/models/:id/settings") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeSettings(request, id: id)
        }

        router.patch("/runtime/models/:id/settings") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeSettingsPatch(request, id: id)
        }

        return router
    }

    private func handleModels(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let includeLoopbackArtifactModels = APIVFXArtifactRoutePolicy.allows(
            remoteAddress: remoteAddress
        )
        var models = try await pool.modelsResponse()
        if !includeLoopbackArtifactModels {
            models.data.removeAll { APIVFXArtifactRoutePolicy.modelIDs.contains($0.id) }
        }
        for modelID in APIServerContract.companionModelIDs(
            includeLoopbackArtifactModels: includeLoopbackArtifactModels
        )
            where !models.data.contains(where: { $0.id == modelID }) {
            models.data.append(
                OpenAIModel(
                    id: modelID,
                    object: "model",
                    created: Int(Date().timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            )
        }
        models.data.sort { $0.id < $1.id }

        let data = try JSONEncoder().encode(models)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func handleChatCompletions(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        // Decode request body
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024) // 10MB limit
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        let openaiRequest: OpenAIChatRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        let admissionLease = try await requestAdmission.acquire()
        let plan: RuntimeChatPlan
        do {
            plan = try await pool.makeChatPlan(
                for: openaiRequest,
                fallbackLoraPath: fallbackLoraPath,
                serverContextSize: contextSize
            )
        } catch {
            await admissionLease.release()
            return runtimeErrorResponse(error)
        }

        if plan.engine == .textChatDeepseekV4Flash {
            do {
                return try await proxyDeepseekV4FlashChatCompletions(
                    body: body,
                    contentType: request.headers[.contentType],
                    lease: plan.lease,
                    admissionLease: admissionLease
                )
            } catch {
                await plan.lease.release()
                await admissionLease.release()
                return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
            }
        }

        do {
            if openaiRequest.stream == true {
                return try await handleStreamingChat(
                    plan.request,
                    modelID: plan.modelID,
                    includeUsage: plan.includeUsage,
                    lease: plan.lease,
                    admissionLease: admissionLease
                )
            } else {
                return try await handleNonStreamingChat(
                    plan.request,
                    modelID: plan.modelID,
                    lease: plan.lease,
                    admissionLease: admissionLease
                )
            }
        } catch let error as APIRequestValidationError {
            await plan.lease.release()
            await admissionLease.release()
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        } catch {
            await plan.lease.release()
            await admissionLease.release()
            return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
        }
    }

    private func handleEmbeddings(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        let openaiRequest: OpenAIEmbeddingRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        let admissionLease: RuntimeRequestAdmissionLease
        do {
            admissionLease = try await requestAdmission.acquire()
        } catch {
            return runtimeErrorResponse(error)
        }

        do {
            let texts = try APIServerContract.embeddingTexts(from: openaiRequest)
            let resolved = try await embeddingModel(for: openaiRequest.model)
            let result = try await sidecarPool.embed(
                modelID: resolved.modelID,
                modelPath: resolved.modelPath,
                texts: texts
            )
            let response = APIServerContract.embeddingResponse(
                modelId: resolved.modelID,
                embeddings: result.embeddings,
                tokenCounts: result.tokenCounts
            )
            let encoded = try jsonResponse(response)
            await admissionLease.release()
            return encoded
        } catch {
            await admissionLease.release()
            return runtimeErrorResponse(error)
        }
    }

    private func handleImageGenerations(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let openaiRequest = try APIServerContract.decodeImageGenerationRequest(
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.imageGenerationPlan(from: openaiRequest)
            let admissionLease = try await requestAdmission.acquire()
            do {
                let outputURL = try await generateImage(plan)
                let response = try APIServerContract.imageResponse(outputURL: outputURL, plan: plan)
                let encoded = try jsonResponse(response)
                await admissionLease.release()
                return encoded
            } catch {
                await admissionLease.release()
                throw error
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleImageEdits(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 100 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let form = try MultipartFormData.parse(
                body: Data(body.readableBytesView),
                boundary: boundary
            )
            let imageFiles = (form.files(named: "image") + form.files(named: "image[]"))
                .filter { !$0.body.isEmpty }
            guard !imageFiles.isEmpty else {
                throw APIRequestValidationError.invalidField("image", "image file is required")
            }
            let inputImageURLs = try imageFiles.map {
                try writeMultipartFile($0, directoryName: "mere-run-api-image-edits")
            }
            let maskImageURL = try form.file(named: "mask").flatMap { mask -> URL? in
                guard !mask.body.isEmpty else { return nil }
                return try writeMultipartFile(mask, directoryName: "mere-run-api-image-edits")
            }
            let plan = try APIServerContract.imageEditPlan(
                from: form,
                inputImageURLs: inputImageURLs,
                maskImageURL: maskImageURL
            )
            let admissionLease = try await requestAdmission.acquire()
            do {
                let outputURL = try await generateImage(plan)
                let response = try APIServerContract.imageResponse(outputURL: outputURL, plan: plan)
                let encoded = try jsonResponse(response)
                await admissionLease.release()
                return encoded
            } catch {
                await admissionLease.release()
                throw error
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionGeometry(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(upTo: 100 * 1024 * 1024)
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                guard let image = form.file(named: "image"), !image.body.isEmpty else {
                    throw APIRequestValidationError.invalidField("image", "image file is required")
                }
                if let contentType = image.contentType?.lowercased(),
                    !contentType.hasPrefix("image/") && contentType != "application/octet-stream"
                {
                    throw APIRequestValidationError.invalidField("image", "uploaded part must be an image")
                }
                let plan = try APIServerContract.geometryPlan(from: form)
                let inputURL = try writeMultipartFile(
                    image,
                    directoryName: "mere-run-api-geometry-inputs",
                    defaultExtension: "png"
                )
                defer { try? FileManager.default.removeItem(at: inputURL) }
                let outputDirectory = try temporaryOutputDirectory(directoryName: "mere-run-api-geometry")
                try MLXBundleSupport.ensureAvailable(quiet: true)
                let generator = MoGe2Generator()
                do {
                    let result = try await generator.generate(
                        imageURL: inputURL,
                        outputDirectory: outputDirectory,
                        model: nil,
                        configuration: plan.configuration,
                        progress: nil
                    )
                    await generator.unload()
                    return try retainedArtifactJSONResponse(
                        APIServerContract.geometryResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    await generator.unload()
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionMultiViewGeometry(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumMultiViewGeometryUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or multi-view upload exceeds 512 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.multiViewGeometryPlan(from: form)
                let imageParts = form.parts.filter {
                    $0.filename != nil && ($0.name == "image" || $0.name == "image[]")
                }
                let inputURLs = try imageParts.map {
                    try writeMultipartFile(
                        $0,
                        directoryName: "mere-run-api-geometry-multiview-inputs",
                        defaultExtension: "png"
                    )
                }
                defer {
                    for url in inputURLs { try? FileManager.default.removeItem(at: url) }
                }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-geometry-multiview"
                )
                try MLXBundleSupport.ensureAvailable(quiet: true)
                let generator = DepthAnything3Generator()
                do {
                    let result = try await generator.generate(
                        imageURLs: inputURLs,
                        model: plan.modelID,
                        knownCameras: plan.knownCameras,
                        referenceViewStrategy: plan.referenceViewStrategy,
                        processResolution: plan.processResolution,
                        progress: nil
                    )
                    let exportStart = Date()
                    let export = try MultiViewGeometryExporter.export(
                        run: result,
                        outputDirectory: outputDirectory,
                        configuration: try MultiViewGeometryExportConfiguration(
                            confidencePercentile: plan.confidencePercentile,
                            maximumPointCount: plan.maximumPointCount
                        )
                    )
                    let exportSeconds = Date().timeIntervalSince(exportStart)
                    await generator.unload()
                    return try retainedArtifactJSONResponse(
                        APIServerContract.multiViewGeometryResponse(
                            from: result,
                            export: export,
                            exportSeconds: exportSeconds
                        ),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    await generator.unload()
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionImageTo3D(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumImageTo3DUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or image upload exceeds 100 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.imageTo3DPlan(from: form)
                guard let image = form.file(named: "image") else {
                    throw APIRequestValidationError.invalidField(
                        "image",
                        "exactly one non-empty uploaded image file is required"
                    )
                }
                let inputURL = try writeMultipartFile(
                    image,
                    directoryName: "mere-run-api-image-to-3d-inputs",
                    defaultExtension: "png"
                )
                defer { try? FileManager.default.removeItem(at: inputURL) }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-image-to-3d"
                )
                try MLXBundleSupport.ensureAvailable(quiet: true)
                let generator = TripoSRGenerator()
                do {
                    let result = try await generator.generate(
                        imageURL: inputURL,
                        outputDirectory: outputDirectory,
                        model: plan.modelID,
                        foregroundPolicy: plan.foregroundPolicy,
                        extractionResolution: plan.extractionResolution,
                        densityThreshold: plan.densityThreshold,
                        includeVertexColors: plan.includesVertexColors,
                        progress: nil
                    )
                    await generator.unload()
                    return try retainedArtifactJSONResponse(
                        APIServerContract.imageTo3DResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    await generator.unload()
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionInstantMesh(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumInstantMeshUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or multi-view upload exceeds 512 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.instantMeshPlan(from: form)
                let uploads = form.parts
                    .filter { $0.filename != nil && ($0.name == "image" || $0.name == "image[]") }
                    .filter { !$0.body.isEmpty }
                let inputURLs = try uploads.map {
                    try writeMultipartFile(
                        $0,
                        directoryName: "mere-run-api-instantmesh-inputs",
                        defaultExtension: "png"
                    )
                }
                defer {
                    for inputURL in inputURLs {
                        try? FileManager.default.removeItem(at: inputURL)
                    }
                }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-instantmesh"
                )
                try MLXBundleSupport.ensureAvailable(quiet: true)
                let generator = InstantMeshGenerator()
                do {
                    let result = try await generator.generate(
                        viewURLs: inputURLs,
                        outputDirectory: outputDirectory,
                        model: plan.modelID,
                        cameras: plan.cameras,
                        extractionResolution: plan.extractionResolution,
                        includeVertexColors: plan.includesVertexColors,
                        progress: nil
                    )
                    await generator.unload()
                    return try retainedArtifactJSONResponse(
                        APIServerContract.instantMeshResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    await generator.unload()
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionDepthVideo(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumDepthVideoUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or video upload exceeds 512 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.depthVideoPlan(from: form)
                guard let video = form.file(named: "video") else {
                    throw APIRequestValidationError.invalidField(
                        "video",
                        "exactly one non-empty uploaded video file is required"
                    )
                }
                let inputURL = try writeMultipartFile(
                    video,
                    directoryName: "mere-run-api-depth-video-inputs",
                    defaultExtension: "mp4"
                )
                defer { try? FileManager.default.removeItem(at: inputURL) }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-depth-video"
                )
                try MLXBundleSupport.ensureAvailable(quiet: true)
                let generator = VideoDepthAnythingGenerator()
                do {
                    let result = try await generator.generate(
                        videoURL: inputURL,
                        outputDirectory: outputDirectory,
                        model: plan.modelID,
                        inputSize: plan.inputSize,
                        maximumFrameCount: plan.maximumFrameCount,
                        progress: nil
                    )
                    await generator.unload()
                    return try retainedArtifactJSONResponse(
                        APIServerContract.depthVideoResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    await generator.unload()
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleAudioSpeech(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(
                status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let openaiRequest = try APIServerContract.decodeSpeechRequest(
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.speechPlan(from: openaiRequest)
            let admissionLease = try await requestAdmission.acquire()
            do {
                let outputURL = try await synthesizeSpeech(plan)
                let responseURL = try speechResponseURL(outputURL, responseFormat: plan.responseFormat)
                let data = try Data(contentsOf: responseURL)
                let response = binaryResponse(
                    data,
                    contentType: APIServerContract.speechContentType(for: plan.responseFormat)
                )
                await admissionLease.release()
                return response
            } catch {
                await admissionLease.release()
                throw error
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleAudioTranscriptions(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 100 * 1024 * 1024)
        } catch {
            return makeErrorResponse(
                status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let form = try MultipartFormData.parse(body: Data(body.readableBytesView), boundary: boundary)
            let plan = try APIServerContract.transcriptionPlan(from: form)
            guard let file = form.file(named: "file"), !file.body.isEmpty else {
                throw APIRequestValidationError.invalidField("file", "audio file is required")
            }
            let audioURL = try writeMultipartFile(file, directoryName: "mere-run-api-audio")
            let admissionLease = try await requestAdmission.acquire()
            do {
                let result = try await transcribeAudio(audioURL: audioURL, plan: plan)
                let response: Response
                switch plan.responseFormat {
                case "text":
                    response = binaryResponse(Data(result.text.utf8), contentType: "text/plain; charset=utf-8")
                case "srt", "vtt":
                    let subtitle = APIServerContract.transcriptionSubtitle(from: result, format: plan.responseFormat)
                    let contentType = plan.responseFormat == "srt"
                        ? "application/x-subrip; charset=utf-8"
                        : "text/vtt; charset=utf-8"
                    response = binaryResponse(Data(subtitle.utf8), contentType: contentType)
                case "verbose_json":
                    response = try jsonResponse(
                        APIServerContract.transcriptionResponse(from: result, verbose: true)
                    )
                default:
                    response = try jsonResponse(
                        APIServerContract.transcriptionResponse(from: result, verbose: false)
                    )
                }
                await admissionLease.release()
                return response
            } catch {
                await admissionLease.release()
                throw error
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleNonStreamingChat(
        _ request: ChatRequest,
        modelID: String,
        lease: RuntimeModelLease,
        admissionLease: RuntimeRequestAdmissionLease
    ) async throws -> Response {
        defer {
            Task {
                await lease.release()
                await admissionLease.release()
            }
        }
        let result = try await lease.chat(request, progressHandler: nil)

        let response = OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelID,
            choices: [
                OpenAIChatChoice(
                    index: 0,
                    message: OpenAIChatMessage(
                        role: "assistant",
                        content: result.response,
                        reasoning_content: result.reasoningContent,
                        tool_calls: openAIToolCalls(from: result.toolCalls)
                    ),
                    finish_reason: Self.openAIFinishReason(for: result)
                )
            ],
            usage: Self.openAIUsage(for: result)
        )

        let data = try JSONEncoder().encode(response)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func embeddingModel(for requestedModel: String) async throws -> (modelID: String, modelPath: String) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if let spec = ManagedModelCatalog.spec(for: normalized),
           spec.id != Qwen3EmbeddingCatalog.modelId {
            throw APIRequestValidationError.invalidField(
                "model",
                "use \(Qwen3EmbeddingCatalog.modelId) or a local Qwen3 embedding model path"
            )
        }

        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: normalized,
            defaultModelID: Qwen3EmbeddingCatalog.modelId,
            progress: nil
        )
        let modelID = resolution.source == .explicitPath ? normalized : resolution.spec.id
        return (modelID, resolution.url.standardizedFileURL.path)
    }

    private func generateImage(_ plan: APIServerContract.ImageGenerationPlan) async throws -> URL {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        if plan.inputImage != nil,
           QwenImageEditRepository.canonicalModelId(for: plan.modelID) != nil {
            return try await generateQwenImageEdit(plan)
        }
        let resolved = try resolveImageModel(plan.modelID)
        let effectiveSteps = plan.steps
            ?? ((resolved.manifest.family == .hidream || resolved.manifest.family == .krea || resolved.manifest.family == .ideogram)
                ? (resolved.manifest.defaults?.steps ?? 4)
                : 4)
        let effectiveCFG = plan.guidanceScale
            ?? ((resolved.manifest.family == .hidream || resolved.manifest.family == .krea || resolved.manifest.family == .ideogram)
                ? (resolved.manifest.defaults?.cfg ?? 1.0)
                : 1.0)
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-images", extension: "png")
        let request = GenerationRequest(
            prompt: plan.prompt,
            negativePrompt: plan.negativePrompt,
            referenceImages: plan.additionalInputImages,
            width: plan.width,
            height: plan.height,
            steps: effectiveSteps,
            guidanceScale: effectiveCFG,
            seed: plan.seed,
            outputURL: outputURL,
            model: resolved.rootURL.path,
            maxSequenceLength: 512,
            lora: nil,
            enhancePrompt: false,
            inputImage: plan.inputImage,
            strength: plan.strength ?? 0.75,
            keepOriginalAspect: false,
            useBetaSigmas: false,
            sigmaShift: resolved.manifest.defaults?.sigmaShift.map { Float($0) }
        )

        switch resolved.manifest.family {
        case .klein:
            _ = try await sidecarPool.generateImage(
                kind: .flux2Klein,
                modelID: resolved.modelID,
                modelPath: resolved.rootURL.path,
                request: request
            )
        case .zimage:
            _ = try await sidecarPool.generateImage(
                kind: .zImageTurbo,
                modelID: resolved.modelID,
                modelPath: resolved.rootURL.path,
                request: request
            )
        case .hidream:
            _ = try await sidecarPool.generateImage(
                kind: .hiDreamO1,
                modelID: resolved.modelID,
                modelPath: resolved.rootURL.path,
                request: request
            )
        case .krea:
            _ = try await sidecarPool.generateImage(
                kind: .krea2,
                modelID: resolved.modelID,
                modelPath: resolved.rootURL.path,
                request: request
            )
        case .ideogram:
            _ = try await sidecarPool.generateImage(
                kind: .ideogram4,
                modelID: resolved.modelID,
                modelPath: resolved.rootURL.path,
                request: request
            )
        case .gemma, .liquid, .qwen, .sam, .falcon, .face, .geometry, .depth, .threeD,
             .tts, .asr, .embed, .code, .ocr, .music, .sfx, .video, .psi, .privacy, .deepseek, nil:
            throw APIRequestValidationError.invalidField(
                "model",
                "model \(resolved.modelID) is not an image generation model"
            )
        }
        return outputURL
    }

    private func generateQwenImageEdit(_ plan: APIServerContract.ImageGenerationPlan) async throws -> URL {
        guard let modelRoot = QwenImageEditRepository.resolveInstalledModelRoot() else {
            throw APIRequestValidationError.invalidField(
                "model",
                "qwen-image-edit is not installed; pull it before serving image edits"
            )
        }
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-images", extension: "png")
        let request = GenerationRequest(
            prompt: plan.prompt,
            negativePrompt: plan.negativePrompt,
            width: plan.width,
            height: plan.height,
            steps: plan.steps ?? 20,
            guidanceScale: plan.guidanceScale ?? 4.0,
            seed: plan.seed,
            outputURL: outputURL,
            model: modelRoot.path,
            maxSequenceLength: 512,
            inputImage: plan.inputImage,
            strength: plan.strength ?? 0.75
        )
        _ = try await sidecarPool.generateImage(
            kind: .qwenImageEdit,
            modelID: QwenImageEditRepository.modelId,
            modelPath: modelRoot.path,
            request: request
        )
        return outputURL
    }

    private func synthesizeSpeech(_ plan: APIServerContract.SpeechPlan) async throws -> URL {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        let selection = try resolveSpeechModel(plan.modelID)
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-speech", extension: "wav")
        let request = TTSRequest(
            text: plan.input,
            voiceDescription: plan.voiceDescription,
            voiceMode: .style,
            cloneReference: nil,
            language: "auto",
            speed: plan.speed,
            temperature: plan.temperature,
            outputURL: outputURL
        )
        _ = try await sidecarPool.synthesizeSpeech(
            modelID: selection.modelID,
            modelPath: selection.modelPath,
            request: request
        )
        return outputURL
    }

    private func speechResponseURL(_ wavURL: URL, responseFormat: String) throws -> URL {
        guard responseFormat != "wav" else {
            return wavURL
        }
        let outputURL = try temporaryOutputURL(
            directoryName: "mere-run-api-speech",
            extension: responseFormat
        )
        try MediaAudioIO.transcode(wavURL, to: outputURL, format: responseFormat)
        return outputURL
    }

    private func transcribeAudio(
        audioURL: URL,
        plan: APIServerContract.TranscriptionPlan
    ) async throws -> ASRResult {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        let selection = try resolveTranscriptionModel(plan.modelID)
        let request = ASRRequest(
            audioURL: audioURL,
            language: plan.language,
            task: plan.task,
            maxTokens: plan.maxTokens
        )
        let execution = try await CLIASRRouting.transcribe(
            request: request,
            preferredBackend: selection.backend,
            modelOverride: selection.modelOverride,
            progressHandler: nil,
            executor: sidecarPool
        )
        return execution.result
    }

    private func resolveImageModel(
        _ requestedModel: String
    ) throws -> (modelID: String, rootURL: URL, manifest: MereRunModelManifest) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            let manifest = try MereRunModelManifest.loadRequired(from: asPath)
            return (manifest.id, asPath, manifest)
        }
        guard let modelID = ModelResolver.ModelID(rawValue: normalized) else {
            throw APIRequestValidationError.invalidField(
                "model",
                "use a mere.run image model id or a local model path"
            )
        }
        let resolution = try ModelResolver().resolve(modelID)
        let manifest = try MereRunModelManifest.loadRequired(from: resolution.rootURL)
        return (modelID.rawValue, resolution.rootURL, manifest)
    }

    private func resolveSpeechModel(
        _ requestedModel: String
    ) throws -> (modelID: String, modelPath: String?) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            return (Qwen3TTSResources.defaultModelId, asPath.path)
        }
        if let spec = ManagedModelCatalog.spec(for: normalized),
           spec.category == .speechTTS {
            return (spec.id, nil)
        }
        throw APIRequestValidationError.invalidField(
            "model",
            "use a mere.run TTS model id or a local Qwen3-TTS model path"
        )
    }

    private func resolveTranscriptionModel(
        _ requestedModel: String
    ) throws -> (modelOverride: String?, backend: ASRBackend) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            return (asPath.path, .auto)
        }
        guard let spec = ManagedModelCatalog.spec(for: normalized),
              spec.category == .speechASR else {
            throw APIRequestValidationError.invalidField(
                "model",
                "use a mere.run ASR model id or a local ASR model path"
            )
        }
        let backend: ASRBackend = spec.id.contains("parakeet") ? .parakeet : .qwen
        return (spec.id, backend)
    }

    private nonisolated func temporaryOutputURL(
        directoryName: String,
        extension pathExtension: String
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("\(UUID().uuidString).\(pathExtension)")
    }

    private nonisolated func temporaryOutputDirectory(directoryName: String) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let directory = parent.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        return directory
    }

    private nonisolated func writeMultipartFile(
        _ file: MultipartFormData.Part,
        directoryName: String,
        defaultExtension: String = "wav"
    ) throws -> URL {
        let pathExtension = sanitizedPathExtension(from: file.filename) ?? defaultExtension
        let outputURL = try temporaryOutputURL(directoryName: directoryName, extension: pathExtension)
        try file.body.write(to: outputURL)
        return outputURL
    }

    private nonisolated func sanitizedPathExtension(from filename: String?) -> String? {
        guard let rawExtension = filename.flatMap({ URL(fileURLWithPath: $0).pathExtension.lowercased() }),
              !rawExtension.isEmpty,
              rawExtension.allSatisfy({ $0.isLetter || $0.isNumber }) else {
            return nil
        }
        return rawExtension
    }

    private func handleRuntimeStatus(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let admission = await requestAdmission.snapshot()
        let sidecars = await sidecarPool.status()
        let data = try JSONEncoder().encode(
            await pool.status(admission: admission, sidecars: sidecars)
        )
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func handleRuntimeLoad(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let admissionLease: RuntimeRequestAdmissionLease
        do {
            admissionLease = try await requestAdmission.acquire()
        } catch {
            return runtimeErrorResponse(error)
        }
        do {
            let snapshot = try await pool.loadModel(idOrAlias: id)
            let response = try jsonResponse(snapshot)
            await admissionLease.release()
            return response
        } catch {
            await admissionLease.release()
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeUnload(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let admissionLease: RuntimeRequestAdmissionLease
        do {
            admissionLease = try await requestAdmission.acquire()
        } catch {
            return runtimeErrorResponse(error)
        }
        do {
            let snapshot = try await pool.unloadModel(idOrAlias: id)
            let response = try jsonResponse(snapshot)
            await admissionLease.release()
            return response
        } catch {
            await admissionLease.release()
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeSettings(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let settings = try await pool.settings(idOrAlias: id)
            return try jsonResponse(settings)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeSettingsPatch(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }
        do {
            let settings = try JSONDecoder().decode(RuntimeModelSettings.self, from: Data(body.readableBytesView))
            let updated = try await pool.updateSettings(idOrAlias: id, settings: settings)
            return try jsonResponse(updated)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func proxyDeepseekV4FlashChatCompletions(
        body: ByteBuffer,
        contentType: String?,
        lease: RuntimeModelLease,
        admissionLease: RuntimeRequestAdmissionLease
    ) async throws -> Response {
        let upstreamURL = try await lease.deepseekChatCompletionsURL(progressHandler: nil)
        let data = Data(body.readableBytesView)

        if !DeepseekV4FlashClient.requestWantsStreamingResponse(data) {
            let upstreamResponse: DeepseekV4FlashClientResponse
            do {
                upstreamResponse = try await DeepseekV4FlashClient.normalizedChatCompletionData(
                    url: upstreamURL,
                    requestBody: data,
                    contentType: contentType
                )
            } catch {
                await lease.release()
                await admissionLease.release()
                throw error
            }
            await lease.release()
            await admissionLease.release()
            var headers: HTTPFields = [:]
            headers[.contentType] = upstreamResponse.contentType
            return Response(
                status: .init(code: upstreamResponse.statusCode),
                headers: headers,
                body: .init(byteBuffer: ByteBuffer(bytes: upstreamResponse.body))
            )
        }

        let upstreamRequest = DeepseekV4FlashClient.makeChatCompletionsRequest(
            url: upstreamURL,
            requestBody: data,
            contentType: contentType
        )
#if os(Linux)
        let (upstreamData, response) = try await URLSession.shared.data(for: upstreamRequest)
        let http = response as? HTTPURLResponse
        var headers: HTTPFields = [:]
        headers[.contentType] = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        if DeepseekV4FlashClient.isEventStreamContentType(headers[.contentType]) {
            headers[.init("Cache-Control")!] = "no-cache"
            headers[.connection] = "keep-alive"
            let stream = AsyncStream<ByteBuffer> { continuation in
                if !upstreamData.isEmpty {
                    continuation.yield(ByteBuffer(bytes: upstreamData))
                }
                Task {
                    await lease.release()
                    await admissionLease.release()
                }
                continuation.finish()
            }
            return Response(
                status: .init(code: http?.statusCode ?? 502),
                headers: headers,
                body: .init(asyncSequence: stream)
            )
        }
        let repairedData = DeepseekV4FlashClient.normalizedChatCompletionBody(
            upstreamData,
            contentType: headers[.contentType]
        )
        await lease.release()
        await admissionLease.release()
        return Response(
            status: .init(code: http?.statusCode ?? 502),
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: repairedData))
        )
#else
        let (upstreamBytes, response) = try await URLSession.shared.bytes(for: upstreamRequest)
        let http = response as? HTTPURLResponse
        var headers: HTTPFields = [:]
        headers[.contentType] = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        if DeepseekV4FlashClient.isEventStreamContentType(headers[.contentType]) {
            headers[.init("Cache-Control")!] = "no-cache"
            headers[.connection] = "keep-alive"
            let stream = AsyncStream<ByteBuffer> { continuation in
                Task {
                    var buffer = Data()
                    buffer.reserveCapacity(4_096)
                    do {
                        for try await byte in upstreamBytes {
                            buffer.append(byte)
                            if byte == 10 || buffer.count >= 4_096 {
                                continuation.yield(ByteBuffer(bytes: buffer))
                                buffer.removeAll(keepingCapacity: true)
                            }
                        }
                        if !buffer.isEmpty {
                            continuation.yield(ByteBuffer(bytes: buffer))
                        }
                        await lease.release()
                        await admissionLease.release()
                        continuation.finish()
                    } catch {
                        await lease.release()
                        await admissionLease.release()
                        continuation.finish()
                    }
                }
            }
            return Response(
                status: .init(code: http?.statusCode ?? 502),
                headers: headers,
                body: .init(asyncSequence: stream)
            )
        }

        var upstreamData = Data()
        do {
            for try await byte in upstreamBytes {
                upstreamData.append(byte)
            }
        } catch {
            await lease.release()
            await admissionLease.release()
            throw error
        }
        await lease.release()
        await admissionLease.release()
        let repairedData = DeepseekV4FlashClient.normalizedChatCompletionBody(
            upstreamData,
            contentType: headers[.contentType]
        )
        return Response(
            status: .init(code: http?.statusCode ?? 502),
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: repairedData))
        )
#endif
    }

    private func handleStreamingChat(
        _ request: ChatRequest,
        modelID: String,
        includeUsage: Bool,
        lease: RuntimeModelLease,
        admissionLease: RuntimeRequestAdmissionLease
    ) async throws -> Response {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let encoder = JSONEncoder()

        // Create async stream for SSE
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()

        // Start generation in a detached task
        Task {
            do {
                let streamedContent = StreamingContentTracker()
                let shouldBufferForToolCalls = request.tools?.isEmpty == false
                let result = try await lease.chat(request) { progress in
                    guard !shouldBufferForToolCalls else { return }
                    if progress.stage == .generating,
                       let token = progress.message,
                       !token.isEmpty,
                       !APIServerContract.isStreamingStatusMessage(token) {
                        streamedContent.markStreamed()
                        let chunk = OpenAIChatResponse(
                            id: id,
                            object: "chat.completion.chunk",
                            created: Int(Date().timeIntervalSince1970),
                            model: modelID,
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
                if let toolCalls = openAIToolCalls(from: result.toolCalls), !toolCalls.isEmpty {
                    let chunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: OpenAIChatDelta(
                                    role: "assistant",
                                    tool_calls: toolCalls.enumerated().map(OpenAIChatToolCallDelta.init(indexAndToolCall:))
                                ),
                                finish_reason: nil
                            )
                        ]
                    )
                    if let data = try? encoder.encode(chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                } else if !streamedContent.didStream, !result.response.isEmpty {
                    let chunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: OpenAIChatDelta(
                                    content: result.response,
                                    reasoning_content: result.reasoningContent
                                ),
                                finish_reason: nil
                            )
                        ]
                    )
                    if let data = try? encoder.encode(chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }

                // Final chunk with finish_reason
                let finalChunk = OpenAIChatResponse(
                    id: id,
                    object: "chat.completion.chunk",
                    created: Int(Date().timeIntervalSince1970),
                    model: modelID,
                    choices: [
                        OpenAIChatChoice(
                            index: 0,
                            delta: OpenAIChatDelta(),
                            finish_reason: Self.openAIFinishReason(for: result)
                        )
                    ]
                )
                if let data = try? encoder.encode(finalChunk),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                }

                if includeUsage {
                    let usageChunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [],
                        usage: Self.openAIUsage(for: result)
                    )
                    if let data = try? encoder.encode(usageChunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }

                continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
                await lease.release()
                await admissionLease.release()
                continuation.finish()
            } catch {
                // Send error in SSE format
                let errorResponse = OpenAIErrorResponse(
                    error: OpenAIError(message: "Request failed.", type: "server_error")
                )
                if let data = try? encoder.encode(errorResponse),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                }
                await lease.release()
                await admissionLease.release()
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

    private nonisolated func openAIToolCalls(from toolCalls: [ToolCall]?) -> [OpenAIChatToolCall]? {
        guard let toolCalls, !toolCalls.isEmpty else { return nil }
        return toolCalls.enumerated().map { index, call in
            OpenAIChatToolCall(
                id: "call_\(index)_\(UUID().uuidString.prefix(8))",
                function: OpenAIChatToolCallFunction(
                    name: call.name,
                    arguments: jsonString(from: call.arguments)
                )
            )
        }
    }

    nonisolated static func openAIFinishReason(for result: ChatResponse) -> String {
        if result.toolCalls?.isEmpty == false {
            return "tool_calls"
        }
        return result.finishReason == .length ? "length" : "stop"
    }

    /// Generators that don't report a prompt token count fall back to zero
    /// rather than omitting the usage object, matching OpenAI's schema.
    nonisolated static func openAIUsage(for result: ChatResponse) -> OpenAIUsage {
        let promptTokens = result.promptTokens ?? 0
        return OpenAIUsage(
            prompt_tokens: promptTokens,
            completion_tokens: result.tokensGenerated,
            total_tokens: promptTokens + result.tokensGenerated
        )
    }

    private nonisolated func jsonString(from arguments: [String: String]) -> String {
        let data = (try? JSONEncoder().encode(arguments)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private func unauthorizedResponseIfNeeded(for request: Request) -> Response? {
        guard let apiKey, !apiKey.isEmpty else { return nil }
        guard request.headers[.authorization] == "Bearer \(apiKey)" else {
            return makeErrorResponse(
                status: .unauthorized,
                message: "Unauthorized.",
                type: "authentication_error",
                extraHeaders: [.init("WWW-Authenticate")!: "Bearer"]
            )
        }
        return nil
    }

    /// Authentication is evaluated first so a remote caller never learns
    /// route details without a valid bearer token on non-loopback binds.
    private func vfxArtifactAccessResponseIfNeeded(
        for request: Request,
        remoteAddress: SocketAddress?
    ) -> Response? {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIVFXArtifactRoutePolicy.allows(remoteAddress: remoteAddress) else {
            return makeErrorResponse(
                status: .forbidden,
                message: APIVFXArtifactRoutePolicy.denialMessage,
                type: "permission_error"
            )
        }
        return nil
    }

    private func retainedArtifactJSONResponse<T: Encodable>(
        _ payload: T,
        outputDirectory: URL
    ) throws -> Response {
        let response = try jsonResponse(payload)
        artifactCleanupScheduler.scheduleCleanup(of: outputDirectory)
        return response
    }

    private nonisolated func jsonResponse<T: Encodable>(
        _ payload: T,
        status: HTTPResponse.Status = .ok
    ) throws -> Response {
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private nonisolated func binaryResponse(_ data: Data, contentType: String) -> Response {
        Response(
            status: .ok,
            headers: [.contentType: contentType],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private nonisolated func runtimeErrorResponse(_ error: Error) -> Response {
        if let status = APIVFXClientErrorPolicy.status(for: error) {
            return makeErrorResponse(
                status: status,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        }
        switch error {
        case let error as RuntimeModelPoolError:
            switch error {
            case .unknownModel, .unsupportedModel, .modelNotInstalled, .incompatibleEngine, .invalidSettings:
                return makeErrorResponse(
                    status: .badRequest,
                    message: error.localizedDescription,
                    type: "invalid_request_error"
                )
            case .unloadConflict:
                return makeErrorResponse(
                    status: .conflict,
                    message: error.localizedDescription,
                    type: "conflict_error"
                )
            case .rawProxyUnavailable:
                return makeErrorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    type: "server_error"
                )
            }
        case let error as APISidecarModelPoolError:
            return makeErrorResponse(
                status: .serviceUnavailable,
                message: error.localizedDescription,
                type: "memory_pressure_error"
            )
        case let error as APIRequestValidationError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as MultipartFormData.ParseError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as ManagedModelResolver.ResolverError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as InstantMeshResourceError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as Qwen3EmbeddingModel.EmbeddingError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        default:
            return makeErrorResponse(
                status: .internalServerError,
                message: "Request failed.",
                type: "server_error"
            )
        }
    }

    private nonisolated func makeErrorResponse(
        status: HTTPResponse.Status,
        message: String,
        type: String,
        extraHeaders: HTTPFields = [:]
    ) -> Response {
        let payload = OpenAIErrorResponse(error: OpenAIError(message: message, type: type))
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{\"error\":{\"message\":\"\(message)\",\"type\":\"\(type)\"}}".utf8)
        var headers: HTTPFields = [.contentType: "application/json"]
        for field in extraHeaders {
            headers.append(field)
        }
        return Response(
            status: status,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}

actor APIRateLimiter {
    private let limitPerMinute: Int
    private var requestTimes: [Date] = []

    init(limitPerMinute: Int) {
        self.limitPerMinute = limitPerMinute
    }

    func allowRequest(now: Date = Date()) -> Bool {
        let cutoff = now.addingTimeInterval(-60)
        requestTimes.removeAll { $0 < cutoff }
        guard requestTimes.count < limitPerMinute else {
            return false
        }
        requestTimes.append(now)
        return true
    }
}

final class StreamingContentTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var streamed = false

    var didStream: Bool {
        lock.lock()
        defer { lock.unlock() }
        return streamed
    }

    func markStreamed() {
        lock.lock()
        streamed = true
        lock.unlock()
    }
}
