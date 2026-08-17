import Foundation
import XCTest
import AudioCore
import MediaIO
import NIOCore
@testable import MereRunCLI
@testable import MereRunCore

private actor VFXRequestAdmissionTestGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var isOpen = false

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func open() {
        isOpen = true
        continuation?.resume()
        continuation = nil
    }
}

private enum VFXRequestAdmissionTestError: Error {
    case expected
}

final class APIServeCommandTests: XCTestCase {
    func testAPICommandExposesServeSubcommand() {
        let commandNames = Set(API.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["serve"]))
    }

    func testAPIServeParsesDefaults() throws {
        let cmd = try APIServe.parse([])

        XCTAssertEqual(cmd.port, 8080)
        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.engine, .textChatQ36)
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.lora)
        XCTAssertNil(cmd.apiKey)
        XCTAssertEqual(cmd.rateLimitPerMinute, 60)
        XCTAssertEqual(cmd.maxActiveRequests, 1)
        XCTAssertEqual(cmd.memoryGuard, .balanced)
        XCTAssertNil(cmd.memoryGuardCustomCeilingGB)
        XCTAssertEqual(cmd.contextSize, 32_768)
        XCTAssertNil(cmd.kvBits)
        XCTAssertNil(cmd.kvQuantScheme)
        XCTAssertNil(cmd.kvGroupSize)
        XCTAssertNil(cmd.quantizedKVStart)
        XCTAssertFalse(cmd.preflight)
        XCTAssertFalse(cmd.json)
    }

    func testAPIServeParsesOverrides() throws {
        let cmd = try APIServe.parse([
            "--host", "0.0.0.0",
            "--port", "11434",
            "--engine", "text-chat-gemma4",
            "--model-path", "/tmp/gemma4",
            "--lora", "/tmp/adapter.safetensors",
            "--api-key", "secret",
            "--rate-limit-per-minute", "120",
            "--max-active-requests", "2",
            "--memory-guard", "custom",
            "--memory-guard-custom-ceiling-gb", "42.5",
            "--context-size", "8192",
            "--kv-bits", "3.5",
            "--kv-quant-scheme", "turboquant",
            "--kv-group-size", "32",
            "--quantized-kv-start", "2048",
        ])

        XCTAssertEqual(cmd.host, "0.0.0.0")
        XCTAssertEqual(cmd.port, 11_434)
        XCTAssertEqual(cmd.engine, .textChatGemma4)
        XCTAssertEqual(cmd.model, "/tmp/gemma4")
        XCTAssertEqual(cmd.lora, "/tmp/adapter.safetensors")
        XCTAssertEqual(cmd.apiKey, "secret")
        XCTAssertEqual(cmd.rateLimitPerMinute, 120)
        XCTAssertEqual(cmd.maxActiveRequests, 2)
        XCTAssertEqual(cmd.memoryGuard, .custom)
        XCTAssertEqual(cmd.memoryGuardCustomCeilingGB, 42.5)
        XCTAssertEqual(cmd.contextSize, 8192)
        XCTAssertEqual(cmd.kvBits, 3.5)
        XCTAssertEqual(cmd.kvQuantScheme, "turboquant")
        XCTAssertEqual(cmd.kvGroupSize, 32)
        XCTAssertEqual(cmd.quantizedKVStart, 2048)
    }

    func testAPIServeParsesMuseGlimmerVisionAgentEngine() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-muse-glimmer",
            "--model", MuseGlimmerResources.modelId,
        ])

        XCTAssertEqual(cmd.engine, .textChatMuseGlimmer)
        XCTAssertEqual(cmd.model, MuseGlimmerResources.modelId)
        XCTAssertEqual(cmd.defaultRuntimeModelID(modelPath: nil), MuseGlimmerResources.modelId)
        XCTAssertTrue(cmd.engine.openAICompatibility.supportsTools)
        XCTAssertTrue(cmd.engine.openAICompatibility.supportsVisionContentParts)
        XCTAssertTrue(cmd.engine.openAICompatibility.supportsReasoningEffort)
        XCTAssertFalse(cmd.engine.openAICompatibility.supportsStructuredOutputs)
    }

    func testAPIServeParsesNemotronHNativeEngine() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-nemotron-h",
            "--model", NemotronHResources.modelID,
        ])

        XCTAssertEqual(cmd.engine, .textChatNemotronH)
        XCTAssertEqual(cmd.model, NemotronHResources.modelID)
        XCTAssertEqual(cmd.defaultRuntimeModelID(modelPath: nil), NemotronHResources.modelID)
        XCTAssertTrue(cmd.engine.openAICompatibility.supportsTools)
        XCTAssertFalse(cmd.engine.openAICompatibility.supportsVisionContentParts)
        XCTAssertFalse(cmd.engine.openAICompatibility.supportsStructuredOutputs)
    }

    func testAPIServeParsesPreflightJSONFlags() throws {
        let cmd = try APIServe.parse([
            "--preflight",
            "--json",
        ])

        XCTAssertTrue(cmd.preflight)
        XCTAssertTrue(cmd.json)
    }

    func testAPIServePreflightReportsLoopbackServerPlan() throws {
        let cmd = try APIServe.parse([
            "--host", "127.0.0.1",
            "--port", "11434",
            "--engine", "text-chat-q36",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            environment: [:],
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.command, ["api", "serve"])
        XCTAssertEqual(envelope.mode, .preflight)
        XCTAssertEqual(envelope.result.server.baseURL, "http://127.0.0.1:11434")
        XCTAssertTrue(envelope.result.server.loopback)
        XCTAssertFalse(envelope.result.server.requiresAPIKey)
        XCTAssertFalse(envelope.result.server.apiKeyPresent)
        XCTAssertEqual(envelope.result.runtime.engine, "text-chat-q36")
        XCTAssertEqual(envelope.result.runtime.runtimeServingEngine, "text-chat-q36")
        XCTAssertEqual(envelope.result.model.defaultModelID, ModelResolver.ModelID.q36Nano.rawValue)
        XCTAssertEqual(envelope.result.model.kind, "managed_model")
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-api-server" })
        XCTAssertTrue(envelope.actions.contains { $0.id == "check-status" })

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(APIServePreflightEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded.result.server.port, 11_434)
        XCTAssertEqual(decoded.request.apiKeySource, "none")
    }

    func testAPIServePreflightBlocksNonLoopbackWithoutAPIKey() throws {
        let cmd = try APIServe.parse([
            "--host", "0.0.0.0",
            "--port", "11434",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            environment: [:],
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "api_key_required_for_non_loopback" })
        let startServer = try XCTUnwrap(envelope.actions.first { $0.id == "start-api-server" })
        XCTAssertFalse(startServer.enabled)
    }

    func testVFXArtifactRoutePolicyUsesPeerSocketAndFailsClosed() throws {
        XCTAssertEqual(APIVFXArtifactRoutePolicy.outputTTLSeconds, 3_600)
        XCTAssertEqual(
            APIArtifactDirectoryCleanupScheduler.productionDelayNanoseconds,
            3_600_000_000_000
        )
        XCTAssertEqual(APIVFXArtifactRoutePolicy.routePaths, [
            "/v1/vision/geometry",
            "/v1/vision/geometry/multiview",
            "/v1/vision/image-to-3d",
            "/v1/vision/image-to-3d-multiview",
            "/v1/vision/depth-video",
            "/v1/videos/generations",
        ])
        XCTAssertTrue(APIVFXArtifactRoutePolicy.denialMessage.contains("loopback-only"))
        XCTAssertTrue(APIVFXArtifactRoutePolicy.denialMessage.contains("file URLs"))

        for address in ["127.0.0.1", "127.42.0.9", "::1", "::ffff:127.0.0.1"] {
            XCTAssertTrue(APIVFXArtifactRoutePolicy.allows(
                remoteAddress: try SocketAddress(ipAddress: address, port: 8_080)
            ), address)
        }
        for address in ["0.0.0.0", "192.0.2.10", "2001:db8::1"] {
            XCTAssertFalse(APIVFXArtifactRoutePolicy.allows(
                remoteAddress: try SocketAddress(ipAddress: address, port: 8_080)
            ), address)
        }
        XCTAssertFalse(APIVFXArtifactRoutePolicy.allows(remoteAddress: nil))
    }

    func testVFXRequestAdmissionSerializesAndReleasesSuccessAndFailure() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let gate = VFXRequestAdmissionTestGate()
        let first = Task {
            try await withVFXRequestAdmission(using: admission) {
                await gate.wait()
                return "first"
            }
        }

        var snapshot = await admission.snapshot()
        for _ in 0..<100 where snapshot.activeRequests == 0 {
            try await Task.sleep(nanoseconds: 10_000_000)
            snapshot = await admission.snapshot()
        }
        XCTAssertEqual(snapshot.activeRequests, 1)

        let second = Task {
            try await withVFXRequestAdmission(using: admission) {
                "second"
            }
        }
        for _ in 0..<100 where snapshot.queuedRequests == 0 {
            try await Task.sleep(nanoseconds: 10_000_000)
            snapshot = await admission.snapshot()
        }
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 1)

        await gate.open()
        let firstResult = try await first.value
        let secondResult = try await second.value
        XCTAssertEqual(firstResult, "first")
        XCTAssertEqual(secondResult, "second")

        do {
            let _: Int = try await withVFXRequestAdmission(using: admission) {
                throw VFXRequestAdmissionTestError.expected
            }
            XCTFail("Expected admitted operation to throw")
        } catch VFXRequestAdmissionTestError.expected {
            // Expected.
        }

        snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 3)
        XCTAssertEqual(snapshot.totalCompletedRequests, 3)
    }

    func testVFXClientInputAndResourceErrorsAreBadRequests() {
        let inputURL = URL(fileURLWithPath: "/tmp/client-input")
        let errors: [Error] = [
            MoGe2TokenGridError.tokenCountOutOfRange(
                actual: 3_601,
                minimum: 1,
                maximum: 3_600
            ),
            VideoDepthAnythingLimitError.inputSizeOutOfRange(
                actual: 0,
                minimum: 14,
                maximum: 1_008
            ),
            DepthAnything3LimitError.viewCountOutOfRange(
                actual: 17,
                minimum: 1,
                maximum: 16
            ),
            MediaIOError.invalidVideoFrameRate(.infinity),
            MediaIOError.imageDecodeFailed(inputURL),
            VFXImageInputValidationError.dimensionLimitExceeded(
                path: inputURL.path,
                width: 9_000,
                height: 1,
                maximum: 8_192
            ),
            DepthAnything3CameraValidationError(index: 0, reason: "invalid focal length"),
            DepthAnything3PreprocessingError.emptyImages,
            VideoDepthAnythingPreprocessingError.emptyFrames,
            VideoDepthAnythingWindowingError.invalidOriginalFrameCount(0),
            MultiViewGeometryExportConfigurationError.invalidMaximumPointCount(0),
            TripoSRPreprocessingError.emptyForeground,
            InstantMeshPreprocessingError.invalidViewCount(3),
        ]

        for error in errors {
            XCTAssertEqual(
                APIVFXClientErrorPolicy.status(for: error),
                .badRequest,
                error.localizedDescription
            )
        }
        XCTAssertNil(APIVFXClientErrorPolicy.status(for: VFXRequestAdmissionTestError.expected))
    }

    func testExtremeAspectMoGeTokenGridErrorIsABadRequest() throws {
        do {
            _ = try MoGe2TokenGrid.resolve(
                imageWidth: 16_384,
                imageHeight: 1,
                requestedTokenCount: MoGe2TokenGrid.maximumTokenCount
            )
            XCTFail("Expected the derived MoGe-2 token grid to exceed the native workload limit")
        } catch let error as MoGe2TokenGridError {
            XCTAssertEqual(
                error,
                .tokenGridExceedsLimit(
                    rows: 1,
                    columns: 7_680,
                    actual: 7_680,
                    maximum: MoGe2TokenGrid.maximumTokenCount
                )
            )
            XCTAssertEqual(APIVFXClientErrorPolicy.status(for: error), .badRequest)
        } catch {
            XCTFail("Expected MoGe2TokenGridError, received \(error)")
        }
    }

    func testRemoteModelListingOmitsLoopbackArtifactModels() {
        let installed = APIVFXArtifactRoutePolicy.modelIDs.union(["image-zimage-nano"])
        let remoteIDs = APIServerContract.companionModelIDs(
            installedModelIDs: installed,
            includeLoopbackArtifactModels: false
        )

        XCTAssertTrue(remoteIDs.contains("image-zimage-nano"))
        XCTAssertTrue(APIVFXArtifactRoutePolicy.modelIDs.isDisjoint(with: remoteIDs))
    }

    func testSuccessfulArtifactDirectoryIsRemovedAfterInjectedShortTTL() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-artifact-ttl-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("artifact".utf8).write(to: root.appendingPathComponent("mesh.glb"))

        let scheduler = APIArtifactDirectoryCleanupScheduler(delayNanoseconds: 25_000_000)
        scheduler.scheduleCleanup(of: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.path))
        for _ in 0..<100 where FileManager.default.fileExists(atPath: root.path) {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    func testAPIServePreflightDoesNotExposeAPIKeyInActionArgv() throws {
        let cmd = try APIServe.parse([
            "--host", "0.0.0.0",
            "--port", "11434",
            "--api-key", "super-secret",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            environment: [:],
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertNotEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.request.apiKeyPresent)
        XCTAssertEqual(envelope.request.apiKeySource, "argument")
        let startServer = try XCTUnwrap(envelope.actions.first { $0.id == "start-api-server" })
        XCTAssertFalse(startServer.command?.argv.contains("super-secret") == true)
        XCTAssertFalse(startServer.command?.argv.contains("--api-key") == true)
    }

    func testAPIServeGemma4TurboModelDefaultsToTurboQuantKVCache() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .turboquant)
        XCTAssertEqual(quantization.groupSize, Gemma4Resources.defaultKVGroupSize)
        XCTAssertEqual(quantization.quantizedStart, Gemma4Resources.defaultTurboQuantizedKVStart)
    }

    func testAPIServeParsesLFM2Engine() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-lfm2",
            "--model-path", LFM2Resources.defaultModelId,
        ])

        XCTAssertEqual(cmd.engine, .textChatLFM2)
        XCTAssertEqual(cmd.model, LFM2Resources.defaultModelId)
    }

    func testAPIServePreservesExplicitDenseLFM2ManagedID() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-lfm2",
            "--model-path", LFM2Resources.denseModelId,
        ])

        XCTAssertEqual(cmd.engine, .textChatLFM2)
        XCTAssertEqual(cmd.model, LFM2Resources.denseModelId)
        XCTAssertEqual(cmd.defaultRuntimeModelID(modelPath: nil), LFM2Resources.denseModelId)
    }

    func testAPIServeParsesLagunaEngineAndCanonicalDefault() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-laguna",
        ])

        XCTAssertEqual(cmd.engine, .textChatLaguna)
        XCTAssertEqual(cmd.engine.runtimeServingEngine, .textChatLaguna)
        XCTAssertEqual(cmd.defaultRuntimeModelID(modelPath: nil), LagunaResources.modelID)
    }

    func testAPIServePreservesExplicitLagunaXSManagedID() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-laguna",
            "--model", LagunaResources.xsModelID,
        ])

        XCTAssertEqual(cmd.engine, .textChatLaguna)
        XCTAssertEqual(
            cmd.defaultRuntimeModelID(modelPath: nil),
            LagunaResources.xsModelID
        )
    }

    func testAPIServeLagunaAcceptsExplicitLocalCheckpointPath() throws {
        let path = "/tmp/Laguna-S-2.1-NVFP4-mlx"
        let cmd = try APIServe.parse([
            "--engine", "text-chat-laguna",
            "--model", path,
        ])

        XCTAssertEqual(try cmd.resolveModelPath(), path)
        XCTAssertEqual(cmd.defaultRuntimeModelID(modelPath: path), "Laguna-S-2.1-NVFP4-mlx")
    }

    func testAPIServeParsesLegacyQ35EngineAlias() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-q35",
        ])

        XCTAssertEqual(cmd.engine, .textChatQ35)
        XCTAssertEqual(cmd.engine.runtimeServingEngine, .textChatQ36)
    }

    func testAPIServeGemma4TurboKVFlagsOverrideIndependently() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
            "--kv-quant-scheme", "uniform",
            "--kv-group-size", "32",
            "--quantized-kv-start", "128",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .uniform)
        XCTAssertEqual(quantization.groupSize, 32)
        XCTAssertEqual(quantization.quantizedStart, 128)
    }

    func testAPIServeGemma4PolarKVFlagsParse() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
            "--kv-bits", "2",
            "--kv-quant-scheme", "polar",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, 2)
        XCTAssertEqual(quantization.scheme, .polar)
    }

    func testHealthContractUsesStablePayload() {
        XCTAssertEqual(APIServerContract.healthStatus(), APIHealthStatus(status: "ok"))
    }

    func testModelsContractUsesProvidedModelID() {
        let response = APIServerContract.modelsResponse(
            modelId: "mererun-test-model",
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.object, "list")
        XCTAssertEqual(response.data.count, 1)
        XCTAssertEqual(response.data.first?.id, "mererun-test-model")
        XCTAssertEqual(response.data.first?.object, "model")
        XCTAssertEqual(response.data.first?.owned_by, "mere.run")
        XCTAssertEqual(response.data.first?.created, 123)
    }

    func testModelsContractCanReturnAliases() {
        let response = APIServerContract.modelsResponse(
            modelIds: ["text-chat-gemma4", "chat-default"],
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.data.map(\.id), ["text-chat-gemma4", "chat-default"])
        XCTAssertEqual(Set(response.data.map(\.owned_by)), Set(["mere.run"]))
    }

    func testChatModelContractDescribesDeepSeekForHarnesses() throws {
        let model = APIServerContract.chatModel(
            id: DeepseekV4FlashResources.defaultModelId,
            name: "DeepSeek V4 Flash",
            profile: .deepseekV4Flash(),
            contextWindow: 32_768,
            maximumOutputTokens: 32_768,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(model.task, "chat.completions")
        XCTAssertEqual(model.reasoning, true)
        XCTAssertEqual(model.thinking_levels, ["off", "minimal", "low", "medium", "high", "xhigh"])
        XCTAssertEqual(model.tool_call, true)
        XCTAssertEqual(model.modalities, OpenAIModelModalities(input: ["text", "image"], output: ["text"]))
        XCTAssertEqual(model.limit, OpenAIModelLimit(context: 32_768, output: 32_768))
        XCTAssertEqual(model.openai_compat?.supports_developer_role, false)
        XCTAssertEqual(model.openai_compat?.supports_reasoning_effort, true)
        XCTAssertEqual(model.openai_compat?.supports_finish_reason, true)
        XCTAssertEqual(model.openai_compat?.max_tokens_field, "max_tokens")
        XCTAssertEqual(model.openai_compat?.thinking_format, "deepseek")
        XCTAssertEqual(model.openai_compat?.thinking_level_map, ["minimal": "low"])
        XCTAssertEqual(model.openai_compat?.requires_reasoning_content_on_assistant_messages, true)

        let json = try XCTUnwrap(String(data: JSONEncoder().encode(model), encoding: .utf8))
        XCTAssertTrue(json.contains("\"tool_call\":true"))
        XCTAssertTrue(json.contains("\"thinking_levels\""))
        XCTAssertTrue(json.contains("\"openai_compat\""))
    }

    func testCompanionModelContractLabelsNonChatTasks() throws {
        let profile = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: ModelResolver.ModelID.qwen3Embedding.rawValue)
        )
        let embedding = APIServerContract.companionModel(
            id: ModelResolver.ModelID.qwen3Embedding.rawValue,
            profile: profile,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(embedding.task, "embeddings")
        XCTAssertEqual(embedding.tool_call, false)
        XCTAssertEqual(
            embedding.modalities,
            OpenAIModelModalities(input: ["text"], output: ["embedding"])
        )
    }

    func testEmbeddingRequestDecodesStringInputAndUnknownFields() throws {
        let data = """
        {
          "model": "text-embed-qwen3-0.6b",
          "input": "semantic search query",
          "encoding_format": "float",
          "user": "rag-test",
          "x-client-extra": { "kept": true }
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: data)

        XCTAssertEqual(request.model, "text-embed-qwen3-0.6b")
        XCTAssertEqual(request.input.texts, ["semantic search query"])
        XCTAssertEqual(request.encoding_format, "float")
        XCTAssertEqual(request.user, "rag-test")
        XCTAssertEqual(request.unknownFields["x-client-extra"]?.objectValue?["kept"]?.boolValue, true)
    }

    func testEmbeddingRequestDecodesArrayInput() throws {
        let data = """
        {
          "model": "text-embed-qwen3-0.6b",
          "input": ["first", "second"]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: data)

        XCTAssertEqual(request.input.texts, ["first", "second"])
    }

    func testEmbeddingContractValidatesSupportedOptions() throws {
        let request = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .array(["first", "second"]),
            encoding_format: "float"
        )

        XCTAssertEqual(try APIServerContract.embeddingTexts(from: request), ["first", "second"])

        let base64Request = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .string("hello"),
            encoding_format: "base64"
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: base64Request)) { error in
            XCTAssertTrue(error.localizedDescription.contains("encoding_format"))
        }

        let dimensionsRequest = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .string("hello"),
            dimensions: 128
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: dimensionsRequest)) { error in
            XCTAssertTrue(error.localizedDescription.contains("dimensions"))
        }

        let emptyRequest = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .array([])
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: emptyRequest)) { error in
            XCTAssertTrue(error.localizedDescription.contains("input"))
        }

        let tooManyInputs = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .array(
                Array(
                    repeating: "x",
                    count: APIServerContract.maxEmbeddingInputCount + 1
                )
            )
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: tooManyInputs)) { error in
            XCTAssertTrue(error.localizedDescription.contains("at most 256 texts"))
        }

        let oversizedInput = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .string(
                String(
                    repeating: "x",
                    count: APIServerContract.maxEmbeddingInputUTF8Bytes + 1
                )
            )
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: oversizedInput)) { error in
            XCTAssertTrue(error.localizedDescription.contains("UTF-8 content"))
        }
    }

    func testEmbeddingContractUsesOpenAICompatibleResponseShape() {
        let response = APIServerContract.embeddingResponse(
            modelId: "text-embed-qwen3-0.6b",
            embeddings: [[0.1, 0.2], [0.3, 0.4]],
            tokenCounts: [3, 5]
        )

        XCTAssertEqual(response.object, "list")
        XCTAssertEqual(response.model, "text-embed-qwen3-0.6b")
        XCTAssertEqual(response.data.count, 2)
        XCTAssertEqual(response.data[0].object, "embedding")
        XCTAssertEqual(response.data[0].index, 0)
        XCTAssertEqual(response.data[0].embedding, [0.1, 0.2])
        XCTAssertEqual(response.usage.prompt_tokens, 8)
        XCTAssertEqual(response.usage.total_tokens, 8)
    }

    func testCompanionModelIDsIncludeNativeOpenAIStyleModalities() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: Set([
            "image-zimage-nano",
            "speech-tts-qwen3-nano",
            "speech-asr-parakeet",
            "text-embed-qwen3-0.6b",
            "qwen-image-edit",
        ]))

        XCTAssertTrue(ids.contains("text-embed-qwen3-0.6b"))
        XCTAssertTrue(ids.contains("image-zimage-nano"))
        XCTAssertTrue(ids.contains("speech-tts-qwen3-nano"))
        XCTAssertTrue(ids.contains("speech-asr-parakeet"))
        XCTAssertTrue(ids.contains("qwen-image-edit"))
        XCTAssertFalse(ids.contains("image-zimage-max"))
        XCTAssertFalse(ids.contains("speech-asr-qwen3"))
    }

    func testCompanionModelIDsIncludeInstalledGeometryPrimitive() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: [
            ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
        ])

        XCTAssertEqual(ids, [ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue])
    }

    func testCompanionModelIDsIncludeInstalledMultiViewGeometryPrimitive() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: [
            ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
        ])

        XCTAssertEqual(ids, [ModelResolver.ModelID.visionGeometryDA3Small.rawValue])
    }

    func testCompanionModelIDsIncludeInstalledImageTo3DPrimitive() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: [
            ModelResolver.ModelID.image3DTripoSR.rawValue,
        ])

        XCTAssertEqual(ids, [ModelResolver.ModelID.image3DTripoSR.rawValue])
    }

    func testCompanionModelIDsIncludeInstalledVideoDepthPrimitives() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: [
            ModelResolver.ModelID.visionDepthVDASmall.rawValue,
            ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
        ])

        XCTAssertEqual(ids, [
            ModelResolver.ModelID.visionDepthVDASmall.rawValue,
            ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
        ])
    }

    func testCompanionModelIDsRejectPinnedPlaceholderInstalls() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-api-pinned-models-\(UUID().uuidString)", isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        for pin in GeometryModelPins.all {
            let modelID = try XCTUnwrap(ModelResolver.ModelID(rawValue: pin.modelID))
            let install = root.appendingPathComponent(pin.modelID, isDirectory: true)
            try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
            for artifact in pin.artifacts {
                try Data([0]).write(to: install.appendingPathComponent(artifact.filename))
            }
            try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
                .write(to: install)
        }

        let ids = APIServerContract.companionModelIDs()
        for pin in GeometryModelPins.all {
            XCTAssertFalse(ids.contains(pin.modelID), pin.modelID)
        }
    }

    func testGeometryContractAcceptsOnlyManagedMoGeAndValidControls() throws {
        let form = MultipartFormData(parts: [
            .init(name: "model", filename: nil, contentType: nil, body: Data("vision-geometry-moge2-small".utf8)),
            .init(name: "resolution_level", filename: nil, contentType: nil, body: Data("5".utf8)),
            .init(name: "token_count", filename: nil, contentType: nil, body: Data("1800".utf8)),
            .init(name: "max_points", filename: nil, contentType: nil, body: Data("250000".utf8)),
        ])
        let plan = try APIServerContract.geometryPlan(from: form)
        XCTAssertEqual(plan.modelID, ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue)
        XCTAssertEqual(plan.resolutionLevel, 5)
        XCTAssertEqual(plan.tokenCount, 1_800)
        XCTAssertEqual(plan.maximumPointCount, 250_000)

        let pathForm = MultipartFormData(parts: [
            .init(name: "model", filename: nil, contentType: nil, body: Data("/tmp/model.onnx".utf8)),
        ])
        XCTAssertThrowsError(try APIServerContract.geometryPlan(from: pathForm))
        let badResolution = MultipartFormData(parts: [
            .init(name: "resolution_level", filename: nil, contentType: nil, body: Data("10".utf8)),
        ])
        XCTAssertThrowsError(try APIServerContract.geometryPlan(from: badResolution))
        let unsafeTokenCount = MultipartFormData(parts: [
            .init(
                name: "token_count",
                filename: nil,
                contentType: nil,
                body: Data(String(MoGe2InferenceConfiguration.maximumTokenCount + 1).utf8)
            ),
        ])
        XCTAssertThrowsError(try APIServerContract.geometryPlan(from: unsafeTokenCount)) { error in
            XCTAssertTrue(error.localizedDescription.contains("token_count"))
        }
    }

    func testGeometryResponseReturnsHashedServerOwnedArtifactURLs() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-geometry-response-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let inputURL = root.appendingPathComponent("input.png")
        try Data("image-input".utf8).write(to: inputURL)
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 1,
            normalizedFY: 1
        )
        let frame = try DenseGeometryFrame(
            width: 2,
            height: 2,
            units: .meters,
            intrinsics: intrinsics,
            depth: [1, 2, 3, 4],
            points: [
                -0.25, -0.25, 1, 0.5, -0.5, 2,
                -0.75, 0.75, 3, 1, 1, 4,
            ],
            normals: [Float](repeating: 0, count: 12),
            validity: [1, 1, 1, 1],
            confidence: [1, 1, 1, 1]
        )
        let export = try GeometryArtifactExporter.export(
            frame: frame,
            inputURL: inputURL,
            outputDirectory: root,
            provenance: GeometryModelProvenance(
                modelID: ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
                upstreamRepository: "Ruicheng/moge-2-vits-normal-onnx",
                upstreamRevision: "pin",
                license: "MIT"
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let result = MoGe2RunResult(
            export: export,
            focalShift: MoGe2FocalShiftSolution(
                focal: 1.5,
                shift: 0.1,
                iterationCount: 3,
                residualMeanSquare: 0
            ),
            metricScale: 2,
            tokenCount: 1_200,
            modelLoadSeconds: 0.1,
            inferenceSeconds: 0.2,
            postprocessSeconds: 0.3
        )
        let response = try APIServerContract.geometryResponse(
            from: result,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.created, 123)
        XCTAssertEqual(response.object, "vision.geometry")
        XCTAssertEqual(response.units, .meters)
        XCTAssertEqual(response.camera.intrinsics, intrinsics)
        XCTAssertEqual(response.artifacts.count, export.manifest.artifacts.count + 1)
        let manifest = try XCTUnwrap(response.artifacts.first { $0.kind == .manifest })
        XCTAssertTrue(manifest.url.hasPrefix("file://"))
        XCTAssertEqual(manifest.sha256.count, 64)
        XCTAssertGreaterThan(manifest.byteCount, 0)
    }

    func testMultiViewGeometryContractAcceptsManagedDA3ControlsAndInlineCameras() throws {
        let camera = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 2,
                imageHeight: 2,
                normalizedFX: 1,
                normalizedFY: 1
            ),
            extrinsics: .identity
        )
        let cameraJSON = try JSONEncoder().encode(
            DepthAnything3CameraDocument(cameras: [camera, camera])
        )
        let form = MultipartFormData(parts: [
            .init(name: "model", filename: nil, contentType: nil, body: Data("vision-geometry-da3-small".utf8)),
            .init(name: "process_resolution", filename: nil, contentType: nil, body: Data("392".utf8)),
            .init(name: "reference_view", filename: nil, contentType: nil, body: Data("first".utf8)),
            .init(name: "confidence_percentile", filename: nil, contentType: nil, body: Data("55.5".utf8)),
            .init(name: "max_points", filename: nil, contentType: nil, body: Data("250000".utf8)),
            .init(name: "cameras", filename: nil, contentType: nil, body: cameraJSON),
            .init(name: "image[]", filename: "a.png", contentType: "image/png", body: Data([1])),
            .init(name: "image", filename: "b.jpg", contentType: "image/jpeg", body: Data([2])),
        ])

        let plan = try APIServerContract.multiViewGeometryPlan(from: form)

        XCTAssertEqual(plan.modelID, ModelResolver.ModelID.visionGeometryDA3Small.rawValue)
        XCTAssertEqual(plan.processResolution, 392)
        XCTAssertEqual(plan.referenceViewStrategy, .first)
        XCTAssertEqual(plan.confidencePercentile, 55.5)
        XCTAssertEqual(plan.maximumPointCount, 250_000)
        XCTAssertEqual(plan.knownCameras, [camera, camera])
        XCTAssertTrue(plan.poseConditioned)
    }

    func testMultiViewGeometryContractAcceptsUploadedCameraDocumentAndDefaults() throws {
        let camera = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 4,
                imageHeight: 3,
                normalizedFX: 0.8,
                normalizedFY: 0.9
            ),
            extrinsics: .identity
        )
        let cameraJSON = try JSONEncoder().encode(
            DepthAnything3CameraDocument(cameras: [camera])
        )
        let plan = try APIServerContract.multiViewGeometryPlan(from: MultipartFormData(parts: [
            .init(name: "image", filename: "view.png", contentType: "image/png", body: Data([1])),
            .init(
                name: "cameras",
                filename: "cameras.json",
                contentType: "application/json; charset=utf-8",
                body: cameraJSON
            ),
        ]))

        XCTAssertEqual(plan.modelID, ModelResolver.ModelID.visionGeometryDA3Small.rawValue)
        XCTAssertEqual(plan.processResolution, 504)
        XCTAssertEqual(plan.referenceViewStrategy, .saddleBalanced)
        XCTAssertEqual(plan.confidencePercentile, 40)
        XCTAssertEqual(plan.maximumPointCount, 1_000_000)
        XCTAssertEqual(plan.knownCameras, [camera])
    }

    func testMultiViewGeometryRoutePolicyRequiresUploadedImagesAndRejectsClientPaths() throws {
        XCTAssertEqual(
            APIServerContract.multiViewGeometryRoutePath,
            "/v1/vision/geometry/multiview"
        )
        XCTAssertEqual(
            APIServerContract.multiViewGeometryRouterPath.description,
            APIServerContract.multiViewGeometryRoutePath
        )
        XCTAssertEqual(
            APIServerContract.maximumMultiViewGeometryUploadByteCount,
            512 * 1024 * 1024
        )
        let upload = MultipartFormData.Part(
            name: "image[]",
            filename: "view.png",
            contentType: "image/png",
            body: Data([1])
        )

        for pathField in [
            "input", "input_path", "image_path", "output", "output_path",
            "output_directory", "model_path", "camera_path", "cameras_path",
        ] {
            XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
                from: MultipartFormData(parts: [
                    .init(name: pathField, filename: nil, contentType: nil, body: Data("/tmp/client-path".utf8)),
                    upload,
                ])
            )) { error in
                XCTAssertTrue(error.localizedDescription.contains(pathField))
            }
        }

        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "model", filename: nil, contentType: nil, body: Data("/tmp/model.safetensors".utf8)),
                upload,
            ])
        ))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [])
        ))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "image", filename: nil, contentType: nil, body: Data("/tmp/view.png".utf8)),
            ])
        ))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "image", filename: "empty.png", contentType: "image/png", body: Data()),
            ])
        ))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "image", filename: "view.txt", contentType: "text/plain", body: Data([1])),
            ])
        ))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                upload,
                .init(name: "checkpoint", filename: "model.safetensors", contentType: "application/octet-stream", body: Data([1])),
            ])
        ))
    }

    func testMultiViewGeometryContractRejectsInvalidControlsAndCameraDocuments() throws {
        let upload = MultipartFormData.Part(
            name: "image",
            filename: "view.png",
            contentType: "image/png",
            body: Data([1])
        )
        for (field, value) in [
            ("process_resolution", "0"),
            ("process_resolution", "13"),
            ("process_resolution", "1009"),
            ("max_points", "0"),
            ("confidence_percentile", "-1"),
            ("confidence_percentile", "101"),
            ("confidence_percentile", "nan"),
            ("reference_view", "last"),
        ] {
            XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
                from: MultipartFormData(parts: [
                    .init(name: field, filename: nil, contentType: nil, body: Data(value.utf8)),
                    upload,
                ])
            ), "field \(field) should reject \(value)")
        }

        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "cameras", filename: nil, contentType: nil, body: Data("/tmp/cameras.json".utf8)),
                upload,
            ])
        ))

        let zeroCameras = try JSONEncoder().encode(DepthAnything3CameraDocument(cameras: []))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "cameras", filename: nil, contentType: nil, body: zeroCameras),
                upload,
            ])
        ))

        let camera = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 2,
                imageHeight: 2,
                normalizedFX: 1,
                normalizedFY: 1
            ),
            extrinsics: .identity
        )
        let validCameraJSON = try JSONEncoder().encode(DepthAnything3CameraDocument(cameras: [camera]))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "cameras", filename: nil, contentType: nil, body: validCameraJSON),
                .init(name: "cameras", filename: "cameras.json", contentType: "application/json", body: validCameraJSON),
                upload,
            ])
        ))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "cameras", filename: "cameras.txt", contentType: "text/plain", body: validCameraJSON),
                upload,
            ])
        ))

        let reflected = DepthAnything3KnownCamera(
            intrinsics: camera.intrinsics,
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, -1],
                translation: [0, 0, 0]
            )
        )
        let reflectedJSON = try JSONEncoder().encode(
            DepthAnything3CameraDocument(cameras: [reflected])
        )
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: [
                .init(name: "cameras", filename: nil, contentType: nil, body: reflectedJSON),
                upload,
            ])
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("determinant +1"))
        }
    }

    func testMultiViewGeometryContractRejectsViewAndProcessedPixelBudgets() throws {
        func uploads(_ count: Int) -> [MultipartFormData.Part] {
            (0..<count).map { index in
                .init(
                    name: "image[]",
                    filename: "view-\(index).png",
                    contentType: "image/png",
                    body: Data([1])
                )
            }
        }

        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: uploads(17) + [
                .init(name: "process_resolution", filename: nil, contentType: nil, body: Data("14".utf8)),
            ])
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("between 1 and 16"))
        }
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(
            from: MultipartFormData(parts: uploads(9))
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("activation budget"))
        }
    }

    func testMultiViewGeometryResponseReturnsPinsCamerasAndEveryHashedArtifact() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-multiview-geometry-response-\(UUID().uuidString)", isDirectory: true)
        let sourceRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-multiview-geometry-inputs-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        defer { try? FileManager.default.removeItem(at: sourceRoot) }
        try FileManager.default.createDirectory(at: sourceRoot, withIntermediateDirectories: true)
        let image = try MediaImage(
            width: 2,
            height: 2,
            rgba8: [UInt8](repeating: 255, count: 16)
        )
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 1,
            normalizedFY: 1
        )
        let secondExtrinsics = try GeometryCameraExtrinsics(
            rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            translation: [-1, 0, 0]
        )
        let sourceURLs = [
            sourceRoot.appendingPathComponent("view-0.png"),
            sourceRoot.appendingPathComponent("view-1.png"),
        ]
        try Data("view zero".utf8).write(to: sourceURLs[0])
        try Data("view one".utf8).write(to: sourceURLs[1])
        let viewResults = [
            DepthAnything3ViewResult(
                index: 0,
                sourceURL: sourceURLs[0],
                inputIdentity: try DepthAnything3InputIdentity.capture(sourceURLs[0]),
                sourceImage: image,
                processedImage: image,
                preprocessingPlan: preprocessingPlan(index: 0),
                depth: [1, 2, 3, 4],
                confidence: [1, 2, 3, 4],
                intrinsics: intrinsics,
                extrinsics: .identity,
                predictedIntrinsics: intrinsics,
                predictedExtrinsics: .identity,
                suppliedCamera: nil
            ),
            DepthAnything3ViewResult(
                index: 1,
                sourceURL: sourceURLs[1],
                inputIdentity: try DepthAnything3InputIdentity.capture(sourceURLs[1]),
                sourceImage: image,
                processedImage: image,
                preprocessingPlan: preprocessingPlan(index: 1),
                depth: [2, 3, 4, 5],
                confidence: [2, 3, 4, 5],
                intrinsics: intrinsics,
                extrinsics: secondExtrinsics,
                predictedIntrinsics: intrinsics,
                predictedExtrinsics: secondExtrinsics,
                suppliedCamera: nil
            ),
        ]
        let checkpoint = DepthAnything3Checkpoint(
            modelID: ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
            repository: "depth-anything/DA3-SMALL",
            revision: String(repeating: "a", count: 40),
            sourceRepository: "ByteDance-Seed/Depth-Anything-3",
            sourceRevision: String(repeating: "b", count: 40),
            license: "Apache-2.0",
            rootURL: root,
            weightsURL: root.appendingPathComponent("model.safetensors"),
            configurationURL: root.appendingPathComponent("config.json"),
            weightsByteCount: 137_248_940,
            weightsSHA256: String(repeating: "c", count: 64),
            configurationByteCount: 1_202,
            configurationSHA256: String(repeating: "d", count: 64)
        )
        let run = DepthAnything3RunResult(
            views: viewResults,
            checkpoint: checkpoint,
            referenceViewStrategy: .saddleBalanced,
            cameraSemantics: .predictedRelative,
            cameraScaleAlignment: "predicted-relative",
            depthScaleDivisor: 1,
            processResolution: 504,
            checkpointVerificationSeconds: 0.1,
            decodingSeconds: 0.2,
            preprocessingSeconds: 0.3,
            modelLoadSeconds: 0.4,
            inferenceSeconds: 0.5,
            postprocessingSeconds: 0.6
        )
        let export = try MultiViewGeometryExporter.export(
            run: run,
            outputDirectory: root,
            configuration: try MultiViewGeometryExportConfiguration(
                confidencePercentile: 0,
                maximumPointCount: 100
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )

        let response = try APIServerContract.multiViewGeometryResponse(
            from: run,
            export: export,
            exportSeconds: 0.7,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.created, 123)
        XCTAssertEqual(response.object, "vision.geometry.multiview")
        XCTAssertEqual(response.model, ModelResolver.ModelID.visionGeometryDA3Small.rawValue)
        XCTAssertEqual(response.checkpoint.weightsSHA256, checkpoint.weightsSHA256)
        XCTAssertEqual(response.checkpoint.configurationSHA256, checkpoint.configurationSHA256)
        XCTAssertEqual(response.units, .relative)
        XCTAssertEqual(response.coordinateSystem, .worldFromCameras)
        XCTAssertEqual(response.cameraSemantics, .predictedRelative)
        XCTAssertEqual(response.referenceViewStrategy, .saddleBalanced)
        XCTAssertEqual(response.viewCount, 2)
        XCTAssertEqual(response.cameraCount, 2)
        XCTAssertEqual(response.cameras.map(\.viewIndex), [0, 1])
        XCTAssertEqual(response.pointCount, 8)
        XCTAssertEqual(response.pointCloudRepresentation, "colored-points-not-mesh")
        XCTAssertFalse(response.containsMesh)
        XCTAssertFalse(response.containsGaussianParameters)
        XCTAssertFalse(response.threeDGaussianHandoff.containsGaussianParameters)
        XCTAssertTrue(response.manifest.url.hasPrefix("file://"))
        XCTAssertEqual(response.manifest.sha256.count, 64)
        XCTAssertGreaterThan(response.manifest.byteCount, 0)
        XCTAssertEqual(response.artifacts.count, export.manifest.artifacts.count)
        XCTAssertTrue(response.artifacts.allSatisfy {
            $0.url.hasPrefix("file://") && $0.sha256.count == 64 && $0.byteCount > 0
        })
        XCTAssertEqual(response.timing.totalSeconds, 2.8, accuracy: 0.000_001)

        let encoded = try JSONEncoder().encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["contains_mesh"] as? Bool, false)
        XCTAssertEqual(object["contains_gaussian_parameters"] as? Bool, false)
        XCTAssertEqual(object["point_cloud_representation"] as? String, "colored-points-not-mesh")
        XCTAssertNotNil(object["checkpoint"])
        XCTAssertNotNil(object["manifest"])
        XCTAssertNotNil(object["artifacts"])
        XCTAssertNotNil(object["timing"])
    }

    func testImageTo3DContractAcceptsOnlyManagedModelAndValidControls() throws {
        let upload = MultipartFormData.Part(
            name: "image",
            filename: "chair.png",
            contentType: "image/png",
            body: Data([1, 2, 3])
        )
        let plan = try APIServerContract.imageTo3DPlan(from: MultipartFormData(parts: [
            .init(
                name: "model",
                filename: nil,
                contentType: nil,
                body: Data(ModelResolver.ModelID.image3DTripoSR.rawValue.utf8)
            ),
            .init(name: "resolution", filename: nil, contentType: nil, body: Data("192".utf8)),
            .init(name: "density_threshold", filename: nil, contentType: nil, body: Data("21.5".utf8)),
            .init(name: "foreground_ratio", filename: nil, contentType: nil, body: Data("0.9".utf8)),
            .init(name: "already_framed", filename: nil, contentType: nil, body: Data("true".utf8)),
            .init(name: "vertex_colors", filename: nil, contentType: nil, body: Data("0".utf8)),
            upload,
        ]))

        XCTAssertEqual(plan.modelID, ModelResolver.ModelID.image3DTripoSR.rawValue)
        XCTAssertEqual(plan.extractionResolution, 192)
        XCTAssertEqual(plan.densityThreshold, 21.5)
        XCTAssertEqual(plan.foregroundRatio, 0.9)
        XCTAssertTrue(plan.alreadyFramed)
        XCTAssertEqual(plan.foregroundPolicy, .alreadyFramed)
        XCTAssertFalse(plan.includesVertexColors)

        let defaults = try APIServerContract.imageTo3DPlan(
            from: MultipartFormData(parts: [upload])
        )
        XCTAssertEqual(defaults.extractionResolution, 256)
        XCTAssertEqual(defaults.densityThreshold, 25)
        XCTAssertEqual(defaults.foregroundRatio, 0.85)
        XCTAssertFalse(defaults.alreadyFramed)
        XCTAssertTrue(defaults.includesVertexColors)
    }

    func testImageTo3DRouteRequiresUploadedBytesAndRejectsClientPaths() {
        XCTAssertEqual(APIServerContract.imageTo3DRoutePath, "/v1/vision/image-to-3d")
        XCTAssertEqual(
            APIServerContract.imageTo3DRouterPath.description,
            APIServerContract.imageTo3DRoutePath
        )
        XCTAssertEqual(APIServerContract.maximumImageTo3DUploadByteCount, 100 * 1024 * 1024)
        let upload = MultipartFormData.Part(
            name: "image",
            filename: "chair.png",
            contentType: "image/png",
            body: Data([1])
        )

        for pathField in [
            "input", "input_path", "image_path", "output", "output_path",
            "output_directory", "model_path", "checkpoint", "checkpoint_path",
        ] {
            XCTAssertThrowsError(try APIServerContract.imageTo3DPlan(
                from: MultipartFormData(parts: [
                    .init(name: pathField, filename: nil, contentType: nil, body: Data("/tmp/client-path".utf8)),
                    upload,
                ])
            )) { error in
                XCTAssertTrue(error.localizedDescription.contains(pathField))
            }
        }

        let invalidForms = [
            MultipartFormData(parts: []),
            MultipartFormData(parts: [
                .init(name: "image", filename: nil, contentType: nil, body: Data("/tmp/chair.png".utf8)),
            ]),
            MultipartFormData(parts: [
                .init(name: "image", filename: "empty.png", contentType: "image/png", body: Data()),
            ]),
            MultipartFormData(parts: [
                .init(name: "image", filename: "chair.txt", contentType: "text/plain", body: Data([1])),
            ]),
            MultipartFormData(parts: [upload, upload]),
            MultipartFormData(parts: [
                .init(name: "model", filename: nil, contentType: nil, body: Data("/tmp/model.ckpt".utf8)),
                upload,
            ]),
            MultipartFormData(parts: [
                .init(name: "checkpoint", filename: "model.ckpt", contentType: "application/octet-stream", body: Data([1])),
                upload,
            ]),
        ]
        for form in invalidForms {
            XCTAssertThrowsError(try APIServerContract.imageTo3DPlan(from: form))
        }

        for (field, value) in [
            ("resolution", "1"),
            ("resolution", "513"),
            ("density_threshold", "nan"),
            ("foreground_ratio", "0"),
            ("foreground_ratio", "1.1"),
            ("already_framed", "sometimes"),
            ("vertex_colors", "yes"),
        ] {
            XCTAssertThrowsError(try APIServerContract.imageTo3DPlan(
                from: MultipartFormData(parts: [
                    .init(name: field, filename: nil, contentType: nil, body: Data(value.utf8)),
                    upload,
                ])
            ), "field \(field) should reject \(value)")
        }
    }

    func testImageTo3DResponseReturnsPinnedIdentityAndEveryHashedMeshArtifact() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-image-to-3d-response-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let inputURL = root.appendingPathComponent("chair.png")
        try Data("temporary multipart upload".utf8).write(to: inputURL)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let checkpoint = TripoSRCheckpoint(
            modelID: ModelResolver.ModelID.image3DTripoSR.rawValue,
            repository: "stabilityai/TripoSR",
            revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            sourceRepository: "VAST-AI-Research/TripoSR",
            sourceRevision: "107cefdc244c39106fa830359024f6a2f1c78871",
            license: "MIT",
            format: .pinnedPyTorch,
            rootURL: root,
            weightsURL: root.appendingPathComponent("model.ckpt"),
            configurationURL: root.appendingPathComponent("config.yaml"),
            weightsByteCount: 1_677_246_742,
            weightsSHA256: "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
            sourceSHA256: "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
            configurationSHA256: "74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62"
        )
        let export = try TripoSRAssetExporter.export(
            mesh: mesh,
            inputURL: inputURL,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "chair",
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let runManifest = try TripoSRRunManifestExporter.export(
            meshExport: export,
            checkpoint: checkpoint,
            inputURL: inputURL,
            sourceWidth: 640,
            sourceHeight: 480,
            preparedWidth: 512,
            preparedHeight: 512,
            foregroundPolicy: "automatic-transparent-alpha",
            foregroundRatio: 0.85,
            croppedTransparentForeground: true,
            extractionResolution: 256,
            densityThreshold: 25,
            includesVertexColors: true
        )
        let run = TripoSRRunResult(
            export: export,
            runManifest: runManifest,
            checkpoint: checkpoint,
            sourceWidth: 640,
            sourceHeight: 480,
            preparedWidth: 512,
            preparedHeight: 512,
            foregroundPolicy: "automatic-transparent-alpha",
            foregroundRatio: 0.85,
            croppedTransparentForeground: true,
            extractionResolution: 256,
            densityThreshold: 25,
            includesVertexColors: true,
            checkpointVerificationSeconds: 0.1,
            decodingSeconds: 0.2,
            preprocessingSeconds: 0.3,
            modelLoadSeconds: 0.4,
            sceneEncodingSeconds: 0.5,
            meshExtractionSeconds: 0.6,
            exportSeconds: 0.7
        )

        let response = try APIServerContract.imageTo3DResponse(
            from: run,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.created, 123)
        XCTAssertEqual(response.object, "vision.image-to-3d")
        XCTAssertEqual(response.model, ModelResolver.ModelID.image3DTripoSR.rawValue)
        XCTAssertEqual(response.checkpoint.weightsSHA256, checkpoint.weightsSHA256)
        XCTAssertEqual(response.units, .normalizedObjectSpace)
        XCTAssertEqual(response.coordinateSystem, .modelXRightYUpZForward)
        XCTAssertTrue(response.inferredUnseenGeometry)
        XCTAssertEqual(response.vertexCount, 3)
        XCTAssertEqual(response.triangleCount, 1)
        XCTAssertTrue(response.manifest.url.hasPrefix("file://"))
        XCTAssertEqual(response.manifest.sha256.count, 64)
        XCTAssertGreaterThan(response.manifest.byteCount, 0)
        XCTAssertTrue(response.meshManifest.url.hasPrefix("file://"))
        XCTAssertEqual(response.meshManifest.sha256.count, 64)
        XCTAssertEqual(response.artifacts.count, 4)
        XCTAssertEqual(
            Set(response.artifacts.map(\.kind)),
            ["obj", "ply", "glb", "mesh-manifest"]
        )
        XCTAssertTrue(response.artifacts.allSatisfy {
            $0.url.hasPrefix("file://") && $0.sha256.count == 64 && $0.byteCount > 0
        })
        XCTAssertEqual(response.foregroundPolicy, "automatic-transparent-alpha")
        XCTAssertEqual(response.foregroundRatio, 0.85)
        XCTAssertEqual(response.timing.totalSeconds, 2.8, accuracy: 0.000_001)

        let encoded = try JSONEncoder().encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["inferred_unseen_geometry"] as? Bool, true)
        XCTAssertEqual(object["mesh_extraction_algorithm"] as? String, "native-marching-tetrahedra")
        XCTAssertNotNil(object["checkpoint"])
        XCTAssertNotNil(object["manifest"])
        XCTAssertNotNil(object["mesh_manifest"])
        XCTAssertNotNil(object["artifacts"])
        XCTAssertNotNil(object["timing"])
    }

    func testDepthVideoContractAcceptsOnlyManagedModelsAndPositiveControls() throws {
        let upload = MultipartFormData.Part(
            name: "video",
            filename: "shot.mp4",
            contentType: "video/mp4",
            body: Data([0, 1, 2, 3])
        )
        let relative = try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [upload]))
        XCTAssertEqual(relative.modelID, ModelResolver.ModelID.visionDepthVDASmall.rawValue)
        XCTAssertEqual(relative.inputSize, 518)
        XCTAssertEqual(relative.maximumFrameCount, VideoDepthAnythingLimits.defaultMaximumFrameCount)

        let metric = try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [
            .init(
                name: "model",
                filename: nil,
                contentType: nil,
                body: Data(ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue.utf8)
            ),
            .init(name: "input_size", filename: nil, contentType: nil, body: Data("756".utf8)),
            .init(name: "max_frames", filename: nil, contentType: nil, body: Data("240".utf8)),
            upload,
        ]))
        XCTAssertEqual(metric.modelID, ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue)
        XCTAssertEqual(metric.inputSize, 756)
        XCTAssertEqual(metric.maximumFrameCount, 240)

        for (field, value) in [("input_size", "0"), ("max_frames", "-1"), ("max_frames", "nope")] {
            let form = MultipartFormData(parts: [
                .init(name: field, filename: nil, contentType: nil, body: Data(value.utf8)),
                upload,
            ])
            XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: form)) { error in
                XCTAssertTrue(error.localizedDescription.contains(field))
            }
        }

        for (field, value) in [
            ("input_size", String(VideoDepthAnythingLimits.maximumInputSize + 1)),
            ("max_frames", String(VideoDepthAnythingLimits.maximumFrameCount + 1)),
        ] {
            let form = MultipartFormData(parts: [
                .init(name: field, filename: nil, contentType: nil, body: Data(value.utf8)),
                upload,
            ])
            XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: form)) { error in
                XCTAssertTrue(error.localizedDescription.contains(field))
            }
        }

        let unmanaged = MultipartFormData(parts: [
            .init(name: "model", filename: nil, contentType: nil, body: Data("/tmp/model.pth".utf8)),
            upload,
        ])
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: unmanaged)) { error in
            XCTAssertTrue(error.localizedDescription.contains("model"))
        }
    }

    func testDepthVideoRoutePolicyRequiresUploadedBytesAndRejectsFilesystemControls() {
        XCTAssertEqual(APIServerContract.depthVideoRoutePath, "/v1/vision/depth-video")
        XCTAssertEqual(
            APIServerContract.depthVideoRouterPath.description,
            APIServerContract.depthVideoRoutePath
        )
        XCTAssertEqual(APIServerContract.maximumDepthVideoUploadByteCount, 512 * 1024 * 1024)

        let upload = MultipartFormData.Part(
            name: "video",
            filename: "shot.mov",
            contentType: "video/quicktime",
            body: Data([1])
        )
        let pathInsteadOfUpload = MultipartFormData(parts: [
            .init(name: "video", filename: nil, contentType: nil, body: Data("/tmp/shot.mov".utf8)),
        ])
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: pathInsteadOfUpload))

        for pathField in ["input", "input_path", "video_path", "output", "output_path", "output_directory", "model_path"] {
            let form = MultipartFormData(parts: [
                .init(name: pathField, filename: nil, contentType: nil, body: Data("/tmp/client-path".utf8)),
                upload,
            ])
            XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: form)) { error in
                XCTAssertTrue(error.localizedDescription.contains(pathField))
            }
        }

        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [])))
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [
            .init(name: "video", filename: "empty.mp4", contentType: "video/mp4", body: Data()),
        ])))
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [
            upload,
            upload,
        ])))
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [
            .init(name: "video", filename: "frame.png", contentType: "image/png", body: Data([1])),
        ])))
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: MultipartFormData(parts: [
            upload,
            .init(name: "checkpoint", filename: "model.pth", contentType: "application/octet-stream", body: Data([1])),
        ])))
    }

    func testDepthVideoResponseReturnsHashedServerOwnedSequenceArtifactsAndExplicitAbsences() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-depth-video-response-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let inputURL = root.appendingPathComponent("upload.mp4")
        try Data("video-input".utf8).write(to: inputURL)
        let frames = try [
            DepthSequenceFrame(
                index: 0,
                timeSeconds: 0,
                width: 2,
                height: 2,
                depth: [1, 2, 3, 4]
            ),
            DepthSequenceFrame(
                index: 1,
                timeSeconds: 0.125,
                width: 2,
                height: 2,
                depth: [2, 3, 4, 5]
            ),
        ]
        let export = try DepthSequenceArtifactExporter.export(
            frames: frames,
            inputURL: inputURL,
            outputDirectory: root,
            fps: 8,
            semantics: .affineRelative,
            provenance: GeometryModelProvenance(
                modelID: ModelResolver.ModelID.visionDepthVDASmall.rawValue,
                upstreamRepository: "depth-anything/Video-Depth-Anything-Small",
                upstreamRevision: "pin",
                license: "Apache-2.0",
                weightsSHA256: String(repeating: "a", count: 64)
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let reviewURL = root.appendingPathComponent("depth-review.mp4")
        try Data("review".utf8).write(to: reviewURL)
        let review = VideoDepthReviewArtifact(
            relativePath: reviewURL.lastPathComponent,
            byteCount: 6,
            sha256: try ModelArtifactPin.fileSHA256(reviewURL)
        )
        let checkpoint = VideoDepthAnythingCheckpoint(
            variant: .relative,
            format: .pinnedPyTorch,
            weightsURL: root.appendingPathComponent("model.pth"),
            weightsByteCount: 123,
            weightsSHA256: String(repeating: "b", count: 64),
            sourceSHA256: String(repeating: "b", count: 64)
        )
        let result = VideoDepthAnythingRunResult(
            export: export,
            reviewVideo: review,
            checkpoint: checkpoint,
            sourceFPS: 8,
            windowCount: 1,
            checkpointVerificationSeconds: 0.1,
            frameExtractionSeconds: 0.2,
            modelLoadSeconds: 0.3,
            inferenceSeconds: 0.4,
            exportSeconds: 0.5
        )

        let response = try APIServerContract.depthVideoResponse(
            from: result,
            createdAt: Date(timeIntervalSince1970: 123)
        )
        XCTAssertEqual(response.created, 123)
        XCTAssertEqual(response.object, "vision.depth-video")
        XCTAssertEqual(response.model, ModelResolver.ModelID.visionDepthVDASmall.rawValue)
        XCTAssertEqual(response.semantics, .affineRelative)
        XCTAssertEqual(response.frameCount, 2)
        XCTAssertEqual(response.windowCount, 1)
        XCTAssertFalse(response.hasConfidence)
        XCTAssertFalse(response.hasCameraIntrinsics)
        XCTAssertFalse(response.hasCameraExtrinsics)
        XCTAssertFalse(response.hasPointCloud)
        XCTAssertTrue(response.manifest.url.hasPrefix("file://"))
        XCTAssertEqual(response.manifest.sha256.count, 64)
        XCTAssertGreaterThan(response.manifest.byteCount, 0)
        XCTAssertEqual(response.review.url, reviewURL.absoluteString)
        XCTAssertEqual(response.review.sha256, review.sha256)
        XCTAssertEqual(response.artifacts.count, 4)
        XCTAssertTrue(response.artifacts.allSatisfy { artifact in
            artifact.url.hasPrefix("file://")
                && artifact.sha256.count == 64
                && artifact.byteCount > 0
                && artifact.frameIndex != nil
        })
        XCTAssertEqual(response.timing.totalSeconds, 1.5, accuracy: 0.000_001)

        let encoded = try JSONEncoder().encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["has_confidence"] as? Bool, false)
        XCTAssertEqual(object["has_camera_intrinsics"] as? Bool, false)
        XCTAssertEqual(object["has_camera_extrinsics"] as? Bool, false)
        XCTAssertEqual(object["has_point_cloud"] as? Bool, false)
        XCTAssertNotNil(object["manifest"])
        XCTAssertNotNil(object["review"])
        XCTAssertNotNil(object["artifacts"])
        XCTAssertNotNil(object["timing"])
    }

    func testCompanionModelIDsHideMissingSidecarModels() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: [])

        XCTAssertFalse(ids.contains("image-zimage-nano"))
        XCTAssertFalse(ids.contains("speech-tts-qwen3-nano"))
        XCTAssertFalse(ids.contains("speech-asr-parakeet"))
        XCTAssertFalse(ids.contains("text-embed-qwen3-0.6b"))
    }

    func testImageGenerationContractMapsOpenAINamesAndValidatesOptions() throws {
        let request = OpenAIImageGenerationRequest(
            prompt: "  workstation in morning light  ",
            model: "dall-e-3",
            n: 1,
            size: "512x768",
            response_format: "url",
            seed: 123,
            negative_prompt: "blur",
            steps: 4,
            guidance_scale: 1.2
        )

        let plan = try APIServerContract.imageGenerationPlan(from: request)

        XCTAssertEqual(plan.modelID, "image-zimage-nano")
        XCTAssertEqual(plan.prompt, "workstation in morning light")
        XCTAssertEqual(plan.width, 512)
        XCTAssertEqual(plan.height, 768)
        XCTAssertEqual(plan.responseFormat, "url")
        XCTAssertEqual(plan.seed, 123)
        XCTAssertEqual(plan.negativePrompt, "blur")
        XCTAssertEqual(plan.steps, 4)
        XCTAssertEqual(plan.guidanceScale, 1.2)
        XCTAssertNil(plan.inputImage)
        XCTAssertNil(plan.strength)

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", n: 2)
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("n"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "large")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("size"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(
                    prompt: "hello",
                    steps: APIServerContract.maxImageInferenceSteps + 1
                )
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("between 1 and 100"))
        }
    }

    func testVideoGenerationContractPreservesNativeLTXOptionsAndArtifactProof() throws {
        let request = OpenAIVideoGenerationRequest(
            prompt: "waves break under moonlight",
            model: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue,
            size: "1024x768",
            num_frames: 97,
            fps: 24,
            seed: 17,
            quality: "final",
            output_mode: "audio-video",
            options: [
                "--ltx-preset", "hq",
                "--num-generated-keyframes", "3",
                "--enhance-prompt",
            ]
        )
        let plan = try APIServerContract.videoGenerationPlan(from: request)

        XCTAssertEqual(APIServerContract.videoGenerationRoutePath, "/v1/videos/generations")
        XCTAssertEqual(plan.width, 1_024)
        XCTAssertEqual(plan.height, 768)
        XCTAssertEqual(plan.numFrames, 97)
        XCTAssertTrue(plan.commandArguments.contains("--num-generated-keyframes"))

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "mere-run-video-api-contract-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("output.mp4")
        try Data("video".utf8).write(to: output)
        let response = try APIServerContract.videoGenerationResponse(
            outputURL: output,
            plan: plan,
            createdAt: Date(timeIntervalSince1970: 12)
        )
        XCTAssertEqual(response.created, 12)
        XCTAssertEqual(response.model, ModelResolver.ModelID.ltxVideo25FullBF16.rawValue)
        XCTAssertEqual(response.artifact.byte_count, 5)
        XCTAssertEqual(response.artifact.url, output.absoluteString)
        XCTAssertEqual(response.artifact.sha256, try ModelArtifactPin.fileSHA256(output))

        XCTAssertThrowsError(try APIServerContract.videoGenerationPlan(from:
            OpenAIVideoGenerationRequest(
                prompt: "unsafe",
                options: ["--output", "/tmp/escape.mp4"]
            )
        ))
        XCTAssertThrowsError(try APIServerContract.videoGenerationPlan(from:
            OpenAIVideoGenerationRequest(
                prompt: "EXR only",
                options: ["--skip-mp4"]
            )
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("retains and hashes an MP4"))
        }
    }

    func testSidecarJSONDecodingMapsMalformedPayloadsToValidationErrors() {
        XCTAssertThrowsError(
            try APIServerContract.decodeImageGenerationRequest(
                from: Data(#"{"prompt":42}"#.utf8)
            )
        ) { error in
            XCTAssertEqual(error as? APIRequestValidationError, .invalidPayload)
            XCTAssertEqual(error.localizedDescription, "Invalid request payload.")
        }

        XCTAssertThrowsError(
            try APIServerContract.decodeSpeechRequest(
                from: Data(#"{"input":["not","text"]}"#.utf8)
            )
        ) { error in
            XCTAssertEqual(error as? APIRequestValidationError, .invalidPayload)
            XCTAssertEqual(error.localizedDescription, "Invalid request payload.")
        }
    }

    func testImageGenerationContractBoundsDimensionsAndPixelAreaWithoutOverflow() throws {
        let boundary = try APIServerContract.imageGenerationPlan(
            from: OpenAIImageGenerationRequest(prompt: "hello", size: "4096x1024")
        )
        XCTAssertEqual(boundary.width, 4_096)
        XCTAssertEqual(boundary.height, 1_024)

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "4097x1")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("at most 4096"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "4096x1025")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("total image area"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "\(Int.max)x2")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("at most 4096"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "1x1")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("at least 16"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "1000x1000")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("divisible by 16"))
        }
    }

    func testImageGenerationResponseCanReturnBase64OrFileURL() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-api-test-\(UUID().uuidString).png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        let plan = APIServerContract.ImageGenerationPlan(
            modelID: "image-zimage-nano",
            prompt: "hello",
            width: 1024,
            height: 1024,
            responseFormat: "b64_json",
            seed: nil,
            negativePrompt: nil,
            steps: nil,
            guidanceScale: nil,
            inputImage: nil,
            strength: nil
        )

        let b64 = try APIServerContract.imageResponse(
            outputURL: tempURL,
            plan: plan,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(b64.created, 123)
        XCTAssertEqual(b64.data.first?.b64_json, Data([0x89, 0x50, 0x4E, 0x47]).base64EncodedString())
        XCTAssertEqual(b64.data.first?.revised_prompt, "hello")

        let urlPlan = APIServerContract.ImageGenerationPlan(
            modelID: plan.modelID,
            prompt: plan.prompt,
            width: plan.width,
            height: plan.height,
            responseFormat: "url",
            seed: nil,
            negativePrompt: nil,
            steps: nil,
            guidanceScale: nil,
            inputImage: nil,
            strength: nil
        )
        let urlResponse = try APIServerContract.imageResponse(outputURL: tempURL, plan: urlPlan)
        XCTAssertEqual(urlResponse.data.first?.url, tempURL.absoluteString)
    }

    func testImageEditContractReadsMultipartFields() throws {
        let inputURL = URL(fileURLWithPath: "/tmp/input.png")
        let secondURL = URL(fileURLWithPath: "/tmp/second.png")
        let maskURL = URL(fileURLWithPath: "/tmp/mask.png")
        let form = MultipartFormData(parts: [
            MultipartFormData.Part(name: "model", filename: nil, contentType: nil, body: Data("gpt-image-1".utf8)),
            MultipartFormData.Part(name: "prompt", filename: nil, contentType: nil, body: Data("  add warm light  ".utf8)),
            MultipartFormData.Part(name: "size", filename: nil, contentType: nil, body: Data("768x512".utf8)),
            MultipartFormData.Part(name: "response_format", filename: nil, contentType: nil, body: Data("url".utf8)),
            MultipartFormData.Part(name: "seed", filename: nil, contentType: nil, body: Data("42".utf8)),
            MultipartFormData.Part(name: "strength", filename: nil, contentType: nil, body: Data("0.4".utf8)),
        ])

        let plan = try APIServerContract.imageEditPlan(
            from: form,
            inputImageURLs: [inputURL, secondURL],
            maskImageURL: maskURL
        )

        XCTAssertEqual(plan.modelID, "image-zimage-nano")
        XCTAssertEqual(plan.prompt, "add warm light")
        XCTAssertEqual(plan.width, 768)
        XCTAssertEqual(plan.height, 512)
        XCTAssertEqual(plan.responseFormat, "url")
        XCTAssertEqual(plan.seed, 42)
        XCTAssertEqual(plan.inputImage, inputURL)
        XCTAssertEqual(plan.additionalInputImages, [secondURL])
        XCTAssertEqual(plan.maskImage, maskURL)
        XCTAssertEqual(plan.strength, 0.4)
    }

    func testSpeechContractMapsOpenAINamesAndValidatesOutputFormats() throws {
        let request = OpenAIAudioSpeechRequest(
            model: "tts-1",
            input: "  hello from mere.run  ",
            voice: "onyx",
            response_format: "mp3",
            speed: 1.25,
            instructions: "Use a friendly pace.",
            temperature: 0.7
        )

        let plan = try APIServerContract.speechPlan(from: request)

        XCTAssertEqual(plan.modelID, "speech-tts-qwen3-nano")
        XCTAssertEqual(plan.input, "hello from mere.run")
        XCTAssertTrue(plan.voiceDescription.contains("deep"))
        XCTAssertTrue(plan.voiceDescription.contains("friendly pace"))
        XCTAssertEqual(plan.responseFormat, "mp3")
        XCTAssertEqual(APIServerContract.speechContentType(for: plan.responseFormat), "audio/mpeg")
        XCTAssertEqual(plan.speed, 1.25)
        XCTAssertEqual(plan.temperature, 0.7)

        XCTAssertThrowsError(
            try APIServerContract.speechPlan(
                from: OpenAIAudioSpeechRequest(input: "hello", response_format: "caf")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("response_format"))
        }


        XCTAssertThrowsError(
            try APIServerContract.speechPlan(
                from: OpenAIAudioSpeechRequest(
                    input: "hello",
                    instructions: String(
                        repeating: "x",
                        count: APIServerContract.maxSpeechPromptUTF8Bytes
                    )
                )
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("UTF-8 bytes"))
        }
    }

    func testMultipartFormDataParserExtractsFieldsAndFile() throws {
        let boundary = "mere-boundary"
        let body = [
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"model\"",
            "",
            "whisper-1",
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"",
            "Content-Type: audio/wav",
            "",
            "WAVDATA",
            "--mere-boundary--",
            "",
        ].joined(separator: "\r\n").data(using: .utf8)!

        let form = try MultipartFormData.parse(body: body, boundary: boundary)

        XCTAssertEqual(form.field("model"), "whisper-1")
        XCTAssertEqual(form.file(named: "file")?.filename, "speech.wav")
        XCTAssertEqual(form.files(named: "file").map(\.filename), ["speech.wav"])
        XCTAssertEqual(form.file(named: "file")?.contentType, "audio/wav")
        XCTAssertEqual(form.file(named: "file")?.body, Data("WAVDATA".utf8))
    }

    func testMultipartFormDataParserExtractsRepeatedFiles() throws {
        let boundary = "mere-boundary"
        let body = [
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"image\"; filename=\"first.png\"",
            "Content-Type: image/png",
            "",
            "FIRST",
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"image[]\"; filename=\"second.png\"",
            "Content-Type: image/png",
            "",
            "SECOND",
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"",
            "Content-Type: image/png",
            "",
            "MASK",
            "--mere-boundary--",
            "",
        ].joined(separator: "\r\n").data(using: .utf8)!

        let form = try MultipartFormData.parse(body: body, boundary: boundary)

        XCTAssertEqual(form.files(named: "image").map(\.filename), ["first.png"])
        XCTAssertEqual(form.files(named: "image[]").map(\.filename), ["second.png"])
        XCTAssertEqual(form.file(named: "mask")?.body, Data("MASK".utf8))
    }

    func testTranscriptionContractMapsOpenAINamesAndVerboseResponse() throws {
        let form = MultipartFormData(parts: [
            MultipartFormData.Part(name: "model", filename: nil, contentType: nil, body: Data("whisper-1".utf8)),
            MultipartFormData.Part(name: "language", filename: nil, contentType: nil, body: Data("en".utf8)),
            MultipartFormData.Part(name: "response_format", filename: nil, contentType: nil, body: Data("verbose_json".utf8)),
            MultipartFormData.Part(name: "task", filename: nil, contentType: nil, body: Data("transcribe".utf8)),
        ])

        let plan = try APIServerContract.transcriptionPlan(from: form)

        XCTAssertEqual(plan.modelID, "speech-asr-parakeet")
        XCTAssertEqual(plan.language, "en")
        XCTAssertEqual(plan.responseFormat, "verbose_json")
        XCTAssertEqual(plan.task, .transcribe)
        XCTAssertEqual(plan.maxTokens, 448)

        let oversizedMaxTokens = MultipartFormData(parts: [
            MultipartFormData.Part(
                name: "max_tokens",
                filename: nil,
                contentType: nil,
                body: Data(String(APIServerContract.maxTranscriptionTokens + 1).utf8)
            ),
        ])
        XCTAssertThrowsError(
            try APIServerContract.transcriptionPlan(from: oversizedMaxTokens)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("between 1 and 4096"))
        }

        let response = APIServerContract.transcriptionResponse(
            from: ASRResult(
                text: "hello",
                language: "en",
                duration: 1.5,
                sentenceAlignments: [
                    ASRSentenceAlignment(
                        text: "hello",
                        startSeconds: 0,
                        durationSeconds: 1.5,
                        tokens: []
                    ),
                ]
            ),
            verbose: true
        )

        XCTAssertEqual(response.text, "hello")
        XCTAssertEqual(response.language, "en")
        XCTAssertEqual(response.duration, 1.5)
        XCTAssertEqual(response.segments?.first?.text, "hello")
    }

    func testTranscriptionContractCanRenderSubtitleFormats() throws {
        let srtForm = MultipartFormData(parts: [
            MultipartFormData.Part(name: "response_format", filename: nil, contentType: nil, body: Data("srt".utf8)),
        ])
        let plan = try APIServerContract.transcriptionPlan(from: srtForm)
        XCTAssertEqual(plan.responseFormat, "srt")

        let result = ASRResult(
            text: "hello",
            language: "en",
            duration: 1.5,
            sentenceAlignments: [
                ASRSentenceAlignment(text: "hello", startSeconds: 0, durationSeconds: 1.5, tokens: []),
            ]
        )
        let srt = APIServerContract.transcriptionSubtitle(from: result, format: "srt")
        XCTAssertTrue(srt.contains("1\n00:00:00,000 --> 00:00:01,500\nhello"))

        let vtt = APIServerContract.transcriptionSubtitle(from: result, format: "vtt")
        XCTAssertTrue(vtt.hasPrefix("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello"))
    }

    func testChatRequestValidationAcceptsBoundedDefaults() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: " /tmp/default.safetensors ",
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.maxTokens, APIServerContract.defaultMaxTokens)
        XCTAssertEqual(chatRequest.maxContextTokens, 4_096)
        XCTAssertEqual(chatRequest.temperature, 1.0)
        XCTAssertEqual(chatRequest.topP, 0.95)
        XCTAssertEqual(chatRequest.minP, 0)
        XCTAssertFalse(chatRequest.showThinking)
        XCTAssertEqual(chatRequest.maxContextTokens, 4_096)
        XCTAssertEqual(chatRequest.messages, [ChatMessage(role: .user, content: "hello")])
        guard case .local(let path, let scale) = chatRequest.lora else {
            return XCTFail("Expected server-configured LoRA to be applied")
        }
        XCTAssertEqual(path, "/tmp/default.safetensors")
        XCTAssertEqual(scale, 1.0)
    }

    func testChatRequestDecodesStructuredOpenAIContentParts() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "hello" },
                { "type": "input_text", "text": "setup mere.run" }
              ]
            },
            {
              "role": "assistant"
            }
          ]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.messages[0].content, "hello\nsetup mere.run")
        XCTAssertEqual(chatRequest.messages[1].content, "")
    }

    func testChatRequestDecodesModernOpenAIFieldsAndUnknownFields() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            { "role": "developer", "content": "Follow repo rules." },
            { "role": "user", "content": "hello" }
          ],
          "max_completion_tokens": 77,
          "stream_options": { "include_usage": true },
          "parallel_tool_calls": true,
          "metadata": { "client": "test" },
          "reasoning_effort": "low",
          "min_p": 0.05,
          "x-client-extra": { "kept": true }
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        XCTAssertEqual(request.max_completion_tokens, 77)
        XCTAssertEqual(request.stream_options?.include_usage, true)
        XCTAssertEqual(request.parallel_tool_calls, true)
        XCTAssertEqual(request.metadata?["client"], "test")
        XCTAssertEqual(request.reasoning_effort, "low")
        XCTAssertEqual(request.min_p, 0.05)
        XCTAssertEqual(request.unknownFields["x-client-extra"]?.objectValue?["kept"]?.boolValue, true)
    }

    func testChatRequestMapsDeveloperRoleAndMaxCompletionTokens() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [
                OpenAIChatMessage(role: "developer", content: "Follow repo rules."),
                OpenAIChatMessage(role: "user", content: "hello"),
            ],
            max_completion_tokens: 77
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.maxTokens, 77)
        XCTAssertEqual(chatRequest.messages[0].role, .system)
        XCTAssertEqual(chatRequest.messages[0].content, "Follow repo rules.")
    }

    func testChatRequestPreservesReasoningAndToolCorrelationFields() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [
                OpenAIChatMessage(
                    role: "assistant",
                    content: "Working...",
                    reasoning_content: "I should write the file.",
                    tool_calls: [
                        OpenAIChatToolCall(
                            id: "call_123",
                            function: OpenAIChatToolCallFunction(
                                name: "write_file",
                                arguments: #"{"path":"note.txt","overwrite":false}"#
                            )
                        ),
                    ]
                ),
                OpenAIChatMessage(
                    role: "tool",
                    content: "Wrote 4 bytes.",
                    name: "write_file",
                    tool_call_id: "call_123"
                ),
            ]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textChatGemma4.openAICompatibility
        )

        XCTAssertEqual(chatRequest.messages[0].reasoningContent, "I should write the file.")
        XCTAssertEqual(
            chatRequest.messages[0].toolCalls,
            [
                ChatMessageToolCall(
                    id: "call_123",
                    name: "write_file",
                    arguments: [
                        "path": .string("note.txt"),
                        "overwrite": .bool(false),
                    ]
                ),
            ]
        )
        XCTAssertEqual(chatRequest.messages[1].name, "write_file")
        XCTAssertEqual(chatRequest.messages[1].toolCallID, "call_123")
    }

    func testChatRequestRejectsInvalidToolCallArguments() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [
                OpenAIChatMessage(
                    role: "assistant",
                    tool_calls: [
                        OpenAIChatToolCall(
                            id: "call_123",
                            function: OpenAIChatToolCallFunction(
                                name: "write_file",
                                arguments: "not-json"
                            )
                        ),
                    ]
                ),
            ]
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: RuntimeServingEngine.textChatGemma4.openAICompatibility
            )
        ) { error in
            XCTAssertEqual(
                error as? APIRequestValidationError,
                .invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a JSON object"
                )
            )
        }
    }

    func testChatRequestMapsSupportedStopSequences() throws {
        let single = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .string("END")
        )
        let singleRequest = try APIServerContract.chatRequest(
            from: single,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textCode.openAICompatibility
        )
        XCTAssertEqual(singleRequest.stopSequences, ["END"])

        let multiple = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .array(["END", "", "\nif __name__"])
        )
        let multipleRequest = try APIServerContract.chatRequest(
            from: multiple,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textCode.openAICompatibility
        )
        XCTAssertEqual(multipleRequest.stopSequences, ["END", "\nif __name__"])
    }

    func testChatRequestValidatesImageContentPartsAgainstEngineCapability() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "describe" },
                { "type": "image_url", "image_url": { "url": "file:///tmp/image.png" } }
              ]
            }
          ]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("image content"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithToolsAndVision
        )
        XCTAssertEqual(chatRequest.messages[0].content, "describe")
        XCTAssertEqual(chatRequest.messages[0].imageUrl, "file:///tmp/image.png")
    }

    func testChatRequestMapsSupportedOpenAITools() throws {
        let tool = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "lookup",
                description: "Look up a value.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("Search query"),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        )
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [tool],
            tool_choice: .mode("auto")
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("tools"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithTools
        )
        XCTAssertEqual(chatRequest.tools?.first?.name, "lookup")
        XCTAssertEqual(chatRequest.tools?.first?.parameters["query"]?.description, "Search query")
        XCTAssertEqual(chatRequest.tools?.first?.required, ["query"])
    }

    func testOpenAIToolArgumentsPreserveJSONTypesFromModelPayloads() throws {
        let json = APIServerContract.openAIToolArgumentsJSON(
            [
                "audience": "local-AI filmmakers",
                "stringBoolean": "false",
                "references": "[\"paper craft\",\"stop motion\"]",
                "generateScore": "false",
                "takes": "2",
                "metadata": "{\"color\":\"red\"}",
            ],
            parameterTypes: [
                "audience": "string",
                "stringBoolean": "string",
                "references": "array",
                "generateScore": "boolean",
                "takes": "integer",
                "metadata": "object",
            ]
        )
        let data = try XCTUnwrap(json.data(using: .utf8))
        let decoded = try JSONDecoder().decode([String: OpenAIJSONValue].self, from: data)

        XCTAssertEqual(decoded["audience"], .string("local-AI filmmakers"))
        XCTAssertEqual(decoded["stringBoolean"], .string("false"))
        XCTAssertEqual(
            decoded["references"],
            .array([.string("paper craft"), .string("stop motion")])
        )
        XCTAssertEqual(decoded["generateScore"], .bool(false))
        XCTAssertEqual(decoded["takes"], .number(2))
        XCTAssertEqual(decoded["metadata"], .object(["color": .string("red")]))
    }

    func testChatRequestAcceptsSpecificFunctionToolChoice() throws {
        let lookup = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "lookup",
                description: "Look up a value.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")]),
                    ]),
                ])
            )
        )
        let summarize = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "summarize",
                description: "Summarize text.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "text": .object(["type": .string("string")]),
                    ]),
                ])
            )
        )
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [lookup, summarize],
            tool_choice: .function(name: "summarize")
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithTools
        )

        XCTAssertEqual(chatRequest.tools?.map(\.name), ["summarize"])

        let missing = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [lookup],
            tool_choice: .function(name: "missing")
        )
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: missing,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localTextWithTools
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("tool_choice"))
        }
    }

    func testChatRequestValidatesStructuredOutputsAgainstEngineCapability() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("response_format"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithStructuredJSON
        )
        XCTAssertTrue(chatRequest.requiresJSON)
    }

    func testGemma4EngineCapabilityAcceptsJSONMode() throws {
        let request = OpenAIChatRequest(
            model: "text-chat-gemma4-12b-4bit",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textChatGemma4.openAICompatibility
        )
        XCTAssertTrue(chatRequest.requiresJSON)
    }

    func testQ36EngineCapabilityAcceptsJSONObjectButNotStrictSchema() throws {
        let capabilities = RuntimeServingEngine.textChatQ36.openAICompatibility
        XCTAssertTrue(capabilities.supportsTools)
        XCTAssertTrue(capabilities.supportsVisionContentParts)
        XCTAssertTrue(capabilities.supportsStructuredOutputs)
        XCTAssertFalse(capabilities.supportsStrictMode)

        let objectRequest = OpenAIChatRequest(
            model: Q35Resources.q36NanoModelId,
            messages: [OpenAIChatMessage(role: "user", content: "Return an object")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )
        let chatRequest = try APIServerContract.chatRequest(
            from: objectRequest,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: capabilities,
            servedModelID: Q35Resources.q36NanoModelId
        )
        XCTAssertTrue(chatRequest.requiresJSON)
        XCTAssertFalse(chatRequest.showThinking)

        let schemaRequest = OpenAIChatRequest(
            model: Q35Resources.q36NanoModelId,
            messages: [OpenAIChatMessage(role: "user", content: "Return an object")],
            response_format: OpenAIResponseFormat(type: "json_schema")
        )
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: schemaRequest,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: capabilities,
                servedModelID: Q35Resources.q36NanoModelId
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("strict JSON schema"))
        }
    }

    func testJSONObjectModeForcesThinkingOffForNativeQ35Family() throws {
        let request = OpenAIChatRequest(
            model: Q35Resources.ornith35BMLXModelId,
            messages: [OpenAIChatMessage(role: "user", content: "Return an object")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )
        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textChatQ35.openAICompatibility,
            servedModelID: Q35Resources.ornith35BMLXModelId
        )

        XCTAssertTrue(chatRequest.requiresJSON)
        XCTAssertFalse(chatRequest.showThinking)
    }

    func testOpenAIFinishReasonReportsLengthForTokenBudgetExhaustion() {
        let result = ChatResponse(
            response: "{\"partial\":true",
            tokensGenerated: 8,
            finishReason: .length
        )

        XCTAssertEqual(CodeGenServer.openAIFinishReason(for: result), "length")
    }

    func testChatRequestRejectsUnsupportedHighImpactFields() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .string("END"),
            seed: 42,
            presence_penalty: 0.5,
            logprobs: true,
            reasoning_effort: "low",
            think: true
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            let message = error.localizedDescription
            XCTAssertTrue(
                message.contains("stop")
                    || message.contains("seed")
                    || message.contains("presence_penalty")
                    || message.contains("logprobs")
                    || message.contains("reasoning_effort")
                    || message.contains("thinking")
            )
        }
    }

    func testMuseReasoningEffortMapsToNativeStrength() throws {
        let request = OpenAIChatRequest(
            model: MuseGlimmerResources.modelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            reasoning_effort: "high"
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textChatMuseGlimmer.openAICompatibility,
            servedModelID: MuseGlimmerResources.modelId
        )

        XCTAssertEqual(chatRequest.reasoningEffort, 0.8)
    }

    func testQ38ReasoningAliasesMapToNativeXHighStrength() throws {
        let profile = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: Q35Resources.q38TwentySevenB4BitModelId)
        )
        let request = OpenAIChatRequest(
            model: Q35Resources.q38TwentySevenB4BitModelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            reasoning_effort: "high"
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .catalog(profile),
            servedModelID: Q35Resources.q38TwentySevenB4BitModelId
        )

        XCTAssertEqual(chatRequest.reasoningEffort, 1)
        XCTAssertTrue(chatRequest.showThinking)
    }

    func testStreamingUsageOptionHonorsCapabilities() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stream_options: OpenAIStreamOptions(include_usage: true)
        )

        XCTAssertTrue(
            try APIServerContract.includeUsageInStreaming(
                request,
                capabilities: .localText
            )
        )

        XCTAssertThrowsError(
            try APIServerContract.includeUsageInStreaming(
                request,
                capabilities: APIEngineCapabilities(supportsUsageInStreaming: false)
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("stream_options.include_usage"))
        }
    }

    func testChatRequestValidationRejectsOversizedMaxTokens() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            max_tokens: 4_097
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("max_tokens"))
        }
    }

    func testChatRequestValidationRejectsPerRequestLora() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            lora: "/tmp/request-controlled.safetensors"
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("lora"))
        }
    }

    func testChatRequestValidationRejectsInvalidSamplingValues() {
        let badTemperature = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            temperature: 3.0
        )
        let badTopP = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            top_p: 1.5
        )
        let badMinP = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            min_p: -0.01
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badTemperature, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("temperature"))
        }
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badTopP, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("top_p"))
        }
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badMinP, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("min_p"))
        }
    }

    func testChatContentTypeValidationRejectsBrowserSimpleTypes() {
        XCTAssertFalse(APIServerContract.acceptsJSONContentType(nil))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType(""))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("text/plain"))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("application/x-www-form-urlencoded"))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("multipart/form-data; boundary=abc"))
    }

    func testChatContentTypeValidationAcceptsJSONTypes() {
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/json"))
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/json; charset=utf-8"))
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/vnd.openai+json"))
    }

    func testStreamingStatusMessagesAreNotTreatedAsTokens() {
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Generating..."))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Generating response"))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Retrying generation"))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("DS4 chat completion"))
        XCTAssertFalse(APIServerContract.isStreamingStatusMessage("Actual generated text"))
    }

    func testOpenAIUsageReportsPromptAndCompletionTokenCounts() {
        let result = ChatResponse(
            response: "hello",
            tokensGenerated: 128,
            promptTokens: 1039
        )

        let usage = CodeGenServer.openAIUsage(for: result)

        XCTAssertEqual(usage.prompt_tokens, 1039)
        XCTAssertEqual(usage.completion_tokens, 128)
        XCTAssertEqual(usage.total_tokens, 1167)
    }

    func testOpenAIUsageFallsBackToZeroPromptTokensWhenUnreported() {
        let result = ChatResponse(response: "hello", tokensGenerated: 7)

        let usage = CodeGenServer.openAIUsage(for: result)

        XCTAssertEqual(usage.prompt_tokens, 0)
        XCTAssertEqual(usage.completion_tokens, 7)
        XCTAssertEqual(usage.total_tokens, 7)
    }

    func testChatRequestDefaultsToThinkingForOrnithLanes() throws {
        let request = OpenAIChatRequest(
            model: Q35Resources.ornith35BMLXModelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            servedModelID: Q35Resources.ornith35BMLXModelId
        )

        XCTAssertTrue(chatRequest.showThinking)
        XCTAssertEqual(chatRequest.topK, 20)
        XCTAssertEqual(chatRequest.temperature, 1.0)
        XCTAssertEqual(chatRequest.topP, 0.95)
    }

    func testChatRequestUsesValidatedLagunaSamplingDefaults() throws {
        let request = OpenAIChatRequest(
            model: LagunaResources.modelID,
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: LagunaResources.defaultContextLength,
            capabilities: RuntimeServingEngine.textChatLaguna.openAICompatibility,
            servedModelID: LagunaResources.modelID
        )

        XCTAssertEqual(chatRequest.temperature, LagunaResources.recommendedTemperature)
        XCTAssertEqual(chatRequest.topP, LagunaResources.recommendedTopP)
        XCTAssertEqual(chatRequest.topK, LagunaResources.recommendedTopK)
        XCTAssertEqual(chatRequest.minP, LagunaResources.recommendedMinP)
        XCTAssertFalse(chatRequest.requiresJSON)
    }

    func testChatRequestExplicitSamplingSkipsRecommendedTopK() throws {
        var request = OpenAIChatRequest(
            model: Q35Resources.ornith35BMLXModelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )
        request.temperature = 0.2

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            servedModelID: Q35Resources.ornith35BMLXModelId
        )

        XCTAssertTrue(chatRequest.showThinking)
        XCTAssertNil(chatRequest.topK)
        XCTAssertEqual(chatRequest.temperature, 0.2)
    }

    func testChatRequestAcceptsMinPAsExplicitSampling() throws {
        let request = OpenAIChatRequest(
            model: Q35Resources.ornith35BMLXModelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            min_p: 0.05
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            servedModelID: Q35Resources.ornith35BMLXModelId
        )

        XCTAssertEqual(chatRequest.minP, 0.05)
        XCTAssertNil(chatRequest.topK)
    }

    func testChatRequestKeepsNoThinkDefaultForNonOrnithLanes() throws {
        let request = OpenAIChatRequest(
            model: Q35Resources.q36NanoModelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            servedModelID: Q35Resources.q36NanoModelId
        )

        XCTAssertFalse(chatRequest.showThinking)
        XCTAssertNil(chatRequest.topK)
    }

    func testChatRequestUsesPublishedBonsaiReasoningAndSamplingDefaults() throws {
        let request = OpenAIChatRequest(
            model: Q35Resources.bonsai27B1BitModelId,
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: Q35Resources.bonsai27B1BitContextLength,
            servedModelID: Q35Resources.bonsai27B1BitModelId
        )

        XCTAssertTrue(chatRequest.showThinking)
        XCTAssertEqual(chatRequest.temperature, 0.7)
        XCTAssertEqual(chatRequest.topP, 0.95)
        XCTAssertEqual(chatRequest.topK, 20)
        XCTAssertEqual(chatRequest.maxContextTokens, 262_144)
    }

    private func preprocessingPlan(index: Int) -> DepthAnything3PreprocessingPlan {
        DepthAnything3PreprocessingPlan(
            sourceWidth: 2,
            sourceHeight: 2,
            processResolution: 504,
            boundaryWidth: 504,
            boundaryHeight: 504,
            divisibleWidth: 504,
            divisibleHeight: 504,
            batchCropLeft: index,
            batchCropTop: 0,
            processedWidth: 2,
            processedHeight: 2
        )
    }
}
