import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

final class ChatExecutionIntegrationTests: XCTestCase {
    func testCLIAndAPIResolveEquivalentExplicitRequests() throws {
        for id in [Q35Resources.ornith35BMLX4BitModelId, LagunaResources.modelID, Gemma4Resources.defaultModelId] {
            let thinking = Q35Resources.thinkingDefault(forModelId: id)
                || ManagedModelCatalog.apiProfile(for: id)?.thinkingLevels == [.high]
            let command = try TextChat.parse([
                "--prompt", "hello", "--model", id, "--max-tokens", "19", "--context-size", "128",
                "--temperature", "0", "--top-p", "1", "--top-k", "0", "--min-p", "0.1", thinking ? "--thinking" : "--no-thinking"
            ])
            let cli = try command.resolvedChatRequest(modelID: id, messages: [.init(role: .user, content: "hello")])
            var wire = OpenAIChatRequest(model: id, messages: [.init(role: "user", content: "hello")])
            wire.max_tokens = 19
            wire.temperature = 0
            wire.top_p = 1
            wire.top_k = 0
            wire.min_p = 0.1
            wire.parallel_tool_calls = false
            let api = try APIServerContract.chatRequest(
                from: wire, fallbackLoraPath: nil, contextSize: 128,
                capabilities: .catalog(try XCTUnwrap(ManagedModelCatalog.apiProfile(for: id))), servedModelID: id
            )
            XCTAssertEqual(cli, api, id)
        }
    }

    func testCLIExecutionAndPreflightRejectTheSameInvalidOptions() throws {
        for arguments in [["--max-tokens", "0"], ["--temperature", "nan"], ["--top-k=-1"], ["--context-size", "0"]] {
            let command = try TextChat.parse(["--prompt", "hello"] + arguments)
            XCTAssertThrowsError(try command.resolvedChatRequest(
                modelID: command.model, messages: [.init(role: .user, content: "hello")]
            ))
            let report = command.makePreflightReport(modelID: command.model, installedModelPath: nil)
            XCTAssertTrue(report.diagnostics.contains { $0.id == "text_chat_request_invalid" && $0.severity == .blocker })
        }
    }

    func testCommandClampsMaxTokensToASmallerContextInsteadOfRejecting() throws {
        let id = Q35Resources.ornith35BMLX4BitModelId
        // The default --max-tokens (2048) must not turn a smaller --context-size into a
        // blocker; the command capped generation against the context before this moved.
        let defaulted = try TextChat.parse(["--prompt", "hello", "--model", id, "--context-size", "1024"])
        let defaultedRequest = try defaulted.resolvedChatRequest(
            modelID: id, messages: [.init(role: .user, content: "hello")]
        )
        XCTAssertEqual(defaultedRequest.maxTokens, 1024)
        XCTAssertEqual(defaultedRequest.maxContextTokens, 1024)
        let report = defaulted.makePreflightReport(modelID: id, installedModelPath: nil)
        XCTAssertFalse(report.diagnostics.contains { $0.id == "text_chat_request_invalid" })

        let explicit = try TextChat.parse([
            "--prompt", "hello", "--model", id, "--max-tokens", "4096", "--context-size", "512"
        ])
        let explicitRequest = try explicit.resolvedChatRequest(
            modelID: id, messages: [.init(role: .user, content: "hello")]
        )
        XCTAssertEqual(explicitRequest.maxTokens, 512)

        // Clamping must not swallow the values the command still rejects.
        for arguments in [["--max-tokens", "0"], ["--context-size", "0"]] {
            let invalid = try TextChat.parse(["--prompt", "hello", "--model", id] + arguments)
            XCTAssertThrowsError(try invalid.resolvedChatRequest(
                modelID: id, messages: [.init(role: .user, content: "hello")]
            ), arguments.joined(separator: " "))
        }
    }

    func testSessionRetainsLeasesThroughResponseConstructionAndCoalescesCleanup() async throws {
        let releaseStarted = expectation(description: "model release started")
        let releaseGate = ChatSessionTestGate()
        let model = ChatSessionTestModel(onRelease: {
            releaseStarted.fulfill()
            await releaseGate.wait()
        })
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let session = try await makeSession(model: model, admission: admission)
        let response = try await session.chat()
        XCTAssertEqual(response.response, "hello")
        let active = await admission.snapshot()
        XCTAssertEqual(active.activeRequests, 1)
        let first = Task { await session.finish() }
        let second = Task { await session.finish() }
        await fulfillment(of: [releaseStarted], timeout: 2)
        let releasing = await admission.snapshot()
        XCTAssertEqual(releasing.activeRequests, 1, "Admission remains held until model release completes")
        await releaseGate.open()
        await first.value
        await second.value
        let finished = await admission.snapshot()
        let releases = await model.releases
        XCTAssertEqual(releases, 1)
        XCTAssertEqual(finished.activeRequests, 0)
        XCTAssertEqual(finished.totalAdmittedRequests, 1)
        XCTAssertEqual(finished.totalCompletedRequests, 1)
    }

    func testServicePreservesAliasSettingsAndExplicitOverridesWithoutDoubleAdmission() async throws {
        let root = try temporaryDirectory()
        let store = RuntimeModelSettingsStore(modelsDir: root)
        let id = Q35Resources.ornith35BMLX4BitModelId
        try store.save(.init(models: [id: .init(
            alias: "fixture", maxContextTokens: 256, maxTokens: 29,
            temperature: 0.3, topP: 0.7, minP: 0.1, kvCacheMode: .affine8
        )]))
        let services = makeServices(root: root)
        var wire = OpenAIChatRequest(model: "fixture", messages: [.init(role: "user", content: "hello")])
        wire.temperature = 0
        let session = try await services.startChat(wire, fallbackLoraPath: nil, contextSize: 4_096)
        XCTAssertEqual(session.modelID, id)
        XCTAssertEqual(session.request.maxTokens, 29)
        XCTAssertEqual(session.request.maxContextTokens, 256)
        XCTAssertEqual(session.request.temperature, 0)
        XCTAssertEqual(session.request.topP, 0.7)
        XCTAssertEqual(session.request.minP, 0.1)
        XCTAssertEqual(session.request.kvCacheMode, .affine8)
        do {
            _ = try await services.models.unloadModel(idOrAlias: id)
            XCTFail("An active request allowed model eviction")
        } catch RuntimeModelPoolError.unloadConflict(_, let active) {
            XCTAssertEqual(active, 1)
        }
        await session.finish()
        let unloaded = try await services.models.unloadModel(idOrAlias: id)
        let admitted = await services.admission.snapshot()
        XCTAssertFalse(unloaded.loaded)
        XCTAssertEqual(admitted.totalAdmittedRequests, 1)
        XCTAssertEqual(admitted.activeRequests, 0)
    }

    func testCancellationDuringPreparationReleasesAdmissionAndResidency() async throws {
        let preparing = expectation(description: "model preparation started")
        let gate = ChatSessionTestGate()
        let services = makeServices(root: try temporaryDirectory(), onPrepare: {
            preparing.fulfill()
            await gate.wait()
        })
        let id = Q35Resources.ornith35BMLX4BitModelId
        let task = Task {
            try await services.startChat(
                .init(model: id, messages: [.init(role: "user", content: "hello")]),
                fallbackLoraPath: nil, contextSize: 4_096
            )
        }
        await fulfillment(of: [preparing], timeout: 2)
        task.cancel()
        await gate.open()
        do {
            let session = try await task.value
            await session.finish()
            XCTFail("A cancelled request returned a session")
        } catch is CancellationError {}
        let snapshot = await services.admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalCancelledRequests, 1)
        _ = try await services.models.unloadModel(idOrAlias: id)
    }

    func testResolutionAndPreparationFailureReleaseAdmission() async throws {
        let root = try temporaryDirectory()
        let services = makeServices(root: root, failPreparation: true)
        for model in ["missing-fixture-model", Q35Resources.ornith35BMLX4BitModelId] {
            do {
                _ = try await services.startChat(
                    .init(model: model, messages: [.init(role: "user", content: "hello")]),
                    fallbackLoraPath: nil, contextSize: 4_096
                )
                XCTFail("Invalid or unprepared model was admitted")
            } catch {}
            let snapshot = await services.admission.snapshot()
            XCTAssertEqual(snapshot.activeRequests, 0)
        }
    }

    func testProxyStreamsExactBytesAndReleasesAfterCompletionOrFailure() async throws {
        for fail in [false, true] {
            let model = ChatSessionTestModel()
            let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
            let session = try await makeSession(model: model, admission: admission)
            let (source, continuation) = AsyncThrowingStream<UInt8, Error>.makeStream()
            let bytes = Array("data: first\n\nlast".utf8)
            bytes.forEach { continuation.yield($0) }
            if fail { continuation.finish(throwing: ChatSessionTestError.expected) } else { continuation.finish() }
            var actual: [UInt8] = []
            for await buffer in RuntimeChatProxyStream.make(from: source, session: session) {
                actual.append(contentsOf: buffer.readableBytesView)
            }
            XCTAssertEqual(actual, fail ? Array("data: first\n\n".utf8) : bytes)
            let releases = await model.releases
            let snapshot = await admission.snapshot()
            XCTAssertEqual(releases, 1)
            XCTAssertEqual(snapshot.activeRequests, 0)
        }
    }

    func testProxyDisconnectCancelsUpstreamAndReleasesBothLeases() async throws {
        let received = expectation(description: "first bytes received")
        let upstreamCancelled = expectation(description: "upstream cancelled")
        let modelReleased = expectation(description: "model released")
        let model = ChatSessionTestModel(onRelease: { modelReleased.fulfill() })
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let session = try await makeSession(model: model, admission: admission)
        let (source, continuation) = AsyncThrowingStream<UInt8, Error>.makeStream()
        continuation.onTermination = { termination in
            if case .cancelled = termination { upstreamCancelled.fulfill() }
        }
        let stream = RuntimeChatProxyStream.make(from: source, session: session)
        let reader = Task {
            for await _ in stream { received.fulfill() }
        }
        for byte in "data: first\n".utf8 { continuation.yield(byte) }
        await fulfillment(of: [received], timeout: 2)
        reader.cancel()
        await reader.value
        await fulfillment(of: [upstreamCancelled, modelReleased], timeout: 2)
        await session.finish(cancelled: true)
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalCancelledRequests, 1)
    }

    private func makeSession(model: ChatSessionTestModel, admission: RuntimeRequestAdmission) async throws -> RuntimeChatSession {
        RuntimeChatSession(
            plan: .init(
                lease: model, request: .init(messages: [.init(role: .user, content: "hello")]),
                modelID: "fixture", engine: .textChatQ35, includeUsage: true
            ), admission: try await admission.acquire()
        )
    }

    private func makeServices(
        root: URL, failPreparation: Bool = false,
        onPrepare: @escaping @Sendable () async -> Void = {}
    ) -> RuntimeServingServices {
        let models = RuntimeModelPool(
            defaultModelID: Q35Resources.ornith35BMLX4BitModelId, defaultEngine: .textChatQ35,
            startupModelPath: root.path, settingsStore: RuntimeModelSettingsStore(modelsDir: root), ensureMLXAvailable: {},
            prepareLoadedModel: { _ in
                await onPrepare()
                if failPreparation { throw ChatSessionTestError.expected }
            },
            unloadLoadedModel: { _ in }, clearMLXCache: {}
        )
        let media = APISidecarModelPool(settingsURL: root.appendingPathComponent("media-settings.json"))
        return RuntimeServingServices(
            models: models, media: media, admission: RuntimeRequestAdmission(maxActiveRequests: 1),
            ensureAvailable: {}, transcriptionExecutor: media
        )
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        return root
    }
}

private enum ChatSessionTestError: Error { case expected }

private actor ChatSessionTestModel: RuntimeChatModelLease {
    private let onRelease: @Sendable () async -> Void
    var releases = 0

    init(onRelease: @escaping @Sendable () async -> Void = {}) { self.onRelease = onRelease }

    func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse {
        progressHandler?(.init(stage: .generating, message: "hello"))
        return ChatResponse(response: "hello", tokensGenerated: 1)
    }

    func deepseekChatCompletionsURL(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> URL {
        URL(string: "http://127.0.0.1:1/v1/chat/completions")!
    }

    func release() async {
        releases += 1
        await onRelease()
    }
}

private actor ChatSessionTestGate {
    private var isOpen = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiting.append($0) }
    }
    func open() {
        isOpen = true
        waiting.forEach { $0.resume() }
        waiting.removeAll()
    }
}
