import XCTest
import MereRunCore
@testable import MereRunCLI

final class StatusCommandTests: XCTestCase {
    func testMereRunCLIExposesStatusCommand() {
        let commandNames = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("status"))
    }

    func testStatusParsesDefaults() throws {
        let cmd = try Status.parse([])

        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.port, 8080)
        XCTAssertNil(cmd.apiKey)
        XCTAssertEqual(cmd.timeoutSeconds, 1.0)
        XCTAssertFalse(cmd.json)
    }

    func testStatusParsesOverrides() throws {
        let cmd = try Status.parse([
            "--host", "localhost",
            "--port", "11434",
            "--api-key", "secret",
            "--timeout-seconds", "2.5",
            "--json",
        ])

        XCTAssertEqual(cmd.host, "localhost")
        XCTAssertEqual(cmd.port, 11_434)
        XCTAssertEqual(cmd.apiKey, "secret")
        XCTAssertEqual(cmd.timeoutSeconds, 2.5)
        XCTAssertTrue(cmd.json)
    }

    func testFormatterShowsLoadedAndInstalledModels() {
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "up",
                detail: nil,
                loadedModels: ["text-chat-gemma4"],
                modelsDetail: nil,
                runtime: nil,
                runtimeDetail: nil
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/Users/test/Library/Application Support/MereRun/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 2,
            installedModels: [
                StatusInstalledModelSnapshot(
                    id: "text-chat-gemma4",
                    category: "text-chat",
                    size: "12 GB"
                ),
            ]
        )

        let output = StatusFormatter.text(snapshot)

        XCTAssertTrue(output.contains("server: up (http://127.0.0.1:8080)"))
        XCTAssertTrue(output.contains("loaded models: text-chat-gemma4"))
        XCTAssertTrue(output.contains("installed models: 1/2"))
        XCTAssertTrue(output.contains("text-chat-gemma4 (text-chat, 12 GB)"))
    }

    func testFormatterShowsUnavailableLoadedModels() {
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "up",
                detail: nil,
                loadedModels: [],
                modelsDetail: "requires API key",
                runtime: nil,
                runtimeDetail: nil
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/Users/test/Library/Application Support/MereRun/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 0,
            installedModels: []
        )

        let output = StatusFormatter.text(snapshot)

        XCTAssertTrue(output.contains("loaded models: unavailable (requires API key)"))
        XCTAssertTrue(output.contains("installed models: 0/0"))
        XCTAssertTrue(output.contains("    none"))
    }

    func testFormatterUsesRuntimeStatusWhenAvailable() {
        let runtime = RuntimeModelPoolStatus(
            object: "runtime.status",
            defaultModel: "text-chat-gemma4",
            settingsPath: "/tmp/models/.mere-run/runtime-model-settings.json",
            activeRequests: 2,
            admission: RuntimeRequestAdmissionSnapshot(
                maxActiveRequests: 1,
                activeRequests: 1,
                queuedRequests: 1,
                totalAdmittedRequests: 3,
                totalCompletedRequests: 2,
                totalCancelledRequests: 1
            ),
            capabilities: .current(
                gemma4PrefixKVCacheEnabled: true,
                gemma4ContinuousBatchingEnabled: true,
                q35ContinuousBatchingEnabled: false,
                q35PrefixKVCacheEnabled: false
            ),
            memory: RuntimeMemorySnapshot(
                physicalBytes: 1024 * 1024 * 1024,
                activeRequests: 2,
                activeModelCount: 1,
                pressure: "unknown"
            ),
            models: [
                RuntimeModelPoolEntrySnapshot(
                    id: "text-chat-gemma4",
                    category: "text-chat",
                    engine: .textChatGemma4,
                    installPath: "/tmp/models/text-chat-gemma4",
                    loaded: true,
                    activeRequests: 2,
                    lastAccess: nil,
                    lastError: nil,
                    pinned: true,
                    alias: "chat-default",
                    ttlSeconds: 600,
                    maxContextTokens: 8192,
                    maxTokens: 512,
                    temperature: 0.4,
                    topP: 0.8,
                    engineOverride: nil,
                    kvCacheMode: .auto,
                    prefixKVCache: Gemma4PrefixKVCacheStats(
                        enabled: true,
                        entries: 1,
                        maxEntries: 4,
                        hits: 2,
                        misses: 1,
                        storedPrefixes: 3,
                        reusedTokens: 256,
                        storedTokens: 512
                    ),
                    continuousBatching: Gemma4ContinuousBatchingStats(
                        enabled: true,
                        activeRows: 2,
                        queuedRows: 1,
                        batchedDecodeSteps: 4,
                        samePositionBatchedSteps: 3,
                        variablePositionBatchedSteps: 1,
                        singleDecodeSteps: 3,
                        totalBatchedRows: 8,
                        maxBatchSize: 2
                    ),
                    mtp: Gemma4MTPStats(
                        available: true,
                        enabled: true,
                        active: true,
                        assistantModelPath: "/tmp/models/text-chat-gemma4-12b-mtp",
                        reason: nil,
                        blockSize: 4,
                        threshold: 2048,
                        rounds: 3,
                        draftedTokens: 9,
                        acceptedTokens: 6,
                        rejectedTokens: 1
                    ),
                    benchmarkStats: RuntimeModelBenchmarkStats(
                        completedRequests: 2,
                        failedRequests: 1,
                        generatedTokens: 40,
                        totalLoadSeconds: 0.2,
                        totalPrefillSeconds: 0.6,
                        totalDecodeSeconds: 2.0,
                        lastCompletedAt: nil
                    )
                ),
            ],
            cacheStats: RuntimeCacheStatsSummary(
                prefixKVCaches: [
                    PrefixKVCacheStats(
                        enabled: true,
                        entries: 1,
                        maxEntries: 4,
                        hits: 2,
                        misses: 1,
                        storedPrefixes: 3,
                        reusedTokens: 256,
                        storedTokens: 512
                    ),
                ],
                decodeBatchers: [
                    RuntimeDecodeBatchingStats(
                        enabled: true,
                        activeRows: 2,
                        queuedRows: 1,
                        batchedDecodeSteps: 4,
                        samePositionBatchedSteps: 3,
                        variablePositionBatchedSteps: 1,
                        singleDecodeSteps: 3,
                        totalBatchedRows: 8,
                        maxBatchSize: 2
                    ),
                ]
            ),
            benchmarkStats: RuntimeBenchmarkStatsSummary(stats: [
                RuntimeModelBenchmarkStats(
                    completedRequests: 2,
                    failedRequests: 1,
                    generatedTokens: 40,
                    totalLoadSeconds: 0.2,
                    totalPrefillSeconds: 0.6,
                    totalDecodeSeconds: 2.0,
                    lastCompletedAt: nil
                ),
            ]),
            sidecars: RuntimeSidecarPoolStatus(
                defaultIdleTTLSeconds: 300,
                pressure: "nominal",
                loadedCount: 1,
                activeRequests: 0,
                queuedRequests: 0,
                residents: [
                    RuntimeSidecarResidentSnapshot(
                        kind: .image,
                        modelID: "image-zimage-nano",
                        modelPath: "/tmp/models/image-zimage-nano",
                        variant: "zImageTurbo",
                        loaded: true,
                        activeRequests: 0,
                        queuedRequests: 0,
                        loadedAt: Date(timeIntervalSince1970: 10),
                        lastAccess: Date(timeIntervalSince1970: 20),
                        lastEvictedAt: nil,
                        lastEvictionReason: nil,
                        pinned: false,
                        ttlSeconds: 45,
                        loadCount: 1,
                        replacementCount: 0,
                        evictionCount: 0,
                        completedRequests: 2,
                        failedRequests: 0
                    ),
                ]
            )
        )
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "up",
                detail: nil,
                loadedModels: [],
                modelsDetail: nil,
                runtime: runtime,
                runtimeDetail: nil
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/tmp/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 1,
            installedModels: []
        )

        let output = StatusFormatter.text(snapshot)

        XCTAssertTrue(output.contains("loaded models: image-zimage-nano, text-chat-gemma4"))
        XCTAssertTrue(output.contains("active requests: 2"))
        XCTAssertTrue(output.contains("request admission: 1/1 active, 1 queued"))
        XCTAssertTrue(output.contains("continuous batching: enabled"))
        XCTAssertTrue(output.contains("prefix KV reuse: enabled"))
        let aggregateCacheLine = "cache stats: prefix 1/4 entries, 2 hits, 256 reused tokens; "
            + "decode batching 4 batched steps, 1 variable-position, max batch 2"
        XCTAssertTrue(output.contains(aggregateCacheLine))
        XCTAssertTrue(output.contains("benchmark stats: 2 completed, 1 failed, 40 tokens"))
        XCTAssertTrue(output.contains("prefill avg 0.30s"))
        XCTAssertTrue(output.contains("decode avg 1.00s"))
        XCTAssertTrue(output.contains("decode 20.00 tok/s"))
        XCTAssertTrue(output.contains("text-chat-gemma4 prefix KV: 1/4 entries, 2 hits, 256 reused tokens"))
        XCTAssertTrue(output.contains(
            "text-chat-gemma4 batching: 4 batched steps, 3 same-position, 1 variable-position, max batch 2, 1 queued rows"
        ))
        XCTAssertTrue(output.contains("text-chat-gemma4 MTP: active, block 4, 6/9 accepted"))
        XCTAssertTrue(output.contains("sidecar residency: 1/1 loaded, 0 active, 0 queued, default TTL 300s"))
        XCTAssertTrue(output.contains("image: image-zimage-nano, loaded, 0 active, 0 queued, TTL 45s, 1 load(s)"))
        XCTAssertTrue(output.contains("runtime settings: /tmp/models/.mere-run/runtime-model-settings.json"))
    }

    func testRuntimeCacheStatsSummaryAggregatesLoadedModelCounters() {
        let summary = RuntimeCacheStatsSummary(
            prefixKVCaches: [
                PrefixKVCacheStats(
                    enabled: true,
                    entries: 2,
                    maxEntries: 4,
                    hits: 3,
                    misses: 1,
                    storedPrefixes: 4,
                    reusedTokens: 128,
                    storedTokens: 256
                ),
                PrefixKVCacheStats(
                    enabled: false,
                    entries: 0,
                    maxEntries: 4,
                    hits: 0,
                    misses: 2,
                    storedPrefixes: 0,
                    reusedTokens: 0,
                    storedTokens: 0
                ),
            ],
            decodeBatchers: [
                RuntimeDecodeBatchingStats(
                    enabled: true,
                    activeRows: 1,
                    queuedRows: 2,
                    batchedDecodeSteps: 5,
                    samePositionBatchedSteps: 4,
                    variablePositionBatchedSteps: 1,
                    singleDecodeSteps: 7,
                    totalBatchedRows: 12,
                    maxBatchSize: 3
                ),
                RuntimeDecodeBatchingStats(
                    enabled: true,
                    activeRows: 2,
                    queuedRows: 0,
                    batchedDecodeSteps: 11,
                    samePositionBatchedSteps: 9,
                    variablePositionBatchedSteps: 2,
                    singleDecodeSteps: 13,
                    totalBatchedRows: 18,
                    maxBatchSize: 4
                ),
            ]
        )

        XCTAssertTrue(summary.available)
        XCTAssertEqual(summary.prefixKVReuse.reportedModelCount, 2)
        XCTAssertEqual(summary.prefixKVReuse.enabledModelCount, 1)
        XCTAssertEqual(summary.prefixKVReuse.entries, 2)
        XCTAssertEqual(summary.prefixKVReuse.maxEntries, 8)
        XCTAssertEqual(summary.prefixKVReuse.hits, 3)
        XCTAssertEqual(summary.prefixKVReuse.misses, 3)
        XCTAssertEqual(summary.prefixKVReuse.reusedTokens, 128)
        XCTAssertEqual(summary.decodeBatching.reportedModelCount, 2)
        XCTAssertEqual(summary.decodeBatching.enabledModelCount, 2)
        XCTAssertEqual(summary.decodeBatching.activeRows, 3)
        XCTAssertEqual(summary.decodeBatching.queuedRows, 2)
        XCTAssertEqual(summary.decodeBatching.batchedDecodeSteps, 16)
        XCTAssertEqual(summary.decodeBatching.samePositionBatchedSteps, 13)
        XCTAssertEqual(summary.decodeBatching.variablePositionBatchedSteps, 3)
        XCTAssertEqual(summary.decodeBatching.singleDecodeSteps, 20)
        XCTAssertEqual(summary.decodeBatching.totalBatchedRows, 30)
        XCTAssertEqual(summary.decodeBatching.maxBatchSize, 4)
        XCTAssertFalse(summary.ssdKVCache.available)
    }

    func testRuntimeBenchmarkStatsSummaryAggregatesCompletedRequests() {
        let summary = RuntimeBenchmarkStatsSummary(stats: [
            RuntimeModelBenchmarkStats(
                completedRequests: 2,
                failedRequests: 1,
                generatedTokens: 20,
                totalLoadSeconds: 0.2,
                totalPrefillSeconds: 0.4,
                totalDecodeSeconds: 1.0,
                lastCompletedAt: nil
            ),
            RuntimeModelBenchmarkStats(
                completedRequests: 1,
                failedRequests: 0,
                generatedTokens: 10,
                totalLoadSeconds: 0.2,
                totalPrefillSeconds: 0.5,
                totalDecodeSeconds: 1.0,
                lastCompletedAt: nil
            ),
        ])

        XCTAssertTrue(summary.available)
        XCTAssertEqual(summary.reportedModelCount, 2)
        XCTAssertEqual(summary.completedRequests, 3)
        XCTAssertEqual(summary.failedRequests, 1)
        XCTAssertEqual(summary.generatedTokens, 30)
        XCTAssertEqual(summary.averageLoadSeconds ?? 0, 0.4 / 3, accuracy: 0.000_001)
        XCTAssertEqual(summary.averagePrefillSeconds ?? 0, 0.9 / 3, accuracy: 0.000_001)
        XCTAssertEqual(summary.averageDecodeSeconds ?? 0, 2.0 / 3, accuracy: 0.000_001)
        XCTAssertEqual(summary.decodeTokensPerSecond ?? 0, 15, accuracy: 0.000_001)
    }
}
