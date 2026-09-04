@testable import StudioKit
import XCTest

final class StudioServingConsoleTests: XCTestCase {
    func testDecodesModernRuntimeSnapshotAcrossEveryOperationalLane() throws {
        let json = """
        {
          "object": "runtime.model_pool",
          "defaultModel": "text-chat-gemma4-12b-4bit",
          "settingsPath": "/tmp/runtime.json",
          "activeRequests": 1,
          "admission": {
            "maxActiveRequests": 4,
            "activeRequests": 1,
            "queuedRequests": 2,
            "totalAdmittedRequests": 10,
            "totalCompletedRequests": 7,
            "totalCancelledRequests": 0,
            "admissionPaused": false,
            "pressure": "nominal"
          },
          "memory": {
            "physicalBytes": 103079215104,
            "residentBytes": 1000,
            "currentBytes": 900,
            "availableBytes": 800,
            "ceilingBytes": 700,
            "softLimitBytes": 600,
            "hardLimitBytes": 650,
            "activeRequests": 1,
            "activeModelCount": 2,
            "guardTier": "balanced",
            "pressure": "nominal"
          },
          "models": [{
            "id": "text-chat-gemma4-12b-4bit",
            "category": "text",
            "engine": "text-chat-gemma4",
            "loaded": true,
            "ready": true,
            "activeRequests": 1,
            "pinned": true,
            "alias": "local",
            "ttlSeconds": 300,
            "benchmarkStats": {
              "completedRequests": 7,
              "failedRequests": 1,
              "generatedTokens": 500,
              "totalLoadSeconds": 0,
              "totalPrefillSeconds": 0,
              "totalDecodeSeconds": 0
            }
          }],
          "cacheStats": {
            "available": true,
            "detail": "live",
            "prefixKVReuse": {
              "reportedModelCount": 1,
              "enabledModelCount": 1,
              "entries": 2,
              "maxEntries": 8,
              "hits": 3,
              "misses": 1,
              "storedPrefixes": 2,
              "reusedTokens": 100,
              "storedTokens": 200
            },
            "decodeBatching": {
              "reportedModelCount": 1,
              "enabledModelCount": 1,
              "activeRows": 1,
              "queuedRows": 2,
              "batchedDecodeSteps": 3,
              "samePositionBatchedSteps": 1,
              "variablePositionBatchedSteps": 2,
              "singleDecodeSteps": 4,
              "totalBatchedRows": 5,
              "maxBatchSize": 2
            }
          },
          "benchmarkStats": {
            "available": true,
            "detail": "observed",
            "reportedModelCount": 1,
            "completedRequests": 7,
            "failedRequests": 1,
            "generatedTokens": 500,
            "averageLoadSeconds": 1,
            "averagePrefillSeconds": 2,
            "averageDecodeSeconds": 3,
            "averageTotalSeconds": 6,
            "decodeTokensPerSecond": 22
          },
          "sidecars": {
            "defaultIdleTTLSeconds": 300,
            "pressure": "nominal",
            "loadedCount": 1,
            "activeRequests": 0,
            "queuedRequests": 0,
            "residents": [{
              "kind": "image",
              "modelID": "image-zimage-nano",
              "loaded": true,
              "ready": true,
              "activeRequests": 0,
              "queuedRequests": 0,
              "pinned": false,
              "ttlSeconds": 300,
              "loadCount": 1,
              "replacementCount": 0,
              "evictionCount": 0,
              "completedRequests": 2,
              "failedRequests": 0
            }]
          },
          "process": {
            "processID": 42,
            "startedAt": 0,
            "uptimeSeconds": 15,
            "cpuPercent": 31.5,
            "thermalState": "nominal",
            "lowPowerModeEnabled": false,
            "metalDeviceName": "Apple GPU",
            "metalCurrentAllocatedBytes": 1234,
            "metalRecommendedMaxWorkingSetBytes": 5678,
            "metalHasUnifiedMemory": true
          }
        }
        """

        let snapshot = try JSONDecoder().decode(
            StudioRuntimeSnapshot.self,
            from: try XCTUnwrap(json.data(using: .utf8))
        )

        XCTAssertEqual(snapshot.loadedTextModels.map(\.id), ["text-chat-gemma4-12b-4bit"])
        XCTAssertEqual(snapshot.loadedSidecars.map(\.kind), ["image"])
        XCTAssertEqual(snapshot.admission?.queuedRequests, 2)
        XCTAssertEqual(snapshot.memory?.availableBytes, 800)
        XCTAssertEqual(snapshot.process?.metalCurrentAllocatedBytes, 1234)
        XCTAssertEqual(snapshot.cacheStats?.prefixKVReuse?.hitRate, 0.75)
        XCTAssertEqual(snapshot.totalFailures, 1)
    }

    func testOlderRuntimeSnapshotWithoutAdditiveFieldsStillDecodes() throws {
        let json = """
        {
          "activeRequests": 0,
          "models": [{
            "id": "text-chat-qwen",
            "loaded": true,
            "activeRequests": 0,
            "pinned": false
          }]
        }
        """

        let snapshot = try JSONDecoder().decode(
            StudioRuntimeSnapshot.self,
            from: try XCTUnwrap(json.data(using: .utf8))
        )

        XCTAssertEqual(snapshot.loadedTextModels.first?.state, "Ready")
        XCTAssertNil(snapshot.process)
        XCTAssertNil(snapshot.sidecars)
        XCTAssertNil(snapshot.memory)
    }

    func testServingSafetyRequiresAuthenticationAwayFromLoopback() {
        XCTAssertEqual(StudioServingSafety.evaluate(host: "127.0.0.1", apiKey: ""), .loopback)
        XCTAssertEqual(StudioServingSafety.evaluate(host: "::1", apiKey: ""), .loopback)
        XCTAssertEqual(
            StudioServingSafety.evaluate(host: "0.0.0.0", apiKey: ""),
            .exposedWithoutAuthentication
        )
        XCTAssertEqual(
            StudioServingSafety.evaluate(host: "192.168.1.10", apiKey: "secret"),
            .protectedLAN
        )
    }

    func testActivitySanitizerRedactsSecretsAndRequestContents() {
        let raw = #"Authorization: Bearer secret-token "prompt":"private story" api_key=topsecret"#
        let sanitized = StudioActivitySanitizer.sanitize(raw)

        XCTAssertFalse(sanitized.contains("secret-token"))
        XCTAssertFalse(sanitized.contains("private story"))
        XCTAssertFalse(sanitized.contains("topsecret"))
        XCTAssertTrue(sanitized.contains("[redacted]"))
    }

    func testActivityDiffReportsPoolLifecycleWithoutRequestBodies() throws {
        let previous = try snapshot(loaded: false, failures: 0)
        let current = try snapshot(loaded: true, failures: 1)

        let events = StudioServiceActivityDiff.events(previous: previous, current: current)

        XCTAssertTrue(events.contains { $0.title == "Text model loaded" })
        XCTAssertTrue(events.contains { $0.title == "Request failure recorded" })
        XCTAssertFalse(events.compactMap(\.detail).joined().contains("prompt"))
    }

    func testAPIServerKeyStaysInEnvironmentInsteadOfArguments() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        var draft = template.defaultDraft()
        draft.host = "0.0.0.0"
        draft.apiKey = "private-key"

        XCTAssertFalse(template.arguments(from: draft).contains("private-key"))
        XCTAssertEqual(
            CommandLaunchEnvironment.overrides(templateID: .apiServe, draft: draft)["MERERUN_API_KEY"],
            "private-key"
        )
    }

    private func snapshot(loaded: Bool, failures: Int) throws -> StudioRuntimeSnapshot {
        let json = """
        {
          "models": [{
            "id": "text-chat-qwen",
            "loaded": \(loaded),
            "ready": \(loaded),
            "activeRequests": 0,
            "pinned": false,
            "benchmarkStats": {
              "completedRequests": 1,
              "failedRequests": \(failures),
              "generatedTokens": 1,
              "totalLoadSeconds": 0,
              "totalPrefillSeconds": 0,
              "totalDecodeSeconds": 0
            }
          }]
        }
        """
        return try JSONDecoder().decode(
            StudioRuntimeSnapshot.self,
            from: try XCTUnwrap(json.data(using: .utf8))
        )
    }
}
