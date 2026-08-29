import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Opt-in: loads the installed 73 GB mixed checkpoint. Never part of the
/// fixture-only gate, and never downloads assets or starts a resident server.
final class Q38FlashNextCheckpointTests: MereRunCoreTestCase {
    private enum Mode {
        case mtpCache, targetCache, cleanTarget
    }

    private enum Context: Int {
        case k32 = 32_768
        case k64 = 65_536
        case k128 = 131_072

        var rows: Int {
            switch self {
            case .k32: 1_110
            case .k64: 2_230
            case .k128: 4_470
            }
        }

        var minimumPromptTokens: Int {
            switch self {
            case .k32: 32_000
            case .k64: 64_000
            case .k128: 128_000
            }
        }

        // Use the same answer allowance at every context size. The 32k
        // checkpoint can spend about 100 tokens explaining the reference.
        var maxTokens: Int { 256 }
    }

    func testInstalled32KRetrievalAndMTPPrefixReuse() async throws {
        try await qualifyInstalled(context: .k32)
    }

    func testInstalled64KRetrievalAndMTPPrefixReuse() async throws {
        try await qualifyInstalled(context: .k64)
    }

    func testInstalled128KRetrievalAndMTPPrefixReuse() async throws {
        try await qualifyInstalled(context: .k128)
    }

    func testInstalled64KCleanTargetBaseline() async throws {
        try await qualifyInstalled(context: .k64, mode: .cleanTarget)
    }

    func testInstalled128KCleanTargetBaseline() async throws {
        try await qualifyInstalled(context: .k128, mode: .cleanTarget)
    }

    func testInstalled64KTargetPrefixReuse() async throws {
        try await qualifyInstalled(context: .k64, mode: .targetCache)
    }

    func testInstalled128KTargetPrefixReuse() async throws {
        try await qualifyInstalled(context: .k128, mode: .targetCache)
    }

    private func qualifyInstalled(context: Context, mode: Mode = .mtpCache) async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS=1 for installed Flash-Next qualification.")
        }
        guard let root = environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"],
              FileManager.default.fileExists(atPath: root + "/model.safetensors.index.json") else {
            throw XCTSkip("Set MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT to the installed mixed checkpoint.")
        }
        let kvCacheMode = try XCTUnwrap(RuntimeKVCacheMode(
            rawValue: environment["MERERUN_TEST_Q38_FLASH_NEXT_KV_CACHE_MODE"] ?? "default"
        ))
        if mode != .mtpCache {
            try XCTSkipUnless(
                environment["MERERUN_Q35_MTP_SPECULATION"] == "0",
                "Target-only qualification requires MERERUN_Q35_MTP_SPECULATION=0."
            )
        }
        let generator = Q35Generator(
            modelId: Q35Resources.q38FlashNextMixedModelId,
            prefixKVCacheEnabled: mode != .cleanTarget,
            continuousBatchingEnabled: false
        )
        do {
            if mode == .cleanTarget {
                try await qualifyCleanTarget(generator, root: root, context: context, kvCacheMode: kvCacheMode)
            } else {
                try await qualify(
                    generator, root: root, context: context, kvCacheMode: kvCacheMode,
                    mtpEnabled: mode == .mtpCache
                )
            }
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func messages(context: Context) -> [ChatMessage] {
        let marker = "spruce-lantern-7429"
        let filler = (1...context.rows).map { index in
            "Archive \(String(format: "%04d", index)): routine weather observations, equipment checks, "
                + "and inventory counts were recorded. No project passphrase appears in this entry."
        }.joined(separator: "\n")
        let prompt = "Read the archive below and retrieve the exact reference entry.\n"
            + "REFERENCE: The project cobalt-harbor has passphrase \(marker).\n\(filler)\n"
            + "Question: What is the passphrase for project cobalt-harbor? "
            + "Return only the exact passphrase, without explanation."
        return [ChatMessage(role: .user, content: prompt)]
    }

    private func qualifyCleanTarget(
        _ generator: Q35Generator, root: String, context: Context, kvCacheMode: RuntimeKVCacheMode
    ) async throws {
        let response = try await generator.chat(
            request(messages(context: context), context: context, kvCacheMode: kvCacheMode),
            modelPath: root, progressHandler: { Self.tracePrefillMemory($0) }
        )
        let cache = await generator.prefixKVCacheStats()
        let line = "[q38-baseline] context=\(context.rawValue) kv=\(kvCacheMode.rawValue) "
            + "prompt_tokens=\(response.promptTokens ?? 0) output_tokens=\(response.tokensGenerated) "
            + "prefill_s=\(response.timing?.prefillSeconds ?? 0) "
            + "decode_s=\(response.timing?.decodeSeconds ?? 0) "
            + "route=\(response.acceleration?.route ?? "none") "
            + "cache_entries=\(cache.entries) cache_hits=\(cache.hits)\n"
            + "[q38-baseline-response] text=\(String(reflecting: response.response))\n"
        FileHandle.standardError.write(Data(line.utf8))
        let promptCount = try XCTUnwrap(response.promptTokens)
        XCTAssertGreaterThanOrEqual(promptCount, context.minimumPromptTokens)
        XCTAssertLessThanOrEqual(promptCount + response.tokensGenerated, context.rawValue)
        XCTAssertLessThan(response.tokensGenerated, context.maxTokens, "Baseline exhausted its answer budget")
        XCTAssertTrue(response.response.contains("spruce-lantern-7429"), "Reference was not retrieved")
        XCTAssertEqual(response.acceleration?.route, "final-target-pipelined")
        XCTAssertEqual(response.acceleration?.draftedTokens ?? 0, 0)
        XCTAssertEqual(cache.entries, 0)
        XCTAssertEqual(cache.hits, 0)
        XCTAssertEqual(cache.reusedTokens, 0)
    }

    private func qualify(
        _ generator: Q35Generator, root: String, context: Context, kvCacheMode: RuntimeKVCacheMode,
        mtpEnabled: Bool
    ) async throws {
        let marker = "spruce-lantern-7429"
        let expectedRoute = mtpEnabled ? "mtp-speculative" : "final-target-pipelined"
        let messages = messages(context: context)
        let cold = try await generator.chat(
            request(messages, context: context, kvCacheMode: kvCacheMode),
            modelPath: root, progressHandler: { Self.tracePrefillMemory($0) }
        )
        Self.traceResponse("cold", cold)
        let promptCount = try XCTUnwrap(cold.promptTokens)
        XCTAssertGreaterThanOrEqual(promptCount, context.minimumPromptTokens)
        XCTAssertLessThanOrEqual(promptCount + cold.tokensGenerated, context.rawValue)
        XCTAssertTrue(cold.response.contains(marker), "The early reference was not retrieved: \(cold.response)")
        XCTAssertLessThan(cold.tokensGenerated, context.maxTokens, "Cold answer exhausted its budget")
        XCTAssertEqual(cold.acceleration?.route, expectedRoute)
        if mtpEnabled {
            XCTAssertGreaterThan(cold.acceleration?.draftedTokens ?? 0, 0)
        } else {
            XCTAssertEqual(cold.acceleration?.draftedTokens ?? 0, 0)
        }
        let before = await generator.prefixKVCacheStats()
        XCTAssertEqual(before.entries, 1, "Cold prefill should retain only its final prompt checkpoint")
        XCTAssertEqual(before.storedPrefixes, 1, "Do not snapshot every intermediate prefill chunk")
        XCTAssertEqual(before.storedTokens, promptCount)

        let cached = try await generator.chat(
            request(messages, context: context, kvCacheMode: kvCacheMode),
            modelPath: root, progressHandler: { Self.tracePrefillMemory($0) }
        )
        Self.traceResponse("cached", cached)
        let after = await generator.prefixKVCacheStats()
        XCTAssertEqual(cached.response, cold.response)
        XCTAssertEqual(cached.tokensGenerated, cold.tokensGenerated)
        XCTAssertEqual(cached.acceleration?.route, expectedRoute)
        XCTAssertEqual(after.hits, before.hits + 1)
        XCTAssertEqual(after.reusedTokens - before.reusedTokens, promptCount)
        XCTAssertLessThanOrEqual(after.entries, after.maxEntries)

        // Compare before the follow-up changes request history, then repeat
        // after it. This separates speculative arithmetic from cache mutation.
        // Quality capture forces the final-target lane while keeping the same
        // immutable prefill checkpoint, without a second model or prefill.
        var serialRequest = request(messages, context: context, kvCacheMode: kvCacheMode)
        if mtpEnabled { serialRequest.logprobCapture = .tokens }
        let serial = try await generator.chat(
            serialRequest, modelPath: root, progressHandler: { Self.tracePrefillMemory($0) }
        )
        Self.traceResponse("serial", serial)
        let serialStats = await generator.prefixKVCacheStats()
        XCTAssertEqual(serial.acceleration?.route, "final-target-pipelined")
        XCTAssertEqual(serialStats.hits, after.hits + 1)
        XCTAssertEqual(serialStats.reusedTokens - after.reusedTokens, promptCount)
        XCTAssertEqual(cold.response, serial.response, "Cached decode changed the final target's greedy response")
        XCTAssertEqual(cold.tokensGenerated, serial.tokensGenerated)

        let followup = messages + [
            ChatMessage(role: .assistant, content: cold.response),
            ChatMessage(role: .user, content: "Repeat the passphrase from the archive's reference entry."),
        ]
        let continued = try await generator.chat(
            request(followup, context: context, kvCacheMode: kvCacheMode),
            modelPath: root, progressHandler: { Self.tracePrefillMemory($0) }
        )
        Self.traceResponse("followup", continued)
        let final = await generator.prefixKVCacheStats()
        XCTAssertTrue(continued.response.contains(marker), "Follow-up lost the reference: \(continued.response)")
        XCTAssertLessThan(continued.tokensGenerated, context.maxTokens, "Follow-up exhausted its answer budget")
        XCTAssertGreaterThan(final.hits, serialStats.hits)
        XCTAssertGreaterThan(final.reusedTokens, serialStats.reusedTokens)
        XCTAssertEqual(continued.acceleration?.route, expectedRoute)
        XCTAssertLessThanOrEqual(final.entries, final.maxEntries)
        XCTAssertLessThanOrEqual((continued.promptTokens ?? 0) + continued.tokensGenerated, context.rawValue)

        let replayedSerial = try await generator.chat(
            serialRequest, modelPath: root, progressHandler: { Self.tracePrefillMemory($0) }
        )
        Self.traceResponse("replayed-serial", replayedSerial)
        let replayedStats = await generator.prefixKVCacheStats()
        XCTAssertEqual(replayedSerial.acceleration?.route, "final-target-pipelined")
        XCTAssertEqual(replayedStats.hits, final.hits + 1)
        XCTAssertEqual(replayedStats.reusedTokens - final.reusedTokens, promptCount)
        XCTAssertEqual(serial.response, replayedSerial.response, "Follow-up mutated the original prompt checkpoint")
        XCTAssertEqual(serial.tokensGenerated, replayedSerial.tokensGenerated)

        for (label, response) in [
            ("cold", cold), ("cached", cached), ("serial", serial),
            ("followup", continued), ("replayed-serial", replayedSerial),
        ] {
            print(
                "[q38-checkpoint] case=\(label) prompt_tokens=\(response.promptTokens ?? 0) "
                    + "output_tokens=\(response.tokensGenerated) "
                    + "prefill_s=\(response.timing?.prefillSeconds ?? 0) "
                    + "decode_s=\(response.timing?.decodeSeconds ?? 0) "
                    + "drafted=\(response.acceleration?.draftedTokens ?? 0) "
                    + "accepted=\(response.acceleration?.acceptedDraftTokens ?? 0)"
            )
            print("[q38-checkpoint-response] case=\(label) text=\(String(reflecting: response.response))")
        }
    }

    private static func tracePrefillMemory(_ progress: ChatProgress) {
        if progress.stage == .generating, let message = progress.message {
            FileHandle.standardError.write(Data("[q38-checkpoint-stream] \(String(reflecting: message))\n".utf8))
        }
        guard progress.stage == .encoding else { return }
        let snapshot = Memory.snapshot()
        let line = "[q38-checkpoint-memory] \(progress.message ?? "") "
            + "active_bytes=\(snapshot.activeMemory) cache_bytes=\(snapshot.cacheMemory) "
            + "memory_limit=\(Memory.memoryLimit) cache_limit=\(Memory.cacheLimit)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static func traceResponse(_ label: String, _ response: ChatResponse) {
        let line = "[q38-checkpoint-stage] case=\(label) text=\(String(reflecting: response.response))\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func request(_ messages: [ChatMessage], context: Context, kvCacheMode: RuntimeKVCacheMode) -> ChatRequest {
        var request = ChatRequest(
            messages: messages, maxTokens: context.maxTokens,
            temperature: 0, topP: 1, showThinking: false, maxContextTokens: context.rawValue
        )
        request.kvCacheMode = kvCacheMode
        return request
    }
}
