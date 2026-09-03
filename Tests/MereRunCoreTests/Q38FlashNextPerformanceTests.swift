import Foundation
import MLX
import XCTest
@_spi(Benchmark) @testable import MereRunCore

/// Opt-in full-checkpoint measurements. Use an optimized test build and a
/// separate process per MTP setting; no logprob capture or prefix reuse.
final class Q38FlashNextPerformanceTests: MereRunCoreTestCase {
    func testInstalledVerificationFrontier() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_FLASH_NEXT_FRONTIER"] == "1",
                          "Flash-Next verification frontier is opt-in.")
        let root = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        let modelID = environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ID"]
            ?? Q35Resources.q38FlashNext3BitNativePLEModelId
        let tokenCount = max(32, Int(environment["MERERUN_TEST_Q38_FLASH_NEXT_FRONTIER_TOKENS"] ?? "") ?? 128)
        let trials = max(1, Int(environment["MERERUN_TEST_Q38_FLASH_NEXT_FRONTIER_TRIALS"] ?? "") ?? 2)
        let widths = try (environment["MERERUN_TEST_Q38_FLASH_NEXT_FRONTIER_WIDTHS"] ?? "1,4,8,16,32")
            .split(separator: ",")
            .map { value -> Int in
                guard let width = Int(value), width > 0 else {
                    throw XCTSkip("Verification widths must be positive integers.")
                }
                return width
            }
        let prompt = "Write a complete Python LRU cache using a dictionary and a doubly linked list."
        let messages = [ChatMessage(role: .user, content: prompt)]
        let generator = Q35Generator(
            modelId: modelID,
            prefixKVCacheEnabled: false,
            continuousBatchingEnabled: false
        )
        defer { Task { await generator.unload() } }
        _ = try await generator.chat(
            ChatRequest(
                messages: messages,
                maxTokens: 1,
                temperature: 0,
                topP: 1,
                showThinking: false,
                stopOnEOS: false,
                maxContextTokens: 8_192
            ),
            modelPath: root,
            progressHandler: nil
        )
        let results = try await generator.benchmarkVerificationFrontier(
            messages: messages,
            tokenCount: tokenCount,
            widths: widths,
            trials: trials
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        for result in results {
            let data = try encoder.encode(result)
            FileHandle.standardError.write(Data("[q38-verification-frontier] ".utf8) + data + Data("\n".utf8))
            if result.width <= 8 {
                XCTAssertTrue(result.greedyOutputParity, "Width \(result.width) changed greedy target output")
            }
        }
    }

    func testInstalledMixedCodeAndProse() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_FLASH_NEXT_BENCHMARK"] == "1",
                          "Full mixed-model benchmark is opt-in.")
        let root = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: root + "/model.safetensors.index.json"))
        let modelID = environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ID"]
            ?? Q35Resources.q38FlashNextMixedModelId
        let supportedModelIDs = [
            Q35Resources.q38FlashNextMixedModelId,
            Q35Resources.q38FlashNext3BitModelId,
            Q35Resources.q38FlashNext3BitNativePLEModelId,
            Q35Resources.q38FlashNext4BitModelId,
        ]
        XCTAssertTrue(supportedModelIDs.contains(modelID), "Unsupported Flash-Next benchmark model id")
        let trials = max(1, Int(environment["MERERUN_TEST_Q38_FLASH_NEXT_BENCHMARK_TRIALS"] ?? "") ?? 4)
        let outputTokens = max(
            64,
            Int(environment["MERERUN_TEST_Q38_FLASH_NEXT_BENCHMARK_TOKENS"] ?? "") ?? 256
        )
        let mtp = environment["MERERUN_Q35_MTP_SPECULATION"] == "1"
        let generator = Q35Generator(
            modelId: modelID,
            prefixKVCacheEnabled: false, continuousBatchingEnabled: false
        )
        do {
            try await benchmark(
                generator,
                root: root,
                modelID: modelID,
                mtp: mtp,
                trials: trials,
                outputTokens: outputTokens
            )
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func benchmark(
        _ generator: Q35Generator,
        root: String,
        modelID: String,
        mtp: Bool,
        trials: Int,
        outputTokens: Int
    ) async throws {
        let prompts = [
            ("code", "Write a complete Python implementation of an LRU cache using a dictionary and a "
                + "doubly linked list, with get and put operations, type hints, and a short usage example. "
                + "Return the code directly without introductory prose."),
            ("prose", "Explain how a small community could turn an abandoned railway station into a public "
                + "library. Discuss the building, accessibility, staffing, the collection, and a realistic "
                + "opening-day scene. Write approximately 400 words of clear, connected prose."),
        ]
        for trial in 0..<trials {
            for (label, prompt) in prompts {
                let started = Date()
                let response = try await generator.chat(
                    ChatRequest(messages: [.init(role: .user, content: prompt)], maxTokens: outputTokens,
                                temperature: 0, topP: 1, showThinking: false, stopOnEOS: false,
                                maxContextTokens: 8_192),
                    modelPath: root, progressHandler: { progress in
                        if progress.stage == .encoding {
                            let line = "[q38-benchmark-progress] \(progress.message ?? "encoding")\n"
                            FileHandle.standardError.write(Data(line.utf8))
                        }
                    }
                )
                let timing = try XCTUnwrap(response.timing)
                let record = Record(
                    modelID: modelID, workload: label, trial: trial, mtp: mtp,
                    mtpBlockSize: mtp ? Q35Generator.mtpBlockSize(modelId: modelID) : nil,
                    promptTokens: response.promptTokens ?? 0, outputTokens: response.tokensGenerated,
                    requestSeconds: Date().timeIntervalSince(started), loadSeconds: timing.loadSeconds,
                    prefillSeconds: timing.prefillSeconds, decodeSeconds: timing.decodeSeconds,
                    tokensPerSecond: Double(response.tokensGenerated) / timing.decodeSeconds,
                    acceleration: response.acceleration, response: response.response,
                    activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                let encoded = try encoder.encode(record)
                FileHandle.standardError.write(Data("[q38-benchmark] ".utf8) + encoded + Data("\n".utf8))
                XCTAssertEqual(response.acceleration?.route, mtp ? "mtp-speculative" : "final-target-pipelined")
                XCTAssertGreaterThanOrEqual(response.tokensGenerated, 64, "Too short for a useful decode measurement")
                XCTAssertTrue(timing.decodeSeconds.isFinite && timing.decodeSeconds > 0)
                XCTAssertFalse(response.response.isEmpty)
            }
        }
        let cache = await generator.prefixKVCacheStats()
        XCTAssertEqual(cache.entries, 0)
        XCTAssertEqual(cache.hits, 0)
    }

    private struct Record: Encodable {
        let modelID: String
        let workload: String
        let trial: Int
        let mtp: Bool
        let mtpBlockSize: Int?
        let promptTokens: Int
        let outputTokens: Int
        let requestSeconds: Double
        let loadSeconds: Double
        let prefillSeconds: Double
        let decodeSeconds: Double
        let tokensPerSecond: Double
        let acceleration: ChatAccelerationDiagnostics?
        let response: String
        let activeBytes: Int
        let cacheBytes: Int
    }
}
