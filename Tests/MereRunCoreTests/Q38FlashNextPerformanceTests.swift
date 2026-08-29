import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Opt-in full-checkpoint measurements. Use an optimized test build and a
/// separate process per MTP setting; no logprob capture or prefix reuse.
final class Q38FlashNextPerformanceTests: MereRunCoreTestCase {
    func testInstalledMixedCodeAndProse() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_FLASH_NEXT_BENCHMARK"] == "1",
                          "Full mixed-model benchmark is opt-in.")
        let root = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: root + "/model.safetensors.index.json"))
        let mtp = environment["MERERUN_Q35_MTP_SPECULATION"] == "1"
        let generator = Q35Generator(
            modelId: Q35Resources.q38FlashNextMixedModelId,
            prefixKVCacheEnabled: false, continuousBatchingEnabled: false
        )
        do {
            try await benchmark(generator, root: root, mtp: mtp)
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func benchmark(_ generator: Q35Generator, root: String, mtp: Bool) async throws {
        let prompts = [
            ("code", "Write a complete Python implementation of an LRU cache using a dictionary and a "
                + "doubly linked list, with get and put operations, type hints, and a short usage example. "
                + "Return the code directly without introductory prose."),
            ("prose", "Explain how a small community could turn an abandoned railway station into a public "
                + "library. Discuss the building, accessibility, staffing, the collection, and a realistic "
                + "opening-day scene. Write approximately 400 words of clear, connected prose."),
        ]
        for trial in 0..<4 {
            for (label, prompt) in prompts {
                let started = Date()
                let response = try await generator.chat(
                    ChatRequest(messages: [.init(role: .user, content: prompt)], maxTokens: 256,
                                temperature: 0, topP: 1, showThinking: false, maxContextTokens: 8_192),
                    modelPath: root, progressHandler: { progress in
                        if progress.stage == .encoding {
                            let line = "[q38-benchmark-progress] \(progress.message ?? "encoding")\n"
                            FileHandle.standardError.write(Data(line.utf8))
                        }
                    }
                )
                let timing = try XCTUnwrap(response.timing)
                let record = Record(
                    workload: label, trial: trial, mtp: mtp,
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
        let workload: String
        let trial: Int
        let mtp: Bool
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
