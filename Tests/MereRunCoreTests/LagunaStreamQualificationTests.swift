import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LagunaStreamQualificationTests: MereRunCoreTestCase {
    private struct Sample: Codable {
        let request: Int
        let phase: String
        let streams: [String]
        let response: String
        let tokens: Int
        let finishReason: String
    }

    private final class Probe: @unchecked Sendable {
        private let lock = NSLock()
        private var values: Set<String> = []

        @discardableResult
        func record() -> Bool {
            let stream = StreamOrDevice.default.stream.description
            return lock.withLock {
                let first = values.isEmpty
                values.insert(stream)
                return first
            }
        }

        var streams: [String] { lock.withLock { values.sorted() } }
    }

    func testInstalledSequentialRequestsReuseExecutionStreams() async throws {
        let (root, output) = try configuration()
        let generator = LagunaGenerator(continuousBatchingEnabled: false, dflashModelPath: nil)
        let request = makeRequest()
        var samples: [Sample] = []
        var streams: Set<String> = []
        do {
            for index in 0..<6 {
                let probe = Probe()
                let response = try await generator.chat(request, modelPath: root, dflashRouting: .targetOnly) { progress in
                    if progress.stage == .generating { probe.record() }
                }
                XCTAssertFalse(response.response.isEmpty)
                XCTAssertGreaterThan(response.tokensGenerated, 0)
                XCTAssertFalse(probe.streams.isEmpty)
                if let first = samples.first {
                    XCTAssertEqual(response.response, first.response)
                    XCTAssertEqual(response.tokensGenerated, first.tokens)
                }
                streams.formUnion(probe.streams)
                samples.append(Sample(request: index, phase: "resident", streams: probe.streams, response: response.response,
                                      tokens: response.tokensGenerated, finishReason: String(describing: response.finishReason)))
                try write(samples, to: output, name: "laguna-sequential")
            }
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
        XCTAssertEqual(streams.count, 1, "Completed requests must reuse a bounded execution stream")
    }

    func testInstalledPreparationCancellationAllowsRetry() async throws {
        let (root, output) = try configuration()
        var samples: [Sample] = []
        for attempt in 0..<4 {
            let generator = LagunaGenerator(continuousBatchingEnabled: false, dflashModelPath: nil)
            let probe = Probe()
            let task = Task {
                try await generator.prepare(modelPath: root) { progress in
                    if progress.message == "Loading Laguna weights" {
                        probe.record()
                        withUnsafeCurrentTask { $0?.cancel() }
                    }
                }
            }
            do {
                try await task.value
                XCTFail("Laguna preparation ignored cancellation")
            } catch is CancellationError {}
            XCTAssertFalse(probe.streams.isEmpty)
            let response = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly)
            if let first = samples.first { XCTAssertEqual(response.response, first.response) }
            samples.append(sample(response, probe: probe, index: attempt, phase: "prepare-cancel-retry"))
            await generator.unload()
            try write(samples, to: output, name: "laguna-preparation")
        }
    }

    func testInstalledCancellationErrorsAndRetry() async throws {
        let (root, output) = try configuration()
        let generator = LagunaGenerator(continuousBatchingEnabled: false, dflashModelPath: nil)
        let expected = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly)
        var samples: [Sample] = []
        for phase in ["encoding", "decode"] {
            for attempt in 0..<4 {
                let probe = Probe()
                let request = makeRequest(maxTokens: 128)
                let task = Task {
                    try await generator.chat(request, modelPath: root, dflashRouting: .targetOnly) { progress in
                        let matched = phase == "encoding" ? progress.stage == .encoding
                            : progress.stage == .generating && progress.message?.isEmpty == false
                        if matched {
                            probe.record()
                            withUnsafeCurrentTask { $0?.cancel() }
                        }
                    }
                }
                do {
                    _ = try await task.value
                    XCTFail("Expected cancellation at \(phase)")
                } catch is CancellationError {}
                XCTAssertFalse(probe.streams.isEmpty)
                let retry = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly)
                XCTAssertEqual(retry.response, expected.response)
                XCTAssertEqual(retry.tokensGenerated, expected.tokensGenerated)
                samples.append(sample(retry, probe: probe, index: attempt, phase: phase + "-cancel-retry"))
            }
        }
        for _ in 0..<4 {
            var invalid = makeRequest()
            invalid.maxContextTokens = 0
            do {
                _ = try await generator.chat(invalid, modelPath: root, dflashRouting: .targetOnly)
                XCTFail("Expected invalid context failure")
            } catch let error as LagunaError {
                guard case .generationFailed = error else { throw error }
            }
            let retry = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly)
            XCTAssertEqual(retry.response, expected.response)
        }
        await generator.unload()
        try write(samples, to: output, name: "laguna-cancellation")
    }

    func testInstalledUnloadAndRecreationReuseStreams() async throws {
        let (root, output) = try configuration()
        let retained = LagunaGenerator(continuousBatchingEnabled: false, dflashModelPath: nil)
        var samples: [Sample] = []
        var streams: Set<String> = []
        for phase in ["retained", "recreated"] {
            for attempt in 0..<10 {
                let generator = phase == "retained" ? retained
                    : LagunaGenerator(continuousBatchingEnabled: false, dflashModelPath: nil)
                let probe = Probe()
                try await generator.prepare(modelPath: root)
                let response = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly) {
                    if $0.stage == .generating { probe.record() }
                }
                if let first = samples.first { XCTAssertEqual(response.response, first.response) }
                samples.append(sample(response, probe: probe, index: attempt, phase: phase))
                streams.formUnion(probe.streams)
                await generator.unload()
                try write(samples, to: output, name: "laguna-lifecycle")
            }
        }
        XCTAssertEqual(streams.count, 1)
    }

    func testInstalledBatchedLoopOwnsStreamsAndSurvivesCancellation() async throws {
        let (root, output) = try configuration()
        let generator = LagunaGenerator(continuousBatchingEnabled: true, dflashModelPath: nil)
        try await generator.prepare(modelPath: root)
        let expected = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly)
        let encoding = Probe()
        let decoding = Probe()
        let firstToken = expectation(description: "A token was emitted for the cancelled row")
        let longRequest = makeRequest(maxTokens: 128)
        let normalRequest = makeRequest()
        let cancelled = Task {
            try await generator.chat(longRequest, modelPath: root, dflashRouting: .targetOnly) { progress in
                if progress.stage == .generating {
                    if progress.message == "" { encoding.record() }
                    else if progress.message?.isEmpty == false, decoding.record() { firstToken.fulfill() }
                }
            }
        }
        let survivor = Task {
            try await generator.chat(normalRequest, modelPath: root, dflashRouting: .targetOnly) { progress in
                if progress.stage == .generating, progress.message == "" { encoding.record() }
            }
        }
        await fulfillment(of: [firstToken], timeout: 30)
        cancelled.cancel()
        do {
            _ = try await cancelled.value
            XCTFail("Expected the selected batch row to cancel")
        } catch is CancellationError {}
        let response = try await survivor.value
        XCTAssertEqual(response.response, expected.response)
        XCTAssertEqual(response.tokensGenerated, expected.tokensGenerated)
        XCTAssertEqual(encoding.streams.count, 2)
        XCTAssertEqual(decoding.streams.count, 1)
        XCTAssertTrue(Set(encoding.streams).isDisjoint(with: decoding.streams))
        let stats = await generator.continuousBatchingStats()
        XCTAssertGreaterThanOrEqual(stats.maxBatchSize, 2)
        XCTAssertEqual(stats.activeRows, 0)
        XCTAssertEqual(stats.queuedRows, 0)
        let retry = try await generator.chat(makeRequest(), modelPath: root, dflashRouting: .targetOnly)
        XCTAssertEqual(retry.response, expected.response)
        await generator.unload()
        try write([
            sample(response, probe: encoding, index: 0, phase: "request-streams"),
            sample(retry, probe: decoding, index: 1, phase: "loop-stream-and-retry"),
        ], to: output, name: "laguna-batched-cancellation")
    }

    private func configuration() throws -> (String, URL) {
        let env = ProcessInfo.processInfo.environment
        guard let models = env["MERERUN_STREAM_QUALIFICATION_MODELS"],
              let output = env["MERERUN_STREAM_QUALIFICATION_OUTPUT"] else {
            throw XCTSkip("Set stream qualification model and output directories for installed Laguna diagnostics.")
        }
        return (URL(fileURLWithPath: models).appendingPathComponent("text-chat-laguna-xs-2-1").path,
                URL(fileURLWithPath: output))
    }

    private func makeRequest(maxTokens: Int = 32) -> ChatRequest {
        ChatRequest(
            messages: [ChatMessage(role: .user, content: "Write consecutive integers starting at 1, one integer per line.")],
            maxTokens: maxTokens, temperature: 0, topP: 1, topK: 0, showThinking: false, maxContextTokens: 2048
        )
    }

    private func sample(_ response: ChatResponse, probe: Probe, index: Int, phase: String) -> Sample {
        Sample(request: index, phase: phase, streams: probe.streams, response: response.response,
               tokens: response.tokensGenerated, finishReason: String(describing: response.finishReason))
    }

    private func write(_ samples: [Sample], to output: URL, name: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(samples).write(to: output.appendingPathComponent(name + ".json"), options: .atomic)
    }
}
