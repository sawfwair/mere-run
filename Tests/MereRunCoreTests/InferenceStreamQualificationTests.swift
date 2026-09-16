import AudioCore
import AudioSTT
import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Opt-in diagnostics use installed checkpoints and write observations for a paired comparison.
final class InferenceStreamQualificationTests: MereRunCoreTestCase {
    private struct Sample: Codable {
        let request: Int
        let streams: [String]
        let response: String
        let generatedTokens: Int?
        let finishReason: String?
        let seconds: Double
    }

    private final class StreamObservations: @unchecked Sendable {
        private let lock = NSLock()
        private var values: Set<String> = []

        func record() {
            let value = StreamOrDevice.default.stream.description
            lock.withLock { _ = values.insert(value) }
        }

        var streams: [String] { lock.withLock { values.sorted() } }
    }

    private func configuration() throws -> (models: URL, output: URL) {
        let environment = ProcessInfo.processInfo.environment
        guard let models = environment["MERERUN_STREAM_QUALIFICATION_MODELS"],
              let output = environment["MERERUN_STREAM_QUALIFICATION_OUTPUT"] else {
            throw XCTSkip("Set stream qualification model and output directories for installed diagnostics.")
        }
        return (URL(fileURLWithPath: models), URL(fileURLWithPath: output))
    }

    private func write(_ samples: [Sample], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(samples).write(to: url, options: .atomic)
    }

    func testInstalledGemmaRequestsRecordExecutionStreams() async throws {
        let paths = try configuration()
        let modelID = "text-chat-gemma4-12b-4bit"
        let root = paths.models.appendingPathComponent(modelID).path
        XCTAssertTrue(FileManager.default.fileExists(atPath: root))
        let generator = Gemma4Generator(modelId: modelID, prefixKVCacheEnabled: true)
        let request = ChatRequest(
            messages: [
                ChatMessage(role: .system, content: "Follow the requested output format."),
                ChatMessage(
                    role: .user,
                    content: "Write consecutive integers starting at 1, one integer per line. Continue until stopped."
                ),
            ],
            maxTokens: 128, temperature: 0, topP: 1, topK: 0,
            showThinking: false, maxContextTokens: 2048
        )
        var samples: [Sample] = []
        do {
            for index in 0..<6 {
                let observations = StreamObservations()
                let started = ProcessInfo.processInfo.systemUptime
                let response = try await generator.chat(request, modelPath: root) { progress in
                    if progress.stage == .generating { observations.record() }
                }
                samples.append(Sample(
                    request: index, streams: observations.streams, response: response.response,
                    generatedTokens: response.tokensGenerated, finishReason: String(describing: response.finishReason),
                    seconds: ProcessInfo.processInfo.systemUptime - started
                ))
                try write(samples, to: paths.output.appendingPathComponent("gemma-streams.json"))
                XCTAssertFalse(observations.streams.isEmpty)
                XCTAssertGreaterThan(response.tokensGenerated, 0)
            }
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }

    func testInstalledParakeetRequestsRecordExecutionStreams() async throws {
        let paths = try configuration()
        guard let audio = ProcessInfo.processInfo.environment["MERERUN_STREAM_QUALIFICATION_AUDIO"] else {
            throw XCTSkip("Set the fixed speech fixture for installed Parakeet diagnostics.")
        }
        let modelID = "speech-asr-parakeet"
        let root = paths.models.appendingPathComponent(modelID).path
        XCTAssertTrue(FileManager.default.fileExists(atPath: root))
        let generator = ParakeetGenerator(modelId: modelID, executionProvider: .mlx)
        let request = ASRRequest(audioURL: URL(fileURLWithPath: audio))
        var samples: [Sample] = []
        do {
            for index in 0..<6 {
                let observations = StreamObservations()
                let started = ProcessInfo.processInfo.systemUptime
                let response = try await generator.transcribe(request, modelPath: root) { progress in
                    if progress.stage == .transcribing { observations.record() }
                }
                samples.append(Sample(
                    request: index, streams: observations.streams, response: response.text,
                    generatedTokens: nil, finishReason: nil,
                    seconds: ProcessInfo.processInfo.systemUptime - started
                ))
                try write(samples, to: paths.output.appendingPathComponent("parakeet-streams.json"))
                XCTAssertFalse(observations.streams.isEmpty)
                XCTAssertFalse(response.text.isEmpty)
            }
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }
    func testInstalledGemmaBatchedLoopOwnsItsStream() async throws {
        let paths = try configuration()
        let modelID = "text-chat-gemma4-12b-4bit"
        let root = paths.models.appendingPathComponent(modelID).path
        let generator = Gemma4Generator(
            modelId: modelID, prefixKVCacheEnabled: true, continuousBatchingEnabled: true
        )
        try await generator.prepare(modelPath: root)
        let request = ChatRequest(
            messages: [
                ChatMessage(role: .system, content: "Follow the requested output format."),
                ChatMessage(
                    role: .user,
                    content: "Write consecutive integers starting at 1, one integer per line. Continue until stopped."
                ),
            ],
            maxTokens: 128, temperature: 0, topP: 1, topK: 0,
            showThinking: false, maxContextTokens: 2048
        )
        let encoding = StreamObservations()
        let decoding = StreamObservations()
        do {
            let results = try await withThrowingTaskGroup(of: ChatResponse.self) { group in
                for _ in 0..<4 {
                    group.addTask {
                        try await generator.chat(request, modelPath: root) { progress in
                            if progress.stage == .generating {
                                if progress.message == "" { encoding.record() } else { decoding.record() }
                            }
                        }
                    }
                }
                var results: [ChatResponse] = []
                for try await result in group { results.append(result) }
                return results
            }
            XCTAssertEqual(results.count, 4)
            XCTAssertEqual(Set(results.map(\.response)).count, 1)
            XCTAssertTrue(results.allSatisfy { $0.tokensGenerated == 20 })
            XCTAssertEqual(encoding.streams.count, 4)
            XCTAssertEqual(decoding.streams.count, 1)
            XCTAssertTrue(Set(encoding.streams).isDisjoint(with: decoding.streams))
            let stats = await generator.continuousBatchingStats()
            XCTAssertGreaterThanOrEqual(stats.maxBatchSize, 2)
            XCTAssertEqual(stats.activeRows, 0)
            XCTAssertEqual(stats.queuedRows, 0)
            try write([
                Sample(request: 0, streams: encoding.streams, response: "encoding", generatedTokens: nil,
                       finishReason: nil, seconds: 0),
                Sample(request: 1, streams: decoding.streams, response: "decoding", generatedTokens: nil,
                       finishReason: nil, seconds: 0),
            ], to: paths.output.appendingPathComponent("gemma-batched-streams.json"))
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }

    func testInstalledPreparationAndUnloadReuseStreams() async throws {
        let paths = try configuration()
        guard let audio = ProcessInfo.processInfo.environment["MERERUN_STREAM_QUALIFICATION_AUDIO"] else {
            throw XCTSkip("Set the fixed speech fixture for installed Parakeet diagnostics.")
        }
        let gemma = Gemma4Generator()
        let parakeet = ParakeetGenerator()
        let chat = ChatRequest(messages: [ChatMessage(role: .user, content: "Say ready.")],
                               maxTokens: 8, temperature: 0)
        let speech = ASRRequest(audioURL: URL(fileURLWithPath: audio))
        var gemmaSamples: [Sample] = []
        var parakeetSamples: [Sample] = []
        do {
            for index in 0..<10 {
                let gemmaStreams = StreamObservations()
                let gemmaPath = paths.models.appendingPathComponent("text-chat-gemma4-12b-4bit").path
                try await gemma.prepare(modelPath: gemmaPath)
                let response = try await gemma.chat(chat, modelPath: gemmaPath) { progress in
                    if progress.stage == .generating { gemmaStreams.record() }
                }
                await gemma.unload()
                gemmaSamples.append(Sample(request: index, streams: gemmaStreams.streams,
                                           response: response.response, generatedTokens: response.tokensGenerated,
                                           finishReason: String(describing: response.finishReason), seconds: 0))
                try write(gemmaSamples, to: paths.output.appendingPathComponent("gemma-prepare-unload-streams.json"))

                let parakeetStreams = StreamObservations()
                let parakeetPath = paths.models.appendingPathComponent("speech-asr-parakeet").path
                try await parakeet.prepare(modelPath: parakeetPath)
                let transcript = try await parakeet.transcribe(speech, modelPath: parakeetPath) { progress in
                    if progress.stage == .transcribing { parakeetStreams.record() }
                }
                await parakeet.unload()
                parakeetSamples.append(Sample(request: index, streams: parakeetStreams.streams,
                                              response: transcript.text, generatedTokens: nil,
                                              finishReason: nil, seconds: 0))
                try write(parakeetSamples, to: paths.output.appendingPathComponent("parakeet-prepare-unload-streams.json"))
            }
        } catch {
            await gemma.unload()
            await parakeet.unload()
            throw error
        }
        for samples in [gemmaSamples, parakeetSamples] {
            XCTAssertEqual(Set(samples.flatMap(\.streams)).count, 1)
            XCTAssertEqual(Set(samples.map(\.response)).count, 1)
            XCTAssertTrue(samples.allSatisfy { !$0.response.isEmpty && !$0.streams.isEmpty })
        }
    }

}
