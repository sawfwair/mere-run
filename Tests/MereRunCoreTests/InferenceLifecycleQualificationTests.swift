import AudioCore
import Foundation
import MLX
import XCTest
@testable import AudioSTT
@testable import MereRunCore

/// Opt-in lifecycle diagnostics. Checkpoints are local and results are not speed claims.
final class InferenceLifecycleQualificationTests: MereRunCoreTestCase {
    private struct Observation: Codable {
        let model: String
        let phase: String
        let streams: [String]
        let output: String
    }

    private final class Probe: @unchecked Sendable {
        private let lock = NSLock()
        private var seen: Set<String> = []
        private var cancelled = false

        func observe(cancel: Bool = false) {
            let stream = StreamOrDevice.default.stream.description
            let shouldCancel = lock.withLock {
                seen.insert(stream)
                if cancel, !cancelled {
                    cancelled = true
                    return true
                }
                return false
            }
            if shouldCancel { withUnsafeCurrentTask { $0?.cancel() } }
        }

        var streams: [String] { lock.withLock { seen.sorted() } }
        var triggered: Bool { lock.withLock { cancelled } }
    }

    private func paths() throws -> (models: URL, output: URL, audio: URL) {
        let env = ProcessInfo.processInfo.environment
        guard let models = env["MERERUN_STREAM_QUALIFICATION_MODELS"],
              let output = env["MERERUN_STREAM_QUALIFICATION_OUTPUT"],
              let audio = env["MERERUN_STREAM_QUALIFICATION_AUDIO"] else {
            throw XCTSkip("Set stream qualification model, output, and audio paths for installed diagnostics.")
        }
        return (URL(fileURLWithPath: models), URL(fileURLWithPath: output), URL(fileURLWithPath: audio))
    }

    private var request: ChatRequest {
        ChatRequest(
            messages: [ChatMessage(role: .user, content: "Write consecutive integers starting at 1, one integer per line.")],
            maxTokens: 24, temperature: 0, topP: 1, topK: 0, showThinking: false, maxContextTokens: 4096
        )
    }

    private func write(_ observations: [Observation], to output: URL, name: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(observations).write(to: output.appendingPathComponent(name + ".json"), options: .atomic)
    }

    func testInstalledPreparationCancellationAndRetry() async throws {
        let paths = try paths()
        var rows: [Observation] = []
        var expected: [String: String] = [:]
        for attempt in 0..<4 {
            let gemmaID = "text-chat-gemma4-12b-4bit"
            let gemmaPath = paths.models.appendingPathComponent(gemmaID).path
            let gemma = Gemma4Generator(modelId: gemmaID)
            let chatProbe = Probe()
            let chatTask = Task {
                try await gemma.prepare(modelPath: gemmaPath) { progress in
                    chatProbe.observe(cancel: progress.message == "Loading Gemma4 weights")
                }
            }
            do {
                try await chatTask.value
                XCTFail("Gemma preparation ignored cancellation")
            } catch is CancellationError {}
            XCTAssertTrue(chatProbe.triggered)
            let response = try await gemma.chat(request, modelPath: gemmaPath) { _ in chatProbe.observe() }
            XCTAssertFalse(response.response.isEmpty)
            if let prior = expected[gemmaID] { XCTAssertEqual(response.response, prior) }
            expected[gemmaID] = response.response
            rows.append(Observation(model: gemmaID, phase: "prepare-cancel-retry-\(attempt)",
                                    streams: chatProbe.streams, output: response.response))
            await gemma.unload()

            let asrID = "speech-asr-parakeet"
            let asrPath = paths.models.appendingPathComponent(asrID).path
            let asr = ParakeetGenerator()
            let asrProbe = Probe()
            let asrTask = Task {
                try await asr.prepare(modelPath: asrPath) { progress in
                    asrProbe.observe(cancel: progress.message == "Loading Parakeet weights...")
                }
            }
            do {
                try await asrTask.value
                XCTFail("Parakeet preparation ignored cancellation")
            } catch is CancellationError {}
            XCTAssertTrue(asrProbe.triggered)
            let transcript = try await asr.transcribe(ASRRequest(audioURL: paths.audio), modelPath: asrPath) { progress in
                if progress.stage == .transcribing { asrProbe.observe() }
            }
            XCTAssertFalse(transcript.text.isEmpty)
            if let prior = expected[asrID] { XCTAssertEqual(transcript.text, prior) }
            expected[asrID] = transcript.text
            rows.append(Observation(model: asrID, phase: "prepare-cancel-retry-\(attempt)",
                                    streams: asrProbe.streams, output: transcript.text))
            await asr.unload()
            try write(rows, to: paths.output, name: "preparation-cancellation")
        }
    }

    func testInstalledGemmaPrefillAndDecodeCancellationAndRetry() async throws {
        let paths = try paths()
        let modelID = "text-chat-gemma4-12b-4bit"
        let root = paths.models.appendingPathComponent(modelID).path
        let generator = Gemma4Generator(modelId: modelID, prefixKVCacheEnabled: true)
        let expected = try await generator.chat(request, modelPath: root, progressHandler: nil)
        var rows: [Observation] = []
        for phase in ["prefill", "decode"] {
            for attempt in 0..<4 {
                let probe = Probe()
                let longRequest = ChatRequest(
                    messages: [ChatMessage(role: .user, content:
                        Array(repeating: "alpha beta gamma delta", count: 400).joined(separator: " ")
                            + "\nWrite all integers from 1 through 500, one per line.")],
                    maxTokens: 128, temperature: 0, topP: 1, topK: 0,
                    showThinking: false, maxContextTokens: 4096
                )
                let task = Task {
                    try await generator.chat(longRequest, modelPath: root) { progress in
                        let matched = phase == "prefill"
                            ? progress.message?.hasPrefix("Prefilling ") == true
                            : progress.stage == .generating && progress.message?.isEmpty == false
                        probe.observe(cancel: matched)
                    }
                }
                do {
                    _ = try await task.value
                    XCTFail("Expected Gemma cancellation at \(phase)")
                } catch is CancellationError {}
                XCTAssertTrue(probe.triggered, "The requested phase must actually be observed")
                let retry = try await generator.chat(request, modelPath: root) { _ in probe.observe() }
                XCTAssertEqual(retry.response, expected.response)
                XCTAssertEqual(retry.tokensGenerated, expected.tokensGenerated)
                rows.append(Observation(model: modelID, phase: "\(phase)-cancel-retry-\(attempt)",
                                        streams: probe.streams, output: retry.response))
                try write(rows, to: paths.output, name: "gemma-phase-cancellation")
            }
        }
        await generator.unload()
    }

    func testInstalledParakeetCancellationAndRecreation() async throws {
        let paths = try paths()
        let modelID = "speech-asr-parakeet"
        let root = paths.models.appendingPathComponent(modelID).path
        let request = ASRRequest(audioURL: paths.audio)
        let generator = ParakeetGenerator()
        let expected = try await generator.transcribe(request, modelPath: root)
        var rows: [Observation] = []
        var streams: Set<String> = []
        for stage: ASRStage in [.extractingFeatures, .transcribing] {
            for attempt in 0..<4 {
                let probe = Probe()
                let task = Task {
                    try await generator.transcribe(request, modelPath: root) { progress in
                        if progress.stage == stage { probe.observe(cancel: true) }
                    }
                }
                do {
                    _ = try await task.value
                    XCTFail("Expected Parakeet cancellation at \(stage)")
                } catch is CancellationError {}
                XCTAssertTrue(probe.triggered)
                let retry = try await generator.transcribe(request, modelPath: root) { progress in
                    if progress.stage == .transcribing { probe.observe() }
                }
                XCTAssertEqual(retry, expected)
                streams.formUnion(probe.streams)
                rows.append(Observation(model: modelID, phase: "\(stage)-cancel-retry-\(attempt)",
                                        streams: probe.streams, output: retry.text))
                try write(rows, to: paths.output, name: "parakeet-phase-cancellation")
            }
        }
        await generator.unload()
        for attempt in 0..<10 {
            let owner = ParakeetGenerator()
            let probe = Probe()
            let result = try await owner.transcribe(request, modelPath: root) { progress in
                if progress.stage == .transcribing { probe.observe() }
            }
            XCTAssertEqual(result, expected)
            streams.formUnion(probe.streams)
            await owner.unload()
            rows.append(Observation(model: modelID, phase: "recreated-asr-\(attempt)",
                                    streams: probe.streams, output: result.text))
            try write(rows, to: paths.output, name: "parakeet-phase-cancellation")
        }
        XCTAssertEqual(streams.count, 1)
    }

    func testInstalledCheckpointSwitchAndRecreation() async throws {
        let paths = try paths()
        let models = ["text-chat-gemma4-12b-4bit", "text-chat-gemma4-12b-3bit"]
        let generator = Gemma4Generator(prefixKVCacheEnabled: true)
        var expected: [String: String] = [:]
        var rows: [Observation] = []
        var streams: Set<String> = []
        for attempt in 0..<10 {
            let id = models[attempt % models.count]
            let probe = Probe()
            let result = try await generator.chat(request, modelPath: paths.models.appendingPathComponent(id).path) {
                if $0.stage == .generating { probe.observe() }
            }
            XCTAssertFalse(result.response.isEmpty)
            if let prior = expected[id] { XCTAssertEqual(result.response, prior) }
            expected[id] = result.response
            streams.formUnion(probe.streams)
            rows.append(Observation(model: id, phase: "checkpoint-switch-\(attempt)",
                                    streams: probe.streams, output: result.response))
            try write(rows, to: paths.output, name: "checkpoint-switch")
        }
        await generator.unload()
        XCTAssertEqual(streams.count, 1)
        for attempt in 0..<10 {
            let id = models[attempt % models.count]
            let owner = Gemma4Generator(modelId: id)
            let probe = Probe()
            let result = try await owner.chat(request, modelPath: paths.models.appendingPathComponent(id).path) {
                if $0.stage == .generating { probe.observe() }
            }
            XCTAssertEqual(result.response, expected[id])
            streams.formUnion(probe.streams)
            await owner.unload()
            rows.append(Observation(model: id, phase: "recreated-checkpoint-\(attempt)",
                                    streams: probe.streams, output: result.response))
            try write(rows, to: paths.output, name: "checkpoint-switch")
        }
        XCTAssertEqual(streams.count, 1)
    }
}
