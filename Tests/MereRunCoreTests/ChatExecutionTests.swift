import Foundation
import XCTest
@testable import MereRunCore

final class ChatExecutionTests: XCTestCase {
    func testCompatibilityDefaultsKeepCommandAndOpenAIBehavior() throws {
        let request = ChatRequest(messages: [.init(role: .user, content: "hello")])
        let fixtures: [(String, Double, Double, Int?, Double)] = [
            ("fixture", 0.7, 0.9, nil, 0),
            (LFM2Resources.visionModelId, 0.2, 0.9, 50, 0),
            (LagunaResources.modelID, 1, 1, 20, 0.02),
            (MuseGlimmerResources.modelId, 1, 0.95, 64, 0),
            (NemotronHResources.modelID, 1, 0.95, nil, 0),
            (NemotronOmniResources.modelID, 0.6, 0.95, nil, 0)
        ]
        for (id, temperature, topP, topK, minP) in fixtures {
            let command = try ChatRequestResolver.resolve(request, modelID: id, sampling: .init(), policy: .command)
            XCTAssertEqual(command.temperature, temperature, id)
            XCTAssertEqual(command.topP, topP, id)
            XCTAssertEqual(command.topK, topK, id)
            XCTAssertEqual(command.minP, minP, id)
        }
        let api = try ChatRequestResolver.resolve(request, modelID: "fixture", sampling: .init(), policy: .openAI)
        XCTAssertEqual(api.temperature, 1)
        XCTAssertEqual(api.topP, 0.95)
        let muse = try ChatRequestResolver.resolve(request, modelID: MuseGlimmerResources.modelId, sampling: .init(), policy: .openAI)
        XCTAssertNil(muse.topK, "The v1 API keeps its existing omitted top-k behavior")
    }

    func testCommandSelectionPreservesFamiliesAndModelPaths() throws {
        let path = "/fixture/model"
        let fixtures = [
            Psi3ChatResources.defaultModelId, InklingResources.modelID,
            DiffusionGemmaResources.modelID, Gemma4Resources.defaultModelId,
            LagunaResources.modelID, Q35Resources.ornith35BMLX4BitModelId,
            LFM2Resources.visionModelId
        ]
        for id in fixtures {
            let runtime = try NativeChatRuntime.command(modelID: id, modelPath: path)
            let selected: String
            let selectedPath: String?
            switch runtime {
            case .textChatPsi(_, let value): (selected, selectedPath) = (fixtures[0], value)
            case .textChatInkling(_, let value): (selected, selectedPath) = (fixtures[1], value)
            case .textChatDiffusionGemma(_, let value): (selected, selectedPath) = (fixtures[2], value)
            case .textChatGemma4(_, let value): (selected, selectedPath) = (fixtures[3], value)
            case .textChatLaguna(_, let value): (selected, selectedPath) = (fixtures[4], value)
            case .textChatQ35(_, let value): (selected, selectedPath) = (fixtures[5], value)
            case .textChatLFM2(_, let value): (selected, selectedPath) = (fixtures[6], value)
            default: XCTFail("Unexpected runtime for \(id)"); continue
            }
            XCTAssertEqual(selected, id)
            XCTAssertEqual(selectedPath, path)
        }
        XCTAssertThrowsError(try NativeChatRuntime.command(modelID: LagunaResources.modelID, modelPath: nil))
    }

    func testEmptyModelSelectionKeepsTheSelectedEngineDefault() throws {
        // Gemma4 claims an empty spec, so the substituted ID has to stay in its family.
        let runtime = try NativeChatRuntime.command(modelID: "", modelPath: nil)
        guard case .textChatGemma4(let generator, _) = runtime else {
            return XCTFail("An empty model spec should keep selecting the Gemma4 family")
        }
        XCTAssertEqual(generator.modelId, Gemma4Resources.defaultModelId)
    }

    func testExplicitOptionsPreserveToolMediaAndDiagnosticRequests() throws {
        let request = ChatRequest(
            messages: [.init(role: .user, content: "inspect", imageUrl: "/image.png", audioUrl: "/audio.wav")],
            maxTokens: 19, presencePenalty: 0.3, frequencyPenalty: -0.2, repetitionPenalty: 1.1,
            seed: 42, reasoningEffort: 0.4, lora: .local(path: "/adapter", scale: 0.7), requiresJSON: true,
            tools: [.init(name: "lookup", description: "Look up a value", parameters: [:])],
            toolChoice: .required, parallelToolCalls: true, stopSequences: ["STOP"],
            kvCacheMode: .affine8, maxContextTokens: 32, logprobCapture: .top(3), showUnmasking: true
        )
        var expected = request
        expected.temperature = 0
        expected.topP = 1
        expected.topK = 0
        expected.minP = 0.1
        expected.showThinking = false
        for policy in [ChatDefaultPolicy.command, .openAI] {
            let effective = try ChatRequestResolver.resolve(
                request, modelID: LagunaResources.modelID,
                sampling: .init(temperature: 0, topP: 1, topK: 0, minP: 0.1, thinking: true), policy: policy
            )
            XCTAssertEqual(effective, expected)
        }
    }

    func testInvalidRequestsNeverReachAnExecutor() async throws {
        let valid = ChatRequest(messages: [.init(role: .user, content: "hello")], maxTokens: 10, maxContextTokens: 20)
        let invalid: [(String, (inout ChatRequest) -> Void)] = [
            ("messages", { $0.messages = [] }), ("max_tokens", { $0.maxTokens = 0 }),
            ("max_tokens", { $0.maxTokens = 21 }), ("context_size", { $0.maxContextTokens = 0 }),
            ("temperature", { $0.temperature = .nan }), ("top_p", { $0.topP = 1.1 }),
            ("top_k", { $0.topK = -1 }), ("min_p", { $0.minP = .infinity }),
            ("presence_penalty", { $0.presencePenalty = 3 }),
            ("repetition_penalty", { $0.repetitionPenalty = .greatestFiniteMagnitude })
        ]
        for (field, change) in invalid {
            var request = valid
            change(&request)
            do {
                _ = try await ChatGenerationOperation.run(request) {
                    XCTFail("Invalid request reached generation")
                    return ChatResponse(response: "unexpected", tokensGenerated: 1)
                }
                XCTFail("Invalid request succeeded")
            } catch let issue as ChatRequestIssue {
                XCTAssertEqual(issue.field, field)
            }
        }
    }

    func testConversationRetainsRuntimeUntilAllResponsesAndAwaitsCleanup() async throws {
        let trace = ChatExecutionTrace()
        let request = ChatRequest(messages: [.init(role: .user, content: "hello")])
        let response = ChatResponse(response: "done", tokensGenerated: 2, toolCalls: [.init(name: "lookup", arguments: [:])])
        let result = try await ChatGenerationOperation.withCleanup {
            for turn in 1...2 {
                let actual = try await ChatGenerationOperation.run(request) {
                    await trace.append("turn-\(turn)")
                    return response
                }
                XCTAssertEqual(actual, response)
            }
            return response
        } cleanup: {
            await trace.append("released")
        }
        XCTAssertEqual(result, response)
        let events = await trace.events
        XCTAssertEqual(events, ["turn-1", "turn-2", "released"])
    }

    func testFailureAndCancellationBeforeOrAfterGenerationAwaitCleanup() async throws {
        for mode in 0...2 {
            let trace = ChatExecutionTrace()
            let task = Task {
                try await ChatGenerationOperation.withCleanup {
                    if mode == 0 { withUnsafeCurrentTask { $0?.cancel() } }
                    return try await ChatGenerationOperation.run(.init(messages: [.init(role: .user, content: "hello")])) {
                        await trace.append("generated")
                        if mode == 1 { throw ChatRequestIssue("fixture", "failed") }
                        withUnsafeCurrentTask { $0?.cancel() }
                        return ChatResponse(response: "must not succeed", tokensGenerated: 1)
                    }
                } cleanup: {
                    await trace.append("released")
                }
            }
            do {
                _ = try await task.value
                XCTFail("Interrupted operation succeeded")
            } catch {
                XCTAssertTrue(mode == 1 ? error is ChatRequestIssue : error is CancellationError)
            }
            let events = await trace.events
            XCTAssertEqual(events, mode == 0 ? ["released"] : ["generated", "released"])
        }
    }
}

private actor ChatExecutionTrace {
    var events: [String] = []
    func append(_ event: String) { events.append(event) }
}
