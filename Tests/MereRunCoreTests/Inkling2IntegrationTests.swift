import Foundation
import Jinja
import XCTest
@testable import MereRunCore

final class Inkling2IntegrationTests: XCTestCase {
    func testManagedCheckpointAndNativeServingRouting() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Inkling2Resources.modelID))
        XCTAssertEqual(spec.validationKind, .inkling2)
        XCTAssertEqual(spec.upstreamRepoId, "thinkingmachines/Inkling-2-Small")
        XCTAssertEqual(spec.hubFallback?.revision, Inkling2Resources.revision)
        XCTAssertFalse(spec.hubFallback?.patterns.contains("mtp.safetensors") ?? true)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatInkling2)
        XCTAssertEqual(spec.apiProfile?.inputModalities, [.text, .image])
        XCTAssertEqual(spec.apiProfile?.reasoningEffortStrengths[.off], 0)
        XCTAssertFalse(spec.apiProfile?.structuredOutput ?? true)
        for name in [Inkling2Resources.modelID, Inkling2Resources.repository] {
            XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: name), .inkling2)
            guard case .textChatInkling2 = try NativeChatRuntime.command(modelID: name, modelPath: nil) else {
                return XCTFail("Expected native Inkling 2 runtime")
            }
        }
        let manifest = MereRunModelManifest.template(for: .inkling2Small)
        XCTAssertEqual(manifest.engine, .inkling2)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.sources?.first?.revision, Inkling2Resources.revision)
        XCTAssertTrue(manifest.supports?.contains(.visionChat) == true)
    }

    func testMiniManagedCheckpointAndNativeServingRouting() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Inkling2Resources.miniModelID))
        XCTAssertEqual(spec.upstreamRepoId, "Sawfwair/Inkling-2-Mini-Speed")
        XCTAssertEqual(spec.hubFallback?.revision, Inkling2Resources.miniRevision)
        let source = try XCTUnwrap(spec.hubFallback)
        XCTAssertTrue(source.patterns.contains("mtp.safetensors"))
        XCTAssertTrue(Set(Inkling2Resources.speedDraftFiles).isSubset(of: Set(source.patterns)))
        XCTAssertTrue(spec.mountedHubFallbacks.isEmpty)
        XCTAssertEqual(spec.validationKind, .inkling2)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatInkling2)
        XCTAssertEqual(spec.apiProfile?.inputModalities, [.text, .image])
        for name in [spec.id, Inkling2Resources.miniRepository] {
            XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: name), .inkling2Mini)
            guard case .textChatInkling2 = try NativeChatRuntime.command(modelID: name, modelPath: nil) else {
                return XCTFail("Expected native Inkling 2 runtime")
            }
        }
        let manifest = MereRunModelManifest.template(for: .inkling2Mini)
        XCTAssertEqual(manifest.engine, .inkling2)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 16)
        XCTAssertEqual(manifest.quantization?.scheme, "mixed-nvfp4-mxfp8-with-affine-q8-attention-and-head")
        XCTAssertEqual(manifest.sources?.first?.revision, Inkling2Resources.miniRevision)
    }

    func testMiniSpeedRequiresBothBundledDraftHeads() throws {
        let fileManager = FileManager.default
        let parent = fileManager.temporaryDirectory.appendingPathComponent("Inkling2MiniSpeed-\(UUID())")
        let snapshot = parent.appendingPathComponent("snapshot")
        let installation = parent.appendingPathComponent("install")
        try fileManager.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: parent) }
        let speed = try XCTUnwrap(ManagedModelCatalog.spec(for: Inkling2Resources.miniModelID))
        let official = try XCTUnwrap(ManagedModelCatalog.spec(for: "thinkingmachines/Inkling-2-Mini"))
        XCTAssertEqual(official.id, Inkling2Resources.miniOfficialModelID)
        XCTAssertEqual(official.upstreamRevision, "ed40e7da625f594fc9c4990f5f66be24873c22d7")
        let filenames = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                         "mtp.safetensors", "model-00001.safetensors"]
        for filename in filenames { try Data().write(to: snapshot.appendingPathComponent(filename)) }
        let index = #"{"weight_map":{"tensor":"model-00001.safetensors"}}"#
        try Data(index.utf8).write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertTrue(official.missingPaths(in: snapshot, fileManager: fileManager).isEmpty)
        XCTAssertEqual(speed.missingPaths(in: snapshot, fileManager: fileManager).count, 6)

        for filename in Inkling2Resources.speedDraftFiles {
            let file = snapshot.appendingPathComponent(filename)
            try fileManager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        _ = try ManagedModelResolver.materializeManagedInstallRoot(
            for: speed, snapshotURL: snapshot, modelDir: installation, fileManager: fileManager
        )
        XCTAssertTrue(speed.isManagedRuntimeReady(installation, fileManager: fileManager))
    }

    func testMiniQ8HeadUsesOnePinnedManagedCheckpoint() throws {
        let modelID = Inkling2Resources.miniQ8HeadModelID
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
        let source = try XCTUnwrap(spec.hubFallback)
        XCTAssertEqual(source.repoId, "Sawfwair/Inkling-2-Mini-Q8-Head")
        XCTAssertEqual(source.revision, "6e2909a5ed4956cec636e6517d365676086f90df")
        XCTAssertTrue(source.patterns.contains("mtp.safetensors"))
        XCTAssertTrue(source.patterns.contains("tiktoken/tokenizer.model"))
        XCTAssertTrue(spec.mountedHubFallbacks.isEmpty)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatInkling2)
        XCTAssertEqual(spec.apiProfile?.inputModalities, [.text, .image])
        XCTAssertEqual(ManagedModelCatalog.spec(for: source.repoId)?.id, modelID)
        let manifest = MereRunModelManifest.template(for: .inkling2MiniQ8Head)
        XCTAssertEqual(manifest.sources?.first?.revision, source.revision)
        XCTAssertEqual(manifest.quantization?.scheme, "mixed-nvfp4-mxfp8-with-affine-q8-head")
    }

    func testMiniSamplingDefaultsForCLIAndAPI() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("Inkling2Mini-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"inkling2_loop"}"#.utf8).write(to: root.appendingPathComponent("config.json"))

        let request = ChatRequest(messages: [ChatMessage(role: .user, content: "hello")])
        for policy in [ChatDefaultPolicy.command, .openAI] {
            for model in [Inkling2Resources.miniModelID, Inkling2Resources.miniRepository,
                          Inkling2Resources.miniQ8HeadModelID, Inkling2Resources.miniQ8HeadRepository, root.path] {
                let resolved = try ChatRequestResolver.resolve(request, modelID: model, sampling: .init(), policy: policy)
                XCTAssertEqual(resolved.temperature, 1)
                XCTAssertEqual(resolved.topP, 1)
                XCTAssertEqual(resolved.topK ?? 0, 0)
                XCTAssertEqual(resolved.minP, 0)
            }
            let explicit = ChatSamplingOptions(temperature: 0, topP: 0.8, topK: 40, minP: 0.05)
            let resolved = try ChatRequestResolver.resolve(request, modelID: Inkling2Resources.miniModelID,
                                                           sampling: explicit, policy: policy)
            XCTAssertEqual(resolved.temperature, 0)
            XCTAssertEqual(resolved.topP, 0.8)
            XCTAssertEqual(resolved.topK, 40)
            XCTAssertEqual(resolved.minP, 0.05)
        }
        let small = try ChatRequestResolver.resolve(request, modelID: Inkling2Resources.modelID, sampling: .init(), policy: .command)
        XCTAssertEqual(small.temperature, 0.7)
        XCTAssertEqual(small.topP, 0.9)
    }

    func testDSparkDefaultsToUnfilteredSamplingWithPackedExperts() throws {
        let chat = ChatRequest(messages: [ChatMessage(role: .user, content: "hello")])
        let defaults = try ChatRequestResolver.resolve(chat, modelID: Inkling2Resources.miniModelID,
                                                       sampling: .init(), policy: .command)
        func usesDraft(_ request: ChatRequest, packed: Bool = true, preference: String? = nil) -> Bool {
            Inkling2Generator.usesDSpark(request, modelType: "inkling2_loop", packedExperts: packed, preference: preference)
        }

        XCTAssertTrue(usesDraft(defaults))
        XCTAssertFalse(usesDraft(defaults, packed: false))
        XCTAssertFalse(usesDraft(defaults, preference: "0"))
        XCTAssertTrue(usesDraft(defaults, packed: false, preference: "1"))
        XCTAssertFalse(Inkling2Generator.usesDSpark(defaults, modelType: "inkling2", packedExperts: true, preference: "1"))

        let options = [
            ChatSamplingOptions(temperature: 0), ChatSamplingOptions(topP: 0.9),
            ChatSamplingOptions(topK: 20), ChatSamplingOptions(minP: 0.05),
        ]
        for sampling in options {
            let request = try ChatRequestResolver.resolve(chat, modelID: Inkling2Resources.miniModelID,
                                                          sampling: sampling, policy: .command)
            XCTAssertFalse(usesDraft(request))
            XCTAssertTrue(usesDraft(request, preference: "1"))
        }
        var incompatible = defaults
        incompatible.stopSequences = ["END"]
        XCTAssertFalse(usesDraft(incompatible))
        XCTAssertFalse(usesDraft(incompatible, preference: "1"))
    }

    private func render(
        _ messages: [ChatMessage], effort: Double = 0.9, tools: [ToolDefinition]? = nil, fixture: String = "chat_template.jinja"
    ) throws -> String {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Inkling2/\(fixture)")
        return try Inkling2ChatTemplate(source: String(contentsOf: url, encoding: .utf8))
            .render(messages: messages, tools: tools, effort: effort)
    }

    func testReleasedTemplateHasNoAssistantSuffixAndPreservesEffort() throws {
        XCTAssertEqual(try render([ChatMessage(role: .user, content: "Hello")], effort: 0.2),
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
            + "<|message_user|><|content_text|>Hello<|end_message|>")
    }

    func testMiniTemplatePreservesAssistantHistoryAndImageMarker() throws {
        let messages = [
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi"),
            ChatMessage(role: .user, content: "Describe", imageUrl: "/image.png"),
        ]
        let prompt = try render(messages, fixture: "mini-chat_template.jinja")
        XCTAssertEqual(prompt, try render(messages))
        XCTAssertTrue(prompt.contains("<|message_model|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>"))
        XCTAssertTrue(prompt.hasSuffix("<|message_user|><|content_image|><|unused_200054|><|end_message|>"
            + "<|message_user|><|content_text|>Describe<|end_message|>"))
    }

    func testReleasedTemplatePreservesTypedToolArgumentsAndImageOrder() throws {
        let call = ChatMessageToolCall(id: "c1", name: "read", arguments: ["path": .string("a.txt")])
        let rendered = try render([
            ChatMessage(role: .assistant, content: "", reasoningContent: "Inspect first.", toolCalls: [call]),
            ChatMessage(role: .tool, content: "data", toolCallID: "c1"),
            ChatMessage(role: .user, content: "Describe", imageUrl: "/image.png"),
        ])
        XCTAssertTrue(rendered.contains("<|message_model|><|content_thinking|>Inspect first.<|end_message|>"))
        XCTAssertTrue(rendered.contains("<|message_model|>read<|content_invoke_tool_json|>{\"name\":\"read\",\"args\":{\"path\":\"a.txt\"}}<|end_message|>"))
        XCTAssertTrue(rendered.contains("<|message_tool|>read<|content_text|>data<|end_message|>"))
        XCTAssertTrue(rendered.hasSuffix("<|message_user|><|content_image|><|unused_200054|><|end_message|>"
            + "<|message_user|><|content_text|>Describe<|end_message|>"))
        XCTAssertFalse(rendered.contains("<|content_text|><|end_message|>"))
    }

    func testToolDeclarationUsesCompactPythonJSONWithoutHTMLEscaping() throws {
        let tools = [ToolDefinition(name: "read", description: "été <&>", parameters: [:])]
        let prompt = try render([ChatMessage(role: .user, content: "Read")], tools: tools)
        XCTAssertTrue(prompt.hasPrefix("<|message_system|>tool_declare<|content_xml|>"))
        XCTAssertTrue(prompt.contains(#""description":"été <&>""#))
        XCTAssertTrue(prompt.contains(#""name":"read","parameters":{"properties":{},"required":[],"type":"object"},"type":"function""#))
        XCTAssertFalse(prompt.contains(#"\u00e9"#))
    }

    func testToolLikeTextInReasoningIsNeverExecutedAndCallsFollowPolicy() {
        let tool = ToolDefinition(name: "read", description: "Read", parameters: ["path": .init(type: "string", description: "Path")], required: ["path"])
        let invocation = #"<|content_invoke_tool_json|>{"name":"read","args":{"path":"a.txt"}}"#
        let text = "<|message_model|><|content_thinking|>Example: " + invocation + "<|end_message|>"
            + "<|message_model|>read" + invocation + "<|end_message|>"
            + "<|message_model|>read" + invocation + "<|end_message|>"
            + "<|message_model|><|content_text|>Done.<|end_message|>"
        let parsed = Inkling2OutputParser.parse(text, tools: [tool], parallel: false)
        XCTAssertEqual(parsed.visible, "Done.")
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertTrue(parsed.reasoning?.contains(invocation) == true)
        XCTAssertTrue(Inkling2OutputParser.parse(text).toolCalls.isEmpty)
    }

    func testStopSequencesMatchBeforeVisibleWhitespaceIsTrimmed() {
        let parsed = Inkling2OutputParser.parse("<|message_model|><|content_text|>Answer END <|end_message|>")
        let stopped = TextGenerationStopSequences.trimming(parsed.visible, sequences: ["END "])
        XCTAssertEqual(stopped.text, "Answer")
        XCTAssertEqual(stopped.matchedSequence, "END ")
    }

    func testUnsupportedControlsFailBeforeCheckpointLoading() async throws {
        let generator = Inkling2Generator()
        var request = ChatRequest(messages: [ChatMessage(role: .user, content: "hello")])
        request.kvCacheMode = .affine4
        do {
            _ = try await generator.chat(request, modelPath: "/missing", progressHandler: nil)
            XCTFail("Expected refusal")
        } catch { XCTAssertEqual((error as? ChatRequestIssue)?.field, "request") }
    }

    func testDefaultCacheModesReachCheckpointLoading() async throws {
        let generator = Inkling2Generator()
        for mode in [nil, RuntimeKVCacheMode.default, .auto] {
            var request = ChatRequest(messages: [ChatMessage(role: .user, content: "hello")])
            request.kvCacheMode = mode
            do {
                _ = try await generator.chat(request, modelPath: nil, progressHandler: nil)
                XCTFail("Expected missing checkpoint error")
            } catch { XCTAssertEqual((error as? ChatRequestIssue)?.field, "model") }
        }
    }
}
