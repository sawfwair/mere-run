import Foundation
@testable import MereRunCore
import XCTest

@MainActor
final class MuseGlimmerPrefixCacheTests: XCTestCase {
    func testStrictToolCacheBoundariesUsePreparedJSONSchema() async throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_GLIMMER_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_GLIMMER_TOKENIZER_ROOT to run the native tokenizer cache-boundary fixture")
        }
        let config = try JSONDecoder().decode(MuseGlimmerGenerationConfig.self, from: Data("{}".utf8))
        let tokenizer = try await MuseGlimmerTokenizerAndTemplate.load(from: URL(fileURLWithPath: path), generationConfig: config)
        let generator = MuseGlimmerGenerator()
        await generator.installCacheFixture(tokenizer)
        let schema: [String: ChatJSONValue] = ["type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["value": .object(["type": .string("string")])]), "required": .array([.string("value")])]
        let tools = [ToolDefinition(name: "first", description: "first", parameterSchema: schema),
                     ToolDefinition(name: "second", description: "second", parameterSchema: schema)]
        var request = ChatRequest(messages: [.init(role: .user, content: "record a literal")], tools: tools, toolChoice: .required)
        let ordinaryIdentity = await generator.prefixIdentity(request)
        request.outputConstraint = try .tools(tools, choice: .required, parallel: true)
        let strictIdentity = await generator.prefixIdentity(request)
        XCTAssertNotEqual(ordinaryIdentity, strictIdentity)
        request.prefixCachePolicy = .init(scope: "fixture", boundaries: [], wholePromptTTL: nil,
            toolBoundaries: [.init(throughTool: "first", ttlSeconds: 300), .init(throughTool: "second", ttlSeconds: 3600)])
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.dateFormat = "yyyy-MM-dd"
        let tokens = try tokenizer.encodeForGeneration(messages: MuseGlimmerTokenizerAndTemplate.requestMessages(request),
            tools: NativeChatToolPolicy.templateTools(request), reasoningStrength: MuseGlimmerGenerator.reasoningStrength(request.reasoningEffort),
            currentDate: formatter.string(from: Date()), maxLength: Int.max, truncate: false)
        let checkpoints = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertEqual(checkpoints.count, 2)
        let firstBoundary = try XCTUnwrap(checkpoints.first(where: { $0.value == 300 })?.key)
        let secondBoundary = try XCTUnwrap(checkpoints.first(where: { $0.value == 3600 })?.key)
        XCTAssertLessThan(firstBoundary, secondBoundary)
        XCTAssertTrue(checkpoints.keys.allSatisfy { $0 < tokens.count })
        var changed = request
        changed.tools = [ToolDefinition(name: "first", description: "changed description", parameterSchema: schema), tools[1]]
        changed.outputConstraint = try .tools(changed.tools!, choice: .required, parallel: true)
        let changedTokens = try tokenizer.encodeForGeneration(messages: MuseGlimmerTokenizerAndTemplate.requestMessages(changed),
            tools: NativeChatToolPolicy.templateTools(changed), reasoningStrength: MuseGlimmerGenerator.reasoningStrength(changed.reasoningEffort),
            currentDate: formatter.string(from: Date()), maxLength: Int.max, truncate: false)
        XCTAssertNotEqual(tokens, changedTokens)
    }

    func testCoveringMarkersIncludeImageStateWithExpandedBoundaries() async throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_GLIMMER_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_GLIMMER_TOKENIZER_ROOT to run the native tokenizer cache-boundary fixture")
        }
        let config = try JSONDecoder().decode(MuseGlimmerGenerationConfig.self, from: Data("{}".utf8))
        let tokenizer = try await MuseGlimmerTokenizerAndTemplate.load(from: URL(fileURLWithPath: path), generationConfig: config)
        let generator = MuseGlimmerGenerator()
        await generator.installCacheFixture(tokenizer)
        var request = ChatRequest(messages: [
            ChatMessage(role: .system, blocks: [.text("Stable instructions")]),
            ChatMessage(role: .user, blocks: [.text("Inspect this image."), .image(reference: "fixture", identity: "one"), .text("Answer.")]),
        ], maxTokens: 64)
        request.prefixCachePolicy = .init(scope: "private", boundaries: [
            .init(messageIndex: 0, blockCount: 1, ttlSeconds: 300),
            .init(messageIndex: 1, blockCount: 1, ttlSeconds: 300),
            .init(messageIndex: 1, blockCount: 3, ttlSeconds: 300),
        ], wholePromptTTL: 300)
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd"
        let raw = try tokenizer.encodeForGeneration(messages: MuseGlimmerTokenizerAndTemplate.requestMessages(request),
            reasoningStrength: MuseGlimmerGenerator.reasoningStrength(request.reasoningEffort),
            currentDate: formatter.string(from: Date()), maxLength: Int.max, truncate: false)
        let tokens = try MuseGlimmerImageProcessor.expandedPromptTokens(raw, tokenCounts: [4],
            imageTokenId: tokenizer.imageTokenId, imageStartTokenId: tokenizer.imageStartTokenId,
            imageEndTokenId: tokenizer.imageEndTokenId)
        let firstImage = try XCTUnwrap(tokens.firstIndex(of: tokenizer.imageStartTokenId))
        let checkpoints = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertFalse(checkpoints.isEmpty)
        XCTAssertTrue(checkpoints.keys.contains { $0 > firstImage })
        let identity = await generator.prefixIdentity(request)
        request.messages[1] = ChatMessage(role: .user, blocks: [
            .text("Inspect this image."), .image(reference: "changed", identity: "two"), .text("Answer."),
        ])
        let changedImage = await generator.prefixIdentity(request)
        XCTAssertEqual(identity, changedImage, "The cached state contains no image-derived values")
        request.prefixCachePolicy = .init(scope: "private", boundaries: [
            .init(messageIndex: 1, blockCount: 3, ttlSeconds: 300),
        ], wholePromptTTL: 300)
        let afterImage = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertEqual(afterImage.count, 1)
        XCTAssertTrue(afterImage.keys.allSatisfy { $0 > firstImage })
        XCTAssertEqual(Array(afterImage.values), [300])
        request.prefixCachePolicy = .init(scope: "private", boundaries: [
            .init(messageIndex: 1, blockCount: 3, ttlSeconds: 3600),
        ], wholePromptTTL: 300)
        let longerTTL = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertEqual(Set(longerTTL.keys), Set(afterImage.keys))
        XCTAssertEqual(Array(longerTTL.values), [3600])
        request.prefixCachePolicy = .init(scope: "private", boundaries: [
            .init(messageIndex: 0, blockCount: 1, ttlSeconds: 300),
        ], wholePromptTTL: nil)
        let movedEarlier = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertFalse(movedEarlier.isEmpty)
        XCTAssertTrue(movedEarlier.keys.allSatisfy { $0 < firstImage })
        request.prefixCachePolicy = .init(scope: "private", boundaries: [], wholePromptTTL: nil)
        let unmarked = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertTrue(unmarked.isEmpty)
        request.prefixCachePolicy = .init(scope: "another-user", boundaries: [], wholePromptTTL: nil)
        let changedScope = await generator.prefixIdentity(request)
        XCTAssertNotEqual(identity, changedScope)
    }
}

private extension MuseGlimmerGenerator {
    func installCacheFixture(_ tokenizer: MuseGlimmerTokenizerAndTemplate) {
        tokenizerAndTemplate = tokenizer
        loadedModelPath = "fixture-artifact"
    }
}
