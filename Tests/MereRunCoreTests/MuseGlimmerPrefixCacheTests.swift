import Foundation
@testable import MereRunCore
import XCTest

@MainActor
final class MuseGlimmerPrefixCacheTests: XCTestCase {
    func testCoveringMarkersCacheOnlyTextBeforeTheFirstImage() async throws {
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
        XCTAssertTrue(checkpoints.keys.allSatisfy { $0 <= firstImage })
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
        XCTAssertEqual(afterImage, [firstImage: 300], "A covering marker retains only the preceding text state")
        request.prefixCachePolicy = .init(scope: "private", boundaries: [
            .init(messageIndex: 1, blockCount: 3, ttlSeconds: 3600),
        ], wholePromptTTL: 300)
        let longerTTL = try await generator.prefixCheckpoints(request, tokens: tokens)
        XCTAssertEqual(longerTTL, [firstImage: 3600])
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
