import XCTest
@testable import MereRunCore

final class HubSnapshotTests: XCTestCase {
    func testHubSnapshotPatternMatchingPreservesNestedPaths() {
        XCTAssertTrue(HubSnapshot.matchesPath(
            "tokenizer/added_tokens.json",
            patterns: ["tokenizer/*"]
        ))
        XCTAssertTrue(HubSnapshot.matchesPath(
            "tokenizer/added_tokens.json",
            patterns: ["tokenizer/added_tokens.json"]
        ))
        XCTAssertTrue(HubSnapshot.matchesPath(
            "Qwen3-Coder-Next-Q4_K_M/subdir/model.gguf",
            patterns: ["Qwen3-Coder-Next-Q4_K_M/*"]
        ))
        XCTAssertTrue(HubSnapshot.matchesPath(
            "model-q4.gguf",
            patterns: ["*.gguf"]
        ))

        XCTAssertFalse(HubSnapshot.matchesPath(
            "added_tokens.json",
            patterns: ["tokenizer/*"]
        ))
        XCTAssertFalse(HubSnapshot.matchesPath(
            "tokenizer_config.json",
            patterns: ["tokenizer/*"]
        ))
    }

    func testRedirectResolutionDoesNotDoubleEncodeNestedHubPath() throws {
        let source = try XCTUnwrap(URL(string: "https://huggingface.co/org/repo/resolve/main/tokenizer/added_tokens.json"))
        let location = "/api/resolve-cache/models/org/repo/commit/tokenizer%2Fadded_tokens.json?etag=%22abc%22"
        let redirected = try XCTUnwrap(HubSnapshot.redirectURL(from: location, relativeTo: source))

        XCTAssertTrue(redirected.absoluteString.contains("tokenizer%2Fadded_tokens.json"))
        XCTAssertFalse(redirected.absoluteString.contains("tokenizer%252Fadded_tokens.json"))
    }
}
