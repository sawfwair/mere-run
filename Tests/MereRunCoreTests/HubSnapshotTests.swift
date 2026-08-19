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

    func testRevisionKeysAreStableAndRevisionSpecific() {
        XCTAssertEqual(HubSnapshot.revisionKey("main"), HubSnapshot.revisionKey("main"))
        XCTAssertNotEqual(HubSnapshot.revisionKey("main"), HubSnapshot.revisionKey("feature"))
        XCTAssertEqual(HubSnapshot.revisionKey("main").count, 64)
    }

    func testMaterializedPatternClosureSupportsExactAndGlobSelections() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-hub-closure-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("cache", isDirectory: true),
            withIntermediateDirectories: true
        )
        try Data("config".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(
            to: root.appendingPathComponent("cache/adaln-9.safetensors")
        )

        XCTAssertTrue(HubSnapshot.containsAllMaterializedPatterns(
            at: root,
            patterns: ["config.json", "cache/*.safetensors"]
        ))
        XCTAssertFalse(HubSnapshot.containsAllMaterializedPatterns(
            at: root,
            patterns: ["config.json", "transformer.safetensors"]
        ))
        XCTAssertFalse(HubSnapshot.containsAllMaterializedPatterns(at: root, patterns: []))
    }

    func testNextPageLinkResolution() throws {
        let source = try XCTUnwrap(URL(string: "https://huggingface.co/api/models/org/repo/tree/main"))
        let header = "<https://huggingface.co/api/models/org/repo/tree/main?cursor=abc>; rel=\"next\", <https://example.test/end>; rel=\"last\""
        let next = HubSnapshot.nextPageURL(from: header, relativeTo: source)

        XCTAssertEqual(next?.absoluteString, "https://huggingface.co/api/models/org/repo/tree/main?cursor=abc")
        XCTAssertNil(HubSnapshot.nextPageURL(from: nil, relativeTo: source))
    }

    func testPayloadIdentityAcceptsHubLFSAndGitBlobETags() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-hub-etag-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let payload = root.appendingPathComponent("payload.bin")
        try Data("abc".utf8).write(to: payload)

        XCTAssertTrue(try HubSnapshot.payloadMatchesETag(
            at: payload,
            etag: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            byteCount: 3
        ))
        XCTAssertTrue(try HubSnapshot.payloadMatchesETag(
            at: payload,
            etag: "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f",
            byteCount: 3
        ))
    }

    func testPayloadIdentityRejectsMismatchedOrOpaqueETags() throws {
        let payload = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-hub-etag-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: payload) }
        try Data("abc".utf8).write(to: payload)

        XCTAssertFalse(try HubSnapshot.payloadMatchesETag(
            at: payload,
            etag: String(repeating: "0", count: 64),
            byteCount: 3
        ))
        XCTAssertFalse(try HubSnapshot.payloadMatchesETag(
            at: payload,
            etag: "opaque-etag",
            byteCount: 3
        ))
    }

    func testContentReferenceFallsBackToSymlinkWhenHardLinksAreUnavailable() throws {
        enum UnsupportedLink: Error { case unavailable }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-hub-reference-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("blob")
        let destination = root.appendingPathComponent("snapshot-payload")
        try Data("shared".utf8).write(to: source)
        try Data("stale".utf8).write(to: destination)

        try HubSnapshot.materializeContentReference(
            from: source,
            to: destination,
            hardLink: { _, _ in throw UnsupportedLink.unavailable }
        )

        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(
                atPath: destination.path
            )).standardizedFileURL,
            source.standardizedFileURL
        )
        XCTAssertEqual(try Data(contentsOf: destination), Data("shared".utf8))
        XCTAssertEqual(HubSnapshot.fileSize(at: destination), 6)
    }
}
