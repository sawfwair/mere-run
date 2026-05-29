import XCTest
@testable import MereRunCore

final class ManagedModelResolverTests: XCTestCase {
    func testMaterializedInstallRootsKeepSharedHubAliasesManifestIsolated() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let snapshot = root.appendingPathComponent("hub/models/google/gemma-4-31B-it", isDirectory: true)
        try writeMinimalGemma4Snapshot(at: snapshot)
        try MereRunModelManifest.template(for: .gemma4, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: snapshot)

        let defaultSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-gemma4"))
        let maxSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-gemma4-max"))
        let defaultInstall = root.appendingPathComponent("models/text-chat-gemma4", isDirectory: true)
        let maxInstall = root.appendingPathComponent("models/text-chat-gemma4-max", isDirectory: true)

        let defaultManifest = try ManagedModelResolver.materializeManagedInstallRoot(
            for: defaultSpec,
            snapshotURL: snapshot,
            modelDir: defaultInstall,
            fileManager: .default
        )
        let maxManifest = try ManagedModelResolver.materializeManagedInstallRoot(
            for: maxSpec,
            snapshotURL: snapshot,
            modelDir: maxInstall,
            fileManager: .default
        )

        XCTAssertEqual(defaultManifest?.id, "text-chat-gemma4")
        XCTAssertEqual(maxManifest?.id, "text-chat-gemma4-max")
        XCTAssertEqual(try MereRunModelManifest.loadRequired(from: snapshot).id, "text-chat-gemma4")
        XCTAssertTrue(ManagedModelResolver.isManagedInstallComplete(spec: defaultSpec, at: defaultInstall))
        XCTAssertTrue(ManagedModelResolver.isManagedInstallComplete(spec: maxSpec, at: maxInstall))

        let defaultTokenizer = defaultInstall.appendingPathComponent("tokenizer.json")
        let maxTokenizer = maxInstall.appendingPathComponent("tokenizer.json")
        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(atPath: defaultTokenizer.path))
                .standardizedFileURL.path,
            snapshot.appendingPathComponent("tokenizer.json").standardizedFileURL.path
        )
        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(atPath: maxTokenizer.path))
                .standardizedFileURL.path,
            snapshot.appendingPathComponent("tokenizer.json").standardizedFileURL.path
        )
    }

    func testManagedInstallCompletenessRejectsMismatchedManifest() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        try writeMinimalGemma4Snapshot(at: root)
        try MereRunModelManifest.template(for: .gemma4, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: root)

        let maxSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-gemma4-max"))
        XCTAssertFalse(ManagedModelResolver.isManagedInstallComplete(spec: maxSpec, at: root))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-managed-model-resolver-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeMinimalGemma4Snapshot(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for file in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data("{}".utf8)
            ))
        }
    }
}
