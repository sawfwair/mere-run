import Crypto
import Foundation
import XCTest
@testable import MereRunCLI

final class PluginBundleTests: XCTestCase {
    private struct InstalledPluginManifest: Decodable {
        let name: String
        let version: String
    }

    private let key = Curve25519.Signing.PrivateKey()
    private var keys: [String: String] { ["test": key.publicKey.rawRepresentation.base64EncodedString()] }

    func testPublisherSignatureBindsExactPayloadAndRejectsUnknownKey() throws {
        let manifest = makeManifest()
        let signed = try envelope(manifest)
        XCTAssertEqual(try signed.verified(trustedKeys: keys), manifest)
        XCTAssertThrowsError(try signed.verified())
        let tampered = PluginBundleEnvelope(contractVersion: signed.contractVersion, keyID: "test",
                                           payload: Data("{}".utf8).base64EncodedString(), signature: signed.signature)
        XCTAssertThrowsError(try tampered.verified(trustedKeys: keys))
        let badSignature = PluginBundleEnvelope(contractVersion: signed.contractVersion, keyID: "test",
                                               payload: signed.payload, signature: Data(repeating: 0, count: 64).base64EncodedString())
        XCTAssertThrowsError(try badSignature.verified(trustedKeys: keys))
    }

    func testManifestRejectsExpiryAndUnsafePathsEvenWithValidSignature() throws {
        for manifest in [makeManifest(package: "../escape"), makeManifest(app: "../Escape.app"),
                         makeManifest(entrypoint: "../tool"), makeManifest(platform: "linux-x86_64"),
                         makeManifest(package: "mere-workflow-tools\n"), makeManifest(entrypoint: "mere-doc-tools\n"),
                         makeManifest(expiry: "2099-01-01T00:00:00Z\n"),
                         makeManifest(artifactURL: "http://example.com/bundle.dmg"), makeManifest(size: -1),
                         makeManifest(sequence: 0)] {
            XCTAssertThrowsError(try envelope(manifest).verified(trustedKeys: keys))
        }
        let expired = try envelope(makeManifest(expiry: "2000-01-01T00:00:00Z"))
        XCTAssertThrowsError(try expired.verified(trustedKeys: keys))
        XCTAssertNoThrow(try expired.verified(trustedKeys: keys, now: nil))
    }

    func testArtifactTamperingAndIdentityMismatchNeverStageCode() throws {
        let root = try temporary()
        let archive = root.appendingPathComponent("test.dmg")
        try Data("original".utf8).write(to: archive)
        let store = testStore(root)
        let manifest = makeManifest(hash: try PluginBundleIO.hash(archive), size: 8)
        let data = try JSONEncoder().encode(envelope(manifest))
        XCTAssertThrowsError(try store.install(envelopeData: data, archive: archive, package: "mere-other", pluginID: "mere-doc-tools"))
        try Data("modified".utf8).write(to: archive)
        XCTAssertThrowsError(try store.install(envelopeData: data, archive: archive, package: manifest.package,
                                               pluginID: "mere-doc-tools", stage: { _, _, _ in XCTFail("Must not stage tampered code") }))
        XCTAssertNil(try store.state(manifest.package))
    }

    func testActivationRollbackAndReplayProtection() throws {
        let root = try temporary()
        let store = testStore(root)
        let first = try installFixture(store, sequence: 1)
        let original = try XCTUnwrap(store.resolve("mere-doc-tools"))
        XCTAssertTrue(original.path.contains(first.artifact.sha256))
        XCTAssertEqual(try store.entrypoints(), ["mere-doc-tools"])
        XCTAssertEqual(try store.package(containing: "mere-doc-tools"), first.package)
        XCTAssertNil(try store.package(containing: "mere-absent-tools"))
        let second = try installFixture(store, sequence: 2)
        XCTAssertEqual(try store.state(first.package)?.previous, first.artifact.sha256)
        XCTAssertThrowsError(try installFixture(store, sequence: 1))
        XCTAssertThrowsError(try installFixture(store, sequence: 2, content: "conflict"))
        XCTAssertEqual(try store.rollback(package: first.package), first)
        XCTAssertEqual(try store.resolve("mere-doc-tools"), original)
        XCTAssertEqual(try store.state(first.package)?.highestSequence, 2)
        XCTAssertEqual(try store.rollback(package: first.package), second)
    }

    func testFailedRelocationCheckPreservesActiveInstallation() throws {
        let root = try temporary()
        var store = testStore(root)
        let first = try installFixture(store, sequence: 1)
        let before = try store.state(first.package)
        store.smoke = { url, _ in
            if url.path.contains("/versions/") { throw CocoaError(.fileReadCorruptFile) }
        }
        XCTAssertThrowsError(try installFixture(store, sequence: 2))
        XCTAssertEqual(try store.state(first.package), before)
    }

    func testFailedPlatformVerificationPreservesActiveInstallation() throws {
        let root = try temporary()
        var store = testStore(root)
        let first = try installFixture(store, sequence: 1)
        let before = try store.state(first.package)
        store.verifyPlatform = { _, _ in throw CocoaError(.fileReadNoPermission) }
        XCTAssertThrowsError(try installFixture(store, sequence: 2))
        XCTAssertEqual(try store.state(first.package), before)
        XCTAssertThrowsError(try store.rollback(package: first.package))
    }

    func testBundleTreeRejectsSymlinkEscape() throws {
        let root = try temporary()
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("escape"),
                                                   withDestinationURL: URL(fileURLWithPath: "/tmp"))
        XCTAssertThrowsError(try PluginBundleIO.validateTree(root))
    }

    func testBundleTreeAllowsOnlyRelativeLinksWithInternalTargets() throws {
        let root = try temporary()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Frameworks"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Resources"), withIntermediateDirectories: true)
        try Data("library".utf8).write(to: root.appendingPathComponent("Frameworks/runtime"))
        let link = root.appendingPathComponent("Resources/runtime")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: "../Frameworks/runtime")
        XCTAssertNoThrow(try PluginBundleIO.validateTree(root))
        try FileManager.default.removeItem(at: link)
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: "../../outside")
        XCTAssertThrowsError(try PluginBundleIO.validateTree(root))
        try FileManager.default.removeItem(at: link)
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: "../Frameworks/missing")
        XCTAssertThrowsError(try PluginBundleIO.validateTree(root))
    }

    func testConcurrentInstallCannotEnterStaging() throws {
        let root = try temporary()
        let store = testStore(root)
        let archive = root.appendingPathComponent("fixture.dmg")
        try Data("fixture".utf8).write(to: archive)
        let manifest = makeManifest(hash: try PluginBundleIO.hash(archive), size: 7)
        let data = try JSONEncoder().encode(envelope(manifest))
        _ = try store.install(envelopeData: data, archive: archive, package: manifest.package,
                              pluginID: "mere-doc-tools", stage: { _, _, app in
            XCTAssertThrowsError(try store.install(envelopeData: data, archive: archive, package: manifest.package,
                                                   pluginID: "mere-doc-tools", stage: { _, _, _ in XCTFail("Lock not held") }))
            try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
        })
    }

    func testCorruptManagedReceiptDoesNotResolveAnExecutable() throws {
        let root = try temporary()
        let store = testStore(root)
        let manifest = try installFixture(store, sequence: 1)
        let receipt = root.appendingPathComponent("store/packages/\(manifest.package)/versions/\(manifest.artifact.sha256)/release.json")
        try Data("{}".utf8).write(to: receipt)
        XCTAssertThrowsError(try store.resolve("mere-doc-tools"))
        XCTAssertThrowsError(try installFixture(store, sequence: 1))
    }

    func testRetainedArtifactCannotBeReboundToDifferentReleaseMetadata() throws {
        let root = try temporary()
        let store = testStore(root)
        let first = try installFixture(store, sequence: 1)
        XCTAssertEqual(try installFixture(store, sequence: 1), first)
        let before = try store.state(first.package)
        XCTAssertThrowsError(try installFixture(store, sequence: 2, content: "fixture-1"))
        XCTAssertEqual(try store.state(first.package), before)
    }

    func testPlatformChecksRequireNoDeveloperToolsAndStopOnFailure() throws {
        #if os(macOS)
        let app = URL(fileURLWithPath: "/tmp/Fixture.app")
        var commands: [String] = []
        try PluginBundleIO.verifyMacOS(app, manifest: makeManifest()) { executable, _ in
            commands.append(executable)
            return Data()
        }
        XCTAssertEqual(commands, ["/usr/bin/codesign", "/usr/bin/syspolicy_check", "/usr/sbin/spctl"])
        commands = []
        XCTAssertThrowsError(try PluginBundleIO.verifyMacOS(app, manifest: makeManifest()) { executable, _ in
            commands.append(executable)
            throw CocoaError(.fileReadNoPermission)
        })
        XCTAssertEqual(commands, ["/usr/bin/codesign"])
        #endif
    }

    func testBundleOptionsRetainExplicitSourceAndOfflineInputs() throws {
        let command = try PluginInstall.parse(["mere-doc-tools", "--bundle-manifest", "/tmp/release.json",
                                              "--bundle-archive", "/tmp/plugin bundle.dmg"])
        XCTAssertFalse(command.source)
        XCTAssertTrue(command.confirmationCommand(channel: "main").contains("--bundle-archive '/tmp/plugin bundle.dmg'"))
        let source = try PluginInstall.parse(["mere-doc-tools", "--source"])
        XCTAssertTrue(source.source)
        let run = try PluginRun.parse(["mere-doc-tools", "process", "--input", "document.csv"])
        XCTAssertEqual(run.arguments, ["process", "--input", "document.csv"])
    }

    func testRealSignedBundleInstallsAndRunsAfterRelocation() throws {
        guard let directory = ProcessInfo.processInfo.environment["MERERUN_BUNDLE_TEST_ARTIFACT"] else {
            throw XCTSkip("Set MERERUN_BUNDLE_TEST_ARTIFACT to a signed bundle directory for the real macOS acceptance test.")
        }
        let fixture = URL(fileURLWithPath: directory)
        let data = try Data(contentsOf: fixture.appendingPathComponent("release.json"))
        let manifest = try JSONDecoder().decode(PluginBundleEnvelope.self, from: data).verified()
        let requested = ProcessInfo.processInfo.environment["MERERUN_BUNDLE_TEST_PLUGIN"]
        let pluginID = try XCTUnwrap(requested ?? (manifest.entrypoints["mere-doc-tools"] == nil
            ? manifest.entrypoints.keys.sorted().first : "mere-doc-tools"))
        XCTAssertNotNil(manifest.entrypoints[pluginID])
        let root = try temporary().appendingPathComponent("Relocated plugin with spaces")
        let store = PluginBundleStore(root: root)
        _ = try store.install(envelopeData: data, archive: fixture.appendingPathComponent("bundle.dmg"),
                              package: manifest.package, pluginID: pluginID)
        let executable = try XCTUnwrap(store.resolve(pluginID))
        let installed = try JSONDecoder().decode(InstalledPluginManifest.self,
            from: PluginBundleIO.capture(executable.path, ["manifest", "--json"]))
        XCTAssertEqual(installed.name, pluginID)
        XCTAssertEqual(installed.version, manifest.version)
        guard pluginID == "mere-doc-tools" else { return }
        let input = root.appendingPathComponent("source.csv")
        try Data("item,count\nNotebook,3\nPencil,4\n".utf8).write(to: input)
        _ = try PluginBundleIO.capture(executable.path, ["process", "--extractor", "anydoc", "--no-redact",
                                                        "--input", input.path, "--output-dir", root.appendingPathComponent("output").path])
        let outputs = FileManager.default.enumerator(at: root.appendingPathComponent("output"), includingPropertiesForKeys: nil)
        let markdown = (outputs?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "md" }
        XCTAssertFalse(markdown.isEmpty)
        XCTAssertTrue(try String(contentsOf: XCTUnwrap(markdown.first), encoding: .utf8).contains("Notebook"))
    }

    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mere-bundle-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func testStore(_ root: URL) -> PluginBundleStore {
        PluginBundleStore(root: root.appendingPathComponent("store"), trustedKeys: keys,
                          verifyPlatform: { _, _ in }, smoke: { _, _ in })
    }

    private func installFixture(_ store: PluginBundleStore, sequence: Int, content: String? = nil) throws -> PluginBundleManifest {
        let archive = store.root.deletingLastPathComponent().appendingPathComponent("fixture-\(UUID().uuidString)")
        let bytes = Data((content ?? "fixture-\(sequence)").utf8)
        try bytes.write(to: archive)
        let manifest = makeManifest(sequence: sequence, hash: try PluginBundleIO.hash(archive), size: Int64(bytes.count))
        return try store.install(envelopeData: JSONEncoder().encode(envelope(manifest)), archive: archive,
                                 package: manifest.package, pluginID: "mere-doc-tools", stage: { _, _, app in
            try FileManager.default.createDirectory(at: app.appendingPathComponent("Contents/MacOS"), withIntermediateDirectories: true)
            try Data("fixture".utf8).write(to: app.appendingPathComponent("Contents/MacOS/mere-doc-tools"))
        })
    }

    private func envelope(_ manifest: PluginBundleManifest) throws -> PluginBundleEnvelope {
        let payload = try JSONEncoder().encode(manifest)
        return PluginBundleEnvelope(contractVersion: "mere.run/plugin-bundle-envelope.v1", keyID: "test",
                                    payload: payload.base64EncodedString(), signature: try key.signature(for: payload).base64EncodedString())
    }

    private func makeManifest(
        package: String = "mere-workflow-tools", sequence: Int = 1, platform: String = "macos-arm64",
        expiry: String = "2099-01-01T00:00:00Z", app: String = "MereWorkflowTools.app", entrypoint: String = "mere-doc-tools",
        artifactURL: String = "https://example.com/plugin.dmg", hash: String = String(repeating: "a", count: 64), size: Int64 = 1
    ) -> PluginBundleManifest {
        PluginBundleManifest(contractVersion: "mere.run/plugin-bundle.v1", package: package, version: "0.4.0",
                             sequence: sequence, sourceCommit: String(repeating: "b", count: 40), platform: platform,
                             minimumOSVersion: "15.0", expiresAt: expiry, appBundle: app,
                             entrypoints: [entrypoint: entrypoint], artifact: .init(url: artifactURL, sha256: hash, size: size))
    }
}
