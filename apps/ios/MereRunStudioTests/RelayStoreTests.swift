import Foundation
import MereRunRelayKit
import XCTest
import mere_run

@MainActor
final class RelayStoreTests: XCTestCase {
    func testLegacyCredentialMigratesAndClientUsesInjectedStorage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("RelayStoreTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let profile = WorkflowExecutorProfile(
            name: "phone",
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: "https://relay.example.test",
            tokenFile: nil
        )
        try WorkflowExecutorProfileStore.save(
            WorkflowExecutorProfiles(schemaVersion: 1, profiles: [profile]),
            to: root.appendingPathComponent("executors.json")
        )
        let legacyURL = RelayAuthentication.defaultTokenFile(
            profileName: profile.name,
            applicationSupportBase: root
        )
        let token = RelayOAuthTokenSet(
            accessToken: "legacy-token",
            refreshToken: "refresh-token",
            tokenType: "Bearer",
            expiresIn: nil,
            obtainedAtEpochSeconds: nil
        )
        try FileCredentialStorage(url: legacyURL).save(token)
        let storage = MemoryCredentialStorage()

        let store = RelayStore(supportBase: root) { _ in storage }

        XCTAssertEqual(store.pairing, .paired)
        XCTAssertNil(store.profile?.tokenFile)
        XCTAssertEqual(try storage.load(), token)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyURL.path))
        XCTAssertEqual(try store.client?.credentialStorage?.load(), token)

        store.unpair()
        XCTAssertNil(try storage.load())
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("executors.json").path))
    }
}

private final class MemoryCredentialStorage: RelayCredentialStorage, @unchecked Sendable {
    private let lock = NSLock()
    private var tokenSet: RelayOAuthTokenSet?

    func load() throws -> RelayOAuthTokenSet? {
        lock.withLock { tokenSet }
    }

    func save(_ tokenSet: RelayOAuthTokenSet) throws {
        lock.withLock { self.tokenSet = tokenSet }
    }

    func clear() throws {
        lock.withLock { tokenSet = nil }
    }
}
