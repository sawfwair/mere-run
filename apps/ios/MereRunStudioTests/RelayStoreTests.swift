import Foundation
import MereRunRelayKit
import XCTest
import mere_run

@MainActor
final class RelayStoreTests: XCTestCase {
    func testRepairingTheSameEndpointStillReplacesConnectionIdentity() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("client-scope-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = MemoryCredentialStorage()
        var count = 0
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in storage }, directAuthorization: { _, _, _ in
            count += 1
            return self.testToken("account-\(count)")
        })
        await store.pairDirect(urlString: "same.test:6373", code: "first")
        let identity = store.connectionID
        let profile = store.profile
        let old = try XCTUnwrap(store.client?.credentialStorage)
        await store.pairDirect(urlString: "same.test:6373", code: "second")

        XCTAssertEqual(store.profile, profile)
        XCTAssertNotEqual(store.connectionID, identity)
        XCTAssertThrowsError(try old.load())
        XCTAssertEqual(try store.client?.credentialStorage?.load(), testToken("account-2"))
    }

    func testFailedProfileReplacementDoesNotRetainTheNewEndpointsToken() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("client-scope-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = MemoryCredentialStorage()
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in storage },
                               directAuthorization: { url, _, _ in self.testToken(url) })
        await store.pairDirect(urlString: "first.test:6373", code: "first")
        let old = try XCTUnwrap(store.client?.credentialStorage)
        let profileURL = root.appendingPathComponent("executors.json")
        try FileManager.default.removeItem(at: profileURL)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: false)
        try Data("block replacement".utf8).write(to: profileURL.appendingPathComponent("blocker"))
        await store.pairDirect(urlString: "second.test:6373", code: "second")

        guard case .failed = store.pairing else { return XCTFail("profile replacement must fail") }
        XCTAssertNil(store.client)
        XCTAssertNil(store.profile, "a failed replacement must not keep presenting the retired endpoint")
        XCTAssertNil(try storage.load(), "an old profile must not be reopened with the new endpoint's token")
        XCTAssertThrowsError(try old.load())
    }

    func testOldClientCannotReadOrOverwriteCredentialsAfterRepairing() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("client-scope-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = MemoryCredentialStorage()
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in storage },
                               directAuthorization: { url, _, _ in self.testToken(url) })
        await store.pairDirect(urlString: "first.test:6373", code: "first")
        let old = try XCTUnwrap(store.client?.credentialStorage)
        await store.pairDirect(urlString: "second.test:6373", code: "second")
        let current = try storage.load()

        XCTAssertThrowsError(try old.load(), "old requests cannot read another connection's token")
        XCTAssertThrowsError(try old.save(testToken("late refresh")))
        XCTAssertThrowsError(try old.clear())
        XCTAssertEqual(try storage.load(), current)
        XCTAssertEqual(try store.client?.credentialStorage?.load(), current)
    }

    func testUnpairPreventsAnOldClientFromRestoringItsRefreshedCredentials() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("client-scope-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = MemoryCredentialStorage()
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in storage },
                               directAuthorization: { _, _, _ in self.testToken() })
        await store.pairDirect(urlString: "first.test:6373", code: "first")
        let old = try XCTUnwrap(store.client?.credentialStorage)
        store.unpair()

        XCTAssertThrowsError(try old.save(testToken("late refresh")))
        XCTAssertNil(try storage.load())
    }

    func testUnpairRejectsLateAuthorizationBeforeSavingCredentialsOrProfile() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("pairing-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = MemoryCredentialStorage()
        let gate = AuthorizationGate()
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in storage },
                               directAuthorization: { _, _, _ in try await gate.authorize() })
        let pairing = Task { await store.pairDirect(urlString: "first.test:6373", code: "test") }
        await gate.waitUntilStarted()
        store.unpair()
        gate.finish(.success(testToken()))
        await pairing.value

        XCTAssertEqual(store.pairing, .unpaired)
        XCTAssertNil(store.profile)
        XCTAssertNil(try storage.load())
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("executors.json").path))
    }

    func testOlderPairingCannotOverwriteTheNewerPairingOrItsCredentials() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("pairing-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = MemoryCredentialStorage()
        let gate = AuthorizationGate()
        let token = testToken()
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in storage }, directAuthorization: { url, _, _ in
            if url.contains("first.test") { return try await gate.authorize() }
            return token
        })
        let old = Task { await store.pairDirect(urlString: "first.test:6373", code: "old") }
        await gate.waitUntilStarted()
        await store.pairDirect(urlString: "second.test:6373", code: "new")
        gate.finish(.success(testToken("old")))
        await old.value

        XCTAssertEqual(store.profile?.url, "http://second.test:6373")
        XCTAssertEqual(try storage.load(), token)
        let profiles = try WorkflowExecutorProfileStore.load(from: root.appendingPathComponent("executors.json"))
        XCTAssertEqual(profiles.profiles.first?.url, "http://second.test:6373")
    }

    func testLateAuthorizationFailureDoesNotUndoUnpair() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("pairing-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = AuthorizationGate()
        let store = RelayStore(supportBase: root, credentialStorageFactory: { _ in MemoryCredentialStorage() },
                               directAuthorization: { _, _, _ in try await gate.authorize() })
        let pairing = Task { await store.pairDirect(urlString: "first.test:6373", code: "test") }
        await gate.waitUntilStarted()
        store.unpair()
        gate.finish(.failure(URLError(.cannotConnectToHost)))
        await pairing.value
        XCTAssertEqual(store.pairing, .unpaired)
    }

    private func testToken(_ token: String = "new") -> RelayOAuthTokenSet {
        RelayOAuthTokenSet(accessToken: token, refreshToken: nil, tokenType: "Bearer", expiresIn: nil,
                          obtainedAtEpochSeconds: nil)
    }

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

@MainActor
private final class AuthorizationGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<RelayOAuthTokenSet, Error>?

    func authorize() async throws -> RelayOAuthTokenSet {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        return try await withCheckedThrowingContinuation { waiter = $0 }
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func finish(_ result: Result<RelayOAuthTokenSet, Error>) {
        waiter?.resume(with: result)
        waiter = nil
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
