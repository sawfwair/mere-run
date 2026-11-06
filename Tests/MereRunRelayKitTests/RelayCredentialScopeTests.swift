import Foundation
import MereRunRelayKit
import XCTest

final class RelayCredentialScopeTests: XCTestCase {
    func testActiveScopePersistsRefreshedCredentials() throws {
        let storage = ScopedTestCredentials()
        let scope = RelayCredentialScope(storage: storage)
        let token = token("refreshed")
        try scope.save(token)
        XCTAssertEqual(try scope.load(), token)
        XCTAssertEqual(try storage.load(), token)
        try scope.clear()
        XCTAssertNil(try storage.load())
    }

    func testInvalidationPreventsOldCredentialReadsWritesAndDeletion() throws {
        let storage = ScopedTestCredentials()
        let old = RelayCredentialScope(storage: storage)
        try old.save(token("old"))
        old.invalidate()
        let replacement = RelayCredentialScope(storage: storage)
        try replacement.save(token("new"))

        XCTAssertThrowsError(try old.load())
        XCTAssertThrowsError(try old.save(token("late refresh")))
        XCTAssertThrowsError(try old.clear())
        XCTAssertEqual(try replacement.load(), token("new"))
    }

    private func token(_ accessToken: String) -> RelayOAuthTokenSet {
        RelayOAuthTokenSet(accessToken: accessToken, refreshToken: nil, tokenType: "Bearer", expiresIn: nil,
                          obtainedAtEpochSeconds: nil)
    }
}

private final class ScopedTestCredentials: RelayCredentialStorage, @unchecked Sendable {
    private let lock = NSLock()
    private var token: RelayOAuthTokenSet?
    func load() throws -> RelayOAuthTokenSet? { lock.withLock { token } }
    func save(_ tokenSet: RelayOAuthTokenSet) throws { lock.withLock { token = tokenSet } }
    func clear() throws { lock.withLock { token = nil } }
}
