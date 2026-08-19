import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import MereRunRelayKit

final class WorkflowRelayExecutorTests: XCTestCase {
    func testExecutorResolvesInjectedCredentialStorageWithoutTokenFile() async throws {
        let storage = FixedCredentialStorage(
            tokenSet: RelayOAuthTokenSet(
                accessToken: "keychain-bearer",
                refreshToken: nil,
                tokenType: "Bearer",
                expiresIn: nil,
                obtainedAtEpochSeconds: nil
            )
        )
        let executor = RelayWorkflowExecutor(
            profile: relayProfile(tokenFile: nil),
            credentialStorage: storage
        )

        let credential = try await executor.resolveCredential()

        XCTAssertEqual(credential.accessToken, "keychain-bearer")
        XCTAssertFalse(credential.refreshable)
    }

    func testTransientServerFailuresRetryUntilSuccess() async throws {
        var attempts = 0
        let result = try await RelayWorkflowExecutor.performWithTransientRetries(
            maximumAttempts: 3,
            delayNanoseconds: 0
        ) {
            attempts += 1
            return (Data(), response(statusCode: attempts < 3 ? 500 : 200))
        }

        XCTAssertEqual(attempts, 3)
        XCTAssertEqual(result.response.statusCode, 200)
    }

    func testClientFailureDoesNotRetry() async throws {
        var attempts = 0
        let result = try await RelayWorkflowExecutor.performWithTransientRetries(
            maximumAttempts: 3,
            delayNanoseconds: 0
        ) {
            attempts += 1
            return (Data(), response(statusCode: 409))
        }

        XCTAssertEqual(attempts, 1)
        XCTAssertEqual(result.response.statusCode, 409)
    }

    func testTransientTransportFailureRetries() async throws {
        var attempts = 0
        let result = try await RelayWorkflowExecutor.performWithTransientRetries(
            maximumAttempts: 2,
            delayNanoseconds: 0
        ) {
            attempts += 1
            if attempts == 1 {
                throw URLError(.networkConnectionLost)
            }
            return (Data(), response(statusCode: 200))
        }

        XCTAssertEqual(attempts, 2)
        XCTAssertEqual(result.response.statusCode, 200)
    }

    private func response(statusCode: Int) -> HTTPURLResponse {
        HTTPURLResponse(
            url: URL(string: "https://relay.example.test")!,
            statusCode: statusCode,
            httpVersion: nil,
            headerFields: nil
        )!
    }

    private func relayProfile(tokenFile: String?) -> WorkflowExecutorProfile {
        WorkflowExecutorProfile(
            name: "phone",
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: "https://relay.example.test",
            tokenFile: tokenFile
        )
    }
}

private struct FixedCredentialStorage: RelayCredentialStorage {
    let tokenSet: RelayOAuthTokenSet

    func load() throws -> RelayOAuthTokenSet? {
        tokenSet
    }

    func save(_ tokenSet: RelayOAuthTokenSet) throws {}

    func clear() throws {}
}
