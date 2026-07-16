import Foundation
import XCTest
@testable import MereRunCLI

final class WorkflowRelayExecutorTests: XCTestCase {
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
}
