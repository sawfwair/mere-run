import Foundation
import XCTest
@testable import MereRunCore

final class R2DownloadRequestBuilderTests: XCTestCase {
    func testPublicFallbackUsesProvidedBaseURL() async throws {
        let result = try await R2DownloadRequestBuilder.makeGETRequest(
            key: "models/test.tar.gz",
            environment: [:],
            defaultPublicBaseURL: "https://example.com"
        )

        XCTAssertEqual(result.mode, .publicBaseURL)
        XCTAssertEqual(result.request.url?.absoluteString, "https://example.com/models/test.tar.gz")
        XCTAssertEqual(result.request.httpMethod, "GET")
    }

    func testExplicitModelSourceBaseURLUsesPublicMode() async throws {
        let result = try await R2DownloadRequestBuilder.makeGETRequest(
            key: "models/test.tar.gz",
            environment: [
                MereRunModelSourceConfiguration.baseURLEnvironmentKey: "https://models.example.com/"
            ]
        )

        XCTAssertEqual(result.mode, .publicBaseURL)
        XCTAssertEqual(result.request.url?.absoluteString, "https://models.example.com/models/test.tar.gz")
        XCTAssertEqual(result.request.httpMethod, "GET")
    }

    func testEnvCredentialsUseSignedMode() async throws {
        let result = try await R2DownloadRequestBuilder.makeGETRequest(
            key: "models/test.tar.gz",
            environment: [
                "MERERUN_R2_ACCOUNT_ID": "account",
                "MERERUN_R2_ACCESS_KEY_ID": "access",
                "MERERUN_R2_SECRET_ACCESS_KEY": "secret",
                "MERERUN_R2_BUCKET": "public"
            ],
            defaultPublicBaseURL: "https://example.com"
        )

        XCTAssertEqual(result.mode, .signed)
        XCTAssertEqual(
            result.request.url?.absoluteString,
            "https://account.r2.cloudflarestorage.com/public/models/test.tar.gz"
        )
        XCTAssertEqual(result.request.httpMethod, "GET")
        XCTAssertNotNil(result.request.value(forHTTPHeaderField: "Authorization"))
    }

    func testInvalidSignedEndpointFallsBackWhenNotRequired() async throws {
        let result = try await R2DownloadRequestBuilder.makeGETRequest(
            key: "models/test.tar.gz",
            environment: [
                "MERERUN_R2_SIGNED_URL_ENDPOINT": "not a valid url"
            ],
            defaultPublicBaseURL: "https://example.com"
        )

        XCTAssertEqual(result.mode, .publicBaseURL)
        XCTAssertEqual(result.request.url?.absoluteString, "https://example.com/models/test.tar.gz")
    }

    func testMissingConfigurationThrowsClearError() async {
        do {
            _ = try await R2DownloadRequestBuilder.makeGETRequest(
                key: "models/test.tar.gz",
                environment: [:]
            )
            XCTFail("Expected missing configuration error")
        } catch let error as R2DownloadRequestBuilder.BuildError {
            guard case .missingConfiguration(let message) = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
            XCTAssertTrue(message.contains("MERERUN_MODEL_SOURCE_BASE_URL"))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testInvalidSignedEndpointThrowsWhenRequired() async {
        do {
            _ = try await R2DownloadRequestBuilder.makeGETRequest(
                key: "models/test.tar.gz",
                environment: [
                    "MERERUN_R2_SIGNED_URL_ENDPOINT": "not a valid url",
                    "MERERUN_R2_SIGNED_URL_REQUIRED": "1"
                ],
                defaultPublicBaseURL: "https://example.com"
            )
            XCTFail("Expected invalid signed URL endpoint error")
        } catch let error as R2DownloadRequestBuilder.BuildError {
            guard case .invalidSignedURLEndpoint = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
}
