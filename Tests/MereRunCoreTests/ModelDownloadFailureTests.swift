import Foundation
import Hub
import XCTest
@testable import MereRunCore

final class ModelDownloadFailureTests: XCTestCase {
    func testNetworkInterruptionAndCancellationSurviveTheResolverBoundary() {
        let errors: [Error] = [URLError(.networkConnectionLost), URLError(.notConnectedToInternet), CancellationError(),
                              Hub.HubClientError.networkError(URLError(.timedOut)), Hub.HubClientError.httpStatusCode(503)]
        for error in errors {
            XCTAssertTrue(ManagedModelResolver.canResumeDownload(after: ManagedModelResolver.downloadFailure(error)))
        }
    }

    func testInvalidCredentialsCertificatesAndPayloadsDoNotAutomaticallyResume() {
        let errors: [Error] = [URLError(.serverCertificateUntrusted), Hub.HubClientError.authorizationRequired,
                              Hub.HubClientError.httpStatusCode(403), Hub.HubClientError.parse,
                              ManagedModelResolver.ResolverError.invalidInstalledModel("invalid checkpoint")]
        for error in errors {
            XCTAssertFalse(ManagedModelResolver.canResumeDownload(after: ManagedModelResolver.downloadFailure(error)))
        }
    }
}
