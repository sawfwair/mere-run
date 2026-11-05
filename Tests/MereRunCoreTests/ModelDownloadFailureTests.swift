import Foundation
import Hub
import XCTest
@testable import MereRunCore

final class ModelDownloadFailureTests: XCTestCase {
    func testNetworkInterruptionAndCancellationSurviveTheResolverBoundary() {
        let errors: [Error] = [URLError(.networkConnectionLost), URLError(.notConnectedToInternet), CancellationError(),
                              Hub.HubClientError.networkError(URLError(.timedOut)), Hub.HubClientError.httpStatusCode(503)]
        for error in errors {
            let failure = ManagedModelResolver.downloadFailure(error)
            guard case .downloadInterrupted(let message) = failure else {
                return XCTFail("Expected an interrupted resolver error for \(error), got \(failure)")
            }
            XCTAssertEqual(message, error.localizedDescription)
            XCTAssertTrue(ManagedModelResolver.canResumeDownload(after: failure))
        }
    }

    func testBackgroundTransferServerUnavailabilityResumesOnEveryPlatform() {
        XCTAssertTrue(ManagedModelResolver.canResumeDownload(after: HubBackgroundTransferError.httpStatus(503)))
        XCTAssertFalse(ManagedModelResolver.canResumeDownload(after: HubBackgroundTransferError.httpStatus(404)))
    }

    func testInvalidCredentialsCertificatesAndPayloadsDoNotAutomaticallyResume() {
        let errors: [Error] = [URLError(.serverCertificateUntrusted), Hub.HubClientError.authorizationRequired,
                              Hub.HubClientError.httpStatusCode(403), Hub.HubClientError.parse,
                              ManagedModelResolver.ResolverError.invalidInstalledModel("invalid checkpoint")]
        for error in errors {
            let failure = ManagedModelResolver.downloadFailure(error)
            guard case .downloadFailed = failure else {
                return XCTFail("Expected a permanent resolver error for \(error), got \(failure)")
            }
            XCTAssertFalse(ManagedModelResolver.canResumeDownload(after: failure))
        }
    }
}
