import Foundation
import MereRunContract
@testable import StudioKit
import StudioTestSupport
import XCTest

/// `catalog resolve` answers: whose they are, when Studio asks, and how long an answer stands.
@MainActor
final class StudioModelIdentityStoreTests: XCTestCase {
    private let video = MereRunCapabilityCatalog.videoGenerate

    /// M-e: each controller asks its own CLI. An answer one controller's resolver gives never
    /// scopes another controller's surfaces, and nothing reads a process-wide store.
    func testEachControllerKeepsItsOwnAnswers() async throws {
        let first = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        let second = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        XCTAssertFalse(first.modelIdentities === second.modelIdentities)
        let folder = "/tmp/model-scope/each-controller"
        first.modelIdentities.use { _ in
            MereRunFamilyResolutionReport(
                capability: "video.generate", family: "h3-ref2va", familyTitle: "MiniMax-H3 Ref2VA",
                model: folder, source: .identified, violations: [], warnings: []
            )
        }
        second.modelIdentities.use { _ in nil }
        let commandLine = ["video", "generate", "a lighthouse", "--model", folder]
        for _ in 0..<200 where first.scopeSource.scope(capability: video, commandLine: commandLine).family == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(first.scopeSource.scope(capability: video, commandLine: commandLine).family?.id, "h3-ref2va")
        XCTAssertNil(second.scopeSource.scope(capability: video, commandLine: commandLine).family)
    }
}
