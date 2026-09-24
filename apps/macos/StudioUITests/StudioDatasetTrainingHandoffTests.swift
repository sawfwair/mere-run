import Foundation
import XCTest
@testable import StudioKit
@testable import StudioUI

/// "Train on it" on a discovered dataset points Image ▸ Train's task draft at the folder — its
/// dataset well, `--data` — and opens the task.
@MainActor
final class StudioDatasetTrainingHandoffTests: XCTestCase {
    private let candidate = StudioDatasetDiscoveryDocument.Candidate(
        id: "portraits", name: "Portraits", path: "/tmp/datasets/portraits", status: "ok", trainable: true,
        images: 12, captions: 12, usablePairs: 12, diagnostics: []
    )

    func testTrainOnItPointsImageTrainAtTheFolderAndOpensIt() throws {
        let sessions = StudioTaskSessions()
        let navigation = NavigationModel()
        StudioDatasetTrainingHandoff.open(candidate, navigation: navigation, sessions: sessions)
        XCTAssertEqual(navigation.destination.task, .imageTrain)
        let draft = try XCTUnwrap(sessions.taskDraft(for: .imageTrain))
        XCTAssertEqual(draft.templateID, .imageTrainLoRA)
        XCTAssertEqual(draft.text("--data"), candidate.path)
        // A fresh draft comes with the page's own defaults, so the handoff reads like the page's.
        XCTAssertEqual(draft.text("--seed"), "42")
    }
}
