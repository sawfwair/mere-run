import Foundation
import XCTest
@testable import StudioKit
@testable import StudioUI

/// "Train on it" on a discovered dataset points Image ▸ Train at the folder and opens it: into
/// the task draft's `--data` once Training is on the shared workspace, into the page's parked
/// `CommandDraft` while the page still owns the task.
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

        if StudioTask.imageTrain.usesTaskDraft {
            let draft = try XCTUnwrap(sessions.taskDraft(for: .imageTrain))
            XCTAssertEqual(draft.text("--data"), candidate.path)
        } else {
            let draft = try XCTUnwrap(sessions.value(for: StudioDatasetTrainingHandoff.legacyDraftKey, default: Optional<CommandDraft>.none))
            XCTAssertEqual(draft.inputPath, candidate.path)
            // An untouched page's defaults come along, so the handoff reads like the page's own fresh draft.
            XCTAssertEqual(draft.seed, "42")
            XCTAssertEqual(draft.checkpointInterval, 250)
            XCTAssertFalse(draft.outputPath.isBlank)
        }
    }
}
