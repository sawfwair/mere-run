import XCTest
@testable import StudioKit

final class StudioLayaCommandTests: XCTestCase {
    func testDecisionTemplateUsesSharedContractAndOrderedRequestFile() throws {
        var draft = CommandDraft()
        draft.inputPath = "/tmp/questions.json"
        draft.model = "text-decide-laya-multilingual"
        draft.outputPath = "/tmp/answers.json"
        draft.force = true
        draft.preflight = true
        XCTAssertEqual(CommandArguments.textDecide(draft), [
            "text", "decide",
            "--input", "/tmp/questions.json", "--model", "text-decide-laya-multilingual",
            "--output", "/tmp/answers.json", "--pretty", "--preflight"
        ])
        XCTAssertEqual(CommandTemplateID.textDecide.capabilityID, "text.decide")
        XCTAssertEqual(CommandTemplateID.textDecide.studioTask, .textDecide)
    }
}
