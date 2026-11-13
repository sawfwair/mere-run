@testable import StudioKit
@testable import StudioUI
import SwiftUI
import XCTest

@MainActor
final class StudioOperationSelectionTests: XCTestCase {
    func testDepthRecoverySwitchesOperationAndPreservesEachModelChoice() throws {
        var draft = StudioTaskDraft(templateID: .visionDepth)
        draft.model = "vision-depth-marigold-v2"
        draft.setArgument(0, "/tmp/photo.png")
        let binding = Binding(get: { draft }, set: { draft = $0 })
        let choices = StudioOperationSelection.task(.visionDepth, draft: binding, source: .contract)
        let operation = try XCTUnwrap(choices.first)
        XCTAssertTrue(operation.chipTitle.hasPrefix("Operation · "))
        operation.value.wrappedValue = CommandTemplateID.visionDepthVideo.rawValue
        draft.model = "vision-depth-vda-small-metric"
        let scope = StudioTaskSchema.modelScope(for: draft, source: .contract)
        XCTAssertTrue(scope.runnableModels?.contains("vision-depth-vda-small-metric") == true)
        XCTAssertFalse(scope.runnableModels?.contains("vision-depth-marigold-v2") == true)
        operation.value.wrappedValue = CommandTemplateID.visionDepth.rawValue
        XCTAssertEqual(draft.model, "vision-depth-marigold-v2")
        XCTAssertEqual(draft.argument(0), "/tmp/photo.png")
        operation.value.wrappedValue = CommandTemplateID.visionDepthVideo.rawValue
        XCTAssertEqual(draft.model, "vision-depth-vda-small-metric")
    }

    func testReadImageRecoveryAndComposerUseTheSameOperationBinding() throws {
        var draft = StudioDraft()
        let binding = Binding(get: { draft }, set: { draft = $0 })
        let choices = StudioOperationSelection.prompt(.readImage, draft: binding, source: .contract)
        let operation = try XCTUnwrap(choices.first { $0.label == "Operation" })
        XCTAssertEqual(operation.choices.map(\.id), StudioReadImageAction.allCases.map(\.rawValue))
        operation.value.wrappedValue = StudioReadImageAction.caption.rawValue
        XCTAssertEqual(draft.readImageAction, .caption)
        XCTAssertEqual(operation.selectedTitle, "Caption")
    }

    func testVoiceModeAndTranscriptionBackendAreLabeledOperationSelectors() {
        for mode in [StudioMode.speak, .listen] {
            var draft = StudioDraft()
            draft.reset(for: mode)
            let binding = Binding(get: { draft }, set: { draft = $0 })
            let selectors = StudioOperationSelection.prompt(mode, draft: binding, source: .contract)
            XCTAssertFalse(selectors.isEmpty, "\(mode)")
            XCTAssertFalse(selectors.contains { $0.id == "--model" })
            for selector in selectors { XCTAssertTrue(selector.chipTitle.hasPrefix(selector.label + " · ")) }
            if mode == .speak {
                XCTAssertEqual(selectors.first { $0.id == "--mode" }?.choices.map(\.id), ["style", "clone"])
            }
        }
    }
}
