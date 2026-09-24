@testable import StudioKit
import XCTest

/// A finished task-draft row's card chips are the composer's own chips — the template's
/// essential options — valued from the recorded command, then the model, whatever mode files
/// the row. A template whose contract declares no tiers shows the model alone, on the card as in
/// the composer; marking an option essential in the contract adds it to both at once.
final class StudioTaskChipsTests: XCTestCase {
    /// `sfx generate` marks duration and seed essential and steps standard: the card carries the
    /// two the composer shows, valued from the argv, and not the third.
    func testEssentialOptionsBecomeChipsValuedFromTheRecordedCommand() throws {
        var draft = StudioTaskDraft(templateID: .sfxGenerate)
        draft.prompt = "footsteps on gravel"
        draft.form["--duration"] = .number(4)
        draft.form["--steps"] = .integer(25)
        draft.form["--seed"] = .integer(3)
        let chips = StudioTaskChips.chips(for: try Self.row(for: draft), titles: .none)
        XCTAssertTrue(chips.contains("Duration 4"), "\(chips)")
        XCTAssertTrue(chips.contains("Seed 3"), "\(chips)")
        XCTAssertFalse(chips.contains("Steps 25"), "steps is standard, not a chip: \(chips)")
        XCTAssertEqual(chips.last, StudioModelNaming.displayName(StudioTaskSchema.modelID(for: draft), titles: .none))
        XCTAssertFalse(chips.contains { $0.hasPrefix("1024") || $0.hasSuffix(" steps") || $0.hasPrefix("seed ") }, "no prompt-mode chips: \(chips)")
    }

    /// Foley's contract declares no tiers, so its composer has no option chips and neither does
    /// its card: the model alone, never Create Image's size, steps, and seed.
    func testAFoleyRowMirrorsItsComposer() throws {
        var draft = StudioTaskDraft(templateID: .sfxVideo)
        draft.prompt = "footsteps on gravel"
        StudioTaskSchema.slots(for: .sfxVideo)[0].attach([URL(fileURLWithPath: "/tmp/walk.mp4")], to: &draft)
        draft.form["--duration"] = .number(4)
        draft.form["--steps"] = .integer(25)
        draft.form["--seed"] = .integer(3)
        let expected = StudioTaskSchema.essentials(for: .soundFoley, draft: draft)
            .filter { $0.overrideID != .variant }
            .map { $0.chipTitle(in: draft) }
            + [StudioModelNaming.displayName(StudioTaskSchema.modelID(for: draft), titles: .none)]
        XCTAssertEqual(StudioTaskChips.chips(for: try Self.row(for: draft), titles: .none), expected)
    }

    func testAThreeDRowShowsItsEssentialsAndModel() throws {
        var draft = StudioTaskDraft(templateID: .imageReconstruct3D)
        draft.setArgument(0, "/tmp/chair.png")
        draft.form["--resolution"] = .integer(64)
        let chips = StudioTaskChips.chips(for: try Self.row(for: draft), titles: .none)
        let essentials = StudioTaskSchema.essentials(for: .threeDFromImage, draft: draft).filter { $0.overrideID != .variant }
        XCTAssertEqual(chips.dropLast().count, essentials.count, "one chip per essential option: \(chips)")
        if essentials.contains(where: { $0.flag == "--resolution" }) {
            XCTAssertTrue(chips.contains("Resolution 64"), "\(chips)")
        }
        XCTAssertEqual(chips.last, "TripoSR", "the model closes the row, named as the cards name it")
        XCTAssertFalse(chips.contains { $0.hasPrefix("Engine") }, "the card is headed by the engine already")
    }

    func testARowWithoutARecordedCommandHasNoChips() {
        let row = StudioLibraryItem(
            id: UUID(), mode: .createImage, prompt: "", inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "", outputText: nil, templateID: .imageReconstruct3D
        )
        XCTAssertEqual(StudioTaskChips.chips(for: row, titles: .none), [])
    }

    /// The Library row the runner would record for `draft`: its command draft and exact argv.
    private static func row(for draft: StudioTaskDraft) throws -> StudioLibraryItem {
        let request = try XCTUnwrap(StudioOutputLocation.destination(for: draft).request())
        return StudioLibraryItem(
            id: request.id, mode: request.mode, prompt: request.draft.prompt, inputURL: nil, outputURL: nil,
            createdAt: Date(), updatedAt: Date(), status: .completed, exitCode: 0, commandPreview: "", outputText: nil,
            templateID: request.templateID, commandDraft: request.draft, commandArguments: request.execution?.arguments
        )
    }
}
