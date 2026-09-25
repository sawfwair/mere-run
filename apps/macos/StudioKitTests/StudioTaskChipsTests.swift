@testable import StudioKit
import XCTest

/// A finished task-draft row's card chips are the composer's own chips — the template's
/// essential options — valued from the recorded command, then the model, whatever mode files
/// the row. Marking an option essential in the contract adds it to both surfaces at once.
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

    /// Foley's duration and steps appear in both places; its seed remains in the inspector.
    func testAFoleyRowMirrorsItsComposer() throws {
        var draft = StudioTaskDraft(templateID: .sfxVideo)
        draft.prompt = "footsteps on gravel"
        StudioTaskSchema.slots(for: .sfxVideo)[0].attach([URL(fileURLWithPath: "/tmp/walk.mp4")], to: &draft)
        draft.form["--duration"] = .number(4)
        draft.form["--steps"] = .integer(25)
        draft.form["--seed"] = .integer(3)
        XCTAssertEqual(StudioTaskSchema.essentials(for: .soundFoley, draft: draft).map(\.flag), ["--duration", "--steps"])
        let expected = StudioTaskSchema.essentials(for: .soundFoley, draft: draft)
            .filter { $0.overrideID != .variant }
            .map { $0.chipTitle(in: draft) }
            + [StudioModelNaming.displayName(StudioTaskSchema.modelID(for: draft), titles: .none)]
        XCTAssertEqual(StudioTaskChips.chips(for: try Self.row(for: draft), titles: .none), expected)
        XCTAssertTrue(expected.contains("Duration 4"))
        XCTAssertTrue(expected.contains("Steps 25"))
        XCTAssertFalse(expected.contains("Seed 3"))
    }

    func testAThreeDRowShowsItsEssentialsAndModel() throws {
        var draft = StudioTaskDraft(templateID: .imageReconstruct3D)
        draft.setArgument(0, "/tmp/chair.png")
        draft.form["--resolution"] = .integer(64)
        let chips = StudioTaskChips.chips(for: try Self.row(for: draft), titles: .none)
        let essentials = StudioTaskSchema.essentials(for: .threeDFromImage, draft: draft).filter { $0.overrideID != .variant }
        XCTAssertEqual(essentials.map(\.flag), ["--resolution"])
        XCTAssertEqual(chips.dropLast().count, essentials.count, "one chip per essential option: \(chips)")
        XCTAssertTrue(chips.contains("Resolution 64"), "\(chips)")
        XCTAssertEqual(chips.last, "TripoSR", "the model closes the row, named as the cards name it")
        XCTAssertFalse(chips.contains { $0.hasPrefix("Engine") }, "the card is headed by the engine already")

        var trellis = StudioTaskDraft(templateID: .imageReconstruct3DTrellis2)
        trellis.setArgument(0, "/tmp/chair.png")
        trellis.form["--seed"] = .integer(7)
        XCTAssertEqual(StudioTaskSchema.essentials(for: .threeDFromImage, draft: trellis).map(\.flag), ["task-variant", "--seed"])
        XCTAssertTrue(StudioTaskChips.chips(for: try Self.row(for: trellis), titles: .none).contains("Seed 7"))

        let multiview = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
        XCTAssertEqual(StudioTaskSchema.essentials(for: .threeDFromImage, draft: multiview).map(\.flag), ["task-variant", "--resolution"])
    }

    func testMusicAnalyzeAndTranscribeShowTheirEssentialSettingsOnCards() throws {
        var analyze = StudioTaskDraft(templateID: .musicAnalyze)
        analyze.setArgument(0, "/tmp/song.wav")
        analyze.form["--duration"] = .number(45)
        XCTAssertEqual(StudioTaskSchema.essentials(for: .musicAnalyze, draft: analyze).map(\.flag), ["--duration"])
        XCTAssertTrue(StudioTaskChips.chips(for: try Self.row(for: analyze), titles: .none).contains("Duration 45"))

        // `--variant` only sizes a local checkpoint without a config; a managed MuScriptor model
        // ignores it, so it is an expert setting rather than a card chip.
        var transcribe = StudioTaskDraft(templateID: .musicTranscribe)
        transcribe.setArgument(0, "/tmp/song.wav")
        transcribe.form["--variant"] = .text("large")
        transcribe.form["--format"] = .text("json")
        XCTAssertEqual(StudioTaskSchema.essentials(for: .musicTranscribe, draft: transcribe).map(\.flag), ["--format"])
        let chips = StudioTaskChips.chips(for: try Self.row(for: transcribe), titles: .none)
        XCTAssertFalse(chips.contains { $0.hasPrefix("Variant ") }, "\(chips)")
        XCTAssertTrue(chips.contains { $0.hasPrefix("Format ") }, "\(chips)")
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
