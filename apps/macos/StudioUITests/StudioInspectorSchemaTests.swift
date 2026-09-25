@testable import StudioKit
@testable import StudioUI
import XCTest

/// The inspector and the composer's chips bind the same draft fields; the schema counts and
/// resets what differs from a mode's defaults; the panel columns and the Library follow the task.
@MainActor
final class StudioInspectorSchemaTests: XCTestCase {
    func testEveryChipFieldIsAnInspectorFieldForTheModesThatShowIt() {
        for mode in StudioMode.allCases {
            // Both follow the model the draft runs, so read them from the same draft.
            let draft = StudioDraft.baseline(for: mode)
            let inspectorFields = StudioContractSchema.draftFieldIDs(for: mode, draft: draft, source: .contract)
            for chip in mode.composerChips(for: draft, source: .contract) {
                for field in chip.kind.draftFieldIDs(for: mode) {
                    XCTAssertTrue(inspectorFields.contains(field), "\(mode) chip \(chip) edits \(field), which its inspector does not show")
                }
            }
        }
    }

    func testInspectorFieldIDsAreUniquePerMode() {
        for mode in StudioMode.allCases {
            let ids = StudioInspectorSchema.sections(for: mode, source: .contract).flatMap { $0.fields.map(\.id) }
                + StudioInspectorSchema.advancedFields(for: mode, source: .contract).map(\.id)
            XCTAssertEqual(ids.count, Set(ids).count, "\(mode) lists a flag twice: \(ids)")
        }
    }

    func testChangedCountAndResetReadAgainstTheBaseline() {
        var baseline = StudioDraft()
        baseline.reset(for: .createImage)
        var draft = baseline
        XCTAssertEqual(StudioInspectorSchema.changedCount(mode: .createImage, draft: draft, baseline: baseline, source: .contract), 0)

        draft.cfgScale = 3.5
        draft.sigmaShift = 3.0
        XCTAssertEqual(StudioInspectorSchema.changedCount(mode: .createImage, draft: draft, baseline: baseline, source: .contract), 2)

        draft.width = 1344
        draft.height = 768
        draft.prompt = "prompts are the composer's, not the inspector's"
        XCTAssertEqual(StudioInspectorSchema.changedCount(mode: .createImage, draft: draft, baseline: baseline, source: .contract), 4)

        let output = StudioInspectorSchema.sections(for: .createImage, source: .contract).first { $0.group == .output }!
        XCTAssertEqual(output.changedCount(draft: draft, baseline: baseline), 2)
        output.reset(&draft, to: baseline)
        XCTAssertEqual(draft.width, 1024)
        XCTAssertEqual(draft.height, 1024)
        XCTAssertEqual(draft.cfgScale, 3.5, "Reset is per section")

        StudioInspectorSchema.resetAdvanced(for: .createImage, &draft, to: baseline, source: .contract)
        XCTAssertEqual(draft.sigmaShift, 0)
        XCTAssertEqual(StudioInspectorSchema.changedCount(mode: .createImage, draft: draft, baseline: baseline, source: .contract), 1)
    }

    func testAspectPresetsSelectAndSwap() {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        XCTAssertEqual(StudioAspectPreset.inspectorPresets(for: .createImage).map(\.label), ["1:1", "3:2", "16:9", "9:16"])
        XCTAssertEqual(StudioAspectPreset.inspectorSelection(for: draft, mode: .createImage)?.label, "1:1")

        StudioAspectPreset.inspectorPresets(for: .createImage)[2].apply(to: &draft)
        XCTAssertEqual([draft.width, draft.height], [1344, 768])
        StudioAspectPreset.swap(&draft)
        XCTAssertEqual([draft.width, draft.height], [768, 1344])
        XCTAssertEqual(StudioAspectPreset.inspectorSelection(for: draft, mode: .createImage)?.label, "9:16")

        draft.width = 1000
        XCTAssertNil(StudioAspectPreset.inspectorSelection(for: draft, mode: .createImage), "a custom size selects no segment")

        var video = StudioDraft()
        video.reset(for: .video)
        XCTAssertEqual(StudioAspectPreset.inspectorSelection(for: video, mode: .video)?.label, "3:2")
    }

    // MARK: Panel columns and the Library

    func testOnlyPromptTasksShowTheLibraryColumn() {
        for task in StudioTask.allCases {
            XCTAssertEqual(task.isPromptTask, task.mode != nil, "\(task)")
        }
        XCTAssertTrue(StudioTask.imageGenerate.isPromptTask)
        XCTAssertTrue(StudioTask.musicCompose.isPromptTask)
        XCTAssertTrue(StudioTask.chatChat.isPromptTask)
        XCTAssertTrue(StudioTask.audioTranscribe.isPromptTask)
        for task in [StudioTask.videoSubjects, .musicRealtime, .modelsInstalled, .imageTrain, .serverServing, .visionDepth] {
            XCTAssertFalse(task.isPromptTask, "\(task) takes the full width")
        }
    }

    func testInspectorIsRememberedPerTaskAndDisplacedByTheCommandColumn() {
        let navigation = NavigationModel()
        XCTAssertFalse(navigation.showsInspector(for: .imageGenerate))

        navigation.toggleInspector(for: .imageGenerate)
        XCTAssertTrue(navigation.showsInspector(for: .imageGenerate))
        XCTAssertFalse(navigation.showsInspector(for: .videoGenerate), "remembered per task")

        navigation.toggleCommandColumn()
        XCTAssertTrue(navigation.showCommandColumn)
        XCTAssertFalse(navigation.showsInspector(for: .imageGenerate), "never side by side")

        navigation.toggleInspector(for: .imageGenerate)
        XCTAssertFalse(navigation.showCommandColumn)
        XCTAssertTrue(navigation.showsInspector(for: .imageGenerate))

        navigation.toggleCommandColumn()
        navigation.open(task: .videoGenerate)
        XCTAssertFalse(navigation.showCommandColumn, "the Command view closes with its task")
        navigation.open(task: .imageGenerate)
        XCTAssertTrue(navigation.showsInspector(for: .imageGenerate), "the inspector memory survives the detour")

        navigation.toggleInspector(for: .videoSubjects)
        XCTAssertFalse(navigation.showsInspector(for: .videoSubjects), "only prompt tasks have an inspector")
        navigation.toggleCommandColumn()
        XCTAssertTrue(navigation.showCommandColumn, "specialist tasks expose the same Command panel")
    }

    func testInspectorTaskMemoryRoundTripsAsSceneStorageText() {
        let tasks: Set<StudioTask> = [.imageGenerate, .chatChat]
        let encoded = StudioInspectorTaskMemory.encode(tasks)
        XCTAssertEqual(encoded, "chat.chat,image.generate")
        XCTAssertEqual(StudioInspectorTaskMemory.decode(encoded), tasks)
        XCTAssertEqual(StudioInspectorTaskMemory.decode(""), [])
        XCTAssertEqual(StudioInspectorTaskMemory.decode("image.generate,not.a.task"), [.imageGenerate])
    }

    /// The scope note shows a folder glyph only for a folder the CLI could not identify; a model
    /// id or the default gets a plain question mark, and a note about unused values its own glyph.
    func testTheScopeNoteGlyphFollowsWhatTheNoteIsAbout() throws {
        let video = try XCTUnwrap(StudioContractSchema.capability(for: .video))
        let note = { (model: String) in
            StudioScopeNotice(
                scope: StudioOptionScope(
                    capability: video, arguments: ["x", "--model", model],
                    identities: StudioFixedModelIdentities([model: .unidentified])
                ),
                hidden: [], replaced: []
            )
        }
        let folder = try XCTUnwrap(note("/Volumes/Models/my-ltx"))
        XCTAssertEqual(folder.kind, .unidentified)
        XCTAssertEqual(StudioScopeNote.systemImage(for: folder), "questionmark.folder")
        let alias = try XCTUnwrap(note("Lightricks/LTX-2.3"))
        XCTAssertEqual(StudioScopeNote.systemImage(for: alias), "questionmark.circle", "an upstream id is not a folder")
        XCTAssertTrue(alias.title.hasSuffix("Lightricks/LTX-2.3"), alias.title)
        let id = try XCTUnwrap(note("video-ltx-av"))
        XCTAssertEqual(StudioScopeNote.systemImage(for: id), "questionmark.circle")
        XCTAssertEqual(StudioScopeNote.systemImage(for: StudioScopeNotice(kind: .unused, title: "Not used", details: [])), "eye.slash")
        XCTAssertNil(StudioScopeNote.systemImage(for: StudioScopeNotice(kind: .identifying, title: "Identifying", details: [])))
    }
}
