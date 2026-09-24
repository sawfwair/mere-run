@testable import StudioKit
import MereRunContract
import UniformTypeIdentifiers
import XCTest

/// The specialist tasks' surface is read from the contract, and their canonical draft is the
/// Command view's form. These tests hold what makes that safe: the argv a task draft builds is
/// the argv the Command view shows for the same form, the routing-owned and launcher-owned flags
/// never reach a control, every well slot is something the contract declares, and switching a
/// task's variant keeps what the two templates share.
final class StudioTaskSchemaTests: XCTestCase {
    /// Every task the plan moves onto the shared workspace, with its templates.
    private var migratingTasks: [StudioTask] {
        StudioTask.allCases.filter { $0.mode == nil && [.generate, .analyze].contains($0.archetype) }
    }

    // MARK: Identity

    func testEveryTaskDraftBuildsTheCommandViewsArgv() throws {
        for task in migratingTasks {
            XCTAssertFalse(task.variantTemplates.isEmpty, "\(task) has no template to run")
            for template in task.variantTemplates {
                let capability = try XCTUnwrap(template.id.capability, "\(template.id) has no contract")
                let draft = StudioTaskDraft(templateID: template.id)
                XCTAssertEqual(draft.arguments, StudioConsoleCommand.arguments(for: capability, draft: draft.form), "\(template.id)")
                XCTAssertEqual(draft.run?.arguments, draft.arguments, "\(template.id) Command view argv")
                XCTAssertEqual(
                    draft.arguments,
                    StudioConsoleCommand.arguments(
                        for: capability,
                        draft: StudioConsoleCommand.seed(template: template, draft: template.defaultDraft())
                    ),
                    "\(template.id) fresh draft is the template's own command"
                )
                let request = try XCTUnwrap(draft.request(), "\(template.id) request")
                XCTAssertEqual(request.mode, template.libraryMode, "attribution comes from the template")
                XCTAssertEqual(request.execution?.arguments, draft.arguments)
            }
        }
    }

    // MARK: Fields

    func testHiddenFlagsNeverReachTheComposerOrInspector() throws {
        for task in migratingTasks {
            for template in task.variantTemplates {
                let capability = try XCTUnwrap(template.id.capability)
                let draft = StudioTaskDraft(templateID: template.id)
                let hidden = StudioTaskSchema.hiddenFlags(for: capability)
                let shown = StudioTaskSchema.sections(for: task, draft: draft).flatMap(\.fields)
                    + StudioTaskSchema.advanced(for: task, draft: draft)
                    + StudioTaskSchema.essentials(for: task, draft: draft)
                for field in shown {
                    XCTAssertFalse(hidden.contains(field.flag), "\(template.id) shows hidden \(field.flag)")
                    for binding in field.bindings where binding.fieldID.hasPrefix("--") {
                        XCTAssertFalse(hidden.contains(binding.fieldID), "\(template.id) binds hidden \(binding.fieldID)")
                    }
                }
                let slotFlags = StudioTaskSchema.slots(for: template.id).compactMap { slot -> String? in
                    if case .flag(let flag) = slot.storage { return flag }
                    if case .flagList(let flag) = slot.storage { return flag }
                    return nil
                }
                for flag in slotFlags {
                    XCTAssertFalse(shown.contains { $0.flag == flag }, "\(template.id) repeats slot \(flag) in the inspector")
                }
                if let output = capability.output.flag, capability.options.contains(where: { $0.flag == output }) {
                    XCTAssertTrue(hidden.contains(output), "\(template.id) shows its destination \(output)")
                }
            }
        }
    }

    func testEverySlotIsADeclaredInputOfItsTemplate() throws {
        for task in migratingTasks {
            for template in task.variantTemplates {
                let capability = try XCTUnwrap(template.id.capability)
                let outputs = StudioTaskSchema.outputFlags(for: capability)
                for slot in StudioTaskSchema.slots(for: template.id) {
                    XCTAssertFalse(slot.acceptedTypes.isEmpty, "\(template.id).\(slot.id) accepts nothing")
                    switch slot.storage {
                    case .argument(let index), .argumentList(let index):
                        XCTAssertLessThan(index, capability.arguments.count, "\(template.id).\(slot.id)")
                        XCTAssertTrue([.file, .directory].contains(capability.arguments[index].kind))
                    case .flag(let flag), .flagList(let flag):
                        let option = try XCTUnwrap(capability.options.first { $0.flag == flag }, "\(template.id) \(flag)")
                        XCTAssertTrue([.file, .directory].contains(option.kind))
                        XCTAssertFalse(outputs.contains(flag), "\(template.id) offers output \(flag) as an input")
                    case .path, .pathList:
                        XCTFail("\(template.id).\(slot.id) binds a prompt-draft field")
                    }
                }
            }
        }
    }

    func testSlotsFollowTheContractsInputs() {
        XCTAssertEqual(StudioTaskSchema.slots(for: .visionFlow).map(\.id), ["from", "to"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .visionFlow).map(\.acceptedTypes), [[.image], [.image]])
        XCTAssertEqual(StudioTaskSchema.slots(for: .visionFaceBatch).map(\.id), ["images", "--input-list"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .visionFaceBatch).first?.allowsMultiple, true)
        XCTAssertEqual(StudioTaskSchema.slots(for: .visionGeometryMultiview).map(\.id), ["images"],
                       "--cameras is a composite editor, not a well slot")
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageDatasetDiscover).map(\.id), ["--root"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageDatasetDiscover).first?.acceptedTypes, [.folder])
        XCTAssertEqual(StudioTaskSchema.slots(for: .audioEnhance).map(\.id), ["audio"], "--model-path is the model's, not an input")
        XCTAssertEqual(StudioTaskSchema.slots(for: .audioEnhance).first?.acceptedTypes, [.audio])
        XCTAssertEqual(StudioTaskSchema.slots(for: .textEmbed), [])
        XCTAssertEqual(StudioTaskSchema.slots(for: .sfxClapScore).map(\.id), ["audio"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3DMultiview).map(\.id), ["--view"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3DMultiview).first?.storage, .flagList("--view"))
    }

    func testSlotsWriteTheTaskDraft() {
        var draft = StudioTaskDraft(templateID: .visionFlow)
        let slots = StudioTaskSchema.slots(for: .visionFlow)
        slots[0].attach([URL(fileURLWithPath: "/tmp/a.png")], to: &draft)
        slots[1].attach([URL(fileURLWithPath: "/tmp/b.png")], to: &draft)
        XCTAssertEqual(draft.form.arguments, ["/tmp/a.png", "/tmp/b.png"])
        XCTAssertEqual(draft.primaryInputPath, "/tmp/a.png")
        XCTAssertEqual(slots[1].caption(in: draft), "b.png")
        slots[0].clear(in: &draft)
        XCTAssertEqual(slots[0].paths(in: draft), [])
        XCTAssertEqual(draft.form.arguments, ["", "/tmp/b.png"], "the second positional keeps its place")

        var batch = StudioTaskDraft(templateID: .visionFaceBatch)
        let images = StudioTaskSchema.slots(for: .visionFaceBatch)[0]
        images.attach([URL(fileURLWithPath: "/tmp/1.png"), URL(fileURLWithPath: "/tmp/2.png")], to: &batch)
        images.attach([URL(fileURLWithPath: "/tmp/2.png"), URL(fileURLWithPath: "/tmp/3.png")], to: &batch)
        XCTAssertEqual(batch.form.arguments, ["/tmp/1.png", "/tmp/2.png", "/tmp/3.png"], "a list appends without repeats")
        XCTAssertEqual(images.caption(in: batch), "1.png +2")
        XCTAssertTrue(batch.attach(dropped: [URL(fileURLWithPath: "/tmp/list.txt")], slots: batch.slots))
        XCTAssertEqual(batch.text("--input-list"), "/tmp/list.txt")
        XCTAssertFalse(batch.attach(dropped: [URL(fileURLWithPath: "/tmp/clip.mp4")], slots: batch.slots))

        var views = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
        StudioTaskSchema.slots(for: .imageReconstruct3DMultiview)[0]
            .attach([URL(fileURLWithPath: "/tmp/f.png"), URL(fileURLWithPath: "/tmp/b.png")], to: &views)
        XCTAssertEqual(views.text("--view"), "/tmp/f.png\n/tmp/b.png")
        XCTAssertEqual(views.arguments.filter { $0 == "--view" }.count, 2, "a repeatable option emits once per line")
    }

    // MARK: Prompt and variant

    func testPromptFieldsFindTheFreeText() throws {
        let embed = try XCTUnwrap(CommandTemplateID.textEmbed.capability)
        XCTAssertEqual(StudioTaskSchema.promptField(for: embed), .argument(0, repeatable: true))
        var draft = StudioTaskDraft(templateID: .textEmbed)
        draft.prompt = "first\n\nsecond"
        XCTAssertEqual(draft.form.arguments, ["first", "second"], "one text per line, blanks dropped")
        XCTAssertEqual(draft.prompt, "first\nsecond")

        let clap = try XCTUnwrap(CommandTemplateID.sfxClapScore.capability)
        XCTAssertEqual(StudioTaskSchema.promptField(for: clap), .argument(0, repeatable: false))
        var score = StudioTaskDraft(templateID: .sfxClapScore)
        score.prompt = "a dog barking"
        StudioTaskSchema.slots(for: .sfxClapScore)[0].attach([URL(fileURLWithPath: "/tmp/bark.wav")], to: &score)
        XCTAssertEqual(Array(score.arguments.prefix(5)), ["sfx", "clap", "score", "a dog barking", "/tmp/bark.wav"])

        let foley = try XCTUnwrap(CommandTemplateID.sfxVideo.capability)
        XCTAssertEqual(StudioTaskSchema.promptField(for: foley), .argument(0, repeatable: false), "a positional prompt wins over --prompt")
        let depth = try XCTUnwrap(CommandTemplateID.visionDepth.capability)
        XCTAssertNil(StudioTaskSchema.promptField(for: depth))
    }

    func testVariantsSwitchTemplatesAndCarryWhatTheyShare() throws {
        let faces = try XCTUnwrap(StudioTaskSchema.variantField(for: .visionFaces))
        XCTAssertEqual(faces.option.choices, StudioTask.visionFaces.variantTemplates.map(\.id.rawValue))
        XCTAssertEqual(faces.overrideID, .variant)
        XCTAssertEqual(faces.tier, .essential)
        XCTAssertNil(StudioTaskSchema.variantField(for: .audioWhoSpoke))
        XCTAssertEqual(StudioTaskSchema.variantTitle(CommandTemplateID.visionFaceCompare.rawValue), "Compare faces")

        var draft = StudioTaskDraft(templateID: .visionFaceDetect)
        draft.setArgument(0, "/tmp/portrait.png")
        draft.form["--score-threshold"] = .text("0.4")
        draft.form["--model"] = .text("vision-face-buffalo-l")
        faces.write(.text(CommandTemplateID.visionFaceEmbed.rawValue), to: &draft)
        XCTAssertEqual(draft.templateID, .visionFaceEmbed)
        XCTAssertEqual(draft.argument(0), "/tmp/portrait.png", "the picture carries to the sibling command")
        XCTAssertEqual(draft.text("--score-threshold"), "0.4", "a shared option carries")
        XCTAssertEqual(draft.text("--model"), "vision-face-buffalo-l", "the same default model stays")

        var depth = StudioTaskDraft(templateID: .visionDepth)
        depth.form["--model"] = .text("vision-depth-marigold-v2")
        depth.switchTemplate(to: .visionDepthVideo)
        XCTAssertEqual(depth.text("--model"), "", "a different default model is not carried")
        XCTAssertEqual(StudioTaskSchema.fields(for: .visionDepth, draft: depth).first?.overrideID, .variant, "the variant leads")
    }

    // MARK: Sections and scope

    func testSectionsFollowTheContractsGroupsAndTiers() {
        let draft = StudioTaskDraft(templateID: .audioEnhance)
        let sections = StudioTaskSchema.sections(for: .audioEnhance, draft: draft)
        XCTAssertEqual(sections.map(\.group), sections.map(\.group).sorted { lhs, rhs in
            StudioContractGroup.allCases.firstIndex(of: lhs)! < StudioContractGroup.allCases.firstIndex(of: rhs)!
        })
        for section in sections {
            XCTAssertFalse(section.fields.contains { $0.tier == .expert }, "\(section.group) holds an expert field")
        }
        XCTAssertTrue(sections.contains { $0.fields.contains { $0.overrideID == .model } }, "the model picker is a section row")
        XCTAssertEqual(StudioTaskSchema.changedCount(for: .audioEnhance, draft: draft), 0)
        var edited = draft
        edited.form["--overlap"] = .integer(4)
        XCTAssertEqual(StudioTaskSchema.changedCount(for: .audioEnhance, draft: edited), 1)
        XCTAssertEqual(StudioTaskSchema.modelScope(for: draft).categories, ["audio"])
        XCTAssertEqual(StudioTaskSchema.modelID(for: draft), "audio-enhance-ap-bwe-16kto48k")
        XCTAssertEqual(StudioModelScope(templateID: .visionDepth).categories, ["vision-depth"])
        XCTAssertEqual(StudioModelScope(templateID: .imageGenerate).categories, StudioMode.createImage.modelCategories)
        XCTAssertTrue(StudioModelScope(templateID: .geoTessera).categories.isEmpty, "no filter when the inventory has no category")
    }

    func testTaskDraftPersistsWithoutSecrets() throws {
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.form["--hf-token"] = .text("hf_secret")
        draft.form.extraArguments = "--api-key sk-live"
        let saved = draft.withoutSessionSecrets
        XCTAssertNil(saved.form.values["--hf-token"])
        XCTAssertFalse(saved.form.extraArguments.contains("sk-live"))
        let data = try JSONEncoder.mereRunApp.encode(draft)
        XCTAssertEqual(try JSONDecoder.mereRunApp.decode(StudioTaskDraft.self, from: data), draft)
    }

    /// A page that persisted a whole `CommandDraft` seeds the task draft once; scalar pages start fresh.
    @MainActor
    func testLegacyPageDraftsSeedTheTaskDraftOnce() throws {
        let sessions = StudioTaskSessions()
        var page = try XCTUnwrap(CommandCatalog.template(id: .audioEnhance)).defaultDraft()
        page.inputPath = "/tmp/voice.wav"
        page.model = "audio-enhance-universr"
        sessions.set(page, for: StudioTask.audioEnhance.rawValue + ".AudioTools.enhanceDraft")
        let imported = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        XCTAssertEqual(imported.templateID, .audioEnhance)
        XCTAssertEqual(imported.primaryInputPath, "/tmp/voice.wav")
        XCTAssertEqual(imported.model, "audio-enhance-universr")
        XCTAssertNil(StudioTaskDraftMigration.legacyKey(for: .visionDepth), "Vision persisted scalars, not a draft")
        XCTAssertEqual(sessions.taskDraft(for: .visionPose)?.templateID, .visionPose, "a fresh draft otherwise")
        sessions.setTaskDraft(StudioTaskDraft(templateID: .audioEdit), for: .audioEnhance)
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.templateID, .audioEdit, "a parked draft wins")
    }
}
