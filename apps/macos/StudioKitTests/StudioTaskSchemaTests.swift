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

    /// A fresh task draft runs the template's own default command (the same option/value pairs
    /// the catalog builder emits, plus the `--json` a page set on every run), and a populated
    /// draft's argv is exactly what `StudioTaskCommandView` shows as "Will run"
    /// (`StudioConsoleRun`), so the composer, the inspector, and the Command view cannot disagree.
    func testEveryTaskDraftBuildsTheCommandViewsArgv() throws {
        for task in migratingTasks {
            XCTAssertFalse(task.variantTemplates.isEmpty, "\(task) has no template to run")
            for template in task.variantTemplates {
                let capability = try XCTUnwrap(template.id.capability, "\(template.id) has no contract")
                let draft = StudioTaskDraft(templateID: template.id)
                let launcher = Set(StudioTaskDraft.launcherDefaults(for: template.id))
                XCTAssertEqual(
                    Self.pairs(of: draft.arguments, capability: capability).subtracting(launcher),
                    Self.pairs(of: template.arguments(from: template.defaultDraft()), capability: capability).subtracting(launcher),
                    "\(template.id) fresh draft is the template's own command"
                )
                for flag in launcher {
                    XCTAssertTrue(draft.arguments.contains(flag), "\(template.id) launcher default \(flag)")
                    XCTAssertTrue(capability.options.contains { $0.flag == flag }, "\(template.id) declares \(flag)")
                }

                var populated = draft
                for slot in populated.slots {
                    slot.attach([URL(fileURLWithPath: "/tmp/input-\(slot.id).png")], to: &populated)
                }
                if capability.options.contains(where: { $0.flag == "--model" }) { populated.model = "some-model" }
                populated.prompt = "a prompt"
                let launch = try XCTUnwrap(StudioConsoleRun(template: template, draft: populated.form, seed: populated.seed))
                XCTAssertEqual(populated.arguments, launch.arguments, "\(template.id) Command view argv")
                XCTAssertEqual(populated.run?.arguments, launch.arguments)
                let request = try XCTUnwrap(populated.request(), "\(template.id) request")
                XCTAssertEqual(request.mode, template.libraryMode, "attribution comes from the template")
                XCTAssertEqual(request.execution?.arguments, launch.arguments)
            }
        }
    }

    /// The same normalization `StudioConsoleDraftTests` uses: the catalog builders and the
    /// contract emit options in different orders.
    private static func pairs(of arguments: [String], capability: MereRunCommandCapability) -> Set<String> {
        let parsed = StudioCommandRows.parse(arguments: arguments, commandPathCount: capability.command.count)
        var units = Set(parsed.positional.enumerated().filter { !$0.element.isEmpty }.map { "\($0.offset)=\($0.element)" })
        for (flag, value) in parsed.flags {
            guard let value else {
                units.insert(flag)
                continue
            }
            guard !value.isEmpty else { continue }
            units.insert("\(flag)=\(value)")
        }
        return units
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

    func testSlotsWriteTheTaskDraft() throws {
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
        // The argv carries every image, and reading that argv back (Use these settings, the
        // Command view) keeps them all as positionals rather than spilling them into extras.
        let batchCapability = try XCTUnwrap(CommandTemplateID.visionFaceBatch.capability)
        XCTAssertEqual(Array(batch.arguments.prefix(6)), ["vision", "face", "batch", "/tmp/1.png", "/tmp/2.png", "/tmp/3.png"])
        let reread = StudioConsoleCommand.seed(capability: batchCapability, arguments: batch.arguments)
        XCTAssertEqual(reread.arguments, ["/tmp/1.png", "/tmp/2.png", "/tmp/3.png"])
        XCTAssertEqual(reread.extraArguments, "")
        XCTAssertEqual(reread.text("--input-list"), "/tmp/list.txt")

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

        // Batch holds one repeatable positional with three values; Compare declares two single
        // ones. Only the first image carries, into Compare's first argument.
        var batch = StudioTaskDraft(templateID: .visionFaceBatch)
        batch.form.arguments = ["/tmp/1.png", "/tmp/2.png", "/tmp/3.png"]
        batch.switchTemplate(to: .visionFaceCompare)
        XCTAssertEqual(batch.form.arguments.filter { !$0.isEmpty }, ["/tmp/1.png"])
        XCTAssertEqual(batch.argument(1), "", "Compare's candidate stays empty")
        var compare = StudioTaskDraft(templateID: .visionFaceCompare)
        compare.form.arguments = ["/tmp/a.png", "/tmp/b.png"]
        compare.switchTemplate(to: .visionFaceBatch)
        XCTAssertEqual(compare.form.arguments, ["/tmp/a.png"], "a second single positional has no place in a repeatable one")
        var empty = StudioTaskDraft(templateID: .visionFaceBatch)
        empty.switchTemplate(to: .visionFaceCompare)
        XCTAssertTrue(empty.form.arguments.allSatisfy(\.isEmpty), "nothing to carry")
    }

    /// The pages turned `--json` on for every run whose surface reads the printed result; a fresh
    /// task draft does the same so a renderer gets JSON, and a parked draft keeps what it ran with.
    func testLauncherDefaultsTurnOnMachineOutputWherePagesDid() {
        XCTAssertTrue(StudioTaskDraft(templateID: .imageRunPlan).arguments.contains("--json"))
        XCTAssertTrue(StudioTaskDraft(templateID: .imageDatasetDiscover).arguments.contains("--json"))
        XCTAssertTrue(StudioTaskDraft(templateID: .visionPose).arguments.contains("--json"))
        XCTAssertFalse(StudioTaskDraft(templateID: .audioEnhance).arguments.contains("--json"), "enhance writes a file")
        let restored = StudioTaskDraft(templateID: .imageRunPlan, form: StudioConsoleDraft(arguments: ["/tmp/plan.json"]))
        XCTAssertFalse(restored.arguments.contains("--json"), "a restored draft is not changed")
        // Preflight, materialize, and the discover output root stay reachable in the inspector.
        let plan = StudioTaskDraft(templateID: .imageRunPlan)
        let planFlags = StudioTaskSchema.fields(for: .imageDatasets, draft: plan).flatMap(\.draftFieldIDs)
        XCTAssertTrue(planFlags.contains("--preflight"))
        XCTAssertTrue(planFlags.contains("--materialize"))
        XCTAssertFalse(planFlags.contains("--json"), "the launcher switch is not a control")
        XCTAssertEqual(StudioTaskSchema.sections(for: .imageDatasets, draft: plan).first { $0.group == .output }?.fields.map(\.flag),
                       ["--materialize"])
        let discover = StudioTaskDraft(templateID: .imageDatasetDiscover)
        XCTAssertTrue(StudioTaskSchema.fields(for: .imageDatasets, draft: discover).contains { $0.flag == "--training-output-root" })
        XCTAssertFalse(StudioTaskSchema.slots(for: .imageDatasetDiscover).contains { $0.id == "--training-output-root" })
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageValidate), [], "validate takes no input; its folders are options")
        XCTAssertTrue(StudioTaskSchema.fields(for: .imageDatasets, draft: StudioTaskDraft(templateID: .imageValidate))
            .contains { $0.flag == "--reference-dir" })
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

    /// The Music Tools page stamped a timestamped `--output` and `--context-output` into the
    /// draft it kept, the way it named every run's files; importing that draft keeps the
    /// settings (input, format, model) and drops the destinations, so routing names the next run
    /// afresh instead of writing over the page's last files.
    @MainActor
    func testImportedPageDraftsDropTheirPerRunDestinations() throws {
        let sessions = StudioTaskSessions()
        var page = try XCTUnwrap(CommandCatalog.template(id: .musicTranscribe)).defaultDraft()
        page.inputPath = "/Users/example/Music/harbor-lights.wav"
        page.outputPath = "/Users/example/Music/mere.run/Music/music-transcribe-20260903-101500.mid"
        page.musicContextOutput = "/Users/example/Music/mere.run/Music/music-transcribe-20260903-101500-context.json"
        page.musicTranscribeFormat = "midi"
        page.musicInstruments = "piano"
        sessions.set(page, for: StudioTask.musicTranscribe.rawValue + ".MusicTools.transcribeDraft")

        let imported = try XCTUnwrap(sessions.taskDraft(for: .musicTranscribe))
        XCTAssertEqual(imported.primaryInputPath, page.inputPath)
        XCTAssertEqual(imported.text("--instruments"), "piano")
        XCTAssertEqual(imported.text("--output"), "", "the page's per-run file is not a setting")
        XCTAssertEqual(imported.text("--context-output"), "")
        XCTAssertFalse(imported.arguments.contains(where: { $0.contains("20260903-101500") }))

        var encode = try XCTUnwrap(CommandCatalog.template(id: .sfxAEEncode)).defaultDraft()
        encode.inputPath = "/tmp/hit.wav"
        encode.outputPath = "/Users/example/Music/mere.run/Sound/sfx-encode-20260903-101500.npy"
        sessions.set(encode, for: StudioTask.soundEncode.rawValue + ".SFXLab.encodeDraft")
        let encodeDraft = try XCTUnwrap(sessions.taskDraft(for: .soundEncode))
        XCTAssertEqual(encodeDraft.primaryInputPath, "/tmp/hit.wav")
        XCTAssertEqual(encodeDraft.text("--output"), "")
    }
}
