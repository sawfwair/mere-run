@testable import StudioKit
import MereRunContract
import StudioTestSupport
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
    /// the catalog builder emits, plus the `--json` and face numbers a page set on every run,
    /// minus the stamped destination that routing names at submit time instead), and a
    /// populated draft's argv is exactly what `StudioTaskCommandView` shows as "Will run"
    /// (`StudioConsoleRun`), so the composer, the inspector, and the Command view cannot disagree.
    func testEveryTaskDraftBuildsTheCommandViewsArgv() throws {
        for task in migratingTasks {
            XCTAssertFalse(task.variantTemplates.isEmpty, "\(task) has no template to run")
            for template in task.variantTemplates {
                let capability = try XCTUnwrap(template.id.capability, "\(template.id) has no contract")
                let draft = StudioTaskDraft(templateID: template.id)
                let launcher = Set(StudioTaskDraft.launcherDefaults(for: template.id))
                let consoleOnly = Set(StudioTaskDraft.consoleOnlyDefaults(for: template.id))
                let destinations = StudioTaskSchema.outputFlags(for: capability)
                let pageValues = Set(StudioTaskDraft.pageValues(for: template.id).keys.map { "\($0)=\(draft.text($0))" })
                XCTAssertEqual(
                    Self.pairs(of: draft.arguments, capability: capability).subtracting(launcher).subtracting(pageValues),
                    Self.pairs(of: template.arguments(from: template.defaultDraft()), capability: capability)
                        .subtracting(launcher).subtracting(consoleOnly)
                        .filter { pair in !destinations.contains { pair.hasPrefix($0 + "=") } },
                    "\(template.id) fresh draft is the template's own command"
                )
                for flag in launcher {
                    XCTAssertTrue(draft.arguments.contains(flag), "\(template.id) launcher default \(flag)")
                    XCTAssertTrue(capability.options.contains { $0.flag == flag }, "\(template.id) declares \(flag)")
                }
                for flag in consoleOnly {
                    XCTAssertFalse(draft.arguments.contains(flag), "\(template.id) fresh draft runs for real, without \(flag)")
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
        // The contract declares the text trainer's resume checkpoint before its dataset; the
        // required option leads the well all the same, so the adapter is named after the data.
        XCTAssertEqual(StudioTaskSchema.slots(for: .textTrainLoRA).map(\.id), ["--data", "--resume-from", "--eval"])
        XCTAssertEqual(StudioTaskSchema.primarySlot(for: .textTrainLoRA)?.isRequired, true)
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageTrainLoRA).map(\.id), ["--data", "--resume-from"])
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

    /// Each variant keeps its own form: leaving InstantMesh for TRELLIS.2 and coming back finds
    /// the ordered views and the camera file as they were, Compare keeps its second picture
    /// across a trip through Detect, and a Library row of another variant ("Use these
    /// settings") parks the variant it replaces too. The parked forms survive the session store
    /// and never carry a secret or a destination the app named.
    @MainActor
    func testEachVariantKeepsItsOwnDraft() throws {
        var mesh = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
        mesh.form["--view"] = .text("/tmp/front.png\n/tmp/right.png\n/tmp/back.png\n/tmp/left.png")
        mesh.form["--cameras"] = .text("/tmp/cameras.json")
        mesh.switchTemplate(to: .imageReconstruct3DTrellis2)
        XCTAssertEqual(mesh.text("--view"), "", "TRELLIS.2 takes no views")
        mesh.setArgument(0, "/tmp/chair.png")
        mesh.switchTemplate(to: .imageReconstruct3DMultiview)
        XCTAssertEqual(StudioAttachmentSlot.separatedPaths(mesh.text("--view")).count, 4, "the views come back")
        XCTAssertEqual(mesh.text("--cameras"), "/tmp/cameras.json")
        mesh.switchTemplate(to: .imageReconstruct3DTrellis2)
        XCTAssertEqual(mesh.argument(0), "/tmp/chair.png", "and TRELLIS.2 finds its own picture")

        var faces = StudioTaskDraft(templateID: .visionFaceCompare)
        faces.form.arguments = ["/tmp/a.png", "/tmp/b.png"]
        faces.switchTemplate(to: .visionFaceDetect)
        faces.switchTemplate(to: .visionFaceCompare)
        XCTAssertEqual(faces.form.arguments, ["/tmp/a.png", "/tmp/b.png"], "the second picture survives")

        var restored = StudioTaskDraft(templateID: .visionFaceDetect)
        restored.setArgument(0, "/tmp/group.png")
        faces.adopt(restored)
        XCTAssertEqual(faces.templateID, .visionFaceDetect)
        XCTAssertEqual(faces.argument(0), "/tmp/group.png")
        faces.switchTemplate(to: .visionFaceCompare)
        XCTAssertEqual(faces.form.arguments, ["/tmp/a.png", "/tmp/b.png"], "restoring a row parks what it replaced")

        let sessions = StudioTaskSessions()
        sessions.setTaskDraft(faces, for: .visionFaces)
        var reloaded = try XCTUnwrap(sessions.taskDraft(for: .visionFaces))
        XCTAssertEqual(reloaded, faces)
        reloaded.switchTemplate(to: .visionFaceDetect)
        XCTAssertEqual(reloaded.argument(0), "/tmp/group.png", "parked forms round-trip through the session store")

        faces.switchTemplate(to: .visionFaceDetect)
        faces.form.extraArguments = "--hf-token hf_secret"
        faces.switchTemplate(to: .visionFaceCompare)
        XCTAssertFalse(faces.withoutSessionSecrets.parked[.visionFaceDetect]?.extraArguments.contains("hf_secret") ?? true,
                       "a parked form is saved without its secrets")
        mesh.switchTemplate(to: .imageReconstruct3DMultiview)
        var named = mesh
        named.switchTemplate(to: .imageReconstruct3DTrellis2)
        named.form["--output"] = .text("/tmp/out/chair-a1b2c3")
        named.switchTemplate(to: .imageReconstruct3DMultiview)
        XCTAssertEqual(named.withoutDestinations().parked[.imageReconstruct3DTrellis2]?.text("--output"), "",
                       "nor with a destination")
    }

    /// Faces ▸ Embed and Compare start on face 0, the face their picker shows as face 1 and the
    /// one the Faces page always sent, rather than leaving the CLI to take the largest face; 3D ▸
    /// InstantMesh starts at the 3D page's grid resolution of 256.
    func testFacesAndInstantMeshStartWhereTheirPagesDid() {
        let embed = StudioTaskDraft(templateID: .visionFaceEmbed)
        XCTAssertEqual(embed.text("--face-index"), "0")
        XCTAssertEqual(embed.arguments.firstIndex(of: "--face-index").map { embed.arguments[$0 + 1] }, "0", "and the argv says so")
        let compare = StudioTaskDraft(templateID: .visionFaceCompare)
        XCTAssertEqual(compare.text("--reference-face-index"), "0")
        XCTAssertEqual(compare.text("--candidate-face-index"), "0")
        XCTAssertEqual(StudioTaskDraft(templateID: .visionFaceDetect).text("--face-index"), "", "Detect has no face to choose")
        XCTAssertEqual(StudioTaskDraft(templateID: .imageReconstruct3DMultiview).text("--resolution"), "256")
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
        XCTAssertTrue(StudioModelScope(templateID: .musicRealtime).categories.isEmpty, "no filter when the inventory has no category")
    }

    /// Readiness asks `model list` about a managed id only. A local checkpoints root the CLI
    /// takes as `--model` (the Woosh commands), a local model location beside it (Chat ▸ Train's
    /// `--model-path`, Music ▸ Train's `--checkpoints-root`) needs no managed model, so such a
    /// run is never refused as "isn't in the model list".
    func testALocalModelLocationNeedsNoManagedModel() {
        var foley = StudioTaskDraft(templateID: .sfxVideo)
        XCTAssertEqual(StudioTaskSchema.requiredModelID(for: foley), StudioTaskSchema.modelID(for: foley))
        for path in ["/Volumes/Models/woosh", "~/woosh", "./woosh", "checkpoints/woosh"] {
            foley.model = path
            XCTAssertEqual(StudioTaskSchema.requiredModelID(for: foley), "", path)
        }
        foley.model = "sfx-woosh-dvflow-8s"
        XCTAssertEqual(StudioTaskSchema.requiredModelID(for: foley), "sfx-woosh-dvflow-8s")

        var chat = StudioTaskDraft(templateID: .textTrainLoRA)
        XCTAssertFalse(StudioTaskSchema.requiredModelID(for: chat).isEmpty)
        chat.form["--model-path"] = .text("/Volumes/Models/inkling")
        XCTAssertEqual(StudioTaskSchema.requiredModelID(for: chat), "")

        var music = StudioTaskDraft(templateID: .musicTrainAdapter)
        XCTAssertFalse(StudioTaskSchema.requiredModelID(for: music).isEmpty)
        music.form["--checkpoints-root"] = .text("/Volumes/Models/acestep")
        XCTAssertEqual(StudioTaskSchema.requiredModelID(for: music), "")
    }

    /// A recipe decides Image ▸ Train's base when `--model` is left to it, so readiness, the
    /// picker, and "Get the model" name the base the CLI will train, not the template's default.
    func testARecipeNamesTheBaseItTrains() {
        var draft = StudioTaskDraft(templateID: .imageTrainLoRA)
        draft.model = ""
        draft.form["--recipe"] = .text("klein-fast-style")
        XCTAssertEqual(StudioTaskSchema.modelID(for: draft), "image-klein-base-9b")
        XCTAssertEqual(StudioTaskSchema.requiredModelID(for: draft), "image-klein-base-9b")
        draft.form["--recipe"] = .text("krea-cinematic-style")
        XCTAssertEqual(StudioTaskSchema.modelID(for: draft), "image-krea2-raw")
        draft.model = "image-klein-base-9b-8bit"
        XCTAssertEqual(StudioTaskSchema.modelID(for: draft), "image-klein-base-9b-8bit", "an explicit model wins, as on the command line")
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

    /// A page that persisted a whole `CommandDraft` seeds the task draft once; a task no page
    /// kept anything for starts fresh.
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
        XCTAssertNil(StudioTaskDraftMigration.legacyKey(for: .musicServe), "Music server keeps its own draft")
        XCTAssertEqual(sessions.taskDraft(for: .visionPose)?.templateID, .visionPose, "a fresh draft otherwise")
        sessions.setTaskDraft(StudioTaskDraft(templateID: .audioEdit), for: .audioEnhance)
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.templateID, .audioEdit, "a parked draft wins")
    }

    /// A page that kept a draft for more than one of a task's commands keeps them all: Audio ▸
    /// Live opens on the transcript's settings with the speaker activity's parked beside them.
    @MainActor
    func testEveryVariantAPageKeptIsImported() throws {
        let sessions = StudioTaskSessions()
        var listen = try XCTUnwrap(CommandCatalog.template(id: .speechListen)).defaultDraft()
        listen.model = "speech-asr-parakeet-tdt-0.6b-v3"
        sessions.set(listen, for: StudioTask.audioLive.rawValue + ".Voice.listenDraft")
        var speakers = try XCTUnwrap(CommandCatalog.template(id: .speechDiarizeLive)).defaultDraft()
        speakers.model = "speech-diarization-nemotron-3"
        sessions.set(speakers, for: StudioTask.audioLive.rawValue + ".Voice.liveDiarizationDraft")

        var imported = try XCTUnwrap(sessions.taskDraft(for: .audioLive))
        XCTAssertEqual(imported.templateID, StudioTask.audioLive.variantTemplates.first?.id)
        XCTAssertEqual(imported.model, "speech-asr-parakeet-tdt-0.6b-v3")
        imported.switchTemplate(to: .speechDiarizeLive)
        XCTAssertEqual(imported.model, "speech-diarization-nemotron-3", "the speaker activity's draft is not dropped")
    }

    /// The SFX Lab page kept Video Foley's renoise mode beside its draft; the inspector reads it
    /// until it keeps a mode of its own.
    @MainActor
    func testTheFoleyPagesRenoiseModeCarriesOver() {
        let sessions = StudioTaskSessions()
        XCTAssertEqual(StudioTaskDraftMigration.renoiseMode(for: .soundFoley, in: sessions), .automatic)
        sessions.set(StudioRenoise.Mode.schedule, for: StudioTask.soundFoley.rawValue + ".SFXLab.videoRenoiseMode")
        XCTAssertEqual(StudioTaskDraftMigration.renoiseMode(for: .soundFoley, in: sessions), .schedule)
        sessions.set(StudioRenoise.Mode.amount, for: StudioTaskDraftMigration.renoiseModeKey(for: .soundFoley))
        XCTAssertEqual(StudioTaskDraftMigration.renoiseMode(for: .soundFoley, in: sessions), .amount, "the inspector's own mode wins")
    }

    /// The inspector and the run's validation read `--renoise` the way the CLI does, whatever
    /// mode the inspector shows: one number is an amount whatever the step count, so neither
    /// objects to it, and a schedule of the wrong length is refused by both.
    func testTheInspectorAndTheRunShareOneRenoiseRule() throws {
        let capability = try XCTUnwrap(CommandTemplateID.sfxVideo.capability)
        var draft = StudioTaskDraft(templateID: .sfxVideo)
        let steps = StudioRenoise.stepCount(in: draft.form, templateID: .sfxVideo)
        for argument in ["0.3", "0.3, 0.2", "", "1.5", String(repeating: "0.1,", count: steps - 1) + "0.1"] {
            draft.form["--renoise"] = argument.isEmpty ? .unset : .text(argument)
            XCTAssertEqual(
                StudioCommandChecks.message(for: capability, draft: draft.form),
                StudioRenoise.problems(argument: argument, steps: steps).first,
                argument
            )
        }
        XCTAssertEqual(StudioRenoise.problems(argument: "0.3", steps: steps), [], "one amount is valid for any step count")
        XCTAssertFalse(StudioRenoise.problems(argument: "0.3, 0.2", steps: 4).isEmpty)
    }

    /// Multi-view geometry refuses a camera file that does not fit the views, as its page did,
    /// instead of running with estimated cameras; one that will not read is refused too.
    func testMultiviewGeometryRefusesCamerasThatDoNotFitTheViews() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("geometry-cameras-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let capability = try XCTUnwrap(CommandTemplateID.visionGeometryMultiview.capability)
        var draft = StudioTaskDraft(templateID: .visionGeometryMultiview)
        draft.form.arguments = ["/tmp/view-a.png", "/tmp/view-b.png"]
        XCTAssertNil(StudioCommandChecks.message(for: capability, draft: draft.form), "no cameras: the model solves them")

        let one = root.appendingPathComponent("one-camera.json")
        try StudioGeometryCameraDocument(cameras: [.identity()]).json().write(to: one)
        draft.form["--cameras"] = .text(one.path)
        XCTAssertEqual(StudioCommandChecks.message(for: capability, draft: draft.form), "Add one camera per view: 2 views, 1 camera.")

        let two = root.appendingPathComponent("two-cameras.json")
        try StudioGeometryCameraDocument(cameras: [.identity(), .identity()]).json().write(to: two)
        draft.form["--cameras"] = .text(two.path)
        XCTAssertNil(StudioCommandChecks.message(for: capability, draft: draft.form))

        let broken = root.appendingPathComponent("broken.json")
        try Data("not json".utf8).write(to: broken)
        draft.form["--cameras"] = .text(broken.path)
        XCTAssertTrue(StudioCommandChecks.message(for: capability, draft: draft.form)?.hasPrefix("The camera file at broken.json could not be read") ?? false)
    }

    /// The 3D page kept scalar keys, not a draft. Its command is rebuilt for every engine the
    /// way it built it: InstantMesh gets the ordered views and the camera document it edited
    /// (written as the editor's draft file), the page's engine is the variant it opens on, and
    /// the other engines keep the source picture and shared settings.
    @MainActor
    func testThe3DPagesKeysBecomeTheTaskDraft() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("legacy-3d-\(UUID().uuidString)")
        StudioTestDefaults.redirectSupport(under: root)
        defer {
            StudioTestDefaults.restore()
            try? FileManager.default.removeItem(at: root)
        }
        let sessions = StudioTaskSessions()
        let key = StudioTask.threeDFromImage.rawValue + ".3DCreation."
        let views = ["/tmp/front.png", "/tmp/right.png", "/tmp/back.png", "/tmp/left.png"]
        let cameras = StudioInstantMeshCameraDocument(cameras: Array(repeating: .example, count: 4))
        sessions.set("InstantMesh", for: key + "engine")
        sessions.set("/tmp/chair.png", for: key + "sourcePath")
        sessions.set(views, for: key + "orderedViews")
        sessions.set("image-3d-instantmesh-large", for: key + "model")
        sessions.set(true, for: key + "suppliesCameras")
        sessions.set(cameras, for: key + "cameras")
        sessions.set(true, for: key + "alreadyFramed")

        var imported = try XCTUnwrap(sessions.taskDraft(for: .threeDFromImage))
        XCTAssertEqual(imported.templateID, .imageReconstruct3DMultiview, "the engine the page had open")
        XCTAssertEqual(StudioAttachmentSlot.separatedPaths(imported.text("--view")), views)
        XCTAssertEqual(imported.model, "image-3d-instantmesh-large")
        XCTAssertEqual(imported.text("--resolution"), "256")
        let camerasFile = imported.text("--cameras")
        XCTAssertTrue(StudioCameraDocuments.isDraft(camerasFile, page: "3D Creation"), camerasFile)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: camerasFile)), try cameras.json())
        XCTAssertNil(StudioCommandChecks.message(for: try XCTUnwrap(imported.capability), draft: imported.form))

        imported.switchTemplate(to: .imageReconstruct3DTrellis2)
        XCTAssertEqual(imported.argument(0), "/tmp/chair.png")
        XCTAssertEqual(imported.model, "image-3d-trellis2-4b", "InstantMesh's model is not TRELLIS.2's")
        imported.switchTemplate(to: .imageReconstruct3D)
        XCTAssertEqual(imported.form["--already-framed"].flag, true)
    }

    /// The Vision page kept scalar keys per task. Faces gets its threshold and face numbers on
    /// every face command; multi-view geometry gets its ordered views and the camera document
    /// it edited as the editor's draft file.
    @MainActor
    func testTheVisionPagesKeysBecomeTheTaskDrafts() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("legacy-vision-\(UUID().uuidString)")
        StudioTestDefaults.redirectSupport(under: root)
        defer {
            StudioTestDefaults.restore()
            try? FileManager.default.removeItem(at: root)
        }
        let sessions = StudioTaskSessions()
        let faces = StudioTask.visionFaces.rawValue + ".VisionLab."
        sessions.set("/tmp/group.png", for: faces + "primaryInput")
        sessions.set(0.4, for: faces + "faceThreshold")
        sessions.set(2, for: faces + "faceIndex")
        var face = try XCTUnwrap(sessions.taskDraft(for: .visionFaces))
        XCTAssertEqual(face.templateID, .visionFaceDetect)
        XCTAssertEqual(face.argument(0), "/tmp/group.png")
        XCTAssertEqual(face.text("--score-threshold"), "0.4")
        face.switchTemplate(to: .visionFaceEmbed)
        XCTAssertEqual(face.text("--face-index"), "2")

        let geometry = StudioTask.visionGeometry.rawValue + ".VisionLab."
        let document = StudioGeometryCameraDocument(cameras: [.identity(), .identity()])
        sessions.set("/tmp/view-a.png", for: geometry + "primaryInput")
        sessions.set(["/tmp/view-b.png"], for: geometry + "additionalInputs")
        sessions.set(true, for: geometry + "suppliesCameras")
        sessions.set(document, for: geometry + "geometryCameras")
        var multiview = try XCTUnwrap(sessions.taskDraft(for: .visionGeometry))
        multiview.switchTemplate(to: .visionGeometryMultiview)
        XCTAssertEqual(multiview.form.arguments.filter { !$0.isEmpty }, ["/tmp/view-a.png", "/tmp/view-b.png"])
        let camerasFile = multiview.text("--cameras")
        XCTAssertTrue(StudioCameraDocuments.isDraft(camerasFile, page: "Vision Geometry"), camerasFile)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: camerasFile)), try document.json())
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
