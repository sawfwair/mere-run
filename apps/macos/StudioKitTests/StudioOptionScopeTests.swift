import Foundation
import MereRunContract
@testable import StudioKit
import XCTest

/// The mechanism behind every scoped surface: one resolution from the argv a surface would run,
/// hidden values kept in the draft but reset before validation and launch, and a note that says
/// what the model leaves out. What each surface shows per family is
/// `StudioModelScopeGoldenTests`; this file holds the behavior that is not a set.
final class StudioOptionScopeTests: XCTestCase {
    private typealias Music = StudioScopeContracts.Music
    private typealias Video = StudioScopeContracts.Video
    private let source = StudioScopeContracts.source([Music.capability, Video.capability])

    // MARK: Hidden values

    func testAHiddenValueSurvivesAModelSwitchAndComesBack() throws {
        var draft = StudioDraft.baseline(for: .music)
        draft.prompt = "Acoustic folk waltz"
        draft.musicTask = "cover"
        draft.musicSourceAudio = "/tmp/source.wav"
        XCTAssertEqual(StudioMode.music.attachmentSlots(for: draft, source: source).map(\.id), ["source", "timbre"])

        draft.model = "music-yue2"
        XCTAssertEqual(StudioMode.music.attachmentSlots(for: draft, source: source).map(\.id), [])
        XCTAssertFalse(StudioContractSchema.fields(for: .music, draft: draft, source: source).contains { $0.flag == "--task-type" })
        let yue2 = try StudioCommandAdapter.makeRequest(mode: .music, draft: draft, source: source)
        XCTAssertEqual(yue2.draft.musicSourceAudio, "")
        XCTAssertEqual(yue2.draft.musicTask, "text2music")
        XCTAssertEqual(draft.musicSourceAudio, "/tmp/source.wav", "the draft keeps what the model hides")

        draft.model = "music-acestep"
        let ace = try StudioCommandAdapter.makeRequest(mode: .music, draft: draft, source: source)
        XCTAssertEqual(ace.draft.musicSourceAudio, "/tmp/source.wav")
        XCTAssertEqual(ace.draft.musicTask, "cover")
    }

    /// H1: a Cover task chosen for ACE-Step, with no source attached yet, blocked a YuE2 run
    /// that has no source well to fill.
    func testValidationIgnoresAHiddenCoverTask() throws {
        var draft = StudioDraft.baseline(for: .music)
        draft.prompt = "Acoustic folk waltz"
        draft.musicTask = "cover"
        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .music, draft: draft, source: source)) { error in
            XCTAssertEqual(error as? StudioCommandError, .missingInput("source audio"))
        }
        draft.model = "music-yue2"
        XCTAssertNoThrow(try StudioCommandAdapter.makeRequest(mode: .music, draft: draft, source: source))
        draft.musicTask = "text2music"
        draft.musicFlowEdit = true
        draft.model = "music-magenta-rt2-small"
        XCTAssertNoThrow(try StudioCommandAdapter.makeRequest(mode: .music, draft: draft, source: source))
    }

    /// H2: keyframes attached for LTX blocked MiniMax-H3 Ref2VA, which takes ordered references
    /// and has no keyframe wells.
    func testValidationIgnoresHiddenKeyframes() throws {
        var draft = StudioDraft.baseline(for: .video)
        draft.prompt = "A lighthouse at dusk"
        draft.inputPath = "/tmp/start.png"
        draft.endImagePath = "/tmp/end.png"
        draft.model = "video-minimax-h3-ref2va-mlx"
        draft.h3ReferenceInputs = ["image:/tmp/keeper.png"]

        // The keyframes alone would refuse the run: Ref2VA takes ordered references only.
        var keyframes = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).defaultDraft()
        keyframes.model = draft.model
        keyframes.prompt = draft.prompt
        keyframes.inputPath = draft.inputPath
        keyframes.endImagePath = draft.endImagePath
        keyframes.h3ReferenceInputs = draft.h3ReferenceInputs
        let unscoped = CommandArguments.videoGenerate(keyframes, scope: source.scope(capability: Video.capability, commandLine: []))
        XCTAssertNotNil(source.scope(capability: Video.capability, commandLine: unscoped).refusal)

        let request = try StudioCommandAdapter.makeRequest(mode: .video, draft: draft, source: source)
        XCTAssertEqual(request.draft.inputPath, "")
        XCTAssertEqual(request.draft.endImagePath, "")
        XCTAssertNil(request.template.validationMessage(for: request.draft, source: .contract))
        XCTAssertEqual(StudioMode.video.attachmentSlots(for: draft, source: source).map(\.id), ["audio"])
    }

    // MARK: One resolution

    func testABlankModelScopesToTheContractDefault() throws {
        var form = StudioConsoleDraft()
        form.arguments = ["A lighthouse at dusk"]
        let blank = source.scope(capability: Video.capability, form: form)
        XCTAssertEqual(blank.family?.id, Video.ltx)
        XCTAssertEqual(blank.managedModel, "video-ltx25-full-bf16")

        // The composer sends the template's model when its draft names none, so a cleared
        // model scopes to what the run sends.
        var draft = StudioDraft.baseline(for: .video)
        draft.model = ""
        XCTAssertEqual(source.scope(mode: .video, draft: draft)?.family?.id, Video.ltx)
    }

    func testModelRootWinsOverModel() throws {
        let source = StudioScopeContracts.source(
            [Video.capability], identities: ["/tmp/checkpoints/h3-ref2va": .identified(.family(Video.ref2va))]
        )
        var draft = StudioDraft.baseline(for: .video)
        draft.model = "video-ltx25-full-bf16"
        var command = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).defaultDraft()
        command.model = draft.model
        command.modelRoot = "/tmp/checkpoints/h3-ref2va"
        let scope = source.scope(
            capability: Video.capability,
            commandLine: try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).unscopedArguments(from: command, source: .contract)
        )
        XCTAssertEqual(scope.family?.id, Video.ref2va)
        XCTAssertFalse(scope.allows("--image"))
    }

    func testScopingNeverChangesTheFamily() throws {
        for model in ["music-acestep", "music-yue2", "music-magenta-rt2-small"] {
            var draft = StudioDraft.baseline(for: .music)
            draft.model = model
            draft.musicTask = "cover"
            draft.musicSourceAudio = "/tmp/source.wav"
            let scope = try XCTUnwrap(source.scope(mode: .music, draft: draft))
            let scoped = draft.scoped(to: scope, mode: .music)
            XCTAssertEqual(source.scope(mode: .music, draft: scoped)?.family, scope.family, model)
        }
    }

    // MARK: Identity

    func testAPendingOrUnidentifiedFolderShowsEveryOptionAndSaysWhy() throws {
        let folder = "/tmp/checkpoints/my-video-model"
        for (identity, kind) in [(StudioModelIdentity.pending, StudioScopeNotice.Kind.identifying), (.unidentified, .unidentified)] {
            let source = StudioScopeContracts.source([Video.capability], identities: [folder: identity])
            var draft = StudioDraft.baseline(for: .video)
            draft.model = folder
            let scope = try XCTUnwrap(source.scope(mode: .video, draft: draft))
            XCTAssertNil(scope.family)
            XCTAssertEqual(scope.options.map(\.flag), Video.capability.options.map(\.flag))
            let notice = try XCTUnwrap(StudioInspectorSchema.notice(for: .video, draft: draft, source: source))
            XCTAssertEqual(notice.kind, kind)
            XCTAssertTrue(notice.title.contains("my-video-model"), notice.title)
            XCTAssertNil(StudioCommandAdapter.capabilityRequirement(for: .video, draft: draft, source: source),
                         "a local folder has no managed model to check")
        }
        let identified = StudioScopeContracts.source([Video.capability], identities: [folder: .identified(.family(Video.ref2va))])
        var draft = StudioDraft.baseline(for: .video)
        draft.model = folder
        XCTAssertEqual(identified.scope(mode: .video, draft: draft)?.family?.id, Video.ref2va)
        XCTAssertNil(StudioInspectorSchema.notice(for: .video, draft: draft, source: identified))
    }

    /// H-a: a local folder whose family is not known yet runs no family's rule. Video's Quality
    /// and FPS, and music's Candidates, are free controls, not another model's locked value.
    func testAnUnplacedFolderLocksNoControl() throws {
        let folder = "/tmp/checkpoints/my-model"
        for identity in [StudioModelIdentity.pending, .unidentified] {
            let source = StudioScopeSource(identities: StudioFixedModelIdentities([folder: identity]))
            for mode in [StudioMode.video, .music, .chat] {
                var draft = StudioDraft.baseline(for: mode)
                draft.model = folder
                let fields = StudioContractSchema.fields(for: mode, draft: draft, source: source)
                XCTAssertFalse(fields.isEmpty)
                let locked = fields.filter { $0.fixedValue != nil || $0.allowedValues != nil }.map(\.flag)
                XCTAssertEqual(locked, [], "\(mode) \(identity)")
            }
        }
        // A model the contract places still locks what its own family fixes.
        var fastH3 = StudioDraft.baseline(for: .video)
        fastH3.model = "video-minimax-h3-fasth3-vsa-datafree-mlx"
        let steps = StudioContractSchema.fields(for: .video, draft: fastH3, source: StudioScopeSource(identities: StudioFixedModelIdentities()))
            .first { $0.flag == "--steps" }
        XCTAssertEqual(steps?.fixedValue, "5")
    }

    func testTheIdentityStoreSendsTheWholeCommandLineAndAsksOncePerRoutingFlags() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("scope-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let store = StudioModelIdentityStore()
        let asked = AskedCommandLines()
        await store.use { commandLine in
            await asked.append(commandLine)
            return MereRunFamilyResolutionReport(
                capability: "video.generate", family: Video.ref2va, familyTitle: "MiniMax-H3 Ref2VA",
                model: folder.path, source: .identified, violations: [], warnings: []
            )
        }
        let arguments = { (root: String, prompt: String) in ["--model-root", root, prompt, "--seed", "7"] }
        let identity = { (arguments: [String]) in store.identity(of: arguments, model: folder.path, for: Video.capability) }
        XCTAssertEqual(identity(arguments(folder.path, "a cat")), .pending)
        // The same folder with a trailing slash, another prompt, or another seed is the same
        // question: only the routing flags pick the family, so it is not asked again.
        XCTAssertNotEqual(identity(arguments(folder.path + "/", "a dog")), .unidentified)
        for _ in 0..<100 where identity(arguments(folder.path, "a cat")) == .pending {
            try await Task.sleep(for: .milliseconds(10))
        }
        let answer = StudioModelIdentity.resolved(.family(id: Video.ref2va, model: folder.path, source: .identified))
        XCTAssertEqual(identity(arguments(folder.path, "a cat")), answer)
        XCTAssertEqual(identity(["--model-root", folder.path, "a bird", "--seed", "9"]), answer)
        let lines = await asked.lines
        XCTAssertEqual(lines, [["video", "generate"] + arguments(folder.path, "a cat")], "the CLI hears the whole command line")
    }

    func testAReportBecomesTheResolutionItDescribes() {
        let report = { (family: String?, model: String?, source: MereRunFamilyResolutionReport.Source) in
            MereRunFamilyResolutionReport(
                capability: "music.generate", family: family, familyTitle: nil, model: model, source: source,
                violations: [], warnings: []
            )
        }
        XCTAssertEqual(StudioModelIdentityStore.identity(from: report(Music.yue2, "music-yue2", .model), in: Music.capability),
                       .resolved(.family(id: Music.yue2, model: "music-yue2", source: .model)))
        XCTAssertEqual(StudioModelIdentityStore.identity(from: report(nil, "/tmp/unknown", .unidentified), in: Music.capability),
                       .unidentified)
    }

    /// A RoFormer model takes any divisor of its chunk size as the overlap: the stepper walks
    /// that set rather than every integer, and a typed value lands on the nearest one.
    func testAChunkOverlapStepsThroughTheValuesItsModelTakes() throws {
        var draft = StudioTaskDraft(templateID: .musicSeparate)
        draft.model = "music-separate-bs-roformer-4stem"
        let field = try XCTUnwrap(StudioTaskSchema.fields(for: .musicSeparate, draft: draft, source: .contract).first { $0.flag == "--overlap" })
        XCTAssertEqual(field.control, .stepper)
        let divisors = (1...485_100).filter { 485_100.isMultiple(of: $0) }.map(Double.init)
        XCTAssertEqual(field.allowedValues, divisors)
        XCTAssertEqual(field.stepped(from: 4, by: 1), 5)
        XCTAssertEqual(field.stepped(from: 7, by: 1), 9, "8 does not divide the 4-stem chunk")
        XCTAssertEqual(field.stepped(from: 1, by: -1), 1)
        XCTAssertEqual(field.stepped(from: 485_100, by: 1), 485_100)
        XCTAssertEqual(field.clamped(.integer(8)), .integer(7))
    }

    // MARK: Pickers and readiness

    func testAnExcludedModelIsNotOfferedAndBlocksWithItsReason() throws {
        let rows = ["vision-depth-marigold-v2", "vision-depth-vda-small"].map {
            StudioModelInventoryRow(id: $0, category: "vision-depth", status: "installed", size: "1 GB", usageTerms: nil)
        }
        XCTAssertEqual(StudioModelScope(templateID: .visionDepth, source: .contract).choices(from: rows).map(\.id), ["vision-depth-marigold-v2"])

        var draft = StudioTaskDraft(templateID: .visionDepth)
        draft.model = "vision-depth-vda-small"
        guard case .unavailable(let reason)? = StudioTaskSchema.requirement(for: draft, source: .contract) else {
            return XCTFail("an excluded model must block the run")
        }
        XCTAssertTrue(reason.contains("depth-video"), reason)
        XCTAssertEqual(StudioTaskSchema.requiredModelID(for: draft, source: .contract), "")
    }

    /// H-b: a picker offers every model the contract runs the command with: the families'
    /// models, the ones whose family depends on what is installed (`identified_models`, such as
    /// `video-ltx-av`), and the defaults, less the excluded. A template's own default is offered.
    func testPickersOfferEveryModelTheCommandRuns() {
        let source = StudioScopeSource(identities: StudioFixedModelIdentities())
        var pickers: [(name: String, capability: MereRunCommandCapability, scope: StudioModelScope, templateDefault: String?)] = []
        for templateID in CommandTemplateID.allCases {
            guard let capability = source.capability(for: templateID), capability.routing != nil else { continue }
            pickers.append(("task \(templateID)", capability, StudioModelScope(templateID: templateID, source: source),
                            CommandCatalog.template(id: templateID)?.defaultModel))
        }
        for mode in StudioMode.allCases {
            let actions: [StudioReadImageAction] = mode == .readImage ? StudioReadImageAction.allCases : [.inspect]
            for action in actions {
                let templateID = mode == .readImage ? action.templateID : mode.defaultTemplateID
                guard let capability = source.capability(for: templateID), capability.routing != nil else { continue }
                pickers.append(("mode \(mode) \(action)", capability, StudioModelScope(mode: mode, readImageAction: action, source: source),
                                CommandCatalog.template(id: templateID)?.defaultModel))
            }
        }
        XCTAssertFalse(pickers.isEmpty)
        for picker in pickers {
            let routing = try? XCTUnwrap(picker.capability.routing)
            guard let routing, let offered = picker.scope.runnableModels else {
                XCTFail("\(picker.name) offers every row for a routed command")
                continue
            }
            let excluded = Set(routing.excludedModels.map(\.id))
            let runs = Set(routing.families.flatMap(\.models) + routing.identifiedModels + routing.defaultModels.flatMap(\.models))
            XCTAssertEqual(runs.subtracting(excluded).subtracting(offered).sorted(), [], "\(picker.name) drops models the command runs")
            XCTAssertEqual(offered.intersection(excluded).sorted(), [], "\(picker.name) offers excluded models")
            if let own = picker.templateDefault, !own.isEmpty, !excluded.contains(own) {
                XCTAssertTrue(offered.contains(own), "\(picker.name) drops its own default \(own)")
            }
            if !picker.scope.defaultModelID.isEmpty {
                XCTAssertTrue(offered.contains(picker.scope.defaultModelID), "\(picker.name) drops its default \(picker.scope.defaultModelID)")
            }
        }
        let video = StudioModelScope(templateID: .videoGenerate, source: source)
        for id in ["video-ltx-av", "video-ltx23-full-mlx", "video-ltx23-a2vid-mlx"] {
            XCTAssertEqual(video.runnableModels?.contains(id), true, id)
        }
    }

    /// The empty canvas offers a starting image only while the model takes one: FastH3 has no
    /// start-frame well.
    func testTheEmptyCanvasOffersOnlyTheAttachmentsTheModelTakes() {
        var draft = StudioDraft.baseline(for: .video)
        draft.model = "video-minimax-h3-fasth3-vsa-datafree-mlx"
        let fastH3 = StudioTaskPresentation(mode: .video, slots: StudioMode.video.attachmentSlots(for: draft, source: .contract))
        XCTAssertFalse(fastH3.emptyMessage.contains("image"), fastH3.emptyMessage)
        draft.model = "video-ltx25-full-bf16"
        let ltx = StudioTaskPresentation(mode: .video, slots: StudioMode.video.attachmentSlots(for: draft, source: .contract))
        XCTAssertEqual(ltx.emptyMessage, StudioMode.video.emptyMessage)
        XCTAssertTrue(ltx.emptyMessage.contains("starting image"))
    }

    // MARK: Argv

    func testTheBuildersDropWhatTheFamilyDoesNotTakeAndItsOwnDefault() throws {
        let scope = { (model: String) in
            self.source.scope(capability: Music.capability, commandLine: [
                "music", "generate", "song", "--model", model, "--quality", "song", "--source-audio", "/tmp/a.wav",
                "--target-peak-db=-1", "--steps", "12",
            ])
        }
        let argv = ["music", "generate", "song", "--model", "music-acestep", "--quality", "song",
                    "--source-audio", "/tmp/a.wav", "--target-peak-db=-1", "--steps", "12"]
        XCTAssertEqual(
            StudioOptionScopes.filtered(argv, scope: scope("music-acestep")),
            ["music", "generate", "song", "--model", "music-acestep", "--source-audio", "/tmp/a.wav",
             "--target-peak-db=-1", "--steps", "12"],
            "ACE-Step runs song when --quality is left off"
        )
        var yue2 = argv
        yue2[4] = "music-yue2"
        XCTAssertEqual(
            StudioOptionScopes.filtered(yue2, scope: scope("music-yue2")),
            ["music", "generate", "song", "--model", "music-yue2", "--target-peak-db=-1", "--steps", "12"]
        )
        var magenta = argv
        magenta[4] = "music-magenta-rt2-small"
        XCTAssertEqual(
            StudioOptionScopes.filtered(magenta + ["--seed", "3"], scope: scope("music-magenta-rt2-small")),
            ["music", "generate", "song", "--model", "music-magenta-rt2-small", "--target-peak-db=-1"],
            "an option the family ignores is dropped too"
        )
    }

    /// A distilled LTX-2.5 folder that holds the diffusion decoder, as `catalog resolve` places it,
    /// offers and sends either decoder; the managed checkpoint shows the convolutional one fixed.
    func testTheVideoDecoderFollowsWhatADistilledFolderHolds() throws {
        let capability = MereRunCapabilityCatalog.videoGenerate
        let folder = "/tmp/checkpoints/ltx25-distilled-with-diffusion"
        let answer = StudioModelIdentity.resolved(.family(id: "ltx25-distilled-diffusion", model: folder, source: .identified))
        let source = StudioScopeSource(identities: StudioFixedModelIdentities([folder: answer]))
        let argv = ["video", "generate", "a cat", "--model-root", folder, "--video-decoder", "diffusion"]
        let diffusion = source.scope(capability: capability, commandLine: argv)
        XCTAssertEqual(diffusion.family?.title, "LTX-2.5 Distilled with the diffusion decoder")
        XCTAssertNil(diffusion.fixedValue("--video-decoder"))
        XCTAssertEqual(diffusion.option("--video-decoder")?.choices, ["diffusion", "convolutional"])
        XCTAssertEqual(diffusion.unusedFlags, [])
        XCTAssertEqual(StudioOptionScopes.filtered(argv, scope: diffusion), argv)

        let managedArgv = ["video", "generate", "a cat", "--model", "video-ltx25-distilled-bf16", "--video-decoder", "diffusion"]
        let managed = StudioScopeSource.contract.scope(capability: capability, commandLine: managedArgv)
        XCTAssertEqual(managed.fixedValue("--video-decoder"), "convolutional")
        XCTAssertEqual(managed.unusedFlags, ["--video-decoder"])
    }

    /// Stage-one previews and IC-LoRA references on a full checkpoint: refused without source
    /// audio, ignored with it. Studio hides both either way and never sends them there.
    func testAFullCheckpointNeverSendsStageOnePreviews() throws {
        let capability = MereRunCapabilityCatalog.videoGenerate
        let base = ["video", "generate", "a cat", "--model", "video-ltx25-full-bf16"]
        for audio in [[], ["--audio", "/tmp/a.wav"]] {
            let argv = base + audio + ["--skip-stage-2"]
            let scope = StudioScopeSource.contract.scope(capability: capability, commandLine: argv)
            XCTAssertFalse(scope.allows("--skip-stage-2"), "\(audio)")
            XCTAssertEqual(scope.unusedFlags, ["--skip-stage-2"], "\(audio)")
            XCTAssertEqual(scope.refusal != nil, audio.isEmpty, "\(audio): \(String(describing: scope.refusal))")
            XCTAssertEqual(StudioOptionScopes.filtered(argv, scope: scope), base + audio)
        }
    }

    /// Source audio drops LTX-2.5 Full's IC-LoRA references, so Studio marks them unused with it;
    /// a checkpoint that takes no source audio drops the audio and keeps the references.
    func testSourceAudioDropsReferencesOnlyWhereItIsSent() throws {
        let capability = MereRunCapabilityCatalog.videoGenerate
        let reference = ["--video-conditioning", "/tmp/r.mp4", "--lora", "/tmp/ic.safetensors"]
        let full = ["video", "generate", "a cat", "--model", "video-ltx25-full-bf16", "--audio", "/tmp/a.wav"]
        let fullScope = StudioScopeSource.contract.scope(capability: capability, commandLine: full + reference)
        XCTAssertNil(fullScope.refusal)
        XCTAssertEqual(fullScope.unusedFlags, ["--video-conditioning"])

        let distilled = ["video", "generate", "a cat", "--model", "video-ltx25-distilled-bf16"]
        let argv = distilled + ["--audio", "/tmp/a.wav"] + reference
        let distilledScope = StudioScopeSource.contract.scope(capability: capability, commandLine: argv)
        XCTAssertEqual(distilledScope.unusedFlags, ["--audio"])
        XCTAssertEqual(StudioOptionScopes.filtered(argv, scope: distilledScope), distilled + reference)
    }

    func testAFixedValueIsKeptOnlyAsTheFamilyRunsIt() throws {
        let scope = source.scope(capability: Video.capability, commandLine: [
            "video", "generate", "x", "--model", "video-minimax-h3-fasth3-vsa-datafree-mlx", "--steps", "30",
        ])
        XCTAssertEqual(scope.fixedValue("--steps"), "5")
        XCTAssertEqual(scope.unusedFlags, ["--steps"])
        XCTAssertEqual(
            StudioOptionScopes.filtered(["video", "generate", "x", "--steps", "30"], scope: scope),
            ["video", "generate", "x"]
        )
        XCTAssertEqual(
            StudioOptionScopes.filtered(["video", "generate", "x", "--steps", "5"], scope: scope),
            ["video", "generate", "x", "--steps", "5"]
        )
    }

    /// L1: argv is read the way ArgumentParser reads it: by the option's kind, and an option takes
    /// the next token only when that token is a value, so a negative number rides with `=`.
    func testArgvIsReadByOptionKind() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "music.generate"))
        let form = StudioConsoleCommand.seed(capability: capability, arguments: [
            "music", "generate", "song", "--target-peak-db=-1", "-s", "12", "--instrumental", "--unknown", "value",
        ])
        XCTAssertEqual(form.text("--target-peak-db"), "-1")
        XCTAssertTrue(StudioConsoleCommand.arguments(for: capability, draft: form).contains("--target-peak-db=-1"),
                      "the console writes a negative value joined to its flag")
        XCTAssertEqual(
            StudioArgvToken.read(["--target-peak-db", "-1", "--steps", "-", "--lyrics", ""], capability: capability),
            [.option(flag: "--target-peak-db", tokens: ["--target-peak-db"], values: []), .undeclared("-1"),
             .option(flag: "--steps", tokens: ["--steps", "-"], values: ["-"]),
             .option(flag: "--lyrics", tokens: ["--lyrics", ""], values: [""])],
            "ArgumentParser refuses `--target-peak-db -1`; a lone \"-\" and \"\" are values"
        )
        XCTAssertEqual(form.text("--steps"), "12", "a short alias lands under its canonical flag")
        XCTAssertEqual(form["--instrumental"], .flag(true))
        XCTAssertEqual(form.arguments, ["song"])
        XCTAssertEqual(form.extraArguments, "--unknown value")
    }

    // MARK: Task drafts and the console

    /// M3: "Will run" and the run are the scoped command; the draft keeps the rest.
    func testATaskDraftRunsItsScopedForm() throws {
        let capability = StudioScopeContracts.routed(
            "audio.enhance",
            families: [
                MereRunRuntimeFamily(id: "ap-bwe", title: "AP-BWE", models: ["audio-enhance-ap-bwe-16kto48k"]),
                MereRunRuntimeFamily(id: "universr", title: "UniverSR", models: ["audio-enhance-universr-audio"]),
            ],
            defaultModel: "audio-enhance-ap-bwe-16kto48k",
            uses: ["--ode-method": ["universr"], "--ode-steps": ["universr"], "--seed": ["universr"]]
        )
        let source = StudioScopeContracts.source([capability])
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.setArgument(0, "/tmp/memo.wav")
        draft.model = "audio-enhance-ap-bwe-16kto48k"
        XCTAssertEqual(draft.text("--seed"), "42", "the template's UniverSR default is in the fresh form")
        let arguments = draft.arguments(source: source)
        XCTAssertFalse(arguments.contains("--ode-method"))
        XCTAssertFalse(arguments.contains("--seed"))
        XCTAssertEqual(draft.run(source: source)?.arguments.contains("--seed"), false)
        XCTAssertFalse(StudioTaskSchema.fields(for: .audioEnhance, draft: draft, source: source).contains { $0.flag == "--seed" })

        draft.model = "audio-enhance-universr-audio"
        XCTAssertTrue(draft.arguments(source: source).contains("--seed"), "switching to UniverSR runs its defaults")

        // The note lists only what moved off the fresh draft, so the template's own defaults for
        // the other model are not called out.
        draft.model = "audio-enhance-ap-bwe-16kto48k"
        XCTAssertNil(StudioTaskSchema.notice(for: draft, source: source))
        draft.form["--ode-steps"] = .integer(8)
        XCTAssertEqual(StudioTaskSchema.notice(for: draft, source: source)?.title, "Not used by AP-BWE: UniverSR ODE steps.")
    }

    /// M4: an override is a change to what runs, compared as each side launches.
    func testTheOverrideBannerComparesScopedCommands() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicGenerate))
        var command = template.defaultDraft()
        command.model = "music-yue2"
        let composed = template.arguments(from: command, source: source)
        var edited = StudioConsoleCommand.seed(capability: Music.capability, arguments: composed)
        edited["--source-audio"] = .text("/tmp/kept-for-ace.wav")
        let hiddenEdit = StudioTaskCommandState(templateID: .musicGenerate, sourceArguments: composed, form: edited)
        XCTAssertFalse(hiddenEdit.overrides(source: composed, scopeSource: source))

        edited["--lyrics"] = .text("la la la")
        let shownEdit = StudioTaskCommandState(templateID: .musicGenerate, sourceArguments: composed, form: edited)
        XCTAssertTrue(shownEdit.overrides(source: composed, scopeSource: source))
    }

    func testTheConsoleShowsTheFamilysOptionsAndNotesTheRest() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicGenerate))
        var form = StudioConsoleCommand.seed(template: template, draft: template.defaultDraft(), source: .contract)
        form["--model"] = .text("music-yue2")
        form["--source-audio"] = .text("/tmp/source.wav")
        form["--task-type"] = .text("cover")
        let scope = source.scope(capability: Music.capability, form: form)
        let shown = Set(StudioConsoleCommand.groups(for: Music.capability, scope: scope).flatMap(\.fields).map(\.flag))
        XCTAssertFalse(shown.contains("--source-audio"))
        XCTAssertTrue(shown.contains("--lyrics"))
        let notice = try XCTUnwrap(scope.notice(form: form))
        XCTAssertEqual(notice.kind, .unused)
        XCTAssertTrue(notice.title.hasPrefix("Not used by YuE2: "), notice.title)
        XCTAssertTrue(notice.title.contains("Source audio"), notice.title)

        let run = try XCTUnwrap(StudioConsoleRun(template: template, draft: form, seed: template.defaultDraft(), source: source))
        XCTAssertFalse(run.arguments.contains("--source-audio"))
        XCTAssertEqual(form.text("--source-audio"), "/tmp/source.wav")
    }

    // MARK: Composer

    /// M1: a chip shows only when the model uses one of its flags, and a fixed value is read-only.
    func testChipsFollowTheModel() {
        var draft = StudioDraft.baseline(for: .music)
        draft.model = "music-magenta-rt2-small"
        XCTAssertEqual(StudioMode.music.composerChips(for: draft, source: source).map(\.kind), [.duration, .model])
        draft.model = "music-acestep"
        XCTAssertEqual(StudioMode.music.composerChips(for: draft, source: source).map(\.kind), [.duration, .steps, .seed, .model])

        var video = StudioDraft.baseline(for: .video)
        video.model = "video-minimax-h3-fasth3-vsa-datafree-mlx"
        let steps = StudioMode.video.composerChips(for: video, source: source).first { $0.kind == .steps }
        XCTAssertEqual(steps?.fixedValue, "5")
        XCTAssertEqual(steps?.fixedBy, "FastH3")
    }

    func testTheNoteNamesWhatTheModelLeavesOutAndReplaces() throws {
        var draft = StudioDraft.baseline(for: .video)
        draft.model = "video-minimax-h3-fasth3-vsa-datafree-mlx"
        draft.endImagePath = "/tmp/end.png"
        draft.timings = true
        draft.h3Steps = 30
        let notice = try XCTUnwrap(StudioInspectorSchema.notice(for: .video, draft: draft, source: source))
        XCTAssertEqual(notice.title, "Not used by FastH3: End image, Timings.")
        XCTAssertEqual(notice.details, [
            "Denoising steps: FastH3 runs 5; your 30 is kept.",
            "Your values are kept for when you switch back.",
        ])
        XCTAssertTrue(notice.accessibilityLabel.hasPrefix(notice.title))

        draft.model = "video-ltx25-full-bf16"
        XCTAssertNil(StudioInspectorSchema.notice(for: .video, draft: draft, source: source))
    }

    /// L4: each attachment slot's flag is the binding that stores the slot's field: exactly one
    /// binding per path slot, or none for a slot that fills the command's positional input.
    func testAttachmentSlotsFillTheFlagsTheirFieldsBind() {
        var flags: [String: String] = [:]
        for mode in StudioMode.allCases {
            for slot in mode.attachmentSlots {
                if let flag = mode.attachmentFlag(for: slot) { flags["\(mode.rawValue).\(slot.id)"] = flag }
                let keyPath: PartialKeyPath<StudioDraft>
                switch slot.storage {
                case .path(let path), .pathList(let path): keyPath = path
                case .flag, .flagList, .argument, .argumentList: continue
                }
                let bound = StudioContractBindings.bindings(for: mode).values.filter { $0.storage == keyPath }
                if slot.storage == .path(\StudioDraft.inputPath), StudioContractSchema.capability(for: mode)?.arguments
                    .contains(where: { [.file, .directory].contains($0.kind) }) == true, bound.isEmpty {
                    continue
                }
                XCTAssertEqual(bound.count, 1, "\(mode) \(slot.id) must be stored by exactly one flag")
            }
        }
        XCTAssertEqual(flags["video.startFrame"], "--image")
        XCTAssertEqual(flags["video.endFrame"], "--end-image")
        XCTAssertEqual(flags["video.audio"], "--audio")
        XCTAssertEqual(flags["music.source"], "--source-audio")
        XCTAssertEqual(flags["music.timbre"], "--reference-audio")
        XCTAssertEqual(flags["createImage.input"], "--input")
        XCTAssertEqual(flags["createImage.references"], "--ref-image")
        XCTAssertEqual(flags["speak.referenceAudio"], "--ref-audio")
        XCTAssertEqual(flags["chat.image"], "--image")
        XCTAssertNil(flags["listen.input"], "a positional input is never scoped away")
    }

    // MARK: Replay

    func testAReplayedLibraryCommandIsRescopedAndSaysSo() throws {
        let recorded = StudioExecution(templateID: .musicGenerate, arguments: [
            "music", "generate", "song", "--model", "music-yue2", "--source-audio", "/tmp/a.wav", "--task-type", "cover",
        ])
        XCTAssertEqual(
            recorded.scoped(source: source).arguments,
            ["music", "generate", "song", "--model", "music-yue2"]
        )
        // Design risk 6: the replay says what it left out.
        let notice = try XCTUnwrap(recorded.rescopeNotice(source: source))
        XCTAssertEqual(notice.kind, .unused)
        XCTAssertEqual(notice.title, "Run again left out Task, Source audio: YuE2 doesn't take them.")
        let current = StudioExecution(templateID: .musicGenerate, arguments: ["music", "generate", "song", "--model", "music-yue2"])
        XCTAssertNil(current.rescopeNotice(source: source), "a command the model runs whole says nothing")
    }
}

private actor AskedCommandLines {
    private(set) var lines: [[String]] = []

    func append(_ line: [String]) {
        lines.append(line)
    }
}
