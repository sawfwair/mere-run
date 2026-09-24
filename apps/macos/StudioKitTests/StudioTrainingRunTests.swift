@testable import StudioKit
import StudioTestSupport
import MereRunContract
import UniformTypeIdentifiers
import XCTest

/// The Train tasks over a task draft: the dataset and checkpoint wells fill the trainers' flags,
/// the pages' editors are the contract's overrides, the pages' stored drafts import once, the
/// argv for the same settings is the argv the pages built, a preflight is the run with its
/// check-only switches, and Music ▸ Train's clip list lands beside the adapter routing names.
final class StudioTrainingRunTests: XCTestCase {
    private let trainers: [(task: StudioTask, templateID: CommandTemplateID)] = [
        (.imageTrain, .imageTrainLoRA), (.chatTrain, .textTrainLoRA), (.musicTrain, .musicTrainAdapter),
    ]

    // MARK: Wells and editors

    func testTheDatasetIsTheWellsPrimarySlotAndTheOtherInputsFollow() throws {
        let image = StudioTaskSchema.slots(for: .imageTrainLoRA)
        XCTAssertEqual(image.map(\.id), ["--data", "--resume-from"])
        XCTAssertEqual(image[0].acceptedTypes, [.folder], "the image dataset is a folder")
        XCTAssertEqual(image[0].storage, .flag("--data"))

        let text = StudioTaskSchema.slots(for: .textTrainLoRA)
        XCTAssertEqual(text.map(\.id), ["--data", "--resume-from", "--eval"], "the dataset names the adapter, not the checkpoint")
        XCTAssertTrue(text[0].isRequired)
        XCTAssertFalse(text[0].acceptedTypes.contains(.folder), "the text dataset is a JSONL file")

        XCTAssertEqual(StudioTaskSchema.slots(for: .musicTrainAdapter), [], "the clip list editor owns music's dataset")

        // A folder slot takes a directory that exists; a file slot takes any file.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("training-slots-\(UUID().uuidString)", isDirectory: true)
        let photos = root.appendingPathComponent("my-style-photos", isDirectory: true)
        try FileManager.default.createDirectory(at: photos, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = root.appendingPathComponent("checkpoint-step750.safetensors")
        try Data().write(to: checkpoint)
        var draft = StudioTaskDraft(templateID: .imageTrainLoRA)
        XCTAssertTrue(image[0].accepts(photos))
        XCTAssertFalse(image[0].accepts(checkpoint), "the dataset slot takes a folder only")
        image[0].attach([photos], to: &draft)
        image[1].attach([checkpoint], to: &draft)
        XCTAssertEqual(draft.primaryInputPath, photos.path)
        XCTAssertEqual(draft.arguments.firstIndex(of: "--data").map { draft.arguments[$0 + 1] }, photos.path)
        XCTAssertEqual(draft.arguments.firstIndex(of: "--resume-from").map { draft.arguments[$0 + 1] }, checkpoint.path)
        XCTAssertEqual(draft.run?.commandDraft.inputPath, photos.path, "the Library row's input")
    }

    func testThePagesEditorsAreTheContractsOverrides() throws {
        let image = StudioTaskSchema.fields(for: .imageTrain, draft: StudioTaskDraft(templateID: .imageTrainLoRA))
        let ranks = try XCTUnwrap(image.first { $0.flag == "--lora-target-ranks" })
        XCTAssertEqual(ranks.overrideID, .targetRanks)
        XCTAssertEqual(image.first { $0.flag == "--model" }?.overrideID, .model)
        XCTAssertNil(image.first { $0.flag == "--data" }, "the well owns the dataset")
        XCTAssertNil(image.first { $0.flag == "--output" }, "routing owns the adapter path")
        XCTAssertNil(image.first { $0.flag == "--json" }, "the launcher owns the report switch")
        XCTAssertNotNil(image.first { $0.flag == "--preflight" }, "preflight stays reachable")

        let music = StudioTaskSchema.fields(for: .musicTrain, draft: StudioTaskDraft(templateID: .musicTrainAdapter))
        XCTAssertEqual(music.first { $0.flag == "--dataset" }?.overrideID, .musicManifest)
        let checkpoints = try XCTUnwrap(music.first { $0.flag == "--checkpoints-root" })
        XCTAssertEqual(checkpoints.group, .model, "ACE-Step's checkpoint layout sits with the model")
        XCTAssertEqual(checkpoints.control, .path, "a folder chooser, not a text field")
        XCTAssertEqual(checkpoints.kind, .directory)
        for flag in ["--decoder-subdirectory", "--vae-subdirectory", "--text-subdirectory"] {
            XCTAssertNotNil(music.first { $0.flag == flag }, "\(flag) stays reachable")
        }

        let text = StudioTaskSchema.fields(for: .chatTrain, draft: StudioTaskDraft(templateID: .textTrainLoRA))
        XCTAssertEqual(text.first { $0.flag == "--model-path" }?.group, .model)
        XCTAssertNil(text.first { $0.flag == "--eval" }, "the well owns the evaluation prompts")
    }

    /// Every flag a page's sections name is one its template declares, no flag is placed twice,
    /// and the well's, the picker's, and routing's flags are never placed: what a section does not
    /// name falls under Advanced, so nothing the command takes is out of reach.
    func testPageSectionsNameDeclaredFlagsOnce() throws {
        for trainer in trainers {
            let capability = try XCTUnwrap(trainer.templateID.capability)
            let declared = Set(capability.options.map(\.flag))
            let sections = StudioTrainingRun.sections(for: trainer.templateID)
            XCTAssertFalse(sections.isEmpty, "\(trainer.templateID) has no sections")
            let placed = sections.flatMap(\.flags) + StudioTrainingRun.modelFlags(for: trainer.templateID)
            XCTAssertEqual(placed.count, Set(placed).count, "\(trainer.templateID) places a flag twice")
            for flag in placed {
                XCTAssertTrue(declared.contains(flag), "\(trainer.templateID) section names \(flag), which it does not declare")
            }
            let slots = Set(StudioTaskSchema.slots(for: trainer.templateID).map(\.id))
            let hidden = StudioTaskSchema.hiddenFlags(for: capability).union(slots).union(["--model", "--dataset"])
            XCTAssertTrue(hidden.isDisjoint(with: placed), "\(trainer.templateID) places a flag another surface owns")
        }
    }

    // MARK: Imports and parity

    /// The pages kept one `CommandDraft` each; the task draft reads it once, its per-run
    /// destination cleared so routing names the next adapter.
    @MainActor
    func testEachPagesStoredDraftImportsIntoItsOwnTaskDraft() throws {
        let sessions = StudioTaskSessions()
        var image = try XCTUnwrap(CommandCatalog.template(id: .imageTrainLoRA)).defaultDraft()
        image.inputPath = "/tmp/my-style-photos"
        image.outputPath = "/Users/example/Documents/mere.run/Image/image-adapter-20260924-1200.safetensors"
        image.trainingRecipe = "klein-fast-style"
        image.trainingResumePath = "/tmp/checkpoint-step750.safetensors"
        image.loraTargetRanks = ".attn.to_q=128,.attn.to_k=64"
        sessions.set(image, for: StudioTask.imageTrain.rawValue + ".Training.imageDraft")
        var text = try XCTUnwrap(CommandCatalog.template(id: .textTrainLoRA)).defaultDraft()
        text.inputPath = "/tmp/train.jsonl"
        text.model = "text-chat-inkling-small"
        text.modelRoot = "/Volumes/models/inkling"
        sessions.set(text, for: StudioTask.chatTrain.rawValue + ".Training.textDraft")
        var music = try XCTUnwrap(CommandCatalog.template(id: .musicTrainAdapter)).defaultDraft()
        music.musicTrainingKind = "lokr"
        music.musicCheckpointsRoot = "/Volumes/models/acestep"
        sessions.set(music, for: StudioTask.musicTrain.rawValue + ".Training.musicDraft")

        let imported = try XCTUnwrap(sessions.taskDraft(for: .imageTrain))
        XCTAssertEqual(imported.templateID, .imageTrainLoRA)
        XCTAssertEqual(imported.primaryInputPath, "/tmp/my-style-photos")
        XCTAssertEqual(imported.text("--recipe"), "klein-fast-style")
        XCTAssertEqual(imported.text("--resume-from"), "/tmp/checkpoint-step750.safetensors")
        XCTAssertEqual(imported.text("--lora-target-ranks"), ".attn.to_q=128,.attn.to_k=64")
        XCTAssertEqual(imported.text("--output"), "", "the page's stamped destination was that run's, not a setting")

        let chat = try XCTUnwrap(sessions.taskDraft(for: .chatTrain))
        XCTAssertEqual(chat.primaryInputPath, "/tmp/train.jsonl")
        XCTAssertEqual(chat.model, "text-chat-inkling-small")
        XCTAssertEqual(chat.text("--model-path"), "/Volumes/models/inkling")

        let adapter = try XCTUnwrap(sessions.taskDraft(for: .musicTrain))
        XCTAssertEqual(adapter.text("--kind"), "lokr")
        XCTAssertEqual(adapter.text("--checkpoints-root"), "/Volumes/models/acestep")
    }

    /// The argv a task draft builds for a page's settings is the argv the page built from its
    /// `CommandDraft`, resume checkpoint and target ranks included.
    func testATaskDraftBuildsThePagesArgvForTheSameSettings() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageTrainLoRA))
        var page = template.defaultDraft()
        page.inputPath = "/tmp/my-style-photos"
        page.outputPath = "/tmp/out/style.safetensors"
        page.seed = "42"
        page.checkpointInterval = 250
        page.sampleInterval = 250
        page.trainingResumePath = "/tmp/checkpoint-step750.safetensors"
        page.loraTargetRanks = ".attn.to_q=128"
        page.loraTargetMode = "suffix"
        page.preflight = true
        page.json = true
        let expected = template.arguments(from: page)

        let draft = StudioTaskDraft(templateID: .imageTrainLoRA, form: StudioConsoleCommand.seed(template: template, draft: page))
        // Compared as sets: the page's builder and the contract emit options in different
        // orders, and the CLI reads them in any order; the count check keeps a pair from
        // appearing twice.
        XCTAssertEqual(Set(draft.arguments), Set(expected), "the same option/value pairs")
        XCTAssertEqual(draft.arguments.count, expected.count)
        let run = try XCTUnwrap(draft.run)
        XCTAssertEqual(run.arguments, draft.arguments, "what the Command view shows is what runs")
        XCTAssertEqual(run.commandDraft.outputPath, "/tmp/out/style.safetensors", "the job lifecycle reads the adapter path")
        XCTAssertEqual(run.commandDraft.inputPath, "/tmp/my-style-photos", "and the Library row's input")
    }

    /// A chosen recipe decides the options the page used to leave off the command line; the
    /// seeded defaults must not override it (the CLI prefers an explicit flag), and the Klein
    /// recipe must not run with the seeded Krea 2 model.
    func testAChosenRecipeClearsTheSeededOptionsItDecides() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageTrainLoRA))
        for recipe in ["krea-fast-style", "klein-fast-style"] {
            var page = template.defaultDraft()
            page.seed = "42"
            page.trainingRecipe = recipe
            page.overrideTrainingRecipe = false
            let expected = template.arguments(from: page)

            var draft = StudioTrainingRun.baseline(for: .imageTrainLoRA)
            draft.form["--recipe"] = .text(recipe)
            let applied = StudioTrainingRun.applyingRecipe(draft)
            XCTAssertEqual(Set(applied.arguments), Set(expected), "\(recipe): the page's command with the recipe")
            XCTAssertEqual(applied.arguments.count, expected.count, "\(recipe)")
            XCTAssertFalse(applied.arguments.contains("--model"), "\(recipe) decides the base model")
            XCTAssertFalse(applied.arguments.contains("--width"))
            XCTAssertEqual(applied.text("--seed"), "42", "the seed is not the recipe's")
            XCTAssertEqual(StudioTrainingRun.applyingRecipe(applied), applied)
            XCTAssertTrue(StudioTrainingRun.isRecipeGoverned("--rank", in: applied))
            XCTAssertFalse(StudioTrainingRun.isRecipeGoverned("--seed", in: applied))
        }

        // A value the user typed is an override the recipe does not take away.
        var typed = StudioTrainingRun.baseline(for: .imageTrainLoRA)
        typed.form["--rank"] = .text("32")
        typed.form["--recipe"] = .text("krea-fast-style")
        XCTAssertEqual(StudioTrainingRun.applyingRecipe(typed).text("--rank"), "32")
        XCTAssertEqual(StudioTrainingRun.applyingRecipe(typed).text("--width"), "")
        XCTAssertFalse(StudioTrainingRun.isRecipeGoverned("--rank", in: StudioTrainingRun.baseline(for: .imageTrainLoRA)), "no recipe, nothing governed")
    }

    /// A Klein launch gets checkpoints and previews every 250 steps unless the user chose a
    /// cadence; Krea 2, which takes neither option, gets nothing added.
    func testAKleinLaunchSetsTheCheckpointAndPreviewCadence() {
        var klein = StudioTrainingRun.baseline(for: .imageTrainLoRA)
        klein.form["--recipe"] = .text("klein-fast-style")
        let launched = StudioTrainingRun.launchDraft(klein)
        XCTAssertTrue(StudioTrainingRun.trainsKlein(klein))
        XCTAssertEqual(launched.text("--checkpoint-interval"), "250")
        XCTAssertEqual(launched.text("--sample-interval"), "250")
        XCTAssertFalse(launched.arguments.contains("--model"), "the recipe still decides the model")

        var chosen = klein
        chosen.form["--checkpoint-interval"] = .integer(100)
        XCTAssertEqual(StudioTrainingRun.launchDraft(chosen).text("--checkpoint-interval"), "100", "a chosen cadence is kept")

        var base = StudioTrainingRun.baseline(for: .imageTrainLoRA)
        base.model = "image-klein-base-9b"
        XCTAssertTrue(StudioTrainingRun.trainsKlein(base))
        XCTAssertEqual(StudioTrainingRun.launchDraft(base).text("--sample-interval"), "250")

        let krea = StudioTrainingRun.launchDraft(StudioTrainingRun.baseline(for: .imageTrainLoRA))
        XCTAssertFalse(StudioTrainingRun.trainsKlein(krea))
        XCTAssertEqual(krea.text("--checkpoint-interval"), "", "Klein-only; it would block a Krea 2 preflight")
        XCTAssertEqual(krea, StudioTrainingRun.baseline(for: .imageTrainLoRA))
    }

    func testTheModelPickerOffersTrainableBasesOnly() {
        XCTAssertTrue(StudioTrainingRun.isTrainableBase("image-krea2-raw", for: .imageTrainLoRA))
        XCTAssertTrue(StudioTrainingRun.isTrainableBase("image-krea2-turbo", for: .imageTrainLoRA))
        XCTAssertTrue(StudioTrainingRun.isTrainableBase("image-klein-base-9b-8bit", for: .imageTrainLoRA))
        XCTAssertFalse(StudioTrainingRun.isTrainableBase("image-klein-9b", for: .imageTrainLoRA), "the distilled Klein is not a training base")
        XCTAssertFalse(StudioTrainingRun.isTrainableBase("image-zimage-nano", for: .imageTrainLoRA))
        XCTAssertTrue(StudioTrainingRun.isTrainableBase("music-acestep-xl-sft", for: .musicTrainAdapter))
        XCTAssertFalse(StudioTrainingRun.isTrainableBase("music-magenta-rt2-medium", for: .musicTrainAdapter))
        XCTAssertTrue(StudioTrainingRun.isTrainableBase("text-chat-inkling-small", for: .textTrainLoRA))
    }

    // MARK: Fresh drafts and preflight

    func testTheImagePagesDefaultsApplyToAFreshDraftOnly() {
        let fresh = StudioTrainingRun.applyingPageDefaults(StudioTaskDraft(templateID: .imageTrainLoRA))
        XCTAssertEqual(fresh.text("--seed"), "42")
        XCTAssertEqual(fresh.text("--checkpoint-interval"), "", "Klein-only; a Krea 2 preflight would block on it")
        XCTAssertEqual(fresh.text("--sample-interval"), "")
        XCTAssertEqual(StudioTrainingRun.applyingPageDefaults(fresh), fresh, "applying twice changes nothing")
        XCTAssertEqual(StudioTrainingRun.baseline(for: .imageTrainLoRA), fresh)

        var edited = StudioTaskDraft(templateID: .imageTrainLoRA)
        edited.form["--seed"] = .integer(7)
        XCTAssertEqual(StudioTrainingRun.applyingPageDefaults(edited).text("--seed"), "7", "a chosen seed is kept")

        let text = StudioTaskDraft(templateID: .textTrainLoRA)
        XCTAssertEqual(StudioTrainingRun.applyingPageDefaults(text), text, "the text trainer's template already seeds 42")
        XCTAssertEqual(text.text("--seed"), "42")
        XCTAssertEqual(StudioTaskDraft(templateID: .musicTrainAdapter).text("--seed"), "42")
    }

    func testPreflightIsTheRunWithItsCheckOnlySwitches() throws {
        var image = StudioTaskDraft(templateID: .imageTrainLoRA)
        image.form["--data"] = .text("/tmp/my-style-photos")
        let imageCheck = try XCTUnwrap(StudioTrainingRun.preflightDraft(image))
        XCTAssertTrue(imageCheck.arguments.contains("--preflight"))
        XCTAssertTrue(imageCheck.arguments.contains("--json"))
        XCTAssertEqual(imageCheck.text("--data"), "/tmp/my-style-photos", "the rest of the request is unchanged")
        XCTAssertFalse(image.arguments.contains("--preflight"), "the draft itself is untouched")

        let textCheck = try XCTUnwrap(StudioTrainingRun.preflightDraft(StudioTaskDraft(templateID: .textTrainLoRA)))
        XCTAssertTrue(textCheck.arguments.contains("--dry-run"))
        XCTAssertTrue(textCheck.arguments.contains("--json"))
        XCTAssertFalse(textCheck.arguments.contains("--preflight"), "the text trainer has no preflight switch")

        XCTAssertNil(StudioTrainingRun.preflightDraft(StudioTaskDraft(templateID: .musicTrainAdapter)), "the clip list's checks are the page's own")
    }

    /// Image ▸ Datasets ▸ Discover's "Train on it" hands a folder to Image ▸ Train's draft: a
    /// fresh page gets its defaults with the dataset, a parked draft keeps its settings.
    @MainActor
    func testAnotherSurfaceCanHandADatasetToTheTrainersDraft() throws {
        let sessions = StudioTaskSessions()
        StudioTrainingRun.attachDataset("/tmp/warm-still-life", to: .imageTrain, sessions: sessions)
        let fresh = try XCTUnwrap(sessions.taskDraft(for: .imageTrain))
        XCTAssertEqual(fresh.primaryInputPath, "/tmp/warm-still-life")
        XCTAssertEqual(fresh.text("--seed"), "42", "a fresh page's defaults come along")
        XCTAssertTrue(sessions.contains(StudioTaskSessions.taskDraftKey(.imageTrain)), "parked, so the page opens on it")

        var parked = fresh
        parked.form["--recipe"] = .text("klein-fast-style")
        parked.form["--seed"] = .integer(7)
        sessions.setTaskDraft(parked, for: .imageTrain)
        StudioTrainingRun.attachDataset("/tmp/linen-textures", to: .imageTrain, sessions: sessions)
        let replaced = try XCTUnwrap(sessions.taskDraft(for: .imageTrain))
        XCTAssertEqual(replaced.primaryInputPath, "/tmp/linen-textures")
        XCTAssertEqual(replaced.text("--recipe"), "klein-fast-style", "the parked settings stay")
        XCTAssertEqual(replaced.text("--seed"), "7")

        StudioTrainingRun.attachDataset("/tmp/replies.jsonl", to: .chatTrain, sessions: sessions)
        XCTAssertEqual(sessions.taskDraft(for: .chatTrain)?.text("--data"), "/tmp/replies.jsonl")
        StudioTrainingRun.attachDataset("/tmp/anything", to: .musicTrain, sessions: sessions)
        XCTAssertFalse(sessions.contains(StudioTaskSessions.taskDraftKey(.musicTrain)), "the clip list has no well to fill")
        XCTAssertNil(StudioTask.imageGenerate.trainingTemplateID)
    }

    // MARK: Destinations

    /// The image adapter is named after the dataset folder inside the Image folder; music's
    /// clip list is written beside the adapter routing names, and preparing that request keeps
    /// both where they are.
    @MainActor
    func testAdaptersFileUnderTheirDomainAndTheClipListLandsBesideTheMusicAdapter() throws {
        try withConfiguredRoot { root in
            var image = StudioTaskDraft(templateID: .imageTrainLoRA)
            image.form["--data"] = .text("/tmp/my-style-photos")
            let imageOutput = StudioOutputLocation.destination(for: image).text("--output")
            XCTAssertTrue(imageOutput.hasPrefix(root.appendingPathComponent("Image").path), imageOutput)
            XCTAssertTrue(URL(fileURLWithPath: imageOutput).lastPathComponent.hasPrefix("my-style-photos-"), imageOutput)
            XCTAssertEqual(URL(fileURLWithPath: imageOutput).pathExtension, "safetensors")

            var text = StudioTaskDraft(templateID: .textTrainLoRA)
            text.form["--data"] = .text("/tmp/support-replies.jsonl")
            let textOutput = StudioOutputLocation.destination(for: text).text("--output")
            XCTAssertTrue(textOutput.hasPrefix(root.appendingPathComponent("Chat").path), textOutput)
            XCTAssertTrue(URL(fileURLWithPath: textOutput).lastPathComponent.hasPrefix("support-replies-"), textOutput)

            let clips = root.appendingPathComponent("clips", isDirectory: true)
            try FileManager.default.createDirectory(at: clips, withIntermediateDirectories: true)
            let clip = clips.appendingPathComponent("loop.wav")
            try Data([0x52, 0x49, 0x46, 0x46]).write(to: clip)
            let manifest = StudioMusicTrainingManifest(clips: [
                StudioMusicTrainingClip(audioPath: clip.path, caption: "short synth loop, steady kick", lyrics: "la la\nla"),
            ])
            var music = StudioTaskDraft(templateID: .musicTrainAdapter)
            music.form["--dataset"] = .text(root.appendingPathComponent("drafts/dataset-abc123.jsonl").path)
            let previewed = StudioOutputLocation.destination(for: music).text("--output")

            let launch = try StudioTrainingRun.musicLaunch(music, manifest: manifest)
            let output = launch.text("--output")
            XCTAssertEqual(output, previewed, "the adapter the Command view previews is the adapter the run writes")
            XCTAssertTrue(output.hasPrefix(root.appendingPathComponent("Music").path), output)
            let manifestURL = URL(fileURLWithPath: launch.text("--dataset"))
            XCTAssertEqual(manifestURL.deletingLastPathComponent().path, URL(fileURLWithPath: output).deletingLastPathComponent().path)
            XCTAssertEqual(manifestURL.lastPathComponent, URL(fileURLWithPath: output).deletingPathExtension().lastPathComponent + ".dataset.jsonl")
            let written = try StudioMusicTrainingManifest.importing(Data(contentsOf: manifestURL), from: manifestURL)
            XCTAssertEqual(written.clips.map(\.caption), ["short synth loop, steady kick"])
            XCTAssertEqual(written.clips.first?.lyrics, "la la\nla")

            let request = try XCTUnwrap(launch.request())
            let prepared = try StudioTaskRunner.prepare(request, sessions: StudioTaskSessions())
            XCTAssertNil(prepared.fallbackReason)
            XCTAssertEqual(prepared.request.draft.outputPath, output, "preparing does not name the adapter again")
            XCTAssertEqual(prepared.request.draft.inputPath, manifestURL.path)
            let argv = try XCTUnwrap(prepared.request.execution).arguments
            XCTAssertEqual(argv.firstIndex(of: "--dataset").map { argv[$0 + 1] }, manifestURL.path)
            XCTAssertEqual(argv.firstIndex(of: "--output").map { argv[$0 + 1] }, output)
            XCTAssertEqual(prepared.request.mode, .music, "attributed by the template")
        }
    }

    private func withConfiguredRoot<Result>(_ body: (URL) throws -> Result) throws -> Result {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("training-run-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        StudioTestDefaults.redirectOutputs(under: root, outputs: root)
        defer {
            StudioTestDefaults.restore()
            try? FileManager.default.removeItem(at: root)
        }
        return try body(root)
    }
}
