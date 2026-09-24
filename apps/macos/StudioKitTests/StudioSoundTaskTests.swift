@testable import StudioKit
import MereRunContract
import UniformTypeIdentifiers
import XCTest

/// The Sound tasks on the shared task workspace: Video Foley and Condition generate; Encode,
/// Decode, and Score analyze. The task draft sends what the SFX Lab page sent for the same
/// settings, the page's drafts import once, renoise keeps the CLI's objections in front of the
/// run, and a CLAP result reads for the gauge.
final class StudioSoundTaskTests: XCTestCase {
    private let soundTasks: [StudioTask] = [.soundFoley, .soundCondition, .soundEncode, .soundDecode, .soundScore]

    func testTheSoundTasksRenderOnTheSharedWorkspace() {
        for task in soundTasks {
            XCTAssertFalse(task.usesLegacyPage, "\(task) still has a page")
            XCTAssertTrue(task.usesTaskDraft, "\(task)")
            XCTAssertTrue(task.showsPromptChrome, "\(task)")
            XCTAssertEqual(task.variantTemplates.count, 1, "\(task) has one template, so no variant chip")
        }
        XCTAssertEqual(StudioTask.soundFoley.archetype, .generate)
        XCTAssertEqual(StudioTask.soundCondition.archetype, .generate)
        XCTAssertNil(StudioTask.soundFoley.analyzeArchetype)
        XCTAssertEqual(StudioTask.soundFoley.generateArchetype?.primaryOutput, .audio)
        XCTAssertEqual(StudioTask.soundScore.analyzeArchetype?.views, [.score])
        XCTAssertEqual(StudioTask.soundEncode.analyzeArchetype?.views, [.tensor, .json])
        XCTAssertEqual(StudioTask.soundDecode.analyzeArchetype?.inputKind, .file)
        XCTAssertEqual(StudioTask.soundDecode.analyzeArchetype?.views, [.audio, .json])
    }

    // MARK: Slots and prompts

    func testTheWellAndPromptFollowTheSFXContracts() throws {
        let foley = StudioTaskSchema.slots(for: .sfxVideo)
        XCTAssertEqual(foley.map(\.id), ["input"])
        XCTAssertEqual(foley.first?.storage, .argument(1))
        XCTAssertEqual(foley.first?.isRequired, true)
        XCTAssertEqual(foley.first?.acceptedTypes, [.movie, .video, .audiovisualContent], "the template's own video input")
        XCTAssertTrue(StudioTaskSchema.slots(for: .sfxConditionText).isEmpty, "Condition takes only words")
        XCTAssertEqual(StudioTaskSchema.slots(for: .sfxAEEncode).map(\.acceptedTypes), [[.audio]])
        XCTAssertEqual(StudioTaskSchema.slots(for: .sfxAEDecode).map(\.acceptedTypes), [[.data]])
        XCTAssertEqual(StudioTaskSchema.slots(for: .sfxClapScore).map(\.storage), [.argument(1)])
        for templateID in [CommandTemplateID.sfxVideo, .sfxConditionText, .sfxClapScore] {
            let capability = try XCTUnwrap(templateID.capability)
            XCTAssertEqual(StudioTaskSchema.promptField(for: capability), .argument(0, repeatable: false), "\(templateID)")
        }
        XCTAssertNil(StudioTaskSchema.promptField(for: try XCTUnwrap(CommandTemplateID.sfxAEEncode.capability)))
        XCTAssertNil(StudioTaskSchema.promptField(for: try XCTUnwrap(CommandTemplateID.sfxAEDecode.capability)))
        XCTAssertEqual(StudioTask.soundFoley.presentation.attaching(foley.first).attachLabel, "Choose video…")
        XCTAssertEqual(StudioTask.soundDecode.presentation.attaching(StudioTaskSchema.primarySlot(for: .sfxAEDecode)).attachLabel, "Choose file…")
    }

    /// Foley's inspector shows renoise once, as its editor, and neither the destination routing
    /// fills nor the machine-output switch the launcher owns; the page's Preflight toggle stays.
    func testFoleyShowsRenoiseAsItsEditorAndHidesWhatRoutingOwns() {
        let fields = StudioTaskSchema.fields(for: .soundFoley, draft: StudioTaskDraft(templateID: .sfxVideo))
        XCTAssertEqual(fields.filter { $0.overrideID == .renoise }.count, 1)
        XCTAssertEqual(fields.first { $0.overrideID == .renoise }?.bindings.map(\.fieldID), ["--renoise"])
        XCTAssertFalse(fields.contains { ["--output", "--json"].contains($0.flag) })
        XCTAssertTrue(fields.contains { $0.flag == "--preflight" })
        XCTAssertTrue(fields.contains { $0.flag == "--synchformer-model" }, "the Synchformer model stays reachable")
        XCTAssertTrue(fields.contains { $0.flag == "--sync-batch-size" })
        XCTAssertTrue(fields.contains { $0.flag == "--clip-batch-size" })
    }

    // MARK: Argv parity

    /// The page's Video Foley form and a task draft holding the same settings send one argv,
    /// and the seeded draft is exactly that form read back.
    func testAFoleyTaskDraftSendsWhatThePageSent() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxVideo))
        var page = template.defaultDraft()
        page.prompt = "footsteps on wet gravel"
        page.inputPath = "/tmp/walk.mp4"
        page.secondaryText = "music, speech"
        page.model = "sfx-woosh-dvflow-8s"
        page.durationSeconds = 6
        page.steps = 8
        page.cfgScale = 3.5
        page.seed = "11"
        page.sfxRenoise = "0.3,0.3,0.3,0.3,0.2,0.2,0.1,0.1"
        page.sfxSynchformerModel = "sfx-woosh-synchformer"
        page.sfxSyncBatchSize = 2
        page.sfxClipBatchSize = 8
        page.outputPath = "/tmp/out/walk.wav"

        var draft = StudioTaskDraft(templateID: .sfxVideo)
        draft.prompt = "footsteps on wet gravel"
        draft.setArgument(1, "/tmp/walk.mp4")
        draft.form["--negative-prompt"] = .text("music, speech")
        draft.model = "sfx-woosh-dvflow-8s"
        draft.form["--duration"] = .number(6)
        draft.form["--steps"] = .integer(8)
        draft.form["--cfg"] = .number(3.5)
        draft.form["--seed"] = .integer(11)
        draft.form["--renoise"] = .text("0.3,0.3,0.3,0.3,0.2,0.2,0.1,0.1")
        draft.form["--synchformer-model"] = .text("sfx-woosh-synchformer")
        draft.form["--sync-batch-size"] = .integer(2)
        draft.form["--clip-batch-size"] = .integer(8)
        draft.form["--output"] = .text("/tmp/out/walk.wav")

        // The console emits options in the contract's order; the page's builder had its own.
        // The CLI reads either, so the settings are compared, not the order.
        XCTAssertEqual(Self.settings(of: draft.arguments), Self.settings(of: template.arguments(from: page)))
        XCTAssertEqual(
            StudioTaskDraft(templateID: .sfxVideo, form: StudioConsoleCommand.seed(template: template, draft: page)).arguments,
            draft.arguments,
            "the page's draft read back is the same command"
        )
        let request = try XCTUnwrap(draft.request())
        XCTAssertEqual(request.mode, .sfx, "the template files under Sound, as the page did")
        XCTAssertEqual(request.execution?.arguments, draft.arguments)
    }

    /// The positionals in order, then each option with its value, sorted: what the CLI reads,
    /// independent of the order the flags were written in.
    private static func settings(of argv: [String]) -> [String] {
        var positionals: [String] = []
        var options: [String] = []
        var index = 0
        while index < argv.count {
            let token = argv[index]
            if token.hasPrefix("--") {
                let next = index + 1 < argv.count ? argv[index + 1] : ""
                if next.hasPrefix("--") || next.isEmpty {
                    options.append(token)
                    index += 1
                } else {
                    options.append("\(token) \(next)")
                    index += 2
                }
            } else {
                positionals.append(token)
                index += 1
            }
        }
        return positionals + options.sorted()
    }

    func testTheOtherSoundDraftsSendWhatTheirPagesSent() throws {
        for (templateID, edit) in [
            (CommandTemplateID.sfxConditionText, { (draft: inout CommandDraft) in
                draft.prompt = "an enormous stone door grinding open"
                draft.outputPath = "/tmp/out/door.safetensors"
            }),
            (.sfxAEEncode, { draft in
                draft.inputPath = "/tmp/hit.wav"
                draft.outputPath = "/tmp/out/hit.npy"
            }),
            (.sfxAEDecode, { draft in
                draft.inputPath = "/tmp/hit.npy"
                draft.outputPath = "/tmp/out/hit.wav"
            }),
            (.sfxClapScore, { draft in
                draft.prompt = "a clean glass bottle breaking on concrete"
                draft.inputPath = "/tmp/bottle.wav"
                draft.model = "sfx-woosh-clap"
            }),
        ] as [(CommandTemplateID, (inout CommandDraft) -> Void)] {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var page = template.defaultDraft()
            edit(&page)
            let draft = StudioTaskDraft(templateID: templateID, form: StudioConsoleCommand.seed(template: template, draft: page))
            XCTAssertEqual(Self.settings(of: draft.arguments), Self.settings(of: template.arguments(from: page)), "\(templateID)")
            XCTAssertEqual(draft.request()?.mode, .sfx, "\(templateID)")
        }
    }

    // MARK: Renoise

    /// The CLI accepts one renoise amount or exactly `--steps` amounts; the shared validation
    /// refuses anything else before a run is recorded, for Foley and for Sound ▸ Generate alike.
    @MainActor
    func testRenoiseProblemsKeepTheRunFromStarting() throws {
        var draft = StudioTaskDraft(templateID: .sfxVideo)
        draft.prompt = "a door"
        draft.setArgument(1, "/tmp/door.mp4")
        draft.form["--steps"] = .integer(4)
        let capability = try XCTUnwrap(draft.capability)
        func message() -> String? { StudioConsoleCommand.validationMessage(for: capability, draft: draft.form) }

        XCTAssertNil(message(), "automatic")
        draft.form["--renoise"] = .text("0.4")
        XCTAssertNil(message(), "one amount")
        draft.form["--renoise"] = .text("0.1,0.2,0.3,0.4")
        XCTAssertNil(message(), "one amount per step")
        draft.form["--renoise"] = .text("0.1,0.2,0.3")
        XCTAssertEqual(message(), "The renoise schedule has 3 values but the run has 4 steps.")
        draft.form["--renoise"] = .text("1.5")
        XCTAssertEqual(message(), "Renoise must be between 0 and 1.")
        draft.form["--renoise"] = .text("0,5")
        XCTAssertEqual(message(), "The renoise schedule has 2 values but the run has 4 steps.", "a comma decimal is two tokens to the CLI")
        draft.form["--steps"] = .unset
        draft.form["--renoise"] = .text("0.1,0.2,0.3")
        XCTAssertNil(message(), "with the step count left to the CLI only the amounts are checked")
        draft.form["--renoise"] = .text("0.1,2")
        XCTAssertEqual(message(), "Renoise values must be between 0 and 1.")

        // The runner refuses the same draft before anything is created or recorded.
        draft.form["--steps"] = .integer(4)
        draft.form["--renoise"] = .text("0.1,0.2,0.3")
        let request = try XCTUnwrap(StudioOutputLocation.destination(for: draft).request())
        XCTAssertThrowsError(try StudioTaskRunner.prepare(request, sessions: StudioTaskSessions())) { error in
            XCTAssertEqual(
                error as? StudioValidationError,
                StudioValidationError(message: "The renoise schedule has 3 values but the run has 4 steps.")
            )
        }

        var generate = StudioTaskDraft(templateID: .sfxGenerate)
        generate.prompt = "a whoosh"
        generate.form["--renoise"] = .text("half")
        XCTAssertEqual(
            StudioConsoleCommand.validationMessage(for: try XCTUnwrap(generate.capability), draft: generate.form),
            "Renoise amounts must be numbers separated by commas, with a point for decimals."
        )
    }

    // MARK: Import

    /// The SFX Lab page's per-template drafts seed the task drafts once; the renoise mode is
    /// read back from the imported argument.
    @MainActor
    func testTheSFXLabDraftsSeedTheTaskDraftsOnce() throws {
        let sessions = StudioTaskSessions()
        var page = try XCTUnwrap(CommandCatalog.template(id: .sfxVideo)).defaultDraft()
        page.prompt = "rain on a tent"
        page.inputPath = "/tmp/tent.mov"
        page.sfxRenoise = "0.25"
        page.sfxSyncBatchSize = 3
        sessions.set(page, for: StudioTask.soundFoley.rawValue + ".SFXLab.videoDraft")

        let imported = try XCTUnwrap(sessions.taskDraft(for: .soundFoley))
        XCTAssertEqual(imported.templateID, .sfxVideo)
        XCTAssertEqual(imported.prompt, "rain on a tent")
        XCTAssertEqual(imported.primaryInputPath, "/tmp/tent.mov")
        XCTAssertEqual(imported.text("--renoise"), "0.25")
        XCTAssertEqual(imported.text("--sync-batch-size"), "3")
        XCTAssertEqual(StudioRenoise.resolvedMode(stored: .automatic, argument: imported.text("--renoise")), .amount,
                       "the override shows an imported amount as Fixed amount")

        XCTAssertEqual(StudioTaskDraftMigration.legacyKey(for: .sfxConditionText), "SFXLab.conditionDraft")
        XCTAssertEqual(StudioTaskDraftMigration.legacyKey(for: .sfxAEEncode), "SFXLab.encodeDraft")
        XCTAssertEqual(StudioTaskDraftMigration.legacyKey(for: .sfxAEDecode), "SFXLab.decodeDraft")
        XCTAssertEqual(StudioTaskDraftMigration.legacyKey(for: .sfxClapScore), "SFXLab.scoreDraft")
        XCTAssertEqual(sessions.taskDraft(for: .soundScore)?.templateID, .sfxClapScore, "a fresh draft where no page draft exists")
    }

    // MARK: Library

    /// A row the page filed under Sound ▸ Generate's mode still feeds its own task, and "Use these
    /// settings" reads it back into the task draft.
    @MainActor
    func testPageRowsFeedTheirTaskAndRestoreItsDraft() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxVideo))
        var page = template.defaultDraft()
        page.prompt = "hooves on cobbles"
        page.inputPath = "/tmp/horse.mp4"
        page.sfxRenoise = "0.2"
        page.outputPath = "/tmp/horse.wav"
        func makeRow(_ templateID: CommandTemplateID) -> StudioLibraryItem {
            StudioLibraryItem(
                id: UUID(), mode: .sfx, prompt: page.prompt, inputURL: URL(fileURLWithPath: page.inputPath),
                outputURL: URL(fileURLWithPath: page.outputPath), createdAt: Date(), updatedAt: Date(),
                status: .completed, exitCode: 0, commandPreview: "mere.run sfx video generate …", outputText: nil,
                templateID: templateID, commandDraft: page, commandArguments: template.arguments(from: page),
                artifactURLs: [URL(fileURLWithPath: page.outputPath)]
            )
        }
        let row = makeRow(.sfxVideo)
        let encodeRow = makeRow(.sfxAEEncode)

        XCTAssertEqual(StudioFeedCardBuilder.cards(items: [row, encodeRow], task: .soundFoley) { _ in nil }.map(\.id), [row.id])
        XCTAssertEqual(StudioFeedCardBuilder.cards(items: [row, encodeRow], task: .soundEncode) { _ in nil }.map(\.id), [encodeRow.id])
        XCTAssertTrue(StudioLibraryDraftRestoration.canRestore(row))
        let restored = try XCTUnwrap(StudioLibraryDraftRestoration.taskDraft(from: row))
        XCTAssertEqual(restored.templateID, .sfxVideo)
        XCTAssertEqual(restored.prompt, "hooves on cobbles")
        XCTAssertEqual(restored.primaryInputPath, "/tmp/horse.mp4")
        XCTAssertEqual(restored.text("--renoise"), "0.2")
        XCTAssertEqual(Self.settings(of: restored.arguments), Self.settings(of: template.arguments(from: page)))
    }

    // MARK: Results

    func testCLAPOutputDecodesForTheGauge() throws {
        let text = "Loading model 2.1 GB\n{\"prompt\":\"door slam\",\"score\":0.42,\"audio\":\"/a.wav\",\"model\":\"sfx-woosh-clap\"}\n"
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(text.utf8)))
        guard case .clap(let output) = document else { return XCTFail("The CLAP result read as \(document)") }
        XCTAssertEqual(output.score, 0.42)
        XCTAssertEqual(output.prompt, "door slam")
        XCTAssertEqual(document.modelID, "sfx-woosh-clap")
        XCTAssertEqual(document.summary(detectionCount: 0), "CLAP score 0.42")
        XCTAssertNil(StudioAnalyzeDocument.decode(Data("error: model missing after 3 tries".utf8)).flatMap { document -> StudioCLAPScore.Output? in
            if case .clap(let output) = document { return output }
            return nil
        })
    }

    func testEncodeAndConditionOutputsReadAsTensorHeaders() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sound-tensors-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let npy = root.appendingPathComponent("hit.npy")
        try Self.npy(descriptor: "<f4", shape: "(1, 128, 87)").write(to: npy)
        guard case .npy(let metadata)? = StudioTensorHeader.load(from: npy) else { return XCTFail("not read as .npy") }
        XCTAssertEqual(metadata.shape, "(1, 128, 87)")
        XCTAssertEqual(StudioAnalyzeDocumentSource.preferredExtensions(for: .sfxAEEncode), ["npy"])

        let safetensors = root.appendingPathComponent("door.safetensors")
        try Self.safetensors(["text_embeddings": [1, 77, 1_024], "pooled": [1, 1_024]]).write(to: safetensors)
        guard case .safetensors(let header)? = StudioTensorHeader.load(from: safetensors) else { return XCTFail("not read as safetensors") }
        XCTAssertEqual(header.tensors.map(\.name), ["pooled", "text_embeddings"])
        XCTAssertEqual(header.tensors.first?.summary, "F32 [1, 1024]")
        XCTAssertEqual(StudioAnalyzeDocumentSource.preferredExtensions(for: .sfxConditionText), ["safetensors"])
    }

    // MARK: Fixtures

    /// A NumPy 1.0 file with an empty payload of the declared shape.
    static func npy(descriptor: String, shape: String) -> Data {
        var header = "{'descr': '\(descriptor)', 'fortran_order': False, 'shape': \(shape), }"
        let remainder = (16 - ((10 + header.utf8.count + 1) % 16)) % 16
        header += String(repeating: " ", count: remainder) + "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        data.append(UInt8(header.utf8.count & 0xff))
        data.append(UInt8((header.utf8.count >> 8) & 0xff))
        data.append(Data(header.utf8))
        data.append(Data(repeating: 0, count: 16))
        return data
    }

    /// A safetensors file of float32 tensors, named in sorted order the way the format's writers
    /// emit them, with zeroed payloads.
    static func safetensors(_ tensors: [String: [Int]]) -> Data {
        var offset = 0
        var entries: [String] = []
        for (name, shape) in tensors.sorted(by: { $0.key < $1.key }) {
            let bytes = shape.reduce(1, *) * 4
            entries.append("\"\(name)\":{\"dtype\":\"F32\",\"shape\":[\(shape.map(String.init).joined(separator: ","))],\"data_offsets\":[\(offset),\(offset + bytes)]}")
            offset += bytes
        }
        let header = "{\(entries.joined(separator: ","))}"
        var data = Data()
        withUnsafeBytes(of: UInt64(header.utf8.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(Data(header.utf8))
        data.append(Data(repeating: 0, count: offset))
        return data
    }
}
