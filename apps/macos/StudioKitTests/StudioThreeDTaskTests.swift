import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// 3D ▸ From image on the shared task workspace: the engine picker is the task's variant, the
/// task draft builds the argv the 3D page built for the same settings, InstantMesh runs are
/// refused without four or six views or with a camera file that does not match them, and the
/// destination is a fresh directory under the 3D folder named after the picture.
@MainActor
final class StudioThreeDTaskTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("three-d-task-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        StudioTestDefaults.redirectOutputs(under: root)
    }

    override func tearDownWithError() throws {
        StudioTestDefaults.restore()
        try FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    // MARK: Engine picker

    func testTheEnginePickerStartsOnTrellisAndOffersTheFourEngines() throws {
        XCTAssertTrue(StudioTask.taskDraftTasks.contains(.threeDFromImage))
        XCTAssertTrue(StudioTask.threeDFromImage.usesTaskDraft)
        XCTAssertEqual(StudioTask.threeDFromImage.archetype, .generate)
        XCTAssertEqual(
            StudioTask.threeDFromImage.variantTemplates.map(\.id),
            [.imageReconstruct3DTrellis2, .imageReconstruct3D, .imageReconstruct3DMultiview, .visionImageTo3DIdol]
        )
        XCTAssertEqual(StudioDomain(templateID: .visionImageTo3DIdol), .threeD)
        let engine = try XCTUnwrap(StudioTaskSchema.variantField(for: .threeDFromImage))
        XCTAssertEqual(engine.label, "Engine")
        XCTAssertEqual(engine.tier, .essential)
        XCTAssertEqual(engine.group, .inputs, "a prompt-less task files its variant with the inputs, not under Prompt")
        XCTAssertEqual(StudioTaskSchema.variantField(for: .visionFaces)?.group, .inputs)
        XCTAssertEqual(StudioModelNaming.displayName("image-3d-triposr", titles: .none), "TripoSR")
        XCTAssertEqual(StudioModelNaming.displayName("image-3d-trellis2-4b", titles: .none), "TRELLIS.2 4B")
        XCTAssertEqual(StudioModelNaming.displayName("image-3d-instantmesh-base", titles: .none), "InstantMesh Base")
        XCTAssertEqual(StudioTaskDraft(task: .threeDFromImage)?.templateID, .imageReconstruct3DTrellis2)
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3DTrellis2).map(\.id), ["input"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3D).map(\.id), ["input"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3DMultiview).map(\.id), ["--view"])
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3DMultiview).first?.acceptedTypes, [.image])
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--cameras", templateID: .imageReconstruct3DMultiview), .cameras)
        XCTAssertNil(StudioTaskSchema.promptField(for: try XCTUnwrap(CommandTemplateID.imageReconstruct3D.capability)), "3D takes no prompt")
    }

    /// Switching engines keeps the picture and the framing the two single-image engines share
    /// and drops the model, which names a different checkpoint per engine.
    func testSwitchingEnginesCarriesThePictureAndClearsTheModel() throws {
        var draft = try XCTUnwrap(StudioTaskDraft(task: .threeDFromImage))
        draft.setArgument(0, "/tmp/chair.png")
        draft.form["--already-framed"] = .flag(true)
        draft.form["--seed"] = .integer(7)
        draft.model = "image-3d-trellis2-4b"

        draft.switchTemplate(to: .imageReconstruct3D)
        XCTAssertEqual(draft.argument(0), "/tmp/chair.png")
        XCTAssertEqual(draft.primaryInputPath, "/tmp/chair.png")
        XCTAssertEqual(draft.form["--already-framed"].flag, true)
        XCTAssertEqual(draft.model, "", "TripoSR runs its own default, not the TRELLIS.2 checkpoint")
        XCTAssertNil(draft.form.values["--seed"], "TripoSR takes no seed")
        XCTAssertEqual(draft.text("--resolution"), "256", "the template's own default")

        draft.switchTemplate(to: .imageReconstruct3DMultiview)
        XCTAssertEqual(draft.form.arguments.filter { !$0.isEmpty }, [], "InstantMesh takes views, not a positional")
        XCTAssertEqual(draft.text("--view"), "")
        XCTAssertEqual(draft.text("--resolution"), "256", "a shared option carries")
        XCTAssertEqual(draft.model, "")
    }

    // MARK: IDOL

    /// IDOL reads three local checkpoint folders rather than a managed model: the inspector draws
    /// each as a folder row, the composer offers no model chip, no installed model stands in for
    /// `--model`, and the run asks the readiness check for nothing.
    func testIdolTakesItsCheckpointsAsFoldersAndNoManagedModel() throws {
        XCTAssertTrue(StudioTaskSchema.takesModelFolder(.visionImageTo3DIdol))
        XCTAssertFalse(StudioTaskSchema.takesModelFolder(.imageReconstruct3DTrellis2))
        XCTAssertNil(StudioTaskSchema.overrideID(forFlag: "--model", templateID: .visionImageTo3DIdol))
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--model", templateID: .imageReconstruct3DTrellis2), .model)
        XCTAssertEqual(StudioTaskSchema.slots(for: .visionImageTo3DIdol).map(\.id), ["input"], "the folders are not well slots")

        var draft = StudioTaskDraft(templateID: .visionImageTo3DIdol)
        let fields = StudioTaskSchema.fields(for: .threeDFromImage, draft: draft, source: .contract)
        for flag in ["--model", "--dino", "--vae"] {
            let field = try XCTUnwrap(fields.first { $0.flag == flag }, flag)
            XCTAssertEqual(field.control, .path, flag)
            XCTAssertEqual(field.kind, .directory, flag)
            XCTAssertEqual(field.group, .model, flag)
        }
        XCTAssertNil(fields.first { $0.flag == "--output" }, "routing fills the destination")

        let rows = [StudioModelInventoryRow(
            id: "image-3d-trellis2-4b", category: "image-3d", status: "installed", size: "1 GB", usageTerms: nil
        )]
        draft.selectInstalledDefault(from: rows, source: .contract)
        XCTAssertEqual(draft.model, "", "no managed model stands in for the IDOL checkpoints")
        XCTAssertNil(StudioTaskSchema.requirement(for: draft, source: .contract))
        draft.model = "/models/idol"
        XCTAssertNil(StudioTaskSchema.requirement(for: draft, source: .contract))
    }

    /// The task form and the catalog builder spell the same command, the folders are required
    /// before anything is created, and switching to IDOL keeps the picture and the framing.
    func testIdolBuildsTheCommandAndRequiresItsFolders() throws {
        var draft = try XCTUnwrap(StudioTaskDraft(task: .threeDFromImage))
        draft.setArgument(0, "/tmp/chair.png")
        draft.form["--already-framed"] = .flag(true)
        draft.model = "image-3d-trellis2-4b"
        draft.switchTemplate(to: .visionImageTo3DIdol)
        XCTAssertEqual(draft.argument(0), "/tmp/chair.png")
        XCTAssertEqual(draft.form["--already-framed"].flag, true)
        XCTAssertEqual(draft.model, "", "a TRELLIS.2 id is not an IDOL checkpoint folder")

        let template = try XCTUnwrap(CommandCatalog.template(id: .visionImageTo3DIdol))
        let run = try XCTUnwrap(StudioConsoleRun(template: template, draft: draft.form, seed: draft.seed, source: .contract))
        XCTAssertEqual(run.validationMessage, "IDOL checkpoints (--model) is required.")

        let output = root.appendingPathComponent("chair-splat").path
        draft.model = "/models/idol"
        draft.form["--dino"] = .text("/models/dinov3")
        draft.form["--vae"] = .text("/models/flux2-vae")
        draft.form["--output"] = .text(output)
        draft.form["--mode"] = .text("real")
        draft.form["--seed"] = .integer(7)
        draft.form["--steps"] = .integer(30)
        draft.form["--num-gaussians"] = .integer(131_072)
        var page = template.defaultDraft()
        page.inputPath = "/tmp/chair.png"
        page.model = "/models/idol"
        page.idolDINOPath = "/models/dinov3"
        page.idolVAEPath = "/models/flux2-vae"
        page.outputPath = output
        page.idolMode = "real"
        page.alreadyFramed = true
        page.seed = "7"
        page.steps = 30
        page.idolGaussianCount = 131_072
        let argv = draft.arguments(source: .contract)
        XCTAssertEqual(Array(argv.prefix(3)), ["vision", "image-to-3d-idol", "/tmp/chair.png"])
        XCTAssertEqual(Self.pairs(argv, template), Self.pairs(template.arguments(from: page, source: .contract), template))
        let filled = try XCTUnwrap(StudioConsoleRun(template: template, draft: draft.form, seed: draft.seed, source: .contract))
        XCTAssertNil(filled.validationMessage)
    }

    // MARK: Argv parity with the 3D page

    /// The page built a `CommandDraft` per engine and ran the catalog builder over it; the task
    /// draft builds the same command from its contract form for the same settings.
    func testTheTaskDraftBuildsThePagesArgvForTheSameSettings() throws {
        let output = root.appendingPathComponent("3d-asset").path
        let tripoSR = try XCTUnwrap(CommandCatalog.template(id: .imageReconstruct3D))
        var pageTripoSR = tripoSR.defaultDraft()
        pageTripoSR.inputPath = "/tmp/chair.png"
        pageTripoSR.outputPath = output
        pageTripoSR.reconstructionResolution = 128
        pageTripoSR.densityThreshold = 30
        pageTripoSR.foregroundRatio = 0.9
        pageTripoSR.alreadyFramed = true
        pageTripoSR.noVertexColors = true
        pageTripoSR.dryRun = true
        pageTripoSR.json = true
        var taskTripoSR = StudioTaskDraft(templateID: .imageReconstruct3D)
        taskTripoSR.setArgument(0, "/tmp/chair.png")
        taskTripoSR.form["--output"] = .text(output)
        taskTripoSR.form["--resolution"] = .integer(128)
        taskTripoSR.form["--density-threshold"] = .number(30)
        taskTripoSR.form["--foreground-ratio"] = .number(0.9)
        taskTripoSR.form["--already-framed"] = .flag(true)
        taskTripoSR.form["--no-vertex-colors"] = .flag(true)
        taskTripoSR.form["--dry-run"] = .flag(true)
        taskTripoSR.form["--json"] = .flag(true)
        XCTAssertEqual(Self.pairs(taskTripoSR.arguments(source: .contract), tripoSR), Self.pairs(tripoSR.arguments(from: pageTripoSR, source: .contract), tripoSR))
        XCTAssertEqual(Array(taskTripoSR.arguments(source: .contract).prefix(3)), ["image", "reconstruct-3d", "/tmp/chair.png"])

        let trellis = try XCTUnwrap(CommandCatalog.template(id: .imageReconstruct3DTrellis2))
        var pageTrellis = trellis.defaultDraft()
        pageTrellis.inputPath = "/tmp/chair.png"
        pageTrellis.outputPath = output
        pageTrellis.seed = "7"
        pageTrellis.trellisTextureSeed = "9"
        pageTrellis.maxTokens = 1_048_576
        pageTrellis.trellisNoRemesh = false
        pageTrellis.trellisRemeshBand = 2
        pageTrellis.trellisSealRadius = 8
        var taskTrellis = StudioTaskDraft(templateID: .imageReconstruct3DTrellis2)
        taskTrellis.setArgument(0, "/tmp/chair.png")
        taskTrellis.form["--output"] = .text(output)
        taskTrellis.form["--seed"] = .integer(7)
        taskTrellis.form["--texture-seed"] = .integer(9)
        taskTrellis.form["--max-tokens"] = .integer(1_048_576)
        taskTrellis.form["--remesh-band"] = .number(2)
        taskTrellis.form["--seal-radius"] = .integer(8)
        XCTAssertEqual(Self.pairs(taskTrellis.arguments(source: .contract), trellis), Self.pairs(trellis.arguments(from: pageTrellis, source: .contract), trellis))

        let views = (1...4).map { "/tmp/view-\($0).png" }
        let instantMesh = try XCTUnwrap(CommandCatalog.template(id: .imageReconstruct3DMultiview))
        var pageInstantMesh = instantMesh.defaultDraft()
        pageInstantMesh.referenceImagePaths = views.joined(separator: "\n")
        pageInstantMesh.outputPath = output
        pageInstantMesh.reconstructionResolution = 256
        pageInstantMesh.camerasPath = "/tmp/cameras.json"
        pageInstantMesh.dryRun = true
        pageInstantMesh.json = true
        var taskInstantMesh = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
        StudioTaskSchema.slots(for: .imageReconstruct3DMultiview)[0].attach(views.map { URL(fileURLWithPath: $0) }, to: &taskInstantMesh)
        taskInstantMesh.form["--output"] = .text(output)
        taskInstantMesh.form["--resolution"] = .integer(256)
        taskInstantMesh.form["--cameras"] = .text("/tmp/cameras.json")
        taskInstantMesh.form["--dry-run"] = .flag(true)
        taskInstantMesh.form["--json"] = .flag(true)
        XCTAssertEqual(Self.pairs(taskInstantMesh.arguments(source: .contract), instantMesh), Self.pairs(instantMesh.arguments(from: pageInstantMesh, source: .contract), instantMesh))
        XCTAssertEqual(taskInstantMesh.arguments(source: .contract).filter { $0 == "--view" }.count, 4)
        // The order of the views is the order the argv carries them.
        let argv = taskInstantMesh.arguments(source: .contract)
        XCTAssertEqual(argv.indices.filter { argv[$0] == "--view" }.map { argv[$0 + 1] }, views)
    }

    // MARK: InstantMesh gates

    /// The page refused to run without four or six views, or with a camera document that did
    /// not match them; the runner (and the Command view, which validates the same form) refuse
    /// the same runs with the same words, before anything is created.
    func testInstantMeshRunsAreRefusedWithoutMatchingViewsAndCameras() throws {
        let views = try (1...4).map { index -> URL in
            let url = root.appendingPathComponent("view-\(index).png")
            try Data().write(to: url)
            return url
        }
        func draft(views: [URL], cameras: URL?) -> StudioTaskDraft {
            var draft = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
            StudioTaskSchema.slots(for: .imageReconstruct3DMultiview)[0].attach(views, to: &draft)
            if let cameras { draft.form["--cameras"] = .text(cameras.path) }
            return draft
        }
        func refusal(_ draft: StudioTaskDraft) throws -> String? {
            let base = try XCTUnwrap(StudioOutputLocation.destination(for: draft, source: .contract).request(source: .contract))
            do {
                _ = try StudioTaskRunner.prepare(base, sessions: StudioTaskSessions(), source: .contract)
                return nil
            } catch let error as StudioValidationError {
                return error.message
            }
        }

        XCTAssertEqual(try refusal(draft(views: Array(views.prefix(3)), cameras: nil)), "Add exactly 4 or 6 ordered source views.")
        XCTAssertEqual(try refusal(draft(views: [], cameras: nil)), "Add exactly 4 or 6 ordered source views.")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("outputs").path),
                       "a refused run creates no destination")
        XCTAssertNil(try refusal(draft(views: views, cameras: nil)), "four views run without cameras")

        let short = root.appendingPathComponent("short.cameras.json")
        try StudioInstantMeshCameraDocument(cameras: (0..<3).map { _ in .example }).json().write(to: short)
        XCTAssertEqual(try refusal(draft(views: views, cameras: short)), "Add one camera per view: 4 views, 3 cameras.")

        let matching = root.appendingPathComponent("matching.cameras.json")
        try StudioInstantMeshCameraDocument(cameras: (0..<4).map { _ in .example }).json().write(to: matching)
        XCTAssertNil(try refusal(draft(views: views, cameras: matching)))

        let broken = root.appendingPathComponent("broken.cameras.json")
        try Data("not cameras".utf8).write(to: broken)
        XCTAssertEqual(try refusal(draft(views: views, cameras: broken))?.hasPrefix("Couldn't read the camera file broken.cameras.json"), true)

        // The Command view's Run validates the same form and shows the same reason.
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageReconstruct3DMultiview))
        let shortDraft = draft(views: views, cameras: short)
        let run = try XCTUnwrap(StudioConsoleRun(template: template, draft: shortDraft.form, seed: shortDraft.seed, source: .contract))
        XCTAssertEqual(run.validationMessage, "Add one camera per view: 4 views, 3 cameras.")
    }

    // MARK: Output routing

    /// Every engine writes a directory; routing names it after the picture under the 3D folder,
    /// and preparing it creates the folder the CLI fills.
    func testTheDestinationIsAFreshDirectoryUnderTheThreeDFolder() throws {
        let picture = root.appendingPathComponent("chair.png")
        try Data().write(to: picture)
        for templateID in [CommandTemplateID.imageReconstruct3DTrellis2, .imageReconstruct3D, .visionImageTo3DIdol] {
            var draft = StudioTaskDraft(templateID: templateID)
            draft.setArgument(0, picture.path)
            if templateID == .visionImageTo3DIdol {
                draft.model = "/models/idol"
                draft.form["--dino"] = .text("/models/dinov3")
                draft.form["--vae"] = .text("/models/flux2-vae")
            }
            let named = StudioOutputLocation.destination(for: draft, source: .contract)
            let output = URL(fileURLWithPath: named.text("--output"))
            XCTAssertEqual(output.deletingLastPathComponent().path, root.appendingPathComponent("outputs/3D").path, "\(templateID)")
            XCTAssertTrue(output.lastPathComponent.hasPrefix("chair"), "\(templateID): \(output.lastPathComponent)")
            XCTAssertEqual(output.pathExtension, "", "a directory, not a file")
            XCTAssertEqual(StudioOutputLocation.destination(for: named, source: .contract).text("--output"), named.text("--output"), "naming is stable")

            let base = try XCTUnwrap(named.request(source: .contract))
            XCTAssertEqual(base.mode, .createImage, "3D rows keep the attribution the page gave them")
            let prepared = try StudioTaskRunner.prepare(base, sessions: StudioTaskSessions(), source: .contract)
            XCTAssertNil(prepared.fallbackReason)
            XCTAssertTrue(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
            let arguments = try XCTUnwrap(prepared.request.execution?.arguments)
            XCTAssertEqual(arguments.firstIndex(of: "--output").map { arguments[$0 + 1] }, output.path)
        }
    }

    // MARK: Library

    /// "Use these settings" on an InstantMesh row brings back every view in order and the camera
    /// file it ran with, and leaves the run's own output directory behind for routing to name anew.
    func testUseTheseSettingsRestoresTheViewsAndCameras() throws {
        let views = (1...6).map { "/tmp/views/\($0).png" }
        var draft = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
        StudioTaskSchema.slots(for: .imageReconstruct3DMultiview)[0].attach(views.map { URL(fileURLWithPath: $0) }, to: &draft)
        draft.form["--cameras"] = .text("/tmp/cameras.json")
        draft.form["--resolution"] = .integer(192)
        let request = try XCTUnwrap(StudioOutputLocation.destination(for: draft, source: .contract).request(source: .contract))
        let row = StudioLibraryItem(
            id: request.id, mode: request.mode, prompt: "", inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "", outputText: nil, templateID: .imageReconstruct3DMultiview,
            commandDraft: request.draft, commandArguments: request.execution?.arguments
        )
        XCTAssertTrue(StudioLibraryDraftRestoration.canRestore(row))
        let restored = try XCTUnwrap(StudioLibraryDraftRestoration.taskDraft(from: row, source: .contract))
        XCTAssertEqual(restored.templateID, .imageReconstruct3DMultiview)
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageReconstruct3DMultiview)[0].paths(in: restored), views)
        XCTAssertEqual(restored.text("--cameras"), "/tmp/cameras.json")
        XCTAssertEqual(restored.text("--resolution"), "192")
        XCTAssertEqual(restored.text("--output"), "", "the run's own directory is not restored")
        let recorded = try XCTUnwrap(request.execution?.arguments)
        let outputIndex = try XCTUnwrap(recorded.firstIndex(of: "--output"))
        var withoutDestination = recorded
        withoutDestination.removeSubrange(outputIndex...(outputIndex + 1))
        XCTAssertEqual(restored.arguments(source: .contract), withoutDestination, "everything but the destination is the recorded command")
    }

    /// The same normalization `StudioTaskSchemaTests` uses: the catalog builder and the contract
    /// emit options in different orders.
    private static func pairs(_ arguments: [String], _ template: CommandTemplate) -> Set<String> {
        let count = template.id.capability?.command.count ?? 2
        let parsed = StudioCommandRows.parse(arguments: arguments, commandPathCount: count)
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
}
