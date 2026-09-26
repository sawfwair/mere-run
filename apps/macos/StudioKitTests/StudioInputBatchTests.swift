import Foundation
import StudioTestSupport
@testable import StudioKit
import UniformTypeIdentifiers
import XCTest

/// Batch inputs: which slots batch (read from the slot schema), how several files land in one,
/// and how a batch runs — checked once, each file its own run with its own destination, the rows
/// sharing a group, a bad file skipped, and the whole batch stopped. Everything lives in a
/// temporary folder; the process runner records launches and starts nothing.
@MainActor
final class StudioInputBatchTests: XCTestCase {
    private var root: URL!
    private var processRunner: RecordingProcessRunner!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var runner: StudioTaskRunner!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("input-batch-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            StudioTestDefaults.redirectOutputs(under: root)
            processRunner = RecordingProcessRunner()
            controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: processRunner, resolvesCLIOnInit: false,
                taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
            library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
            library.observe(controller: controller)
            runner = StudioTaskRunner(controller: controller, library: library)
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            controller.terminateAllProcesses()
            StudioTestDefaults.restore()
            runner = nil
            library = nil
            controller = nil
            processRunner = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    private func files(_ names: [String]) throws -> [URL] {
        try names.map { name in
            let url = root.appendingPathComponent(name)
            try Data([0]).write(to: url)
            return url
        }
    }

    // MARK: - Which slots batch

    /// Every slot of every task's well, in both drafts' schemas: only a task's first slot may
    /// batch, and only when it holds one required file that stays in the well, on a Generate or
    /// Analyze task. Lists, folders, per-turn pictures, optional inputs, sessions, and projects
    /// never do; the one-file inputs the pages are about all do.
    func testOnlyARequiredSingleFileFirstSlotOnAGenerateOrAnalyzeTaskBatches() throws {
        var batching: Set<String> = []
        for task in StudioTask.allCases {
            var wells: [(name: String, slots: [StudioAttachmentSlot])] = []
            if let mode = task.mode, mode.task == task { wells.append((mode.rawValue, mode.attachmentSlots)) }
            if task.usesTaskDraft {
                wells += task.variantTemplates.map { ($0.id.rawValue, StudioTaskSchema.slots(for: $0.id)) }
            }
            for well in wells {
                for (index, slot) in well.slots.enumerated() {
                    let expected = index == 0 && !slot.allowsMultiple && slot.isRequired && !slot.isTransient
                        && !slot.acceptedTypes.contains(.folder)
                        && [.generate, .analyze].contains(task.archetype) && task.mode?.isConversational != true
                    XCTAssertEqual(slot.batches, expected, "\(well.name) · \(slot.id)")
                    if slot.batches { batching.insert(well.name) }
                }
            }
        }
        for name in ["listen", "readImage", "findObjects", "segment", "track", "audioEnhance", "musicSeparate",
                     "visionDepth", "visionFaceDetect", "speechDiarize"] {
            XCTAssertTrue(batching.contains(name), "\(name) batches; batching: \(batching.sorted())")
        }
        for name in ["createImage", "chat", "speak", "music", "video", "visionFaceBatch", "visionGeometryMultiview",
                     "imageDatasetDiscover", "imageTrainLoRA", "speechListen"] {
            XCTAssertFalse(batching.contains(name), "\(name) keeps one file or its own list")
        }
        XCTAssertEqual(StudioAttachmentRequirement(slot: try XCTUnwrap(StudioTaskSchema.primarySlot(for: .audioEnhance))).allowsMultiple,
                       true, "the open panel and the Library picker take several files for a batching slot")
    }

    // MARK: - Filling a batch

    func testSeveralFilesDroppedOnAOneFileInputBecomeABatchThatGrowsShrinksAndClears() throws {
        let clips = try files(["a.wav", "b.wav", "c.wav"])
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        let slots = draft.slots(source: .contract)
        let slot = try XCTUnwrap(slots.first)

        XCTAssertTrue(draft.attach(dropped: clips.prefix(2) + [root.appendingPathComponent("notes.txt")], slots: slots))
        XCTAssertEqual(draft.batchInputPaths, clips.prefix(2).map(\.path))
        XCTAssertEqual(draft.primaryInputPath, clips[0].path, "the first file is the slot's, for the canvas and Command view")
        XCTAssertEqual(slot.caption(in: draft), "2 files")
        XCTAssertEqual(slots.batchRunCount(in: draft), 2)

        slot.attach([clips[2], clips[0]], to: &draft)
        XCTAssertEqual(slot.runPaths(in: draft), clips.map(\.path), "a batch grows, never repeating a file")

        slot.removeFromBatch(clips[0].path, in: &draft)
        XCTAssertEqual(draft.primaryInputPath, clips[1].path, "the next file moves into the slot")
        slot.removeFromBatch(clips[1].path, in: &draft)
        XCTAssertEqual(draft.batchInputPaths, [], "a batch of one is a plain attachment")
        XCTAssertEqual(draft.primaryInputPath, clips[2].path)
        XCTAssertNil(slots.batchRunCount(in: draft))

        slot.attach([clips[0]], to: &draft)
        XCTAssertEqual(draft.primaryInputPath, clips[0].path, "one file on one file still replaces it")
        slot.attach(clips, to: &draft)
        slot.clear(in: &draft)
        XCTAssertEqual(draft.batchInputPaths, [])
        XCTAssertEqual(draft.primaryInputPath, "")
    }

    /// Two pictures dropped on Compare fill its reference and candidate as they always did; only
    /// files beyond the slots that take them batch. A prompt task's well batches the same way.
    func testDroppedFilesFillOtherSlotsBeforeBatchingAndPromptDraftsBatchToo() throws {
        let pictures = try files(["one.png", "two.png", "three.png"])
        var compare = StudioTaskDraft(templateID: .visionFaceCompare)
        let slots = compare.slots(source: .contract)
        compare.attach(dropped: Array(pictures.prefix(2)), slots: slots)
        XCTAssertEqual(compare.batchInputPaths, [])
        XCTAssertEqual(slots.map { $0.paths(in: compare) }, [[pictures[0].path], [pictures[1].path]])
        compare.attach(dropped: [pictures[2]], slots: slots)
        XCTAssertEqual(compare.batchInputPaths, [], "one more picture replaces the reference, as it always has")
        let more = try files(["four.png", "five.png"])
        compare.attach(dropped: more, slots: slots)
        XCTAssertEqual(compare.batchInputPaths, more.map(\.path), "two more batch the reference")
        XCTAssertEqual(slots[1].paths(in: compare), [pictures[1].path], "and leave the candidate")

        let clips = try files(["x.m4a", "y.m4a"])
        var listen = StudioDraft()
        XCTAssertTrue(listen.attach(dropped: clips, for: .listen, source: .contract))
        XCTAssertEqual(listen.batchInputPaths, clips.map(\.path))
        XCTAssertEqual(listen.inputPath, clips[0].path)
        var face = StudioTaskDraft(templateID: .visionFaceBatch)
        face.attach(dropped: pictures, slots: face.slots(source: .contract))
        XCTAssertEqual(face.batchInputPaths, [], "a list slot keeps its own list")
        XCTAssertEqual(face.primaryInputPath, pictures[0].path)
    }

    /// Saved drafts and Library rows from before batches read unchanged; a draft with a batch
    /// round-trips it; a variant switch keeps the batch only where the new input takes it.
    func testBatchesAreAdditiveInSavedStateAndFollowAVariantThatTakesThem() throws {
        let fresh = StudioTaskDraft(templateID: .audioEnhance)
        let legacy = try JSONEncoder.mereRunApp.encode(fresh)
        XCTAssertFalse(String(decoding: legacy, as: UTF8.self).contains("batchInputPaths"), "a single run writes no key")
        XCTAssertEqual(try JSONDecoder.mereRunApp.decode(StudioTaskDraft.self, from: legacy), fresh)
        XCTAssertNil(try JSONDecoder.mereRunApp.decode(StudioDraft.self, from: JSONEncoder.mereRunApp.encode(StudioDraft())).batchInputs)

        let pictures = try files(["p.png", "q.png"])
        var faces = StudioTaskDraft(templateID: .visionFaceDetect)
        faces.attach(dropped: pictures, slots: faces.slots(source: .contract))
        let decoded = try JSONDecoder.mereRunApp.decode(StudioTaskDraft.self, from: JSONEncoder.mereRunApp.encode(faces))
        XCTAssertEqual(decoded.batchInputPaths, pictures.map(\.path))
        faces.switchTemplate(to: .visionFaceEmbed)
        XCTAssertEqual(faces.batchInputPaths, pictures.map(\.path), "Embed takes one picture per run too")
        faces.switchTemplate(to: .visionFaceBatch)
        XCTAssertEqual(faces.batchInputPaths, [], "Batch takes its pictures as its own list")

        let row = StudioLibraryItem(
            id: UUID(), mode: .listen, prompt: "", inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "mere.run speech transcribe", outputText: nil
        )
        let written = try JSONEncoder.mereRunApp.encode(row)
        XCTAssertFalse(String(decoding: written, as: UTF8.self).contains("batchGroup"), "a single run's row is written as before")
        XCTAssertNil(try JSONDecoder.mereRunApp.decode(StudioLibraryItem.self, from: written).batchGroup)
    }

    func testAddingAndRemovingBatchFilesAreNamedUndoSteps() throws {
        let clips = try files(["a.wav", "b.wav", "c.wav"])
        let manager = UndoManager()
        let sessions = controller.taskSessions
        sessions.undo.manager = manager
        var draft = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        let slot = try XCTUnwrap(draft.slots(source: .contract).first)

        slot.attach(clips, to: &draft)
        sessions.setTaskDraft(draft, for: .audioEnhance)
        RunLoop.current.run(until: Date())
        XCTAssertEqual(manager.undoActionName, "Add Files")
        slot.removeFromBatch(clips[1].path, in: &draft)
        sessions.setTaskDraft(draft, for: .audioEnhance)
        RunLoop.current.run(until: Date())
        XCTAssertEqual(manager.undoActionName, "Remove File")

        manager.undo()
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.batchInputPaths, clips.map(\.path))
        manager.undo()
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.batchInputPaths, [])
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.primaryInputPath, "")
    }

    // MARK: - Running a batch

    func testABatchRunsOncePerFileInOrderWithItsOwnDestinationAndOneGroup() throws {
        controller.readinessByTask[.audioEnhance] = .ready
        let clips = try files(["intro.wav", "interview.wav", "outro.wav"])
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.attach(dropped: clips, slots: draft.slots(source: .contract))

        let review = try XCTUnwrap(runner.reviewBatch(draft, task: .audioEnhance))
        XCTAssertEqual(review.decision, .runAll)
        XCTAssertTrue(library.items.isEmpty, "checking submits nothing")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("outputs").path), "nor creates a folder")

        let submission = runner.runBatch(draft, task: .audioEnhance, paths: review.runnable)
        XCTAssertEqual(submission.requests.count, 3)
        XCTAssertEqual(submission.failures, [])
        let outputs = submission.requests.map(\.draft.outputPath)
        XCTAssertEqual(Set(outputs).count, 3, "each file writes its own result: \(outputs)")
        for (output, name) in zip(outputs, ["intro", "interview", "outro"]) {
            XCTAssertTrue(URL(fileURLWithPath: output).lastPathComponent.hasPrefix(name), output)
        }
        let rows = submission.requests.compactMap { request in library.items.first { $0.id == request.id } }
        XCTAssertEqual(rows.map(\.inputURL?.lastPathComponent), ["intro.wav", "interview.wav", "outro.wav"])
        XCTAssertEqual(Set(rows.map(\.batchGroup)), [submission.group])
        let lane = controller.jobs.running(in: .inference) + controller.jobs.queued(in: .inference)
        XCTAssertEqual(lane.map(\.request.requestID), submission.requests.map(\.id), "the lane takes the files in the batch's order")
        XCTAssertEqual(controller.jobs.queued(in: .inference).count, 3 - JobLane.inference.capacity, "the rest wait in the queue")

        let progress = try XCTUnwrap(runner.activeBatch(for: .audioEnhance))
        XCTAssertEqual(progress.group, submission.group)
        XCTAssertEqual(progress.summary, "0 of 3 done")
        XCTAssertEqual(progress.title, "Enhance · 3 files")

        let reloaded = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        XCTAssertEqual(Set(reloaded.items.map(\.batchGroup)), [submission.group], "the group is saved with the rows")
    }

    func testReadinessIsCheckedOnceAndABatchThatCannotRunSubmitsNothing() throws {
        let clips = try files(["a.wav", "b.wav"])
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.attach(dropped: clips, slots: draft.slots(source: .contract))
        controller.readinessByTask[.audioEnhance] = .missingModel("audio-enhance-ap-bwe-16kto48k")
        XCTAssertThrowsError(try runner.reviewBatch(draft, task: .audioEnhance)) { error in
            XCTAssertTrue(error.localizedDescription.contains("isn't on this Mac"), error.localizedDescription)
        }

        controller.readinessByTask[.audioEnhance] = .ready
        draft.form["--overlap"] = .text("not-a-number")
        let review = try XCTUnwrap(runner.reviewBatch(draft, task: .audioEnhance))
        XCTAssertEqual(review.decision, .refuse("Neither file can run. AP-BWE overlap must be a whole number."))
        XCTAssertNil(try runner.reviewBatch(StudioTaskDraft(templateID: .audioEnhance), task: .audioEnhance), "no batch, no review")
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(processRunner.starts.isEmpty)
    }

    func testAFileThatCannotRunIsNamedBeforeSubmittingAndCanBeSkipped() throws {
        controller.readinessByTask[.audioEnhance] = .ready
        let clips = try files(["good.wav", "fine.wav"])
        let missing = root.appendingPathComponent("moved.wav")
        let folder = root.appendingPathComponent("stems.wav", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        let slot = try XCTUnwrap(draft.slots(source: .contract).first)
        slot.setBatch([clips[0].path, missing.path, folder.path, clips[1].path], in: &draft)

        let review = try XCTUnwrap(runner.reviewBatch(draft, task: .audioEnhance))
        XCTAssertEqual(review.decision, .confirmSkipping)
        XCTAssertEqual(review.rejected.map(\.fileName), ["moved.wav", "stems.wav"])
        XCTAssertEqual(review.confirmationTitle, "2 of 4 files can't run")
        XCTAssertEqual(review.confirmationMessage(), "moved.wav: It's no longer on disk.\nstems.wav: It's a folder, not a file.")
        XCTAssertEqual(review.skipTitle, "Skip and run 2")
        XCTAssertTrue(library.items.isEmpty, "nothing runs until the user says so")

        let submission = runner.runBatch(draft, task: .audioEnhance, paths: review.runnable)
        XCTAssertEqual(submission.requests.map { URL(fileURLWithPath: $0.draft.inputPath).lastPathComponent }, ["good.wav", "fine.wav"])
        XCTAssertEqual(library.items.count, 2)
    }

    func testAPromptTasksBatchRunsThroughTheRunnerWithReadinessOnce() throws {
        let prompt = StudioPromptTaskController(controller: controller, library: library)
        _ = prompt.activate(.listen, preferredID: nil)
        controller.readinessByMode[.listen] = .ready
        let clips = try files(["monday.m4a", "tuesday.m4a"])
        var draft = prompt.draft
        draft.attach(dropped: clips + [root.appendingPathComponent("gone.m4a")], for: .listen, source: .contract)
        prompt.draft = draft

        let review = try XCTUnwrap(prompt.reviewPromptBatch())
        XCTAssertEqual(review.rejected.map(\.fileName), ["gone.m4a"])
        let submission = try XCTUnwrap(prompt.runPromptBatch(paths: review.runnable))
        XCTAssertEqual(submission.requests.map { URL(fileURLWithPath: $0.draft.inputPath).lastPathComponent }, ["monday.m4a", "tuesday.m4a"])
        XCTAssertEqual(Set(submission.requests.map(\.draft.outputPath)).count, 2)
        XCTAssertEqual(Set(library.items.map(\.batchGroup)), [submission.group])
        XCTAssertEqual(runner.activeBatch(for: .audioTranscribe)?.total, 2)

        controller.readinessByMode[.listen] = .notChecked
        XCTAssertThrowsError(try prompt.reviewPromptBatch(), "readiness gates the batch once")
    }

    /// Stop batch takes the waiting runs out of the queue and stops the running one; finished
    /// runs and runs outside the batch are left alone.
    func testStoppingABatchCancelsItsQueuedAndRunningRunsOnly() async throws {
        controller.readinessByTask[.audioEnhance] = .ready
        let clips = try files(["one.wav", "two.wav", "three.wav", "four.wav"])
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.attach(dropped: Array(clips.prefix(3)), slots: draft.slots(source: .contract))
        let batch = runner.runBatch(draft, task: .audioEnhance, paths: clips.prefix(3).map(\.path))
        var single = StudioTaskDraft(templateID: .audioEnhance)
        single.setArgument(0, clips[3].path)
        let other = try runner.run(single, task: .audioEnhance)

        processRunner.starts[0].termination(0)
        for _ in 0..<6 { await Task.yield() }
        XCTAssertEqual(runner.activeBatch(for: .audioEnhance)?.summary, "1 of 3 done")

        runner.stopBatch(batch.group)
        // The stopped processes report their exit, as SIGTERM's would.
        for index in processRunner.processes.indices where processRunner.processes[index].terminateCallCount > 0 {
            processRunner.starts[index].termination(15)
        }
        for _ in 0..<6 { await Task.yield() }

        let rows = batch.requests.compactMap { request in library.items.first { $0.id == request.id } }
        XCTAssertEqual(rows.map(\.status), [.completed, .cancelled, .cancelled])
        XCTAssertNil(runner.activeBatch(for: .audioEnhance))
        XCTAssertEqual(runner.batches().first?.summary, "1 of 3 done · 2 didn't finish")
        XCTAssertTrue(controller.jobs.job(requestID: other.id)?.state.isActive == true, "a run outside the batch keeps going")
    }
}
