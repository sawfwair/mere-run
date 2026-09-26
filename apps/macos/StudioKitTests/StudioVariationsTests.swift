import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// "Run variations" submits one command N times through the task runner, each with its own
/// recorded random seed, as one Library group — and is offered only where the command takes a
/// seed for the model it runs. Compare lists each pane's seed, model, and the options that differ.
@MainActor
final class StudioVariationsTests: XCTestCase {
    private var root: URL!
    private var processRunner: RecordingProcessRunner!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var prompt: StudioPromptTaskController!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("variations-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            StudioTestDefaults.redirectOutputs(under: root)
            processRunner = RecordingProcessRunner()
            controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: processRunner, resolvesCLIOnInit: false,
                taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
            controller.modelIdentities.use(nil)
            library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
            library.observe(controller: controller)
            prompt = StudioPromptTaskController(controller: controller, library: library)
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            controller.terminateAllProcesses()
            controller.taskSessions.flush()
            StudioTestDefaults.restore()
            prompt = nil
            library = nil
            controller = nil
            processRunner = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    private func activate(_ mode: StudioMode) {
        _ = prompt.activate(mode, preferredID: nil)
        controller.readinessByMode[mode] = .ready
    }

    private func seed(of arguments: [String]?) -> String? {
        guard let arguments, let flag = arguments.firstIndex(of: "--seed"), flag + 1 < arguments.count else { return nil }
        return arguments[flag + 1]
    }

    private func output(of arguments: [String]?) -> String? {
        guard let arguments, let flag = arguments.firstIndex(of: "--output"), flag + 1 < arguments.count else { return nil }
        return arguments[flag + 1]
    }

    // MARK: Seeds

    func testSeedsAreDistinctAndWithinTheRangeEveryFamilyAccepts() {
        var generator = SeededGenerator(state: 7)
        for count in StudioVariationCount.allCases {
            let seeds = StudioVariations.seeds(count: count.rawValue, using: &generator)
            XCTAssertEqual(seeds.count, count.rawValue)
            XCTAssertEqual(Set(seeds).count, count.rawValue, "no two variations share a seed")
            XCTAssertTrue(seeds.allSatisfy { Int($0).map(StudioVariations.seedRange.contains) == true }, "\(seeds)")
        }
    }

    // MARK: Composer

    func testComposerVariationsSubmitEveryRunWithItsOwnSeedAsOneGroup() throws {
        activate(.createImage)
        prompt.draft.prompt = "A harbor at dusk"
        prompt.draft.seed = "42"
        XCTAssertTrue(prompt.offersVariations)
        let seeds = ["101", "202", "303", "404"]

        let requests = try prompt.runPromptVariations(seeds: seeds).requests

        XCTAssertEqual(requests.count, 4)
        XCTAssertEqual(controller.jobs.all.count, 4, "every variation is a job in the run queue")
        XCTAssertEqual(
            controller.taskSessions.value(for: StudioTask.imageGenerate.rawValue + ".requestID", default: Optional<UUID>.none),
            requests.first?.id, "Stop acts on the run that starts first"
        )
        let rows = requests.compactMap { request in library.items.first { $0.id == request.id } }
        XCTAssertEqual(rows.count, 4, "every variation has its Library row")
        XCTAssertEqual(rows.map { seed(of: $0.commandArguments) }, seeds, "each row records the seed it ran with")
        XCTAssertEqual(rows.map(\.commandDraft?.seed), seeds)
        let group = try XCTUnwrap(rows.first?.variationGroup)
        XCTAssertTrue(rows.allSatisfy { $0.variationGroup == group }, "one group for the whole submission")
        XCTAssertEqual(StudioVariations.members(of: group, in: library.items).map(\.id), requests.map(\.id))
        XCTAssertEqual(Set(rows.compactMap { output(of: $0.commandArguments) }).count, 4, "no variation writes over another")
        XCTAssertTrue(rows.allSatisfy { $0.prompt == "A harbor at dusk" })
        XCTAssertEqual(prompt.draft.seed, "42", "the composer keeps the seed the user chose")
        XCTAssertEqual(
            StudioVariations.positions(in: library.items)[requests[2].id],
            StudioVariationPosition(group: group, index: 3, count: 4)
        )
    }

    /// A seed the Command view pinned is replaced in each run's argv, so the group never runs
    /// one picture four times.
    func testComposerVariationsReplaceASeedTheCommandViewPinned() throws {
        activate(.createImage)
        prompt.draft.prompt = "A lighthouse"
        let base = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: prompt.draft, source: .contract)
        var form = StudioConsoleCommand.seed(template: base.template, draft: base.draft, source: .contract)
        form["--seed"] = .text("7")
        controller.taskSessions.set(StudioTaskCommandState(templateID: base.templateID,
            sourceArguments: base.template.arguments(from: base.draft, source: .contract), form: form),
            for: base.templateID.studioTask.rawValue + ".commandOverride")

        let requests = try prompt.runPromptVariations(seeds: ["11", "22"]).requests

        XCTAssertEqual(requests.map { seed(of: $0.execution?.arguments) }, ["11", "22"])
    }

    func testVariationsAreHiddenWhereTheCommandTakesNoSeed() throws {
        activate(.listen)
        XCTAssertFalse(prompt.offersVariations, "transcription takes no seed")
        activate(.chat)
        XCTAssertFalse(prompt.offersVariations, "a conversation never varies")
        activate(.video)
        XCTAssertTrue(prompt.offersVariations)

        var enhance = StudioTaskDraft(templateID: .audioEnhance)
        XCTAssertFalse(StudioVariations.applies(to: enhance, source: .contract), "AP-BWE, the default, takes no seed")
        enhance.model = "audio-enhance-universr-audio"
        XCTAssertTrue(StudioVariations.applies(to: enhance, source: .contract), "UniverSR does")

        var transcript = StudioLibraryItem(
            id: UUID(), mode: .listen, prompt: "", inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "mere.run speech transcribe memo.wav", outputText: "Hello"
        )
        transcript.templateID = .speechTranscribe
        transcript.commandDraft = CommandCatalog.template(id: .speechTranscribe)?.defaultDraft()
        XCTAssertFalse(StudioVariations.applies(to: transcript, source: .contract))
    }

    // MARK: Library rows

    func testLibraryVariationsReplayTheRecordedCommandWithNewSeeds() throws {
        activate(.createImage)
        prompt.draft.prompt = "A red kite"
        prompt.draft.seed = "5"
        let original = try XCTUnwrap(prompt.runPrompt(inventory: []).map(\.request))
        let row = try XCTUnwrap(library.items.first { $0.id == original.id })
        XCTAssertTrue(StudioVariations.applies(to: row, source: controller.scopeSource))

        let requests = try prompt.runner.replayVariations(of: row, seeds: ["8", "9"])

        let rows = requests.compactMap { request in library.items.first { $0.id == request.id } }
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows.map { seed(of: $0.commandArguments) }, ["8", "9"])
        XCTAssertTrue(rows.allSatisfy { $0.parentID == row.id }, "each variation remembers the run it came from")
        XCTAssertNotNil(rows[0].variationGroup)
        XCTAssertEqual(rows[0].variationGroup, rows[1].variationGroup)
        XCTAssertNil(library.items.first { $0.id == row.id }?.variationGroup, "the original row is not part of the group")
        func unchanged(_ arguments: [String]?) -> [String] {
            var remaining = arguments ?? []
            for flag in ["--seed", "--output"] {
                if let index = remaining.firstIndex(of: flag) { remaining.removeSubrange(index...index + 1) }
            }
            return remaining
        }
        XCTAssertTrue(rows.allSatisfy { unchanged($0.commandArguments) == unchanged(row.commandArguments) },
                      "everything but the seed and the destination is the recorded command")
        XCTAssertFalse(rows.contains { output(of: $0.commandArguments) == output(of: row.commandArguments) })
    }

    // MARK: Task workspace

    func testTaskDraftVariationsRunTheFormOncePerSeed() throws {
        controller.readinessByTask[.audioEnhance] = .ready
        let input = root.appendingPathComponent("voice-memo.wav")
        try Data().write(to: input)
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.setArgument(0, input.path)
        draft.model = "audio-enhance-universr-audio"

        let requests = try prompt.runner.runVariations(draft, task: .audioEnhance, seeds: ["1", "2"])

        XCTAssertEqual(requests.map { seed(of: $0.execution?.arguments) }, ["1", "2"])
        XCTAssertEqual(Set(requests.compactMap { output(of: $0.execution?.arguments) }).count, 2)
        let groups = Set(requests.compactMap { request in library.items.first { $0.id == request.id }?.variationGroup })
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(controller.taskSessions.submittingTask(of: requests[1].id), .audioEnhance)
    }

    // MARK: Persistence

    func testTheGroupSurvivesARelaunchAndOlderRowsDecodeWithoutIt() throws {
        activate(.createImage)
        prompt.draft.prompt = "Two boats"
        let requests = try prompt.runPromptVariations(seeds: ["1", "2"]).requests
        let reloaded = StudioLibraryStore(libraryURL: library.libraryURL)
        XCTAssertEqual(Set(reloaded.items.filter { requests.map(\.id).contains($0.id) }.map(\.variationGroup)).count, 1)
        XCTAssertNotNil(reloaded.items.first?.variationGroup)

        let legacy = Data("""
        [{"id":"\(UUID().uuidString)","mode":"createImage","prompt":"Old","createdAt":"2026-01-01T00:00:00Z",
          "updatedAt":"2026-01-01T00:00:00Z","status":"completed","commandPreview":"mere.run image generate"}]
        """.utf8)
        let legacyURL = root.appendingPathComponent("legacy.json")
        try legacy.write(to: legacyURL)
        let old = StudioLibraryStore(libraryURL: legacyURL)
        XCTAssertEqual(old.items.count, 1)
        XCTAssertNil(old.items[0].variationGroup)
    }
}

/// A deterministic generator for the seed tests (SplitMix64).
private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
