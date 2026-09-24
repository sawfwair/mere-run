@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import XCTest

final class StudioSpecialistPageTests: XCTestCase {
    /// `sfx clap` prints one JSON object; the score comes from its `score` field, wherever that
    /// sits in the object and whatever the log around it says.
    func testTheCLAPScoreIsReadFromTheJSONByName() {
        XCTAssertEqual(
            StudioCLAPScore.parse("Loading model 2.1 GB\n{\"prompt\":\"door slam\",\"score\":0.42,\"audio\":\"/a.wav\",\"model\":\"m\"}\n"),
            0.42
        )
        XCTAssertNil(StudioCLAPScore.parse("error: model missing after 3 tries"))
    }

    /// A page's Run with an incomplete form goes through the runner shim and still ends as a
    /// failed Library row the page's result view shows, rather than vanishing.
    @MainActor
    func testTheSpecialistRunnerShimRecordsAnInvalidRunAsAFailedRow() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("shim-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let processRunner = RecordingProcessRunner()
        let controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: processRunner, resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        defer { controller.terminateAllProcesses() }
        let library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        library.observe(controller: controller)
        let template = try XCTUnwrap(CommandCatalog.template(id: .geoFlood))
        var draft = template.defaultDraft()
        draft.outputPath = root.appendingPathComponent("flood.safetensors").path

        let id = try XCTUnwrap(StudioSpecialistRunner.submit(
            templateID: .geoFlood, mode: .readImage, draft: draft, controller: controller, library: library
        ))
        for _ in 0..<6 { await Task.yield() }

        let row = try XCTUnwrap(library.items.first { $0.id == id })
        XCTAssertEqual(row.status, .failed)
        XCTAssertEqual(row.mode, .readImage, "the page's attribution is kept")
        XCTAssertEqual(row.outputText?.contains("required"), true, row.outputText ?? "")
        XCTAssertTrue(processRunner.starts.isEmpty)
        XCTAssertEqual(controller.taskSessions.value(for: StudioTask.earthFlood.rawValue + ".requestID", default: Optional<UUID>.none), id)
    }

    /// The checklists match what each command validates.
    func testEarthChecklistsMatchTheCommands() {
        XCTAssertEqual(StudioGeoTool.flood.tensorRequirement, .init(required: ["S2L2A", "S1RTC", "DEM"]))
        XCTAssertEqual(
            StudioGeoTool.tessera.tensorRequirement,
            .init(required: ["S2", "S2_DOY"], oneOf: ["S1_ASC + S1_ASC_DOY", "S1_DESC + S1_DESC_DOY"])
        )
        XCTAssertEqual(StudioGeoTool.olmoEarth.tensorRequirement, .init(required: ["TIMESTAMPS"], oneOf: ["S2L2A", "S1RTC", "LANDSAT"]))
    }
}
