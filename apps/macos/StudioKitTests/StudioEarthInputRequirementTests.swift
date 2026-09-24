@testable import StudioKit
import StudioTestSupport
import XCTest

/// The Earth tasks on the shared task workspace: the tensor checklist says what each command
/// checks, reads an attached bundle's header without the tensors behind it, and words a missing
/// tensor the way the command would refuse it; the task draft builds the argv the Geo page built;
/// TESSERA's dimensions stay a constrained picker; the page's drafts import once.
final class StudioEarthInputRequirementTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("earth-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    // MARK: Checklist

    /// The checklists match what each command validates (`GeoFloodCommand`, `GeoFireCommand`,
    /// `GeoTESSERACommand`, `GeoOlmoEarthCommand`).
    func testEarthChecklistsMatchTheCommands() {
        XCTAssertEqual(StudioEarthInputRequirement.requirement(for: .geoFlood), .init(required: ["S2L2A", "S1RTC", "DEM"]))
        XCTAssertEqual(StudioEarthInputRequirement.requirement(for: .geoFire), .init(required: ["S2L2A", "S1RTC", "DEM"]))
        XCTAssertEqual(
            StudioEarthInputRequirement.requirement(for: .geoTessera),
            .init(required: ["S2", "S2_DOY"], oneOf: [.init("S1_ASC", "S1_ASC_DOY"), .init("S1_DESC", "S1_DESC_DOY")])
        )
        XCTAssertEqual(
            StudioEarthInputRequirement.requirement(for: .geoOlmoEarth),
            .init(required: ["TIMESTAMPS"], oneOf: [.init("S2L2A"), .init("S1RTC"), .init("LANDSAT")])
        )
        XCTAssertNil(StudioEarthInputRequirement.requirement(for: .audioEnhance))
        for task in [StudioTask.earthFlood, .earthFire, .earthTessera, .earthOlmoEarth] {
            XCTAssertTrue(StudioTask.migratedTasks.contains(task), "\(task) is on the shared workspace")
            XCTAssertEqual(task.variantTemplates.count, 1, "\(task) has one command; no variant chip")
        }
    }

    /// The empty well's hint names the tensors in one sentence.
    func testTheHintReadsAsOneSentence() throws {
        XCTAssertEqual(try XCTUnwrap(StudioEarthInputRequirement.requirement(for: .geoFlood)).hint, "Needs S2L2A, S1RTC, and DEM.")
        XCTAssertEqual(
            try XCTUnwrap(StudioEarthInputRequirement.requirement(for: .geoTessera)).hint,
            "Needs S2 and S2_DOY, plus S1_ASC + S1_ASC_DOY or S1_DESC + S1_DESC_DOY."
        )
        XCTAssertEqual(
            try XCTUnwrap(StudioEarthInputRequirement.requirement(for: .geoOlmoEarth)).hint,
            "Needs TIMESTAMPS, plus S2L2A, S1RTC, or LANDSAT."
        )
    }

    /// An attached bundle is read against the requirement: a satisfied row carries the tensor's
    /// dtype and shape, a pair counts only when both halves are present, and the message names
    /// what the command will refuse the file for.
    func testTheChecklistReadsAnAttachedBundle() throws {
        let tessera = try XCTUnwrap(StudioEarthInputRequirement.requirement(for: .geoTessera))
        let complete = try XCTUnwrap(StudioSafetensorsHeader.decode(SafetensorsFixture.data(tensors: Self.tesseraTensors)))
        let satisfied = tessera.check(complete)
        XCTAssertTrue(satisfied.isSatisfied)
        XCTAssertNil(satisfied.message)
        XCTAssertEqual(satisfied.required.map(\.title), ["S2", "S2_DOY"])
        XCTAssertEqual(satisfied.required.map(\.detail), ["F32 [1, 4, 10]", "F32 [1, 4]"])
        XCTAssertEqual(satisfied.oneOf.map(\.isPresent), [true, false])
        XCTAssertEqual(satisfied.oneOf[0].detail, "F32 [1, 4, 2] · F32 [1, 4]")

        // Half a pair is no pair: the command throws `incompleteInputPair` for it.
        let halfPair = try XCTUnwrap(StudioSafetensorsHeader.decode(SafetensorsFixture.data(tensors: [
            .float32("S2", shape: [1, 4, 10]), .float32("S2_DOY", shape: [1, 4]), .float32("S1_DESC", shape: [1, 4, 2]),
        ])))
        let unpaired = tessera.check(halfPair)
        XCTAssertFalse(unpaired.isSatisfied)
        XCTAssertTrue(unpaired.oneOf[1].isIncomplete)
        XCTAssertEqual(unpaired.oneOf[1].missing, ["S1_DESC_DOY"])
        XCTAssertEqual(unpaired.message, "S1_DESC needs S1_DESC_DOY.")

        // A complete ascending pair beside a lone descending tensor: the command still refuses
        // the bundle (`incompleteInputPair`), so the checklist must not call it ready.
        let strayDescending = try XCTUnwrap(StudioSafetensorsHeader.decode(SafetensorsFixture.data(
            tensors: Self.tesseraTensors + [.float32("S1_DESC", shape: [1, 4, 2])]
        )))
        let stray = tessera.check(strayDescending)
        XCTAssertFalse(stray.isSatisfied)
        XCTAssertEqual(stray.oneOf.map(\.isPresent), [true, false])
        XCTAssertEqual(stray.oneOf.map(\.isIncomplete), [false, true])
        XCTAssertEqual(stray.message, "S1_DESC needs S1_DESC_DOY.")
        let noOrbit = try XCTUnwrap(StudioSafetensorsHeader.decode(SafetensorsFixture.data(
            tensors: Array(Self.tesseraTensors.prefix(2))
        )))
        XCTAssertEqual(tessera.check(noOrbit).message, "Needs S1_ASC + S1_ASC_DOY or S1_DESC + S1_DESC_DOY.")

        let flood = try XCTUnwrap(StudioEarthInputRequirement.requirement(for: .geoFlood))
        let noDEM = try XCTUnwrap(StudioSafetensorsHeader.decode(SafetensorsFixture.data(tensors: [
            .float32("S2L2A", shape: [1, 12, 4, 8, 8]), .float32("S1RTC", shape: [1, 2, 4, 8, 8]),
        ])))
        XCTAssertEqual(flood.check(noDEM).message, "Missing DEM.")
        XCTAssertEqual(flood.check(noDEM).required.map(\.isPresent), [true, true, false])
        let empty = try XCTUnwrap(StudioSafetensorsHeader.decode(SafetensorsFixture.data(tensors: [.float32("logits", shape: [1])])))
        XCTAssertEqual(flood.check(empty).message, "Missing S2L2A, S1RTC, and DEM.")
        let olmo = try XCTUnwrap(StudioEarthInputRequirement.requirement(for: .geoOlmoEarth))
        XCTAssertEqual(olmo.check(empty).message, "Missing TIMESTAMPS. Needs S2L2A, S1RTC, or LANDSAT.")
    }

    /// The header comes off the front of the file; the tensors behind it are never read, so a
    /// tile bundle of any size is checked in a moment.
    func testTheHeaderIsReadWithoutTheTensors() throws {
        let url = root.appendingPathComponent("tile.safetensors")
        try SafetensorsFixture.write(to: url, tensors: Self.tesseraTensors, metadata: ["source": "test"])
        // Append junk after the tensors: a reader that swallowed the whole file would still
        // decode, but `byteCount` proves the header path reports the file's size, not its read.
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(repeating: 0, count: 4_096))
        try handle.close()
        let header = try XCTUnwrap(StudioSafetensorsHeader.loadHeader(from: url))
        XCTAssertEqual(header.tensors.map(\.name), ["S2", "S2_DOY", "S1_ASC", "S1_ASC_DOY"], "written order, by offset")
        XCTAssertEqual(header.metadata, ["source": "test"])
        XCTAssertEqual(header.byteCount, try XCTUnwrap(FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int))
        XCTAssertNil(StudioSafetensorsHeader.loadHeader(from: root.appendingPathComponent("missing.safetensors")))
        let text = root.appendingPathComponent("notes.txt")
        try "not a tensor file at all, just words".write(to: text, atomically: true, encoding: .utf8)
        XCTAssertNil(StudioSafetensorsHeader.loadHeader(from: text))
        XCTAssertNil(StudioTensorHeader.loadHeader(from: text))
        guard case .safetensors(let viaTensorHeader)? = StudioTensorHeader.loadHeader(from: url) else {
            return XCTFail("The tensor header reader should read a safetensors file's front")
        }
        XCTAssertEqual(viaTensorHeader, header)

        // An `.npy` file's header is a short dictionary; its dtype and shape are read from the
        // front and the size from the file.
        let npy = root.appendingPathComponent("latents.npy")
        var bytes = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        let dictionary = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 64, 1875), }".padding(toLength: 118, withPad: " ", startingAt: 0) + "\n"
        bytes.append(contentsOf: [UInt8(dictionary.utf8.count & 0xFF), UInt8(dictionary.utf8.count >> 8)])
        bytes.append(Data(dictionary.utf8))
        bytes.append(Data(repeating: 0, count: 1 * 64 * 1_875 * 4))
        try bytes.write(to: npy)
        guard case .npy(let metadata)? = StudioTensorHeader.loadHeader(from: npy) else {
            return XCTFail("The tensor header reader should read an .npy file's front")
        }
        XCTAssertEqual(metadata.descriptor, "<f4")
        XCTAssertEqual(metadata.shape, "(1, 64, 1875)")
        XCTAssertEqual(metadata.byteCount, bytes.count)
        XCTAssertEqual(StudioTensorHeader.fileExtensions, ["safetensors", "npy"])
    }

    // MARK: Argv parity

    /// The task draft builds exactly the argv the Geo page's `CommandDraft` built for the same
    /// settings, template by template, including TESSERA's dimensions and OlmoEarth's options.
    func testTaskDraftsBuildTheGeoPagesArgv() throws {
        func page(_ id: CommandTemplateID, _ edit: (inout CommandDraft) -> Void) throws -> (CommandDraft, [String]) {
            let template = try XCTUnwrap(CommandCatalog.template(id: id))
            var draft = template.defaultDraft()
            draft.inputPath = "/tiles/batch.safetensors"
            draft.outputPath = "/out/result.safetensors"
            edit(&draft)
            return (draft, template.arguments(from: draft))
        }
        func task(_ id: CommandTemplateID, _ edit: (inout StudioTaskDraft) -> Void) -> [String] {
            var draft = StudioTaskDraft(templateID: id)
            draft.setArgument(0, "/tiles/batch.safetensors")
            draft.form["--output"] = .text("/out/result.safetensors")
            edit(&draft)
            return draft.arguments
        }

        let (_, flood) = try page(.geoFlood) { $0.model = "vision-flood-terramind-base"; $0.preflight = true }
        XCTAssertEqual(
            task(.geoFlood) { $0.model = "vision-flood-terramind-base"; $0.form["--preflight"] = .flag(true) }, flood
        )
        XCTAssertEqual(flood, [
            "geo", "flood", "/tiles/batch.safetensors", "--output", "/out/result.safetensors",
            "--model", "vision-flood-terramind-base", "--preflight", "--json",
        ])
        let (_, fire) = try page(.geoFire) { _ in }
        XCTAssertEqual(task(.geoFire) { _ in }, fire)
        XCTAssertEqual(fire, ["geo", "fire", "/tiles/batch.safetensors", "--output", "/out/result.safetensors", "--json"])

        let (_, tessera) = try page(.geoTessera) { $0.geoDimensions = "64" }
        XCTAssertEqual(task(.geoTessera) { $0.form["--dimensions"] = .integer(64) }, tessera)
        XCTAssertEqual(tessera, [
            "geo", "tessera", "/tiles/batch.safetensors", "--output", "/out/result.safetensors", "--dimensions", "64", "--json",
        ])
        let (_, checkpointDefault) = try page(.geoTessera) { $0.geoDimensions = "" }
        XCTAssertEqual(task(.geoTessera) { _ in }, checkpointDefault, "a fresh draft leaves the width to the checkpoint")
        XCTAssertFalse(checkpointDefault.contains("--dimensions"))

        let (_, olmo) = try page(.geoOlmoEarth) { $0.geoPatchSize = 2; $0.geoInputResolution = 30; $0.geoIncludeTokens = true }
        let edited = task(.geoOlmoEarth) {
            $0.form["--patch-size"] = .integer(2)
            $0.form["--input-resolution"] = .number(30)
            $0.form["--include-tokens"] = .flag(true)
        }
        XCTAssertEqual(edited, [
            "geo", "olmoearth", "/tiles/batch.safetensors", "--output", "/out/result.safetensors",
            "--patch-size", "2", "--input-resolution", "30", "--include-tokens", "--json",
        ])
        // The page printed its Double with a trailing ".0"; the inspector's number is written
        // without one. `--input-resolution` is a Float to the CLI, so both read as 30 m.
        XCTAssertEqual(olmo, edited.map { $0 == "30" ? "30.0" : $0 })
        let (_, olmoDefaults) = try page(.geoOlmoEarth) { _ in }
        XCTAssertEqual(task(.geoOlmoEarth) { _ in }, olmoDefaults, "the page's 4 px / 10 m defaults are the fresh draft's")
    }

    // MARK: Surface

    /// One well slot for the bundle, the dimensions row drawn by its own picker, the rest plain;
    /// `--json` is the launcher's and `--output` is routing's, so neither is a control.
    func testTheSurfaceComesFromTheContract() throws {
        for id in [CommandTemplateID.geoFlood, .geoFire, .geoTessera, .geoOlmoEarth] {
            XCTAssertEqual(StudioTaskSchema.slots(for: id).map(\.id), ["input"], "\(id)")
            XCTAssertEqual(StudioTaskSchema.slots(for: id).first?.isRequired, true)
            let task = id.studioTask
            let draft = StudioTaskDraft(templateID: id)
            let shown = StudioTaskSchema.sections(for: task, draft: draft).flatMap(\.fields)
            XCTAssertFalse(shown.contains { $0.flag == "--json" || $0.flag == "--output" }, "\(id)")
            XCTAssertTrue(shown.contains { $0.flag == "--preflight" }, "\(id) keeps Preflight reachable")
            XCTAssertTrue(shown.contains { $0.overrideID == .model }, "\(id)")
            XCTAssertTrue(draft.arguments.contains("--json"), "\(id) fresh draft asks for the JSON the panel reads")
        }
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--dimensions", templateID: .geoTessera), .earthDimensions)
        XCTAssertNil(StudioTaskSchema.overrideID(forFlag: "--dimensions", templateID: .imageGenerate),
                     "another command's --dimensions is its own")
        let tessera = StudioTaskSchema.sections(for: .earthTessera, draft: StudioTaskDraft(templateID: .geoTessera)).flatMap(\.fields)
        XCTAssertEqual(tessera.first { $0.flag == "--dimensions" }?.overrideID, .earthDimensions)
        // Patch size and resolution are one editor (the four sizes the command accepts, metres
        // above zero) drawn where the first of its flags is declared; the tokens switch is plain.
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--patch-size", templateID: .geoOlmoEarth), .earthSampling)
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--input-resolution", templateID: .geoOlmoEarth), .earthSampling)
        let olmo = StudioTaskSchema.sections(for: .earthOlmoEarth, draft: StudioTaskDraft(templateID: .geoOlmoEarth)).flatMap(\.fields)
        let sampling = try XCTUnwrap(olmo.first { $0.overrideID == .earthSampling })
        XCTAssertEqual(sampling.bindings.map(\.fieldID), ["--patch-size", "--input-resolution"])
        XCTAssertEqual(olmo.filter { $0.overrideID == .earthSampling }.count, 1, "one row for the pair")
        XCTAssertEqual(olmo.first { $0.flag == "--include-tokens" }?.overrideID, nil)
        XCTAssertEqual(StudioModelScope(templateID: .geoFlood).categories, ["vision-flood"])
        XCTAssertEqual(StudioModelScope(templateID: .geoFire).categories, ["vision-fire"])
        XCTAssertEqual(StudioModelScope(templateID: .geoTessera).categories, ["vision-embed"])
        XCTAssertEqual(StudioModelScope(templateID: .geoOlmoEarth).categories, ["vision-embed"])
    }

    /// The page's dictionary key, a `String` enum without `CodingKeyRepresentable`, which
    /// `JSONEncoder` writes as an array of alternating keys and values — the shape on disk.
    private enum PageTool: String, Codable, Hashable {
        case flood
        case fire
        case tessera
        case olmoEarth
    }

    /// The Geo page kept every workflow's draft in one dictionary per task, keyed by its tool
    /// enum; the task's own entry seeds its task draft once, without the page's stamped output
    /// path, and a parked draft wins.
    @MainActor
    func testTheGeoPagesDraftsImportOnce() throws {
        let sessions = StudioTaskSessions()
        var tessera = try XCTUnwrap(CommandCatalog.template(id: .geoTessera)).defaultDraft()
        tessera.inputPath = "/tiles/year.safetensors"
        tessera.outputPath = "/Users/example/Documents/mere.run/Earth/tessera-20260101.safetensors"
        tessera.model = "vision-embed-tessera-v2-large"
        tessera.geoDimensions = "32"
        var flood = try XCTUnwrap(CommandCatalog.template(id: .geoFlood)).defaultDraft()
        flood.inputPath = "/tiles/flood.safetensors"
        let page: [PageTool: CommandDraft] = [.tessera: tessera, .flood: flood]
        let encoded = try JSONEncoder().encode(page)
        XCTAssertEqual(encoded.first, UInt8(ascii: "["), "the page's dictionary is an array on disk, not an object")
        XCTAssertNil(try? JSONDecoder().decode([String: CommandDraft].self, from: encoded), "a string-keyed read would miss it")
        sessions.set(page, for: StudioTask.earthTessera.rawValue + "." + StudioTaskDraftMigration.earthPageKey)

        let imported = try XCTUnwrap(sessions.taskDraft(for: .earthTessera))
        XCTAssertEqual(imported.templateID, .geoTessera)
        XCTAssertEqual(imported.primaryInputPath, "/tiles/year.safetensors")
        XCTAssertEqual(imported.model, "vision-embed-tessera-v2-large")
        XCTAssertEqual(imported.text("--dimensions"), "32")
        XCTAssertEqual(imported.text("--output"), "", "the page's destination was that run's, not a setting")
        XCTAssertEqual(sessions.taskDraft(for: .earthFlood)?.primaryInputPath, "",
                       "Flood's task reads its own scope, which holds nothing")
        sessions.setTaskDraft(StudioTaskDraft(templateID: .geoTessera), for: .earthTessera)
        XCTAssertEqual(sessions.taskDraft(for: .earthTessera)?.primaryInputPath, "", "a parked draft wins")
    }

    /// Routing files the run under Earth, named after the bundle, as a safetensors file.
    func testTheDestinationLandsUnderEarth() throws {
        let suiteName = "StudioEarthInputRequirementTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.set(root.path, forKey: StudioOutputLocation.rootDefaultsKey)
        StudioOutputLocation.defaults = defaults
        defer {
            StudioOutputLocation.defaults = .standard
            defaults.removePersistentDomain(forName: suiteName)
        }
        var draft = StudioTaskDraft(templateID: .geoOlmoEarth)
        draft.setArgument(0, "/tiles/valley-2024.safetensors")
        let named = StudioOutputLocation.destination(for: draft)
        let output = URL(fileURLWithPath: named.text("--output"))
        XCTAssertEqual(output.deletingLastPathComponent().path, root.appendingPathComponent("Earth").path)
        XCTAssertTrue(output.lastPathComponent.hasPrefix("valley-2024-"), output.path)
        XCTAssertEqual(output.pathExtension, "safetensors")
        XCTAssertEqual(StudioOutputLocation.destination(for: named).text("--output"), output.path, "stable on its own result")
        XCTAssertEqual(named.request()?.mode, .readImage, "Library rows keep the page's attribution")
    }

    static let tesseraTensors: [SafetensorsFixture.Tensor] = [
        .float32("S2", shape: [1, 4, 10], value: 1_200),
        .float32("S2_DOY", shape: [1, 4], value: 120),
        .float32("S1_ASC", shape: [1, 4, 2], value: -12),
        .float32("S1_ASC_DOY", shape: [1, 4], value: 118),
    ]
}
