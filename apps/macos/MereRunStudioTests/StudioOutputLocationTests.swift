@testable import MereRunApp
import XCTest

final class StudioOutputLocationTests: XCTestCase {
    private let home = URL(fileURLWithPath: "/Users/example", isDirectory: true)

    override func setUp() {
        super.setUp()
        // The per-media defaults are what these assertions describe; a root someone configured in
        // this process's defaults would move every path.
        UserDefaults.standard.removeObject(forKey: StudioOutputLocation.rootDefaultsKey)
    }

    // MARK: - Slugs

    func testSlugLowercasesStripsPunctuationAndJoinsWithHyphens() {
        XCTAssertEqual(
            StudioOutputLocation.slug("A ceramic coffee mug, in soft morning light!"),
            "a-ceramic-coffee-mug-in-soft-morning-light"
        )
        XCTAssertEqual(StudioOutputLocation.slug("  spaced   out  "), "spaced-out")
        XCTAssertEqual(StudioOutputLocation.slug("Seed #8812 (v2)"), "seed-8812-v2")
    }

    func testSlugFoldsDiacriticsAndDropsNonASCII() {
        XCTAssertEqual(StudioOutputLocation.slug("café au lait"), "cafe-au-lait")
        // Nothing usable survives, so the caller's fallback stem takes over.
        XCTAssertEqual(StudioOutputLocation.slug("日本語"), "")
        XCTAssertEqual(StudioOutputLocation.slug("!!! ??? ---"), "")
    }

    func testSlugStopsAtAWordBoundaryWithinTheLimit() {
        let prompt = "a tiny brass astronaut watering a bonsai tree in cinematic macro detail"
        let slug = StudioOutputLocation.slug(prompt)
        XCTAssertLessThanOrEqual(slug.count, StudioOutputLocation.maximumSlugLength)
        XCTAssertEqual(slug, "a-tiny-brass-astronaut-watering-a-bonsai-tree-in-cinematic")
        XCTAssertFalse(slug.hasSuffix("-"))
    }

    func testSlugTruncatesASingleOverlongWord() {
        let slug = StudioOutputLocation.slug(String(repeating: "x", count: 200))
        XCTAssertEqual(slug.count, StudioOutputLocation.maximumSlugLength)
    }

    func testStemFallsBackWhenThePromptHasNothingToSlug() {
        XCTAssertEqual(StudioOutputLocation.stem(prompt: "", fallbackStem: "narration"), "narration")
        XCTAssertEqual(StudioOutputLocation.stem(prompt: "???", fallbackStem: "Transcribe audio"), "transcribe-audio")
        XCTAssertEqual(StudioOutputLocation.stem(prompt: "", fallbackStem: "???"), "output")
    }

    // MARK: - Identifiers

    func testIdentifierPrefersTheSeedAndIsOtherwiseStable() {
        XCTAssertEqual(StudioOutputLocation.identifier(seed: "8812", fingerprint: "anything"), "8812")
        let first = StudioOutputLocation.identifier(seed: "", fingerprint: "imageGenerate|a mug")
        let second = StudioOutputLocation.identifier(seed: "  ", fingerprint: "imageGenerate|a mug")
        XCTAssertEqual(first, second, "the same run must preview the path it will write")
        XCTAssertEqual(first.count, 6)
        XCTAssertNotEqual(first, StudioOutputLocation.identifier(seed: "", fingerprint: "imageGenerate|a plate"))
    }

    // MARK: - Collisions

    func testUniqueFileNameTakesANumericSuffixUntilItIsFree() {
        var taken: Set<String> = ["mug-8812.png", "mug-8812-2.png"]
        let name = StudioOutputLocation.uniqueFileName(
            stem: "mug", identifier: "8812", fileExtension: "png", exists: { taken.contains($0) }
        )
        XCTAssertEqual(name, "mug-8812-3.png")

        taken = []
        XCTAssertEqual(
            StudioOutputLocation.uniqueFileName(
                stem: "mug", identifier: "8812", fileExtension: "png", exists: { taken.contains($0) }
            ),
            "mug-8812.png"
        )
    }

    func testUniqueFileNameHandlesADirectoryDestinationWithNoExtension() {
        let taken: Set<String> = ["run-a1b2c3"]
        XCTAssertEqual(
            StudioOutputLocation.uniqueFileName(
                stem: "run", identifier: "a1b2c3", fileExtension: "", exists: { taken.contains($0) }
            ),
            "run-a1b2c3-2"
        )
    }

    func testUniqueFileNameGivesUpOnTheNumericSuffixRatherThanLooping() {
        let name = StudioOutputLocation.uniqueFileName(
            stem: "mug", identifier: "8812", fileExtension: "png", exists: { _ in true }
        )
        XCTAssertTrue(name.hasPrefix("mug-8812-"))
        XCTAssertTrue(name.hasSuffix(".png"))
    }

    // MARK: - Roots

    func testDefaultRootsFilePicturesMusicAndDocumentsByMedia() {
        XCTAssertEqual(
            StudioOutputLocation.directory(domain: .image, kind: .image, configuredRoot: "", home: home).path,
            "/Users/example/Pictures/mere.run/Image"
        )
        XCTAssertEqual(
            StudioOutputLocation.directory(domain: .video, kind: .video, configuredRoot: "", home: home).path,
            "/Users/example/Pictures/mere.run/Video"
        )
        XCTAssertEqual(
            StudioOutputLocation.directory(domain: .voice, kind: .audio, configuredRoot: "", home: home).path,
            "/Users/example/Music/mere.run/Voice"
        )
        XCTAssertEqual(
            StudioOutputLocation.directory(domain: .music, kind: .audio, configuredRoot: "", home: home).path,
            "/Users/example/Music/mere.run/Music"
        )
        XCTAssertEqual(
            StudioOutputLocation.directory(domain: .vision, kind: .text, configuredRoot: "", home: home).path,
            "/Users/example/Documents/mere.run/Vision"
        )
        XCTAssertEqual(
            StudioOutputLocation.directory(domain: .threeD, kind: .model3D, configuredRoot: "", home: home).path,
            "/Users/example/Documents/mere.run/3D"
        )
    }

    func testAConfiguredRootOverridesEveryMediaFolder() {
        for kind in [StudioOutputFileKind.image, .audio, .text] {
            XCTAssertEqual(
                StudioOutputLocation.directory(
                    domain: .image, kind: kind, configuredRoot: "~/Creative", home: home
                ).path,
                NSString(string: "~/Creative").expandingTildeInPath + "/Image"
            )
        }
    }

    func testOutputURLNamesTheFileAfterThePromptInTheDomainFolder() {
        let url = StudioOutputLocation.outputURL(
            domain: .image,
            prompt: "A ceramic coffee mug in soft morning light",
            seed: "8812",
            fallbackStem: "Generate image",
            fileExtension: "png",
            configuredRoot: "",
            home: home
        )
        XCTAssertEqual(
            url.path,
            "/Users/example/Pictures/mere.run/Image/a-ceramic-coffee-mug-in-soft-morning-light-8812.png"
        )
    }

    // MARK: - Fallback

    func testPreparingDestinationCreatesTheFolderAndLeavesTheDraftAlone() throws {
        let root = try temporaryDirectory()
        var draft = CommandDraft()
        draft.outputPath = root.appendingPathComponent("Pictures/mere.run/Image/mug-8812.png").path

        let prepared = StudioOutputLocation.preparingDestination(of: draft)

        XCTAssertNil(prepared.fallbackReason)
        XCTAssertEqual(prepared.draft, draft)
        var isDirectory: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("Pictures/mere.run/Image").path, isDirectory: &isDirectory
        ))
        XCTAssertTrue(isDirectory.boolValue)
    }

    func testAnUncreatableDestinationFallsBackToAppOutputsWithItsSidecars() throws {
        let root = try temporaryDirectory()
        // A regular file where the destination's parent folder needs to be: creation must fail.
        let blocker = root.appendingPathComponent("blocked")
        try Data("not a folder".utf8).write(to: blocker)
        let intended = blocker.appendingPathComponent("Image")

        var draft = CommandDraft()
        draft.outputPath = intended.appendingPathComponent("mug-8812.png").path
        draft.visionJSONOutputPath = intended.appendingPathComponent("mug-8812.json").path
        draft.visionMaskOutputDirectory = intended.appendingPathComponent("mug-8812-masks").path
        draft.timingsOutputPath = "/Users/example/elsewhere/timings.json"

        let prepared = StudioOutputLocation.preparingDestination(of: draft)

        let fallback = StudioOutputLocation.appOutputsRoot()
        XCTAssertNotNil(prepared.fallbackReason)
        XCTAssertEqual(prepared.draft.outputPath, fallback.appendingPathComponent("mug-8812.png").path)
        XCTAssertEqual(prepared.draft.visionJSONOutputPath, fallback.appendingPathComponent("mug-8812.json").path)
        XCTAssertEqual(
            prepared.draft.visionMaskOutputDirectory,
            fallback.appendingPathComponent("mug-8812-masks").path
        )
        XCTAssertEqual(
            prepared.draft.timingsOutputPath, "/Users/example/elsewhere/timings.json",
            "a path the user chose elsewhere is not dragged into the fallback"
        )
    }

    func testPreparingADraftWithNoOutputIsANoOp() {
        let draft = CommandDraft()
        let prepared = StudioOutputLocation.preparingDestination(of: draft)
        XCTAssertNil(prepared.fallbackReason)
        XCTAssertEqual(prepared.draft, draft)
    }

    func testAbbreviateShortensTheHomeDirectory() {
        XCTAssertEqual(
            StudioOutputLocation.abbreviate(
                URL(fileURLWithPath: "/Users/example/Pictures/mere.run/Image"), home: "/Users/example"
            ),
            "~/Pictures/mere.run/Image"
        )
        XCTAssertEqual(
            StudioOutputLocation.abbreviate(URL(fileURLWithPath: "/Volumes/Work/out"), home: "/Users/example"),
            "/Volumes/Work/out"
        )
    }

    // MARK: - Through the command adapter

    func testStudioRunsAreNamedAfterThePromptUnderTheDomainFolder() throws {
        var studio = StudioDraft()
        studio.reset(for: .createImage)
        studio.prompt = "A ceramic coffee mug in soft morning light"
        studio.seed = "8812"

        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: studio)
        let url = URL(fileURLWithPath: request.draft.outputPath)

        XCTAssertEqual(url.lastPathComponent, "a-ceramic-coffee-mug-in-soft-morning-light-8812.png")
        XCTAssertEqual(url.deletingLastPathComponent().lastPathComponent, "Image")
        XCTAssertEqual(url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent, "mere.run")
        XCTAssertFalse(request.draft.outputPath.contains("App Outputs"))
    }

    func testVisionSidecarsSitBesideTheNamedOutput() throws {
        var studio = StudioDraft()
        studio.reset(for: .findObjects)
        studio.prompt = "every coffee cup"
        studio.inputPath = "/tmp/mug.png"

        let request = try StudioCommandAdapter.makeRequest(mode: .findObjects, draft: studio)
        let output = URL(fileURLWithPath: request.draft.outputPath)
        let stem = output.deletingPathExtension()

        XCTAssertEqual(request.draft.visionJSONOutputPath, stem.appendingPathExtension("json").path)
        XCTAssertTrue(request.draft.visionMaskOutputDirectory.hasPrefix(stem.path))
        XCTAssertEqual(output.deletingLastPathComponent().lastPathComponent, "Vision")
    }

    func testAPreviewAndItsRunAgreeOnThePath() throws {
        var studio = StudioDraft()
        studio.reset(for: .createImage)
        studio.prompt = "a rainy diner window at dusk"

        let preview = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: studio, validating: false)
        let run = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: studio)
        XCTAssertEqual(preview.draft.outputPath, run.draft.outputPath)
    }

    func testAnInputFirstTaskIsNamedAfterItsAttachment() throws {
        var studio = StudioDraft()
        studio.reset(for: .listen)
        studio.inputPath = "/tmp/Morning Standup.wav"

        let request = try StudioCommandAdapter.makeRequest(mode: .listen, draft: studio)
        XCTAssertTrue(
            URL(fileURLWithPath: request.draft.outputPath).lastPathComponent.hasPrefix("morning-standup-"),
            request.draft.outputPath
        )
    }

    func testReplayedSidecarsMoveWithTheNewOutput() {
        var draft = CommandDraft()
        draft.outputPath = "/out/Vision/cup-1.png"
        draft.visionJSONOutputPath = "/out/Vision/cup-1.json"
        draft.visionMaskOutputDirectory = "/out/Vision/cup-1-masks"
        draft.outputPath = "/out/Vision/cup-2.png"

        StudioVisionResultPaths.rederive(in: &draft, previousOutputPath: "/out/Vision/cup-1.png")

        XCTAssertEqual(draft.visionJSONOutputPath, "/out/Vision/cup-2.json")
        XCTAssertEqual(draft.visionMaskOutputDirectory, "/out/Vision/cup-2-masks")
    }

    func testAHandChosenSidecarSurvivesAReplay() {
        var draft = CommandDraft()
        draft.visionJSONOutputPath = "/Users/example/reports/detections.json"
        draft.outputPath = "/out/Vision/cup-2.png"

        StudioVisionResultPaths.rederive(in: &draft, previousOutputPath: "/out/Vision/cup-1.png")

        XCTAssertEqual(draft.visionJSONOutputPath, "/Users/example/reports/detections.json")
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioOutputLocationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
