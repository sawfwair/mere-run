import Foundation
import MLX
import MereRunMLXTestSupport
import MereRunSheetSageModel
import XCTest
@testable import MereRunCore

final class YuE2SavedStageTests: XCTestCase {
    override func setUp() { super.setUp(); MLXTestSupport.ensureMetalLibraryAvailable() }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func receipt(root: URL, stage: YuE2SavedStage.Stage = .acoustic) throws -> YuE2SavedStage {
        for name in ["model.safetensors", "qwen.tiktoken", "config.json"] {
            try Data(name.utf8).write(to: root.appendingPathComponent(name))
        }
        return try YuE2SavedStage(stage: stage, plan: YuE2GenerationPlan(style: "piano", lyrics: "", planning: .off),
            abc: nil, scoreTokens: [], codecTokens: stage == .score ? [] : [0, 1, 2],
            scoreTruncated: false, musicTruncated: false, resources: YuE2Resources(rootURL: root))
    }
    func testSavedLatentsRoundTripAndTamperingFailsInspection() throws {
        let root = try temporary()
        let directory = root.appendingPathComponent("stage")
        let stage = try receipt(root: root)
        let latents = MLXArray(0..<192).asType(.float32).reshaped(3, 64)
        try stage.save(to: directory, latents: latents)
        let loaded = try YuE2SavedStage.load(from: directory)
        XCTAssertEqual(loaded.codecTokens, [0, 1, 2])
        XCTAssertEqual(loaded.plan, stage.plan)
        XCTAssertEqual(try MLX.loadArray(url: directory.appendingPathComponent("latent.npy")).asArray(Float.self), latents.asArray(Float.self))
        XCTAssertThrowsError(try stage.save(to: directory, latents: latents))
        XCTAssertNoThrow(try YuE2SavedStage.load(from: directory))
        try Data("corrupt".utf8).write(to: directory.appendingPathComponent("latent.npy"))
        XCTAssertThrowsError(try YuE2SavedStage.load(from: directory))
    }
    func testMissingInvalidAndEscapingLatentsAreRejectedWithoutPartialOutput() throws {
        let root = try temporary()
        let stage = try receipt(root: root)
        let destination = root.appendingPathComponent("missing")
        XCTAssertThrowsError(try stage.save(to: destination))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertThrowsError(try stage.save(to: destination, latents: MLXArray.zeros([2, 64])))
        XCTAssertThrowsError(try YuE2SavedStage.validateLatents(MLXArray(Array(repeating: Float.nan, count: 192)).reshaped(3, 64)))
        try stage.save(to: destination, latents: MLXArray.zeros([3, 64]))
        let outside = root.appendingPathComponent("outside.npy")
        try FileManager.default.moveItem(at: destination.appendingPathComponent("latent.npy"), to: outside)
        try FileManager.default.createSymbolicLink(at: destination.appendingPathComponent("latent.npy"), withDestinationURL: outside)
        XCTAssertThrowsError(try YuE2SavedStage.load(from: destination))
    }

    func testSavedArtifactsAcceptDirectoryURLsCreatedBeforeTheDirectoryExists() throws {
        let root = try temporary()
        let directory = URL(fileURLWithPath: root.appendingPathComponent("fresh-stage").path, isDirectory: false)
        let stage = try receipt(root: root)
        try stage.save(to: directory, latents: MLXArray.zeros([3, 64]))
        XCTAssertNoThrow(try YuE2SavedStage.load(from: directory))
        XCTAssertNoThrow(try YuE2SavedStage.load(from: URL(fileURLWithPath: directory.path, isDirectory: true)))
    }

    func testOfflineExportEnumeratesManagedLicenseDirectoryAndFileLinks() throws {
        let root = try temporary()
        let snapshot = root.appendingPathComponent("snapshot")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        let notice = snapshot.appendingPathComponent("notice.txt")
        try Data("license notice".utf8).write(to: notice)
        try FileManager.default.createSymbolicLink(at: snapshot.appendingPathComponent("linked-notice.txt"), withDestinationURL: notice)
        try Data("configuration".utf8).write(to: snapshot.appendingPathComponent("config.json"))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("licenses"), withDestinationURL: snapshot)
        let resources = YuE2Resources(rootURL: root)
        XCTAssertEqual(try resources.licensePaths(prefix: "").sorted(), ["licenses/linked-notice.txt", "licenses/notice.txt"])
        XCTAssertEqual(try resources.licensePaths(prefix: "vae/"), [])
    }
    func testWindowPlanningPreservesAcceptanceCoverage() throws {
        let windows = try SheetSage2Window.plan(duration: 650)
        XCTAssertEqual(windows.map(\.start), [0, 100, 200, 300, 350])
        XCTAssertEqual(windows.map(\.acceptStart), [0, 200, 300, 400, 500])
        XCTAssertEqual(windows.map(\.acceptEnd), [200, 300, 400, 500, 650])
        XCTAssertThrowsError(try SheetSage2Window.plan(duration: 0))
        XCTAssertThrowsError(try SheetSage2Window.plan(duration: 10, overlap: 100, lookahead: 200))
    }

    func testImportedLatentsNormalizeUpstreamLayoutAndRejectBadShape() throws {
        let root = try temporary()
        let imported = root.appendingPathComponent("import.npy")
        let frames = MLXArray(0..<192).asType(.float32).reshaped(3, 64)
        try MLX.save(array: frames.T.expandedDimensions(axis: 0), url: imported)
        XCTAssertEqual(try YuE2SavedStage.loadLatents(from: imported).asArray(Float.self), frames.asArray(Float.self))
        try MLX.save(array: MLXArray.zeros([3, 63]), url: imported)
        XCTAssertThrowsError(try YuE2SavedStage.loadLatents(from: imported))
    }

    func testOverlapPrefixCarriesContextWithoutInventingRhythmOrTimestamps() throws {
        let vocabulary = SheetSage2Vocabulary.self
        let prior = SheetSage2TimedEvent(time: 99, globalSubbeat: 0,
            tokens: [517, 30537, 30709, 30968, 30988, 31041])
        let timestamp = SheetSage2TimedEvent(time: 100, globalSubbeat: 8, tokens: [517, 31398 + 60, 31654])
        let beat = SheetSage2TimedEvent(time: 100.5, globalSubbeat: 12, tokens: [30711])
        let window = try SheetSage2Window.plan(duration: 650)[1]
        let (prefix, base) = SheetSage2Transcriber.overlapPrefix([prior, timestamp, beat], window: window)
        XCTAssertEqual(base, 8)
        var grammar = vocabulary.Grammar()
        for token in prefix.dropFirst(vocabulary.prefix.count) { try grammar.update(token) }
        let decoded = try vocabulary.events(prefix + [2])
        XCTAssertNil(decoded[0].meter, "A timestamp-only event cannot acquire a meter without an eighth position.")
        XCTAssertEqual(decoded[0].tokens.filter(vocabulary.structure.contains), [30968])
        XCTAssertNil(decoded[1].timestamp, "An inferred event time must not become a decoded timestamp.")

        let (beatPrefix, _) = SheetSage2Transcriber.overlapPrefix([prior, beat], window: window)
        let beatEvents = try vocabulary.events(beatPrefix + [2])
        XCTAssertEqual(beatEvents[0].meter?.numerator, 4)
        XCTAssertEqual(beatEvents[0].meter?.denominator, 4)
        var beatGrammar = vocabulary.Grammar()
        for token in beatPrefix.dropFirst(vocabulary.prefix.count) { try beatGrammar.update(token) }
    }

    func testSheetSageNotationPreservesEventsAndReportsOverlapClipping() throws {
        let events: [SheetSage2TimedEvent] = (0...4).map { beat in
            var tokens = [517 + beat * 50, 30537, 30709 + beat % 4 * 2]
            if beat == 0 { tokens += [30988, 31041, 31398 + 60, 31654 + 5] }
            if beat == 1 { tokens += [31398 + 62, 31654 + 5] }
            return SheetSage2TimedEvent(time: Double(beat) / 2, globalSubbeat: beat * 4, tokens: tokens)
        }
        let result = try SheetSage2Notation.render(events: events, duration: 2, melodyOnly: false)
        let score = try YuE2Score(abc: result.abc)
        XCTAssertEqual(score.bpm, 120)
        XCTAssertEqual(score.vocal.map(\.pitch), [60, 62])
        XCTAssertEqual(score.vocal.map(\.duration), [256, 512])
        XCTAssertEqual(score.chords.map(\.symbol), ["C"])
        XCTAssertFalse(result.diagnostics.isEmpty)
        let melody = try SheetSage2Notation.render(events: events, duration: 2, melodyOnly: true)
        XCTAssertTrue(try YuE2Score(abc: melody.abc).chords.isEmpty)
        XCTAssertThrowsError(try SheetSage2Notation.render(events: [], duration: 2, melodyOnly: true))
    }
}
