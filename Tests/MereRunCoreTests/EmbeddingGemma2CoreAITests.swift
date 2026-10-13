import XCTest
import MereRunCoreAIModel
@testable import MereRunCore

final class EmbeddingGemma2CoreAITests: XCTestCase {
    private var configData: Data {
        get throws {
            try Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appending(path: "Fixtures/EmbeddingGemma2CoreAI/config.json"))
        }
    }

    func testPublishedBundleGeometryAndUnsafeAssetPaths() throws {
        try JSONDecoder().decode(BundleConfig.self, from: configData).validate()
        let json = String(decoding: try configData, as: UTF8.self)
        for (old, new) in [("embeddinggemma-2-text-mm.aimodel", "../outside.aimodel"),
                           (#""hidden_size": 512"#, #""hidden_size": 513"#),
                           (#""max_tokens": 2048"#, #""max_tokens": 8192"#),
                           ("audio_3000", "audio_2999")] {
            let invalid = json.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(invalid, json)
            XCTAssertThrowsError(try JSONDecoder().decode(BundleConfig.self, from: Data(invalid.utf8)).validate())
        }
    }

    func testNestedArtifactLayoutRequiresAllGraphFilesWithoutLoadingRuntime() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = root.appending(path: "model")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try configData.write(to: bundle.appending(path: "config.json"))
        XCTAssertEqual(EmbeddingGemma2CoreAICatalog.bundleRoot(root), bundle)
        let missing = EmbeddingGemma2CoreAICatalog.missingFiles(root: root)
        XCTAssertEqual(missing.count, 12)
        for file in missing {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([0]).write(to: file)
        }
        XCTAssertTrue(EmbeddingGemma2CoreAICatalog.missingFiles(root: root).isEmpty)
        XCTAssertTrue(EmbeddingGemma2CoreAICatalog.missingFiles(root: bundle).isEmpty)
        try EmbeddingGemma2CoreAICatalog.validateConfiguration(root: root)
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: EmbeddingGemma2CoreAICatalog.modelID))
        XCTAssertTrue(spec.validationMessages(in: root).isEmpty)
        try MereRunModelManifest.template(for: .embeddingGemma2CoreAI).write(to: root)
        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: spec.id)
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertTrue(report.warnings.isEmpty, report.warnings.joined(separator: "\n"))
        let staged = root.appending(path: "staged")
        _ = try ManagedModelResolver.materializeManagedInstallRoot(for: spec, snapshotURL: root,
                                                                   modelDir: staged, fileManager: .default)
        XCTAssertTrue(spec.validationMessages(in: staged).isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: staged.appending(path: "model/embeddinggemma-2-text-mm.aimodel/main.mlirb").path))
        let graph = bundle.appending(path: "embeddinggemma-2-audio.aimodel/main.mlirb")
        try FileManager.default.removeItem(at: graph)
        XCTAssertEqual(EmbeddingGemma2CoreAICatalog.missingFiles(root: root), [graph])
    }

    func testInterleavedMediaSlotsStripOnlyPerMediaBoundaries() {
        var seq = MultimodalSequence(ids: [2, 10, 11])
        let first = MultimodalSequence(ids: [2, 255999, 258880, 258882, 1], softTokens: [Float16](repeating: 3, count: 512), slots: [2])
        seq.appendMedia(first)
        seq.ids.append(12)
        let second = MultimodalSequence(ids: [2, 256000, 258881, 258881, 258883, 1],
                                        softTokens: [Float16](repeating: 4, count: 1024), slots: [2, 3])
        seq.appendMedia(second)
        seq.ids.append(1)
        XCTAssertEqual(seq.ids, [2, 10, 11, 255999, 258880, 258882, 12, 256000, 258881, 258881, 258883, 1])
        XCTAssertEqual(seq.slots, [4, 8, 9])
        XCTAssertEqual(seq.softTokens.count, 1536)
        XCTAssertEqual(Array(seq.softTokens.prefix(512)), first.softTokens)
        XCTAssertEqual(Array(seq.softTokens.suffix(1024)), second.softTokens)
    }

    func testPreflightRejectsVideoEmptyRecordsAndInvalidBudgets() throws {
        let text = EmbeddingGemma2Input(content: [.text("hello")])
        let validate: ([EmbeddingGemma2Input], Int, Int?) throws -> Void = { inputs, dim, limit in
            try EmbeddingGemma2CoreAIRuntime.validateInputs(inputs, task: .raw, title: nil, dimensions: dim, maxTokens: limit)
        }
        try validate([text], 128, 2)
        try validate([text], 768, 8192) // runtime clamps to the export's limit
        XCTAssertThrowsError(try validate([], 768, nil))
        XCTAssertThrowsError(try validate([EmbeddingGemma2Input(content: [])], 768, nil))
        XCTAssertThrowsError(try validate([text], 1024, nil))
        XCTAssertThrowsError(try validate([text], 768, 1))
        for part: EmbeddingGemma2Content in [.video(URL(fileURLWithPath: "/tmp/video.mp4")), .videoFrames([]),
                                            .image(URL(string: "https://example.com/image.png")!)] {
            XCTAssertThrowsError(try validate([EmbeddingGemma2Input(content: [part])], 768, nil))
        }
    }

    func testUnsupportedBuildFailsBeforeReadingArtifacts() async throws {
        guard !EmbeddingGemma2CoreAIRuntime.isAvailable else { return }
        do {
            _ = try await EmbeddingGemma2CoreAIRuntime.embed(root: URL(fileURLWithPath: "/missing-coreai-model"),
                inputs: [EmbeddingGemma2Input(content: [.text("hello")])])
            XCTFail("Expected platform error")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("macOS 27"))
            XCTAssertFalse(error.localizedDescription.contains("Missing Core AI resources"))
        }
    }

    #if os(macOS) && arch(arm64)
    func testExactTokenizerJSONPreservesBOMAndDistinctUnicodeKeys() throws {
        let json = #"{"\ufeff":"\ufeff","é":1,"e\u0301":2}"#
        let parsed = try ExactJSON.parse(Data(json.utf8))
        XCTAssertEqual(parsed["\u{feff}"]?.stringBytes, [0xef, 0xbb, 0xbf])
        XCTAssertEqual(parsed["é"]?.int, 1)
        XCTAssertEqual(parsed["e\u{301}"]?.int, 2)
        XCTAssertEqual(parsed.pairs?.count, 3)
    }

    func testNativePreprocessorsPreservePatchAndAudioMaskGeometry() throws {
        let image = ImagePreprocessor.patchify(resizedRGB: Array(repeating: 64, count: 48 * 48 * 3), width: 48, height: 48)
        XCTAssertEqual(image.softTokenCount, 1)
        XCTAssertEqual(image.patches.count, 2520 * 768)
        XCTAssertEqual(image.patchMask.filter { $0 == 1 }.count, 9)
        XCTAssertEqual(Array(image.patches.prefix(768)), Array(repeating: Float16(64), count: 768))
        XCTAssertTrue(image.patches.dropFirst(9 * 768).allSatisfy { $0 == 0 })
        let silence = try AudioFrontend.prepare(wav16k: Array(repeating: 0, count: 16000))
        XCTAssertEqual(silence.bucket, 192)
        XCTAssertEqual(silence.softTokenCount, AudioFrontend.softTokenCount(frames: silence.frameCount))
        XCTAssertEqual(silence.frameMask.filter { $0 == 1 }.count, silence.frameCount)
        XCTAssertEqual(silence.tokenMask.filter { $0 == 1 }.count, silence.softTokenCount)
        XCTAssertEqual(silence.features[0], Float16(log(Float(0.001))))
        XCTAssertTrue(silence.features.dropFirst(silence.frameCount * 128).allSatisfy { $0 == 0 })
        let strip = try ImagePreprocessor.targetSize(height: 1, width: 300)
        XCTAssertEqual(strip.height, 48)
        XCTAssertLessThanOrEqual(strip.height * strip.width / 256, 2520)
    }
    #endif
}
