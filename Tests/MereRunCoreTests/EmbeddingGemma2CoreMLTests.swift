import XCTest
import MereRunCoreMLModel
@testable import MereRunCore

final class EmbeddingGemma2CoreMLTests: XCTestCase {
    private var configData: Data {
        get throws {
            try Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appending(path: "Fixtures/EmbeddingGemma2CoreML/config.json"))
        }
    }

    func testPublishedBundleGeometryAndUnsafeAssetPaths() throws {
        try JSONDecoder().decode(BundleConfig.self, from: configData).validate()
        let json = String(decoding: try configData, as: UTF8.self)
        for (old, new) in [("embeddings.bf16", "../outside.bf16"),
                           (#""hidden_size": 512"#, #""hidden_size": 513"#),
                           (#""waveform_samples": 160160"#, #""waveform_samples": 160000"#),
                           ("embed_512", "embed_2048")] {
            let invalid = json.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(invalid, json)
            XCTAssertThrowsError(try JSONDecoder().decode(BundleConfig.self, from: Data(invalid.utf8)).validate())
        }
    }

    func testArtifactMaterializationRequiresAllGraphFilesWithoutLoadingCoreML() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try configData.write(to: root.appending(path: "config.json"))
        let missing = EmbeddingGemma2CoreMLCatalog.missingFiles(root: root)
        XCTAssertEqual(missing.count, 12)
        for file in missing {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([0]).write(to: file)
        }
        // Sparse files validate the published byte geometry without allocating real tables.
        for (name, bytes) in [("embeddings.bf16", 262144 * 512 * 2), ("position_embeddings.f16", 2 * 1024 * 768 * 2)] {
            let handle = try FileHandle(forWritingTo: root.appending(path: name))
            try handle.truncate(atOffset: UInt64(bytes)); try handle.close()
        }
        XCTAssertTrue(EmbeddingGemma2CoreMLCatalog.missingFiles(root: root).isEmpty)
        try EmbeddingGemma2CoreMLCatalog.validateConfiguration(root: root)
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: EmbeddingGemma2CoreMLCatalog.modelID))
        XCTAssertTrue(spec.validationMessages(in: root).isEmpty)
        try MereRunModelManifest.template(for: .embeddingGemma2CoreML).write(to: root)
        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: spec.id)
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertTrue(report.warnings.isEmpty, report.warnings.joined(separator: "\n"))
        let staged = root.appending(path: "staged")
        _ = try ManagedModelResolver.materializeManagedInstallRoot(for: spec, snapshotURL: root,
                                                                   modelDir: staged, fileManager: .default)
        XCTAssertTrue(spec.validationMessages(in: staged).isEmpty, spec.validationMessages(in: staged).joined(separator: "\n"))
        let graph = root.appending(path: "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
        try FileManager.default.removeItem(at: graph)
        XCTAssertEqual(EmbeddingGemma2CoreMLCatalog.missingFiles(root: root), [graph])
    }

    func testBF16LookupPreservesSignsScalingAndRejectsOutOfRangeIDs() throws {
        let values: [Float] = [1, -1, 0.5, 2, 3, 4]
        var words = values.map { UInt16($0.bitPattern >> 16).littleEndian }
        let data = words.withUnsafeMutableBytes { Data($0) }
        let table = try TokenEmbeddingTable(data: data, rows: 3, columns: 2, scale: 2)
        XCTAssertEqual(try table.embeddings(ids: [2, 0, 1]), [6, 8, 2, -2, 1, 4].map(Float16.init))
        XCTAssertThrowsError(try table.embeddings(ids: [-1]))
        XCTAssertThrowsError(try table.embeddings(ids: [3]))
        XCTAssertThrowsError(try TokenEmbeddingTable(data: data.dropLast(), rows: 3, columns: 2, scale: 2))
    }

    func testInterleavedMediaSlotsStripOnlyPerMediaBoundaries() {
        var seq = MultimodalSequence(ids: [2, 10, 11])
        let first = MultimodalSequence(ids: [2, 255999, 0, 258882, 1], softTokens: [Float16](repeating: 3, count: 512), slots: [2])
        seq.appendMedia(first)
        seq.ids.append(12)
        let second = MultimodalSequence(ids: [2, 256000, 0, 0, 258883, 1],
                                        softTokens: [Float16](repeating: 4, count: 1024), slots: [2, 3])
        seq.appendMedia(second)
        seq.ids.append(1)
        XCTAssertEqual(seq.ids, [2, 10, 11, 255999, 0, 258882, 12, 256000, 0, 0, 258883, 1])
        XCTAssertEqual(seq.slots, [4, 8, 9])
        XCTAssertEqual(seq.softTokens.count, 1536)
        XCTAssertEqual(Array(seq.softTokens.prefix(512)), first.softTokens)
        XCTAssertEqual(Array(seq.softTokens.suffix(1024)), second.softTokens)
    }

    func testPreflightRejectsVideoEmptyRecordsAndInvalidBudgets() throws {
        let text = EmbeddingGemma2Input(content: [.text("hello")])
        let validate: ([EmbeddingGemma2Input], Int, Int?) throws -> Void = { inputs, dim, limit in
            try EmbeddingGemma2CoreMLRuntime.validateInputs(inputs, task: .raw, title: nil, dimensions: dim, maxTokens: limit)
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

    func testAvailableWithTheInstalledMacOSSDK() {
        #if canImport(CoreML) && os(macOS) && arch(arm64)
        XCTAssertTrue(EmbeddingGemma2CoreMLRuntime.isAvailable)
        XCTAssertNoThrow(try EmbeddingGemma2CoreMLRuntime.requireAvailable())
        #else
        XCTAssertFalse(EmbeddingGemma2CoreMLRuntime.isAvailable)
        XCTAssertThrowsError(try EmbeddingGemma2CoreMLRuntime.requireAvailable())
        #endif
    }

    func testAudioMasksFollowTenSecondGraphGeometry() {
        XCTAssertEqual(CoreMLAudioGeometry.validFrames(samples: 160), 0)
        XCTAssertEqual(CoreMLAudioGeometry.validFrames(samples: 161), 1)
        XCTAssertEqual(CoreMLAudioGeometry.tokenCount(samples: 161), 1)
        XCTAssertEqual(CoreMLAudioGeometry.validFrames(samples: 160000), 999)
        XCTAssertEqual(CoreMLAudioGeometry.tokenCount(samples: 160000), 250)
    }

    private func expandedConfigData() throws -> Data {
        let base = String(decoding: try configData, as: UTF8.self)
        let extra = #""extended":{"version":1,"text_package":"EmbeddingGemma2TextExtended.mlpackage","lengths":[1024,2048,4096,8192],"audio_package":"EmbeddingGemma2AudioExtended.mlpackage","audio_frames":3000,"audio_seconds":30,"source_revision":"914f7f89142e33e77833254d9c9b90c3cef7303b","source_sha256":"197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"},"#
        return Data(("{" + extra + base.dropFirst()).utf8)
    }

    func testExpandedBundleUsesCorrectBucketsAndRetainsShortInputGeometry() throws {
        let base = try JSONDecoder().decode(BundleConfig.self, from: configData)
        XCTAssertEqual(base.maximumTokens, 512)
        XCTAssertEqual(base.maximumAudioSeconds, 10)
        XCTAssertThrowsError(try base.textLength(for: 513))
        let expanded = try JSONDecoder().decode(BundleConfig.self, from: expandedConfigData())
        try expanded.validate()
        XCTAssertEqual(expanded.maximumTokens, 8192)
        XCTAssertEqual(expanded.maximumAudioSeconds, 30)
        for (tokens, expected) in [(1, 32), (512, 512), (513, 1024), (1024, 1024), (1025, 2048),
                                   (2049, 4096), (4097, 8192), (8192, 8192)] {
            XCTAssertEqual(try expanded.textLength(for: tokens), expected)
        }
        XCTAssertThrowsError(try expanded.textLength(for: 0))
        XCTAssertThrowsError(try expanded.textLength(for: 8193))
        let json = String(decoding: try expandedConfigData(), as: UTF8.self)
        for (old, new) in [("TextExtended.mlpackage", "../outside.mlpackage"), ("\"audio_frames\":3000", "\"audio_frames\":4000"),
                           ("[1024,2048,4096,8192]", "[1024,8192]"), ("\"version\":1", "\"version\":2")] {
            let invalid = json.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(invalid, json)
            XCTAssertThrowsError(try JSONDecoder().decode(BundleConfig.self, from: Data(invalid.utf8)).validate())
        }
    }

    func testExpandedArtifactValidationRequiresBothNewGraphsAndConversionReceipt() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try expandedConfigData().write(to: root.appending(path: "config.json"))
        XCTAssertThrowsError(try EmbeddingGemma2CoreMLCatalog.validateConfiguration(root: root))
        for package in ["EmbeddingGemma2TextExtended.mlpackage", "EmbeddingGemma2AudioExtended.mlpackage"] {
            for file in ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"] {
                let url = root.appending(path: package + "/" + file)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data([0]).write(to: url)
            }
        }
        XCTAssertThrowsError(try EmbeddingGemma2CoreMLCatalog.validateConfiguration(root: root))
        try Data("{}".utf8).write(to: root.appending(path: "conversion.json"))
        for (name, bytes) in [("embeddings.bf16", 262144 * 512 * 2), ("position_embeddings.f16", 2 * 1024 * 768 * 2)] {
            let url = root.appending(path: name)
            try Data().write(to: url)
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: UInt64(bytes)); try handle.close()
        }
        try EmbeddingGemma2CoreMLCatalog.validateConfiguration(root: root)
    }

    #if os(macOS) && arch(arm64)
    func testExpandedAudioFrontendUsesThirtySecondBoundaries() throws {
        XCTAssertEqual(AudioFrontend.validFrameCount(samples: 160), 0)
        XCTAssertEqual(AudioFrontend.validFrameCount(samples: 161), 1)
        XCTAssertEqual(AudioFrontend.validFrameCount(samples: 480000), 2999)
        XCTAssertEqual(AudioFrontend.softTokenCount(frames: 2999), 750)
        let silence = AudioFrontend.logMel([Float](repeating: 0, count: 161))
        XCTAssertEqual(silence.frames, 1)
        XCTAssertEqual(silence.features.count, 128)
        XCTAssertTrue(silence.features.allSatisfy { abs($0 - Float(log(0.001))) < 0.000001 })
    }
    func testExactTokenizerJSONPreservesBOMAndDistinctUnicodeKeys() throws {
        let json = #"{"\ufeff":"\ufeff","é":1,"e\u0301":2}"#
        let parsed = try ExactJSON.parse(Data(json.utf8))
        XCTAssertEqual(parsed["\u{feff}"]?.stringBytes, [0xef, 0xbb, 0xbf])
        XCTAssertEqual(parsed["é"]?.int, 1)
        XCTAssertEqual(parsed["e\u{301}"]?.int, 2)
        XCTAssertEqual(parsed.pairs?.count, 3)
    }

    func testVisionUsesRowMajorNormalizedPatchesAndExplicitPooling() throws {
        var rgb = [UInt8](repeating: 0, count: 96 * 48 * 3)
        for y in 0..<48 { for x in 0..<96 { rgb[(y * 96 + x) * 3] = UInt8(x / 16 + (y / 16) * 6) } }
        var words = [UInt16](repeating: 0, count: 2 * 1024 * 768)
        for x in 0..<6 { words[x * 768] = Float16(x).bitPattern.littleEndian }
        for y in 0..<3 { words[(1024 + y) * 768] = Float16(y * 10).bitPattern.littleEndian }
        let table = words.withUnsafeMutableBytes { Data($0) }
        let image = try ImagePreprocessor.patchify(rgb: rgb, width: 96, height: 48, positionTable: table, tokens: 70)
        XCTAssertEqual(image.tokenCount, 2)
        XCTAssertEqual(image.valid.filter { $0 == 1 }.count, 18)
        XCTAssertEqual(image.positions.prefix(4), [0, 0, 1, 0])
        XCTAssertEqual(image.patches[768], Float16(1.0 / 255))
        XCTAssertEqual(image.positionEmbeddings[7 * 768], 11)
        let firstPool = Array(image.pool.prefix(630)).indices.filter { image.pool[$0] != 0 }
        XCTAssertEqual(firstPool, [0, 1, 2, 6, 7, 8, 12, 13, 14])
        XCTAssertTrue(image.patches.dropFirst(18 * 768).allSatisfy { $0 == 0 })
        XCTAssertTrue(image.positions.dropFirst(36).allSatisfy { $0 == -1 })
        let strip = try ImagePreprocessor.targetSize(height: 1, width: 300)
        XCTAssertEqual(strip.height, 48)
        XCTAssertLessThanOrEqual(strip.height * strip.width / 256, 2520)
    }
    #endif
}
