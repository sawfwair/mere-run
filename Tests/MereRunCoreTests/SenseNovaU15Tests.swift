import MLX
@testable import MereRunCore
import XCTest

final class SenseNovaU15Tests: MereRunCoreTestCase {
    func testRecommendedScheduleMatchesEndpointsAndShift() {
        let values = SenseNovaU15Scheduler.timesteps(steps: 2, shift: 3)
        XCTAssertEqual(values[0], 0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[2], 1, accuracy: 1e-6)
    }

    func testResolutionNoiseScaleUsesMergedImageTokens() {
        XCTAssertEqual(
            SenseNovaU15Scheduler.noiseScale(
                imageTokenCount: 4_096,
                baseImageTokenCount: 64,
                baseScale: 1,
                maximum: 16,
                mode: "resolution"
            ),
            8,
            accuracy: 1e-6
        )
    }

    func testPatchifyRoundTripPreservesRawPixels() {
        let source = MLXArray(Array(0..<48).map(Float.init), [1, 4, 4, 3])
        let patches = SenseNovaU15Generator.patchify(source, patchSize: 2)
        let restored = SenseNovaU15Generator.unpatchify(
            patches,
            patchSize: 2,
            height: 4,
            width: 4
        )
        MLX.eval(restored)
        XCTAssertEqual(restored.asArray(Float.self), source.asArray(Float.self))
    }

    func testManagedModelIsPinnedAndUsesUnifiedManifest() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: SenseNovaU15Resources.modelID))
        XCTAssertEqual(spec.hubFallback?.repoId, SenseNovaU15Resources.repository)
        XCTAssertEqual(spec.hubFallback?.revision, SenseNovaU15Resources.revision)
        XCTAssertEqual(spec.validationKind, .senseNovaU15)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)

        let manifest = MereRunModelManifest.template(for: .senseNovaU15)
        XCTAssertEqual(manifest.engine, .senseNovaU15)
        XCTAssertEqual(manifest.family, .senseNova)
        XCTAssertEqual(manifest.defaults?.steps, 50)
        XCTAssertEqual(manifest.defaults?.cfg, 4)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 3)
        XCTAssertNil(manifest.components?.vae)
        XCTAssertNil(manifest.components?.scheduler)
    }

    func testTokenizerLoadsOfficialVocabAndMergesLayout() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("sensenova-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "{}".write(to: root.appendingPathComponent("vocab.json"), atomically: true, encoding: .utf8)
        try "#version: 0.2\n".write(
            to: root.appendingPathComponent("merges.txt"),
            atomically: true,
            encoding: .utf8
        )
        let config = """
        {
          "tokenizer_class": "Qwen2Tokenizer",
          "added_tokens_decoder": {
            "151669": {"content":"<IMG_CONTEXT>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true},
            "151670": {"content":"<img>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true},
            "151671": {"content":"</img>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true}
          }
        }
        """
        try config.write(
            to: root.appendingPathComponent("tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizer = try await SenseNovaU15Tokenizer.load(from: .init(rootURL: root))
        XCTAssertEqual(tokenizer.imageContextTokenID, 151_669)
        XCTAssertEqual(tokenizer.imageStartTokenID, 151_670)
        XCTAssertEqual(tokenizer.imageEndTokenID, 151_671)
    }
}
