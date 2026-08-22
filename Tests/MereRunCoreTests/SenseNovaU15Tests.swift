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

    func testCachedVisionRoPEMatchesLegacyFormulaExactly() {
        let input = MLXArray(Array(0..<48).map { Float($0) / 13 }, [6, 8])
            .asType(.bfloat16)
        let cached = SenseNovaU15VisionEmbeddings.makeRoPETables(
            height: 2,
            width: 3,
            hiddenSize: 8,
            base: 10_000,
            dtype: input.dtype
        )
        MLX.eval(cached.arrays)

        let reused = SenseNovaU15VisionEmbeddings.applyVisionRoPE(input, tables: cached)
        let legacy = legacyVisionRoPE(input, height: 2, width: 3, base: 10_000)
        MLX.eval(reused, legacy)

        XCTAssertEqual(reused.asArray(Float.self), legacy.asArray(Float.self))
    }

    func testCachedNoiseEmbeddingMatchesFreshRepeatedInputExactly() {
        let embedder = SenseNovaU15TimestepEmbedder(hiddenSize: 8)
        let values = MLXArray([Float](repeating: 0.375, count: 4))
        let cached = embedder(values)
        MLX.eval(cached)
        let fresh = embedder(values)
        MLX.eval(fresh)

        XCTAssertEqual(cached.asArray(Float.self), fresh.asArray(Float.self))
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

    private func legacyVisionRoPE(
        _ input: MLXArray,
        height: Int,
        width: Int,
        base: Float
    ) -> MLXArray {
        let half = input.dim(-1) / 2
        let xPositions = (0..<height).flatMap { _ in (0..<width).map(Float.init) }
        let yPositions = (0..<height).flatMap { row in [Float](repeating: Float(row), count: width) }
        return MLX.concatenated([
            legacyInterleavedRoPE(input[0..., 0..<half], positions: xPositions, base: base),
            legacyInterleavedRoPE(input[0..., half...], positions: yPositions, base: base),
        ], axis: -1)
    }

    private func legacyInterleavedRoPE(
        _ input: MLXArray,
        positions: [Float],
        base: Float
    ) -> MLXArray {
        let dimension = input.dim(-1)
        let indices = MLXArray(stride(from: Float(0), to: Float(dimension), by: 2))
        let inverseFrequencies = 1 / MLX.pow(MLXArray(base), indices / Float(dimension))
        let frequencies = MLXArray(positions)[0..., .newAxis] * inverseFrequencies[.newAxis, 0...]
        let cosine = MLX.cos(frequencies).asType(input.dtype)
        let sine = MLX.sin(frequencies).asType(input.dtype)
        let even = input[0..., .stride(by: 2)]
        let odd = input[0..., .stride(from: 1, by: 2)]
        let rotatedEven = even * cosine - odd * sine
        let rotatedOdd = even * sine + odd * cosine
        return MLX.stacked([rotatedEven, rotatedOdd], axis: -1).reshaped(input.shape)
    }
}
