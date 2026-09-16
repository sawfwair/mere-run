import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

final class YuE2Tests: XCTestCase {
    private var fixture: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/YuE2")
    }

    override func setUp() {
        super.setUp()
        MLXTestSupport.ensureMetalLibraryAvailable()
    }

    private func load<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: fixture.appendingPathComponent(name)))
    }

    private func arrays(_ name: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fixture.appendingPathComponent(name))
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        guard actual.shape == expected.shape else { return }
        let error = max(abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
        XCTAssertLessThanOrEqual(error, tolerance, "Maximum absolute error \(error)", file: file, line: line)
    }

    func testTransformerAndAcousticMatchPinnedUpstream() throws {
        let config = try load(YuE2Configuration.self, "model-config.json")
        let expected = try arrays("expected.safetensors")
        let tokens = try XCTUnwrap(expected["tokens"]).asArray(Int32.self).map(Int.init)
        let noise = try XCTUnwrap(expected["noise"])
        for label in ["fp32", "bf16"] {
            let model = try YuE2Model(configuration: config, arrays: arrays("model-\(label).safetensors"))
            let cache = model.makeCache()
            let tolerance: Float = label == "fp32" ? 2e-6 : 0.003
            assertClose(try model.logits(tokens, cache: cache), try XCTUnwrap(expected["\(label).prefill"]), tolerance: tolerance)
            assertClose(try model.logits([11], cache: cache), try XCTUnwrap(expected["\(label).cached"]), tolerance: tolerance)
            assertClose(try model.logits(tokens + [11], cache: model.makeCache()),
                        try XCTUnwrap(expected["\(label).uncached"]), tolerance: tolerance)
            let chunkedCache = model.makeCache()
            MLX.eval(try model.logits(Array(tokens.prefix(3)), cache: chunkedCache))
            assertClose(try model.logits(Array(tokens.dropFirst(3)) + [11], cache: chunkedCache),
                        try XCTUnwrap(expected["\(label).uncached"]), tolerance: tolerance)
            let acoustic = try YuE2Acoustic(model: model, tokens: tokens)
            let velocity = try acoustic.velocity(noise.asType(model.embedding.dtype), rawTime: 0.4)
            assertClose(velocity, try XCTUnwrap(expected["\(label).velocity"]), tolerance: label == "fp32" ? 2e-6 : 0.006)
            var completed: [Int] = []
            let solved = try acoustic.solve(noise: noise, steps: 3) { completed.append($0) }
            XCTAssertEqual(completed, [1, 2, 3])
            assertClose(solved, try XCTUnwrap(expected["\(label).midpoint"]), tolerance: label == "fp32" ? 3e-6 : 0.025)
        }
    }

    func testDecoderAndExactTileBoundariesMatchUpstream() throws {
        let config = try load(YuE2VAEConfiguration.self, "decoder-config.json")
        let decoder = try YuE2Decoder(configuration: config, arrays: arrays("decoder.safetensors"))
        let expected = try arrays("expected.safetensors")
        let latents = try XCTUnwrap(expected["decoder.latents"])
        XCTAssertEqual(decoder.requiredHalo(coreFrames: 8), 12)
        XCTAssertEqual(decoder.outputLength(frames: 35), 67_136)
        assertClose(try decoder.decode(latents)[0], try XCTUnwrap(expected["decoder.full"]), tolerance: 2e-5)
        var completed: [Int] = []
        let tiled = try decoder.decodeTiled(latents) { completed.append($0); XCTAssertEqual($1, 35) }
        XCTAssertEqual(completed, [8, 16, 24, 32, 35])
        XCTAssertEqual(tiled.channels, 2)
        XCTAssertEqual(tiled.sampleRate, 48000)
        let interleaved = MLXArray(tiled.samples).reshaped(67_136, 2)
        assertClose(interleaved, try XCTUnwrap(expected["decoder.tiled"]), tolerance: 2e-5)
    }

    func testWeightCoverageRejectsMissingUnexpectedAndWrongShapes() throws {
        let config = try load(YuE2Configuration.self, "model-config.json")
        var weights = try arrays("model-fp32.safetensors")
        let original = weights.removeValue(forKey: "model.layers.0.nar_self_attn.k_norm.weight")
        XCTAssertThrowsError(try YuE2Model(configuration: config, arrays: weights))
        weights["model.layers.0.nar_self_attn.k_norm.weight"] = original
        weights["extra.weight"] = MLXArray.zeros([1])
        XCTAssertThrowsError(try YuE2Model(configuration: config, arrays: weights))
        weights.removeValue(forKey: "extra.weight")
        weights["llm2vae.weight"] = MLXArray.zeros([1, 1])
        XCTAssertThrowsError(try YuE2Model(configuration: config, arrays: weights))
    }

    func testInstalledDecoderTileBoundariesWhenProvided() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_YUE2_ROOT"] else {
            throw XCTSkip("Set MERERUN_YUE2_ROOT to compare full and tiled decoding with the released VAE.")
        }
        let resources = YuE2Resources(rootURL: URL(fileURLWithPath: path))
        var config = try resources.vaeConfiguration()
        // Exercise multiple exact tile cores with the released weights and a short latent sequence.
        config.decodeCoreFrames = 16
        let decoder = try YuE2Decoder(configuration: config, arrays: SafetensorsStreamingLoader.loadArrays(
            url: resources.vaeURL.appendingPathComponent("model.safetensors"),
            where: { $0.hasPrefix("decoder.") }, dtype: .float32
        ))
        let latents = (sin(MLXArray(0..<(40 * 64)).asType(.float32) * 0.01) * 0.2).reshaped(40, 64)
        let full = try decoder.decode(latents)[0]
        MLX.eval(full)
        let tiled = try decoder.decodeTiled(latents) { _, _ in }
        assertClose(MLXArray(tiled.samples).reshaped(-1, 2), full, tolerance: 2e-5)
    }

    func testTokenizerNormalizesUnicodeAndTreatsSpecialStringsAsText() throws {
        let tokenizer = try syntheticTokenizer()
        XCTAssertEqual(try tokenizer.encode("hello e\u{301} <abc>\n"), [259, 32, 260, 32, 60, 97, 98, 99, 62, 10])
        let multilingual = "海の歌 🎵\n1234\tWe're here."
        XCTAssertEqual(try tokenizer.decode(tokenizer.encode(multilingual)), multilingual)
        XCTAssertFalse(try tokenizer.encode("<abc>").contains(151847))
        XCTAssertThrowsError(try tokenizer.decode([151847]))
        XCTAssertThrowsError(try YuE2Tokenizer(contents: "YQ== 0\nYQ== 1", vocabularySize: 2))
    }

    func testCheckpointTokenizerWhenProvided() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_YUE2_TOKENIZER"] else {
            throw XCTSkip("Set MERERUN_YUE2_TOKENIZER for the pinned checkpoint tokenizer acceptance case.")
        }
        let tokenizer = try YuE2Tokenizer(url: URL(fileURLWithPath: path))
        // Frozen using tiktoken 0.12.0 against the pinned qwen.tiktoken.
        let text = "hello e\u{301} <abc>\n海の歌 🎵\n1234\tWe're here."
        let expectedURL = fixture.appendingPathComponent("tokenizer-reference.json")
        let expected = try JSONDecoder().decode([Int].self, from: Data(contentsOf: expectedURL))
        XCTAssertEqual(try tokenizer.encode(text), expected)
        XCTAssertEqual(try tokenizer.decode(expected), text.precomposedStringWithCanonicalMapping)
    }

    func testPromptBranchesRetainExactScoreAndHaveDifferentNegativeOffPrefix() throws {
        let tokenizer = try syntheticTokenizer()
        let plan = try YuE2GenerationPlan(style: "piano", lyrics: "hello")
        let prompt = try YuE2Protocol.prefix(plan: plan, tokenizer: tokenizer, score: nil)
        XCTAssertEqual(prompt.first, 151643)
        XCTAssertEqual(prompt.last, 151847)
        let prefix = try YuE2Protocol.prefix(plan: plan, tokenizer: tokenizer, score: [1, 2, 3])
        let negative = try YuE2Protocol.negative(plan: plan, tokenizer: tokenizer, score: [1, 2, 3])
        XCTAssertEqual(Array(prefix.suffix(6)), [151847, 1, 2, 3, 151848, 151851])
        XCTAssertEqual(Array(negative.suffix(6)), Array(prefix.suffix(6)))
        let off = try YuE2GenerationPlan(style: "piano", lyrics: "", planning: .off)
        XCTAssertEqual(off.guidanceScale, 1.01)
        XCTAssertEqual(Array(try YuE2Protocol.prefix(plan: off, tokenizer: tokenizer, score: nil).suffix(3)),
                       [151847, 151848, 151851])
        XCTAssertFalse(try YuE2Protocol.negative(plan: off, tokenizer: tokenizer, score: []).contains(151847))
        XCTAssertThrowsError(try YuE2Protocol.prefix(plan: plan, tokenizer: tokenizer, score: [151643]))
    }

    func testSamplingMasksVocabulariesCountsFrequencyAndHonorsMinimum() throws {
        var sampling = YuE2Sampling()
        sampling.temperature = 0
        sampling.repetitionPenalty = 2
        sampling.minimumTokens = 2
        let logits = MLXArray.zeros([184704])
        logits[42] = MLXArray(Float(1000)) // Forbidden in semantic generation.
        logits[151853] = MLXArray(Float(8))
        logits[151854] = MLXArray(Float(-2))
        logits[151852] = MLXArray(Float(100))
        let early = YuE2Sampler.scores(logits: logits, sampling: sampling, history: [151853, 151853, 151854],
                                      step: 0, phase: .semantic, legacyOff: false).asArray(Float.self)
        XCTAssertEqual(early[0], 2)
        XCTAssertEqual(early[1], -4)
        XCTAssertEqual(early.last, -Float.infinity)
        let late = YuE2Sampler.scores(logits: logits, sampling: sampling, history: [], step: 2, phase: .semantic, legacyOff: false)
        XCTAssertEqual(argMax(late).item(Int.self), 32768)
        let score = YuE2Sampler.scores(logits: logits, sampling: .score, history: [], step: 0, phase: .score, legacyOff: false)
        XCTAssertEqual(argMax(score).item(Int.self), 42)
    }

    func testRequestsAndAcousticChunksRejectInvalidBoundaries() throws {
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "", lyrics: ""))
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "piano", lyrics: "", planning: .off, abc: "X:1"))
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "piano", lyrics: "", guidanceScale: .nan))
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "piano", lyrics: "", seed: UInt64.max))
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "piano", lyrics: "", steps: 0))
        XCTAssertEqual(try YuE2Protocol.chunkRanges(frames: 10, prefixTokens: 5, context: 16), [0..<4, 4..<8, 8..<10])
        XCTAssertThrowsError(try YuE2Protocol.chunkRanges(frames: 1, prefixTokens: 24574))
        XCTAssertThrowsError(try YuE2Protocol.chunkRanges(frames: 0, prefixTokens: 5))
        let configuration = try load(YuE2Configuration.self, "model-config.json")
        XCTAssertThrowsError(try configuration.validateReleased())
        try YuE2Configuration().validateReleased()
        try YuE2VAEConfiguration().validateReleased()
    }

    func testSamplingDistributionMatchesUpstreamIncludingBF16OffMode() throws {
        let expected = try arrays("expected.safetensors")
        var sampling = YuE2Sampling()
        sampling.topP = 0.6
        sampling.topK = 4
        sampling.minimumTokens = 0
        sampling.maximumTokens = 32
        for label in ["fp32", "bf16"] {
            let dtype: DType = label == "fp32" ? .float32 : .bfloat16
            let logits = MLXArray.zeros([184704], dtype: dtype)
            logits[42] = MLXArray(Float(1000)).asType(dtype)
            logits[151853..<151858] = MLXArray([Float(2.2345), 2.2234, 1.1, -2.3456, 0]).asType(dtype)
            logits[151852] = MLXArray(Float(1.5)).asType(dtype)
            let actual = YuE2Sampler.scores(logits: logits, sampling: sampling, history: [151853, 151853, 151854],
                                           step: 5, phase: .semantic, legacyOff: label == "bf16")
            let reference = try XCTUnwrap(expected["\(label).sampling"])
            XCTAssertTrue(all(isFinite(actual) .== isFinite(reference)).item(Bool.self))
            assertClose(which(isFinite(actual), actual, MLXArray(Float(0))),
                        which(isFinite(reference), reference, MLXArray(Float(0))), tolerance: 1e-6)
        }
    }

    func testCatalogPinsBothModelsAndRequiresLicenseAcceptance() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: YuE2Resources.modelID))
        XCTAssertEqual(spec.hubFallback?.revision, YuE2Resources.revision)
        XCTAssertEqual(spec.mountedHubFallbacks.first?.hubFallback.revision, YuE2Resources.vaeRevision)
        XCTAssertEqual(spec.mountedHubFallbacks.first?.destinationPath, "vae")
        XCTAssertEqual(spec.usageRestriction?.terms.count, 2)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultCLICommands, ["music generate"])
        XCTAssertEqual(spec.validationKind, .yue2)
    }

    private func syntheticTokenizer() throws -> YuE2Tokenizer {
        let tokens = (0...255).map { Data([UInt8($0)]) } + ["he", "ll", "hell", "hello", "é"].map { Data($0.utf8) }
        let contents = tokens.enumerated().map { "\($0.element.base64EncodedString()) \($0.offset)" }.joined(separator: "\n")
        return try YuE2Tokenizer(contents: contents, vocabularySize: tokens.count)
    }
}
