import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunAudioModels

final class AuKRuntimeTests: MLXTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/AuK")
    }
    private func fixture(_ name: String) throws -> [String: MLXArray] {
        try loadArrays(url: root.appendingPathComponent(name + ".safetensors"))
    }
    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 2e-5,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), tolerance, file: file, line: line)
    }

    func testDiTMatchesPinnedUpstreamForPlainReferenceAndGuidedInputs() throws {
        let fixtures = AuKTensorStore(try fixture("expected"))
        var config = AuKDiTConfiguration()
        config.dimension = 16
        config.heads = 4
        config.doubleLayers = 1
        config.singleLayers = 1
        let dit = try AuKDiT(weights: AuKTensorStore(fixture("dit")),
                             frequencies: fixtures.tensor("inv_freq").asArray(Float.self), configuration: config)
        for (name, reference, guidance): (String, Bool, Float) in [
            ("plain", false, 0), ("reference", true, 0), ("guided", true, 2)
        ] {
            let actual = try dit.velocity(latent: fixtures.tensor("latent"), text: fixtures.tensor("text"),
                time: 0.3, reference: reference ? fixtures.tensor("reference_input") : nil, guidance: guidance)
            assertClose(actual, try fixtures.tensor(name))
        }
    }

    func testAntiAliasedActivationMatchesUpstreamAndPreservesLength() throws {
        let fixtures = AuKTensorStore(try fixture("expected"))
        let vae = AuKVAE(weights: AuKTensorStore([
            "test.act.alpha": try fixtures.tensor("act_alpha"),
            "test.act.beta": try fixtures.tensor("act_beta")
        ]))
        let actual = try vae.activation(fixtures.tensor("act_input"), "test")
        assertClose(actual, try fixtures.tensor("act_output"), tolerance: 2e-6)
        XCTAssertEqual(actual.shape, [1, 17, 3])
        XCTAssertEqual(AuKVAE.sincFilter.sum().item(Float.self), 1, accuracy: 1e-6)
    }

    func testThinkerFusionMatchesUpstreamIncludingWindowedAudio() throws {
        let fixtures = AuKTensorStore(try fixture("expected"))
        let text = AuKThinkerConfiguration.Text(hiddenSize: 16, numHiddenLayers: 2, numAttentionHeads: 4,
            numKeyValueHeads: 2, intermediateSize: 32, vocabSize: 48, ropeTheta: 1_000_000, rmsNormEps: 1e-6)
        let audio = AuKThinkerConfiguration.Audio(dModel: 16, encoderLayers: 2, encoderAttentionHeads: 4,
            encoderFfnDim: 32, outputDim: 16, nWindow: 4, numMelBins: 4)
        let config = AuKThinkerConfiguration(textConfig: text, audioConfig: audio, audioTokenIndex: 40)
        let thinker = AuKThinker(weights: AuKTensorStore(try fixture("thinker")), configuration: config)
        for hasAudio in [false, true] {
            let output = try thinker.encode(tokens: [1, 2, 40, 40, 40, 3],
                audio: hasAudio ? fixtures.tensor("mel") : nil,
                layerWeights: fixtures.tensor("layer_weights"), layerScale: MLXArray(Float(1.7)))
            assertClose(output, try fixtures.tensor(hasAudio ? "thinker_audio" : "thinker_text"))
        }
        XCTAssertThrowsError(try thinker.encode(tokens: [1, 40, 3], audio: fixtures.tensor("mel"),
            layerWeights: fixtures.tensor("layer_weights"), layerScale: MLXArray(Float(1))))
    }

    func testVAEEncodeDecodeMatchesUpstreamWithAllSixStages() throws {
        let fixtures = AuKTensorStore(try fixture("expected"))
        let vae = AuKVAE(weights: AuKTensorStore(try fixture("vae")))
        assertClose(try vae.encode(fixtures.tensor("vae_input")), try fixtures.tensor("vae_encoded"))
        let decoded = try vae.decode(fixtures.tensor("vae_latent"))
        assertClose(decoded, try fixtures.tensor("vae_decoded"))
        XCTAssertEqual(decoded.shape, [1, 960, 1])
    }

    func testEulerIntegratesOnNonuniformGridAndRejectsInvalidSchedules() throws {
        let initial = MLXArray([Float(1), 2])
        let result = try AuKSampling.integrate(initial: initial, schedule: AuKSampling.flashGrid) { x, _ in
            ones(like: x) * 3
        }
        assertClose(result, MLXArray([Float(4), 5]), tolerance: 1e-6)
        XCTAssertThrowsError(try AuKSampling.integrate(initial: initial, schedule: [0, 0.5, 0.4, 1]) { x, _ in x })
        XCTAssertEqual(try AuKSampling.frameCount(seconds: 1), 50)
        XCTAssertEqual(try AuKSampling.frameCount(seconds: 0.021), 2)
        XCTAssertThrowsError(try AuKSampling.frameCount(seconds: .nan))
        XCTAssertEqual(try AuKSampling.schedule(variant: .flash, steps: 32), AuKSampling.flashGrid)
        let base = try AuKSampling.schedule(variant: .base, steps: 4)
        for (actual, expected) in zip(base, AuKSampling.flashGrid) { XCTAssertEqual(actual, expected, accuracy: 1e-7) }
    }

    func testThinkerLoadsOnlyShardsContainingTextOrAudioTensors() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let index = ["weight_map": [
            "thinker.model.embed_tokens.weight": "first.safetensors",
            "thinker.audio_tower.conv1.weight": "second.safetensors",
            "talker.unused.weight": "absent.safetensors"
        ]]
        try JSONEncoder().encode(index).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        let embedding = MLXArray([Float(1), 2]).reshaped(1, 2)
        let convolution = MLXArray(0..<24).asType(.float32).reshaped(2, 3, 4)
        try save(arrays: ["thinker.model.embed_tokens.weight": embedding], url: root.appendingPathComponent("first.safetensors"))
        try save(arrays: ["thinker.audio_tower.conv1.weight": convolution], url: root.appendingPathComponent("second.safetensors"))
        let loaded = try AuKCheckpoint.thinker(root)
        assertClose(try loaded.tensor("embed_tokens.weight"), embedding)
        assertClose(try loaded.tensor("audio_tower.conv1.weight"), convolution.transposed(0, 2, 1))
    }

    func testDiffusionCheckpointConversionPreservesRotaryAndConvolutionLayout() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("weights.safetensors")
        let conv = MLXArray(0..<24).asType(.float32).reshaped(2, 3, 4)
        try save(arrays: ["transformer.audio_embed.conv_pos_embed.conv1d.2.weight": conv,
                          "transformer.rotary_embed.inv_freq": MLXArray([Float(1), 0.125]),
                          "layer_weights": MLXArray([Float(0.2)]),
                          "text_encoder.unused": MLXArray([Float(5)])], url: url)
        let (weights, fusion) = try AuKCheckpoint.diffusion(url)
        assertClose(try weights.tensor("audio_embed.conv_pos_embed.conv1d.1.weight"), conv.transposed(0, 2, 1))
        XCTAssertEqual(try fusion.tensor("inv_freq").asArray(Float.self), [1, 0.125])
        XCTAssertThrowsError(try weights.tensor("text_encoder.unused"))
        XCTAssertThrowsError(try weights.tensor("missing"))
        try save(arrays: ["transformer.proj_out.weight": MLXArray([UInt32(1)])], url: url)
        XCTAssertThrowsError(try AuKCheckpoint.diffusion(url))
    }
}
