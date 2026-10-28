import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioPhononModel

final class PhononModelTests: MLXTestCase {
    func testSyntheticCheckpointRunsWholeGraphAndResetsState() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixtures = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures")
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("mini-config.json"), to: directory.appendingPathComponent("config.json"))
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("tokenizer.json"), to: directory.appendingPathComponent("tokenizer.json"))
        var weights = syntheticWeights()
        weights["flow_lm.out_eos.bias"] = MLXArray([Float(5)])
        weights["mimi.decoder.model.11.conv.bias"] = MLXArray([Float(0.125)])
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let voiceURL = directory.appendingPathComponent("default-voice.safetensors")
        try MLX.save(arrays: ["speaker_wavs": MLXArray.ones([1, 8, 2])], url: voiceURL)
        let model = try PhononModel(directory: directory)
        let voice = try model.loadVoice(url: voiceURL)
        XCTAssertEqual(voice.shape, [1, 2, 8])
        var first: [Float] = [], second: [Float] = []
        var frameSizes: [Int] = []
        try model.generate(text: "Hello world.", voice: voice, temperature: 0, seed: 42) { first += $0; frameSizes.append($0.count) }
        try model.generate(text: "Hello world.", voice: voice, temperature: 0, seed: 42) { second += $0 }
        XCTAssertEqual(frameSizes, [64, 64, 64, 64])
        XCTAssertEqual(first, second)
        XCTAssertEqual(first, [Float](repeating: 0.125, count: 256))
        XCTAssertEqual(model.sampleRate, 800)
        // Nonzero projections exercise seeded sampling and reset the KV/codec states.
        var nonzero = weights
        for (name, value) in weights where name.hasSuffix(".weight") && !name.contains("norm") && !name.contains("in_ln") {
            nonzero[name] = MLXArray((0..<value.size).map { sin(Float($0 + 1)) * 0.08 }, value.shape)
        }
        nonzero["flow_lm.emb_std"] = MLXArray.ones([2])
        nonzero["flow_lm.out_eos.weight"] = MLXArray.zeros([1, 8])
        try MLX.save(arrays: nonzero, url: directory.appendingPathComponent("model.safetensors"))
        let seeded = try PhononModel(directory: directory)
        let seededVoice = try seeded.loadVoice(url: voiceURL)
        var seededFirst: [Float] = [], seededAgain: [Float] = [], otherSeed: [Float] = []
        try seeded.generate(text: "Hello world.", voice: seededVoice, seed: 42) { seededFirst += $0 }
        try seeded.generate(text: "Hello world.", voice: seededVoice, seed: 42) { seededAgain += $0 }
        try seeded.generate(text: "Hello world.", voice: seededVoice, seed: 43) { otherSeed += $0 }
        XCTAssertEqual(seededFirst, seededAgain)
        XCTAssertNotEqual(seededFirst, otherSeed)
        XCTAssertTrue(seededFirst.allSatisfy(\.isFinite))
        try MLX.save(arrays: ["speaker_wavs": MLXArray.ones([1, 7, 2])], url: voiceURL)
        XCTAssertThrowsError(try model.loadVoice(url: voiceURL))
        try MLX.save(arrays: ["emb": MLXArray.ones([1, 2, 8])], metadata: ["model_ext": "wrong@1"], url: voiceURL)
        XCTAssertThrowsError(try model.loadVoice(url: voiceURL))
        // A missing EOS must fail instead of returning truncated audio as success.
        weights["flow_lm.out_eos.bias"] = MLXArray([Float(-5)])
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let noEOS = try PhononModel(directory: directory)
        do {
            try noEOS.generate(text: "Hello.", voice: voice, temperature: 0, seed: 42) { _ in }
            XCTFail("A frame-budget overrun was accepted as a complete utterance")
        } catch PhononError.frameBudgetExceeded { /* Expected: partial audio is not success. */ }
    }

    private func syntheticWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        func zero(_ name: String, _ shape: [Int]) { weights[name] = MLXArray.zeros(shape) }
        func linear(_ name: String, _ input: Int, _ output: Int, bias: Bool = false) {
            zero(name + ".weight", [output, input]); if bias { zero(name + ".bias", [output]) }
        }
        func norm(_ name: String, _ width: Int) {
            weights[name + ".weight"] = MLXArray.ones([width]); zero(name + ".bias", [width])
        }
        func transformer(_ prefix: String, scaled: Bool) {
            let p = prefix + ".layers.0"
            norm(p + ".norm1", 8); norm(p + ".norm2", 8)
            linear(p + ".self_attn.in_proj", 8, 24); linear(p + ".self_attn.out_proj", 8, 8)
            linear(p + ".linear1", 8, 16); linear(p + ".linear2", 16, 8)
            if scaled { zero(p + ".layer_scale_1.scale", [8]); zero(p + ".layer_scale_2.scale", [8]) }
        }
        func convolution(_ name: String, input: Int, output: Int, kernel: Int, transpose: Bool = false, groups: Int = 1, bias: Bool = true) {
            zero(name + ".weight", transpose ? [input, output / groups, kernel] : [output, input / groups, kernel])
            if bias { zero(name + ".bias", [output]) }
        }
        transformer("flow_lm.transformer", scaled: false)
        transformer("mimi.decoder_transformer.transformer", scaled: true)
        for name in ["bos_emb", "emb_mean", "emb_std"] { zero("flow_lm." + name, [2]) }
        linear("flow_lm.input_linear", 2, 8); norm("flow_lm.out_norm", 8); linear("flow_lm.out_eos", 8, 1, bias: true)
        let conditions = "flow_lm.condition_provider.conditioners."
        zero(conditions + "transcript_in_segment.embed.weight", [4001, 8])
        linear(conditions + "speaker_wavs.output_proj", 8, 8)
        linear(conditions + "num_speakers.output_proj", 2, 8)
        zero(conditions + "num_speakers.embed.weight", [32, 2])
        linear(conditions + "duration_delta.output_proj", 4, 8)
        let flow = "flow_lm.flow_net"
        linear(flow + ".input_proj", 2, 8, bias: true); linear(flow + ".cond_embed", 8, 8, bias: true)
        for index in 0..<2 {
            let p = flow + ".time_embed.\(index)"
            zero(p + ".freqs", [128]); linear(p + ".mlp.0", 256, 8, bias: true)
            linear(p + ".mlp.2", 8, 8, bias: true); zero(p + ".mlp.3.alpha", [8])
        }
        let residual = flow + ".res_blocks.0"
        norm(residual + ".in_ln", 8); linear(residual + ".adaLN_modulation.1", 8, 24, bias: true)
        linear(residual + ".mlp.0", 8, 8, bias: true); linear(residual + ".mlp.2", 8, 8, bias: true)
        linear(flow + ".final_layer.adaLN_modulation.1", 8, 16, bias: true)
        linear(flow + ".final_layer.linear", 8, 2, bias: true)
        zero("mimi.quantizer.output_proj.weight", [8, 2, 1])
        convolution("mimi.upsample.convtr.convtr", input: 8, output: 8, kernel: 16, transpose: true, groups: 8, bias: false)
        convolution("mimi.decoder.model.0.conv", input: 8, output: 16, kernel: 7)
        for (index, channels) in [16, 8, 4].enumerated() {
            let p = "mimi.decoder.model.\(2 + 3 * index)"
            convolution(p + ".convtr", input: channels, output: channels / 2, kernel: 4, transpose: true)
            let r = "mimi.decoder.model.\(3 + 3 * index).block"
            convolution(r + ".1.conv", input: channels / 2, output: channels / 4, kernel: 3)
            convolution(r + ".3.conv", input: channels / 4, output: channels / 2, kernel: 1)
        }
        convolution("mimi.decoder.model.11.conv", input: 2, output: 1, kernel: 3)
        return weights
    }
}
