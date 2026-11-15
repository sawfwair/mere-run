import Foundation
import AudioCodecs
import AudioSOVTranscribeModel
import MLX
import MLXNN
import MereRunTensor
import Tokenizers
import XCTest
@testable import AudioSTT
@testable import MereRunGemmaModel

struct SOVLowBitSample: Codable {
    let id: String
    let language: String
    let audio: String
    let reference: String
}

/// Optional real-checkpoint calibration; never runs in the ordinary test gate.
final class SOVLowBitProfileTests: MereRunCoreTestCase {
    struct Capture {
        let input: MLXArray
        let indices: MLXArray
        let weights: MLXArray
        let reference: MLXArray
    }
    struct Score: Codable {
        let layer: Int
        let bits: Int
        let relativeRMSE: Double
        let parameterBytes: Int
    }

    func testCalibrateExpertPrecision() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let root = env["MERERUN_TEST_SOV_PATH"],
              let manifest = env["MERERUN_SOV_CALIBRATION"], let output = env["MERERUN_SOV_LOWBIT_OUTPUT"] else {
            throw XCTSkip("Set GPU device, original SOV root, calibration manifest, and low-bit output directory.")
        }
        let samples = try JSONDecoder().decode([SOVLowBitSample].self, from: Data(contentsOf: URL(fileURLWithPath: manifest)))
        try await MLXRequestStreams.withStream {
            let root = URL(fileURLWithPath: root)
            let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
            let target = Gemma4TextCausalLM(config: config.textConfig)
            target.prepareForFullPrecisionExpertWeights()
            try SOVTranscribeWeights.load(root: root, into: target, target: true)
            let encoder = SOVAudioEncoder(config: config.audioConfig)
            let projector = SOVAudioProjector(config: config)
            try SOVTranscribeWeights.load(root: root, into: encoder, sourcePrefix: "model.audio_tower.")
            try SOVTranscribeWeights.load(root: root, into: projector, sourcePrefix: "model.embed_audio.")
            let tokenizer = try await AutoTokenizer.from(modelFolder: root)
            var captures = [[Capture]](repeating: [], count: target.languageModel.layers.count)
            for sample in samples {
                let audio = try AudioReader.readAudio(from: URL(fileURLWithPath: sample.audio))
                let features = try SOVTranscribeGenerator.audioFeatures(samples: audio)
                let projected = projector(try encoder(features)).reshaped(1, -1, config.textConfig.hiddenSize)
                let prompt = SOVTranscribeGenerator.prompt(instruction: nil, language: nil, audioTokens: projected.dim(1))
                let promptIDs = tokenizer.encode(text: prompt, addSpecialTokens: false)
                let ids = promptIDs + Array(tokenizer.encode(text: sample.reference, addSpecialTokens: false).prefix(64))
                let first = try XCTUnwrap(ids.firstIndex(of: config.audioTokenID))
                let text = target.inputEmbeddings(for: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count))
                var hidden = concatenated([text[0..., ..<first, 0...], projected,
                    text[0..., (first + projected.dim(1))..., 0...]], axis: 1)
                let positions = (0..<32).map { Int32($0 * (promptIDs.count - 1) / 31) }
                    + (promptIDs.count..<min(ids.count, promptIDs.count + 16)).map(Int32.init)
                let selected = MLXArray(positions)
                let cache = target.makeAttentionCache()
                for (index, layer) in target.languageModel.layers.enumerated() {
                    let context = cache[target.languageModel.layerIndexToCacheIndex[index]]
                    let attention = layer.selfAttention(layer.inputLayerNorm(hidden), cache: Gemma4ForwardAttentionCache(context.fork()))
                    let residual = hidden + layer.postAttentionLayerNorm(attention)
                    let route = layer.router!(residual)
                    let input = take(layer.preFeedforwardLayerNorm2!(residual), selected, axis: 1)
                    let indices = take(route.indices, selected, axis: 1)
                    let weights = take(route.weights, selected, axis: 1)
                    let reference = layer.postFeedforwardLayerNorm2!(layer.experts!(input, indices: indices, weights: weights))
                    hidden = layer(hidden, cache: Gemma4ForwardAttentionCache(context), perLayerInput: nil)
                    eval(hidden, input, indices, weights, reference)
                    captures[index].append(Capture(input: input, indices: indices, weights: weights, reference: reference))
                }
                Memory.clearCache()
                print("SOV calibration captured \(sample.id)")
            }
            var scores: [Score] = []
            for (index, layer) in target.languageModel.layers.enumerated() {
                for bits in [4, 6, 8] {
                    let packed = Gemma4SwitchGLU(packing: layer.experts!.switchGLU, groupSize: 64, bits: bits)
                    var numerator = 0.0, denominator = 0.0
                    for capture in captures[index] {
                        let routed = packed(capture.input, indices: capture.indices)
                        let mixed = (routed * capture.weights.expandedDimensions(axis: -1)).asType(capture.input.dtype).sum(axis: -2)
                        let result = layer.postFeedforwardLayerNorm2!(mixed).asType(.float32)
                        let reference = capture.reference.asType(.float32)
                        numerator += Double(square(result - reference).sum().item(Float.self))
                        denominator += Double(square(reference).sum().item(Float.self))
                    }
                    scores.append(Score(layer: index, bits: bits, relativeRMSE: sqrt(numerator / denominator),
                        parameterBytes: packed.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }))
                    Memory.clearCache()
                }
                print("SOV calibration scored layer \(index)")
            }
            let destination = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            let json = JSONEncoder(); json.outputFormatting = [.prettyPrinted, .sortedKeys]
            try json.encode(scores).write(to: destination.appendingPathComponent("calibration-scores.json"))
            for (name, threshold) in [("mixed-001", 0.01), ("mixed-002", 0.02)] {
                let bits = captures.indices.map { index in
                    scores.filter { $0.layer == index && $0.relativeRMSE <= threshold }.map(\.bits).min() ?? 8
                }
                try json.encode(SOVExpertRecipe(layerBits: bits)).write(to: destination.appendingPathComponent(name + ".json"))
            }
        }
    }
}
