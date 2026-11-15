import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunGemmaModel

extension SOVGenerationProfileTests {
    private struct Routes: Encodable {
        let length: Int
        let layer: Int
        let expertIDs: [Int]
        let uniqueExperts: Int
    }

    func profileRemaining(target: Gemma4TextCausalLM, cache: [Gemma4AttentionCache], candidates: MLXArray,
                          embeddings: MLXArray, output: String) throws {
        var routes: [Routes] = []
        var arrays: [String: MLXArray] = [:]
        for length in [4, 8, 12, 16] {
            let trial = cache.map { $0.fork() }
            var hidden = target.inputEmbeddings(for: candidates[0..., ..<length])
            for (index, layer) in target.languageModel.layers.enumerated() {
                let context = trial[target.languageModel.layerIndexToCacheIndex[index]]
                let attention = layer.selfAttention(layer.inputLayerNorm(hidden), cache: Gemma4ForwardAttentionCache(context.fork()))
                let residual = hidden + layer.postAttentionLayerNorm(attention)
                let route = layer.router!(residual)
                let ids = route.indices.asArray(Int.self)
                routes.append(Routes(length: length, layer: index, expertIDs: ids, uniqueExperts: Set(ids).count))
                if length == 16 && index == 0 {
                    arrays["verification_input"] = layer.preFeedforwardLayerNorm2!(residual)
                    arrays["verification_indices"] = route.indices
                    arrays["verification_weights"] = route.weights
                    arrays["verification_output"] = layer.experts!(arrays["verification_input"]!, indices: route.indices, weights: route.weights)
                }
                hidden = layer(hidden, cache: Gemma4ForwardAttentionCache(context), perLayerInput: nil)
            }
            eval(hidden)
        }
        let layer = target.languageModel.layers[0]
        let attention = layer.selfAttention(layer.inputLayerNorm(embeddings), cache: Gemma4ForwardAttentionCache(target.makeAttentionCache()[0]))
        let residual = embeddings + layer.postAttentionLayerNorm(attention)
        let route = layer.router!(residual)
        arrays["prefill_input"] = layer.preFeedforwardLayerNorm2!(residual)
        arrays["prefill_indices"] = route.indices
        arrays["prefill_weights"] = route.weights
        arrays["prefill_output"] = layer.experts!(arrays["prefill_input"]!, indices: route.indices, weights: route.weights)
        eval(arrays)
        try MLX.save(arrays: arrays, url: URL(fileURLWithPath: output + "-routes.safetensors"))
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: layer.experts!.parameters().flattened()),
            url: URL(fileURLWithPath: output + "-experts.safetensors"))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(routes).write(to: URL(fileURLWithPath: output + "-routes.json"))

        let input = target.inputEmbeddings(for: candidates)
        let baseline = target.forwardEmbeddings(input, inputIDs: candidates, cache: cache.map { $0.fork() }).logits
        eval(baseline)
        target.setExperimentalCompiledPost(true)
        let compiled = target.forwardEmbeddings(input, inputIDs: candidates, cache: cache.map { $0.fork() }).logits
        eval(compiled)
        let error = abs(compiled.asType(.float32) - baseline.asType(.float32)).max().item(Float.self)
        let agreement = (argMax(compiled, axis: -1) .== argMax(baseline, axis: -1)).all().item(Bool.self)
        target.setExperimentalCompiledPost(false)
        try encoder.encode(["compiledLogitMaxError": Double(error), "compiledPredictionsAgree": agreement ? 1.0 : 0.0])
            .write(to: URL(fileURLWithPath: output + "-compiled.json"))
    }
}
