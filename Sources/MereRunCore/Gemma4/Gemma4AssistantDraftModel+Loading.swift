import Foundation
import MLX
import MLXNN

extension Gemma4AssistantDraftModel {
    func loadWeights(from resources: Gemma4MTPResources) throws {
        let include: (String) -> Bool = { key in
            !key.hasPrefix("masked_embedding.")
                && !(key == "lm_head.weight" && self.config.tieWordEmbeddings)
        }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard include(key) else { return [] }
            return [(key, value)]
        }
        let quantizedModuleResolver: HFSafetensorsWeightsLoader.QuantizedModuleResolver = { _, _, _, _, biases, fallbackGroupSize, fallbackBits in
            if biases != nil || fallbackBits > 4 {
                return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.affine)
            }
            return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.nvfp4)
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if try Self.indexContainsQuantizedWeights(resources.modelIndexURL) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: self,
                    groupSize: 64,
                    bits: 4,
                    quantizedModuleResolver: quantizedModuleResolver,
                    mapper: mapper
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: self,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: mapper
                )
            }
        } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: self,
                dtype: .bfloat16,
                verify: .none,
                include: include,
                mapper: mapper,
                batchSize: 24
            )
        } else {
            throw Gemma4Error.missingFiles([
                resources.modelIndexURL.lastPathComponent,
                resources.modelWeightsURL.lastPathComponent,
            ])
        }
    }

    private static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }
}
