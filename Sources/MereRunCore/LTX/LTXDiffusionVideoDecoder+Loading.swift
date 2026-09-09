import Foundation
import MLX
import MLXNN

extension LTXDiffusionVideoDecoder {
    public static func load(
        weightsURL: URL,
        dtype: DType = .bfloat16,
        fileManager: FileManager = .default
    ) throws -> LTXDiffusionVideoDecoder {
        guard fileManager.fileExists(atPath: weightsURL.path) else {
            throw LTXDiffusionVideoDecoderError.missingWeights(weightsURL)
        }
        let decoder = LTXDiffusionVideoDecoder()
        let statistics = try SafetensorsStreamingLoader.loadArrays(
            url: weightsURL,
            where: {
                $0 == "per_channel_statistics.mean-of-means"
                    || $0 == "per_channel_statistics.std-of-means"
            },
            dtype: .float32
        )
        if let mean = statistics["per_channel_statistics.mean-of-means"] {
            decoder.latentsMean = mean
        }
        if let standardDeviation = statistics["per_channel_statistics.std-of-means"] {
            decoder.latentsStd = standardDeviation
        }
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: weightsURL,
            to: decoder,
            dtype: dtype,
            verify: [.noUnusedKeys, .shapeMismatch],
            include: { $0.hasPrefix("decoder.") && $0 != "decoder.type_emb" },
            mapper: { key, value in
                mapLTXDiffusionVideoDecoderWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 16
        )
        return decoder
    }
}
