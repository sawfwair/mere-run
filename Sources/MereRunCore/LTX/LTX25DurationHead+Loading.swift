import Foundation
import MLX
import MLXNN

extension LTX25DurationHead {
    public static func load(
        weightsURL: URL,
        dtype: DType = .bfloat16,
        fileManager: FileManager = .default
    ) throws -> LTX25DurationHead {
        guard fileManager.fileExists(atPath: weightsURL.path) else {
            throw LTX25DurationHeadError.missingWeights(weightsURL)
        }
        let head = LTX25DurationHead()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: weightsURL,
            to: head,
            dtype: dtype,
            verify: .all,
            include: { $0.hasPrefix("duration_head.") },
            mapper: { key, value in
                mapLTX25DurationHeadWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 15
        )
        MLX.eval(head)
        return head
    }
}
