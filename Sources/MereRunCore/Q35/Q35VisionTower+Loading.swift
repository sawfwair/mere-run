import Foundation
import MereRunQwenModel
import MLX

extension Q35VisionTower {
    public func loadWeights(from resources: Q35Resources) throws {
        let arrays: [String: MLXArray]
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            let data = try Data(contentsOf: resources.modelIndexURL)
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
            let filenames = Set(index.weightMap.compactMap { key, filename in
                Self.mapVisionWeightKey(key) == nil ? nil : filename
            })
            var selected: [String: MLXArray] = [:]
            for filename in filenames.sorted() {
                let shard = try SafetensorsStreamingLoader.loadArrays(
                    url: resources.rootURL.appendingPathComponent(filename),
                    where: { index.weightMap[$0] == filename && Self.mapVisionWeightKey($0) != nil },
                    dtype: .bfloat16
                )
                selected.merge(shard) { _, replacement in replacement }
            }
            arrays = selected
        } else {
            arrays = try SafetensorsStreamingLoader.loadArrays(
                url: resources.modelWeightsURL,
                where: { Self.mapVisionWeightKey($0) != nil },
                dtype: .bfloat16
            )
        }
        let mapped = Dictionary(uniqueKeysWithValues: arrays.flatMap { Self.mapVisionWeight($0.key, $0.value) })
        try installMappedWeights(mapped)
    }
}
