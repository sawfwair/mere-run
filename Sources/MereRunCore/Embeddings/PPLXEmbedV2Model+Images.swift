import Foundation
import MediaIO
import MereRunQwenModel
import MLX

extension PPLXEmbedV2Model {
    private struct ProcessorConfig: Decodable {
        let imageProcessor: ImageProcessor
        enum CodingKeys: String, CodingKey { case imageProcessor = "image_processor" }
        struct ImageProcessor: Decodable {
            let patchSize: Int, temporalPatchSize: Int, mergeSize: Int
            let imageMean: [Float], imageStd: [Float], size: Size
            let rescaleFactor: Float, resample: Int
            enum CodingKeys: String, CodingKey {
                case patchSize = "patch_size", temporalPatchSize = "temporal_patch_size", mergeSize = "merge_size"
                case imageMean = "image_mean", imageStd = "image_std", size, resample, rescaleFactor = "rescale_factor"
            }
            struct Size: Decodable {
                let shortestEdge: Int, longestEdge: Int
                enum CodingKeys: String, CodingKey { case shortestEdge = "shortest_edge", longestEdge = "longest_edge" }
            }
        }
    }

    public func embed(images: [URL], maxTokens: Int? = nil) throws -> PPLXEmbedV2Result {
        guard !config.isContextual, !images.isEmpty, let visionConfig = config.backbone.visionConfig,
              let imageID = config.backbone.imageTokenId,
              let startID = config.backbone.visionStartTokenId, let endID = config.backbone.visionEndTokenId else {
            throw PPLXEmbedV2Error.invalidInput("Image documents require a PPLX late model with a vision tower.")
        }
        let processor = try JSONDecoder().decode(ProcessorConfig.self,
            from: Data(contentsOf: resources.rootURL.appending(path: "processor_config.json"))).imageProcessor
        guard processor.patchSize == visionConfig.patchSize, processor.mergeSize == visionConfig.spatialMergeSize,
              processor.temporalPatchSize == visionConfig.temporalPatchSize,
              processor.imageMean == [0.5, 0.5, 0.5], processor.imageStd == [0.5, 0.5, 0.5],
              abs(processor.rescaleFactor - 1 / 255) < 1e-8, processor.resample == 3,
              processor.size.shortestEdge > 0, processor.size.longestEdge >= processor.size.shortestEdge else {
            throw PPLXEmbedV2Error.invalidConfiguration("Unsupported PPLX image processor configuration.")
        }
        let limit = try tokenLimit(task: .document, maxTokens: maxTokens)
        var prepared: [(pixels: MLXArray, grid: (Int, Int, Int), ids: [Int])] = []
        for url in images {
            guard url.isFileURL else { throw PPLXEmbedV2Error.invalidInput("PPLX images must be local files.") }
            let source = try MediaImageIO.decode(url)
            let size = try Q35Generator.qwen3VLTargetSize(originalWidth: source.width, originalHeight: source.height,
                patchSize: processor.patchSize, spatialMergeSize: processor.mergeSize,
                minPixels: processor.size.shortestEdge, maxPixels: processor.size.longestEdge)
            let grid = (1, size.height / processor.patchSize, size.width / processor.patchSize)
            let count = grid.1 * grid.2 / (processor.mergeSize * processor.mergeSize)
            guard count + 3 <= limit else {
                throw PPLXEmbedV2Error.invalidInput("Image needs \(count + 3) tokens, exceeding \(limit); resize it before encoding.")
            }
            let resized = try MediaImageIO.bicubicResizedRGB(source, width: size.width, height: size.height)
            let values = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true)
            let ids = [tokenizer.documentID, startID] + Array(repeating: imageID, count: count) + [endID]
            prepared.append((MLXArray(values, [1, 3, size.height, size.width]), grid, ids))
        }
        if vision == nil {
            let tower = Q35VisionTower(config: config.backbone, checkpointDType: .float32)
            let required = Set(tower.parameters().flattened().map(\.0))
            var loaded: Set<String> = []
            let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
                let mapped = Q35VisionTower.mapVisionWeight(key, value)
                loaded.formUnion(mapped.map(\.0))
                return mapped
            }
            if FileManager.default.fileExists(atPath: resources.indexURL.path) {
                try HFSafetensorsWeightsLoader.applyShardedWeights(indexURL: resources.indexURL, to: tower,
                    dtype: .float32, verify: [.noUnusedKeys, .shapeMismatch], mapper: mapper)
            } else {
                try HFSafetensorsWeightsLoader.applyWeights(url: resources.weightsURL, to: tower,
                    dtype: .float32, verify: [.noUnusedKeys, .shapeMismatch], mapper: mapper)
            }
            guard required.isSubset(of: loaded) else {
                throw PPLXEmbedV2Error.invalidConfiguration("Missing PPLX vision tensors: " + required.subtracting(loaded).sorted().joined(separator: ", "))
            }
            vision = tower
        }
        let rows = try prepared.enumerated().map { index, input in
            let visual = try vision!.encodeImage(pixelValues: input.pixels, gridTHW: input.grid)
            let ids = MLXArray(input.ids.map(Int32.init), [1, input.ids.count])
            let embeddings = encoder.embedTokens(ids)
            embeddings[0, 2..<(input.ids.count - 1), 0...] = visual
            let positions = Self.imagePositions(grid: input.grid, mergeSize: processor.mergeSize)
            let hidden = encoder(inputIDs: ids, embeddings: embeddings, positionIDs: positions)
            let vectors = lateVectors(hidden: hidden, ids: input.ids, task: .document)
            return PPLXEmbedV2Result.Row(index: index, embeddings: Self.readVectors(vectors), tokenCount: input.ids.count)
        }
        return result(rows: rows, dimensions: 128, normalize: true)
    }

    static func imagePositions(grid: (Int, Int, Int), mergeSize: Int) -> MLXArray {
        let height = grid.1 / mergeSize, width = grid.2 / mergeSize
        var axes = Array(repeating: [Int32(0), Int32(1)], count: 3)
        for time in 0..<grid.0 {
            for row in 0..<height {
                for column in 0..<width {
                    axes[0].append(Int32(2 + time)); axes[1].append(Int32(2 + row)); axes[2].append(Int32(2 + column))
                }
            }
        }
        for axis in axes.indices { axes[axis].append(Int32(2 + max(grid.0, height, width))) }
        return MLXArray(axes.flatMap { $0 }, [3, 1, axes[0].count])
    }
}
