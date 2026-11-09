import Foundation
import MLX

extension MuseGlimmerGenerator {
    func cachedImageIdentities(tokens: [Int], imageBatch: MuseGlimmerImageBatch?) throws -> [MuseGlimmerCachedImage] {
        guard let imageBatch, let tokenizerAndTemplate else { return [] }
        let imageToken = tokenizerAndTemplate.imageTokenId
        let offsets = tokens.indices.filter { tokens[$0] == imageToken && ($0 == 0 || tokens[$0 - 1] != imageToken) }
        guard offsets.count == imageBatch.identities.count else {
            throw MuseGlimmerError.unsupportedConfiguration("Image identities do not match the prepared prompt's image blocks.")
        }
        return zip(offsets, imageBatch.identities).map { MuseGlimmerCachedImage(offset: $0, identity: $1) }
    }

    func imageInputEmbeddings(model: MuseGlimmerModel, tokens: [Int], batch: MuseGlimmerImageBatch,
                              identity: String, retain: Bool, offset: Int = 0) throws -> MLXArray {
        let ids = MLXArray(tokens.dropFirst(offset).map(Int32.init)).reshaped(1, tokens.count - offset)
        guard retain else { return try model.inputEmbeddings(for: ids, imageBatch: batch) }
        var features: [MLXArray] = []
        var pixelOffset = 0
        var featureOffset = 0
        let positions = tokens.indices.filter { tokens[$0] == model.config.imageTokenId }
        guard positions.count == batch.tokenCounts.reduce(0, +) else {
            throw MuseGlimmerError.unsupportedConfiguration("Prepared image tokens do not match the image feature count.")
        }
        for (index, grid) in batch.grids.enumerated() {
            try Task.checkCancellation()
            let count = grid.temporal * grid.height * grid.width
            let imageCount = batch.tokenCounts[index]
            let skipped = positions[featureOffset..<(featureOffset + imageCount)].prefix { $0 < offset }.count
            defer { pixelOffset += count; featureOffset += imageCount }
            if skipped == imageCount { continue }
            let featureTokens = Array(repeating: 0, count: imageCount)
            let key = identity + "\u{0}" + batch.identities[index]
            if let hit = imageFeatureCache.lookup(identity: key, tokens: featureTokens) {
                features.append(hit.value[skipped..., 0...])
            } else {
                let single = MuseGlimmerImageBatch(pixelValues: batch.pixelValues[pixelOffset..<(pixelOffset + count), 0...],
                    grids: [grid], tokenCounts: [batch.tokenCounts[index]], identities: [batch.identities[index]])
                let value = model.model.encodeImages(single)
                MLX.eval(value)
                imageFeatureCache.store(identity: key, tokens: featureTokens, value: value, ttl: 300)
                features.append(value[skipped..., 0...])
            }
        }
        if features.isEmpty { return model.inputEmbeddings(for: ids) }
        return try model.inputEmbeddings(for: ids, imageBatch: batch, imageFeatures: concatenated(features, axis: 0))
    }
}
