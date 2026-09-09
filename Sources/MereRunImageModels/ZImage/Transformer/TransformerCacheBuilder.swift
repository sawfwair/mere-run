import Foundation
@preconcurrency import MLX
public struct TransformerCacheKey: Hashable, Sendable {
    public let batch: Int
    public let height: Int
    public let width: Int
    public let frames: Int
    public let capOriLen: Int

    public init(batch: Int, height: Int, width: Int, frames: Int, capOriLen: Int) {
        self.batch = batch
        self.height = height
        self.width = width
        self.frames = frames
        self.capOriLen = capOriLen
    }
}

public struct TransformerCache: @unchecked Sendable {
    public let capFreqs: MLXArray
    public let capPadMask: MLXArray?
    public let capSeqLen: Int
    public let capPad: Int

    public let imgFreqs: MLXArray
    public let imgPadMask: MLXArray?
    public let imgSeqLen: Int
    public let imgPad: Int
    public let imageTokens: Int

    public let unifiedFreqsCis: MLXArray

    public let fTokens: Int
    public let hTokens: Int
    public let wTokens: Int

    public init(
        capFreqs: MLXArray,
        capPadMask: MLXArray?,
        capSeqLen: Int,
        capPad: Int,
        imgFreqs: MLXArray,
        imgPadMask: MLXArray?,
        imgSeqLen: Int,
        imgPad: Int,
        imageTokens: Int,
        unifiedFreqsCis: MLXArray,
        fTokens: Int,
        hTokens: Int,
        wTokens: Int
    ) {
        self.capFreqs = capFreqs
        self.capPadMask = capPadMask
        self.capSeqLen = capSeqLen
        self.capPad = capPad
        self.imgFreqs = imgFreqs
        self.imgPadMask = imgPadMask
        self.imgSeqLen = imgSeqLen
        self.imgPad = imgPad
        self.imageTokens = imageTokens
        self.unifiedFreqsCis = unifiedFreqsCis
        self.fTokens = fTokens
        self.hTokens = hTokens
        self.wTokens = wTokens
    }
}

public struct TransformerCacheBuilder {
    public static let seqMultiOf = 32
    public static func build(
        batch: Int,
        height: Int,
        width: Int,
        frames: Int,
        capOriLen: Int,
        patchSize: Int,
        fPatchSize: Int,
        ropeEmbedder: ZImageRopeEmbedder
    ) -> TransformerCache {
        let capPad = (seqMultiOf - (capOriLen % seqMultiOf)) % seqMultiOf
        let capSeqLen = capOriLen + capPad

        let capPosIds = ZImageCoordinateUtils.createCoordinateGrid(
            size: (capSeqLen, 1, 1),
            start: (1, 0, 0)
        ).reshaped(capSeqLen, 3)
        let capFreqs = ropeEmbedder(ids: capPosIds)

        var capPadMask: MLXArray? = nil
        if capPad > 0 {
            let capPadMask1d = MLX.concatenated([
                MLX.zeros([capOriLen], dtype: .bool),
                MLX.ones([capPad], dtype: .bool)
            ], axis: 0)
            capPadMask = MLX.broadcast(capPadMask1d.reshaped(1, capSeqLen), to: [batch, capSeqLen])
        }

        let fTokens = frames / fPatchSize
        let hTokens = height / patchSize
        let wTokens = width / patchSize
        let imageTokens = fTokens * hTokens * wTokens
        let imgPad = (seqMultiOf - (imageTokens % seqMultiOf)) % seqMultiOf
        let imgSeqLen = imageTokens + imgPad

        let imgPos = ZImageCoordinateUtils.createCoordinateGrid(
            size: (fTokens, hTokens, wTokens),
            start: (capSeqLen + 1, 0, 0)
        ).reshaped(imageTokens, 3)
        let imgPadPos = ZImageCoordinateUtils.createCoordinateGrid(
            size: (imgPad, 1, 1),
            start: (0, 0, 0)
        ).reshaped(imgPad, 3)
        let imgPosIds = MLX.concatenated([imgPos, imgPadPos], axis: 0)
        let imgFreqs = ropeEmbedder(ids: imgPosIds)

        var imgPadMask: MLXArray? = nil
        if imgPad > 0 {
            let imgPadMask1d = MLX.concatenated([
                MLX.zeros([imageTokens], dtype: .bool),
                MLX.ones([imgPad], dtype: .bool)
            ], axis: 0)
            imgPadMask = MLX.broadcast(imgPadMask1d.reshaped(1, imgSeqLen), to: [batch, imgSeqLen])
        }

        let unifiedFreqsCis = MLX.concatenated([imgFreqs, capFreqs], axis: 0)

        return TransformerCache(
            capFreqs: capFreqs,
            capPadMask: capPadMask,
            capSeqLen: capSeqLen,
            capPad: capPad,
            imgFreqs: imgFreqs,
            imgPadMask: imgPadMask,
            imgSeqLen: imgSeqLen,
            imgPad: imgPad,
            imageTokens: imageTokens,
            unifiedFreqsCis: unifiedFreqsCis,
            fTokens: fTokens,
            hTokens: hTokens,
            wTokens: wTokens
        )
    }
}
