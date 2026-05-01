import Foundation
import MLX
import MLXFast

enum ACEStepAttentionMasks {
    static func bidirectionalMask(
        attentionMask: MLXArray?,
        seqLen: Int,
        slidingWindow: Int?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let slidingWindow, slidingWindow > 0 {
            let mask = makeBidirectionalSlidingMask(attentionMask: attentionMask, seqLen: seqLen, window: slidingWindow)
            return .array(mask)
        }

        guard let attentionMask else { return .none }
        let mask = makePaddingKeyMask(attentionMask: attentionMask, queryLen: seqLen)
        return .array(mask)
    }

    private static func makePaddingKeyMask(attentionMask: MLXArray, queryLen: Int) -> MLXArray {
        // attentionMask: [B, K] (1 = keep, 0 = mask)
        let B = attentionMask.dim(0)
        let K = attentionMask.dim(1)

        let keep = (attentionMask .> MLXArray(Float(0))).asType(.bool)
        let keepExpanded = keep.reshaped(B, 1, 1, K)
        let keepBroadcast = broadcast(keepExpanded, to: [B, 1, queryLen, K])

        let zeros = MLX.zeros([B, 1, queryLen, K], dtype: .float32)
        let negInf = MLXArray(-1.0e9)
        return MLX.where(keepBroadcast, zeros, zeros + negInf)
    }

    private static func makeBidirectionalSlidingMask(
        attentionMask: MLXArray?,
        seqLen: Int,
        window: Int
    ) -> MLXArray {
        // Base geometry mask: [1, 1, L, L]
        let idx = MLXArray((0..<seqLen).map { Int32($0) }).asType(.int32)
        let i = idx.reshaped(seqLen, 1)
        let j = idx.reshaped(1, seqLen)
        let dist = MLX.abs(i - j)
        let keepGeom = (dist .<= MLXArray(Int32(window))).asType(.bool)
        var keep = broadcast(keepGeom.reshaped(1, 1, seqLen, seqLen), to: [1, 1, seqLen, seqLen])

        if let attentionMask {
            let B = attentionMask.dim(0)
            let K = attentionMask.dim(1)
            precondition(K == seqLen, "Expected attentionMask length \(seqLen), got \(K)")

            let keyKeep = (attentionMask .> MLXArray(Float(0))).asType(.bool).reshaped(B, 1, 1, seqLen)
            let keyKeepBroad = broadcast(keyKeep, to: [B, 1, seqLen, seqLen])
            let geomBroad = broadcast(keep, to: [B, 1, seqLen, seqLen])
            keep = logicalAnd(geomBroad, keyKeepBroad)
        }

        let B = keep.dim(0)
        let zeros = MLX.zeros([B, 1, seqLen, seqLen], dtype: .float32)
        let negInf = MLXArray(-1.0e9)
        return MLX.where(keep, zeros, zeros + negInf)
    }
}

