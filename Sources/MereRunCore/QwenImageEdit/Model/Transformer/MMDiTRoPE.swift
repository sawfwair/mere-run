import Foundation
import MLX
import MLXNN

/// Rotary Position Embeddings for MMDiT
/// Supports 2D (height, width) and 3D (temporal, height, width) position encoding.
public final class MMDiTRoPE {
    public let theta: Float
    public let headDim: Int
    public let axesDims: [Int]  // RoPE dimension per axis [temporal, height, width]
    private var freqsCache: [Int: MLXArray] = [:]

    public init(theta: Float = 10000.0, headDim: Int, axesDims: [Int]? = nil) {
        self.theta = theta
        self.headDim = headDim
        // Default: split evenly for 2D (no temporal)
        self.axesDims = axesDims ?? [headDim / 2, headDim / 2]
    }

    /// Precompute rotary frequencies for a given sequence length
    /// - Parameter maxSeqLen: Maximum sequence length to precompute
    /// - Returns: Frequencies tensor [maxSeqLen, headDim/2, 2] containing (cos, sin)
    public func getFreqs(seqLen: Int) -> MLXArray {
        if let cached = freqsCache[seqLen] {
            return cached
        }

        let halfDim = headDim / 2
        let idx = MLXArray(0..<halfDim).asType(.float32) * 2.0
        let exponent = -idx / MLXArray(Float(headDim))
        let freqs = MLX.pow(MLXArray(theta), exponent)

        let positions = MLXArray(0..<seqLen).asType(.float32)
        let angles = positions[.ellipsis, .newAxis] * freqs[.newAxis]

        let cosVals = MLX.cos(angles)
        let sinVals = MLX.sin(angles)

        // Stack as [seqLen, halfDim, 2] for easy indexing
        let result = MLX.stacked([cosVals, sinVals], axis: -1)
        freqsCache[seqLen] = result
        return result
    }

    /// Compute 2D position frequencies for image patches
    /// - Parameters:
    ///   - height: Number of patches in height
    ///   - width: Number of patches in width
    /// - Returns: Frequencies tensor [height*width, headDim, 2]
    public func getFreqs2D(height: Int, width: Int) -> MLXArray {
        let halfDim = headDim / 2
        let quarterDim = halfDim / 2

        // Separate frequencies for height and width
        let idxH = MLXArray(0..<quarterDim).asType(.float32) * 2.0
        let exponentH = -idxH / MLXArray(Float(halfDim))
        let freqsH = MLX.pow(MLXArray(theta), exponentH)

        let idxW = MLXArray(0..<quarterDim).asType(.float32) * 2.0
        let exponentW = -idxW / MLXArray(Float(halfDim))
        let freqsW = MLX.pow(MLXArray(theta), exponentW)

        // Create position grids
        let posH = MLXArray(0..<height).asType(.float32)
        let posW = MLXArray(0..<width).asType(.float32)

        // Compute angles for height
        let anglesH = posH[.ellipsis, .newAxis] * freqsH[.newAxis]  // [H, quarterDim]
        let cosH = MLX.cos(anglesH)
        let sinH = MLX.sin(anglesH)

        // Compute angles for width
        let anglesW = posW[.ellipsis, .newAxis] * freqsW[.newAxis]  // [W, quarterDim]
        let cosW = MLX.cos(anglesW)
        let sinW = MLX.sin(anglesW)

        // Expand to grid: [H, W, quarterDim]
        let cosHGrid = MLX.broadcast(cosH[.ellipsis, .newAxis, 0...], to: [height, width, quarterDim])
        let sinHGrid = MLX.broadcast(sinH[.ellipsis, .newAxis, 0...], to: [height, width, quarterDim])
        let cosWGrid = MLX.broadcast(cosW[.newAxis], to: [height, width, quarterDim])
        let sinWGrid = MLX.broadcast(sinW[.newAxis], to: [height, width, quarterDim])

        // Concatenate height and width components: [H, W, halfDim]
        let cosAll = MLX.concatenated([cosHGrid, cosWGrid], axis: -1)
        let sinAll = MLX.concatenated([sinHGrid, sinWGrid], axis: -1)

        // Reshape to [H*W, halfDim, 2]
        let seqLen = height * width
        let result = MLX.stacked([cosAll.reshaped(seqLen, halfDim), sinAll.reshaped(seqLen, halfDim)], axis: -1)

        return result
    }

    /// Apply rotary position embedding to query and key tensors
    /// - Parameters:
    ///   - query: Query tensor [batch, seqLen, heads, headDim]
    ///   - key: Key tensor [batch, seqLen, heads, headDim]
    ///   - freqsCis: Precomputed frequencies [seqLen, halfDim, 2] or broadcastable
    /// - Returns: Rotated (query, key) tuple
    public func apply(
        query: MLXArray,
        key: MLXArray,
        freqsCis: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Split into real and imaginary parts (rotate pairs of features)
        let halfDim = query.dim(-1) / 2

        let qReal = query[0..., 0..., 0..., 0..<halfDim]
        let qImag = query[0..., 0..., 0..., halfDim...]
        let kReal = key[0..., 0..., 0..., 0..<halfDim]
        let kImag = key[0..., 0..., 0..., halfDim...]

        // Extract cos and sin from freqsCis
        // freqsCis shape: [seqLen, halfDim, 2] where last dim is (cos, sin)
        let cos = freqsCis[0..., 0..., 0]  // [seqLen, halfDim]
        let sin = freqsCis[0..., 0..., 1]  // [seqLen, halfDim]

        // Add head dimension for broadcasting: [1, seqLen, 1, halfDim]
        let cosExpanded = cos[.newAxis, .ellipsis, .newAxis, 0...]
        let sinExpanded = sin[.newAxis, .ellipsis, .newAxis, 0...]

        // Apply rotation: (x + iy) * (cos + i*sin) = (x*cos - y*sin) + i(x*sin + y*cos)
        let qRotReal = qReal * cosExpanded - qImag * sinExpanded
        let qRotImag = qReal * sinExpanded + qImag * cosExpanded
        let kRotReal = kReal * cosExpanded - kImag * sinExpanded
        let kRotImag = kReal * sinExpanded + kImag * cosExpanded

        // Concatenate back
        let qRot = MLX.concatenated([qRotReal, qRotImag], axis: -1)
        let kRot = MLX.concatenated([kRotReal, kRotImag], axis: -1)

        return (qRot, kRot)
    }
}

/// Position ID builder for MMDiT
public struct MMDiTPositionBuilder {
    /// Build position IDs for image patches in 2D grid
    /// - Parameters:
    ///   - height: Number of patches in height
    ///   - width: Number of patches in width
    /// - Returns: Position IDs [height*width, 2] where each row is (y, x)
    public static func build2D(height: Int, width: Int) -> MLXArray {
        let seqLen = height * width
        var hPositions: [Int32] = []
        var wPositions: [Int32] = []
        hPositions.reserveCapacity(seqLen)
        wPositions.reserveCapacity(seqLen)

        for h in 0..<height {
            for w in 0..<width {
                hPositions.append(Int32(h))
                wPositions.append(Int32(w))
            }
        }

        let hArr = MLXArray(hPositions).reshaped(seqLen, 1)
        let wArr = MLXArray(wPositions).reshaped(seqLen, 1)
        return MLX.concatenated([hArr, wArr], axis: 1)
    }

    /// Build position IDs for 3D patches (temporal, height, width)
    /// - Parameters:
    ///   - temporal: Number of frames
    ///   - height: Number of patches in height
    ///   - width: Number of patches in width
    /// - Returns: Position IDs [temporal*height*width, 3] where each row is (t, y, x)
    public static func build3D(temporal: Int, height: Int, width: Int) -> MLXArray {
        let seqLen = temporal * height * width
        var tPositions: [Int32] = []
        var hPositions: [Int32] = []
        var wPositions: [Int32] = []
        tPositions.reserveCapacity(seqLen)
        hPositions.reserveCapacity(seqLen)
        wPositions.reserveCapacity(seqLen)

        for t in 0..<temporal {
            for h in 0..<height {
                for w in 0..<width {
                    tPositions.append(Int32(t))
                    hPositions.append(Int32(h))
                    wPositions.append(Int32(w))
                }
            }
        }

        let tArr = MLXArray(tPositions).reshaped(seqLen, 1)
        let hArr = MLXArray(hPositions).reshaped(seqLen, 1)
        let wArr = MLXArray(wPositions).reshaped(seqLen, 1)
        return MLX.concatenated([tArr, hArr, wArr], axis: 1)
    }
}
