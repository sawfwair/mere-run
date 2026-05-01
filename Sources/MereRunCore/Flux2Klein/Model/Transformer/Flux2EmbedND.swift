import Foundation
import MLX
import MLXNN

/// N-dimensional RoPE (Rotary Position Embedding) for FLUX.2 Klein
/// Generates position embeddings for text IDs and image IDs.
/// Returns (cos, sin) tuple format compatible with apply_rotary_emb.
final class Flux2PosEmbed: Module {
    let theta: Float
    let axesDim: [Int]

    init(theta: Float = 2000, axesDim: [Int] = [32, 32, 32, 32]) {
        self.theta = theta
        self.axesDim = axesDim
        super.init()
    }

    /// Compute rotary embeddings for position IDs
    /// - Parameter ids: Position IDs of shape [seq_len, num_axes]
    /// - Returns: Tuple of (cos, sin) each of shape [seq_len, total_dim]
    func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        var cosOut: [MLXArray] = []
        var sinOut: [MLXArray] = []
        let pos = ids.asType(.float32)

        for i in 0..<axesDim.count {
            let axisPos = pos[0..., i]  // [seq_len]
            let (cos, sin) = Self.get1DRotaryPosEmbed(
                dim: axesDim[i],
                pos: axisPos,
                theta: theta
            )
            cosOut.append(cos)
            sinOut.append(sin)
        }

        let freqsCos = MLX.concatenated(cosOut, axis: -1)
        let freqsSin = MLX.concatenated(sinOut, axis: -1)
        return (freqsCos, freqsSin)
    }

    /// Compute 1D rotary position embedding for a single axis
    /// Returns (cos, sin) each of shape [seq_len, dim/2]
    /// mflux uses arange(0, dim, 2) which gives dim/2 frequencies
    static func get1DRotaryPosEmbed(dim: Int, pos: MLXArray, theta: Float) -> (MLXArray, MLXArray) {
        // pos shape: [seq_len]
        // mflux: scale = arange(0, dim, 2) / dim → gives dim/2 values
        let scale = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
        let omega = 1.0 / pow(MLXArray(theta), scale)

        // pos: [seq_len], omega: [dim/2]
        let posExpanded = pos.expandedDimensions(axis: -1)  // [seq_len, 1]
        let angles = posExpanded * omega  // [seq_len, dim/2]

        let cosAngles = MLX.cos(angles)
        let sinAngles = MLX.sin(angles)

        return (cosAngles, sinAngles)  // each [seq_len, dim/2]
    }

    /// Apply rotary embedding to query/key tensors using mflux's apply_rope_bshd
    /// - Parameters:
    ///   - x: Input tensor of shape [batch, heads, seq_len, head_dim]
    ///   - freqs: Tuple of (cos, sin) each of shape [seq_len, head_dim/2]
    /// - Returns: Rotated tensor
    static func applyRotaryEmb(_ x: MLXArray, freqs: (MLXArray, MLXArray)) -> MLXArray {
        let (freqsCos, freqsSin) = freqs
        let outDtype = x.dtype
        let xFloat = x.asType(.float32)

        // cos/sin: [seq, dim/2] → [1, 1, seq, dim/2]
        let cosB = freqsCos.reshaped([1, 1, freqsCos.shape[0], freqsCos.shape[1]])
        let sinB = freqsSin.reshaped([1, 1, freqsSin.shape[0], freqsSin.shape[1]])

        // x: [batch, heads, seq, dim] → [batch, heads, seq, dim/2, 2]
        let xShape = xFloat.shape
        let x2 = xFloat.reshaped([xShape[0], xShape[1], xShape[2], -1, 2])

        // Complex-like rotation: (real, imag) * (cos, sin)
        let real = x2[0..., 0..., 0..., 0..., 0]  // [B, H, S, D/2]
        let imag = x2[0..., 0..., 0..., 0..., 1]  // [B, H, S, D/2]

        // out_real = real * cos - imag * sin
        // out_imag = imag * cos + real * sin
        let outReal = real * cosB + (-imag) * sinB
        let outImag = imag * cosB + real * sinB

        // Stack and reshape back to [B, H, S, D]
        let out = MLX.stacked([outReal, outImag], axis: -1)
        let result = out.reshaped(xShape)

        return result.asType(outDtype)
    }

    /// Prepare text position IDs (all zeros for 4 axes)
    /// - Parameter seqLen: Text sequence length
    /// - Returns: Position IDs of shape [seq_len, 4]
    static func prepareTextIds(seqLen: Int, numAxes: Int = 4) -> MLXArray {
        // FLUX.2 uses 4 RoPE axes. Text tokens vary only along the last axis (token index).
        // Matches draw-things/diffusers convention: [0, 0, 0, token_index]
        guard numAxes == 4 else {
            return MLX.zeros([seqLen, numAxes])
        }

        let zeros = MLX.zeros([seqLen, 3]).asType(.float32)
        let positions = MLXArray(0..<seqLen).asType(.float32).reshaped([seqLen, 1])
        return MLX.concatenated([zeros, positions], axis: 1)
    }

    /// Prepare image position IDs for FLUX.2 Klein
    /// Uses 4 axes: [0, y, x, 0]
    /// - Parameters:
    ///   - height: Patched latent height
    ///   - width: Patched latent width
    /// - Returns: Position IDs of shape [height * width, 4]
    static func prepareImageIds(height: Int, width: Int) -> MLXArray {
        let count = height * width
        var ids = [Float](repeating: 0, count: count * 4)

        for y in 0..<height {
            for x in 0..<width {
                let base = (y * width + x) * 4
                ids[base + 0] = 0
                ids[base + 1] = Float(y)
                ids[base + 2] = Float(x)
                ids[base + 3] = 0
            }
        }

        return MLXArray(ids).reshaped([count, 4])
    }

    /// Prepare position IDs for multiple images (Klein Edit multi-reference)
    /// Each image gets unique t_coord for RoPE to distinguish references.
    /// Format: [t_coord, y, x, 0] where t_coord = 10, 20, 30... for refs, 0 for generated
    /// - Parameters:
    ///   - imageCount: Number of images (reference images + 1 generated image)
    ///   - height: Patched latent height per image
    ///   - width: Patched latent width per image
    ///   - tCoords: Per-image t_coord values (e.g. [10, 20, 0] for 2 refs + generated)
    /// - Returns: Position IDs of shape [imageCount * height * width, 4]
    static func prepareMultiImageIds(
        imageCount: Int,
        height: Int,
        width: Int,
        tCoords: [Int]
    ) -> MLXArray {
        precondition(tCoords.count == imageCount, "tCoords must have one value per image")

        let seqLenPerImage = height * width
        let totalSeqLen = imageCount * seqLenPerImage
        var ids = [Float](repeating: 0, count: totalSeqLen * 4)

        for imgIdx in 0..<imageCount {
            let tCoord = Float(tCoords[imgIdx])
            for y in 0..<height {
                for x in 0..<width {
                    let tokenIdx = imgIdx * seqLenPerImage + y * width + x
                    let base = tokenIdx * 4
                    ids[base + 0] = tCoord
                    ids[base + 1] = Float(y)
                    ids[base + 2] = Float(x)
                    ids[base + 3] = 0
                }
            }
        }

        return MLXArray(ids).reshaped([totalSeqLen, 4])
    }
}
