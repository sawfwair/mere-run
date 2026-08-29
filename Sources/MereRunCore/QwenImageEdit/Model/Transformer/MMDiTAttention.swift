import Foundation
import MLX
import MLXFast
import MLXNN

/// Multi-stream attention for MMDiT
/// Supports both joint attention (text + image) and single-stream attention.
public final class MMDiTAttention: Module {
    public let dim: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float
    public let useQKNorm: Bool

    // Projections for the primary stream (image/latent)
    @ModuleInfo(key: "to_q") var toQ: any DenseLayer
    @ModuleInfo(key: "to_k") var toK: any DenseLayer
    @ModuleInfo(key: "to_v") var toV: any DenseLayer
    @ModuleInfo(key: "to_out") var toOut: any DenseLayer

    // QK normalization (optional)
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm?
    @ModuleInfo(key: "norm_k") var normK: RMSNorm?

    // Additional projections for context stream in joint attention
    @ModuleInfo(key: "add_q_proj") var addQProj: (any DenseLayer)?
    @ModuleInfo(key: "add_k_proj") var addKProj: (any DenseLayer)?
    @ModuleInfo(key: "add_v_proj") var addVProj: (any DenseLayer)?
    @ModuleInfo(key: "to_add_out") var toAddOut: (any DenseLayer)?
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm?
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm?

    /// Initialize MMDiT attention
    /// - Parameters:
    ///   - dim: Hidden dimension
    ///   - numHeads: Number of attention heads
    ///   - numKVHeads: Number of key-value heads (for GQA)
    ///   - headDim: Dimension per head (defaults to dim/numHeads)
    ///   - qkNorm: Whether to apply QK normalization
    ///   - hasContextProjection: Whether to include context stream projections (for joint blocks)
    ///   - contextDim: Dimension of context stream (defaults to dim)
    ///   - normEps: Epsilon for normalization layers
    ///   - factory: Factory for creating dense layers
    ///   - basePath: Base path for weight keys
    public init(
        dim: Int,
        numHeads: Int,
        numKVHeads: Int? = nil,
        headDim: Int? = nil,
        qkNorm: Bool = true,
        hasContextProjection: Bool = false,
        contextDim: Int? = nil,
        normEps: Float = 1e-6,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads
        self.headDim = headDim ?? (dim / numHeads)
        self.scale = 1.0 / sqrt(Float(self.headDim))
        self.useQKNorm = qkNorm

        let innerDim = numHeads * self.headDim
        let kvDim = self.numKVHeads * self.headDim
        let path = basePath.isEmpty ? "" : "\(basePath)."

        // Primary stream projections
        self._toQ.wrappedValue = factory.makeDenseLayer(path: "\(path)to_q", inputDim: dim, outputDim: innerDim, bias: true)
        self._toK.wrappedValue = factory.makeDenseLayer(path: "\(path)to_k", inputDim: dim, outputDim: kvDim, bias: true)
        self._toV.wrappedValue = factory.makeDenseLayer(path: "\(path)to_v", inputDim: dim, outputDim: kvDim, bias: true)
        self._toOut.wrappedValue = factory.makeDenseLayer(path: "\(path)to_out", inputDim: innerDim, outputDim: dim, bias: true)

        // QK normalization
        if qkNorm {
            self._normQ.wrappedValue = RMSNorm(dimensions: self.headDim, eps: normEps)
            self._normK.wrappedValue = RMSNorm(dimensions: self.headDim, eps: normEps)
        }

        // Context projections for joint attention
        if hasContextProjection {
            let ctxDim = contextDim ?? dim
            self._addQProj.wrappedValue = factory.makeDenseLayer(path: "\(path)add_q_proj", inputDim: ctxDim, outputDim: innerDim, bias: true)
            self._addKProj.wrappedValue = factory.makeDenseLayer(path: "\(path)add_k_proj", inputDim: ctxDim, outputDim: kvDim, bias: true)
            self._addVProj.wrappedValue = factory.makeDenseLayer(path: "\(path)add_v_proj", inputDim: ctxDim, outputDim: kvDim, bias: true)
            self._toAddOut.wrappedValue = factory.makeDenseLayer(path: "\(path)to_add_out", inputDim: innerDim, outputDim: ctxDim, bias: true)

            if qkNorm {
                self._normAddedQ.wrappedValue = RMSNorm(dimensions: self.headDim, eps: normEps)
                self._normAddedK.wrappedValue = RMSNorm(dimensions: self.headDim, eps: normEps)
            }
        }

        super.init()
    }

    /// Single-stream self-attention
    /// - Parameters:
    ///   - x: Input tensor [batch, seqLen, dim]
    ///   - freqsCis: Optional RoPE frequencies
    ///   - attnMask: Optional attention mask
    /// - Returns: Output tensor [batch, seqLen, dim]
    public func callAsFunction(
        _ x: MLXArray,
        freqsCis: MLXArray? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        // Project to Q, K, V
        var q = toQ(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = toK(x).reshaped(batch, seqLen, numKVHeads, headDim)
        let v = toV(x).reshaped(batch, seqLen, numKVHeads, headDim)

        // Apply QK normalization
        if useQKNorm {
            if let normQ { q = normQ(q) }
            if let normK { k = normK(k) }
        }

        // Apply RoPE if provided
        if let freqsCis {
            (q, k) = applyRoPE(query: q, key: k, freqsCis: freqsCis)
        }

        // Transpose to [batch, heads, seqLen, headDim]
        let qT = q.transposed(0, 2, 1, 3)
        var kT = k.transposed(0, 2, 1, 3)
        var vT = v.transposed(0, 2, 1, 3)

        // Handle GQA by repeating KV heads
        if numKVHeads < numHeads {
            let repeats = numHeads / numKVHeads
            kT = repeatKV(kT, repeats: repeats)
            vT = repeatKV(vT, repeats: repeats)
        }

        // Scaled dot-product attention
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qT,
            keys: kT,
            values: vT,
            scale: scale,
            mask: attnMask
        )

        // Reshape back: [batch, seqLen, innerDim]
        let output = attnOut.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)

        return toOut(output)
    }

    /// Joint attention for text and image streams
    /// - Parameters:
    ///   - x: Image/latent stream [batch, imgSeqLen, dim]
    ///   - context: Text/context stream [batch, ctxSeqLen, contextDim]
    ///   - xFreqsCis: RoPE frequencies for image stream
    ///   - contextFreqsCis: RoPE frequencies for context stream (optional)
    ///   - attnMask: Optional attention mask
    /// - Returns: (imageOut, contextOut) tuple
    public func jointAttention(
        x: MLXArray,
        context: MLXArray,
        xFreqsCis: MLXArray? = nil,
        contextFreqsCis: MLXArray? = nil,
        attnMask: MLXArray? = nil
    ) -> (image: MLXArray, context: MLXArray) {
        guard let addQProj, let addKProj, let addVProj, let toAddOut else {
            fatalError("Joint attention requires context projections (hasContextProjection=true)")
        }

        let batch = x.dim(0)
        let imgSeqLen = x.dim(1)
        let ctxSeqLen = context.dim(1)

        // Project image stream
        var qImg = toQ(x).reshaped(batch, imgSeqLen, numHeads, headDim)
        var kImg = toK(x).reshaped(batch, imgSeqLen, numKVHeads, headDim)
        let vImg = toV(x).reshaped(batch, imgSeqLen, numKVHeads, headDim)

        // Project context stream
        var qCtx = addQProj(context).reshaped(batch, ctxSeqLen, numHeads, headDim)
        var kCtx = addKProj(context).reshaped(batch, ctxSeqLen, numKVHeads, headDim)
        let vCtx = addVProj(context).reshaped(batch, ctxSeqLen, numKVHeads, headDim)

        // Apply QK normalization
        if useQKNorm {
            if let normQ { qImg = normQ(qImg) }
            if let normK { kImg = normK(kImg) }
            if let normAddedQ { qCtx = normAddedQ(qCtx) }
            if let normAddedK { kCtx = normAddedK(kCtx) }
        }

        // Apply RoPE separately to each stream
        if let xFreqsCis {
            (qImg, kImg) = applyRoPE(query: qImg, key: kImg, freqsCis: xFreqsCis)
        }
        if let contextFreqsCis {
            (qCtx, kCtx) = applyRoPE(query: qCtx, key: kCtx, freqsCis: contextFreqsCis)
        }

        // Qwen's joint stream is ordered [text, image].
        let q = MLX.concatenated([qCtx, qImg], axis: 1)
        let k = MLX.concatenated([kCtx, kImg], axis: 1)
        let v = MLX.concatenated([vCtx, vImg], axis: 1)

        // Transpose for attention
        let qT = q.transposed(0, 2, 1, 3)
        var kT = k.transposed(0, 2, 1, 3)
        var vT = v.transposed(0, 2, 1, 3)

        // Handle GQA
        if numKVHeads < numHeads {
            let repeats = numHeads / numKVHeads
            kT = repeatKV(kT, repeats: repeats)
            vT = repeatKV(vT, repeats: repeats)
        }

        // Scaled dot-product attention
        let mask = jointAttentionMask(
            textMask: attnMask,
            imageSequenceLength: imgSeqLen
        )
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qT,
            keys: kT,
            values: vT,
            scale: scale,
            mask: mask
        )

        // Reshape and split back
        let output = attnOut.transposed(0, 2, 1, 3).reshaped(batch, imgSeqLen + ctxSeqLen, numHeads * headDim)

        let ctxOut = toAddOut(output[0..., 0..<ctxSeqLen, 0...])
        let imgOut = toOut(output[0..., ctxSeqLen..., 0...])

        return (imgOut, ctxOut)
    }

    // MARK: - Private Helpers

    private func applyRoPE(
        query: MLXArray,
        key: MLXArray,
        freqsCis: MLXArray
    ) -> (MLXArray, MLXArray) {
        let halfDim = query.dim(-1) / 2
        let queryPairs = query.asType(.float32).reshaped(
            query.dim(0), query.dim(1), query.dim(2), halfDim, 2
        )
        let keyPairs = key.asType(.float32).reshaped(
            key.dim(0), key.dim(1), key.dim(2), halfDim, 2
        )
        let qReal = queryPairs[0..., 0..., 0..., 0..., 0]
        let qImag = queryPairs[0..., 0..., 0..., 0..., 1]
        let kReal = keyPairs[0..., 0..., 0..., 0..., 0]
        let kImag = keyPairs[0..., 0..., 0..., 0..., 1]

        // Extract cos and sin: [seqLen, halfDim]
        let cos = freqsCis[0..., 0..., 0]
        let sin = freqsCis[0..., 0..., 1]

        // Expand for broadcasting: [1, seqLen, 1, halfDim]
        let cosExp = cos[.newAxis, .ellipsis, .newAxis, 0...]
        let sinExp = sin[.newAxis, .ellipsis, .newAxis, 0...]

        // Apply rotation
        let qRotReal = qReal * cosExp - qImag * sinExp
        let qRotImag = qReal * sinExp + qImag * cosExp
        let kRotReal = kReal * cosExp - kImag * sinExp
        let kRotImag = kReal * sinExp + kImag * cosExp

        let qRot = MLX.stacked([qRotReal, qRotImag], axis: -1)
            .reshaped(query.shape)
            .asType(query.dtype)
        let kRot = MLX.stacked([kRotReal, kRotImag], axis: -1)
            .reshaped(key.shape)
            .asType(key.dtype)

        return (qRot, kRot)
    }

    private func jointAttentionMask(
        textMask: MLXArray?,
        imageSequenceLength: Int
    ) -> MLXArray? {
        guard let textMask else {
            return nil
        }
        let imageMask = MLXArray.ones([textMask.dim(0), imageSequenceLength], dtype: .float32)
        let jointMask = MLX.concatenated([textMask.asType(.float32), imageMask], axis: 1)
        let additiveMask = (1 - jointMask) * -1e9
        return additiveMask.reshaped(additiveMask.dim(0), 1, 1, additiveMask.dim(1))
    }

    private func repeatKV(_ x: MLXArray, repeats: Int) -> MLXArray {
        if repeats == 1 { return x }
        // x shape: [batch, kvHeads, seqLen, headDim]
        // Output: [batch, kvHeads * repeats, seqLen, headDim]
        let batch = x.dim(0)
        let kvHeads = x.dim(1)
        let seqLen = x.dim(2)
        let hDim = x.dim(3)

        // Expand and reshape
        let expanded = x[.ellipsis, .newAxis, 0...]  // [batch, kvHeads, 1, seqLen, headDim]
        let tiled = MLX.broadcast(expanded, to: [batch, kvHeads, repeats, seqLen, hDim])
        return tiled.reshaped(batch, kvHeads * repeats, seqLen, hDim)
    }
}
