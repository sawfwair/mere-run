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
        jointAttention(
            x: x,
            context: context,
            xFreqsCis: xFreqsCis,
            contextFreqsCis: contextFreqsCis,
            attnMask: attnMask,
            dynamicSparseRuntime: nil,
            layerIndex: 0
        )
    }

    func jointAttention(
        x: MLXArray,
        context: MLXArray,
        xFreqsCis: MLXArray? = nil,
        contextFreqsCis: MLXArray? = nil,
        attnMask: MLXArray? = nil,
        dynamicSparseRuntime: DynamicSparseAttentionRuntime? = nil,
        layerIndex: Int = 0
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

        // Concatenate streams for joint attention
        let q = MLX.concatenated([qImg, qCtx], axis: 1)  // [batch, imgSeq+ctxSeq, heads, headDim]
        let k = MLX.concatenated([kImg, kCtx], axis: 1)
        let v = MLX.concatenated([vImg, vCtx], axis: 1)

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

        let sparse: MLXArray? = attnMask == nil ? dynamicSparseRuntime.flatMap { runtime in
            // Qwen stores image tokens first. Reorder only the attention axes
            // so context stays exact and the generated image becomes the tail.
            let reorderedQ = MLX.concatenated([
                qT[0..., 0..., imgSeqLen..., 0...],
                qT[0..., 0..., 0..<imgSeqLen, 0...],
            ], axis: 2)
            let reorderedK = MLX.concatenated([
                kT[0..., 0..., imgSeqLen..., 0...],
                kT[0..., 0..., 0..<imgSeqLen, 0...],
            ], axis: 2)
            let reorderedV = MLX.concatenated([
                vT[0..., 0..., imgSeqLen..., 0...],
                vT[0..., 0..., 0..<imgSeqLen, 0...],
            ], axis: 2)
            guard let reorderedOutput = runtime.call(
                queries: reorderedQ,
                keys: reorderedK,
                values: reorderedV,
                layerIndex: layerIndex,
                prefixTokenCount: ctxSeqLen,
                scale: scale
            ) else { return nil }
            return MLX.concatenated([
                reorderedOutput[0..., 0..., ctxSeqLen..., 0...],
                reorderedOutput[0..., 0..., 0..<ctxSeqLen, 0...],
            ], axis: 2)
        } : nil
        let attnOut = sparse ?? MLXFast.scaledDotProductAttention(
            queries: qT,
            keys: kT,
            values: vT,
            scale: scale,
            mask: attnMask
        )

        // Reshape and split back
        let output = attnOut.transposed(0, 2, 1, 3).reshaped(batch, imgSeqLen + ctxSeqLen, numHeads * headDim)

        let imgOut = toOut(output[0..., 0..<imgSeqLen, 0...])
        let ctxOut = toAddOut(output[0..., imgSeqLen..., 0...])

        return (imgOut, ctxOut)
    }

    // MARK: - Private Helpers

    private func applyRoPE(
        query: MLXArray,
        key: MLXArray,
        freqsCis: MLXArray
    ) -> (MLXArray, MLXArray) {
        // freqsCis shape: [seqLen, halfDim, 2] where last dim is (cos, sin)
        let halfDim = query.dim(-1) / 2

        let qReal = query[0..., 0..., 0..., 0..<halfDim]
        let qImag = query[0..., 0..., 0..., halfDim...]
        let kReal = key[0..., 0..., 0..., 0..<halfDim]
        let kImag = key[0..., 0..., 0..., halfDim...]

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

        let qRot = MLX.concatenated([qRotReal, qRotImag], axis: -1)
        let kRot = MLX.concatenated([kRotReal, kRotImag], axis: -1)

        return (qRot, kRot)
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
