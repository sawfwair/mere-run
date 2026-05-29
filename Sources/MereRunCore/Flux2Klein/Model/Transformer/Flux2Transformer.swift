import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Flux2TransformerConfiguration: Sendable {
    public let hiddenSize: Int           // 3072
    public let numHeads: Int             // 24
    public let headDim: Int              // 128
    public let numLayers: Int            // 5 (joint blocks)
    public let numSingleLayers: Int      // 20 (single blocks)
    public let inChannels: Int           // 128 (latent channels after patchify)
    public let contextDim: Int           // 7680 (text encoder output dim)
    public let mlpRatio: Float           // 3.0
    public let eps: Float                // 1e-6
    public let ropeTheta: Float          // 2000
    public let axesDimsRope: [Int]       // [32, 32, 32, 32]
    public let guidanceEmbeds: Bool      // false for base models
    public let quantized: Bool           // true to use QuantizedLinear layers

    public init(
        hiddenSize: Int = 3072,
        numHeads: Int = 24,
        headDim: Int = 128,
        numLayers: Int = 5,
        numSingleLayers: Int = 20,
        inChannels: Int = 128,
        contextDim: Int = 7680,
        mlpRatio: Float = 3.0,
        eps: Float = 1e-6,
        ropeTheta: Float = 2000,
        axesDimsRope: [Int] = [32, 32, 32, 32],
        guidanceEmbeds: Bool = false,
        quantized: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.headDim = headDim
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.inChannels = inChannels
        self.contextDim = contextDim
        self.mlpRatio = mlpRatio
        self.eps = eps
        self.ropeTheta = ropeTheta
        self.axesDimsRope = axesDimsRope
        self.guidanceEmbeds = guidanceEmbeds
        self.quantized = quantized
    }

    public var mlpHiddenSize: Int { Int(Float(hiddenSize) * mlpRatio) }
}

// MARK: - Linear Factory for Quantized Support

/// Factory for creating Linear or QuantizedLinear layers based on configuration.
/// For quantized models, weights must be provided; for non-quantized, dimensions suffice.
public enum Flux2LinearFactory {
    /// Creates a Linear layer (non-quantized)
    public static func makeLinear(_ inputDim: Int, _ outputDim: Int, bias: Bool = false) -> Linear {
        Linear(inputDim, outputDim, bias: bias)
    }

    /// Creates a QuantizedLinear layer from pre-quantized weights
    public static func makeQuantizedLinear(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        bias: MLXArray? = nil,
        groupSize: Int = 64,
        bits: Int = 4
    ) -> QuantizedLinear {
        PortableQuantizedLinear(
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits
        )
    }
}

// MARK: - Timestep Embedder

final class Flux2TimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inDim: Int = 256, hiddenSize: Int = 3072) {
        self._linear1.wrappedValue = Linear(inDim, hiddenSize, bias: false)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = silu(h)
        h = linear2(h)
        return h
    }
}

// MARK: - Time Guidance Embed

final class Flux2TimeGuidanceEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: Flux2TimestepEmbedder

    init(hiddenSize: Int = 3072) {
        self._timestepEmbedder.wrappedValue = Flux2TimestepEmbedder(inDim: 256, hiddenSize: hiddenSize)
        super.init()
    }

    func callAsFunction(timestep: MLXArray, guidance: MLXArray? = nil) -> MLXArray {
        let timestepProj = Self.timestepProjection(timestep, dim: 256)
        let timestepEmb = timestepEmbedder(timestepProj)

        // FLUX.2 Klein 4B doesn't have guidance layers (guidance_embeds=false)
        // mflux always passes guidance=None for this model
        // Only models with guidance_embeds=true use the guidance parameter
        return timestepEmb
    }

    /// Sinusoidal timestep projection (flip_sin_to_cos=True, downscale_freq_shift=0)
    static func timestepProjection(_ timesteps: MLXArray, dim: Int = 256) -> MLXArray {
        let halfDim = dim / 2
        let indices = MLXArray(0..<halfDim).asType(.float32)
        let exponent = -log(Float(10000)) * indices / Float(halfDim)
        let emb = exp(exponent)
        let t = timesteps.asType(.float32)

        // t: [batch], emb: [halfDim]
        let tExpanded = t.expandedDimensions(axis: -1)
        let angles = tExpanded * emb

        // flip_sin_to_cos=True: cos first, then sin
        let cosAngles = cos(angles)
        let sinAngles = sin(angles)
        let combined = concatenated([cosAngles, sinAngles], axis: -1)
        return combined
    }
}

// MARK: - Modulation

final class Flux2Modulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let modParamSets: Int

    init(hiddenSize: Int, modParamSets: Int = 2) {
        self.modParamSets = modParamSets
        // 3 params (shift, scale, gate) per set
        self._linear.wrappedValue = Linear(hiddenSize, hiddenSize * 3 * modParamSets, bias: false)
        super.init()
    }

    /// Returns array of (shift, scale, gate) tuples
    func callAsFunction(_ temb: MLXArray) -> [[(MLXArray, MLXArray, MLXArray)]] {
        var mod = silu(temb)
        mod = linear(mod)

        // Expand for broadcasting: [batch, 1, hidden*3*sets]
        if mod.ndim == 2 {
            mod = mod.expandedDimensions(axis: 1)
        }

        // Split into 3*modParamSets chunks
        let chunks = MLX.split(mod, parts: 3 * modParamSets, axis: -1)

        var result: [[(MLXArray, MLXArray, MLXArray)]] = []
        for i in 0..<modParamSets {
            // Diffusers order: shift, scale, gate
            let shift = chunks[3 * i]
            let scale = chunks[3 * i + 1]
            let gate = chunks[3 * i + 2]
            result.append([(shift, scale, gate)])
        }
        return result
    }
}

// MARK: - SwiGLU Activation

final class Flux2SwiGLU: Module {
    override init() {
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(x, parts: 2, axis: -1)
        return silu(parts[0]) * parts[1]
    }
}

// MARK: - Feed Forward

final class Flux2FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    let actFn: Flux2SwiGLU

    init(dim: Int, mult: Float = 3.0) {
        let innerDim = Int(Float(dim) * mult)
        self._linearIn.wrappedValue = Linear(dim, innerDim * 2, bias: false)
        self._linearOut.wrappedValue = Linear(innerDim, dim, bias: false)
        self.actFn = Flux2SwiGLU()
        super.init()
    }

    /// Placeholder init for iOS - creates minimal 1x1 Linear that will be replaced
    init(placeholder: Bool) {
        self._linearIn.wrappedValue = Linear(1, 1, bias: false)
        self._linearOut.wrappedValue = Linear(1, 1, bias: false)
        self.actFn = Flux2SwiGLU()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linearIn(x)
        h = actFn(h)
        h = linearOut(h)
        return h
    }
}

// MARK: - Joint Attention

final class Flux2Attention: Module {
    let numHeads: Int
    let headDim: Int

    // Image stream
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    // Context (text) stream
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    init(hiddenSize: Int = 3072, numHeads: Int = 24, headDim: Int = 128, eps: Float = 1e-6) {
        self.numHeads = numHeads
        self.headDim = headDim

        self._toQ.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._toK.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._toV.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toOut.wrappedValue = [Linear(hiddenSize, hiddenSize, bias: false)]

        self._addQProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._addKProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._addVProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toAddOut.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)

        super.init()
    }

    /// Placeholder init for iOS - creates minimal 1x1 Linear that will be replaced
    init(numHeads: Int, headDim: Int, eps: Float, placeholder: Bool) {
        self.numHeads = numHeads
        self.headDim = headDim

        self._toQ.wrappedValue = Linear(1, 1, bias: false)
        self._toK.wrappedValue = Linear(1, 1, bias: false)
        self._toV.wrappedValue = Linear(1, 1, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toOut.wrappedValue = [Linear(1, 1, bias: false)]

        self._addQProj.wrappedValue = Linear(1, 1, bias: false)
        self._addKProj.wrappedValue = Linear(1, 1, bias: false)
        self._addVProj.wrappedValue = Linear(1, 1, bias: false)
        self._normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toAddOut.wrappedValue = Linear(1, 1, bias: false)

        super.init()
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        rotaryEmb: (MLXArray, MLXArray)
    ) -> (MLXArray, MLXArray) {
        let batch = hiddenStates.shape[0]
        let hiddenSize = numHeads * headDim

        // Image stream QKV
        var q = toQ(hiddenStates)
        var k = toK(hiddenStates)
        var v = toV(hiddenStates)

        q = q.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)

        q = normQ(q)
        k = normK(k)

        // Context stream QKV
        var addQ = addQProj(encoderHiddenStates)
        var addK = addKProj(encoderHiddenStates)
        var addV = addVProj(encoderHiddenStates)

        addQ = addQ.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        addK = addK.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        addV = addV.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)

        addQ = normAddedQ(addQ)
        addK = normAddedK(addK)

        // Concatenate for joint attention: [text, image] (context first, then image)
        let jointQ = MLX.concatenated([addQ, q], axis: 2)
        let jointK = MLX.concatenated([addK, k], axis: 2)
        let jointV = MLX.concatenated([addV, v], axis: 2)

        // Apply RoPE
        let jointQRope = Flux2PosEmbed.applyRotaryEmb(jointQ, freqs: rotaryEmb)
        let jointKRope = Flux2PosEmbed.applyRotaryEmb(jointK, freqs: rotaryEmb)

        // Scaled dot-product attention
        let attnOut = Self.attention(jointQRope, jointKRope, jointV)

        // Reshape back: [batch, heads, seq, dim] -> [batch, seq, hidden]
        var out = attnOut.transposed(0, 2, 1, 3)
        out = out.reshaped([batch, -1, hiddenSize])

        // Split back to context (text) and image - text is first
        let txtLen = encoderHiddenStates.shape[1]
        let contextOut = out[0..., 0..<txtLen, 0...]
        let hiddenOut = out[0..., txtLen..., 0...]

        // Project outputs
        let imgOut = toOut[0](hiddenOut)
        let txtOut = toAddOut(contextOut)

        return (imgOut, txtOut)
    }

	    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
	        let scale = 1.0 / sqrt(Float(q.shape.last!))
#if os(iOS)
	        // Avoid materializing a full [B, H, T, T] attention matrix on memory-constrained devices.
	        // For typical FLUX.2 seq lengths (~4k+), the naive path can exceed iOS jetsam limits.
	        let queryLen = q.shape[2]
	        let chunkSize = 128
	        if queryLen > chunkSize {
	            let kT = k.transposed(0, 1, 3, 2)
	            var outputs: [MLXArray] = []
	            outputs.reserveCapacity((queryLen + chunkSize - 1) / chunkSize)

	            for start in stride(from: 0, to: queryLen, by: chunkSize) {
	                let end = min(start + chunkSize, queryLen)
	                let out: MLXArray = {
	                    let qChunk = q[0..., 0..., start..<end, 0...]
	                    let scores = (qChunk * scale).matmul(kT)
	                    let attn = softmax(scores, axis: -1)
	                    return attn.matmul(v)
	                }()
	                MLX.eval(out)
	                outputs.append(out)
	            }

	            return MLX.concatenated(outputs, axis: 2)
	        }

	        let scores = (q * scale).matmul(k.transposed(0, 1, 3, 2))
	        let attn = softmax(scores, axis: -1)
	        return attn.matmul(v)
#else
	        return MLXFast.scaledDotProductAttention(
	            queries: q,
	            keys: k,
	            values: v,
	            scale: scale,
	            mask: .none
	        )
#endif
	    }
	}

// MARK: - Joint Transformer Block

final class Flux2TransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: Flux2Attention
    @ModuleInfo(key: "ff") var ff: Flux2FeedForward
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm

    let hiddenSize: Int
    let eps: Float

    init(config: Flux2TransformerConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.eps = config.eps

        self._attn.wrappedValue = Flux2Attention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numHeads,
            headDim: config.headDim,
            eps: config.eps
        )
        self._ff.wrappedValue = Flux2FeedForward(dim: config.hiddenSize, mult: config.mlpRatio)
        self._ffContext.wrappedValue = Flux2FeedForward(dim: config.hiddenSize, mult: config.mlpRatio)

        // LayerNorm without elementwise_affine
        self._norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        self._norm1Context.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        self._norm2Context.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)

        super.init()
    }

    /// Placeholder init for iOS - creates minimal sub-modules that will be replaced
    init(config: Flux2TransformerConfiguration, placeholder: Bool) {
        self.hiddenSize = config.hiddenSize
        self.eps = config.eps

        self._attn.wrappedValue = Flux2Attention(
            numHeads: config.numHeads,
            headDim: config.headDim,
            eps: config.eps,
            placeholder: true
        )
        self._ff.wrappedValue = Flux2FeedForward(placeholder: true)
        self._ffContext.wrappedValue = Flux2FeedForward(placeholder: true)

        self._norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        self._norm1Context.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        self._norm2Context.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)

        super.init()
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        modImg: ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)),
        modTxt: ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)),
        rotaryEmb: (MLXArray, MLXArray)
    ) -> (MLXArray, MLXArray) {
        // Unpack modulation: (msa params, mlp params)
        let ((shiftMsaImg, scaleMsaImg, gateMsaImg), (shiftMlpImg, scaleMlpImg, gateMlpImg)) = modImg
        let ((shiftMsaTxt, scaleMsaTxt, gateMsaTxt), (shiftMlpTxt, scaleMlpTxt, gateMlpTxt)) = modTxt

        // Norm + modulate image
        var normImg = norm1(hiddenStates)
        normImg = (1 + scaleMsaImg) * normImg + shiftMsaImg

        // Norm + modulate text
        var normTxt = norm1Context(encoderHiddenStates)
        normTxt = (1 + scaleMsaTxt) * normTxt + shiftMsaTxt

        // Joint attention
        let (attnOutImg, attnOutTxt) = attn(
            hiddenStates: normImg,
            encoderHiddenStates: normTxt,
            rotaryEmb: rotaryEmb
        )

        // Gated residual for image
        var imgOut = hiddenStates + gateMsaImg * attnOutImg

        // Norm + modulate + FFN for image
        var normImg2 = norm2(imgOut)
        normImg2 = (1 + scaleMlpImg) * normImg2 + shiftMlpImg
        imgOut = imgOut + gateMlpImg * ff(normImg2)

        // Gated residual for text
        var txtOut = encoderHiddenStates + gateMsaTxt * attnOutTxt

        // Norm + modulate + FFN for text
        var normTxt2 = norm2Context(txtOut)
        normTxt2 = (1 + scaleMlpTxt) * normTxt2 + shiftMlpTxt
        txtOut = txtOut + gateMlpTxt * ffContext(normTxt2)

        return (txtOut, imgOut)
    }
}

// MARK: - Parallel Self Attention (for Single Blocks)

final class Flux2ParallelSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let mlpHiddenDim: Int

    @ModuleInfo(key: "to_qkv_mlp_proj") var toQkvMlpProj: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(hiddenSize: Int, numHeads: Int, headDim: Int, mlpRatio: Float, eps: Float) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.mlpHiddenDim = Int(Float(hiddenSize) * mlpRatio)

        // Fused projection: QKV (3*hidden) + MLP gate+up (2*mlpHidden)
        let qkvDim = 3 * hiddenSize
        let mlpDim = 2 * mlpHiddenDim
        self._toQkvMlpProj.wrappedValue = Linear(hiddenSize, qkvDim + mlpDim, bias: false)

        // Fused output: attn (hidden) + mlp_down (mlpHidden)
        self._toOut.wrappedValue = Linear(hiddenSize + mlpHiddenDim, hiddenSize, bias: false)

        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)

        super.init()
    }

    /// Placeholder init for iOS - creates minimal 1x1 Linear that will be replaced
    init(hiddenSize: Int, numHeads: Int, headDim: Int, mlpRatio: Float, eps: Float, placeholder: Bool) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.mlpHiddenDim = Int(Float(hiddenSize) * mlpRatio)

        self._toQkvMlpProj.wrappedValue = Linear(1, 1, bias: false)
        self._toOut.wrappedValue = Linear(1, 1, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, rotaryEmb: (MLXArray, MLXArray)) -> MLXArray {
        let batch = x.shape[0]

        // Fused QKV + MLP projection
        let projected = toQkvMlpProj(x)

        // Split into QKV and MLP parts
        let qkvDim = 3 * hiddenSize
        let qkv = projected[0..., 0..., 0..<qkvDim]
        let mlpIn = projected[0..., 0..., qkvDim...]

        // Split QKV
        let qkvParts = MLX.split(qkv, parts: 3, axis: -1)
        var q = qkvParts[0]
        var k = qkvParts[1]
        let v = qkvParts[2]

        // Reshape for multi-head attention
        q = q.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vReshaped = v.reshaped([batch, -1, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Normalize Q and K
        q = normQ(q)
        k = normK(k)

        // Apply RoPE
        let qRope = Flux2PosEmbed.applyRotaryEmb(q, freqs: rotaryEmb)
        let kRope = Flux2PosEmbed.applyRotaryEmb(k, freqs: rotaryEmb)

        // Attention
        let attnOut = Flux2Attention.attention(qRope, kRope, vReshaped)
        var attnReshaped = attnOut.transposed(0, 2, 1, 3)
        attnReshaped = attnReshaped.reshaped([batch, -1, hiddenSize])

        // MLP with SwiGLU: silu(gate) * up
        let mlpParts = MLX.split(mlpIn, parts: 2, axis: -1)
        let mlpOut = silu(mlpParts[0]) * mlpParts[1]

        // Fused output projection
        let combined = MLX.concatenated([attnReshaped, mlpOut], axis: -1)
        return toOut(combined)
    }
}

// MARK: - Single Transformer Block

final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: Flux2ParallelSelfAttention
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let eps: Float

    init(config: Flux2TransformerConfiguration) {
        self.eps = config.eps
        self._attn.wrappedValue = Flux2ParallelSelfAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numHeads,
            headDim: config.headDim,
            mlpRatio: config.mlpRatio,
            eps: config.eps
        )
        self._norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        super.init()
    }

    /// Placeholder init for iOS - creates minimal sub-modules that will be replaced
    init(config: Flux2TransformerConfiguration, placeholder: Bool) {
        self.eps = config.eps
        self._attn.wrappedValue = Flux2ParallelSelfAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numHeads,
            headDim: config.headDim,
            mlpRatio: config.mlpRatio,
            eps: config.eps,
            placeholder: true
        )
        self._norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps, affine: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mod: (MLXArray, MLXArray, MLXArray),
        rotaryEmb: (MLXArray, MLXArray)
    ) -> MLXArray {
        let (shift, scale, gate) = mod

        // Norm + modulate
        var normed = norm(x)
        normed = (1 + scale) * normed + shift

        // Attention + FFN (fused)
        let out = attn(normed, rotaryEmb: rotaryEmb)

        // Gated residual
        return x + gate * out
    }
}

// MARK: - AdaLN Continuous (for output)

final class Flux2AdaLNContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let hiddenSize: Int
    let eps: Float

    init(hiddenSize: Int, eps: Float) {
        self.hiddenSize = hiddenSize
        self.eps = eps
        self._linear.wrappedValue = Linear(hiddenSize, hiddenSize * 2, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let params = linear(silu(cond))
        let parts = MLX.split(params, parts: 2, axis: -1)
        // Diffusers order: scale, shift
        let scale = parts[0]
        let shift = parts[1]

        // LayerNorm without affine
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axis: -1, keepDims: true)
        var normed = (x - mean) / MLX.sqrt(variance + eps)

        // Expand params for broadcasting if needed
        if shift.ndim == 2 {
            normed = normed * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
        } else {
            normed = normed * (1 + scale) + shift
        }
        return normed
    }
}

// MARK: - Full Transformer

public final class Flux2Transformer2DModel: Module {
    public let config: Flux2TransformerConfiguration

    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimeGuidanceEmbed

    @ModuleInfo(key: "double_stream_modulation_img") var doubleStreamModImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleStreamModTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var singleStreamMod: Flux2Modulation

    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]

    @ModuleInfo(key: "norm_out") var normOut: Flux2AdaLNContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let posEmbed: Flux2PosEmbed
    private var cachedImgIdsId: ObjectIdentifier?
    private var cachedTxtIdsId: ObjectIdentifier?
    private var cachedRotary: (MLXArray, MLXArray)?
    private static let compileEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_FLUX2_COMPILE"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public init(config: Flux2TransformerConfiguration) {
        self.config = config

        self._xEmbedder.wrappedValue = Linear(config.inChannels, config.hiddenSize, bias: false)
        self._contextEmbedder.wrappedValue = Linear(config.contextDim, config.hiddenSize, bias: false)
        self._timeGuidanceEmbed.wrappedValue = Flux2TimeGuidanceEmbed(hiddenSize: config.hiddenSize)

        // Modulation: 2 param sets for double stream (MSA + MLP), 1 for single
        self._doubleStreamModImg.wrappedValue = Flux2Modulation(hiddenSize: config.hiddenSize, modParamSets: 2)
        self._doubleStreamModTxt.wrappedValue = Flux2Modulation(hiddenSize: config.hiddenSize, modParamSets: 2)
        self._singleStreamMod.wrappedValue = Flux2Modulation(hiddenSize: config.hiddenSize, modParamSets: 1)

        self._transformerBlocks.wrappedValue = (0..<config.numLayers).map { _ in
            Flux2TransformerBlock(config: config)
        }
        self._singleTransformerBlocks.wrappedValue = (0..<config.numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(config: config)
        }

        self._normOut.wrappedValue = Flux2AdaLNContinuous(hiddenSize: config.hiddenSize, eps: config.eps)
        self._projOut.wrappedValue = Linear(config.hiddenSize, config.inChannels, bias: false)

        self.posEmbed = Flux2PosEmbed(theta: config.ropeTheta, axesDim: config.axesDimsRope)

        super.init()
    }

    /// iOS-optimized init with pre-built blocks
    /// Accepts pre-built block arrays to avoid post-init mutation
    init(
        config: Flux2TransformerConfiguration,
        transformerBlocks: [Flux2TransformerBlock],
        singleTransformerBlocks: [Flux2SingleTransformerBlock]
    ) {
        self.config = config

        // Placeholder Linear layers - will be replaced
        self._xEmbedder.wrappedValue = Linear(1, 1, bias: false)
        self._contextEmbedder.wrappedValue = Linear(1, 1, bias: false)
        self._timeGuidanceEmbed.wrappedValue = Flux2TimeGuidanceEmbed(hiddenSize: config.hiddenSize)

        self._doubleStreamModImg.wrappedValue = Flux2Modulation(hiddenSize: config.hiddenSize, modParamSets: 2)
        self._doubleStreamModTxt.wrappedValue = Flux2Modulation(hiddenSize: config.hiddenSize, modParamSets: 2)
        self._singleStreamMod.wrappedValue = Flux2Modulation(hiddenSize: config.hiddenSize, modParamSets: 1)

        // Use pre-built blocks
        self._transformerBlocks.wrappedValue = transformerBlocks
        self._singleTransformerBlocks.wrappedValue = singleTransformerBlocks

        self._normOut.wrappedValue = Flux2AdaLNContinuous(hiddenSize: config.hiddenSize, eps: config.eps)
        self._projOut.wrappedValue = Linear(1, 1, bias: false)

        self.posEmbed = Flux2PosEmbed(theta: config.ropeTheta, axesDim: config.axesDimsRope)

        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray,
        guidance: MLXArray? = nil
    ) -> MLXArray {
        // Conditional timestep scaling (matching mflux):
        // If timestep <= 1.0, multiply by 1000. Otherwise keep as-is.
        let timestepMax = timestep.max()
        let timestepScale = MLX.where(timestepMax .<= MLXArray(1.0), MLXArray(1000.0), MLXArray(1.0))
        let scaledTimestep = timestep * timestepScale
        let temb = timeGuidanceEmbed(timestep: scaledTimestep, guidance: guidance)

        // Compute modulation parameters
        let doubleModImg = doubleStreamModImg(temb)
        let doubleModTxt = doubleStreamModTxt(temb)
        let singleMod = singleStreamMod(temb)

        // Embed inputs
        var img = xEmbedder(hiddenStates)
        var txt = contextEmbedder(encoderHiddenStates)

        let concatRotary: (MLXArray, MLXArray)
        if Self.compileEnabled {
            let imgRotary = posEmbed(imgIds)
            let txtRotary = posEmbed(txtIds)
            concatRotary = (
                MLX.concatenated([txtRotary.0, imgRotary.0], axis: 0),
                MLX.concatenated([txtRotary.1, imgRotary.1], axis: 0)
            )
        } else {
            let imgIdsId = ObjectIdentifier(imgIds)
            let txtIdsId = ObjectIdentifier(txtIds)
            if let cachedRotary, cachedImgIdsId == imgIdsId, cachedTxtIdsId == txtIdsId {
                concatRotary = cachedRotary
            } else {
                let imgRotary = posEmbed(imgIds)
                let txtRotary = posEmbed(txtIds)
                let combined = (
                    MLX.concatenated([txtRotary.0, imgRotary.0], axis: 0),
                    MLX.concatenated([txtRotary.1, imgRotary.1], axis: 0)
                )
                cachedImgIdsId = imgIdsId
                cachedTxtIdsId = txtIdsId
                cachedRotary = combined
                concatRotary = combined
            }
        }

        // Extract modulation params for double blocks
        let modImgParams = (doubleModImg[0][0], doubleModImg[1][0])
        let modTxtParams = (doubleModTxt[0][0], doubleModTxt[1][0])

        // Joint transformer blocks
        for block in transformerBlocks {
            (txt, img) = block(
                hiddenStates: img,
                encoderHiddenStates: txt,
                modImg: modImgParams,
                modTxt: modTxtParams,
                rotaryEmb: concatRotary
            )
        }

        // Concatenate for single stream: [txt, img] (text first, then image)
        var combined = MLX.concatenated([txt, img], axis: 1)

        // Extract single stream modulation
        let singleModParams = singleMod[0][0]

        // Single transformer blocks
        for block in singleTransformerBlocks {
            combined = block(combined, mod: singleModParams, rotaryEmb: concatRotary)
        }

        // Extract image part (text is first, image is after)
        let numTxtTokens = txt.shape[1]
        img = combined[0..., numTxtTokens..., 0...]

        // Output projection
        img = normOut(img, cond: temb)
        img = projOut(img)

        return img
    }
}
