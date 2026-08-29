import Foundation
import MLX
import MLXFast
import MLXNN

private func requireConv2d(_ module: Any, context: String) -> Conv2d {
    guard let conv = module as? Conv2d else {
        preconditionFailure("Expected Conv2d in \(context).")
    }
    return conv
}

private func qwenAsymmetricDownsamplePad(_ x: MLXArray) -> MLXArray {
    padded(x, widths: [
        [0, 0],  // batch
        [0, 1],  // height: bottom-only zero pad before stride-2 conv
        [0, 1],  // width: right-only zero pad before stride-2 conv
        [0, 0]   // channels
    ])
}

// MARK: - Configuration

public struct VAE3DConfig {
    public let inChannels: Int
    public let outChannels: Int
    public let latentChannels: Int
    public let blockOutChannels: [Int]
    public let layersPerBlock: Int
    public let normNumGroups: Int
    public let scalingFactor: Float
    public let shiftFactor: Float
    public let temporalCompressionRatio: Int
    public let midBlockAddAttention: Bool

    public init(
        inChannels: Int = 3,
        outChannels: Int = 3,
        latentChannels: Int = 16,
        blockOutChannels: [Int] = [96, 192, 384, 384],
        layersPerBlock: Int = 2,
        normNumGroups: Int = 32,
        scalingFactor: Float = 0.476986,
        shiftFactor: Float = 0.0,
        temporalCompressionRatio: Int = 4,
        midBlockAddAttention: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.latentChannels = latentChannels
        self.blockOutChannels = blockOutChannels
        self.layersPerBlock = layersPerBlock
        self.normNumGroups = normNumGroups
        self.scalingFactor = scalingFactor
        self.shiftFactor = shiftFactor
        self.temporalCompressionRatio = temporalCompressionRatio
        self.midBlockAddAttention = midBlockAddAttention
    }

    public var spatialCompressionRatio: Int { 1 << (blockOutChannels.count - 1) }
}

// MARK: - 3D Spatial Norm

/// Qwen image RMS normalization over channels for [B, C, T, H, W].
private final class VAE3DSpatialNorm: Module {
    @ModuleInfo(key: "gamma") var gamma: MLXArray
    let eps: Float

    init(channels: Int, eps: Float = 1e-12) {
        self.eps = eps
        self._gamma.wrappedValue = MLXArray.ones([channels, 1, 1, 1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xFloat = x.asType(.float32)
        let variance = xFloat.square().mean(axis: 1, keepDims: true)
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * gamma.asType(.float32)).asType(dtype)
    }
}

// MARK: - CausalConv3d

/// 3D convolution with causal zero padding in the time dimension.
private final class CausalConv3d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let stride: (Int, Int, Int)
    let padding: (Int, Int, Int)
    let temporalPadding: Int
    let hasBias: Bool

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (1, 1, 1),
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.temporalPadding = padding.0 * 2
        self.hasBias = bias

        // Weight: [out_ch, in_ch, kT, kH, kW]
        let scale = sqrt(2.0 / Float(inputChannels * kernelSize.0 * kernelSize.1 * kernelSize.2))
        self._weight.wrappedValue = MLXRandom.normal([outputChannels, inputChannels, kernelSize.0, kernelSize.1, kernelSize.2]) * scale

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dim(2) == 1 && stride.0 == 1 {
            return callSingleFrame(x)
        }

        var hidden = x  // [B, C, T, H, W]

        // Qwen's causal conv pads missing history with zeros.
        if temporalPadding > 0 {
            let padFrames = MLX.zeros(
                [hidden.dim(0), hidden.dim(1), temporalPadding, hidden.dim(3), hidden.dim(4)],
                dtype: hidden.dtype
            )
            hidden = concatenated([padFrames, hidden], axis: 2)
        }

        // Spatial padding
        if padding.1 > 0 || padding.2 > 0 {
            hidden = padded(hidden, widths: [
                [0, 0],  // batch
                [0, 0],  // channel
                [0, 0],  // time (already padded)
                [padding.1, padding.1],  // height
                [padding.2, padding.2]   // width
            ])
        }

        // Conv3d: input [B, C, D, H, W], weight [O, I, kD, kH, kW]
        hidden = conv3d(hidden, weight, stride: [stride.0, stride.1, stride.2])

        if let b = bias {
            hidden = hidden + b.reshaped(1, -1, 1, 1, 1)
        }

        return hidden
    }

    private func callSingleFrame(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let temporalIndex = weight.dim(2) - 1

        var frame = x[0..., 0..., 0, 0..., 0...].transposed(0, 2, 3, 1)
        if padding.1 > 0 || padding.2 > 0 {
            frame = padded(frame, widths: [
                [0, 0],
                [padding.1, padding.1],
                [padding.2, padding.2],
                [0, 0]
            ])
        }

        let kernel = weight[0..., 0..., temporalIndex, 0..., 0...].transposed(0, 2, 3, 1)
        var hidden = MLX.conv2d(
            frame,
            kernel,
            stride: .init((stride.1, stride.2)),
            padding: 0
        )
        if let b = bias {
            hidden = hidden + b
        }
        let outHeight = hidden.dim(1)
        let outWidth = hidden.dim(2)
        return hidden.transposed(0, 3, 1, 2).reshaped(batch, hidden.dim(3), 1, outHeight, outWidth)
    }
}

// MARK: - 3D ResNet Block

private final class VAE3DResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: VAE3DSpatialNorm
    @ModuleInfo(key: "norm2") var norm2: VAE3DSpatialNorm
    @ModuleInfo(key: "conv1") var conv1: CausalConv3d
    @ModuleInfo(key: "conv2") var conv2: CausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: CausalConv3d?

    let hasShortcut: Bool

    init(inChannels: Int, outChannels: Int) {
        self._norm1.wrappedValue = VAE3DSpatialNorm(channels: inChannels)
        self._norm2.wrappedValue = VAE3DSpatialNorm(channels: outChannels)
        self._conv1.wrappedValue = CausalConv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )
        self._conv2.wrappedValue = CausalConv3d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )

        self.hasShortcut = inChannels != outChannels
        if hasShortcut {
            self._convShortcut.wrappedValue = CausalConv3d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: (1, 1, 1),
                padding: (0, 0, 0)
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = silu(norm1(x))
        hidden = conv1(hidden)
        hidden = silu(norm2(hidden))
        hidden = conv2(hidden)

        let residual = hasShortcut ? convShortcut!(x) : x
        return residual + hidden
    }
}

// MARK: - 3D Attention (Spatial)

/// Self-attention over spatial dimensions (T*H*W flattened)
/// Uses channels-last format internally to match MLX Conv2d expectations
private final class VAE3DAttention: Module {
    @ModuleInfo(key: "norm") var norm: VAE3DSpatialNorm2D
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d  // Fused QKV projection (1x1 conv)
    @ModuleInfo(key: "proj") var proj: Conv2d     // Output projection (1x1 conv)

    let channels: Int
    let numHeads: Int
    let headDim: Int

    init(channels: Int, numHeads: Int = 1) {
        self.channels = channels
        self.numHeads = numHeads
        self.headDim = channels / numHeads

        self._norm.wrappedValue = VAE3DSpatialNorm2D(channels: channels)
        self._toQKV.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels * 3,
            kernelSize: 1
        )
        self._proj.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))

        // Reshape [B, C, T, H, W] -> [B*T, H, W, C] for channels-last 2D operations
        var hidden = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h, w, c)

        // Normalize (channels-last)
        hidden = norm(hidden)

        // QKV projection: [B*T, H, W, C] -> [B*T, H, W, 3*C]
        let qkv = toQKV(hidden)

        // Split and reshape for attention
        // [B*T, H, W, 3*C] -> [B*T, H*W, 3, numHeads, headDim]
        let qkvReshaped = qkv.reshaped(b * t, h * w, 3, numHeads, headDim)

        // Extract Q, K, V and arrange for attention: [B*T, numHeads, H*W, headDim]
        let q = qkvReshaped[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        let k = qkvReshaped[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkvReshaped[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        let scale = 1.0 / sqrt(Float(headDim))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )

        // Reshape back: [B*T, numHeads, H*W, headDim] -> [B*T, H, W, C]
        hidden = attn.transposed(0, 2, 1, 3).reshaped(b * t, h, w, c)

        // Output projection
        hidden = proj(hidden)

        // Reshape back to 5D: [B*T, H, W, C] -> [B, C, T, H, W]
        hidden = hidden.reshaped(b, t, h, w, c).transposed(0, 4, 1, 2, 3)
        return x + hidden
    }
}

/// Qwen image RMS normalization for attention in channels-last form [B, H, W, C].
private final class VAE3DSpatialNorm2D: Module {
    @ModuleInfo(key: "gamma") var gamma: MLXArray
    let eps: Float

    init(channels: Int, eps: Float = 1e-12) {
        self.eps = eps
        self._gamma.wrappedValue = MLXArray.ones([channels, 1, 1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xFloat = x.asType(.float32)
        let variance = xFloat.square().mean(axis: -1, keepDims: true)
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * gamma.asType(.float32).transposed(2, 1, 0)).asType(dtype)
    }
}

// MARK: - Mid Block

private final class VAE3DMidBlock: Module {
    @ModuleInfo(key: "attentions") var attentions: [VAE3DAttention]
    @ModuleInfo(key: "resnets") var resnets: [VAE3DResnetBlock]

    init(channels: Int, addAttention: Bool) {
        self._resnets.wrappedValue = [
            VAE3DResnetBlock(inChannels: channels, outChannels: channels),
            VAE3DResnetBlock(inChannels: channels, outChannels: channels)
        ]
        self._attentions.wrappedValue = addAttention ? [VAE3DAttention(channels: channels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = resnets[0](x)
        if !attentions.isEmpty {
            hidden = attentions[0](hidden)
        }
        hidden = resnets[1](hidden)
        return hidden
    }
}

// MARK: - Upsampler

/// Spatial and temporal upsampling
private final class VAE3DUpsampler: Module {
    @ModuleInfo(key: "resample") var resample: [Any]  // [Upsample2D, Conv2d]
    @ModuleInfo(key: "time_conv") var timeConv: CausalConv3d?

    let inChannels: Int
    let outChannels: Int
    let hasTimeConv: Bool
    let temporalScale: Int

    init(inChannels: Int, outChannels: Int, temporalScale: Int = 1) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.hasTimeConv = temporalScale > 1
        self.temporalScale = temporalScale

        // resample.1 is the conv (resample.0 is nn.Upsample in PyTorch, we do it manually)
        let conv = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._resample.wrappedValue = [conv]

        if hasTimeConv {
            // time_conv upsamples temporally: [B, C, T, H, W] -> [B, C*2, T, H, W] then pixel shuffle
            self._timeConv.wrappedValue = CausalConv3d(
                inputChannels: inChannels,
                outputChannels: inChannels * temporalScale,
                kernelSize: (3, 1, 1),
                padding: (1, 0, 0)
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x  // [B, C, T, H, W]
        let (b, _, t, h, w) = (hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3), hidden.dim(4))

        // Temporal upsample first (if needed AND T > 1)
        // Skip temporal upsampling for single-image (T=1) inputs
        if hasTimeConv, let tc = timeConv, t > 1 {
            // time_conv: [B, C, T, H, W] -> [B, C*scale, T, H, W]
            hidden = tc(hidden)
            // Pixel shuffle in time: [B, C*scale, T, H, W] -> [B, C, T*scale, H, W]
            let currentT = hidden.dim(2)
            let newC = hidden.dim(1) / temporalScale
            hidden = hidden.reshaped(b, newC, temporalScale, currentT, h, w)
            hidden = hidden.transposed(0, 1, 3, 2, 4, 5)  // [B, C, T, scale, H, W]
            hidden = hidden.reshaped(b, newC, currentT * temporalScale, h, w)
        }

        // Spatial upsample: nearest neighbor 2x
        let newT = hidden.dim(2)
        let currentC = hidden.dim(1)
        let currentH = hidden.dim(3)
        let currentW = hidden.dim(4)
        let newH = currentH * 2
        let newW = currentW * 2

        // Reshape to [B*T, H, W, C] for channels-last 2D operations
        hidden = hidden.transposed(0, 2, 3, 4, 1).reshaped(b * newT, currentH, currentW, currentC)

        // Nearest neighbor upsample (channels-last)
        hidden = upsampleNearest2xChannelsLast(hidden)

        // Apply conv (resample.1) - Conv2d expects [B, H, W, C]
        let conv = requireConv2d(resample[0], context: "VAE3DUpsampler.resample[0]")
        hidden = conv(hidden)

        // Reshape back to [B, C_out, T, H*2, W*2]
        let outC = hidden.dim(3)  // channels are last now
        hidden = hidden.reshaped(b, newT, newH, newW, outC).transposed(0, 4, 1, 2, 3)

        return hidden
    }

    private func upsampleNearest2xChannelsLast(_ x: MLXArray) -> MLXArray {
        // x: [B, H, W, C] - channels-last
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let c = x.dim(3)

        // Expand and interleave in spatial dimensions
        var expanded = x.expandedDimensions(axes: [2, 4])  // [B, H, 1, W, 1, C]
        expanded = tiled(expanded, repetitions: [1, 1, 2, 1, 2, 1])  // [B, H, 2, W, 2, C]
        return expanded.reshaped(b, h * 2, w * 2, c)
    }
}

// MARK: - Up Block

private final class VAE3DUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAE3DResnetBlock]
    @ModuleInfo(key: "upsamplers") var upsamplers: [VAE3DUpsampler]

    init(
        inChannels: Int,
        outChannels: Int,
        numLayers: Int,
        hasUpsampler: Bool,
        temporalScale: Int = 1
    ) {
        var blocks: [VAE3DResnetBlock] = []
        for i in 0..<numLayers {
            let blockIn = i == 0 ? inChannels : outChannels
            blocks.append(VAE3DResnetBlock(inChannels: blockIn, outChannels: outChannels))
        }
        self._resnets.wrappedValue = blocks

        if hasUpsampler {
            self._upsamplers.wrappedValue = [
                VAE3DUpsampler(inChannels: outChannels, outChannels: outChannels / 2, temporalScale: temporalScale)
            ]
        } else {
            self._upsamplers.wrappedValue = []
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for resnet in resnets {
            hidden = resnet(hidden)
        }
        for upsampler in upsamplers {
            hidden = upsampler(hidden)
        }
        return hidden
    }
}

// MARK: - Decoder

private final class VAE3DDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: VAE3DMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [VAE3DUpBlock]
    @ModuleInfo(key: "norm_out") var normOut: VAE3DSpatialNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    init(config: VAE3DConfig) {
        let channels = config.blockOutChannels  // [96, 192, 384, 384]
        let reversed = Array(channels.reversed())  // [384, 384, 192, 96]

        self._convIn.wrappedValue = CausalConv3d(
            inputChannels: config.latentChannels,
            outputChannels: reversed[0],
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )

        self._midBlock.wrappedValue = VAE3DMidBlock(
            channels: reversed[0],
            addAttention: config.midBlockAddAttention
        )

        // Build up blocks
        var blocks: [VAE3DUpBlock] = []
        let numBlocks = channels.count

        // Determine which blocks have temporal upsampling (based on temporalCompressionRatio)
        // ratio=4 means 2 temporal upsamplings (2^2=4), at blocks 0 and 1
        let numTemporalUps = Int(log2(Double(config.temporalCompressionRatio)))

        for i in 0..<numBlocks {
            let isLast = i == numBlocks - 1
            let inCh = i == 0 ? reversed[0] : reversed[i - 1] / 2  // After upsampler halves channels
            let outCh = reversed[i]
            let hasUpsampler = !isLast
            let temporalScale = (i < numTemporalUps && hasUpsampler) ? 2 : 1

            blocks.append(VAE3DUpBlock(
                inChannels: inCh,
                outChannels: outCh,
                numLayers: config.layersPerBlock + 1,
                hasUpsampler: hasUpsampler,
                temporalScale: temporalScale
            ))
        }
        self._upBlocks.wrappedValue = blocks

        self._normOut.wrappedValue = VAE3DSpatialNorm(channels: reversed.last!)
        self._convOut.wrappedValue = CausalConv3d(
            inputChannels: reversed.last!,
            outputChannels: config.outChannels,
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )
        super.init()
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        var hidden = convIn(latents)
        hidden = midBlock(hidden)
        for block in upBlocks {
            hidden = block(hidden)
        }
        let normalized = normOut(hidden)
        hidden = silu(normalized)
        hidden = convOut(hidden)
        return hidden
    }
}

// MARK: - Downsampler

/// Spatial and temporal downsampling for encoder
private final class VAE3DDownsampler: Module {
    @ModuleInfo(key: "resample") var resample: [Any]  // [AvgPool, Conv2d]
    @ModuleInfo(key: "time_conv") var timeConv: CausalConv3d?

    let inChannels: Int
    let hasTimeConv: Bool

    init(inChannels: Int, temporalDownsample: Bool) {
        self.inChannels = inChannels
        self.hasTimeConv = temporalDownsample

        // resample.1 is the conv (resample.0 is nn.AvgPool in PyTorch, we do stride conv)
        let conv = Conv2d(inputChannels: inChannels, outputChannels: inChannels, kernelSize: 3, stride: 2)
        self._resample.wrappedValue = [conv]

        if hasTimeConv {
            // time_conv: temporal stride 2 via 3D conv
            self._timeConv.wrappedValue = CausalConv3d(
                inputChannels: inChannels,
                outputChannels: inChannels,
                kernelSize: (3, 1, 1),
                stride: (2, 1, 1),
                padding: (0, 0, 0)
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x  // [B, C, T, H, W]
        let (b, c, t, _, _) = (hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3), hidden.dim(4))

        // Temporal downsample first (if needed AND T > 1)
        if hasTimeConv, let tc = timeConv, t > 1 {
            hidden = tc(hidden)
        }

        // Spatial downsample: stride-2 conv
        let newT = hidden.dim(2)
        let currentH = hidden.dim(3)
        let currentW = hidden.dim(4)

        // Reshape to [B*T, H, W, C] for channels-last 2D operations
        hidden = hidden.transposed(0, 2, 3, 4, 1).reshaped(b * newT, currentH, currentW, c)
        hidden = qwenAsymmetricDownsamplePad(hidden)

        // Apply conv (resample.1) - Conv2d expects [B, H, W, C]
        let conv = requireConv2d(resample[0], context: "VAE3DDownsampler.resample[0]")
        hidden = conv(hidden)

        // Reshape back to [B, C, T, H/2, W/2]
        let newH = hidden.dim(1)
        let newW = hidden.dim(2)
        hidden = hidden.reshaped(b, newT, newH, newW, c).transposed(0, 4, 1, 2, 3)

        return hidden
    }
}

// MARK: - Encoder Down Block (flat structure to match weights)

/// Individual resnet block for encoder (can be part of any down_blocks.X)
private final class VAE3DEncoderResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: VAE3DSpatialNorm
    @ModuleInfo(key: "norm2") var norm2: VAE3DSpatialNorm
    @ModuleInfo(key: "conv1") var conv1: CausalConv3d
    @ModuleInfo(key: "conv2") var conv2: CausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: CausalConv3d?

    let hasShortcut: Bool

    init(inChannels: Int, outChannels: Int) {
        self._norm1.wrappedValue = VAE3DSpatialNorm(channels: inChannels)
        self._norm2.wrappedValue = VAE3DSpatialNorm(channels: outChannels)
        self._conv1.wrappedValue = CausalConv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )
        self._conv2.wrappedValue = CausalConv3d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )

        self.hasShortcut = inChannels != outChannels
        if hasShortcut {
            self._convShortcut.wrappedValue = CausalConv3d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: (1, 1, 1),
                padding: (0, 0, 0)
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = silu(norm1(x))
        hidden = conv1(hidden)
        hidden = silu(norm2(hidden))
        hidden = conv2(hidden)

        let residual = hasShortcut ? convShortcut!(x) : x
        return residual + hidden
    }
}

/// Downsampler block for encoder (used at blocks 2, 5, 8)
private final class VAE3DEncoderDownsampleBlock: Module {
    @ModuleInfo(key: "resample") var resample: [Any]
    @ModuleInfo(key: "time_conv") var timeConv: CausalConv3d?

    let channels: Int
    let hasTimeConv: Bool

    init(channels: Int, temporalDownsample: Bool) {
        self.channels = channels
        self.hasTimeConv = temporalDownsample

        let conv = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2)
        self._resample.wrappedValue = [conv]

        if temporalDownsample {
            self._timeConv.wrappedValue = CausalConv3d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: (3, 1, 1),
                stride: (2, 1, 1),
                padding: (0, 0, 0)
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        let (b, c, t, _, _) = (hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3), hidden.dim(4))

        // Temporal downsample first (if needed AND T > 1)
        if hasTimeConv, let tc = timeConv, t > 1 {
            hidden = tc(hidden)
        }

        let newT = hidden.dim(2)
        let currentH = hidden.dim(3)
        let currentW = hidden.dim(4)

        // Reshape to channels-last for Conv2d
        hidden = hidden.transposed(0, 2, 3, 4, 1).reshaped(b * newT, currentH, currentW, c)
        hidden = qwenAsymmetricDownsamplePad(hidden)

        let conv = requireConv2d(resample[0], context: "VAE3DEncoderDownsampleBlock.resample[0]")
        hidden = conv(hidden)

        let newH = hidden.dim(1)
        let newW = hidden.dim(2)
        hidden = hidden.reshaped(b, newT, newH, newW, c).transposed(0, 4, 1, 2, 3)

        return hidden
    }
}

// MARK: - Encoder

private final class VAE3DEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [Module]  // Mix of resnet and downsample blocks
    @ModuleInfo(key: "mid_block") var midBlock: VAE3DMidBlock
    @ModuleInfo(key: "norm_out") var normOut: VAE3DSpatialNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    init(config: VAE3DConfig) {
        let channels = config.blockOutChannels  // [96, 192, 384, 384]

        self._convIn.wrappedValue = CausalConv3d(
            inputChannels: config.inChannels,
            outputChannels: channels[0],
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )

        // Build down blocks based on Qwen VAE structure:
        // 0, 1: 96 ch resnets
        // 2: spatial downsample
        // 3, 4: 192 ch resnets (3 has shortcut)
        // 5: spatial + temporal downsample
        // 6, 7: 384 ch resnets (6 has shortcut)
        // 8: spatial + temporal downsample
        // 9, 10: 384 ch resnets
        var blocks: [Module] = []

        // Block 0: 96 → 96
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[0], outChannels: channels[0]))
        // Block 1: 96 → 96
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[0], outChannels: channels[0]))
        // Block 2: spatial downsample (no temporal)
        blocks.append(VAE3DEncoderDownsampleBlock(channels: channels[0], temporalDownsample: false))
        // Block 3: 96 → 192 (has shortcut)
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[0], outChannels: channels[1]))
        // Block 4: 192 → 192
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[1], outChannels: channels[1]))
        // Block 5: spatial + temporal downsample
        blocks.append(VAE3DEncoderDownsampleBlock(channels: channels[1], temporalDownsample: true))
        // Block 6: 192 → 384 (has shortcut)
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[1], outChannels: channels[2]))
        // Block 7: 384 → 384
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[2], outChannels: channels[2]))
        // Block 8: spatial + temporal downsample
        blocks.append(VAE3DEncoderDownsampleBlock(channels: channels[2], temporalDownsample: true))
        // Block 9: 384 → 384
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[2], outChannels: channels[2]))
        // Block 10: 384 → 384
        blocks.append(VAE3DEncoderResnetBlock(inChannels: channels[2], outChannels: channels[2]))

        self._downBlocks.wrappedValue = blocks

        self._midBlock.wrappedValue = VAE3DMidBlock(
            channels: channels.last!,
            addAttention: config.midBlockAddAttention
        )

        self._normOut.wrappedValue = VAE3DSpatialNorm(channels: channels.last!)

        // conv_out: 384 → 32 (latent_channels * 2 for mean + logvar, but we only use mean)
        self._convOut.wrappedValue = CausalConv3d(
            inputChannels: channels.last!,
            outputChannels: config.latentChannels * 2,
            kernelSize: (3, 3, 3),
            padding: (1, 1, 1)
        )
        super.init()
    }

    func callAsFunction(_ images: MLXArray) -> MLXArray {
        var hidden = convIn(images)

        for block in downBlocks {
            if let resnet = block as? VAE3DEncoderResnetBlock {
                hidden = resnet(hidden)
            } else if let downsample = block as? VAE3DEncoderDownsampleBlock {
                hidden = downsample(hidden)
            }
        }

        hidden = midBlock(hidden)
        hidden = silu(normOut(hidden))
        hidden = convOut(hidden)

        return hidden
    }
}

// MARK: - AutoencoderKL3D

public final class AutoencoderKL3D: Module {
    public let config: VAE3DConfig
    @ModuleInfo(key: "encoder") private var encoder: VAE3DEncoder
    @ModuleInfo(key: "quantConv") private var quantConv: CausalConv3d
    @ModuleInfo(key: "postQuantConv") private var postQuantConv: CausalConv3d
    @ModuleInfo(key: "decoder") private var decoder: VAE3DDecoder

    public init(config: VAE3DConfig = .init()) {
        self.config = config
        self._encoder.wrappedValue = VAE3DEncoder(config: config)
        self._quantConv.wrappedValue = CausalConv3d(
            inputChannels: config.latentChannels * 2,
            outputChannels: config.latentChannels * 2,
            kernelSize: (1, 1, 1),
            padding: (0, 0, 0)
        )
        self._postQuantConv.wrappedValue = CausalConv3d(
            inputChannels: config.latentChannels,
            outputChannels: config.latentChannels,
            kernelSize: (1, 1, 1),
            padding: (0, 0, 0)
        )
        self._decoder.wrappedValue = VAE3DDecoder(config: config)
        super.init()
    }

    /// Encode images to latents
    /// - Parameter images: [B, C, T, H, W] in [-1, 1] range
    /// - Returns: [B, latent_channels, T/temporalScale, H/spatialScale, W/spatialScale] latent representation
    public func encode(_ images: MLXArray) -> MLXArray {
        let mean = encodeUnscaled(images)
        // Scale latents
        var scaled = mean * MLXArray(config.scalingFactor)
        if config.shiftFactor != 0 {
            scaled = scaled - MLXArray(config.shiftFactor)
        }
        return scaled
    }

    /// Encode to the raw posterior mode without scalar wrapper normalization.
    public func encodeUnscaled(_ images: MLXArray) -> MLXArray {
        let encoded = quantConv(encoder(images))
        return encoded[0..., 0..<config.latentChannels, 0..., 0..., 0...]
    }

    /// Encode single image (convenience)
    /// - Parameter images: [B, C, H, W] single-frame images in [-1, 1] range
    /// - Returns: [B, latent_channels, H/8, W/8] latent representation
    public func encodeImage(_ images: MLXArray) -> MLXArray {
        // Add temporal dimension: [B, C, H, W] -> [B, C, 1, H, W]
        let images5d = images.expandedDimensions(axis: 2)
        let encoded = encode(images5d)
        // Remove temporal dimension: [B, C, T, H, W] -> [B, C, H, W]
        return encoded.squeezed(axis: 2)
    }

    /// Encode one image to the raw posterior mode.
    public func encodeImageUnscaled(_ images: MLXArray) -> MLXArray {
        encodeUnscaled(images.expandedDimensions(axis: 2)).squeezed(axis: 2)
    }

    /// Decode latents to images/video
    /// - Parameter latents: [B, C, T, H, W] in latent space
    /// - Returns: [B, 3, T*temporalScale, H*spatialScale, W*spatialScale] decoded output
    public func decode(_ latents: MLXArray) -> MLXArray {
        // Un-scale latents
        var x = latents / MLXArray(config.scalingFactor)
        if config.shiftFactor != 0 {
            x = x + MLXArray(config.shiftFactor)
        }
        return decodeUnscaled(x)
    }

    /// Decode raw checkpoint latents without scalar wrapper normalization.
    public func decodeUnscaled(_ latents: MLXArray) -> MLXArray {
        decoder(postQuantConv(latents))
    }

    /// Decode single image (convenience)
    /// - Parameter latents: [B, C, H, W] single-frame latents
    /// - Returns: [B, 3, H*8, W*8] decoded image
    public func decodeImage(_ latents: MLXArray) -> MLXArray {
        // Add temporal dimension: [B, C, H, W] -> [B, C, 1, H, W]
        let latents5d = latents.expandedDimensions(axis: 2)
        let decoded = decode(latents5d)
        // Remove temporal dimension: [B, 3, T, H, W] -> [B, 3, H, W]
        return decoded.squeezed(axis: 2)
    }

    /// Decode one raw latent image without scalar wrapper normalization.
    public func decodeImageUnscaled(_ latents: MLXArray) -> MLXArray {
        decodeUnscaled(latents.expandedDimensions(axis: 2)).squeezed(axis: 2)
    }
}

// MARK: - Conv3d helper

/// 3D convolution (not in MLXNN but available in MLX)
private func conv3d(
    _ x: MLXArray,
    _ weight: MLXArray,
    stride: [Int] = [1, 1, 1],
    groups: Int = 1
) -> MLXArray {
    // MLX convGeneral expects:
    //   input: [N, D, H, W, C_in] (channels last)
    //   weight: [C_out, kD, kH, kW, C_in] (output channels first, input channels last)
    // Our input: [B, C, D, H, W], weight: [O, I, kD, kH, kW] (PyTorch format)

    // Transpose input: [B, C, D, H, W] -> [B, D, H, W, C]
    let xT = x.transposed(0, 2, 3, 4, 1)

    // Transpose weight: [O, I, kD, kH, kW] -> [O, kD, kH, kW, I]
    let wT = weight.transposed(0, 2, 3, 4, 1)

    // MLX conv3d - padding is 0 (we handle padding manually in CausalConv3d)
    let result = MLX.conv3d(
        xT,
        wT,
        stride: .init(stride),
        padding: 0,
        groups: groups
    )

    // Transpose output: [B, D', H', W', C'] -> [B, C', D', H', W']
    return result.transposed(0, 4, 1, 2, 3)
}
