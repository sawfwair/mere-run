import Foundation
import MLX
import MLXFast
import MLXNN

struct SAM31VisionContext: @unchecked Sendable {
    let featurePyramid: SAM31VisionFeaturePyramid
    let src: MLXArray
    let posFlat: MLXArray
    let spatialShape: (height: Int, width: Int)

    var detFeatures: [MLXArray] { featurePyramid.detFeatures }
    var interactiveFeatures: [MLXArray] { featurePyramid.interactiveFeatures }
    var propagationFeatures: [MLXArray] { featurePyramid.propagationFeatures }
}

struct SAM31DetectorOutput: @unchecked Sendable {
    let predLogits: MLXArray
    let predBoxes: MLXArray
    let predMasks: MLXArray
    let presenceLogits: MLXArray
}

struct SAM31VisionFeaturePyramid: @unchecked Sendable {
    let detFeatures: [MLXArray]
    let interactiveFeatures: [MLXArray]
    let propagationFeatures: [MLXArray]
}

private func sam31InverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let clipped = MLX.clip(x, min: eps, max: 1 - eps)
    return MLX.log(clipped / (1 - clipped))
}

private func sam31ZeroPadNHWC(_ x: MLXArray, padHeight: Int, padWidth: Int) -> MLXArray {
    var out = x
    if padHeight > 0 {
        let pad = MLX.zeros([x.dim(0), padHeight, x.dim(2), x.dim(3)], dtype: x.dtype)
        out = MLX.concatenated([out, pad], axis: 1)
    }
    if padWidth > 0 {
        let pad = MLX.zeros([out.dim(0), out.dim(1), padWidth, out.dim(3)], dtype: x.dtype)
        out = MLX.concatenated([out, pad], axis: 2)
    }
    return out
}

private func sam31ComputeAxialCis(
    dim: Int,
    width: Int,
    height: Int,
    theta: Float = 10_000
) -> (cos: MLXArray, sin: MLXArray) {
    let freqs = MLXArray(
        stride(from: 0, to: dim, by: 4).map { index in
            Float(1.0 / Foundation.pow(Double(theta), Double(index) / Double(dim)))
        }
    )
    let positions = Array(0..<(width * height))
    let xPositions = MLXArray(positions.map { Float($0 % width) }).expandedDimensions(axis: 1)
    let yPositions = MLXArray(positions.map { Float($0 / width) }).expandedDimensions(axis: 1)
    let freqsX = xPositions * freqs
    let freqsY = yPositions * freqs
    let joined = MLX.concatenated([freqsX, freqsY], axis: -1)
    let doubled = MLX.stacked([joined, joined], axis: -1).reshaped(joined.dim(0), joined.dim(1) * 2)
    return (MLX.cos(doubled), MLX.sin(doubled))
}

private func sam31RotatePairwise(_ x: MLXArray) -> MLXArray {
    let reshaped = x.reshaped(x.dim(0), x.dim(1), x.dim(2), x.dim(3) / 2, 2)
    let first = reshaped[0..., 0..., 0..., 0..., 0]
    let second = reshaped[0..., 0..., 0..., 0..., 1]
    return MLX.stacked([-second, first], axis: -1).reshaped(x.shape)
}

private func sam31ApplyRotary(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cosExpanded = cos.expandedDimensions(axes: [0, 1])
    let sinExpanded = sin.expandedDimensions(axes: [0, 1])
    let qOut = q * cosExpanded + sam31RotatePairwise(q) * sinExpanded
    let kOut = k * cosExpanded + sam31RotatePairwise(k) * sinExpanded
    return (qOut, kOut)
}

private func sam31PositionTiling(_ pos: MLXArray, targetHeight: Int, targetWidth: Int) -> MLXArray {
    let pretrainSize = Int(Double(pos.dim(1)).squareRoot())
    if pretrainSize == targetHeight && pretrainSize == targetWidth {
        return pos
    }

    let hiddenSize = pos.dim(2)
    let reshaped = pos.reshaped(1, pretrainSize, pretrainSize, hiddenSize)
    let repeatHeight = targetHeight / pretrainSize + 1
    let repeatWidth = targetWidth / pretrainSize + 1
    let tiled = MLX.tiled(reshaped, repetitions: [1, repeatHeight, repeatWidth, 1])
    let cropped = tiled[0..., 0..<targetHeight, 0..<targetWidth, 0...]
    return cropped.reshaped(1, targetHeight * targetWidth, hiddenSize)
}

private func sam31WindowPartition(_ x: MLXArray, windowSize: Int) -> (windows: MLXArray, paddedHW: (Int, Int)) {
    let batch = x.dim(0)
    let height = x.dim(1)
    let width = x.dim(2)
    let channels = x.dim(3)

    let padHeight = (windowSize - height % windowSize) % windowSize
    let padWidth = (windowSize - width % windowSize) % windowSize
    let padded = sam31ZeroPadNHWC(x, padHeight: padHeight, padWidth: padWidth)
    let paddedHeight = padded.dim(1)
    let paddedWidth = padded.dim(2)
    let nH = paddedHeight / windowSize
    let nW = paddedWidth / windowSize

    let windows = padded
        .reshaped(batch, nH, windowSize, nW, windowSize, channels)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(batch * nH * nW, windowSize, windowSize, channels)
    return (windows, (paddedHeight, paddedWidth))
}

private func sam31WindowUnpartition(
    _ x: MLXArray,
    windowSize: Int,
    paddedHW: (Int, Int),
    originalHW: (Int, Int)
) -> MLXArray {
    let paddedHeight = paddedHW.0
    let paddedWidth = paddedHW.1
    let height = originalHW.0
    let width = originalHW.1
    let nH = paddedHeight / windowSize
    let nW = paddedWidth / windowSize
    let batch = x.dim(0) / (nH * nW)
    let channels = x.dim(3)

    let restored = x
        .reshaped(batch, nH, nW, windowSize, windowSize, channels)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(batch, paddedHeight, paddedWidth, channels)

    return restored[0..., 0..<height, 0..<width, 0...]
}

final class SAM31PositionEmbeddingSine: Module {
    let numPosFeats: Int
    let temperature: Float
    let scale: Float

    init(numPosFeats: Int = 128, temperature: Float = 10_000, scale: Float = 2 * .pi) {
        self.numPosFeats = numPosFeats
        self.temperature = temperature
        self.scale = scale
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)

        let yBase = MLXArray((0..<height).map { Float($0 + 1) })
        let xBase = MLXArray((0..<width).map { Float($0 + 1) })
        let yEmbed = MLX.broadcast(yBase.reshaped(1, height, 1), to: [batch, height, width]) / Float(height) * scale
        let xEmbed = MLX.broadcast(xBase.reshaped(1, 1, width), to: [batch, height, width]) / Float(width) * scale

        let dimT = MLXArray((0..<numPosFeats).map { index in
            Float(Foundation.pow(Double(temperature), Double(2 * (index / 2)) / Double(numPosFeats)))
        })

        let posX = (xEmbed.expandedDimensions(axis: 3) / dimT).reshaped(batch, height, width, numPosFeats / 2, 2)
        let posY = (yEmbed.expandedDimensions(axis: 3) / dimT).reshaped(batch, height, width, numPosFeats / 2, 2)

        let posXEven = MLX.sin(posX[0..., 0..., 0..., 0..., 0])
        let posXOdd = MLX.cos(posX[0..., 0..., 0..., 0..., 1])
        let posYEven = MLX.sin(posY[0..., 0..., 0..., 0..., 0])
        let posYOdd = MLX.cos(posY[0..., 0..., 0..., 0..., 1])

        let posXStacked = MLX.stacked([posXEven, posXOdd], axis: -1).reshaped(batch, height, width, numPosFeats)
        let posYStacked = MLX.stacked([posYEven, posYOdd], axis: -1).reshaped(batch, height, width, numPosFeats)
        return MLX.concatenated([posYStacked, posXStacked], axis: -1)
    }
}

final class SAM31PatchProjection: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(config: SAM31ViTConfig) {
        self._projection.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            padding: IntOrPair(0),
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(x)
    }
}

final class SAM31PatchEmbeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: SAM31PatchProjection
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    init(config: SAM31ViTConfig) {
        let patchCount = Int(Foundation.pow(Double(config.pretrainImageSize / config.patchSize), 2))
        self._patchEmbeddings.wrappedValue = SAM31PatchProjection(config: config)
        self._positionEmbeddings.wrappedValue = MLX.zeros([1, patchCount, config.hiddenSize], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = patchEmbeddings(x)
        let batch = projected.dim(0)
        let height = projected.dim(1)
        let width = projected.dim(2)
        let channels = projected.dim(3)
        return projected.reshaped(batch, height * width, channels)
    }
}

final class SAM31ViTAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(config: SAM31ViTConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        self._oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray? = nil, sin: MLXArray? = nil) -> MLXArray {
        let inputShape = x.shape
        let flattened: MLXArray
        if x.ndim == 4 {
            flattened = x.reshaped(x.dim(0), x.dim(1) * x.dim(2), x.dim(3))
        } else {
            flattened = x
        }

        let batch = flattened.dim(0)
        let sequenceLength = flattened.dim(1)
        let hiddenSize = flattened.dim(2)

        var q = qProj(flattened).reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(flattened).reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(flattened).reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

        if let cos, let sin {
            (q, k) = sam31ApplyRotary(q: q, k: k, cos: cos, sin: sin)
        }

        // SAM's shipped ViT uses 64-wide heads. mlx-swift 0.31.5 dispatches
        // qLen > 8 with headDim 64 to its tiled full-attention Metal kernel,
        // avoiding the old graph's two materialized [B,H,Q,K] matrices.
        var out: MLXArray
        if FusedAttentionPolicy.enabled {
            out = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none
            )
        } else {
            var attention = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
            attention = softmax(attention, axis: -1)
            out = MLX.matmul(attention, v)
        }
        out = out.transposed(0, 2, 1, 3).reshaped(batch, sequenceLength, hiddenSize)
        let projected = oProj(out)

        if x.ndim == 4 {
            return projected.reshaped(inputShape)
        }
        return projected
    }
}

final class SAM31ViTMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(config: SAM31ViTConfig) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class SAM31ViTBlock: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: SAM31ViTAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAM31ViTMLP

    let windowSize: Int
    let isGlobal: Bool

    init(config: SAM31ViTConfig, isGlobal: Bool) {
        self.windowSize = isGlobal ? 0 : config.windowSize
        self.isGlobal = isGlobal
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = SAM31ViTAttention(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = SAM31ViTMLP(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var hidden = layerNorm1(x)
        if windowSize > 0 {
            let partition = sam31WindowPartition(hidden, windowSize: windowSize)
            let attended = attention(partition.windows, cos: cos, sin: sin)
            hidden = sam31WindowUnpartition(
                attended,
                windowSize: windowSize,
                paddedHW: partition.paddedHW,
                originalHW: (x.dim(1), x.dim(2))
            )
        } else {
            hidden = attention(hidden, cos: cos, sin: sin)
        }

        let residual = x + hidden
        return residual + mlp(layerNorm2(residual))
    }
}

final class SAM31ViTBackbone: Module {
    let config: SAM31ViTConfig
    let featSize: Int
    let globalCos: MLXArray
    let globalSin: MLXArray
    let windowCos: MLXArray
    let windowSin: MLXArray

    @ModuleInfo(key: "embeddings") var embeddings: SAM31PatchEmbeddings
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [SAM31ViTBlock]

    init(config: SAM31ViTConfig) {
        self.config = config
        self.featSize = config.imageSize / config.patchSize
        let global = sam31ComputeAxialCis(
            dim: config.hiddenSize / config.numAttentionHeads,
            width: featSize,
            height: featSize,
            theta: config.ropeTheta
        )
        self.globalCos = global.cos
        self.globalSin = global.sin
        let window = sam31ComputeAxialCis(
            dim: config.hiddenSize / config.numAttentionHeads,
            width: config.windowSize,
            height: config.windowSize,
            theta: config.ropeTheta
        )
        self.windowCos = window.cos
        self.windowSin = window.sin
        self._embeddings.wrappedValue = SAM31PatchEmbeddings(config: config)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        let globalSet = Set(config.globalAttnIndexes)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { index in
            SAM31ViTBlock(config: config, isGlobal: globalSet.contains(index))
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(1) / config.patchSize
        let width = x.dim(2) / config.patchSize
        let embedded = embeddings(x)
        let pos = sam31PositionTiling(embeddings.positionEmbeddings, targetHeight: height, targetWidth: width)
        var hidden = layerNorm((embedded + pos).reshaped(batch, height, width, config.hiddenSize))

        let globalPair: (MLXArray, MLXArray)
        if height != featSize || width != featSize {
            globalPair = sam31ComputeAxialCis(
                dim: config.hiddenSize / config.numAttentionHeads,
                width: width,
                height: height,
                theta: config.ropeTheta
            )
        } else {
            globalPair = (globalCos, globalSin)
        }

        for layer in layers {
            if layer.isGlobal {
                hidden = layer(hidden, cos: globalPair.0, sin: globalPair.1)
            } else {
                hidden = layer(hidden, cos: windowCos, sin: windowSin)
            }
        }
        return hidden
    }
}

final class SAM31ScaleLayers: Module {
    @ModuleInfo(key: "first") var first: ConvTransposed2d?
    @ModuleInfo(key: "second") var second: ConvTransposed2d?

    init(inChannels: Int, scaleFactor: Float, kernelSize: Int, stride: Int) {
        if scaleFactor >= 2 {
            let mid = inChannels / 2
            self._first.wrappedValue = ConvTransposed2d(
                inputChannels: inChannels,
                outputChannels: mid,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(0),
                dilation: IntOrPair(1),
                bias: true
            )
            if scaleFactor >= 4 {
                self._second.wrappedValue = ConvTransposed2d(
                    inputChannels: mid,
                    outputChannels: mid / 2,
                    kernelSize: IntOrPair(kernelSize),
                    stride: IntOrPair(stride),
                    padding: IntOrPair(0),
                    dilation: IntOrPair(1),
                    bias: true
                )
            } else {
                self._second.wrappedValue = nil
            }
        } else {
            self._first.wrappedValue = nil
            self._second.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        if let first {
            hidden = first(hidden)
            if let second {
                hidden = gelu(hidden)
                hidden = second(hidden)
            }
        }
        return hidden
    }
}

final class SAM31FPNLayer: Module {
    @ModuleInfo(key: "scale_layers") var scaleLayers: SAM31ScaleLayers?
    @ModuleInfo(key: "proj1") var proj1: Conv2d
    @ModuleInfo(key: "proj2") var proj2: Conv2d

    let scaleFactor: Float

    init(
        inChannels: Int,
        outChannels: Int,
        scaleFactor: Float,
        kernelSize: Int,
        stride: Int
    ) {
        self.scaleFactor = scaleFactor
        if scaleFactor >= 2 {
            self._scaleLayers.wrappedValue = SAM31ScaleLayers(
                inChannels: inChannels,
                scaleFactor: scaleFactor,
                kernelSize: kernelSize,
                stride: stride
            )
        } else {
            self._scaleLayers.wrappedValue = nil
        }

        let currentChannels: Int
        if scaleFactor >= 4 {
            currentChannels = inChannels / 4
        } else if scaleFactor >= 2 {
            currentChannels = inChannels / 2
        } else {
            currentChannels = inChannels
        }

        self._proj1.wrappedValue = Conv2d(
            inputChannels: currentChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        self._proj2.wrappedValue = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(1),
            padding: IntOrPair(1),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        if let scaleLayers {
            hidden = scaleLayers(hidden)
        }
        hidden = proj1(hidden)
        hidden = proj2(hidden)
        return hidden
    }
}

final class SAM31TriViTDetNeck: Module {
    @ModuleInfo(key: "convs") var convs: [SAM31FPNLayer]
    @ModuleInfo(key: "interactive_convs") var interactiveConvs: [SAM31FPNLayer]
    @ModuleInfo(key: "propagation_convs") var propagationConvs: [SAM31FPNLayer]

    init(config: SAM31VisionEncoderConfig) {
        self._convs.wrappedValue = config.scaleFactors.map { scale in
            SAM31FPNLayer(
                inChannels: config.backboneConfig.hiddenSize,
                outChannels: config.fpnHiddenSize,
                scaleFactor: scale,
                kernelSize: config.fpnKernelSize,
                stride: config.fpnStride
            )
        }
        self._interactiveConvs.wrappedValue = config.scaleFactors.map { scale in
            SAM31FPNLayer(
                inChannels: config.backboneConfig.hiddenSize,
                outChannels: config.fpnHiddenSize,
                scaleFactor: scale,
                kernelSize: config.fpnKernelSize,
                stride: config.fpnStride
            )
        }
        self._propagationConvs.wrappedValue = config.scaleFactors.map { scale in
            SAM31FPNLayer(
                inChannels: config.backboneConfig.hiddenSize,
                outChannels: config.fpnHiddenSize,
                scaleFactor: scale,
                kernelSize: config.fpnKernelSize,
                stride: config.fpnStride
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> SAM31VisionFeaturePyramid {
        SAM31VisionFeaturePyramid(
            detFeatures: convs.map { $0(x) },
            interactiveFeatures: interactiveConvs.map { $0(x) },
            propagationFeatures: propagationConvs.map { $0(x) }
        )
    }
}

final class SAM31VisionEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: SAM31ViTBackbone
    @ModuleInfo(key: "neck") var neck: SAM31TriViTDetNeck

    init(config: SAM31VisionEncoderConfig) {
        self._backbone.wrappedValue = SAM31ViTBackbone(config: config.backboneConfig)
        self._neck.wrappedValue = SAM31TriViTDetNeck(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> SAM31VisionFeaturePyramid {
        neck(backbone(x))
    }
}

final class SAM31CLIPAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(config: SAM31TextEncoderConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let hiddenSize = x.dim(2)
        let q = qProj(x).reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

        // The shipped CLIP tower also uses 64-wide heads, including its array
        // mask, which is supported by MLX's fused full-attention Metal path.
        let attentionOutput: MLXArray
        if FusedAttentionPolicy.enabled {
            attentionOutput = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask
            )
        } else {
            var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
            if let mask {
                scores = scores + mask.asType(scores.dtype)
            }
            scores = softmax(scores, axis: -1)
            attentionOutput = MLX.matmul(scores, v)
        }
        let attended = attentionOutput.transposed(0, 2, 1, 3)
            .reshaped(batch, sequenceLength, hiddenSize)
        return outProj(attended)
    }
}

final class SAM31CLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(config: SAM31TextEncoderConfig) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class SAM31CLIPEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SAM31CLIPAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAM31CLIPMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(config: SAM31TextEncoderConfig) {
        self._selfAttn.wrappedValue = SAM31CLIPAttention(config: config)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = SAM31CLIPMLP(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let hidden = x + selfAttn(layerNorm1(x), mask: mask)
        return hidden + mlp(layerNorm2(hidden))
    }
}

final class SAM31CLIPEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [SAM31CLIPEncoderLayer]

    init(config: SAM31TextEncoderConfig) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in SAM31CLIPEncoderLayer(config: config) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return hidden
    }
}

final class SAM31CLIPTextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(config: SAM31TextEncoderConfig) {
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        let seqLen = inputIDs.dim(1)
        let positionIDs = MLXArray((0..<seqLen).map(Int32.init))
        return tokenEmbedding(inputIDs) + positionEmbedding(positionIDs)
    }
}

final class SAM31CLIPTextModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SAM31CLIPTextEmbeddings
    @ModuleInfo(key: "encoder") var encoder: SAM31CLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: SAM31TextEncoderConfig) {
        self._embeddings.wrappedValue = SAM31CLIPTextEmbeddings(config: config)
        self._encoder.wrappedValue = SAM31CLIPEncoder(config: config)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let hidden = embeddings(inputIDs)
        let seqLen = inputIDs.dim(1)
        var causalValues = [Float](repeating: 0, count: seqLen * seqLen)
        for row in 0..<seqLen {
            for column in (row + 1)..<seqLen {
                causalValues[row * seqLen + column] = -1e9
            }
        }
        var causal = MLXArray(causalValues, [1, 1, seqLen, seqLen]).asType(hidden.dtype)
        if let attentionMask {
            let padMask = (1 - attentionMask.asType(hidden.dtype)).expandedDimensions(axes: [1, 2]) * -1e9
            causal = causal + padMask
        }
        return finalLayerNorm(encoder(hidden, mask: causal))
    }
}

final class SAM31TextEncoder: Module {
    @ModuleInfo(key: "text_model") var textModel: SAM31CLIPTextModel

    init(config: SAM31TextEncoderConfig) {
        self._textModel.wrappedValue = SAM31CLIPTextModel(config: config)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        textModel(inputIDs, attentionMask: attentionMask)
    }
}

final class SAM31MultiheadAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int, kvDim: Int? = nil) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        let resolvedKVDim = kvDim ?? hiddenSize
        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(resolvedKVDim, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(resolvedKVDim, hiddenSize, bias: true)
        self._oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray, _ value: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batch = query.dim(0)
        let qLen = query.dim(1)
        let kLen = key.dim(1)
        let hiddenSize = qProj.weight.dim(0)

        let q = qProj(query).reshaped(batch, qLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(batch, kLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(batch, kLen, numHeads, headDim).transposed(0, 2, 1, 3)

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask {
            scores = scores + mask.asType(scores.dtype)
        }
        scores = softmax(scores, axis: -1)
        let attended = MLX.matmul(scores, v).transposed(0, 2, 1, 3).reshaped(batch, qLen, hiddenSize)
        return oProj(attended)
    }
}

final class SAM31MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let activation: String

    init(hiddenSize: Int, intermediateSize: Int, activation: String = "relu") {
        self.activation = activation
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = fc1(x)
        if activation == "relu" {
            return fc2(relu(hidden))
        }
        return fc2(gelu(hidden))
    }
}

final class SAM31DETREncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SAM31MultiheadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: SAM31MultiheadAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAM31MLP

    init(config: SAM31DETREncoderConfig) {
        self._selfAttn.wrappedValue = SAM31MultiheadAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._crossAttn.wrappedValue = SAM31MultiheadAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._layerNorm3.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = SAM31MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        super.init()
    }

    func callAsFunction(_ src: MLXArray, pos: MLXArray, textMemory: MLXArray, textMask: MLXArray?) -> MLXArray {
        let selfNorm = layerNorm1(src)
        let selfHidden = selfAttn(selfNorm + pos, selfNorm + pos, selfNorm)
        let afterSelf = src + selfHidden

        let crossMask: MLXArray?
        if let textMask {
            crossMask = (1 - textMask.asType(.float32)).expandedDimensions(axes: [1, 2]) * -1e9
        } else {
            crossMask = nil
        }

        let crossNorm = layerNorm2(afterSelf)
        let crossHidden = crossAttn(crossNorm, textMemory, textMemory, mask: crossMask)
        let afterCross = afterSelf + crossHidden
        return afterCross + mlp(layerNorm3(afterCross))
    }
}

final class SAM31DETREncoder: Module {
    @ModuleInfo(key: "layers") var layers: [SAM31DETREncoderLayer]

    init(config: SAM31DETREncoderConfig) {
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in SAM31DETREncoderLayer(config: config) }
        super.init()
    }

    func callAsFunction(_ src: MLXArray, pos: MLXArray, textMemory: MLXArray, textMask: MLXArray?) -> MLXArray {
        var hidden = src
        for layer in layers {
            hidden = layer(hidden, pos: pos, textMemory: textMemory, textMask: textMask)
        }
        return hidden
    }
}

final class SAM31BoxHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(hiddenSize: Int) {
        self._layer1.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._layer2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._layer3.wrappedValue = Linear(hiddenSize, 4, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer3(relu(layer2(relu(layer1(x)))))
    }
}

final class SAM31PresenceHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(hiddenSize: Int) {
        self._layer1.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._layer2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._layer3.wrappedValue = Linear(hiddenSize, 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer3(relu(layer2(relu(layer1(x)))))
    }
}

final class SAM31RefPointHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(hiddenSize: Int) {
        self._layer1.wrappedValue = Linear(hiddenSize * 2, hiddenSize, bias: true)
        self._layer2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(layer2(relu(layer1(x))))
    }
}

final class SAM31BoxRPBEmbed: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(numHeads: Int, hiddenSize: Int) {
        self._layer1.wrappedValue = Linear(2, hiddenSize, bias: true)
        self._layer2.wrappedValue = Linear(hiddenSize, numHeads, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer2(relu(layer1(x)))
    }
}

final class SAM31SinePositionEmbeddingForBoxes {
    let numPosFeats: Int
    let temperature: Float
    let scale: Float

    init(numPosFeats: Int = 128, temperature: Float = 10_000, scale: Float = 2 * .pi) {
        self.numPosFeats = numPosFeats
        self.temperature = temperature
        self.scale = scale
    }

    func encodeBoxes(_ boxes: MLXArray) -> MLXArray {
        let dimT = MLXArray((0..<numPosFeats).map { index in
            Float(Foundation.pow(Double(temperature), Double(2 * (index / 2)) / Double(numPosFeats)))
        })
        let coords = [
            boxes[0..., 0..., 1..<2] * scale,
            boxes[0..., 0..., 0..<1] * scale,
            boxes[0..., 0..., 2..<3] * scale,
            boxes[0..., 0..., 3..<4] * scale,
        ]

        var encodings: [MLXArray] = []
        encodings.reserveCapacity(coords.count)
        for coord in coords {
            let pos = (coord / dimT).reshaped(coord.dim(0), coord.dim(1), numPosFeats / 2, 2)
            let even = MLX.sin(pos[0..., 0..., 0..., 0])
            let odd = MLX.cos(pos[0..., 0..., 0..., 1])
            encodings.append(MLX.stacked([even, odd], axis: -1).reshaped(coord.dim(0), coord.dim(1), numPosFeats))
        }
        return MLX.concatenated(encodings, axis: -1)
    }
}

final class SAM31DETRDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SAM31MultiheadAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "text_cross_attn") var textCrossAttn: SAM31MultiheadAttention
    @ModuleInfo(key: "text_cross_attn_layer_norm") var textCrossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "vision_cross_attn") var visionCrossAttn: SAM31MultiheadAttention
    @ModuleInfo(key: "vision_cross_attn_layer_norm") var visionCrossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAM31MLP
    @ModuleInfo(key: "mlp_layer_norm") var mlpLayerNorm: LayerNorm

    init(config: SAM31DETRDecoderConfig) {
        self._selfAttn.wrappedValue = SAM31MultiheadAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._textCrossAttn.wrappedValue = SAM31MultiheadAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._textCrossAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._visionCrossAttn.wrappedValue = SAM31MultiheadAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._visionCrossAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = SAM31MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self._mlpLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        queryPos: MLXArray,
        inputsEmbeds: MLXArray,
        visionFeatures: MLXArray,
        visionPosEncoding: MLXArray,
        textMask: MLXArray?,
        visionMask: MLXArray?
    ) -> MLXArray {
        let qk = hiddenStates + queryPos
        let selfHidden = selfAttn(qk, qk, hiddenStates)
        let afterSelf = selfAttnLayerNorm(hiddenStates + selfHidden)

        let textHidden = textCrossAttn(afterSelf + queryPos, inputsEmbeds, inputsEmbeds, mask: textMask)
        let afterText = textCrossAttnLayerNorm(afterSelf + textHidden)

        let visionHidden = visionCrossAttn(
            afterText + queryPos,
            visionFeatures + visionPosEncoding,
            visionFeatures,
            mask: visionMask
        )
        let afterVision = visionCrossAttnLayerNorm(afterText + visionHidden)
        return mlpLayerNorm(afterVision + mlp(afterVision))
    }
}

final class SAM31DETRDecoder: Module {
    let hiddenSize: Int
    let numHeads: Int
    let clampPresenceLogitMaxValue: Float = 10
    let positionEmbedding: SAM31SinePositionEmbeddingForBoxes

    @ModuleInfo(key: "layers") var layers: [SAM31DETRDecoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm
    @ModuleInfo(key: "query_embed") var queryEmbed: Embedding
    @ModuleInfo(key: "reference_points") var referencePoints: Embedding
    @ModuleInfo(key: "presence_token") var presenceToken: Embedding
    @ModuleInfo(key: "presence_head") var presenceHead: SAM31PresenceHead
    @ModuleInfo(key: "presence_layer_norm") var presenceLayerNorm: LayerNorm
    @ModuleInfo(key: "box_head") var boxHead: SAM31BoxHead
    @ModuleInfo(key: "ref_point_head") var refPointHead: SAM31RefPointHead
    @ModuleInfo(key: "box_rpb_embed_x") var boxRPBEmbedX: SAM31BoxRPBEmbed
    @ModuleInfo(key: "box_rpb_embed_y") var boxRPBEmbedY: SAM31BoxRPBEmbed

    init(config: SAM31DETRDecoderConfig) {
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.positionEmbedding = SAM31SinePositionEmbeddingForBoxes(numPosFeats: config.hiddenSize / 2)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in SAM31DETRDecoderLayer(config: config) }
        self._outputLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._queryEmbed.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: config.hiddenSize)
        self._referencePoints.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: 4)
        self._presenceToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        self._presenceHead.wrappedValue = SAM31PresenceHead(hiddenSize: config.hiddenSize)
        self._presenceLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._boxHead.wrappedValue = SAM31BoxHead(hiddenSize: config.hiddenSize)
        self._refPointHead.wrappedValue = SAM31RefPointHead(hiddenSize: config.hiddenSize)
        self._boxRPBEmbedX.wrappedValue = SAM31BoxRPBEmbed(numHeads: config.numAttentionHeads, hiddenSize: config.hiddenSize)
        self._boxRPBEmbedY.wrappedValue = SAM31BoxRPBEmbed(numHeads: config.numAttentionHeads, hiddenSize: config.hiddenSize)
        super.init()
    }

    func callAsFunction(
        visionFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        visionPosEncoding: MLXArray,
        textMask: MLXArray?,
        spatialShape: (Int, Int)
    ) -> (hiddenStates: MLXArray, boxes: MLXArray, presence: MLXArray) {
        let batch = visionFeatures.dim(0)
        let numQueries = queryEmbed.weight.dim(0)

        let queryEmbeds = MLX.broadcast(queryEmbed.weight.expandedDimensions(axis: 0), to: [batch, numQueries, hiddenSize])
        let presence = MLX.broadcast(presenceToken.weight.expandedDimensions(axis: 0), to: [batch, 1, hiddenSize])
        var hiddenStates = MLX.concatenated([presence, queryEmbeds], axis: 1)
        var referenceBoxes = MLX.sigmoid(MLX.broadcast(referencePoints.weight.expandedDimensions(axis: 0), to: [batch, numQueries, 4]))

        let textCrossMask: MLXArray?
        if let textMask {
            textCrossMask = (1 - textMask.asType(.float32)).expandedDimensions(axes: [1, 2]) * -1e9
        } else {
            textCrossMask = nil
        }

        var intermediateHS: [MLXArray] = []
        var intermediateBoxes: [MLXArray] = []
        var intermediatePresence: [MLXArray] = []
        intermediateHS.reserveCapacity(layers.count)
        intermediateBoxes.reserveCapacity(layers.count)
        intermediatePresence.reserveCapacity(layers.count)

        for layer in layers {
            let querySineEmbed = positionEmbedding.encodeBoxes(referenceBoxes)
            let queryPos = refPointHead(querySineEmbed)
            let queryPosPadded = MLX.concatenated(
                [MLX.zeros([batch, 1, hiddenSize], dtype: queryPos.dtype), queryPos],
                axis: 1
            )
            let visionMask = computeRPB(referenceBoxes: referenceBoxes, spatialShape: spatialShape)
            let visionMaskPadded = MLX.concatenated(
                [MLX.zeros([batch, visionMask.dim(1), 1, visionMask.dim(3)], dtype: visionMask.dtype), visionMask],
                axis: 2
            )

            hiddenStates = layer(
                hiddenStates,
                queryPos: queryPosPadded,
                inputsEmbeds: inputsEmbeds,
                visionFeatures: visionFeatures,
                visionPosEncoding: visionPosEncoding,
                textMask: textCrossMask,
                visionMask: visionMaskPadded
            )

            let queryHidden = hiddenStates[0..., 1..., 0...]
            let queryHiddenNorm = outputLayerNorm(queryHidden)
            let delta = boxHead(queryHiddenNorm)
            let newReference = MLX.sigmoid(sam31InverseSigmoid(referenceBoxes) + delta)
            referenceBoxes = newReference
            intermediateHS.append(queryHiddenNorm)
            intermediateBoxes.append(newReference)

            let presenceHidden = hiddenStates[0..., 0..<1, 0...]
            let presenceLogit = MLX.clip(
                presenceHead(presenceLayerNorm(presenceHidden)).squeezed(axis: -1),
                min: -clampPresenceLogitMaxValue,
                max: clampPresenceLogitMaxValue
            )
            intermediatePresence.append(presenceLogit)
        }

        return (
            hiddenStates: MLX.stacked(intermediateHS, axis: 0),
            boxes: MLX.stacked(intermediateBoxes, axis: 0),
            presence: MLX.stacked(intermediatePresence, axis: 0)
        )
    }

    private func computeRPB(referenceBoxes: MLXArray, spatialShape: (Int, Int)) -> MLXArray {
        let height = spatialShape.0
        let width = spatialShape.1
        let batch = referenceBoxes.dim(0)
        let queryCount = referenceBoxes.dim(1)

        let cx = referenceBoxes[0..., 0..., 0]
        let cy = referenceBoxes[0..., 0..., 1]
        let w = referenceBoxes[0..., 0..., 2]
        let h = referenceBoxes[0..., 0..., 3]
        let boxes = MLX.stacked([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis: -1)

        let coordsH = MLXArray((0..<height).map { (Float($0) + 0.5) / Float(height) })
        let coordsW = MLXArray((0..<width).map { (Float($0) + 0.5) / Float(width) })

        let yBounds = MLX.stacked([boxes[0..., 0..., 1], boxes[0..., 0..., 3]], axis: -1)
            .reshaped(batch * queryCount, 1, 2)
        let xBounds = MLX.stacked([boxes[0..., 0..., 0], boxes[0..., 0..., 2]], axis: -1)
            .reshaped(batch * queryCount, 1, 2)
        let deltasY = (coordsH.reshaped(1, height, 1) - yBounds).reshaped(batch, queryCount, height, 2)
        let deltasX = (coordsW.reshaped(1, width, 1) - xBounds).reshaped(batch, queryCount, width, 2)

        let scaledX = MLX.sign(deltasX * 8) * (MLX.log2(MLX.abs(deltasX * 8) + 1) / 3.0)
        let scaledY = MLX.sign(deltasY * 8) * (MLX.log2(MLX.abs(deltasY * 8) + 1) / 3.0)

        let rpbX = boxRPBEmbedX(scaledX)
        let rpbY = boxRPBEmbedY(scaledY)
        let combined = (rpbY.expandedDimensions(axis: 3) + rpbX.expandedDimensions(axis: 2))
            .reshaped(batch, queryCount, height * width, numHeads)
            .transposed(0, 3, 1, 2)
        return combined
    }
}

final class SAM31PixelDecoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [Conv2d]
    @ModuleInfo(key: "norms") var norms: [GroupNorm]

    init(hiddenSize: Int, stageCount: Int = 3) {
        self._convLayers.wrappedValue = (0..<stageCount).map { _ in
            Conv2d(
                inputChannels: hiddenSize,
                outputChannels: hiddenSize,
                kernelSize: IntOrPair(3),
                stride: IntOrPair(1),
                padding: IntOrPair(1),
                bias: true
            )
        }
        self._norms.wrappedValue = (0..<stageCount).map { _ in
            GroupNorm(groupCount: 8, dimensions: hiddenSize, affine: true, pytorchCompatible: true)
        }
        super.init()
    }

    func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        var hidden = features[features.count - 1]
        for (index, feature) in features.dropLast().reversed().enumerated() {
            let targetHeight = feature.dim(1)
            let targetWidth = feature.dim(2)
            if hidden.dim(1) != targetHeight || hidden.dim(2) != targetWidth {
                let scale = FloatOrArray.array([
                    Float(targetHeight) / Float(hidden.dim(1)),
                    Float(targetWidth) / Float(hidden.dim(2)),
                ])
                hidden = Upsample(scaleFactor: scale, mode: .nearest)(hidden)
            }
            hidden = hidden + feature
            hidden = relu(norms[index](convLayers[index](hidden)))
        }
        return hidden
    }
}

final class SAM31MaskEmbedder: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(hiddenSize: Int) {
        self._layers.wrappedValue = [
            Linear(hiddenSize, hiddenSize, bias: true),
            Linear(hiddenSize, hiddenSize, bias: true),
            Linear(hiddenSize, hiddenSize, bias: true),
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden)
            if index < layers.count - 1 {
                hidden = relu(hidden)
            }
        }
        return hidden
    }
}

final class SAM31TextScoringMLP: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(hiddenSize: Int) {
        self._layer1.wrappedValue = Linear(hiddenSize, hiddenSize * 8, bias: true)
        self._layer2.wrappedValue = Linear(hiddenSize * 8, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer2(relu(layer1(x)))
    }
}

final class SAM31DotProductScoring: Module {
    let scale: Float
    let clampMaxValue: Float = 12

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ModuleInfo(key: "text_mlp") var textMLP: SAM31TextScoringMLP
    @ModuleInfo(key: "text_mlp_out_norm") var textMLPOutNorm: LayerNorm

    init(hiddenSize: Int) {
        self.scale = 1.0 / sqrt(Float(hiddenSize))
        self._queryProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._textProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._textMLP.wrappedValue = SAM31TextScoringMLP(hiddenSize: hiddenSize)
        self._textMLPOutNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ hs: MLXArray, inputsEmbeds: MLXArray, textMask: MLXArray?) -> MLXArray {
        let processed = textMLPOutNorm(textMLP(inputsEmbeds) + inputsEmbeds)
        let pooled: MLXArray
        if let textMask {
            let valid = textMask.asType(.float32).expandedDimensions(axis: 2)
            let denom = MLX.maximum(valid.sum(axis: 1), 1.0)
            pooled = (processed * valid).sum(axis: 1) / denom
        } else {
            pooled = processed.mean(axis: 1)
        }
        let projText = textProj(pooled)
        let projQueries = queryProj(hs)
        let scores = MLX.matmul(projQueries, projText.expandedDimensions(axis: 0).expandedDimensions(axis: 3))
        return MLX.clip(scores * scale, min: -clampMaxValue, max: clampMaxValue)
    }
}

final class SAM31MaskDecoder: Module {
    @ModuleInfo(key: "pixel_decoder") var pixelDecoder: SAM31PixelDecoder
    @ModuleInfo(key: "mask_embedder") var maskEmbedder: SAM31MaskEmbedder
    @ModuleInfo(key: "prompt_cross_attn") var promptCrossAttn: SAM31MultiheadAttention
    @ModuleInfo(key: "prompt_cross_attn_norm") var promptCrossAttnNorm: LayerNorm
    @ModuleInfo(key: "semantic_projection") var semanticProjection: Conv2d
    @ModuleInfo(key: "instance_projection") var instanceProjection: Conv2d

    init(config: SAM31DetectorMaskDecoderConfig) {
        self._pixelDecoder.wrappedValue = SAM31PixelDecoder(
            hiddenSize: config.hiddenSize,
            stageCount: config.numUpsamplingStages
        )
        self._maskEmbedder.wrappedValue = SAM31MaskEmbedder(hiddenSize: config.hiddenSize)
        self._promptCrossAttn.wrappedValue = SAM31MultiheadAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._promptCrossAttnNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._semanticProjection.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: 1,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        self._instanceProjection.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(
        objQueries: MLXArray,
        backboneFeatures: [MLXArray],
        encoderHiddenStates: MLXArray?,
        promptFeatures: MLXArray?,
        promptMask: MLXArray?
    ) -> MLXArray {
        var encodedStates = encoderHiddenStates
        if let promptFeatures, let encoderHiddenStates {
            let mask: MLXArray?
            if let promptMask {
                mask = (1 - promptMask.asType(.float32)).expandedDimensions(axes: [1, 2]) * -1e9
            } else {
                mask = nil
            }
            let normed = promptCrossAttnNorm(encoderHiddenStates)
            encodedStates = encoderHiddenStates + promptCrossAttn(normed, promptFeatures, promptFeatures, mask: mask)
        }

        var features = backboneFeatures
        if let encodedStates {
            let finest = features[features.count - 1]
            let height = finest.dim(1)
            let width = finest.dim(2)
            let channels = finest.dim(3)
            features[features.count - 1] = encodedStates[0..., 0..<(height * width), 0...].reshaped(
                encodedStates.dim(0),
                height,
                width,
                channels
            )
        }

        let pixelEmbed = pixelDecoder(features)
        let instanceEmbed = instanceProjection(pixelEmbed)
        let maskEmbeddings = maskEmbedder(objQueries)
        let flat = instanceEmbed.reshaped(instanceEmbed.dim(0), instanceEmbed.dim(1) * instanceEmbed.dim(2), instanceEmbed.dim(3))
        return MLX.matmul(maskEmbeddings, flat.transposed(0, 2, 1)).reshaped(
            maskEmbeddings.dim(0),
            maskEmbeddings.dim(1),
            instanceEmbed.dim(1),
            instanceEmbed.dim(2)
        )
    }
}

final class SAM31DetectorModel: Module {
    let config: SAM31DetectorConfig
    let posEnc: SAM31PositionEmbeddingSine

    @ModuleInfo(key: "vision_encoder") var visionEncoder: SAM31VisionEncoder
    @ModuleInfo(key: "text_encoder") var textEncoder: SAM31TextEncoder
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "detr_encoder") var detrEncoder: SAM31DETREncoder
    @ModuleInfo(key: "detr_decoder") var detrDecoder: SAM31DETRDecoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: SAM31MaskDecoder
    @ModuleInfo(key: "dot_product_scoring") var dotProductScoring: SAM31DotProductScoring

    init(config: SAM31DetectorConfig) {
        self.config = config
        self.posEnc = SAM31PositionEmbeddingSine(numPosFeats: config.detrEncoderConfig.hiddenSize / 2)
        self._visionEncoder.wrappedValue = SAM31VisionEncoder(config: config.visionConfig)
        self._textEncoder.wrappedValue = SAM31TextEncoder(config: config.textConfig)
        self._textProjection.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.detrEncoderConfig.hiddenSize,
            bias: true
        )
        self._detrEncoder.wrappedValue = SAM31DETREncoder(config: config.detrEncoderConfig)
        self._detrDecoder.wrappedValue = SAM31DETRDecoder(config: config.detrDecoderConfig)
        self._maskDecoder.wrappedValue = SAM31MaskDecoder(config: config.maskDecoderConfig)
        self._dotProductScoring.wrappedValue = SAM31DotProductScoring(hiddenSize: config.detrEncoderConfig.hiddenSize)
        super.init()
    }

    func getInputEmbeddings(_ inputIDs: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        textProjection(textEncoder(inputIDs, attentionMask: attentionMask))
    }

    func prepareVision(_ pixelValues: MLXArray) -> SAM31VisionContext {
        let featurePyramid = visionEncoder(pixelValues)
        let detFeatures = featurePyramid.detFeatures
        let pos = posEnc(detFeatures[detFeatures.count - 1])
        let coarse = detFeatures[detFeatures.count - 1]
        return SAM31VisionContext(
            featurePyramid: featurePyramid,
            src: coarse.reshaped(coarse.dim(0), coarse.dim(1) * coarse.dim(2), coarse.dim(3)),
            posFlat: pos.reshaped(pos.dim(0), pos.dim(1) * pos.dim(2), pos.dim(3)),
            spatialShape: (coarse.dim(1), coarse.dim(2))
        )
    }

    func detect(visionContext: SAM31VisionContext, inputsEmbeds: MLXArray, attentionMask: MLXArray?) -> SAM31DetectorOutput {
        let encoded = detrEncoder(visionContext.src, pos: visionContext.posFlat, textMemory: inputsEmbeds, textMask: attentionMask)
        let decoded = detrDecoder(
            visionFeatures: encoded,
            inputsEmbeds: inputsEmbeds,
            visionPosEncoding: visionContext.posFlat,
            textMask: attentionMask,
            spatialShape: visionContext.spatialShape
        )

        let boxStates = decoded.boxes[decoded.boxes.dim(0) - 1]
        let cx = boxStates[0..., 0..., 0]
        let cy = boxStates[0..., 0..., 1]
        let w = boxStates[0..., 0..., 2]
        let h = boxStates[0..., 0..., 3]
        let predBoxes = MLX.stacked([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis: -1)
        let queryStates = decoded.hiddenStates[decoded.hiddenStates.dim(0) - 1]
        let predLogits = dotProductScoring(decoded.hiddenStates, inputsEmbeds: inputsEmbeds, textMask: attentionMask)[decoded.hiddenStates.dim(0) - 1].squeezed(axis: -1)
        let predMasks = maskDecoder(
            objQueries: queryStates,
            backboneFeatures: visionContext.detFeatures,
            encoderHiddenStates: encoded,
            promptFeatures: inputsEmbeds,
            promptMask: attentionMask
        )

        return SAM31DetectorOutput(
            predLogits: predLogits,
            predBoxes: predBoxes,
            predMasks: predMasks,
            presenceLogits: decoded.presence[decoded.presence.dim(0) - 1]
        )
    }
}

final class SAM31Model: Module {
    let config: SAM31ModelConfig

    @ModuleInfo(key: "detector_model") var detectorModel: SAM31DetectorModel
    @ModuleInfo(key: "tracker_model") var trackerModel: SAM31TrackerModel?

    init(config: SAM31ModelConfig) {
        self.config = config
        self._detectorModel.wrappedValue = SAM31DetectorModel(config: config.detectorConfig)
        if let trackerConfig = config.trackerConfig {
            self._trackerModel.wrappedValue = SAM31TrackerModel(config: trackerConfig)
        } else {
            self._trackerModel.wrappedValue = nil
        }
        super.init()
    }
}

final class SAM31TrackerModel: Module {
    let config: SAM31TrackerConfig

    @ModuleInfo(key: "interactive_sam_prompt_encoder") var interactiveSAMPromptEncoder: SAM31InteractivePromptEncoder
    @ModuleInfo(key: "interactive_sam_mask_decoder") var interactiveSAMMaskDecoder: SAM31InteractiveMaskDecoder

    init(config: SAM31TrackerConfig) {
        self.config = config
        var interactiveMaskConfig = config.maskDecoderConfig
        interactiveMaskConfig.numMultimaskOutputs = max(interactiveMaskConfig.numMultimaskOutputs, 4)
        interactiveMaskConfig.multiplexCount = 1
        self._interactiveSAMPromptEncoder.wrappedValue = SAM31InteractivePromptEncoder(config: config.promptEncoderConfig)
        self._interactiveSAMMaskDecoder.wrappedValue = SAM31InteractiveMaskDecoder(config: interactiveMaskConfig)
        super.init()
    }

    func segment(
        featurePyramid: SAM31VisionFeaturePyramid,
        boxPrompt: MLXArray? = nil,
        pointPrompt: SAM31PointPromptTensor? = nil,
        maskPrompt: MLXArray? = nil,
        multimaskOutput: Bool = false
    ) -> SAM31InteractiveMaskOutput {
        let interactiveFeatures = featurePyramid.interactiveFeatures
        let coarse = interactiveFeatures[interactiveFeatures.count - 1]
        let batch = coarse.dim(0)
        let targetHeight = coarse.dim(1)
        let targetWidth = coarse.dim(2)
        let flattened = coarse.reshaped(batch, targetHeight * targetWidth, coarse.dim(3))

        let resizedMaskPrompt: MLXArray?
        if let maskPrompt {
            let maskHeight = interactiveFeatures[0].dim(1)
            let maskWidth = interactiveFeatures[0].dim(2)
            if maskPrompt.dim(1) == maskHeight && maskPrompt.dim(2) == maskWidth {
                resizedMaskPrompt = maskPrompt
            } else {
                let scale = FloatOrArray.array([
                    Float(maskHeight) / Float(maskPrompt.dim(1)),
                    Float(maskWidth) / Float(maskPrompt.dim(2)),
                ])
                resizedMaskPrompt = Upsample(
                    scaleFactor: scale,
                    mode: .linear(alignCorners: false)
                )(maskPrompt)
            }
        } else {
            resizedMaskPrompt = nil
        }

        let promptEncodings = interactiveSAMPromptEncoder(
            points: pointPrompt,
            boxes: boxPrompt,
            masks: resizedMaskPrompt,
            targetHeight: targetHeight,
            targetWidth: targetWidth
        )
        let highResFeatures = [
            interactiveFeatures.count > 1 ? interactiveFeatures[1] : interactiveFeatures[0],
            interactiveFeatures[0],
        ]

        return interactiveSAMMaskDecoder(
            imageEmbeddings: flattened,
            imagePE: promptEncodings.imagePE,
            sparsePromptEmbeddings: promptEncodings.sparseEmbeddings,
            densePromptEmbeddings: promptEncodings.denseEmbeddings,
            multimaskOutput: multimaskOutput,
            highResFeatures: highResFeatures
        )
    }
}
