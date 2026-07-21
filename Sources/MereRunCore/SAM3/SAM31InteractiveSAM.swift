import Foundation
import MLX
import MLXFast
import MLXNN

struct SAM31InteractiveMaskOutput: @unchecked Sendable {
    let predMasks: MLXArray
    let iouScores: MLXArray
    let objectScores: MLXArray
}

struct SAM31PointPromptTensor: @unchecked Sendable {
    let coords: MLXArray
    let labels: MLXArray
}

final class SAM31LayerNorm2d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray
    let eps: Float

    init(numChannels: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLX.ones([numChannels], dtype: .float32)
        self._bias.wrappedValue = MLX.zeros([numChannels], dtype: .float32)
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

final class SAM31OutputMLP: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self._layer1.wrappedValue = Linear(inputDim, hiddenDim, bias: true)
        self._layer2.wrappedValue = Linear(hiddenDim, hiddenDim, bias: true)
        self._layer3.wrappedValue = Linear(hiddenDim, outputDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer3(relu(layer2(relu(layer1(x)))))
    }
}

final class SAM31SAMAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int, downsampleRate: Int = 1) {
        let internalDim = hiddenSize / downsampleRate
        self.numHeads = numHeads
        self.headDim = internalDim / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qProj.wrappedValue = Linear(hiddenSize, internalDim, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, internalDim, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, internalDim, bias: true)
        self._oProj.wrappedValue = Linear(internalDim, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray, _ value: MLXArray) -> MLXArray {
        let batch = query.dim(0)
        let qLen = query.dim(1)
        let kLen = key.dim(1)
        let internalDim = qProj.weight.dim(0)

        let q = qProj(query).reshaped(batch, qLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(batch, kLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(batch, kLen, numHeads, headDim).transposed(0, 2, 1, 3)

        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)
        let out = MLX.matmul(attn, v).transposed(0, 2, 1, 3).reshaped(batch, qLen, internalDim)
        return oProj(out)
    }
}

final class SAM31TransformerMLPBlock: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    let activation: String

    init(inputDim: Int, hiddenDim: Int, activation: String = "relu") {
        self._projIn.wrappedValue = Linear(inputDim, hiddenDim, bias: true)
        self._projOut.wrappedValue = Linear(hiddenDim, inputDim, bias: true)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = projIn(x)
        let activated = activation == "gelu" ? gelu(hidden) : relu(hidden)
        return projOut(activated)
    }
}

final class SAM31TwoWayAttentionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SAM31SAMAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "cross_attn_token_to_image") var crossAttnTokenToImage: SAM31SAMAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAM31TransformerMLPBlock
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "cross_attn_image_to_token") var crossAttnImageToToken: SAM31SAMAttention
    @ModuleInfo(key: "layer_norm4") var layerNorm4: LayerNorm

    init(hiddenSize: Int, numHeads: Int, mlpDim: Int = 2048, attentionDownsampleRate: Int = 2) {
        self._selfAttn.wrappedValue = SAM31SAMAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._crossAttnTokenToImage.wrappedValue = SAM31SAMAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            downsampleRate: attentionDownsampleRate
        )
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._mlp.wrappedValue = SAM31TransformerMLPBlock(inputDim: hiddenSize, hiddenDim: mlpDim)
        self._layerNorm3.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._crossAttnImageToToken.wrappedValue = SAM31SAMAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            downsampleRate: attentionDownsampleRate
        )
        self._layerNorm4.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        queryPE: MLXArray,
        keyPE: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray) {
        let q = queries + queryPE
        let selfAttended = selfAttn(q, q, queries)
        let nextQueries = layerNorm1(queries + selfAttended)

        let q2 = nextQueries + queryPE
        let k2 = keys + keyPE
        let tokenToImage = crossAttnTokenToImage(q2, k2, keys)
        let queriesAfterCross = layerNorm2(nextQueries + tokenToImage)

        let mlpOut = mlp(queriesAfterCross)
        let queriesAfterMLP = layerNorm3(queriesAfterCross + mlpOut)

        let imageToToken = crossAttnImageToToken(keys + keyPE, queriesAfterMLP + queryPE, queriesAfterMLP)
        let nextKeys = layerNorm4(keys + imageToToken)
        return (queriesAfterMLP, nextKeys)
    }
}

final class SAM31TwoWayTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [SAM31TwoWayAttentionBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttnTokenToImage: SAM31SAMAttention
    @ModuleInfo(key: "layer_norm_final_attn") var layerNormFinalAttn: LayerNorm

    init(hiddenSize: Int, numHeads: Int, numLayers: Int, mlpDim: Int, attentionDownsampleRate: Int) {
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            SAM31TwoWayAttentionBlock(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                mlpDim: mlpDim,
                attentionDownsampleRate: attentionDownsampleRate
            )
        }
        self._finalAttnTokenToImage.wrappedValue = SAM31SAMAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            downsampleRate: attentionDownsampleRate
        )
        self._layerNormFinalAttn.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(
        imageEmbedding: MLXArray,
        imagePE: MLXArray,
        pointEmbedding: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray) {
        var queries = pointEmbedding
        var keys = imageEmbedding

        for layer in layers {
            let result = layer(queries: queries, keys: keys, queryPE: pointEmbedding, keyPE: imagePE)
            queries = result.queries
            keys = result.keys
        }

        let q = queries + pointEmbedding
        let k = keys + imagePE
        let attended = finalAttnTokenToImage(q, k, keys)
        queries = layerNormFinalAttn(queries + attended)
        return (queries, keys)
    }
}

final class SAM31MaskEmbedConvs: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "layer_norm1") var layerNorm1: SAM31LayerNorm2d
    @ModuleInfo(key: "layer_norm2") var layerNorm2: SAM31LayerNorm2d

    init(embedDim: Int, maskInputChannels: Int, eps: Float = 1e-6) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: maskInputChannels / 4,
            kernelSize: IntOrPair(2),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            bias: true
        )
        self._conv2.wrappedValue = Conv2d(
            inputChannels: maskInputChannels / 4,
            outputChannels: maskInputChannels,
            kernelSize: IntOrPair(2),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            bias: true
        )
        self._conv3.wrappedValue = Conv2d(
            inputChannels: maskInputChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        self._layerNorm1.wrappedValue = SAM31LayerNorm2d(numChannels: maskInputChannels / 4, eps: eps)
        self._layerNorm2.wrappedValue = SAM31LayerNorm2d(numChannels: maskInputChannels, eps: eps)
        super.init()
    }

    func callAsFunction(_ masks: MLXArray) -> MLXArray {
        var hidden = gelu(layerNorm1(conv1(masks)))
        hidden = gelu(layerNorm2(conv2(hidden)))
        hidden = conv3(hidden)
        return hidden.reshaped(hidden.dim(0), hidden.dim(1) * hidden.dim(2), hidden.dim(3))
    }
}

final class SAM31RandomPositionalEmbedding: Module {
    @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray

    init(numPosFeats: Int = 128) {
        self._positionalEmbedding.wrappedValue = MLX.zeros([2, numPosFeats], dtype: .float32)
        super.init()
    }

    func callAsFunction(size: (Int, Int)) -> MLXArray {
        let height = size.0
        let width = size.1
        let gridY = MLXArray((0..<height).map { Float($0) / Float(max(height, 1)) })
        let gridX = MLXArray((0..<width).map { Float($0) / Float(max(width, 1)) })
        let yy = MLX.broadcast(gridY.reshaped(height, 1), to: [height, width])
        let xx = MLX.broadcast(gridX.reshaped(1, width), to: [height, width])
        let coords = MLX.stacked([xx.reshaped(height * width), yy.reshaped(height * width)], axis: -1)
        return forwardWithCoords(coords.expandedDimensions(axis: 0))[0]
    }

    func forwardWithCoords(_ coords: MLXArray) -> MLXArray {
        let centered = coords * 2 - 1
        let projected = MLX.matmul(centered, positionalEmbedding)
        let angles = projected * (2 * Float.pi)
        return MLX.concatenated([MLX.sin(angles), MLX.cos(angles)], axis: -1)
    }
}

final class SAM31InteractivePromptEncoder: Module {
    let embedDim: Int
    let imageEmbeddingSize: (height: Int, width: Int)
    let inputImageSize: (height: Int, width: Int)

    @ModuleInfo(key: "point_embed") var pointEmbed: Embedding
    @ModuleInfo(key: "not_a_point_embed") var notAPointEmbed: Embedding
    @ModuleInfo(key: "mask_embed") var maskEmbed: SAM31MaskEmbedConvs
    @ModuleInfo(key: "no_mask_embed") var noMaskEmbed: Embedding
    @ModuleInfo(key: "shared_embedding") var sharedEmbedding: SAM31RandomPositionalEmbedding

    init(config: SAM31PromptEncoderConfig) {
        self.embedDim = config.hiddenSize
        self.imageEmbeddingSize = (config.imageSize / config.patchSize, config.imageSize / config.patchSize)
        self.inputImageSize = (config.imageSize, config.imageSize)
        self._pointEmbed.wrappedValue = Embedding(embeddingCount: config.numPointEmbeddings, dimensions: config.hiddenSize)
        self._notAPointEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        self._maskEmbed.wrappedValue = SAM31MaskEmbedConvs(
            embedDim: config.hiddenSize,
            maskInputChannels: config.maskInputChannels,
            eps: config.layerNormEps
        )
        self._noMaskEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        self._sharedEmbedding.wrappedValue = SAM31RandomPositionalEmbedding(numPosFeats: config.hiddenSize / 2)
        super.init()
    }

    func getDensePE(targetHeight: Int? = nil, targetWidth: Int? = nil) -> MLXArray {
        let height = targetHeight ?? imageEmbeddingSize.height
        let width = targetWidth ?? imageEmbeddingSize.width
        return sharedEmbedding(size: (height, width)).expandedDimensions(axis: 0)
    }

    func callAsFunction(
        points: SAM31PointPromptTensor? = nil,
        boxes: MLXArray? = nil,
        masks: MLXArray? = nil,
        targetHeight: Int,
        targetWidth: Int
    ) -> (sparseEmbeddings: MLXArray, denseEmbeddings: MLXArray, imagePE: MLXArray) {
        var batch = 1
        var sparseEmbeddings = MLX.zeros([1, 0, embedDim], dtype: .float32)

        if let points {
            batch = points.coords.dim(0)
            let pointEmbeddings = embedPoints(points.coords, labels: points.labels, targetHeight: targetHeight, targetWidth: targetWidth)
            sparseEmbeddings = pointEmbeddings
        }

        if let boxes {
            batch = boxes.dim(0)
            let boxEmbeddings = embedBoxes(boxes, targetHeight: targetHeight, targetWidth: targetWidth)
            if sparseEmbeddings.dim(1) == 0 {
                sparseEmbeddings = boxEmbeddings
            } else {
                sparseEmbeddings = MLX.concatenated([sparseEmbeddings, boxEmbeddings], axis: 1)
            }
        }

        let denseEmbeddings: MLXArray
        if let masks {
            batch = masks.dim(0)
            denseEmbeddings = maskEmbed(masks)
            if sparseEmbeddings.dim(1) == 0 {
                sparseEmbeddings = MLX.broadcast(
                    notAPointEmbed.weight.reshaped(1, 1, embedDim),
                    to: [batch, 1, embedDim]
                )
            }
        } else {
            let base = noMaskEmbed.weight.reshaped(1, 1, embedDim)
            denseEmbeddings = MLX.broadcast(base, to: [batch, targetHeight * targetWidth, embedDim])
        }

        let imagePE = MLX.broadcast(getDensePE(targetHeight: targetHeight, targetWidth: targetWidth), to: [batch, targetHeight * targetWidth, embedDim])
        if sparseEmbeddings.dim(0) != batch {
            sparseEmbeddings = MLX.broadcast(sparseEmbeddings, to: [batch, sparseEmbeddings.dim(1), sparseEmbeddings.dim(2)])
        }
        return (sparseEmbeddings, denseEmbeddings, imagePE)
    }

    private func embedPoints(_ coords: MLXArray, labels: MLXArray, targetHeight: Int, targetWidth: Int) -> MLXArray {
        let shifted = coords + 0.5
        let normalizer = MLXArray(
            [Float(max(inputImageSize.width, 1)), Float(max(inputImageSize.height, 1))],
            [1, 1, 2]
        ).asType(.float32)
        var pointEmbeddings = sharedEmbedding.forwardWithCoords(shifted / normalizer)

        let count = labels.dim(labels.ndim - 1)
        for index in 0..<count {
            let labelSlice = labels[0..., index..<(index + 1)]
            let validMask = labelSlice .>= MLXArray(0)
            let safeLabels = MLX.maximum(labelSlice, 0)
            let labelEmbedding = pointEmbed(safeLabels.asType(.int32))
            pointEmbeddings = pointEmbeddings + labelEmbedding * validMask.expandedDimensions(axis: -1).asType(labelEmbedding.dtype)

            let paddingMask = labelSlice .== MLXArray(-1)
            if MLX.any(paddingMask).item(Bool.self) {
                let notPoint = notAPointEmbed.weight.reshaped(1, 1, embedDim)
                pointEmbeddings = MLX.where(paddingMask.expandedDimensions(axis: -1), notPoint, pointEmbeddings)
            }
        }
        return pointEmbeddings
    }

    private func embedBoxes(_ boxes: MLXArray, targetHeight: Int, targetWidth: Int) -> MLXArray {
        let reshaped = (boxes + 0.5).reshaped(boxes.dim(0), boxes.dim(1) * 2, 2)
        let normalizer = MLXArray(
            [Float(max(inputImageSize.width, 1)), Float(max(inputImageSize.height, 1))],
            [1, 1, 2]
        ).asType(.float32)
        let cornerEmbeddings = sharedEmbedding.forwardWithCoords(reshaped / normalizer)
        let firstLabel = MLXArray([2], [1, 1]).asType(.int32)
        let secondLabel = MLXArray([3], [1, 1]).asType(.int32)
        for boxIndex in 0..<boxes.dim(1) {
            let start = boxIndex * 2
            cornerEmbeddings[0..., start..<(start + 1), 0...] = cornerEmbeddings[0..., start..<(start + 1), 0...] + pointEmbed(firstLabel)
            cornerEmbeddings[0..., (start + 1)..<(start + 2), 0...] = cornerEmbeddings[0..., (start + 1)..<(start + 2), 0...] + pointEmbed(secondLabel)
        }
        return cornerEmbeddings
    }
}

final class SAM31InteractiveMaskDecoder: Module {
    let numMaskTokens: Int

    @ModuleInfo(key: "transformer") var transformer: SAM31TwoWayTransformer
    @ModuleInfo(key: "iou_token") var iouToken: Embedding
    @ModuleInfo(key: "mask_tokens") var maskTokens: Embedding
    @ModuleInfo(key: "obj_score_token") var objScoreToken: Embedding
    @ModuleInfo(key: "output_hypernetworks_mlps") var outputHypernetworksMLPs: [SAM31OutputMLP]
    @ModuleInfo(key: "iou_prediction_head") var iouPredictionHead: SAM31OutputMLP
    @ModuleInfo(key: "pred_obj_score_head") var predObjScoreHead: SAM31OutputMLP
    @ModuleInfo(key: "upscale_conv1") var upscaleConv1: ConvTransposed2d
    @ModuleInfo(key: "upscale_conv2") var upscaleConv2: ConvTransposed2d
    @ModuleInfo(key: "upscale_layer_norm") var upscaleLayerNorm: SAM31LayerNorm2d
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d

    init(config: SAM31TrackerMaskDecoderConfig) {
        self.numMaskTokens = max(config.numMultimaskOutputs, 1)
        self._transformer.wrappedValue = SAM31TwoWayTransformer(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            mlpDim: config.mlpDim,
            attentionDownsampleRate: config.attentionDownsampleRate
        )
        self._iouToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        self._maskTokens.wrappedValue = Embedding(embeddingCount: self.numMaskTokens, dimensions: config.hiddenSize)
        self._objScoreToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        self._outputHypernetworksMLPs.wrappedValue = (0..<self.numMaskTokens).map { _ in
            SAM31OutputMLP(inputDim: config.hiddenSize, hiddenDim: config.hiddenSize, outputDim: config.hiddenSize / 8)
        }
        self._iouPredictionHead.wrappedValue = SAM31OutputMLP(
            inputDim: config.hiddenSize,
            hiddenDim: config.hiddenSize,
            outputDim: self.numMaskTokens
        )
        self._predObjScoreHead.wrappedValue = SAM31OutputMLP(inputDim: config.hiddenSize, hiddenDim: config.hiddenSize, outputDim: 1)
        self._upscaleConv1.wrappedValue = ConvTransposed2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize / 4,
            kernelSize: IntOrPair(2),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            dilation: IntOrPair(1),
            bias: true
        )
        self._upscaleConv2.wrappedValue = ConvTransposed2d(
            inputChannels: config.hiddenSize / 4,
            outputChannels: config.hiddenSize / 8,
            kernelSize: IntOrPair(2),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            dilation: IntOrPair(1),
            bias: true
        )
        self._upscaleLayerNorm.wrappedValue = SAM31LayerNorm2d(numChannels: config.hiddenSize / 4)
        self._convS0.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize / 8,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        self._convS1.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize / 4,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(
        imageEmbeddings: MLXArray,
        imagePE: MLXArray,
        sparsePromptEmbeddings: MLXArray,
        densePromptEmbeddings: MLXArray,
        multimaskOutput: Bool,
        highResFeatures: [MLXArray]
    ) -> SAM31InteractiveMaskOutput {
        let batch = imageEmbeddings.dim(0)
        let hiddenSize = imageEmbeddings.dim(2)

        let tokens = MLX.concatenated([
            MLX.broadcast(iouToken.weight.expandedDimensions(axis: 0), to: [batch, 1, hiddenSize]),
            MLX.broadcast(maskTokens.weight.expandedDimensions(axis: 0), to: [batch, numMaskTokens, hiddenSize]),
            MLX.broadcast(objScoreToken.weight.expandedDimensions(axis: 0), to: [batch, 1, hiddenSize]),
            sparsePromptEmbeddings,
        ], axis: 1)

        let transformed = transformer(
            imageEmbedding: imageEmbeddings + densePromptEmbeddings,
            imagePE: imagePE,
            pointEmbedding: tokens
        )
        let hs = transformed.queries
        let src = transformed.keys

        let iouTokenOut = hs[0..., 0..<1, 0...]
        let maskTokensOut = hs[0..., 1..<(1 + numMaskTokens), 0...]
        let objScoreTokenOut = hs[0..., (1 + numMaskTokens)..<(2 + numMaskTokens), 0...]

        let hw = src.dim(1)
        let side = Int(Double(hw).squareRoot())
        var upscaled = src.reshaped(batch, side, side, hiddenSize)
        upscaled = gelu(upscaleLayerNorm(upscaleConv1(upscaled)))

        if !highResFeatures.isEmpty {
            let s1 = convS1(highResFeatures[0])
            if s1.dim(1) == upscaled.dim(1) && s1.dim(2) == upscaled.dim(2) {
                upscaled = upscaled + s1
            }
        }

        upscaled = gelu(upscaleConv2(upscaled))
        if highResFeatures.count > 1 {
            let s0 = convS0(highResFeatures[1])
            if s0.dim(1) == upscaled.dim(1) && s0.dim(2) == upscaled.dim(2) {
                upscaled = upscaled + s0
            }
        }

        let upscaledFlat = upscaled.reshaped(batch, upscaled.dim(1) * upscaled.dim(2), upscaled.dim(3))
        var maskList: [MLXArray] = []
        maskList.reserveCapacity(numMaskTokens)
        for index in 0..<numMaskTokens {
            let hyperOut = outputHypernetworksMLPs[index](maskTokensOut[0..., index, 0...])
            let mask = (upscaledFlat * hyperOut.expandedDimensions(axis: 1)).sum(axis: -1).reshaped(batch, upscaled.dim(1), upscaled.dim(2))
            maskList.append(mask.expandedDimensions(axis: 1))
        }
        let allMasks = MLX.concatenated(maskList, axis: 1)
        let iouScores = iouPredictionHead(iouTokenOut.squeezed(axis: 1))
        let objectScores = predObjScoreHead(objScoreTokenOut.squeezed(axis: 1))

        guard !multimaskOutput, numMaskTokens > 1 else {
            return SAM31InteractiveMaskOutput(
                predMasks: allMasks,
                iouScores: iouScores,
                objectScores: objectScores
            )
        }

        let bestIndex = iouScores.argMax(axis: -1).item(Int32.self)
        return SAM31InteractiveMaskOutput(
            predMasks: allMasks[0..., Int(bestIndex)..<(Int(bestIndex) + 1), 0..., 0...],
            iouScores: iouScores[0..., Int(bestIndex)..<(Int(bestIndex) + 1)],
            objectScores: objectScores
        )
    }
}
