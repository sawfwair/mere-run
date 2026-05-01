import Foundation
@preconcurrency import MLX
import MLXNN

private let falconPerceptionImageNetMean: [Float] = [0.485, 0.456, 0.406]
private let falconPerceptionImageNetStd: [Float] = [0.229, 0.224, 0.225]

final class FalconPerceptionAnyUpResBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "shortcut") var shortcut: Conv2d?

    private let usesShortcut: Bool

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 1, numGroups: Int = 8) {
        let padding = kernelSize / 2
        self._norm1.wrappedValue = GroupNorm(
            groupCount: numGroups,
            dimensions: inChannels,
            affine: true,
            pytorchCompatible: true
        )
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            padding: IntOrPair(padding),
            bias: false
        )
        self._norm2.wrappedValue = GroupNorm(
            groupCount: numGroups,
            dimensions: outChannels,
            affine: true,
            pytorchCompatible: true
        )
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            padding: IntOrPair(padding),
            bias: false
        )

        self.usesShortcut = inChannels != outChannels
        if usesShortcut {
            self._shortcut.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: 1,
                bias: false
            )
        } else {
            self._shortcut.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = silu(norm1(x))
        hidden = conv1(hidden)
        hidden = silu(norm2(hidden))
        hidden = conv2(hidden)
        let residual = usesShortcut ? shortcut!(x) : x
        return hidden + residual
    }
}

private func falconPerceptionReflectPad(_ x: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return x }

    var paddedArray = x
    let height = paddedArray.dim(1)
    let width = paddedArray.dim(2)
    precondition(height > pad && width > pad, "Reflect padding requires spatial dims larger than pad.")

    let top = paddedArray[0..., 1..<(pad + 1), 0..., 0...][0..., .stride(by: -1), 0..., 0...]
    let bottom = paddedArray[0..., (height - pad - 1)..<(height - 1), 0..., 0...][0..., .stride(by: -1), 0..., 0...]
    paddedArray = concatenated([top, paddedArray, bottom], axis: 1)

    let left = paddedArray[0..., 0..., 1..<(pad + 1), 0...][0..., 0..., .stride(by: -1), 0...]
    let right = paddedArray[0..., 0..., (width - pad - 1)..<(width - 1), 0...][0..., 0..., .stride(by: -1), 0...]
    return concatenated([left, paddedArray, right], axis: 2)
}

final class FalconPerceptionAnyUpEncoder: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "blocks") var blocks: [FalconPerceptionAnyUpResBlock]

    private let reflectPadding: Bool
    private let reflectPaddingSize: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        numBlocks: Int = 2,
        blockKernelSize: Int = 1,
        reflectPadding: Bool = false
    ) {
        self.reflectPadding = reflectPadding && kernelSize > 1
        self.reflectPaddingSize = kernelSize / 2
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            padding: IntOrPair(self.reflectPadding ? 0 : (kernelSize / 2)),
            bias: false
        )
        self._blocks.wrappedValue = (0..<numBlocks).map { _ in
            FalconPerceptionAnyUpResBlock(
                inChannels: outChannels,
                outChannels: outChannels,
                kernelSize: blockKernelSize
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = reflectPadding ? falconPerceptionReflectPad(x, pad: reflectPaddingSize) : x
        hidden = conv(hidden)
        for block in blocks {
            hidden = block(hidden)
        }
        return hidden
    }
}

final class FalconPerceptionAnyUpLearnedFeatureUnification: Module {
    @ParameterInfo(key: "basis") var basis: MLXArray

    let outChannels: Int
    let kernelSize: Int

    init(outChannels: Int, kernelSize: Int) {
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self._basis.wrappedValue = MLX.zeros([outChannels, kernelSize, kernelSize, 1], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let batch = features.dim(0)
        let height = features.dim(1)
        let width = features.dim(2)
        let channels = features.dim(3)
        let padding = kernelSize / 2

        var x = features.transposed(0, 3, 1, 2).reshaped(batch * channels, height, width, 1)
        x = padded(x, widths: [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        var convolved = conv2d(x, basis, stride: 1, padding: 0)

        var mask = MLX.ones([1, height, width, 1], dtype: features.dtype)
        mask = padded(mask, widths: [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        let onesKernel = MLX.ones([1, kernelSize, kernelSize, 1], dtype: features.dtype)
        let denominator = conv2d(mask, onesKernel, stride: 1, padding: 0)
        convolved = convolved / denominator

        let viewed = convolved
            .reshaped(batch, channels, height, width, outChannels)
            .transposed(0, 1, 4, 2, 3)
            .reshaped(batch, channels * outChannels, height, width)
            .reshaped(batch, outChannels, channels, height, width)
        let attention = softmax(viewed, axis: 1)
        let result = mean(attention, axes: [2])
        return result.transposed(0, 2, 3, 1)
    }
}

final class FalconPerceptionAnyUpLFUEncoder: Module {
    @ModuleInfo(key: "lfu") var lfu: FalconPerceptionAnyUpLearnedFeatureUnification
    @ModuleInfo(key: "blocks") var blocks: [FalconPerceptionAnyUpResBlock]

    init(qkDim: Int, kernelSizeLFU: Int = 5, numBlocks: Int = 2, blockKernelSize: Int = 1) {
        self._lfu.wrappedValue = FalconPerceptionAnyUpLearnedFeatureUnification(
            outChannels: qkDim,
            kernelSize: kernelSizeLFU
        )
        self._blocks.wrappedValue = (0..<numBlocks).map { _ in
            FalconPerceptionAnyUpResBlock(
                inChannels: qkDim,
                outChannels: qkDim,
                kernelSize: blockKernelSize
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = lfu(x)
        for block in blocks {
            hidden = block(hidden)
        }
        return hidden
    }
}

final class FalconPerceptionAnyUpRoPE: Module {
    @ParameterInfo(key: "freqs") var freqs: MLXArray

    let dim: Int

    init(dim: Int) {
        self.dim = dim
        self._freqs.wrappedValue = MLX.zeros([2, dim], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, coords: MLXArray) -> MLXArray {
        let angle = matmul(coords, freqs)
        let cosAngle = MLX.cos(angle)
        let sinAngle = MLX.sin(angle)
        let parts = split(x, axis: -1)
        let rotated = concatenated([parts.1 * MLXArray(-1.0), parts.0], axis: -1)
        return (x * cosAngle) + (rotated * sinAngle)
    }
}

private func falconPerceptionWindowMaskChunk(
    queryStart: Int,
    chunkSize: Int,
    height: Int,
    width: Int,
    keyHeight: Int,
    keyWidth: Int,
    windowRatio: Float
) -> MLXArray {
    let queryIndices = MLXArray(Array(Int32(queryStart)..<Int32(queryStart + chunkSize)), [chunkSize]).asType(.float32)
    let queryRows = floor(queryIndices / MLXArray(Float(width)))
    let queryCols = queryIndices - (queryRows * MLXArray(Float(width)))

    let queryRowsNorm = (queryRows + MLXArray(0.5)) / MLXArray(Float(height))
    let queryColsNorm = (queryCols + MLXArray(0.5)) / MLXArray(Float(width))

    let rowLower = floor(clip(queryRowsNorm - MLXArray(windowRatio), min: 0.0, max: 1.0) * MLXArray(Float(keyHeight))).asType(.int32)
    let rowUpper = ceil(clip(queryRowsNorm + MLXArray(windowRatio), min: 0.0, max: 1.0) * MLXArray(Float(keyHeight))).asType(.int32)
    let colLower = floor(clip(queryColsNorm - MLXArray(windowRatio), min: 0.0, max: 1.0) * MLXArray(Float(keyWidth))).asType(.int32)
    let colUpper = ceil(clip(queryColsNorm + MLXArray(windowRatio), min: 0.0, max: 1.0) * MLXArray(Float(keyWidth))).asType(.int32)

    let keyRows = MLXArray(Array(Int32(0)..<Int32(keyHeight)), [keyHeight])
    let keyCols = MLXArray(Array(Int32(0)..<Int32(keyWidth)), [keyWidth])

    let rowMask = (keyRows.reshaped(1, keyHeight) .>= rowLower.reshaped(chunkSize, 1))
        .&& (keyRows.reshaped(1, keyHeight) .< rowUpper.reshaped(chunkSize, 1))
    let colMask = (keyCols.reshaped(1, keyWidth) .>= colLower.reshaped(chunkSize, 1))
        .&& (keyCols.reshaped(1, keyWidth) .< colUpper.reshaped(chunkSize, 1))

    return (rowMask.reshaped(chunkSize, keyHeight, 1) .&& colMask.reshaped(chunkSize, 1, keyWidth))
        .reshaped(chunkSize, keyHeight * keyWidth)
}

final class FalconPerceptionAnyUpCrossAttention: Module {
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(qkDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = max(1, qkDim / max(1, numHeads))
        self.scale = pow(Float(self.headDim), -0.5)
        self._normQ.wrappedValue = RMSNorm(dimensions: qkDim)
        self._normK.wrappedValue = RMSNorm(dimensions: qkDim)
        self._qProj.wrappedValue = Linear(qkDim, qkDim)
        self._kProj.wrappedValue = Linear(qkDim, qkDim)
        super.init()
    }

    func callAsFunction(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        height: Int? = nil,
        width: Int? = nil,
        keyHeight: Int? = nil,
        keyWidth: Int? = nil,
        windowRatio: Float = 0.1,
        chunkSize: Int = 4096
    ) -> MLXArray {
        let batch = query.dim(0)
        let queryLength = query.dim(1)
        let keyLength = key.dim(1)
        let valueDim = value.dim(2)
        let valueHeadDim = max(1, valueDim / max(1, numHeads))

        let q = qProj(normQ(query))
        let k = kProj(normK(key))
        let kHeads = k.reshaped(batch, keyLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let vHeads = value.reshaped(batch, keyLength, numHeads, valueHeadDim).transposed(0, 2, 1, 3)

        let usesWindow = height != nil && width != nil && keyHeight != nil && keyWidth != nil
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(max(1, (queryLength + chunkSize - 1) / chunkSize))

        for start in stride(from: 0, to: queryLength, by: chunkSize) {
            let currentChunk = min(chunkSize, queryLength - start)
            let qChunk = q[0..., start..<(start + currentChunk), 0...]
                .reshaped(batch, currentChunk, numHeads, headDim)
                .transposed(0, 2, 1, 3)

            var scores = matmul(qChunk, kHeads.transposed(0, 1, 3, 2)) * MLXArray(scale)
            if usesWindow {
                let mask = falconPerceptionWindowMaskChunk(
                    queryStart: start,
                    chunkSize: currentChunk,
                    height: height!,
                    width: width!,
                    keyHeight: keyHeight!,
                    keyWidth: keyWidth!,
                    windowRatio: windowRatio
                )
                scores = MLX.where(mask.reshaped(1, 1, currentChunk, keyLength), scores, MLXArray(-1e9))
            }

            let weights = softmax(scores, axis: -1)
            let outChunk = matmul(weights, vHeads)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, currentChunk, valueDim)
            outputs.append(outChunk)
        }

        return concatenated(outputs, axis: 1)
    }
}

final class FalconPerceptionAnyUpCrossDecodeBlock: Module {
    @ModuleInfo(key: "cross_attn") var crossAttn: FalconPerceptionAnyUpCrossAttention
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(qkDim: Int, numHeads: Int) {
        self._crossAttn.wrappedValue = FalconPerceptionAnyUpCrossAttention(qkDim: qkDim, numHeads: numHeads)
        self._conv.wrappedValue = Conv2d(
            inputChannels: qkDim,
            outputChannels: qkDim,
            kernelSize: 3,
            padding: 1,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ query: MLXArray, key: MLXArray, value: MLXArray, windowRatio: Float = 0.1) -> MLXArray {
        let batch = query.dim(0)
        let height = query.dim(1)
        let width = query.dim(2)
        let keyHeight = key.dim(1)
        let keyWidth = key.dim(2)

        let queryConv = conv(query)
        let queryFlat = queryConv.reshaped(batch, height * width, queryConv.dim(3))
        let keyFlat = key.reshaped(batch, keyHeight * keyWidth, key.dim(3))
        let valueFlat = value.reshaped(batch, keyHeight * keyWidth, value.dim(3))
        let decoded = crossAttn(
            query: queryFlat,
            key: keyFlat,
            value: valueFlat,
            height: height,
            width: width,
            keyHeight: keyHeight,
            keyWidth: keyWidth,
            windowRatio: windowRatio
        )
        return decoded.reshaped(batch, height, width, value.dim(3))
    }
}

final class FalconPerceptionAnyUp: Module {
    @ModuleInfo(key: "image_encoder") var imageEncoder: FalconPerceptionAnyUpEncoder
    @ModuleInfo(key: "key_encoder") var keyEncoder: FalconPerceptionAnyUpEncoder
    @ModuleInfo(key: "query_encoder") var queryEncoder: FalconPerceptionAnyUpEncoder
    @ModuleInfo(key: "key_features_encoder") var keyFeaturesEncoder: FalconPerceptionAnyUpLFUEncoder
    @ModuleInfo(key: "aggregation") var aggregation: FalconPerceptionAnyUpEncoder
    @ModuleInfo(key: "cross_decode") var crossDecode: FalconPerceptionAnyUpCrossDecodeBlock
    @ModuleInfo(key: "rope") var rope: FalconPerceptionAnyUpRoPE

    let qkDim: Int

    init(inputDim: Int = 3, qkDim: Int = 128, numHeads: Int = 4) {
        self.qkDim = qkDim
        self._imageEncoder.wrappedValue = FalconPerceptionAnyUpEncoder(
            inChannels: inputDim,
            outChannels: qkDim,
            kernelSize: 1,
            reflectPadding: true
        )
        self._keyEncoder.wrappedValue = FalconPerceptionAnyUpEncoder(
            inChannels: qkDim,
            outChannels: qkDim,
            kernelSize: 1,
            reflectPadding: true
        )
        self._queryEncoder.wrappedValue = FalconPerceptionAnyUpEncoder(
            inChannels: qkDim,
            outChannels: qkDim,
            kernelSize: 1,
            reflectPadding: true
        )
        self._keyFeaturesEncoder.wrappedValue = FalconPerceptionAnyUpLFUEncoder(qkDim: qkDim, kernelSizeLFU: 5)
        self._aggregation.wrappedValue = FalconPerceptionAnyUpEncoder(
            inChannels: 2 * qkDim,
            outChannels: qkDim,
            kernelSize: 3,
            reflectPadding: true
        )
        self._crossDecode.wrappedValue = FalconPerceptionAnyUpCrossDecodeBlock(qkDim: qkDim, numHeads: numHeads)
        self._rope.wrappedValue = FalconPerceptionAnyUpRoPE(dim: qkDim)
        super.init()
    }

    static func adaptiveAveragePool2D(_ x: MLXArray, outputSize: (Int, Int)) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)
        let channels = x.dim(3)
        let outputHeight = outputSize.0
        let outputWidth = outputSize.1

        if height == outputHeight && width == outputWidth {
            return x
        }
        if height % outputHeight == 0 && width % outputWidth == 0 {
            let kernelH = height / outputHeight
            let kernelW = width / outputWidth
            return mean(
                x.reshaped(batch, outputHeight, kernelH, outputWidth, kernelW, channels),
                axes: [2, 4]
            )
        }

        var rows: [MLXArray] = []
        rows.reserveCapacity(outputHeight)
        for row in 0..<outputHeight {
            let startH = (row * height) / outputHeight
            let endH = ((row + 1) * height) / outputHeight
            var columns: [MLXArray] = []
            columns.reserveCapacity(outputWidth)
            for column in 0..<outputWidth {
                let startW = (column * width) / outputWidth
                let endW = ((column + 1) * width) / outputWidth
                columns.append(mean(x[0..., startH..<endH, startW..<endW, 0...], axes: [1, 2], keepDims: true))
            }
            rows.append(concatenated(columns, axis: 2))
        }
        return concatenated(rows, axis: 1)
    }

    func callAsFunction(images: MLXArray, features: MLXArray) -> MLXArray {
        let batch = images.dim(0)
        let lowResHeight = features.dim(1)
        let lowResWidth = features.dim(2)

        let meanArray = MLXArray(falconPerceptionImageNetMean, [1, 1, 1, 3]).asType(features.dtype)
        let stdArray = MLXArray(falconPerceptionImageNetStd, [1, 1, 1, 3]).asType(features.dtype)
        var normalizedImage = (images * MLXArray(0.5)) + MLXArray(0.5)
        normalizedImage = (normalizedImage - meanArray) / stdArray
        normalizedImage = normalizedImage.asType(features.dtype)

        var encoded = imageEncoder(normalizedImage)
        let yCoords = MLXArray((0..<encoded.dim(1)).map { Float($0) / Float(max(1, encoded.dim(1) - 1)) }, [encoded.dim(1)])
        let xCoords = MLXArray((0..<encoded.dim(2)).map { Float($0) / Float(max(1, encoded.dim(2) - 1)) }, [encoded.dim(2)])
        let yy = broadcast(yCoords.reshaped(encoded.dim(1), 1), to: [encoded.dim(1), encoded.dim(2)])
        let xx = broadcast(xCoords.reshaped(1, encoded.dim(2)), to: [encoded.dim(1), encoded.dim(2)])
        let coords = stacked([yy.reshaped(-1), xx.reshaped(-1)], axis: -1).reshaped(1, encoded.dim(1) * encoded.dim(2), 2)

        let encodedFlat = encoded.reshaped(batch, encoded.dim(1) * encoded.dim(2), qkDim)
        encoded = rope(encodedFlat, coords: coords).reshaped(batch, encoded.dim(1), encoded.dim(2), qkDim)

        let query = queryEncoder(encoded)
        let pooledKey = FalconPerceptionAnyUp.adaptiveAveragePool2D(keyEncoder(encoded), outputSize: (lowResHeight, lowResWidth))

        let normDenominator = sqrt(clip(sum(features * features, axis: -1, keepDims: true), min: 1e-12))
        let normalizedFeatures = features / normDenominator
        let keyFeatures = keyFeaturesEncoder(normalizedFeatures)
        let aggregatedKey = aggregation(concatenated([pooledKey, keyFeatures], axis: -1))

        return crossDecode(query, key: aggregatedKey, value: features)
    }
}
