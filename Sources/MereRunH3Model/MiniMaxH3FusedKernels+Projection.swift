import MLX
import MLXFast

extension MiniMaxH3FusedKernels {
    package static func projectHeadMajorQKVAffineInt8(
        input: MLXArray,
        weightCodes: MLXArray,
        weightScales: MLXArray,
        weightBiases: MLXArray,
        queryNormWeight: MLXArray,
        keyNormWeight: MLXArray,
        ropeCosine: MLXArray,
        ropeSine: MLXArray,
        eps: Float
    ) -> MiniMaxH3HeadMajorQKV? {
        #if os(macOS) || os(iOS)
        let projectionWidth = 3 * attentionInnerDimension
        let scaleGroups = hiddenSize / affineGroupSize
        guard Device.defaultDevice().deviceType == .gpu,
              eps.isFinite,
              eps > 0,
              [.bfloat16, .float32].contains(input.dtype),
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) > 0,
              input.dim(2) == hiddenSize,
              weightCodes.dtype == .uint32,
              weightCodes.shape == [projectionWidth, hiddenSize / 4],
              weightScales.dtype == .bfloat16,
              weightScales.shape == [projectionWidth, scaleGroups],
              weightBiases.dtype == .bfloat16,
              weightBiases.shape == weightScales.shape,
              queryNormWeight.dtype == .bfloat16,
              queryNormWeight.shape == [attentionHeadDimension],
              keyNormWeight.dtype == .bfloat16,
              keyNormWeight.shape == [attentionHeadDimension],
              ropeCosine.dtype == ropeSine.dtype,
              [.bfloat16, .float32].contains(ropeCosine.dtype),
              ropeCosine.shape == [1, input.dim(1), 1, rotaryDimension],
              ropeSine.shape == ropeCosine.shape else {
            return nil
        }

        let rows = input.dim(1)
        let outputShape = [1, attentionHeadCount, rows, attentionHeadDimension]
        let usesLegacyBF16Contract = input.dtype == .bfloat16
            && ropeCosine.dtype == .bfloat16
        let projected = (input.dtype == .bfloat16
            ? projectHeadMajorQKVAffineInt8Kernel
            : projectHeadMajorQKVAffineInt8MixedKernel)(
                [input, weightCodes, weightScales, weightBiases],
                grid: ((projectionWidth / 8) * 64, rows, 1),
                threadGroup: (64, 1, 1),
                outputShapes: [outputShape, outputShape, outputShape],
                outputDTypes: [input.dtype, input.dtype, input.dtype]
            )
        if input.dtype == .bfloat16, ropeCosine.dtype == .float32 {
            func normalizeAndRotate(_ headMajor: MLXArray, weight: MLXArray) -> MLXArray {
                let rowMajor = headMajor.transposed(0, 2, 1, 3)
                let normalized = MLXFast.rmsNorm(rowMajor, weight: weight, eps: eps)
                let rotary = normalized[0..., 0..., 0..., 0..<rotaryDimension]
                let passthrough = normalized[0..., 0..., 0..., rotaryDimension...]
                let halves = MLX.split(rotary, parts: 2, axis: -1)
                let rotated = MLX.concatenated([-halves[1], halves[0]], axis: -1)
                return MLX.concatenated(
                    [rotary * ropeCosine + rotated * ropeSine, passthrough],
                    axis: -1
                ).transposed(0, 2, 1, 3).contiguous()
            }
            return MiniMaxH3HeadMajorQKV(
                query: normalizeAndRotate(projected[0], weight: queryNormWeight),
                key: normalizeAndRotate(projected[1], weight: keyNormWeight),
                value: projected[2]
            )
        }
        let normalizationKernel: MLXFast.MLXFastKernel
        let normalizationDType: DType
        if usesLegacyBF16Contract {
            normalizationKernel = normalizeHeadMajorQKVRoPEKernel
            normalizationDType = .bfloat16
        } else if input.dtype == .bfloat16 {
            normalizationKernel = normalizeHeadMajorQKVRoPEBF16ToFloatKernel
            normalizationDType = .float32
        } else {
            normalizationKernel = normalizeHeadMajorQKVRoPEFloatKernel
            normalizationDType = .float32
        }
        let normalized = normalizationKernel(
            [
                projected[0], projected[1], queryNormWeight, keyNormWeight,
                ropeCosine, ropeSine, eps,
            ],
            grid: (32, attentionHeadCount, rows),
            threadGroup: (32, 1, 1),
            outputShapes: [outputShape, outputShape],
            outputDTypes: [normalizationDType, normalizationDType]
        )
        return MiniMaxH3HeadMajorQKV(
            query: normalized[0],
            key: normalized[1],
            value: projected[2]
        )
        #else
        return nil
        #endif
    }

    /// K3 consumes MLX SDPA's head-major BF16 output without materializing the
    /// row-major transpose/reshape and applies the managed checkpoint's affine
    /// Q8/group-64 output projection directly. This preserves the released
    /// artifact's weight arithmetic; it does not introduce activation INT8.
    package static func projectHeadMajorAttentionAffineInt8(
        attention: MLXArray,
        weightCodes: MLXArray,
        weightScales: MLXArray,
        weightBiases: MLXArray
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        let scaleGroups = attentionInnerDimension / affineGroupSize
        guard Device.defaultDevice().deviceType == .gpu,
              [.bfloat16, .float32].contains(attention.dtype),
              attention.ndim == 4,
              attention.dim(0) == 1,
              attention.dim(1) == attentionHeadCount,
              attention.dim(2) > 0,
              attention.dim(3) == attentionHeadDimension,
              weightCodes.dtype == .uint32,
              weightCodes.shape == [hiddenSize, attentionInnerDimension / 4],
              weightScales.dtype == .bfloat16,
              weightScales.shape == [hiddenSize, scaleGroups],
              weightBiases.dtype == .bfloat16,
              weightBiases.shape == weightScales.shape else {
            return nil
        }

        let rows = attention.dim(2)
        return projectHeadMajorAttentionAffineInt8Kernel(
            [attention, weightCodes, weightScales, weightBiases],
            template: [("T", attention.dtype)],
            grid: ((hiddenSize / 8) * 64, rows, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, rows, hiddenSize]],
            outputDTypes: [attention.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// K4a applies the managed affine Q8/group-64 FC1 weights and folds the
    /// immediately following SwiGLU into the projection dispatch. The two FC1
    /// output slabs are reduced independently and rounded to BF16 before the
    /// elementwise activation, matching the current graph's QMM boundary while
    /// avoiding the `[1, rows, 28672]` materialization.
    package static func projectFeedForwardInputAffineInt8SwiGLU(
        input: MLXArray,
        weightCodes: MLXArray,
        weightScales: MLXArray,
        weightBiases: MLXArray
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        let scaleGroups = hiddenSize / affineGroupSize
        guard Device.defaultDevice().deviceType == .gpu,
              [.bfloat16, .float32].contains(input.dtype),
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) > 0,
              input.dim(2) == hiddenSize,
              weightCodes.dtype == .uint32,
              weightCodes.shape == [2 * feedForwardSize, hiddenSize / 4],
              weightScales.dtype == .bfloat16,
              weightScales.shape == [2 * feedForwardSize, scaleGroups],
              weightBiases.dtype == .bfloat16,
              weightBiases.shape == weightScales.shape else {
            return nil
        }

        let rows = input.dim(1)
        let kernel = input.dtype == .bfloat16
            ? projectFeedForwardInputAffineInt8SwiGLUKernel
            : projectFeedForwardInputAffineInt8SwiGLUFloatKernel
        return kernel(
            [input, weightCodes, weightScales, weightBiases],
            grid: ((feedForwardSize / 8) * 64, rows, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, rows, feedForwardSize]],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// Matrix-tiled K4a candidate that keeps the gate and up projections in
    /// registers through the SwiGLU epilogue. Matrix operands and outputs keep
    /// the residual stream's BF16 or Float32 type, with Float32 accumulation.
    package static func projectFeedForwardInputAffineInt8SwiGLUTiled(
        input: MLXArray,
        weightCodes: MLXArray,
        weightScales: MLXArray,
        weightBiases: MLXArray
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        let scaleGroups = hiddenSize / affineGroupSize
        guard Device.defaultDevice().deviceType == .gpu,
              [.bfloat16, .float32].contains(input.dtype),
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) > 0,
              input.dim(2) == hiddenSize,
              weightCodes.dtype == .uint32,
              weightCodes.shape == [2 * feedForwardSize, hiddenSize / 4],
              weightScales.dtype == .bfloat16,
              weightScales.shape == [2 * feedForwardSize, scaleGroups],
              weightBiases.dtype == .bfloat16,
              weightBiases.shape == weightScales.shape else {
            return nil
        }

        let rows = input.dim(1)
        let outputTileCount = feedForwardSize / 32
        let rowTileCount = (rows + 31) / 32
        return projectFeedForwardInputAffineInt8SwiGLUTiledKernel(
            [input, weightCodes, weightScales, weightBiases],
            template: [("T", input.dtype)],
            grid: (32 * outputTileCount, 4 * rowTileCount, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, rows, feedForwardSize]],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// K4b applies H3's exact 14336 -> 5376 affine Q8/group-64 FC2 without
    /// relying on a generic shape selection path. It consumes K4a's compact
    /// SwiGLU result directly and preserves the managed artifact arithmetic.
    package static func projectFeedForwardOutputAffineInt8(
        input: MLXArray,
        weightCodes: MLXArray,
        weightScales: MLXArray,
        weightBiases: MLXArray
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        let scaleGroups = feedForwardSize / affineGroupSize
        guard Device.defaultDevice().deviceType == .gpu,
              [.bfloat16, .float32].contains(input.dtype),
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) > 0,
              input.dim(2) == feedForwardSize,
              weightCodes.dtype == .uint32,
              weightCodes.shape == [hiddenSize, feedForwardSize / 4],
              weightScales.dtype == .bfloat16,
              weightScales.shape == [hiddenSize, scaleGroups],
              weightBiases.dtype == .bfloat16,
              weightBiases.shape == weightScales.shape else {
            return nil
        }

        let rows = input.dim(1)
        let kernel = input.dtype == .bfloat16
            ? projectFeedForwardOutputAffineInt8Kernel
            : projectFeedForwardOutputAffineInt8FloatKernel
        return kernel(
            [input, weightCodes, weightScales, weightBiases],
            grid: ((hiddenSize / 8) * 64, rows, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, rows, hiddenSize]],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// SIMD-group matrix candidate for K4b. Each workgroup dequantizes one
    /// 64-output by 32-input weight tile once, shares a 32-row activation
    /// tile, and accumulates eight 8x8 matrix products per SIMD group.
    /// The managed FastH3 Q8 recipe selects this after realistic-shape timing
    /// and installed-checkpoint validation.
    package static func projectFeedForwardOutputAffineInt8Tiled(
        input: MLXArray,
        weightCodes: MLXArray,
        weightScales: MLXArray,
        weightBiases: MLXArray
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        let scaleGroups = feedForwardSize / affineGroupSize
        guard Device.defaultDevice().deviceType == .gpu,
              [.bfloat16, .float32].contains(input.dtype),
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) > 0,
              input.dim(2) == feedForwardSize,
              weightCodes.dtype == .uint32,
              weightCodes.shape == [hiddenSize, feedForwardSize / 4],
              weightScales.dtype == .bfloat16,
              weightScales.shape == [hiddenSize, scaleGroups],
              weightBiases.dtype == .bfloat16,
              weightBiases.shape == weightScales.shape else {
            return nil
        }

        let rows = input.dim(1)
        let outputTileCount = hiddenSize / 64
        let rowTileCount = (rows + 31) / 32
        return projectFeedForwardOutputAffineInt8TiledKernel(
            [input, weightCodes, weightScales, weightBiases],
            template: [("T", input.dtype)],
            grid: (32 * outputTileCount, 4 * rowTileCount, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, rows, hiddenSize]],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }

}
