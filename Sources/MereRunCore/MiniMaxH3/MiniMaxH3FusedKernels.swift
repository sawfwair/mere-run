import MLX
import MLXFast

struct MiniMaxH3GateAdaLNOutput {
    let residual: MLXArray
    let feedForwardInput: MLXArray
    let feedForwardGate: MLXArray
}

struct MiniMaxH3DynamicInt8Rows {
    let values: MLXArray
    let scales: MLXArray
}

struct MiniMaxH3GateAdaLNQuantizedOutput {
    let residual: MLXArray
    let feedForwardInput: MiniMaxH3DynamicInt8Rows
}

struct MiniMaxH3HeadMajorQKV {
    let query: MLXArray
    let key: MLXArray
    let value: MLXArray
}

/// Exact-shape Metal experiments for MiniMax-H3's 5,376-wide DiT blocks.
/// Production keeps the decomposed MLX graph until a fused candidate passes
/// tensor parity, realistic-shape timing, and an installed-checkpoint A/B.
enum MiniMaxH3FusedKernels {
    private static let hiddenSize = 5_376
    private static let modulationPartCount = 6
    private static let threadCount = 256
    private static let attentionHeadCount = 56
    private static let attentionHeadDimension = 128
    private static let attentionInnerDimension = attentionHeadCount * attentionHeadDimension
    private static let rotaryDimension = 96
    private static let affineGroupSize = 64
    private static let feedForwardSize = 14_336

    /// Fuses the first H3 block boundary: RMSNorm followed by the row-indexed
    /// attention scale and shift. This is the Metal counterpart of FastVideo's
    /// `fused_rmsnorm_modulate` Triton kernel.
    static func prepareAttentionInput(
        input: MLXArray,
        normWeight: MLXArray,
        modulation: MLXArray,
        rowIndices: MLXArray,
        eps: Float
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        guard eps.isFinite,
              eps > 0,
              supportsAdaLNShapeContract(
                  input: input,
                  normWeight: normWeight,
                  modulation: modulation,
                  rowIndices: rowIndices
              ) else {
            return nil
        }

        let kernel = input.dtype == .bfloat16
            ? attentionAdaLNKernel
            : attentionAdaLNMixedKernel
        return kernel(
            [input, normWeight, modulation, rowIndices, eps],
            grid: (threadCount, input.dim(1), 1),
            threadGroup: (threadCount, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    static func gateAttentionAndPrepareFeedForward(
        residual: MLXArray,
        attentionOutput: MLXArray,
        normWeight: MLXArray,
        modulation: MLXArray,
        rowIndices: MLXArray,
        eps: Float
    ) -> MiniMaxH3GateAdaLNOutput? {
        #if os(macOS) || os(iOS)
        guard eps.isFinite,
              eps > 0,
              supportsGateAdaLNShapeContract(
                  residual: residual,
                  branch: attentionOutput,
                  normWeight: normWeight,
                  modulation: modulation,
                  rowIndices: rowIndices
              ) else {
            return nil
        }

        let inputs: [any ScalarOrArray] = [
            residual, attentionOutput, normWeight, modulation, rowIndices, eps,
        ]
        let outputs: [MLXArray]
        if residual.dtype == .bfloat16, attentionOutput.dtype == .bfloat16 {
            outputs = gateAdaLNKernel(
                inputs,
                grid: (threadCount, residual.dim(1), 1),
                threadGroup: (threadCount, 1, 1),
                outputShapes: [residual.shape, residual.shape, residual.shape],
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
            )
        } else if [.bfloat16, .float32].contains(residual.dtype),
                  attentionOutput.dtype == .float32 {
            outputs = gateAdaLNMixedKernel(
                inputs,
                grid: (threadCount, residual.dim(1), 1),
                threadGroup: (threadCount, 1, 1),
                outputShapes: [residual.shape, residual.shape, residual.shape],
                outputDTypes: [.float32, .float32, .bfloat16]
            )
        } else {
            return nil
        }
        return MiniMaxH3GateAdaLNOutput(
            residual: outputs[0],
            feedForwardInput: outputs[1],
            feedForwardGate: outputs[2]
        )
        #else
        return nil
        #endif
    }

    /// Fuses the K1 boundary through h3.c-compatible per-row symmetric INT8
    /// activation quantization. Stock MLX `QuantizedLinear` still accepts a
    /// floating activation, so production dispatch must not select this until
    /// a matching INT8 activation x weight projection is available.
    static func gateAttentionAndQuantizeFeedForward(
        residual: MLXArray,
        attentionOutput: MLXArray,
        normWeight: MLXArray,
        modulation: MLXArray,
        rowIndices: MLXArray,
        eps: Float
    ) -> MiniMaxH3GateAdaLNQuantizedOutput? {
        #if os(macOS) || os(iOS)
        guard eps.isFinite,
              eps > 0,
              supportsGateAdaLNInputs(
                  residual: residual,
                  branch: attentionOutput,
                  normWeight: normWeight,
                  modulation: modulation,
                  rowIndices: rowIndices
              ) else {
            return nil
        }

        let rows = residual.dim(1)
        let outputs = gateAdaLNQuantizeKernel(
            [residual, attentionOutput, normWeight, modulation, rowIndices, eps],
            grid: (threadCount, rows, 1),
            threadGroup: (threadCount, 1, 1),
            outputShapes: [residual.shape, residual.shape, [rows]],
            outputDTypes: [.bfloat16, .int8, .float32]
        )
        return MiniMaxH3GateAdaLNQuantizedOutput(
            residual: outputs[0],
            feedForwardInput: MiniMaxH3DynamicInt8Rows(
                values: outputs[1],
                scales: outputs[2]
            )
        )
        #else
        return nil
        #endif
    }

    /// Standalone oracle for the activation quantizer that h3.c otherwise
    /// folds into gate/AdaLN. It is also the unfused release-benchmark arm.
    static func quantizeRowsSymmetricInt8(_ input: MLXArray) -> MiniMaxH3DynamicInt8Rows? {
        #if os(macOS) || os(iOS)
        guard Device.defaultDevice().deviceType == .gpu,
              input.dtype == .bfloat16,
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) > 0,
              input.dim(2) == hiddenSize else {
            return nil
        }

        let rows = input.dim(1)
        let outputs = quantizeRowsKernel(
            [input],
            grid: (threadCount, rows, 1),
            threadGroup: (threadCount, 1, 1),
            outputShapes: [input.shape, [rows]],
            outputDTypes: [.int8, .float32]
        )
        return MiniMaxH3DynamicInt8Rows(values: outputs[0], scales: outputs[1])
        #else
        return nil
        #endif
    }

    /// K2a consumes the released checkpoint's three global Q/K/V slabs and
    /// replaces split, reshape, Q/K RMSNorm, partial RoPE, transpose, and
    /// contiguous copies with one head-major write. K2b will move this output
    /// contract into the projection itself.
    static func prepareHeadMajorQKV(
        projected: MLXArray,
        queryNormWeight: MLXArray,
        keyNormWeight: MLXArray,
        ropeCosine: MLXArray,
        ropeSine: MLXArray,
        eps: Float
    ) -> MiniMaxH3HeadMajorQKV? {
        #if os(macOS) || os(iOS)
        guard Device.defaultDevice().deviceType == .gpu,
              eps.isFinite,
              eps > 0,
              [.bfloat16, .float32].contains(projected.dtype),
              projected.ndim == 3,
              projected.dim(0) == 1,
              projected.dim(1) > 0,
              projected.dim(2) == 3 * attentionInnerDimension,
              queryNormWeight.dtype == .bfloat16,
              queryNormWeight.shape == [attentionHeadDimension],
              keyNormWeight.dtype == .bfloat16,
              keyNormWeight.shape == [attentionHeadDimension],
              [.bfloat16, .float32].contains(ropeCosine.dtype),
              ropeSine.dtype == ropeCosine.dtype,
              ropeCosine.shape == [1, projected.dim(1), 1, rotaryDimension],
              ropeSine.shape == ropeCosine.shape else {
            return nil
        }

        let rows = projected.dim(1)
        let outputShape = [1, attentionHeadCount, rows, attentionHeadDimension]
        let queryKeyDType: DType = projected.dtype == .float32
            || ropeCosine.dtype == .float32
            ? .float32
            : .bfloat16
        let outputs = prepareHeadMajorQKVKernel(
            [projected, queryNormWeight, keyNormWeight, ropeCosine, ropeSine, eps],
            template: [("T", projected.dtype), ("Q", queryKeyDType)],
            grid: (32, attentionHeadCount, rows),
            threadGroup: (32, 1, 1),
            outputShapes: [outputShape, outputShape, outputShape],
            outputDTypes: [queryKeyDType, queryKeyDType, projected.dtype]
        )
        return MiniMaxH3HeadMajorQKV(
            query: outputs[0],
            key: outputs[1],
            value: outputs[2]
        )
        #else
        return nil
        #endif
    }

    /// K2b applies the managed checkpoint's affine Q8/group-64 QKV weights
    /// directly into three head-major tensors, then fuses Q/K RMSNorm and RoPE
    /// without ever materializing the `[1, rows, 21504]` global projection
    /// slab. The raw head-major Q/K tensors remain explicit intermediates
    /// because standalone MLXFast Metal outputs cannot donate their storage;
    /// V flows directly from the projection kernel into SDPA.
    static func projectHeadMajorQKVAffineInt8(
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
    static func projectHeadMajorAttentionAffineInt8(
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
    static func projectFeedForwardInputAffineInt8SwiGLU(
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
    static func projectFeedForwardInputAffineInt8SwiGLUTiled(
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
    static func projectFeedForwardOutputAffineInt8(
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
    static func projectFeedForwardOutputAffineInt8Tiled(
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

    #if os(macOS) || os(iOS)
    private static func supportsAdaLNShapeContract(
        input: MLXArray,
        normWeight: MLXArray,
        modulation: MLXArray,
        rowIndices: MLXArray
    ) -> Bool {
        Device.defaultDevice().deviceType == .gpu
            && [.bfloat16, .float32].contains(input.dtype)
            && input.ndim == 3
            && input.dim(0) == 1
            && input.dim(1) > 0
            && input.dim(2) == hiddenSize
            && normWeight.dtype == .bfloat16
            && normWeight.shape == [hiddenSize]
            && modulation.dtype == .bfloat16
            && modulation.ndim == 2
            && modulation.dim(0) > 0
            && modulation.dim(1) == modulationPartCount * hiddenSize
            && rowIndices.dtype == .int32
            && rowIndices.shape == [input.dim(1)]
    }

    private static func supportsGateAdaLNShapeContract(
        residual: MLXArray,
        branch: MLXArray,
        normWeight: MLXArray,
        modulation: MLXArray,
        rowIndices: MLXArray
    ) -> Bool {
        residual.shape == branch.shape
            && supportsAdaLNShapeContract(
                input: residual,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices
            )
    }

    private static func supportsGateAdaLNInputs(
        residual: MLXArray,
        branch: MLXArray,
        normWeight: MLXArray,
        modulation: MLXArray,
        rowIndices: MLXArray
    ) -> Bool {
        supportsGateAdaLNShapeContract(
            residual: residual,
            branch: branch,
            normWeight: normWeight,
            modulation: modulation,
            rowIndices: rowIndices
        )
            && residual.dtype == .bfloat16
            && branch.dtype == .bfloat16
    }

    private static let attentionAdaLNKernel = MLXFast.metalKernel(
        name: "mere_h3_attention_adaln_bf16_h5376_v1",
        inputNames: [
            "input", "norm_weight", "modulation", "row_indices", "epsilon",
        ],
        outputNames: ["output"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float value = float(input[row_offset + dimension]);
                sum += value * value;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                bfloat16_t normalized = bfloat16_t(
                    float(input[row_offset + dimension]) * inverse_rms[0]);
                bfloat16_t weighted = bfloat16_t(
                    float(norm_weight[dimension]) * float(normalized));
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + hidden_size + dimension]));
                bfloat16_t scaled = bfloat16_t(
                    float(weighted) * float(one_plus_scale));
                output[row_offset + dimension] = bfloat16_t(
                    float(scaled) + float(modulation[
                        modulation_offset + dimension]));
            }
        """,
        ensureRowContiguous: true
    )

    private static let attentionAdaLNMixedKernel = MLXFast.metalKernel(
        name: "mere_h3_attention_adaln_mixed_h5376_v1",
        inputNames: [
            "input", "norm_weight", "modulation", "row_indices", "epsilon",
        ],
        outputNames: ["output"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float value = float(input[row_offset + dimension]);
                sum += value * value;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float weighted = float(input[row_offset + dimension])
                    * inverse_rms[0] * float(norm_weight[dimension]);
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + hidden_size + dimension]));
                output[row_offset + dimension] = weighted
                    * float(one_plus_scale) + float(modulation[
                        modulation_offset + dimension]);
            }
        """,
        ensureRowContiguous: true
    )

    private static let gateAdaLNKernel = MLXFast.metalKernel(
        name: "mere_h3_gate_adaln_bf16_h5376_v1",
        inputNames: [
            "residual", "attention_output", "norm_weight", "modulation",
            "row_indices", "epsilon",
        ],
        outputNames: ["residual_out", "feed_forward_input", "feed_forward_gate"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup bfloat16_t rounded_residual[hidden_size];
            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float gate = float(modulation[
                    modulation_offset + 2 * hidden_size + dimension]);
                bfloat16_t gated_attention = bfloat16_t(
                    gate * float(attention_output[row_offset + dimension]));
                bfloat16_t value = bfloat16_t(
                    float(residual[row_offset + dimension])
                    + float(gated_attention));
                rounded_residual[dimension] = value;
                residual_out[row_offset + dimension] = value;
                float widened = float(value);
                sum += widened * widened;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                bfloat16_t normalized = bfloat16_t(
                    float(rounded_residual[dimension]) * inverse_rms[0]);
                bfloat16_t weighted = bfloat16_t(
                    float(norm_weight[dimension]) * float(normalized));
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + 4 * hidden_size + dimension]));
                bfloat16_t scaled = bfloat16_t(
                    float(weighted) * float(one_plus_scale));
                feed_forward_input[row_offset + dimension] = bfloat16_t(
                    float(scaled) + float(modulation[
                        modulation_offset + 3 * hidden_size + dimension]));
                feed_forward_gate[row_offset + dimension] = modulation[
                    modulation_offset + 5 * hidden_size + dimension];
            }
        """,
        ensureRowContiguous: true
    )

    private static let gateAdaLNMixedKernel = MLXFast.metalKernel(
        name: "mere_h3_gate_adaln_mixed_h5376_v1",
        inputNames: [
            "residual", "attention_output", "norm_weight", "modulation",
            "row_indices", "epsilon",
        ],
        outputNames: ["residual_out", "feed_forward_input", "feed_forward_gate"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup float attended[hidden_size];
            threadgroup float partial_sums[simd_width];
            threadgroup float inverse_rms[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float gate = float(modulation[
                    modulation_offset + 2 * hidden_size + dimension]);
                float value = float(residual[row_offset + dimension])
                    + gate * float(attention_output[row_offset + dimension]);
                attended[dimension] = value;
                residual_out[row_offset + dimension] = value;
                sum += value * value;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_sums[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_sums[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float weighted = attended[dimension] * inverse_rms[0]
                    * float(norm_weight[dimension]);
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + 4 * hidden_size + dimension]));
                feed_forward_input[row_offset + dimension] = weighted
                    * float(one_plus_scale) + float(modulation[
                        modulation_offset + 3 * hidden_size + dimension]);
                feed_forward_gate[row_offset + dimension] = modulation[
                    modulation_offset + 5 * hidden_size + dimension];
            }
        """,
        ensureRowContiguous: true
    )

    private static let gateAdaLNQuantizeKernel = MLXFast.metalKernel(
        name: "mere_h3_gate_adaln_quantize_i8_h5376_v1",
        inputNames: [
            "residual", "attention_output", "norm_weight", "modulation",
            "row_indices", "epsilon",
        ],
        outputNames: ["residual_out", "quantized_input", "quantized_scales"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint modulation_parts = 6;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            uint modulation_offset = uint(row_indices[row])
                * modulation_parts * hidden_size;

            threadgroup bfloat16_t rounded_values[hidden_size];
            threadgroup float partial_values[simd_width];
            threadgroup float shared_value[1];

            float sum = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                float gate = float(modulation[
                    modulation_offset + 2 * hidden_size + dimension]);
                bfloat16_t gated_attention = bfloat16_t(
                    gate * float(attention_output[row_offset + dimension]));
                bfloat16_t value = bfloat16_t(
                    float(residual[row_offset + dimension])
                    + float(gated_attention));
                rounded_values[dimension] = value;
                residual_out[row_offset + dimension] = value;
                float widened = float(value);
                sum += widened * widened;
            }

            sum = simd_sum(sum);
            if (simd_group == 0) {
                partial_values[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_values[simd_group] = sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partial_values[simd_lane]);
                if (simd_lane == 0) {
                    shared_value[0] = metal::precise::rsqrt(
                        sum / float(hidden_size) + float(epsilon));
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            float local_max = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                bfloat16_t normalized = bfloat16_t(
                    float(rounded_values[dimension]) * shared_value[0]);
                bfloat16_t weighted = bfloat16_t(
                    float(norm_weight[dimension]) * float(normalized));
                bfloat16_t one_plus_scale = bfloat16_t(
                    1.0f + float(modulation[
                        modulation_offset + 4 * hidden_size + dimension]));
                bfloat16_t scaled = bfloat16_t(
                    float(weighted) * float(one_plus_scale));
                bfloat16_t value = bfloat16_t(
                    float(scaled) + float(modulation[
                        modulation_offset + 3 * hidden_size + dimension]));
                rounded_values[dimension] = value;
                local_max = metal::max(local_max, metal::fabs(float(value)));
            }

            local_max = simd_max(local_max);
            if (simd_group == 0) {
                partial_values[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_values[simd_group] = local_max;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                local_max = simd_max(partial_values[simd_lane]);
                if (simd_lane == 0) {
                    shared_value[0] = local_max;
                    quantized_scales[row] = local_max > 0.0f
                        ? local_max / 127.0f
                        : 1.0f / 127.0f;
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            float quantize_scale = shared_value[0] > 0.0f
                ? 127.0f / shared_value[0]
                : 127.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                int value = int(rint(float(rounded_values[dimension]) * quantize_scale));
                quantized_input[row_offset + dimension] = int8_t(
                    metal::clamp(value, -127, 127));
            }
        """,
        ensureRowContiguous: true
    )

    private static let quantizeRowsKernel = MLXFast.metalKernel(
        name: "mere_h3_quantize_rows_i8_h5376_v1",
        inputNames: ["input"],
        outputNames: ["quantized", "scales"],
        source: """
            constexpr uint hidden_size = 5376;
            constexpr uint simd_width = 32;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * hidden_size;
            threadgroup float partial_maxima[simd_width];
            threadgroup float maximum[1];

            float local_max = 0.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                local_max = metal::max(
                    local_max,
                    metal::fabs(float(input[row_offset + dimension])));
            }
            local_max = simd_max(local_max);
            if (simd_group == 0) {
                partial_maxima[simd_lane] = 0.0f;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_lane == 0) {
                partial_maxima[simd_group] = local_max;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                local_max = simd_max(partial_maxima[simd_lane]);
                if (simd_lane == 0) {
                    maximum[0] = local_max;
                    scales[row] = local_max > 0.0f
                        ? local_max / 127.0f
                        : 1.0f / 127.0f;
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            float quantize_scale = maximum[0] > 0.0f
                ? 127.0f / maximum[0]
                : 127.0f;
            for (uint dimension = tid; dimension < hidden_size;
                 dimension += threads_per_threadgroup.x) {
                int value = int(rint(float(input[row_offset + dimension]) * quantize_scale));
                quantized[row_offset + dimension] = int8_t(
                    metal::clamp(value, -127, 127));
            }
        """,
        ensureRowContiguous: true
    )

    private static let prepareHeadMajorQKVKernel = MLXFast.metalKernel(
        name: "mere_h3_qkv_norm_rope_head_major_mixed_v2",
        inputNames: [
            "projected", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query", "key", "value"],
        source: """
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint inner_dimension = heads * head_dimension;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(projected_shape[1]);
            uint projected_row = row * 3 * inner_dimension;
            uint query_base = projected_row + head * head_dimension;
            uint key_base = query_base + inner_dimension;
            uint value_base = key_base + inner_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = float(projected[query_base + dimension]);
                float key_element = float(projected[key_base + dimension]);
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint output_base = (head * rows + row) * head_dimension;
            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                T query_normalized = T(
                    float(projected[query_base + dimension]) * query_inverse);
                T key_normalized = T(
                    float(projected[key_base + dimension]) * key_inverse);
                T query_weighted = T(
                    float(query_normalized) * float(query_norm_weight[dimension]));
                T key_weighted = T(
                    float(key_normalized) * float(key_norm_weight[dimension]));
                float query_output = float(query_weighted);
                float key_output = float(key_weighted);

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    T query_pair_normalized = T(
                        float(projected[query_base + pair]) * query_inverse);
                    T key_pair_normalized = T(
                        float(projected[key_base + pair]) * key_inverse);
                    T query_pair = T(
                        float(query_pair_normalized) * float(query_norm_weight[pair]));
                    T key_pair = T(
                        float(key_pair_normalized) * float(key_norm_weight[pair]));
                    float cosine = float(rope_cosine[rope_base + dimension]);
                    float sine = float(rope_sine[rope_base + dimension]);
                    if (dimension < rotary_half) {
                        query_output = float(query_weighted) * cosine
                            - float(query_pair) * sine;
                        key_output = float(key_weighted) * cosine
                            - float(key_pair) * sine;
                    } else {
                        query_output = float(query_weighted) * cosine
                            + float(query_pair) * sine;
                        key_output = float(key_weighted) * cosine
                            + float(key_pair) * sine;
                    }
                }

                query[output_base + dimension] = Q(query_output);
                key[output_base + dimension] = Q(key_output);
                value[output_base + dimension] = T(
                    projected[value_base + dimension]);
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectHeadMajorQKVAffineInt8Kernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qkv_projection_i8g64_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["query", "key", "value"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint inner_dimension = heads * head_dimension;
            constexpr uint projection_width = 3 * inner_dimension;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint rows = uint(input_shape[1]);
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device bfloat16_t* scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* values = input
                + row * input_width + lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value_element = float(values[index]);
                    input_values[index] = value_element;
                    input_sum += value_element;
                }

                for (uint output_index = 0;
                     output_index < results_per_simdgroup;
                     ++output_index) {
                    const device uint8_t* row_codes = codes
                        + output_index * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output_index] += float(
                        scales[output_index * scale_groups]) * dot
                        + float(biases[output_index * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                values += block_size;
            }

            for (uint output_index = 0;
                 output_index < results_per_simdgroup;
                 ++output_index) {
                float projected = simd_sum(result[output_index]);
                if (lane == 0) {
                    uint global_dimension = output_row + output_index;
                    uint slab = global_dimension / inner_dimension;
                    uint head_dimension_index = global_dimension
                        - slab * inner_dimension;
                    uint head = head_dimension_index / head_dimension;
                    uint dimension = head_dimension_index - head * head_dimension;
                    uint destination = (head * rows + row) * head_dimension + dimension;
                    if (slab == 0) {
                        query[destination] = bfloat16_t(projected);
                    } else if (slab == 1) {
                        key[destination] = bfloat16_t(projected);
                    } else {
                        value[destination] = bfloat16_t(projected);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectHeadMajorQKVAffineInt8MixedKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qkv_projection_i8g64_mixed_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["query", "key", "value"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint inner_dimension = heads * head_dimension;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint rows = uint(input_shape[1]);
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device bfloat16_t* scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            uint input_column = lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value_element = float(input[
                        row * input_width + input_column + index]);
                    input_values[index] = value_element;
                    input_sum += value_element;
                }

                for (uint output_index = 0;
                     output_index < results_per_simdgroup;
                     ++output_index) {
                    const device uint8_t* row_codes = codes
                        + output_index * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output_index] += float(
                        scales[output_index * scale_groups]) * dot
                        + float(biases[output_index * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                input_column += block_size;
            }

            for (uint output_index = 0;
                 output_index < results_per_simdgroup;
                 ++output_index) {
                float projected = simd_sum(result[output_index]);
                if (lane == 0) {
                    uint global_dimension = output_row + output_index;
                    uint slab = global_dimension / inner_dimension;
                    uint head_dimension_index = global_dimension
                        - slab * inner_dimension;
                    uint head = head_dimension_index / head_dimension;
                    uint dimension = head_dimension_index - head * head_dimension;
                    uint destination = (head * rows + row) * head_dimension + dimension;
                    if (slab == 0) {
                        query[destination] = projected;
                    } else if (slab == 1) {
                        key[destination] = projected;
                    } else {
                        value[destination] = projected;
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let normalizeHeadMajorQKVRoPEKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qk_norm_rope_bf16_v1",
        inputNames: [
            "query_input", "key_input", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query_output", "key_output"],
        source: """
            constexpr uint head_dimension = 128;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(query_input_shape[2]);
            uint input_base = (head * rows + row) * head_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = float(query_input[input_base + dimension]);
                float key_element = float(key_input[input_base + dimension]);
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                bfloat16_t query_normalized = bfloat16_t(
                    float(query_input[input_base + dimension]) * query_inverse);
                bfloat16_t key_normalized = bfloat16_t(
                    float(key_input[input_base + dimension]) * key_inverse);
                bfloat16_t query_weighted = bfloat16_t(
                    float(query_normalized) * float(query_norm_weight[dimension]));
                bfloat16_t key_weighted = bfloat16_t(
                    float(key_normalized) * float(key_norm_weight[dimension]));
                float query_value = float(query_weighted);
                float key_value = float(key_weighted);

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    bfloat16_t query_pair_normalized = bfloat16_t(
                        float(query_input[input_base + pair]) * query_inverse);
                    bfloat16_t key_pair_normalized = bfloat16_t(
                        float(key_input[input_base + pair]) * key_inverse);
                    bfloat16_t query_pair = bfloat16_t(
                        float(query_pair_normalized) * float(query_norm_weight[pair]));
                    bfloat16_t key_pair = bfloat16_t(
                        float(key_pair_normalized) * float(key_norm_weight[pair]));
                    float cosine = float(rope_cosine[rope_base + dimension]);
                    float sine = float(rope_sine[rope_base + dimension]);
                    if (dimension < rotary_half) {
                        query_value = float(query_weighted) * cosine
                            - float(query_pair) * sine;
                        key_value = float(key_weighted) * cosine
                            - float(key_pair) * sine;
                    } else {
                        query_value = float(query_weighted) * cosine
                            + float(query_pair) * sine;
                        key_value = float(key_weighted) * cosine
                            + float(key_pair) * sine;
                    }
                }

                query_output[input_base + dimension] = bfloat16_t(query_value);
                key_output[input_base + dimension] = bfloat16_t(key_value);
            }
        """,
        ensureRowContiguous: true
    )

    private static let normalizeHeadMajorQKVRoPEBF16ToFloatKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qk_norm_rope_bf16_f32_v1",
        inputNames: [
            "query_input", "key_input", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query_output", "key_output"],
        source: """
            constexpr uint head_dimension = 128;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(query_input_shape[2]);
            uint input_base = (head * rows + row) * head_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = float(query_input[input_base + dimension]);
                float key_element = float(key_input[input_base + dimension]);
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                bfloat16_t query_weighted = bfloat16_t(
                    float(query_input[input_base + dimension]) * query_inverse
                        * float(query_norm_weight[dimension]));
                bfloat16_t key_weighted = bfloat16_t(
                    float(key_input[input_base + dimension]) * key_inverse
                        * float(key_norm_weight[dimension]));
                float query_value = float(query_weighted);
                float key_value = float(key_weighted);

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    bfloat16_t query_pair = bfloat16_t(
                        float(query_input[input_base + pair]) * query_inverse
                            * float(query_norm_weight[pair]));
                    bfloat16_t key_pair = bfloat16_t(
                        float(key_input[input_base + pair]) * key_inverse
                            * float(key_norm_weight[pair]));
                    float cosine = float(rope_cosine[rope_base + dimension]);
                    float sine = float(rope_sine[rope_base + dimension]);
                    float sign = dimension < rotary_half ? -1.0f : 1.0f;
                    query_value = float(query_weighted) * cosine
                        + sign * float(query_pair) * sine;
                    key_value = float(key_weighted) * cosine
                        + sign * float(key_pair) * sine;
                }

                query_output[input_base + dimension] = query_value;
                key_output[input_base + dimension] = key_value;
            }
        """,
        ensureRowContiguous: true
    )

    private static let normalizeHeadMajorQKVRoPEFloatKernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_qk_norm_rope_f32_v1",
        inputNames: [
            "query_input", "key_input", "query_norm_weight", "key_norm_weight",
            "rope_cosine", "rope_sine", "epsilon",
        ],
        outputNames: ["query_output", "key_output"],
        source: """
            constexpr uint head_dimension = 128;
            constexpr uint rotary_dimension = 96;
            constexpr uint rotary_half = rotary_dimension / 2;

            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.y;
            uint row = threadgroup_position_in_grid.z;
            uint rows = uint(query_input_shape[2]);
            uint input_base = (head * rows + row) * head_dimension;

            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_element = query_input[input_base + dimension];
                float key_element = key_input[input_base + dimension];
                query_sum += query_element * query_element;
                key_sum += key_element * key_element;
            }
            query_sum = simd_sum(query_sum);
            key_sum = simd_sum(key_sum);
            float query_inverse = metal::precise::rsqrt(
                query_sum / float(head_dimension) + float(epsilon));
            float key_inverse = metal::precise::rsqrt(
                key_sum / float(head_dimension) + float(epsilon));

            uint rope_base = row * rotary_dimension;
            for (uint dimension = lane; dimension < head_dimension; dimension += 32) {
                float query_weighted = query_input[input_base + dimension]
                    * query_inverse * float(query_norm_weight[dimension]);
                float key_weighted = key_input[input_base + dimension]
                    * key_inverse * float(key_norm_weight[dimension]);
                float query_value = query_weighted;
                float key_value = key_weighted;

                if (dimension < rotary_dimension) {
                    uint pair = dimension < rotary_half
                        ? dimension + rotary_half
                        : dimension - rotary_half;
                    float query_pair = query_input[input_base + pair]
                        * query_inverse * float(query_norm_weight[pair]);
                    float key_pair = key_input[input_base + pair]
                        * key_inverse * float(key_norm_weight[pair]);
                    float cosine = rope_cosine[rope_base + dimension];
                    float sine = rope_sine[rope_base + dimension];
                    float sign = dimension < rotary_half ? -1.0f : 1.0f;
                    query_value = query_weighted * cosine + sign * query_pair * sine;
                    key_value = key_weighted * cosine + sign * key_pair * sine;
                }

                query_output[input_base + dimension] = query_value;
                key_output[input_base + dimension] = key_value;
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectHeadMajorAttentionAffineInt8Kernel = MLXFast.metalKernel(
        name: "mere_h3_head_major_affine_oproj_i8g64_v1",
        inputNames: [
            "attention", "weight_codes", "weight_scales", "weight_biases",
        ],
        outputNames: ["projected"],
        source: """
            constexpr uint heads = 56;
            constexpr uint head_dimension = 128;
            constexpr uint input_width = heads * head_dimension;
            constexpr uint output_width = 5376;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint rows = uint(attention_shape[2]);
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device bfloat16_t* scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            uint column = lane * values_per_thread;
            for (uint block = 0; block < input_width; block += block_size) {
                uint head = column / head_dimension;
                uint dimension = column - head * head_dimension;
                uint input_offset = (head * rows + row) * head_dimension + dimension;
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = float(attention[input_offset + index]);
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* row_codes = codes + output * input_width;
                    float scale = float(scales[output * scale_groups]);
                    float bias = float(biases[output * scale_groups]);
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output] += scale * dot + bias * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                column += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                result[output] = simd_sum(result[output]);
                if (lane == 0) {
                    projected[row * output_width + output_row + output]
                        = T(result[output]);
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectFeedForwardInputAffineInt8SwiGLUTiledKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc1_swiglu_i8g64_tiled_v5",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["activated"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint output_width = 14336;
            constexpr uint group_size = 64;
            constexpr uint scale_groups = input_width / group_size;
            constexpr uint output_tile_columns = 32;
            constexpr uint row_tile_rows = 32;
            constexpr uint reduction_tile = 32;

            uint output_tile = threadgroup_position_in_grid.x;
            uint row_tile = threadgroup_position_in_grid.y;
            uint output_start = output_tile * output_tile_columns;
            uint row_start = row_tile * row_tile_rows;
            uint rows = uint(input_shape[1]);
            uint valid_rows = metal::min(row_tile_rows, rows - row_start);
            uint thread_linear = thread_position_in_threadgroup.y * 32
                + thread_position_in_threadgroup.x;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint quad = lane / 4;
            uint matrix_row = (quad & 4) + ((lane / 2) % 4);
            uint matrix_column = (quad & 2) * 2 + (lane % 2) * 2;

            threadgroup T gate_weight_tile[
                output_tile_columns * reduction_tile
            ];
            threadgroup T up_weight_tile[
                output_tile_columns * reduction_tile
            ];
            threadgroup T input_tile[row_tile_rows * reduction_tile];

            thread simdgroup_matrix<float, 8, 8> gate_accumulated[4];
            thread simdgroup_matrix<float, 8, 8> up_accumulated[4];
            #pragma clang loop unroll(full)
            for (uint index = 0; index < 4; ++index) {
                gate_accumulated[index].thread_elements()[0] = 0.0f;
                gate_accumulated[index].thread_elements()[1] = 0.0f;
                up_accumulated[index].thread_elements()[0] = 0.0f;
                up_accumulated[index].thread_elements()[1] = 0.0f;
            }

            const device uint8_t* codes = (const device uint8_t*)weight_codes;
            uint local_output = thread_linear / 4;
            uint weight_eight = 8 * (thread_linear % 4);
            uint local_input_row = thread_linear / 4;
            uint input_eight = 8 * (thread_linear % 4);
            for (uint reduction_start = 0;
                 reduction_start < input_width;
                 reduction_start += reduction_tile) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint gate_column = output_start + local_output;
                uint up_column = gate_column + output_width;
                uint scale_group = reduction_start / group_size;
                uint gate_scale_index = gate_column * scale_groups + scale_group;
                uint up_scale_index = up_column * scale_groups + scale_group;
                float gate_scale = float(weight_scales[gate_scale_index]);
                float gate_bias = float(weight_biases[gate_scale_index]);
                float up_scale = float(weight_scales[up_scale_index]);
                float up_bias = float(weight_biases[up_scale_index]);
                uint weight_reduction_start = reduction_start + weight_eight;
                const device uint8_t* gate_values = codes
                    + gate_column * input_width + weight_reduction_start;
                const device uint8_t* up_values = codes
                    + up_column * input_width + weight_reduction_start;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 8; ++index) {
                    uint tile_index = local_output * reduction_tile
                        + weight_eight + index;
                    gate_weight_tile[tile_index] = T(
                        gate_scale * float(gate_values[index]) + gate_bias
                    );
                    up_weight_tile[tile_index] = T(
                        up_scale * float(up_values[index]) + up_bias
                    );
                }

                uint safe_row = metal::min(local_input_row, valid_rows - 1);
                const device T* input_values = input
                    + uint64_t(row_start + safe_row) * uint64_t(input_width)
                    + reduction_start + input_eight;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 8; ++index) {
                    input_tile[local_input_row * reduction_tile + input_eight + index]
                        = T(input_values[index]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup const T* gate_fragments = gate_weight_tile
                    + (simd_group % 2) * 16 * reduction_tile;
                threadgroup const T* up_fragments = up_weight_tile
                    + (simd_group % 2) * 16 * reduction_tile;
                threadgroup const T* input_fragments = input_tile
                    + (simd_group / 2) * 16 * reduction_tile;
                thread simdgroup_matrix<T, 8, 8> gate_weights[2];
                thread simdgroup_matrix<T, 8, 8> up_weights[2];
                thread simdgroup_matrix<T, 8, 8> inputs[2];
                #pragma clang loop unroll(full)
                for (uint reduction_fragment = 0;
                     reduction_fragment < 4;
                     ++reduction_fragment) {
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint output_fragment = 0;
                         output_fragment < 2;
                         ++output_fragment) {
                        simdgroup_load(
                            gate_weights[output_fragment],
                            gate_fragments
                                + output_fragment * 8 * reduction_tile
                                + reduction_fragment * 8,
                            reduction_tile,
                            0,
                            true
                        );
                        simdgroup_load(
                            up_weights[output_fragment],
                            up_fragments
                                + output_fragment * 8 * reduction_tile
                                + reduction_fragment * 8,
                            reduction_tile,
                            0,
                            true
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint row_fragment = 0; row_fragment < 2; ++row_fragment) {
                        simdgroup_load(
                            inputs[row_fragment],
                            input_fragments
                                + row_fragment * 8 * reduction_tile
                                + reduction_fragment * 8,
                            reduction_tile,
                            0,
                            false
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint result = 0; result < 4; ++result) {
                        thread simdgroup_matrix<float, 8, 8> next_gate;
                        thread simdgroup_matrix<float, 8, 8> next_up;
                        simdgroup_multiply_accumulate(
                            next_gate,
                            inputs[result / 2],
                            gate_weights[result % 2],
                            gate_accumulated[result]
                        );
                        simdgroup_multiply_accumulate(
                            next_up,
                            inputs[result / 2],
                            up_weights[result % 2],
                            up_accumulated[result]
                        );
                        gate_accumulated[result] = next_gate;
                        up_accumulated[result] = next_up;
                    }
                }
            }

            uint simd_output_start = (simd_group & 1) * 16;
            uint simd_row_start = (simd_group >> 1) * 16;
            #pragma clang loop unroll(full)
            for (uint result = 0; result < 4; ++result) {
                uint output_row = simd_row_start + (result / 2) * 8 + matrix_row;
                uint output_column = simd_output_start
                    + (result % 2) * 8 + matrix_column;
                if (output_row < valid_rows) {
                    uint64_t output_index = uint64_t(row_start + output_row)
                        * uint64_t(output_width) + uint64_t(output_start + output_column);
                    float gate0 = float(T(
                        gate_accumulated[result].thread_elements()[0]
                    ));
                    float gate1 = float(T(
                        gate_accumulated[result].thread_elements()[1]
                    ));
                    float up0 = float(T(
                        up_accumulated[result].thread_elements()[0]
                    ));
                    float up1 = float(T(
                        up_accumulated[result].thread_elements()[1]
                    ));
                    float sigmoid0 = 1.0f / (1.0f + metal::precise::exp(-gate0));
                    float sigmoid1 = 1.0f / (1.0f + metal::precise::exp(-gate1));
                    activated[output_index] = T(gate0 * sigmoid0 * up0);
                    activated[output_index + 1] = T(
                        gate1 * sigmoid1 * up1
                    );
                }
            }
        """,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    private static let projectFeedForwardInputAffineInt8SwiGLUKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc1_swiglu_i8g64_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["activated"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint output_width = 14336;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* gate_codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device uint8_t* up_codes = (const device uint8_t*)weight_codes
                + (output_row + output_width) * input_width
                + lane * values_per_thread;
            const device bfloat16_t* gate_scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_scales = weight_scales
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            const device bfloat16_t* gate_biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_biases = weight_biases
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            const device bfloat16_t* values = input
                + row * input_width + lane * values_per_thread;

            thread float gate_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            thread float up_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = float(values[index]);
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* gate_row = gate_codes + output * input_width;
                    const device uint8_t* up_row = up_codes + output * input_width;
                    float gate_dot = 0.0f;
                    float up_dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        float value = input_values[index];
                        gate_dot += value * float(gate_row[index]);
                        up_dot += value * float(up_row[index]);
                    }
                    gate_result[output] += float(
                        gate_scales[output * scale_groups]) * gate_dot
                        + float(gate_biases[output * scale_groups]) * input_sum;
                    up_result[output] += float(
                        up_scales[output * scale_groups]) * up_dot
                        + float(up_biases[output * scale_groups]) * input_sum;
                }

                gate_codes += block_size;
                up_codes += block_size;
                gate_scales += block_size / group_size;
                up_scales += block_size / group_size;
                gate_biases += block_size / group_size;
                up_biases += block_size / group_size;
                values += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                float gate_sum = simd_sum(gate_result[output]);
                float up_sum = simd_sum(up_result[output]);
                if (lane == 0) {
                    float gate = float(bfloat16_t(gate_sum));
                    float up = float(bfloat16_t(up_sum));
                    float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
                    activated[row * output_width + output_row + output]
                        = bfloat16_t(gate * sigmoid * up);
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectFeedForwardInputAffineInt8SwiGLUFloatKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc1_swiglu_i8g64_f32_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["activated"],
        source: """
            constexpr uint input_width = 5376;
            constexpr uint output_width = 14336;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* gate_codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device uint8_t* up_codes = (const device uint8_t*)weight_codes
                + (output_row + output_width) * input_width
                + lane * values_per_thread;
            const device bfloat16_t* gate_scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_scales = weight_scales
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            const device bfloat16_t* gate_biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* up_biases = weight_biases
                + (output_row + output_width) * scale_groups
                + lane / scale_step_per_thread;
            uint input_column = lane * values_per_thread;

            thread float gate_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            thread float up_result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = input[row * input_width + input_column + index];
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* gate_row = gate_codes + output * input_width;
                    const device uint8_t* up_row = up_codes + output * input_width;
                    float gate_dot = 0.0f;
                    float up_dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        gate_dot += input_values[index] * float(gate_row[index]);
                        up_dot += input_values[index] * float(up_row[index]);
                    }
                    gate_result[output] += float(
                        gate_scales[output * scale_groups]) * gate_dot
                        + float(gate_biases[output * scale_groups]) * input_sum;
                    up_result[output] += float(
                        up_scales[output * scale_groups]) * up_dot
                        + float(up_biases[output * scale_groups]) * input_sum;
                }

                gate_codes += block_size;
                up_codes += block_size;
                gate_scales += block_size / group_size;
                up_scales += block_size / group_size;
                gate_biases += block_size / group_size;
                up_biases += block_size / group_size;
                input_column += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                float gate = simd_sum(gate_result[output]);
                float up = simd_sum(up_result[output]);
                if (lane == 0) {
                    float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
                    activated[row * output_width + output_row + output]
                        = gate * sigmoid * up;
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectFeedForwardOutputAffineInt8Kernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc2_i8g64_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["projected"],
        source: """
            constexpr uint input_width = 14336;
            constexpr uint output_width = 5376;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device bfloat16_t* scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* values = input
                + row * input_width + lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = float(values[index]);
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* row_codes = codes + output * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output] += float(scales[output * scale_groups]) * dot
                        + float(biases[output * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                values += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                result[output] = simd_sum(result[output]);
                if (lane == 0) {
                    projected[row * output_width + output_row + output]
                        = bfloat16_t(result[output]);
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let projectFeedForwardOutputAffineInt8TiledKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc2_i8g64_tiled_v5",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["projected"],
        source: """
            constexpr uint input_width = 14336;
            constexpr uint output_width = 5376;
            constexpr uint group_size = 64;
            constexpr uint scale_groups = input_width / group_size;
            constexpr uint output_tile_columns = 64;
            constexpr uint row_tile_rows = 32;
            constexpr uint reduction_tile = 32;
            constexpr uint simdgroups_per_workgroup = 4;

            uint output_tile = threadgroup_position_in_grid.x;
            uint row_tile = threadgroup_position_in_grid.y;
            uint output_start = output_tile * output_tile_columns;
            uint row_start = row_tile * row_tile_rows;
            uint rows = uint(input_shape[1]);
            uint valid_rows = metal::min(row_tile_rows, rows - row_start);
            uint thread_linear = thread_position_in_threadgroup.y * 32
                + thread_position_in_threadgroup.x;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint quad = lane / 4;
            uint matrix_row = (quad & 4) + ((lane / 2) % 4);
            uint matrix_column = (quad & 2) * 2 + (lane % 2) * 2;

            threadgroup T weight_tile[output_tile_columns * reduction_tile];
            threadgroup T input_tile[row_tile_rows * reduction_tile];

            thread simdgroup_matrix<float, 8, 8> accumulated[8];
            #pragma clang loop unroll(full)
            for (uint index = 0; index < 8; ++index) {
                accumulated[index].thread_elements()[0] = 0.0f;
                accumulated[index].thread_elements()[1] = 0.0f;
            }

            const device uint8_t* codes = (const device uint8_t*)weight_codes;
            uint local_output = thread_linear / 2;
            uint weight_half = thread_linear % 2;
            uint local_input_row = thread_linear / 4;
            uint input_eight = 8 * (thread_linear % 4);
            for (uint reduction_start = 0;
                 reduction_start < input_width;
                 reduction_start += reduction_tile) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint output_column = output_start + local_output;
                uint scale_index = output_column * scale_groups
                    + reduction_start / group_size;
                float scale = float(weight_scales[scale_index]);
                float bias = float(weight_biases[scale_index]);
                uint weight_reduction_start = reduction_start + 16 * weight_half;
                const device uint8_t* weight_values = codes
                    + output_column * input_width + weight_reduction_start;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 16; ++index) {
                    uint section_x = 2 * weight_half + index / 8;
                    uint section_y = local_output / 8;
                    uint local_x = local_output % 8;
                    uint local_y = index % 8;
                    uint block = 8 * section_x + section_y;
                    weight_tile[64 * block + 8 * local_y + local_x] = T(
                        scale * float(weight_values[index]) + bias
                    );
                }

                uint safe_row = metal::min(local_input_row, valid_rows - 1);
                uint input_section_x = thread_linear % 4;
                uint input_section_y = local_input_row / 8;
                uint input_local_y = local_input_row % 8;
                uint input_block = 4 * input_section_x + input_section_y;
                const device T* input_values = input
                    + uint64_t(row_start + safe_row) * uint64_t(input_width)
                    + reduction_start + input_eight;
                #pragma clang loop unroll(full)
                for (uint index = 0; index < 8; ++index) {
                    input_tile[64 * input_block + 8 * input_local_y + index]
                        = T(input_values[index]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup const T* weight_fragments = weight_tile
                    + 4 * 64 * (simd_group % 2);
                threadgroup const T* input_fragments = input_tile
                    + 2 * 64 * (simd_group / 2);
                thread simdgroup_matrix<T, 8, 8> weights[4];
                thread simdgroup_matrix<T, 8, 8> inputs[2];
                #pragma clang loop unroll(full)
                for (uint reduction_fragment = 0;
                     reduction_fragment < 4;
                     ++reduction_fragment) {
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint output_fragment = 0;
                         output_fragment < 4;
                         ++output_fragment) {
                        simdgroup_load(
                            weights[output_fragment],
                            weight_fragments + 64 * output_fragment,
                            8,
                            0,
                            false
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint row_fragment = 0; row_fragment < 2; ++row_fragment) {
                        simdgroup_load(
                            inputs[row_fragment],
                            input_fragments + 64 * row_fragment,
                            8,
                            0,
                            false
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint result = 0; result < 8; ++result) {
                        thread simdgroup_matrix<float, 8, 8> next;
                        simdgroup_multiply_accumulate(
                            next,
                            inputs[result / 4],
                            weights[result % 4],
                            accumulated[result]
                        );
                        accumulated[result] = next;
                    }
                    weight_fragments += 8 * 64;
                    input_fragments += 4 * 64;
                }
            }

            uint simd_output_start = (simd_group & 1) * 32;
            uint simd_row_start = (simd_group >> 1) * 16;
            #pragma clang loop unroll(full)
            for (uint result = 0; result < 8; ++result) {
                uint output_row = simd_row_start + (result / 4) * 8 + matrix_row;
                uint output_column = simd_output_start
                    + (result % 4) * 8 + matrix_column;
                if (output_row < valid_rows) {
                    uint64_t output_index = uint64_t(row_start + output_row)
                        * uint64_t(output_width) + uint64_t(output_start + output_column);
                    projected[output_index] = T(
                        accumulated[result].thread_elements()[0]
                    );
                    projected[output_index + 1] = T(
                        accumulated[result].thread_elements()[1]
                    );
                }
            }
        """,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    private static let projectFeedForwardOutputAffineInt8FloatKernel = MLXFast.metalKernel(
        name: "mere_h3_affine_fc2_i8g64_f32_v1",
        inputNames: ["input", "weight_codes", "weight_scales", "weight_biases"],
        outputNames: ["projected"],
        source: """
            constexpr uint input_width = 14336;
            constexpr uint output_width = 5376;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint number_simdgroups = 2;
            constexpr uint group_size = 64;
            constexpr uint scale_step_per_thread = group_size / values_per_thread;
            constexpr uint scale_groups = input_width / group_size;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_row = tile
                * (number_simdgroups * results_per_simdgroup)
                + simd_group * results_per_simdgroup;

            const device uint8_t* codes = (const device uint8_t*)weight_codes
                + output_row * input_width + lane * values_per_thread;
            const device bfloat16_t* scales = weight_scales
                + output_row * scale_groups + lane / scale_step_per_thread;
            const device bfloat16_t* biases = weight_biases
                + output_row * scale_groups + lane / scale_step_per_thread;
            uint input_column = lane * values_per_thread;

            thread float result[results_per_simdgroup] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            for (uint block = 0; block < input_width; block += block_size) {
                thread float input_values[values_per_thread];
                float input_sum = 0.0f;
                for (uint index = 0; index < values_per_thread; ++index) {
                    float value = input[row * input_width + input_column + index];
                    input_values[index] = value;
                    input_sum += value;
                }

                for (uint output = 0; output < results_per_simdgroup; ++output) {
                    const device uint8_t* row_codes = codes + output * input_width;
                    float dot = 0.0f;
                    for (uint index = 0; index < values_per_thread; ++index) {
                        dot += input_values[index] * float(row_codes[index]);
                    }
                    result[output] += float(scales[output * scale_groups]) * dot
                        + float(biases[output * scale_groups]) * input_sum;
                }

                codes += block_size;
                scales += block_size / group_size;
                biases += block_size / group_size;
                input_column += block_size;
            }

            for (uint output = 0; output < results_per_simdgroup; ++output) {
                float value = simd_sum(result[output]);
                if (lane == 0) {
                    projected[row * output_width + output_row + output] = value;
                }
            }
        """,
        ensureRowContiguous: true
    )
    #endif
}
