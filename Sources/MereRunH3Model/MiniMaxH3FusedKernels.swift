import MereRunTensor
import MLX
import MLXFast

package struct MiniMaxH3GateAdaLNOutput {
    package let residual: MLXArray
    package let feedForwardInput: MLXArray
    package let feedForwardGate: MLXArray
}

package struct MiniMaxH3DynamicInt8Rows {
    package let values: MLXArray
    package let scales: MLXArray
}

package struct MiniMaxH3GateAdaLNQuantizedOutput {
    package let residual: MLXArray
    package let feedForwardInput: MiniMaxH3DynamicInt8Rows
}

package struct MiniMaxH3HeadMajorQKV {
    package let query: MLXArray
    package let key: MLXArray
    package let value: MLXArray
}

/// Exact-shape Metal experiments for MiniMax-H3's 5,376-wide DiT blocks.
/// Production keeps the decomposed MLX graph until a fused candidate passes
/// tensor parity, realistic-shape timing, and an installed-checkpoint A/B.
package enum MiniMaxH3FusedKernels {
    static let hiddenSize = 5_376
    static let modulationPartCount = 6
    static let threadCount = 256
    static let attentionHeadCount = 56
    static let attentionHeadDimension = 128
    static let attentionInnerDimension = attentionHeadCount * attentionHeadDimension
    static let rotaryDimension = 96
    static let affineGroupSize = 64
    static let feedForwardSize = 14_336

    /// Fuses the first H3 block boundary: RMSNorm followed by the row-indexed
    /// attention scale and shift. This is the Metal counterpart of FastVideo's
    /// `fused_rmsnorm_modulate` Triton kernel.
    package static func prepareAttentionInput(
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

    package static func gateAttentionAndPrepareFeedForward(
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
    package static func gateAttentionAndQuantizeFeedForward(
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
    package static func quantizeRowsSymmetricInt8(_ input: MLXArray) -> MiniMaxH3DynamicInt8Rows? {
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
    package static func prepareHeadMajorQKV(
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
    #if os(macOS) || os(iOS)
    static func supportsAdaLNShapeContract(
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

    static func supportsGateAdaLNShapeContract(
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

    static func supportsGateAdaLNInputs(
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

    #endif
}
