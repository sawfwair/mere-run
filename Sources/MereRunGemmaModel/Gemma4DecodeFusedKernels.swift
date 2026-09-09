import Foundation
import MLX
import MLXFast

/// Custom fused Metal kernels for the seq==1 decode hot path.
///
/// Decode throughput is bounded by per-token graph weight — every extra node
/// costs schedule/encode time on the CPU and a launch gap on the GPU — so these
/// kernels collapse the elementwise chains between matmuls into single
/// dispatches: (1) QKV head split + q/k RMSNorm + value no-scale norm,
/// (2) post-attention norm + residual + pre-FFN norm, (3) gelu(gate)·up over
/// the fused gate/up buffer, (4) post-FFN norm + residual + layer scalar.
/// Numerics follow MLXFast.rmsNorm (float32 accumulation, rsqrt(mean+eps)) and
/// MLXNN.geluApproximate (tanh form).
enum Gemma4DecodeFusedKernels {
    private struct Key: Hashable {
        enum Kind: Hashable {
            case qkvNorms
            case residualDoubleNorm
            case geluMul
            case ffnResidualScale
        }

        let kind: Kind
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let hidden: Int
        let scalarType: String
    }

    /// Metal scalar type for kernel output stores. mlx's bfloat16_t/half have
    /// explicit float constructors, so float expressions must be cast on write.
    private static func metalScalarType(for dtype: DType) -> String {
        switch dtype {
        case .bfloat16: return "bfloat16_t"
        case .float16: return "half"
        default: return "float"
        }
    }

    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [Key: MLXFast.MLXFastKernel] = [:]

    private static func kernel(
        key: Key,
        inputNames: [String],
        outputNames: [String],
        source: @autoclosure () -> String
    ) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }

        if let existing = kernels[key] {
            return existing
        }

        let kernel = MLXFast.metalKernel(
            name: "gemma4_decode_\(key.kind)_h\(key.numHeads)_kv\(key.numKVHeads)_d\(key.headDim)_hs\(key.hidden)",
            inputNames: inputNames,
            outputNames: outputNames,
            source: source()
        )
        kernels[key] = kernel
        return kernel
    }

    /// Splits the fused QKV projection row into per-head q/k/v in transposed
    /// [B, heads, 1, dim] layout while applying qNorm, kNorm, and the value
    /// no-scale RMS norm. One simdgroup per head.
    static func qkvNorms(
        qkv: MLXArray,
        qNormWeight: MLXArray,
        kNormWeight: MLXArray,
        eps: MLXArray,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let batch = qkv.dim(0)
        let totalHeads = numHeads + 2 * numKVHeads
        let scalarType = metalScalarType(for: qkv.dtype)
        let key = Key(
            kind: .qkvNorms,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            hidden: 0,
            scalarType: scalarType
        )
        let kernel = kernel(
            key: key,
            inputNames: ["qkv", "q_weight", "k_weight", "eps"],
            outputNames: ["q_out", "k_out", "v_out"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto head_slot = thread_position_in_grid.y;
                auto batch = thread_position_in_grid.z;
                if (head_slot >= TotalHeads) {
                    return;
                }

                auto row = qkv + batch * (NumHeads + 2 * NumKVHeads) * Dim;

                int source_offset;
                if (head_slot < NumHeads) {
                    source_offset = head_slot * Dim;
                } else if (head_slot < NumHeads + NumKVHeads) {
                    source_offset = (NumHeads + (head_slot - NumHeads)) * Dim;
                } else {
                    source_offset = (NumHeads + NumKVHeads + (head_slot - NumHeads - NumKVHeads)) * Dim;
                }
                auto source_ptr = row + source_offset;

                float acc = 0.0f;
                for (int d = lane; d < Dim; d += 32) {
                    float value = static_cast<float>(source_ptr[d]);
                    acc += value * value;
                }
                acc = simd_sum(acc);
                float normalizer = metal::precise::rsqrt(acc / static_cast<float>(Dim) + static_cast<float>(eps[0]));

                if (head_slot < NumHeads) {
                    auto out = q_out + ((batch * NumHeads + head_slot) * Dim);
                    for (int d = lane; d < Dim; d += 32) {
                        out[d] = \(scalarType)(static_cast<float>(source_ptr[d]) * normalizer * static_cast<float>(q_weight[d]));
                    }
                } else if (head_slot < NumHeads + NumKVHeads) {
                    auto out = k_out + ((batch * NumKVHeads + (head_slot - NumHeads)) * Dim);
                    for (int d = lane; d < Dim; d += 32) {
                        out[d] = \(scalarType)(static_cast<float>(source_ptr[d]) * normalizer * static_cast<float>(k_weight[d]));
                    }
                } else {
                    auto out = v_out + ((batch * NumKVHeads + (head_slot - NumHeads - NumKVHeads)) * Dim);
                    for (int d = lane; d < Dim; d += 32) {
                        out[d] = \(scalarType)(static_cast<float>(source_ptr[d]) * normalizer);
                    }
                }
                """
        )
        let outputs = kernel(
            [qkv, qNormWeight, kNormWeight, eps],
            template: [
                ("NumHeads", numHeads),
                ("NumKVHeads", numKVHeads),
                ("Dim", headDim),
                ("TotalHeads", totalHeads),
            ],
            grid: (32, totalHeads, batch),
            threadGroup: (32, 1, 1),
            outputShapes: [
                [batch, numHeads, 1, headDim],
                [batch, numKVHeads, 1, headDim],
                [batch, numKVHeads, 1, headDim],
            ],
            outputDTypes: [qkv.dtype, qkv.dtype, qkv.dtype]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    /// mlp_residual = residual + postNorm(attn_out); mlp_input = preNorm(mlp_residual).
    /// One threadgroup per batch row, two chained reductions.
    static func residualDoubleNorm(
        attentionOutput: MLXArray,
        residual: MLXArray,
        postNormWeight: MLXArray,
        preNormWeight: MLXArray,
        eps: MLXArray,
        hidden: Int
    ) -> (MLXArray, MLXArray) {
        let batch = attentionOutput.dim(0)
        let perThread = (hidden + 255) / 256
        let scalarType = metalScalarType(for: attentionOutput.dtype)
        let key = Key(
            kind: .residualDoubleNorm,
            numHeads: 0,
            numKVHeads: 0,
            headDim: 0,
            hidden: hidden,
            scalarType: scalarType
        )
        let kernel = kernel(
            key: key,
            inputNames: ["attn_out", "residual", "post_weight", "pre_weight", "eps"],
            outputNames: ["mlp_residual", "mlp_input"],
            source: """
                auto tid = thread_position_in_grid.x;
                auto batch = thread_position_in_grid.z;
                auto simd_slot = tid / 32;
                auto attn_row = attn_out + batch * Hidden;
                auto residual_row = residual + batch * Hidden;
                auto mlp_residual_row = mlp_residual + batch * Hidden;
                auto mlp_input_row = mlp_input + batch * Hidden;

                threadgroup float shared[8];
                float epsilon = static_cast<float>(eps[0]);

                float acc = 0.0f;
                for (int i = tid; i < Hidden; i += 256) {
                    float value = static_cast<float>(attn_row[i]);
                    acc += value * value;
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    shared[simd_slot] = acc;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                float total = 0.0f;
                for (int s = 0; s < 8; s++) {
                    total += shared[s];
                }
                float normalizer = metal::precise::rsqrt(total / static_cast<float>(Hidden) + epsilon);
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);

                float local[PerThread];
                float acc2 = 0.0f;
                int slot = 0;
                for (int i = tid; i < Hidden; i += 256, slot++) {
                    float normed = static_cast<float>(attn_row[i]) * normalizer * static_cast<float>(post_weight[i]);
                    float combined = static_cast<float>(residual_row[i]) + normed;
                    local[slot] = combined;
                    mlp_residual_row[i] = \(scalarType)(combined);
                    acc2 += combined * combined;
                }
                acc2 = simd_sum(acc2);
                if (thread_index_in_simdgroup == 0) {
                    shared[simd_slot] = acc2;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                float total2 = 0.0f;
                for (int s = 0; s < 8; s++) {
                    total2 += shared[s];
                }
                float normalizer2 = metal::precise::rsqrt(total2 / static_cast<float>(Hidden) + epsilon);

                slot = 0;
                for (int i = tid; i < Hidden; i += 256, slot++) {
                    mlp_input_row[i] = \(scalarType)(local[slot] * normalizer2 * static_cast<float>(pre_weight[i]));
                }
                """
        )
        let outputs = kernel(
            [attentionOutput, residual, postNormWeight, preNormWeight, eps],
            template: [
                ("Hidden", hidden),
                ("PerThread", perThread),
            ],
            grid: (256, 1, batch),
            threadGroup: (256, 1, 1),
            outputShapes: [
                [batch, 1, hidden],
                [batch, 1, hidden],
            ],
            outputDTypes: [attentionOutput.dtype, attentionOutput.dtype]
        )
        return (outputs[0], outputs[1])
    }

    /// act[i] = geluApproximate(gate_up[i]) * gate_up[Intermediate + i] over the
    /// fused [B, 1, 2*Intermediate] gate/up projection output.
    static func geluMul(gateUp: MLXArray, intermediate: Int) -> MLXArray {
        let batch = gateUp.dim(0)
        let scalarType = metalScalarType(for: gateUp.dtype)
        let key = Key(
            kind: .geluMul,
            numHeads: 0,
            numKVHeads: 0,
            headDim: 0,
            hidden: intermediate,
            scalarType: scalarType
        )
        let kernel = kernel(
            key: key,
            inputNames: ["gate_up"],
            outputNames: ["act"],
            source: """
                auto i = thread_position_in_grid.x;
                auto batch = thread_position_in_grid.z;
                if (i >= Intermediate) {
                    return;
                }
                auto row = gate_up + batch * 2 * Intermediate;
                float gate = static_cast<float>(row[i]);
                float up = static_cast<float>(row[Intermediate + i]);
                float cubed = gate * gate * gate;
                float inner = 0.7978845608028654f * (gate + 0.044715f * cubed);
                float gelu = 0.5f * gate * (1.0f + metal::precise::tanh(inner));
                act[batch * Intermediate + i] = \(scalarType)(gelu * up);
                """
        )
        return kernel(
            [gateUp],
            template: [("Intermediate", intermediate)],
            grid: (intermediate, 1, batch),
            threadGroup: (256, 1, 1),
            outputShapes: [[batch, 1, intermediate]],
            outputDTypes: [gateUp.dtype]
        )[0]
    }

    /// x_next = (mlp_residual + postFFNorm(down_out)) * layer_scalar.
    static func ffnResidualScale(
        downOutput: MLXArray,
        mlpResidual: MLXArray,
        postNormWeight: MLXArray,
        layerScalar: MLXArray,
        eps: MLXArray,
        hidden: Int
    ) -> MLXArray {
        let batch = downOutput.dim(0)
        let scalarType = metalScalarType(for: downOutput.dtype)
        let key = Key(
            kind: .ffnResidualScale,
            numHeads: 0,
            numKVHeads: 0,
            headDim: 0,
            hidden: hidden,
            scalarType: scalarType
        )
        let kernel = kernel(
            key: key,
            inputNames: ["down_out", "mlp_residual", "post_weight", "layer_scalar", "eps"],
            outputNames: ["x_next"],
            source: """
                auto tid = thread_position_in_grid.x;
                auto batch = thread_position_in_grid.z;
                auto simd_slot = tid / 32;
                auto down_row = down_out + batch * Hidden;
                auto residual_row = mlp_residual + batch * Hidden;
                auto out_row = x_next + batch * Hidden;

                threadgroup float shared[8];

                float acc = 0.0f;
                for (int i = tid; i < Hidden; i += 256) {
                    float value = static_cast<float>(down_row[i]);
                    acc += value * value;
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    shared[simd_slot] = acc;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                float total = 0.0f;
                for (int s = 0; s < 8; s++) {
                    total += shared[s];
                }
                float normalizer = metal::precise::rsqrt(total / static_cast<float>(Hidden) + static_cast<float>(eps[0]));
                float scalar = static_cast<float>(layer_scalar[0]);

                for (int i = tid; i < Hidden; i += 256) {
                    float normed = static_cast<float>(down_row[i]) * normalizer * static_cast<float>(post_weight[i]);
                    out_row[i] = \(scalarType)((static_cast<float>(residual_row[i]) + normed) * scalar);
                }
                """
        )
        return kernel(
            [downOutput, mlpResidual, postNormWeight, layerScalar, eps],
            template: [("Hidden", hidden)],
            grid: (256, 1, batch),
            threadGroup: (256, 1, 1),
            outputShapes: [[batch, 1, hidden]],
            outputDTypes: [downOutput.dtype]
        )[0]
    }
}
