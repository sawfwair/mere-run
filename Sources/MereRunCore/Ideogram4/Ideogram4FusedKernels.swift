import Foundation
import MLX
import MLXFast

enum Ideogram4FusedKernelPolicy {
    /// Opt-in for the experimental Apple-Silicon fused denoiser kernels.
    /// The exact portable MLX graph remains the production default until the
    /// custom kernels demonstrate an end-to-end checkpoint win.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_IDEOGRAM4_FUSED_KERNELS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw == "1" || raw == "true" || raw == "on"
    }()
}

struct Ideogram4FusedKernelSavings: Equatable, Sendable {
    let explicitDispatchesPerLayer: Int
    let stridedQKElementsPerLayer: Int
    let uniformMaskElementsPerLayer: Int
    let maskedScoreElementsPerLayer: Int

    static func estimate(batch: Int, sequence: Int, heads: Int, headDim: Int) -> Self {
        // q/k normalization: 2 -> 1 dispatch. Two pre-norm modulations and
        // two gated residuals each collapse a norm plus elementwise dispatch
        // into one dispatch: five explicit launches removed per layer.
        Self(
            explicitDispatchesPerLayer: 5,
            stridedQKElementsPerLayer: 2 * batch * sequence * heads * headDim,
            uniformMaskElementsPerLayer: batch * sequence * sequence,
            maskedScoreElementsPerLayer: batch * heads * sequence * sequence
        )
    }
}

enum Ideogram4FusedKernels {
    private enum Kind: Hashable {
        case qkvNorms
        case scaledRMSNorm
        case gatedResidualRMSNorm
    }

    private struct Key: Hashable {
        let kind: Kind
        let width: Int
        let heads: Int
        let scalarType: String
    }

    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [Key: MLXFast.MLXFastKernel] = [:]

    private static func scalarType(for dtype: DType) -> (metal: String, tag: String)? {
        switch dtype {
        case .bfloat16: return ("bfloat16_t", "bf16")
        case .float16: return ("half", "f16")
        case .float32: return ("float", "f32")
        default: return nil
        }
    }

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
            name: "ideogram4_\(key.kind)_w\(key.width)_h\(key.heads)_\(key.scalarType)",
            inputNames: inputNames,
            outputNames: outputNames,
            source: source()
        )
        kernels[key] = kernel
        return kernel
    }

    /// Split a packed QKV projection directly into attention layout while
    /// applying q/k RMSNorm. This replaces two normalization dispatches and
    /// avoids materializing the strided q/k slices that those kernels require.
    static func qkvNorms(
        qkv: MLXArray,
        qWeight: MLXArray,
        kWeight: MLXArray,
        eps: Float,
        numHeads: Int,
        headDim: Int
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray)? {
        #if os(macOS) || os(iOS)
        guard Device.defaultDevice().deviceType == .gpu,
              qkv.ndim == 3,
              qkv.dim(-1) == 3 * numHeads * headDim,
              qWeight.shape == [headDim],
              kWeight.shape == [headDim],
              qWeight.dtype == qkv.dtype,
              kWeight.dtype == qkv.dtype,
              let scalar = scalarType(for: qkv.dtype) else {
            return nil
        }

        let batch = qkv.dim(0)
        let sequence = qkv.dim(1)
        let key = Key(kind: .qkvNorms, width: headDim, heads: numHeads, scalarType: scalar.tag)
        let kernel = kernel(
            key: key,
            inputNames: ["qkv", "q_weight", "k_weight", "epsilon"],
            outputNames: ["q_out", "k_out", "v_out"],
            source: """
                auto tid = thread_position_in_threadgroup.x;
                auto simd_lane = thread_index_in_simdgroup;
                auto simd_group = simdgroup_index_in_threadgroup;
                auto thread_count = threads_per_threadgroup.x;
                auto head_row = thread_position_in_grid.y;
                auto head = head_row % Heads;
                auto token_row = head_row / Heads;
                auto token = token_row % Sequence;
                auto batch = token_row / Sequence;

                auto packed_row = qkv + token_row * 3 * Heads * Dim;
                auto q_row = packed_row + head * Dim;
                auto k_row = packed_row + Heads * Dim + head * Dim;
                auto v_row = packed_row + 2 * Heads * Dim + head * Dim;

                threadgroup float q_sums[32];
                threadgroup float k_sums[32];
                threadgroup float inverse_rms[2];
                float q_sum = 0.0f;
                float k_sum = 0.0f;
                for (int start = 0; start < Dim; start += thread_count * 4) {
                    int base = start + tid * 4;
                    for (int index = 0; index < 4; index++) {
                        int d = base + index;
                        if (d < Dim) {
                            float q_value = static_cast<float>(q_row[d]);
                            float k_value = static_cast<float>(k_row[d]);
                            q_sum += q_value * q_value;
                            k_sum += k_value * k_value;
                        }
                    }
                }
                q_sum = simd_sum(q_sum);
                k_sum = simd_sum(k_sum);
                if (simd_group == 0) {
                    q_sums[simd_lane] = 0.0f;
                    k_sums[simd_lane] = 0.0f;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    q_sums[simd_group] = q_sum;
                    k_sums[simd_group] = k_sum;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    q_sum = simd_sum(q_sums[simd_lane]);
                    k_sum = simd_sum(k_sums[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_rms[0] = metal::precise::rsqrt(
                            q_sum / static_cast<float>(Dim) + static_cast<float>(epsilon));
                        inverse_rms[1] = metal::precise::rsqrt(
                            k_sum / static_cast<float>(Dim) + static_cast<float>(epsilon));
                    }
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);

                auto output_offset = ((batch * Heads + head) * Sequence + token) * Dim;
                for (int start = 0; start < Dim; start += thread_count * 4) {
                    int base = start + tid * 4;
                    for (int index = 0; index < 4; index++) {
                        int d = base + index;
                        if (d < Dim) {
                            auto q_normalized = \(scalar.metal)(
                                static_cast<float>(q_row[d]) * inverse_rms[0]);
                            auto k_normalized = \(scalar.metal)(
                                static_cast<float>(k_row[d]) * inverse_rms[1]);
                            q_out[output_offset + d] = \(scalar.metal)(
                                static_cast<float>(q_weight[d]) * static_cast<float>(q_normalized));
                            k_out[output_offset + d] = \(scalar.metal)(
                                static_cast<float>(k_weight[d]) * static_cast<float>(k_normalized));
                            v_out[output_offset + d] = \(scalar.metal)(static_cast<float>(v_row[d]));
                        }
                    }
                }
                """
        )
        let outputs = kernel(
            [qkv, qWeight, kWeight, eps],
            template: [
                ("Heads", numHeads),
                ("Dim", headDim),
                ("Sequence", sequence),
            ],
            grid: (normalizationThreadgroup(width: headDim), batch * sequence * numHeads, 1),
            threadGroup: (normalizationThreadgroup(width: headDim), 1, 1),
            outputShapes: [
                [batch, numHeads, sequence, headDim],
                [batch, numHeads, sequence, headDim],
                [batch, numHeads, sequence, headDim],
            ],
            outputDTypes: [qkv.dtype, qkv.dtype, qkv.dtype]
        )
        return (outputs[0], outputs[1], outputs[2])
        #else
        return nil
        #endif
    }

    /// `RMSNorm(x, weight) * (1 + modulation[index])` in one Metal dispatch.
    /// The explicit cast preserves normalization output rounding before the
    /// modulation that the portable graph would otherwise launch separately.
    static func scaledRMSNorm(
        _ x: MLXArray,
        weight: MLXArray,
        modulation: MLXArray,
        modulationIndex: Int,
        eps: Float
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        guard (0..<4).contains(modulationIndex),
              let shape = normalizedShape(x, weight: weight, modulation: modulation),
              Device.defaultDevice().deviceType == .gpu,
              let scalar = scalarType(for: x.dtype) else {
            return nil
        }

        let key = Key(kind: .scaledRMSNorm, width: shape.hidden, heads: 0, scalarType: scalar.tag)
        let kernel = kernel(
            key: key,
            inputNames: ["input", "weight", "modulation", "epsilon"],
            outputNames: ["output"],
            source: """
                auto tid = thread_position_in_threadgroup.x;
                auto simd_lane = thread_index_in_simdgroup;
                auto simd_group = simdgroup_index_in_threadgroup;
                auto thread_count = threads_per_threadgroup.x;
                auto row = thread_position_in_grid.y;
                auto batch = row / Sequence;
                auto input_row = input + row * Hidden;
                auto output_row = output + row * Hidden;
                auto modulation_row = modulation
                    + (ModulationRows == 1 ? 0 : batch * 4 * Hidden)
                    + ModulationIndex * Hidden;

                threadgroup float shared[32];
                threadgroup float inverse_rms[1];
                float sum = 0.0f;
                for (int start = 0; start < Hidden; start += thread_count * 4) {
                    int base = start + tid * 4;
                    for (int index = 0; index < 4; index++) {
                        int d = base + index;
                        if (d < Hidden) {
                            float value = static_cast<float>(input_row[d]);
                            sum += value * value;
                        }
                    }
                }
                sum = simd_sum(sum);
                if (simd_group == 0) {
                    shared[simd_lane] = 0.0f;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    shared[simd_group] = sum;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    sum = simd_sum(shared[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_rms[0] = metal::precise::rsqrt(
                            sum / static_cast<float>(Hidden) + static_cast<float>(epsilon));
                    }
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int start = 0; start < Hidden; start += thread_count * 4) {
                    int base = start + tid * 4;
                    for (int index = 0; index < 4; index++) {
                        int d = base + index;
                        if (d < Hidden) {
                            auto normalized = \(scalar.metal)(
                                static_cast<float>(input_row[d]) * inverse_rms[0]);
                            auto weighted = \(scalar.metal)(
                                static_cast<float>(weight[d]) * static_cast<float>(normalized));
                            output_row[d] = \(scalar.metal)(
                                static_cast<float>(weighted)
                                    * (1.0f + static_cast<float>(modulation_row[d])));
                        }
                    }
                }
                """
        )
        return kernel(
            [x, weight, modulation, eps],
            template: [
                ("Hidden", shape.hidden),
                ("Sequence", shape.sequence),
                ("ModulationRows", shape.modulationRows),
                ("ModulationIndex", modulationIndex),
            ],
            grid: (normalizationThreadgroup(width: shape.hidden), shape.batch * shape.sequence, 1),
            threadGroup: (normalizationThreadgroup(width: shape.hidden), 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    /// `residual + tanh(modulation[index]) * RMSNorm(x, weight)` in one
    /// Metal dispatch.
    static func gatedResidualRMSNorm(
        _ x: MLXArray,
        residual: MLXArray,
        weight: MLXArray,
        modulation: MLXArray,
        modulationIndex: Int,
        eps: Float
    ) -> MLXArray? {
        #if os(macOS) || os(iOS)
        guard residual.shape == x.shape,
              residual.dtype == x.dtype,
              (0..<4).contains(modulationIndex),
              let shape = normalizedShape(x, weight: weight, modulation: modulation),
              Device.defaultDevice().deviceType == .gpu,
              let scalar = scalarType(for: x.dtype) else {
            return nil
        }

        let key = Key(kind: .gatedResidualRMSNorm, width: shape.hidden, heads: 0, scalarType: scalar.tag)
        let kernel = kernel(
            key: key,
            inputNames: ["input", "residual", "weight", "modulation", "epsilon"],
            outputNames: ["output"],
            source: """
                auto tid = thread_position_in_threadgroup.x;
                auto simd_lane = thread_index_in_simdgroup;
                auto simd_group = simdgroup_index_in_threadgroup;
                auto thread_count = threads_per_threadgroup.x;
                auto row = thread_position_in_grid.y;
                auto batch = row / Sequence;
                auto input_row = input + row * Hidden;
                auto residual_row = residual + row * Hidden;
                auto output_row = output + row * Hidden;
                auto modulation_row = modulation
                    + (ModulationRows == 1 ? 0 : batch * 4 * Hidden)
                    + ModulationIndex * Hidden;

                threadgroup float shared[32];
                threadgroup float inverse_rms[1];
                float sum = 0.0f;
                for (int start = 0; start < Hidden; start += thread_count * 4) {
                    int base = start + tid * 4;
                    for (int index = 0; index < 4; index++) {
                        int d = base + index;
                        if (d < Hidden) {
                            float value = static_cast<float>(input_row[d]);
                            sum += value * value;
                        }
                    }
                }
                sum = simd_sum(sum);
                if (simd_group == 0) {
                    shared[simd_lane] = 0.0f;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    shared[simd_group] = sum;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    sum = simd_sum(shared[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_rms[0] = metal::precise::rsqrt(
                            sum / static_cast<float>(Hidden) + static_cast<float>(epsilon));
                    }
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int start = 0; start < Hidden; start += thread_count * 4) {
                    int base = start + tid * 4;
                    for (int index = 0; index < 4; index++) {
                        int d = base + index;
                        if (d < Hidden) {
                            auto normalized = \(scalar.metal)(
                                static_cast<float>(input_row[d]) * inverse_rms[0]);
                            auto weighted = \(scalar.metal)(
                                static_cast<float>(weight[d]) * static_cast<float>(normalized));
                            float gate = metal::precise::tanh(static_cast<float>(modulation_row[d]));
                            float combined = static_cast<float>(residual_row[d])
                                + gate * static_cast<float>(weighted);
                            output_row[d] = \(scalar.metal)(combined);
                        }
                    }
                }
                """
        )
        return kernel(
            [x, residual, weight, modulation, eps],
            template: [
                ("Hidden", shape.hidden),
                ("Sequence", shape.sequence),
                ("ModulationRows", shape.modulationRows),
                ("ModulationIndex", modulationIndex),
            ],
            grid: (normalizationThreadgroup(width: shape.hidden), shape.batch * shape.sequence, 1),
            threadGroup: (normalizationThreadgroup(width: shape.hidden), 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    private static func normalizedShape(
        _ x: MLXArray,
        weight: MLXArray,
        modulation: MLXArray
    ) -> (batch: Int, sequence: Int, hidden: Int, modulationRows: Int)? {
        guard x.ndim == 3,
              weight.ndim == 1,
              weight.dim(0) == x.dim(2),
              weight.dtype == x.dtype,
              modulation.dtype == x.dtype else {
            return nil
        }
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let hidden = x.dim(2)
        let modulationRows: Int
        if modulation.size == 4 * hidden {
            modulationRows = 1
        } else if modulation.size == batch * 4 * hidden {
            modulationRows = batch
        } else {
            return nil
        }
        return (batch, sequence, hidden, modulationRows)
    }

    private static func normalizationThreadgroup(width: Int) -> Int {
        let readsPerThread = 4
        if width > 4_096 {
            return 1_024
        }
        let required = (width + readsPerThread - 1) / readsPerThread
        return max(32, ((required + 31) / 32) * 32)
    }
}
