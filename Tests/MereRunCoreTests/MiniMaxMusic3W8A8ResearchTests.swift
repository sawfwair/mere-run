import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest
@testable import MereRunCore

/// Env-gated production-shape oracle for MiniMax Music 3 ConvRot W8A8.
///
/// This intentionally stays out of the runtime until the complete projection,
/// including activation rotation and quantization, beats MLX's BF16 GEMM on a
/// supported Apple GPU. Run with:
///
/// ```bash
/// MERERUN_TEST_MLX_DEVICE=gpu MERERUN_MINIMAX_MUSIC3_W8A8_BENCH=1 \
///   swift test -c release --filter MiniMaxMusic3W8A8ResearchTests
/// ```
final class MiniMaxMusic3W8A8ResearchTests: MereRunCoreTestCase {
    func testConvRotW8A8ProjectionAtFlowShape() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_MINIMAX_MUSIC3_W8A8_BENCH"] == "1" else {
            throw XCTSkip("Set MERERUN_MINIMAX_MUSIC3_W8A8_BENCH=1 to run the W8A8 research lane.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The W8A8 research lane requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }

        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_MINIMAX_MUSIC3_W8A8_ROWS"] ?? "") ?? 690)
        let outputs = max(
            MiniMaxMusic3W8A8Oracle.outputsPerThreadgroup,
            Int(environment["MERERUN_MINIMAX_MUSIC3_W8A8_OUTPUTS"] ?? "") ?? 16_384
        )
        let rounds = max(1, Int(environment["MERERUN_MINIMAX_MUSIC3_W8A8_ROUNDS"] ?? "") ?? 2)
        let iterations = max(
            1,
            Int(environment["MERERUN_MINIMAX_MUSIC3_W8A8_ITERATIONS"] ?? "") ?? 2
        )

        MLXRandom.seed(8_803)
        let input = MLXRandom.normal([rows, MiniMaxMusic3W8A8Oracle.inputWidth])
            .asType(.bfloat16)
        let weight = MLXRandom.normal([outputs, MiniMaxMusic3W8A8Oracle.inputWidth])
            .asType(.bfloat16)
        let preparedWeight = try XCTUnwrap(MiniMaxMusic3W8A8Oracle.rotateAndQuantize(weight))
        MLX.eval(preparedWeight.values, preparedWeight.scales)

        let reference = MLX.matmul(input, weight.transposed())
        let candidate = try XCTUnwrap(
            MiniMaxMusic3W8A8Oracle.project(input: input, weight: preparedWeight)
        )
        MLX.eval(reference, candidate)

        let referenceF32 = reference.asType(.float32)
        let candidateF32 = candidate.asType(.float32)
        let error = candidateF32 - referenceF32
        let cosine = (
            MLX.sum(referenceF32 * candidateF32)
                / MLX.sqrt(MLX.sum(referenceF32 * referenceF32)
                    * MLX.sum(candidateF32 * candidateF32))
        ).item(Float.self)
        let relativeRMSE = (
            MLX.sqrt(MLX.mean(error * error))
                / MLX.sqrt(MLX.mean(referenceF32 * referenceF32))
        ).item(Float.self)

        XCTAssertGreaterThan(cosine, 0.999)
        XCTAssertLessThan(relativeRMSE, 0.05)

        let denseMilliseconds = benchmarkMilliseconds(
            rounds: rounds,
            iterations: iterations
        ) {
            MLX.matmul(input, weight.transposed())
        }
        let w8a8Milliseconds = benchmarkMilliseconds(
            rounds: rounds,
            iterations: iterations
        ) {
            MiniMaxMusic3W8A8Oracle.project(input: input, weight: preparedWeight)!
        }
        let report: [String: Any] = [
            "candidate_ms": w8a8Milliseconds,
            "cosine": cosine,
            "dense_bf16_ms": denseMilliseconds,
            "input_width": MiniMaxMusic3W8A8Oracle.inputWidth,
            "output_width": outputs,
            "relative_rmse": relativeRMSE,
            "rows": rows,
            "speedup": denseMilliseconds / w8a8Milliseconds,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: report,
            options: [.prettyPrinted, .sortedKeys]
        )
        print(String(decoding: data, as: UTF8.self))
    }

    private func benchmarkMilliseconds(
        rounds: Int,
        iterations: Int,
        _ body: () -> MLXArray
    ) -> Double {
        MLX.eval(body())
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<rounds {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations {
                MLX.eval(body())
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            best = min(best, Double(elapsed) / 1_000_000 / Double(iterations))
        }
        return best
    }
}

private struct MiniMaxMusic3W8A8Rows {
    let values: MLXArray
    let scales: MLXArray
}

private enum MiniMaxMusic3W8A8Oracle {
    static let inputWidth = 2_048
    static let convRotGroupSize = 64
    static let threadCount = 256
    static let outputsPerThreadgroup = 8

    static func rotateAndQuantize(_ input: MLXArray) -> MiniMaxMusic3W8A8Rows? {
        guard Device.defaultDevice().deviceType == .gpu,
              input.dtype == .bfloat16,
              input.ndim == 2,
              input.dim(0) > 0,
              input.dim(1) == inputWidth else {
            return nil
        }
        let rows = input.dim(0)
        let outputs = rotateAndQuantizeKernel(
            [input],
            grid: (threadCount, rows, 1),
            threadGroup: (threadCount, 1, 1),
            outputShapes: [input.shape, [rows]],
            outputDTypes: [.int8, .float32]
        )
        return MiniMaxMusic3W8A8Rows(values: outputs[0], scales: outputs[1])
    }

    static func project(
        input: MLXArray,
        weight: MiniMaxMusic3W8A8Rows
    ) -> MLXArray? {
        guard let quantizedInput = rotateAndQuantize(input),
              weight.values.dtype == .int8,
              weight.values.ndim == 2,
              weight.values.dim(1) == inputWidth,
              weight.scales.shape == [weight.values.dim(0)] else {
            return nil
        }
        let rows = input.dim(0)
        let outputs = weight.values.dim(0)
        let tiles = (outputs + outputsPerThreadgroup - 1) / outputsPerThreadgroup
        return projectionKernel(
            [
                quantizedInput.values,
                quantizedInput.scales,
                weight.values,
                weight.scales,
            ],
            grid: (tiles * 64, rows, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[rows, outputs]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    /// Applies Comfy-Kitchen's normalized regular-Hadamard H64 independently
    /// to every 64-value group, then uses one symmetric INT8 scale per row.
    private static let rotateAndQuantizeKernel = MLXFast.metalKernel(
        name: "mere_minimax_music3_convrot_h64_quantize_i8_h2048_v1",
        inputNames: ["input"],
        outputNames: ["quantized", "scales"],
        source: """
            constexpr uint input_width = 2048;
            constexpr uint convrot_group_size = 64;
            constexpr uint simd_width = 32;
            constexpr float normalization = 0.125f;

            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_index_in_threadgroup;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint row_offset = row * input_width;
            threadgroup float rotated[input_width];
            threadgroup float partial_maxima[simd_width];
            threadgroup float maximum[1];

            for (uint dimension = tid; dimension < input_width;
                 dimension += threads_per_threadgroup.x) {
                rotated[dimension] = float(input[row_offset + dimension]);
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (uint stride = 1; stride < convrot_group_size; stride *= 4) {
                for (uint tuple = tid; tuple < input_width / 4;
                     tuple += threads_per_threadgroup.x) {
                    uint block = tuple / stride;
                    uint offset = tuple - block * stride;
                    uint base = block * 4 * stride + offset;
                    float a = rotated[base];
                    float b = rotated[base + stride];
                    float c = rotated[base + 2 * stride];
                    float d = rotated[base + 3 * stride];
                    rotated[base] = a + b + c - d;
                    rotated[base + stride] = a + b - c + d;
                    rotated[base + 2 * stride] = a - b + c + d;
                    rotated[base + 3 * stride] = -a + b + c + d;
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }

            float local_max = 0.0f;
            for (uint dimension = tid; dimension < input_width;
                 dimension += threads_per_threadgroup.x) {
                float value = rotated[dimension] * normalization;
                rotated[dimension] = value;
                local_max = metal::max(local_max, metal::fabs(value));
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
            for (uint dimension = tid; dimension < input_width;
                 dimension += threads_per_threadgroup.x) {
                int value = int(rint(rotated[dimension] * quantize_scale));
                quantized[row_offset + dimension] = int8_t(
                    metal::clamp(value, -127, 127));
            }
        """,
        ensureRowContiguous: true
    )

    /// Portable packed W8A8 oracle. Apple GPU generations without native INT8
    /// matrix units execute these integer products in ordinary shader lanes;
    /// that is precisely the physical-path question this benchmark answers.
    private static let projectionKernel = MLXFast.metalKernel(
        name: "mere_minimax_music3_convrot_w8a8_projection_h2048_v1",
        inputNames: ["input", "input_scales", "weight", "weight_scales"],
        outputNames: ["output"],
        source: """
            constexpr uint input_width = 2048;
            constexpr uint values_per_thread = 8;
            constexpr uint block_size = 256;
            constexpr uint results_per_simdgroup = 4;
            constexpr uint projection_simdgroups = 2;
            constexpr uint outputs_per_threadgroup =
                results_per_simdgroup * projection_simdgroups;

            uint tile = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint output_width = uint(weight_shape[0]);
            uint output_row = tile * outputs_per_threadgroup
                + simd_group * results_per_simdgroup;

            const device int8_t* input_values = input
                + row * input_width + lane * values_per_thread;
            const device int8_t* weight_values = weight
                + output_row * input_width + lane * values_per_thread;
            float results[results_per_simdgroup] = {0.0f};

            for (uint block = 0; block < input_width; block += block_size) {
                int activation[values_per_thread];
                for (uint index = 0; index < values_per_thread; ++index) {
                    activation[index] = int(input_values[index]);
                }
                for (uint output_index = 0;
                     output_index < results_per_simdgroup;
                     ++output_index) {
                    uint global_output = output_row + output_index;
                    if (global_output < output_width) {
                        const device int8_t* weight_row = weight_values
                            + output_index * input_width;
                        int dot = 0;
                        for (uint index = 0; index < values_per_thread; ++index) {
                            dot += activation[index] * int(weight_row[index]);
                        }
                        results[output_index] += float(dot);
                    }
                }
                input_values += block_size;
                weight_values += block_size;
            }

            float input_scale = input_scales[row];
            for (uint output_index = 0;
                 output_index < results_per_simdgroup;
                 ++output_index) {
                uint global_output = output_row + output_index;
                if (global_output < output_width) {
                    float sum = simd_sum(results[output_index]);
                    if (lane == 0) {
                        output[row * output_width + global_output] = bfloat16_t(
                            sum * input_scale * weight_scales[global_output]);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )
}
