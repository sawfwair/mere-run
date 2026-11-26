// Adapted from oMLX (633d6cf62971262149c2b2477361487958da3915), Apache-2.0.
// The upstream code includes Apple MLX Metal routines, copyright (c) 2024-25 Apple Inc., MIT.
// See THIRD_PARTY_NOTICES.md.
#if os(macOS)
import Foundation
import MLX

enum Inkling2BandedPrefill {
    static let enabled = ProcessInfo.processInfo.environment["MERERUN_INKLING2_NAX_PREFILL"] == "1"
        && GPU.deviceInfo().architecture == "applegpu_g17s"

    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_nax_banded_prefill",
        inputNames: ["q", "k", "v", "mask", "sinks", "state", "params"],
        outputNames: ["out", "state_out"],
        source: #"""
            threadgroup float xchg[2 * 2 * 16 * 32];
            omlx_nax::attention_nax_bdv<bfloat16_t, 32, 32, 128, 128, 2, 2,
                false, false, false, true, true, true, true, FIRST_PASS, LAST_PASS, bool, float>(
                q, k, v, out,
                reinterpret_cast<const device omlx_nax::AttnParams*>(params),
                q_strides, k_strides, v_strides, mask_strides,
                mask, sinks, state, state_out, xchg,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, threadgroup_position_in_grid);
            """#,
        header: metalSource + "\n",
        ensureRowContiguous: false
    )

    static func apply(query: MLXArray, keys: MLXArray, values: MLXArray, relative: MLXArray,
                      sinks: MLXArray) -> MLXArray {
        let rows = query.dim(1), heads = query.dim(2), width = query.dim(3)
        let keyRows = keys.dim(2)
        let queryGroups = (rows + 31) / 32, keyGroups = (keyRows + 31) / 32
        let passes = max(1, Int((Double(keyGroups * 32) / 4096).rounded()))
        let q = query.transposed(0, 2, 1, 3)
        let bias = relative.asType(.bfloat16).contiguous()
        let packedSinks = sinks.asType(.bfloat16).contiguous()
        var output = MLXArray(Float(0))
        var state = MLXArray.zeros([8], dtype: .float32)
        for pass in 0..<passes {
            let first = pass == 0, last = pass == passes - 1
            // AttnParams uses 32-bit fields. The scale occupies its FP32 bit pattern.
            let scalars: [Int] = [
                1, heads, width, rows, keyRows, heads / keys.dim(1),
                Int(Int32(bitPattern: (1 / sqrt(Float(width))).bitPattern)),
                queryGroups, keyGroups, rows / 32, keyRows / 32, rows % 32, keyRows % 32, keyRows - rows,
                pass * keyGroups / passes, (pass + 1) * keyGroups / passes, relative.dim(-1), 0, 0,
            ]
            let params = MLXArray(scalars.map(Int32.init)).view(dtype: .uint8)
            let results = kernel(
                [q, keys, values, bias, packedSinks, state, params],
                template: [("FIRST_PASS", first), ("LAST_PASS", last)],
                grid: (queryGroups * 128, heads, 1), threadGroup: (128, 1, 1),
                outputShapes: [last ? [1, rows, heads, width] : [1],
                               last ? [1] : [heads * queryGroups * 32 * (width + 2)]],
                outputDTypes: [.bfloat16, .float32]
            )
            output = results[0]
            state = results[1]
        }
        return output.reshaped(1, rows, heads * width)
    }
}
#endif
