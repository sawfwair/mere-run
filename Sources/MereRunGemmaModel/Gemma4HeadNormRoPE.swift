import Foundation
import MLX
import Cmlx

/// One dispatch for BF16 Q/K/V normalization and half-layout rotary positions.
/// The normalization and learned scale each retain their BF16 rounding point.
final class Gemma4HeadNormRoPE {
    private struct Shape: Hashable {
        let batch: Int
        let count: Int
    }
    private let kernel: mlx_fast_metal_kernel
    private var configurations: [Shape: mlx_fast_metal_kernel_config] = [:]
    private let numHeads: Int
    private let numKVHeads: Int
    private let dim: Int
    private let rotHalf: Int
    private let proportional: Bool
    private let epsilon: MLXArray
    private let frequencies: MLXArray
    private let logBase: MLXArray

    init(heads: Int, kvHeads: Int, dim: Int, rotatedDims: Int, proportional: Bool, base: Float, eps: Float) throws {
        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.dim = dim
        self.rotHalf = rotatedDims / 2
        self.proportional = proportional
        self.epsilon = MLXArray([eps])
        self.frequencies = pow(MLXArray(base), MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dim))
        self.logBase = MLXArray([log2(base)])
        let url = Bundle.module.url(forResource: "Gemma4HeadNormRoPE", withExtension: "metal", subdirectory: "Kernels")!
        let source = try String(contentsOf: url, encoding: .utf8)
        let inputs = mlx_vector_string_new(), outputs = mlx_vector_string_new()
        defer { mlx_vector_string_free(inputs); mlx_vector_string_free(outputs) }
        for name in ["q_raw", "k_raw", "v_raw", "q_weight", "k_weight", "eps", "offset", "freqs", "log_base"] {
            mlx_vector_string_append_value(inputs, name)
        }
        for name in ["q_out", "k_out", "v_out"] { mlx_vector_string_append_value(outputs, name) }
        self.kernel = mlx_fast_metal_kernel_new("gemma4_head_norm_rope", inputs, outputs, source, "", true, false)
    }

    deinit {
        for value in configurations.values { mlx_fast_metal_kernel_config_free(value) }
        mlx_fast_metal_kernel_free(kernel)
    }

    private func configuration(batch: Int, count: Int) -> mlx_fast_metal_kernel_config {
        let shape = Shape(batch: batch, count: count)
        if let config = configurations[shape] { return config }
        let config = mlx_fast_metal_kernel_config_new()
        for (name, value) in [("QH", numHeads), ("KH", numKVHeads), ("D", dim), ("T", count), ("RotHalf", rotHalf)] {
            mlx_fast_metal_kernel_config_add_template_arg_int(config, name, Int32(value))
        }
        mlx_fast_metal_kernel_config_add_template_arg_bool(config, "Proportional", proportional)
        mlx_fast_metal_kernel_config_set_grid(config, Int32(dim / 4), Int32(batch * count * (numHeads + 2 * numKVHeads)), 1)
        mlx_fast_metal_kernel_config_set_thread_group(config, Int32(dim / 4), 1, 1)
        for heads in [numHeads, numKVHeads, numKVHeads] {
            let dimensions = [batch, heads, count, dim].map(Int32.init)
            mlx_fast_metal_kernel_config_add_output_arg(config, dimensions, dimensions.count, MLX_BFLOAT16)
        }
        configurations[shape] = config
        return config
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray,
                        queryWeight: MLXArray, keyWeight: MLXArray, offset: MLXArray) -> (MLXArray, MLXArray, MLXArray)? {
        guard Gemma4MetalExperts.isSupported, [256, 512].contains(dim), proportional || rotHalf == dim / 2,
              queries.ndim == 3, queries.dim(1) <= 16,
              queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16,
              queryWeight.dtype == .bfloat16, keyWeight.dtype == .bfloat16 else { return nil }
        let batch = queries.dim(0), count = queries.dim(1)
        let arrays = [queries, keys, values, queryWeight, keyWeight, epsilon, offset, frequencies, logBase]
        let stream = StreamOrDevice.default
        let inputs = withExtendedLifetime(arrays) { mlx_vector_array_new_data(arrays.map(\.ctx), arrays.count) }
        defer { mlx_vector_array_free(inputs) }
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        _ = withExtendedLifetime(stream) {
            mlx_fast_metal_kernel_apply(&result, kernel, inputs, configuration(batch: batch, count: count), stream.ctx)
        }
        let outputs = (0..<3).map { index in
            var array = mlx_array_new()
            mlx_vector_array_get(&array, result, index)
            return MLXArray(array)
        }
        return (outputs[0], outputs[1], outputs[2])
    }
}
