#if os(macOS)
import MLX
import MLXFast

/// Runs independent affine-Q3 gather QMV routes in one Metal launch while
/// retaining MLX's exact one-vector reduction order for every route.
enum SmallBatchAffineGatherQMV {
    private static let fastKernel = makeKernel(
        name: "mere_affine3_g64_gather_qmv_fast_v1",
        implementation: "qmv_fast_impl"
    )
    private static let regularKernel = makeKernel(
        name: "mere_affine3_g64_gather_qmv_v1",
        implementation: "qmv_impl"
    )

    private static func makeKernel(name: String, implementation: String) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: name,
            inputNames: ["w", "scales", "biases", "x", "indices"],
            outputNames: ["y"],
            source: """
                const int route = int(threadgroup_position_in_grid.x);
                const int k = x_shape[x_ndim - 1];
                const int n = w_shape[w_ndim - 2];
                const int packed = w_shape[w_ndim - 1];
                const int groups = k / 64;
                const int expert = int(indices[route]);
                const device uint32_t* route_w = w + expert * n * packed;
                const device bfloat16_t* route_scales = scales + expert * n * groups;
                const device bfloat16_t* route_biases = biases + expert * n * groups;
                const device bfloat16_t* route_x = x + route * k;
                device bfloat16_t* route_y = y + route * n;
                const uint3 local_tid = uint3(0, threadgroup_position_in_grid.y, 0);
                \(implementation)<bfloat16_t, 64, 3>(
                    route_w, route_scales, route_biases, route_x, route_y,
                    x_shape[x_ndim - 1], w_shape[w_ndim - 2],
                    local_tid, simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
                """,
            header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n",
            ensureRowContiguous: true
        )
    }

    static func matmul(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        indices: MLXArray,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              groupSize == 64, bits == 3,
              x.dtype == .bfloat16,
              weight.dtype == .uint32,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              indices.dtype == .uint32,
              x.ndim == 3, x.dim(1) == 1,
              weight.ndim == 3,
              scales.ndim == 3,
              biases.ndim == 3 else {
            return nil
        }

        let routeCount = x.dim(0)
        let inputSize = x.dim(2)
        let outputSize = weight.dim(1)
        guard routeCount == indices.size,
              inputSize % groupSize == 0,
              outputSize >= 8, outputSize % 8 == 0,
              weight.dim(2) * 32 == inputSize * bits,
              scales.shape == [weight.dim(0), outputSize, inputSize / groupSize],
              biases.shape == scales.shape else {
            return nil
        }

        let kernel = inputSize.isMultiple(of: 512) ? fastKernel : regularKernel
        return kernel(
            [weight, scales, biases, x, indices],
            grid: (routeCount * 32, (outputSize / 8) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[routeCount, 1, outputSize]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
#endif
