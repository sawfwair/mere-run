import MLX

#if os(macOS)
/// Updates the tentative convolution history and applies the rounded residual in one dispatch.
enum Inkling2StaticResidualConvolution {
    private static let kernel = MLXFast.metalKernel(
        name: "inkling_probe_static_residual_convolution", inputNames: ["x", "prior", "weights", "residual"],
        outputNames: ["history", "out"], source: """
            const uint elem = thread_position_in_grid.x;
            const uint channel = elem % CHANNELS;
            const uint token = elem / CHANNELS;
            if (token < TAPS - 1) {
                history[elem] = prior[elem];
                return;
            }
            const uint row = token - (TAPS - 1);
            const uint index = row * CHANNELS + channel;
            const float current = float(x[index]);
            history[elem] = current;
            float sum = current;
            #pragma unroll
            for (uint tap = 0; tap < TAPS; ++tap) {
                const uint position = row + tap;
                const float value = position < TAPS - 1 ? prior[position * CHANNELS + channel]
                                                        : float(x[(position - (TAPS - 1)) * CHANNELS + channel]);
                sum += value * float(weights[channel * TAPS + tap]);
            }
            const float rounded = float(T(sum / sqrt(float(TAPS + 1))));
            out[index] = O(ADD_RESIDUAL ? float(residual[index]) + rounded : rounded);
            """
    )

    static func apply(_ input: MLXArray, prior: MLXArray, weights: MLXArray, residual: MLXArray,
                      taps: Int, addResidual: Bool, outputDType: DType) -> (MLXArray, MLXArray) {
        let channels = input.dim(2), count = input.dim(1) + taps - 1
        let results = kernel(
            [input, prior, weights, residual],
            template: [("T", input.dtype), ("O", outputDType), ("CHANNELS", channels), ("TAPS", taps), ("ADD_RESIDUAL", addResidual)],
            grid: (channels * count, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, count, channels], input.shape], outputDTypes: [.float32, outputDType]
        )
        return (results[0], results[1])
    }
}
#endif
