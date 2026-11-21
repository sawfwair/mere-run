import MLX

#if os(macOS)
enum Inkling2ShortConvolution {
    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_short_convolution",
        inputNames: ["context", "taps", "dynamic"],
        outputNames: ["output"],
        source: """
            const uint elem = thread_position_in_grid.x;
            const uint channel = elem % CHANNELS;
            const uint token = elem / CHANNELS;
            const uint batch = token / SEQUENCE;
            const uint first = elem + batch * (TAPS - 1) * CHANNELS;
            const uint current = first + (TAPS - 1) * CHANNELS;
            float sum = context[current];
            if constexpr (DYNAMIC) {
                const uint head = channel / (CHANNELS / HEADS);
                const uint projection = (token * HEADS + head) * TAPS;
                for (uint lag = 0; lag < TAPS; ++lag) {
                    const float tap = float(taps[lag * CHANNELS + channel]) + float(dynamic[projection + lag]);
                    sum += context[current - lag * CHANNELS] * tap;
                }
                output[elem] = T(sum / sqrt(float(2 * TAPS + 1)));
            } else {
                for (uint tap = 0; tap < TAPS; ++tap) {
                    sum += context[first + tap * CHANNELS] * float(taps[channel * TAPS + tap]);
                }
                output[elem] = T(sum / sqrt(float(TAPS + 1)));
            }
            """
    )

    static func apply(_ input: MLXArray, context: MLXArray, taps: MLXArray, kernelSize: Int,
                      dynamic: MLXArray? = nil, heads: Int = 1) -> MLXArray {
        kernel([context, taps, dynamic ?? taps],
               template: [("T", input.dtype), ("CHANNELS", input.dim(2)), ("SEQUENCE", input.dim(1)),
                          ("TAPS", kernelSize), ("HEADS", heads), ("DYNAMIC", dynamic != nil)],
               grid: (input.size, 1, 1), threadGroup: (min(input.size, 256), 1, 1),
               outputShapes: [input.shape], outputDTypes: [input.dtype])[0]
    }
}
#endif
