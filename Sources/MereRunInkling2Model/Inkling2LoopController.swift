import MLX

#if os(macOS)
struct Inkling2LoopController {
    private static let prepare = MLXFast.metalKernel(
        name: "mererun_inkling2_controller_input",
        inputNames: ["input", "message", "epsilon"],
        outputNames: ["paired"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const uint base = row * WIDTH;
            threadgroup float2 sums[4];
            float2 squares = 0;
            for (uint d = tid; d < WIDTH; d += 128) {
                const float2 v = float2(float(input[base + d]), float(message[base + d]));
                squares += v * v;
            }
            squares = simd_sum(squares);
            if (lane == 0) sums[simd] = squares;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float2 scale = rsqrt((sums[0] + sums[1] + sums[2] + sums[3]) / float(WIDTH) + epsilon);
            for (uint d = tid; d < WIDTH; d += 128) {
                paired[2 * base + d] = T(float(input[base + d]) * scale.x);
                paired[2 * base + WIDTH + d] = T(float(message[base + d]) * scale.y);
            }
            """
    )

    private static let finish = MLXFast.metalKernel(
        name: "mererun_inkling2_controller_norm",
        inputNames: ["input", "message", "gates", "weight", "epsilon"],
        outputNames: ["output"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint controller = row / input_shape[1];
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const uint base = row * WIDTH;
            threadgroup float sums[4];
            threadgroup float mixed[WIDTH];
            float squares = 0;
            for (uint d = tid; d < WIDTH; d += 128) {
                const float forget = tanh(gates[2 * base + d]);
                const float gain = tanh(gates[2 * base + WIDTH + d]);
                const float value = (1 - forget) * float(input[base + d]) + gain * float(message[base + d]);
                mixed[d] = value;
                squares += value * value;
            }
            squares = simd_sum(squares);
            if (lane == 0) sums[simd] = squares;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scale = rsqrt((sums[0] + sums[1] + sums[2] + sums[3]) / float(WIDTH) + epsilon);
            for (uint d = tid; d < WIDTH; d += 128) {
                output[base + d] = T(mixed[d] * scale * float(weight[controller * WIDTH + d]));
            }
            """
    )

    let down: MLXArray
    let up: MLXArray
    let normWeights: MLXArray

    func apply(_ inputs: [MLXArray], messages: [MLXArray], epsilon: Float, dtype: DType) -> [MLXArray] {
        let shape = inputs[0].shape
        let width = shape[2], channels = inputs.count, rows = shape[0] * shape[1]
        let x = stacked(inputs, axis: 0).reshaped(channels, rows, width)
        let m = stacked(messages, axis: 0).reshaped(channels, rows, width)
        let work = channels * rows * 128
        let paired = Self.prepare(
            [x, m, epsilon], template: [("WIDTH", width), ("T", down.dtype)],
            grid: (work, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[channels, rows, 2 * width]], outputDTypes: [down.dtype]
        )[0]
        let hidden = matmul(paired, down).asType(.float32)
        let gates = matmul(hidden, up)
        let output = Self.finish(
            [x, m, gates, normWeights, epsilon], template: [("WIDTH", width), ("T", dtype)],
            grid: (work, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[channels] + shape], outputDTypes: [dtype]
        )[0]
        return (0..<channels).map { output[$0] }
    }
}
#endif
