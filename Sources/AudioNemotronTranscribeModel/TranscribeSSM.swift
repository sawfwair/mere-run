import MLX
import MLXFast
import MLXNN

#if os(macOS)
private enum TranscribeSSMMetalKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_transcribe_ssm_scan",
        inputNames: [
            "x", "B", "C", "dt", "A_log", "D", "dt_bias", "state_in",
            "dt_min", "dt_max", "T",
        ],
        outputNames: ["y", "state_out"],
        source: """
            auto bh = thread_position_in_grid.z;
            auto b = bh / H;
            auto h = bh % H;
            auto d = thread_position_in_grid.y;
            auto lane = thread_index_in_simdgroup;
            auto group = h / (H / G);

            auto x_ = x + (b * T * H + h) * HD + d;
            auto b_ = B + (b * T * G + group) * N;
            auto c_ = C + (b * T * G + group) * N;
            auto dt_ = dt + b * T * H + h;
            auto y_ = y + (b * T * H + h) * HD + d;
            auto i_state = state_in + (bh * HD + d) * N;
            auto o_state = state_out + (bh * HD + d) * N;

            constexpr int n_per_lane = N / 32;
            float state[n_per_lane];
            for (int i = 0; i < n_per_lane; ++i) {
                state[i] = static_cast<float>(i_state[lane * n_per_lane + i]);
            }
            float a = -exp(static_cast<float>(A_log[h]));
            float skip = static_cast<float>(D[h]);
            float bias = static_cast<float>(dt_bias[h]);

            for (int t = 0; t < T; ++t) {
                float raw_dt = static_cast<float>(*dt_) + bias;
                float delta = max(raw_dt, 0.0f) + log1p(exp(-abs(raw_dt)));
                delta = clamp(delta, static_cast<float>(dt_min), static_cast<float>(dt_max));
                float decay = exp(a * delta);
                float xv = static_cast<float>(*x_);
                float partial = 0.0f;
                for (int i = 0; i < n_per_lane; ++i) {
                    auto n = lane * n_per_lane + i;
                    state[i] = state[i] * decay
                        + delta * static_cast<float>(b_[n]) * xv;
                    partial += state[i] * static_cast<float>(c_[n]);
                }
                partial = simd_sum(partial);
                if (lane == 0) {
                    *y_ = static_cast<InT>(partial + xv * skip);
                }
                x_ += H * HD;
                b_ += G * N;
                c_ += G * N;
                dt_ += H;
                y_ += H * HD;
            }
            for (int i = 0; i < n_per_lane; ++i) {
                o_state[lane * n_per_lane + i] = state[i];
            }
            """
    )
}
#endif

func transcribeSSMUpdate(
    x: MLXArray,
    b: MLXArray,
    c: MLXArray,
    dt: MLXArray,
    aLog: MLXArray,
    d: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    minimumTimeStep: Float,
    maximumTimeStep: Float,
    useMetalKernel: Bool = true
) -> (MLXArray, MLXArray) {
    let batch = x.dim(0)
    let sequence = x.dim(1)
    let heads = x.dim(2)
    let headDimensions = x.dim(3)
    let groups = b.dim(2)
    let stateSize = b.dim(3)
    let initial = state ?? MLXArray.zeros(
        [batch, heads, headDimensions, stateSize],
        dtype: .float32
    )

    #if os(macOS)
    if useMetalKernel,
       Device.defaultDevice().deviceType == .gpu,
       stateSize.isMultiple(of: 32) {
        let outputs = TranscribeSSMMetalKernel.kernel(
            [
                x, b, c, dt, aLog, d, dtBias, initial,
                MLXArray(minimumTimeStep), MLXArray(maximumTimeStep), sequence,
            ],
            template: [
                ("InT", x.dtype),
                ("H", heads),
                ("HD", headDimensions),
                ("G", groups),
                ("N", stateSize),
            ],
            grid: (32, headDimensions, batch * heads),
            threadGroup: (32, 4, 1),
            outputShapes: [x.shape, initial.shape],
            outputDTypes: [x.dtype, .float32]
        )
        return (outputs[0], outputs[1])
    }
    #endif

    let repeatFactor = heads / groups
    let expandedB = MLX.repeated(b.asType(.float32), count: repeatFactor, axis: 2)
    let expandedC = MLX.repeated(c.asType(.float32), count: repeatFactor, axis: 2)
    let delta = MLX.minimum(
        MLX.maximum(
            MLXNN.softplus(dt.asType(.float32) + dtBias.reshaped(1, 1, heads)),
            MLXArray(minimumTimeStep)
        ),
        MLXArray(maximumTimeStep)
    )
    let a = -MLX.exp(aLog.asType(.float32)).reshaped(1, heads, 1, 1)
    var recurrent = initial
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(sequence)
    for index in 0..<sequence {
        let xStep = x[0..., index, 0..., 0...].asType(.float32)
        let dtStep = delta[0..., index, 0...]
        let decay = MLX.exp(a * dtStep.reshaped(batch, heads, 1, 1))
        let bStep = expandedB[0..., index, 0..., 0...]
        recurrent = recurrent * decay
            + xStep.expandedDimensions(axis: -1)
            * dtStep.reshaped(batch, heads, 1, 1)
            * bStep.expandedDimensions(axis: -2)
        let cStep = expandedC[0..., index, 0..., 0...]
        let output = (recurrent * cStep.expandedDimensions(axis: -2)).sum(axis: -1)
            + xStep * d.reshaped(1, heads, 1)
        outputs.append(output.asType(x.dtype))
    }
    return (MLX.stacked(outputs, axis: 1), recurrent)
}
