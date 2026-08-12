import MLX
import MLXFast
import MLXNN

public final class NemotronHMambaCache: @unchecked Sendable {
    var convolutionState: MLXArray?
    var recurrentState: MLXArray?
    public private(set) var offset = 0

    public init() {}

    public func fork() -> NemotronHMambaCache {
        let copy = NemotronHMambaCache()
        copy.convolutionState = convolutionState
        copy.recurrentState = recurrentState
        copy.offset = offset
        return copy
    }

    func advance(_ count: Int) {
        offset += count
    }
}

final class NemotronHMambaRMSNormGated: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    private let groupCount: Int
    private let groupSize: Int
    private let eps: Float

    init(dimensions: Int, groups: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.groupCount = groups
        self.groupSize = dimensions / groups
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let gated = MLXNN.silu(gate.asType(.float32)) * x.asType(.float32)
        let grouped = gated.reshaped(x.dim(0), x.dim(1), groupCount, groupSize)
        let normalized = grouped * rsqrt(
            (grouped * grouped).mean(axis: -1, keepDims: true) + eps
        )
        return (normalized.reshaped(x.shape) * weight.asType(.float32)).asType(x.dtype)
    }
}

#if os(macOS)
private enum NemotronHSSMMetalKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_h_ssm_scan",
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

func nemotronHSSMUpdate(
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
        let outputs = NemotronHSSMMetalKernel.kernel(
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

final class NemotronHMamba: NemotronHMixer {
    @ModuleInfo(key: "in_proj") var inputProjection: Linear
    @ModuleInfo(key: "conv1d") var convolution: Conv1d
    @ModuleInfo(key: "dt_bias") var timeStepBias: MLXArray
    @ModuleInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "D") var d: MLXArray
    @ModuleInfo(key: "norm") var norm: NemotronHMambaRMSNormGated
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    private let heads: Int
    private let headDimensions: Int
    private let stateSize: Int
    private let groups: Int
    private let convolutionDimensions: Int
    private let convolutionKernel: Int
    private let minimumTimeStep: Float
    private let maximumTimeStep: Float

    init(config: NemotronHConfig) {
        heads = config.mambaNumHeads
        headDimensions = config.mambaHeadDim
        stateSize = config.ssmStateSize
        groups = config.nGroups
        convolutionKernel = config.convKernel
        let intermediate = heads * headDimensions
        convolutionDimensions = intermediate + 2 * groups * stateSize
        minimumTimeStep = config.timeStepMin
        // Transformers currently treats the upper limit as infinity. A very
        // large finite template value keeps the Metal kernel portable.
        maximumTimeStep = Float.greatestFiniteMagnitude
        self._inputProjection.wrappedValue = Linear(
            config.hiddenSize,
            intermediate + convolutionDimensions + heads,
            bias: false
        )
        self._convolution.wrappedValue = Conv1d(
            inputChannels: convolutionDimensions,
            outputChannels: convolutionDimensions,
            kernelSize: config.convKernel,
            groups: convolutionDimensions,
            bias: true
        )
        self._timeStepBias.wrappedValue = MLXArray.ones([heads])
        self._aLog.wrappedValue = MLXArray.zeros([heads])
        self._d.wrappedValue = MLXArray.ones([heads])
        self._norm.wrappedValue = NemotronHMambaRMSNormGated(
            dimensions: intermediate,
            groups: config.nGroups,
            eps: config.normEps
        )
        self._outputProjection.wrappedValue = Linear(intermediate, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: NemotronHMambaCache?) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let intermediate = heads * headDimensions
        let projected = inputProjection(x)
        let gate = projected[.ellipsis, ..<intermediate]
        let convolutionInput = projected[
            .ellipsis,
            intermediate..<(intermediate + convolutionDimensions)
        ]
        let dt = projected[.ellipsis, (intermediate + convolutionDimensions)...]
        let previous = cache?.convolutionState ?? MLXArray.zeros(
            [batch, convolutionKernel - 1, convolutionDimensions],
            dtype: x.dtype
        )
        let padded = MLX.concatenated([previous, convolutionInput], axis: 1)
        cache?.convolutionState = padded[
            0...,
            (padded.dim(1) - convolutionKernel + 1)...,
            0...
        ]
        let convolved = MLXNN.silu(convolution(padded))
        let hidden = convolved[.ellipsis, ..<intermediate]
            .reshaped(batch, sequence, heads, headDimensions)
        let bStart = intermediate
        let cStart = bStart + groups * stateSize
        let b = convolved[.ellipsis, bStart..<cStart]
            .reshaped(batch, sequence, groups, stateSize)
        let c = convolved[.ellipsis, cStart...]
            .reshaped(batch, sequence, groups, stateSize)
        let result = nemotronHSSMUpdate(
            x: hidden,
            b: b,
            c: c,
            dt: dt,
            aLog: aLog,
            d: d,
            dtBias: timeStepBias,
            state: cache?.recurrentState,
            minimumTimeStep: minimumTimeStep,
            maximumTimeStep: maximumTimeStep
        )
        cache?.recurrentState = result.1
        cache?.advance(sequence)
        return outputProjection(norm(result.0.reshaped(batch, sequence, intermediate), gate: gate))
    }

    override func callAsFunction(_ x: MLXArray, cache: NemotronHLayerCache?) -> MLXArray {
        guard case .mamba(let mambaCache)? = cache else {
            return callAsFunction(x, cache: nil as NemotronHMambaCache?)
        }
        return callAsFunction(x, cache: mambaCache)
    }
}
