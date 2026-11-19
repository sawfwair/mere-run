import MLX
import MLXFast
import MLXNN

class HaruCodecLayer: Module {
    let stateShape: [Int]

    init(stateShape: [Int]) {
        self.stateShape = stateShape
    }

    func callAsFunction(_ input: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("Haru codec layers implement stateful inference.")
    }
}

final class HaruCausalConvolution: HaruCodecLayer {
    let weight: MLXArray
    let bias: MLXArray
    private let kernel: Int
    private let hop: Int

    init(input: Int, output: Int, kernel: Int, hop: Int = 1) {
        weight = MLXArray.zeros([output, kernel * input])
        bias = MLXArray.zeros([output])
        self.kernel = kernel
        self.hop = hop
        super.init(stateShape: [kernel - hop, input])
    }

    override func callAsFunction(_ input: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        let remainder = input.dim(0) % hop
        let rows = remainder == 0 ? input : concatenated([
            input, MLXArray.zeros([hop - remainder, input.dim(1)], dtype: input.dtype),
        ])
        let context = concatenated([state, rows])
        let windows: MLXArray
        if hop == 1 {
            windows = concatenated((0..<kernel).map { context[$0..<($0 + rows.dim(0))] }, axis: 1)
        } else {
            let hops = context.reshaped(-1, hop * input.dim(1))
            windows = concatenated([hops[..<(hops.dim(0) - 1)], hops[1...]], axis: 1)
        }
        return (addMM(bias, windows, weight.T), context[rows.dim(0)...])
    }
}

final class HaruCausalUpsampling: HaruCodecLayer {
    let weight: MLXArray
    let bias: MLXArray
    private let stride: Int
    private let channels: Int

    init(input: Int, output: Int, stride: Int) {
        weight = MLXArray.zeros([2 * stride * output, input])
        bias = MLXArray.zeros([output])
        self.stride = stride
        self.channels = output
        super.init(stateShape: [stride, output])
    }

    override func callAsFunction(_ input: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        let halves = matmul(input, weight.T).reshaped(-1, channels, 2, stride).transposed(0, 2, 3, 1)
        let first = halves[0..., 0]
        let second = halves[0..., 1]
        let overlap = concatenated([state.expandedDimensions(axis: 0), second[..<(second.dim(0) - 1)]])
        let output = (first + overlap).reshaped(-1, channels) + bias
        return (output, second[second.dim(0) - 1])
    }
}

private final class HaruDepthwiseFilter: Module {
    let weight: MLXArray
    let bias: MLXArray

    init(channels: Int) {
        weight = MLXArray.zeros([channels, 7])
        bias = MLXArray.zeros([channels])
    }
}

final class HaruCodecBlock: HaruCodecLayer {
    private let norm: HaruGain
    private let conv: HaruDepthwiseFilter
    private let gamma: MLXArray
    @ModuleInfo(key: "ffn_norm") private var ffnNorm: HaruGain
    private let fc1: Linear
    private let fc2: Linear
    @ParameterInfo(key: "ffn_gamma") private var ffnGamma: MLXArray
    private let epsilon: Float
    private let channels: Int

    init(channels: Int, epsilon: Float) {
        self.channels = channels
        self.epsilon = epsilon
        self.norm = HaruGain(width: channels)
        self.conv = HaruDepthwiseFilter(channels: channels)
        self.gamma = MLXArray.ones([channels])
        self._ffnNorm.wrappedValue = HaruGain(width: channels)
        self.fc1 = Linear(channels, 4 * channels)
        self.fc2 = Linear(4 * channels, channels)
        self._ffnGamma.wrappedValue = MLXArray.ones([channels])
        super.init(stateShape: [6, channels])
    }

    override func callAsFunction(_ input: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        let normalized = MLXFast.rmsNorm(input, weight: norm.weight, eps: epsilon)
        let context = concatenated([state, normalized])
        let filter = conv.weight.expandedDimensions(axis: -1).asType(.float32)
        let mixed = conv1d(context.expandedDimensions(axis: 0).asType(.float32), filter, groups: channels)[0]
        let residual = (input.asType(.float32) + gamma.asType(.float32) * (mixed + conv.bias.asType(.float32)))
            .asType(input.dtype)
        let ffnInput = MLXFast.rmsNorm(residual, weight: ffnNorm.weight, eps: epsilon)
        let output = residual + ffnGamma * fc2(gelu(fc1(ffnInput)))
        return (output, context[input.dim(0)...])
    }
}
