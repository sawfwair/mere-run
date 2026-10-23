// Adapted from Gradium xn-ptts; MIT license, see THIRD_PARTY_NOTICES.md.
import MLX
import MLXNN

/// Convolutions use MLX's NTC layout. The checkpoint stores PyTorch OIK/IOK kernels.
struct PhononConv {
    let weight: MLXArray
    let bias: MLXArray?
    let kernel, stride, groups, inputChannels, outputChannels: Int
    let transposed: Bool
    init(_ weights: PhononWeights, _ prefix: String, input: Int, output: Int,
         kernel: Int, stride: Int = 1, groups: Int = 1, bias: Bool = true, transposed: Bool = false) throws {
        self.kernel = kernel
        self.stride = stride
        self.groups = groups
        inputChannels = input
        outputChannels = output
        self.transposed = transposed
        let shape = transposed ? [input, output / groups, kernel] : [output, input / groups, kernel]
        let original = try weights.tensor(prefix + ".weight", shape)
        // Depthwise transpose kernels already have the output channel as their first axis.
        weight = transposed && groups == 1 ? original.transposed(1, 2, 0) : original.transposed(0, 2, 1)
        self.bias = try bias ? weights.tensor(prefix + ".bias", [output]) : nil
    }
    func makeState() -> MLXArray {
        MLXArray.zeros([1, kernel - stride, transposed ? outputChannels : inputChannels])
    }
    func callAsFunction(_ input: MLXArray, state: inout MLXArray) -> MLXArray {
        let overlap = kernel - stride
        if transposed {
            var result = convTransposed1d(input, weight, stride: stride, groups: groups)
            if let bias { result = result + bias }
            if overlap > 0 {
                let beginning = result[0..., ..<overlap, 0...] + state
                result = concatenated([beginning, result[0..., overlap..., 0...]], axis: 1)
                let end = result.dim(1) - overlap
                let tail = result[0..., end..., 0...]
                state = bias.map { tail - $0 } ?? tail
                return result[0..., ..<end, 0...]
            }
            return result
        }
        let joined = overlap > 0 ? concatenated([state, input], axis: 1) : input
        var result = conv1d(joined, weight, stride: stride, groups: groups)
        if let bias { result = result + bias }
        if overlap > 0 {
            state = joined[0..., (joined.dim(1) - overlap)..., 0...]
        }
        return result
    }
}

struct PhononMimi {
    struct Block {
        let upsample, residual1, residual2: PhononConv
    }
    struct State {
        var upsample, first, last: MLXArray
        var blocks: [[MLXArray]]
        var attention: [PhononAttentionState]
        var arrays: [MLXArray] {
            [upsample, first, last] + blocks.flatMap { $0 } + attention.flatMap(\.arrays)
        }
    }
    let projection: PhononLinear
    let upsample: PhononConv?
    let transformer: PhononTransformer
    let first, last: PhononConv
    let blocks: [Block]
    init(_ weights: PhononWeights, config: PhononConfig.Mimi) throws {
        let cfg = config
        // Kernel-size-one projection stored as OIK, represented as a linear layer.
        let projectionWeight = try weights.tensor("mimi.quantizer.output_proj.weight", [cfg.dimension, cfg.quantizerDimension, 1])
        projection = try PhononLinear(PhononWeights(arrays: ["projection.weight": projectionWeight.squeezed(axis: 2)]),
                                     "projection", cfg.quantizerDimension, cfg.dimension)
        let hop = cfg.ratios.reduce(1, *)
        let rate = Int((Double(cfg.sampleRate) / cfg.frameRate / Double(hop)).rounded())
        upsample = try rate > 1 ? PhononConv(weights, "mimi.upsample.convtr.convtr", input: cfg.dimension,
                                           output: cfg.dimension, kernel: 2 * rate, stride: rate,
                                           groups: cfg.dimension, bias: false, transposed: true) : nil
        transformer = try PhononTransformer(weights, "mimi.decoder_transformer.transformer", width: cfg.dimension,
                                            heads: cfg.transformerNumHeads, count: cfg.transformerNumLayers,
                                            feedforward: cfg.transformerDimFeedforward,
                                            period: cfg.transformerMaxPeriod, context: cfg.transformerContext)
        let prefix = "mimi.decoder.model"
        var channels = cfg.nFilters * (1 << cfg.ratios.count)
        first = try PhononConv(weights, prefix + ".0.conv", input: cfg.dimension, output: channels, kernel: cfg.kernelSize)
        var blocks: [Block] = []
        for (index, ratio) in cfg.ratios.enumerated() {
            let layer = 1 + 3 * index, next = channels / 2
            blocks.append(try Block(
                upsample: PhononConv(weights, prefix + ".\(layer + 1).convtr", input: channels,
                                     output: next, kernel: 2 * ratio, stride: ratio, transposed: true),
                residual1: PhononConv(weights, prefix + ".\(layer + 2).block.1.conv", input: next,
                                      output: next / cfg.compress, kernel: cfg.residualKernelSize),
                residual2: PhononConv(weights, prefix + ".\(layer + 2).block.3.conv", input: next / cfg.compress,
                                      output: next, kernel: 1)
            ))
            channels = next
        }
        self.blocks = blocks
        last = try PhononConv(weights, prefix + ".\(2 + 3 * cfg.ratios.count).conv",
                             input: cfg.nFilters, output: cfg.channels, kernel: cfg.lastKernelSize)
    }
    func makeState() -> State {
        State(upsample: upsample?.makeState() ?? MLXArray.zeros([1, 0, 1]), first: first.makeState(), last: last.makeState(),
              blocks: blocks.map { [$0.upsample.makeState(), $0.residual1.makeState(), $0.residual2.makeState()] },
              attention: transformer.makeState())
    }
    func decode(_ latent: MLXArray, state: inout State) -> MLXArray {
        var x = projection(latent)
        if let upsample { x = upsample(x, state: &state.upsample) }
        x = transformer(x, state: &state.attention)
        x = first(x, state: &state.first)
        for (index, block) in blocks.enumerated() {
            x = block.upsample(elu(x), state: &state.blocks[index][0])
            let hidden = block.residual1(elu(x), state: &state.blocks[index][1])
            x = x + block.residual2(elu(hidden), state: &state.blocks[index][2])
        }
        return last(elu(x), state: &state.last).reshaped(-1)
    }
}
