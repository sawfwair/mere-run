import Foundation
import MLX
import MLXFast
import MLXNN

final class ParakeetDwStridingSubsampling: Module {
    let conv: [Module]
    @ModuleInfo(key: "out") var out: Linear

    let samplingNum: Int
    let convChannels: Int
    let stride: Int = 2
    let kernelSize: Int = 3
    let padding: Int = 1

    init(config: ParakeetEncoderConfig) {
        self.samplingNum = max(1, Int(log2(Double(max(1, config.subsamplingFactor)))))
        self.convChannels = config.subsamplingConvChannels

        precondition(config.subsamplingFactor > 0 && (config.subsamplingFactor & (config.subsamplingFactor - 1) == 0))
        precondition(config.subsamplingFactor <= 8, "Parakeet subsampling factors above 8 are not supported.")

        var finalFreqDim = config.featIn
        for _ in 0..<samplingNum {
            finalFreqDim = ((finalFreqDim + 2 * padding - kernelSize) / stride) + 1
        }

        self.conv = [
            Conv2d(
                inputChannels: 1,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: 1,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: convChannels,
                bias: true
            ),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                padding: IntOrPair(0),
                groups: 1,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: convChannels,
                bias: true
            ),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                padding: IntOrPair(0),
                groups: 1,
                bias: true
            )
        ]

        self._out.wrappedValue = Linear(convChannels * finalFreqDim, config.modelDim)
    }

    func callAsFunction(_ x: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        func convLayer(at index: Int) -> Conv2d {
            guard let layer = conv[index] as? Conv2d else {
                preconditionFailure("Expected Conv2d at conv[\(index)] in ParakeetDwStridingSubsampling.")
            }
            return layer
        }

        var outLengths = lengths
        for _ in 0..<samplingNum {
            outLengths = outLengths.map { length in
                let value = Int(floor(Double(length + 2 * padding - kernelSize) / Double(stride)) + 1)
                return max(0, value)
            }
        }

        var hidden = x.expandedDimensions(axis: 1) // [B,1,T,F]
        hidden = hidden.transposed(0, 2, 3, 1) // [B,T,F,1]

        hidden = relu(convLayer(at: 0)(hidden))
        if samplingNum >= 2 {
            hidden = convLayer(at: 2)(hidden)
            hidden = convLayer(at: 3)(hidden)
            hidden = relu(hidden)
        }
        if samplingNum >= 3 {
            hidden = convLayer(at: 5)(hidden)
            hidden = convLayer(at: 6)(hidden)
            hidden = relu(hidden)
        }

        hidden = hidden.transposed(0, 3, 1, 2) // [B,C,T,F]
        let batch = hidden.dim(0)
        let channels = hidden.dim(1)
        let time = hidden.dim(2)
        let freq = hidden.dim(3)
        hidden = hidden.transposed(0, 2, 1, 3).reshaped(batch, time, channels * freq)
        hidden = out(hidden)

        return (hidden, outLengths)
    }
}

final class ParakeetConformer: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: ParakeetDwStridingSubsampling
    @ModuleInfo(key: "layers") var layers: [ParakeetConformerBlock]

    let config: ParakeetEncoderConfig
    let posEnc: ParakeetRelPositionalEncoding?

    init(config: ParakeetEncoderConfig) {
        self.config = config
        self._preEncode.wrappedValue = ParakeetDwStridingSubsampling(config: config)
        self._layers.wrappedValue = (0..<config.layers).map { _ in ParakeetConformerBlock(config: config) }

        if config.selfAttentionModel == "rel_pos" {
            self.posEnc = ParakeetRelPositionalEncoding(
                modelDim: config.modelDim,
                maxLen: config.posEmbMaxLen,
                scaleInput: config.xScaling
            )
        } else {
            self.posEnc = nil
        }
    }

    func callAsFunction(_ x: MLXArray, lengths: [Int]? = nil) -> (MLXArray, [Int]) {
        let inLengths = lengths ?? Array(repeating: x.dim(1), count: x.dim(0))
        var (hidden, outLengths) = preEncode(x, lengths: inLengths)

        var posEmb: MLXArray?
        if let posEnc {
            let output = posEnc(hidden)
            hidden = output.0
            posEmb = output.1
        }

        for layer in layers {
            hidden = layer(hidden, posEmb: posEmb)
        }

        return (hidden, outLengths)
    }
}
