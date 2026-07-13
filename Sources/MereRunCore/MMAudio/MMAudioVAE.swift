import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

private enum MMAudioVAETensorOps {
    static func normalizeChannels(_ x: MLXArray, eps: Float = 1e-4) -> MLXArray {
        let channels = x.dim(-1)
        let norm = MLX.sqrt((x.asType(.float32) * x.asType(.float32)).sum(axis: -1, keepDims: true))
        let stabilized = eps + norm / sqrt(Float(channels))
        return x / stabilized.asType(x.dtype)
    }

    static func magnitudePreservingSiLU(_ x: MLXArray) -> MLXArray {
        MLXNN.silu(x) / 0.596
    }

    static func magnitudePreservingSum(_ residual: MLXArray, _ update: MLXArray) -> MLXArray {
        (0.7 * residual + 0.3 * update) / sqrt(Float(0.58))
    }
}

private final class MMAudioMPConv1D: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let kernelSize: Int
    private let inputChannels: Int

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self.inputChannels = inputChannels
        self._weight.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels])
    }

    func callAsFunction(_ x: MLXArray, gain: MLXArray? = nil) -> MLXArray {
        let effectiveWeight = effectiveWeight(gain: gain)
        return MLX.conv1d(
            x,
            effectiveWeight.asType(x.dtype),
            stride: 1,
            padding: kernelSize / 2
        )
    }

    func effectiveWeight(gain: MLXArray? = nil) -> MLXArray {
        let flattened = weight.asType(.float32).reshaped(weight.dim(0), -1)
        let norm = MLX.sqrt((flattened * flattened).sum(axis: 1, keepDims: true))
        let normalized = flattened / (1e-4 + norm / sqrt(Float(inputChannels * kernelSize)))
        var effectiveWeight = normalized.reshaped(weight.shape) / sqrt(Float(inputChannels * kernelSize))
        if let gain {
            effectiveWeight = effectiveWeight * gain
        }
        return effectiveWeight
    }
}

private final class MMAudioVAEResidualBlock: Module {
    @ModuleInfo(key: "conv1") var first: MMAudioMPConv1D
    @ModuleInfo(key: "conv2") var second: MMAudioMPConv1D
    @ModuleInfo(key: "nin_shortcut") var shortcut: MMAudioMPConv1D?

    init(inputChannels: Int, outputChannels: Int) {
        self._first.wrappedValue = MMAudioMPConv1D(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 3
        )
        self._second.wrappedValue = MMAudioMPConv1D(
            inputChannels: outputChannels,
            outputChannels: outputChannels,
            kernelSize: 3
        )
        self._shortcut.wrappedValue = inputChannels == outputChannels ? nil : MMAudioMPConv1D(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 1
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let normalized = MMAudioVAETensorOps.normalizeChannels(input)
        var hidden = first(MMAudioVAETensorOps.magnitudePreservingSiLU(normalized))
        hidden = second(MMAudioVAETensorOps.magnitudePreservingSiLU(hidden))
        let residual = shortcut?(normalized) ?? normalized
        return MMAudioVAETensorOps.magnitudePreservingSum(residual, hidden)
    }
}

private final class MMAudioVAEAttentionBlock: Module {
    @ModuleInfo(key: "qkv") var qkv: MMAudioMPConv1D
    @ModuleInfo(key: "proj_out") var output: MMAudioMPConv1D

    init(channels: Int) {
        self._qkv.wrappedValue = MMAudioMPConv1D(
            inputChannels: channels,
            outputChannels: channels * 3,
            kernelSize: 1
        )
        self._output.wrappedValue = MMAudioMPConv1D(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let channels = input.dim(2)
        let packed = qkv(input).reshaped(batch, sequence, 1, channels, 3)
        var query = packed[0..., 0..., 0..., 0..., 0].transposed(0, 2, 1, 3)
        var key = packed[0..., 0..., 0..., 0..., 1].transposed(0, 2, 1, 3)
        var value = packed[0..., 0..., 0..., 0..., 2].transposed(0, 2, 1, 3)
        query = MMAudioVAETensorOps.normalizeChannels(query)
        key = MMAudioVAETensorOps.normalizeChannels(key)
        value = MMAudioVAETensorOps.normalizeChannels(value)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / sqrt(Float(channels)),
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, sequence, channels)
        return MMAudioVAETensorOps.magnitudePreservingSum(input, output(attended))
    }
}

private final class MMAudioVAEMiddle: Module {
    @ModuleInfo(key: "block_1") var first: MMAudioVAEResidualBlock
    @ModuleInfo(key: "attn_1") var attention: MMAudioVAEAttentionBlock
    @ModuleInfo(key: "block_2") var second: MMAudioVAEResidualBlock

    init(channels: Int) {
        self._first.wrappedValue = MMAudioVAEResidualBlock(inputChannels: channels, outputChannels: channels)
        self._attention.wrappedValue = MMAudioVAEAttentionBlock(channels: channels)
        self._second.wrappedValue = MMAudioVAEResidualBlock(inputChannels: channels, outputChannels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        second(attention(first(x)))
    }
}

private final class MMAudioVAEUpsample: Module {
    @ModuleInfo(key: "conv") var conv: MMAudioMPConv1D

    init(channels: Int) {
        self._conv.wrappedValue = MMAudioMPConv1D(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(MLX.repeated(x, count: 2, axis: 1))
    }
}

private final class MMAudioVAEUpLevel: Module {
    @ModuleInfo(key: "block") var blocks: [MMAudioVAEResidualBlock]
    @ModuleInfo(key: "upsample") var upsample: MMAudioVAEUpsample?

    init(inputChannels: Int, outputChannels: Int, upsample: Bool) {
        self._blocks.wrappedValue = [
            MMAudioVAEResidualBlock(inputChannels: inputChannels, outputChannels: outputChannels),
            MMAudioVAEResidualBlock(inputChannels: outputChannels, outputChannels: outputChannels),
            MMAudioVAEResidualBlock(inputChannels: outputChannels, outputChannels: outputChannels),
        ]
        self._upsample.wrappedValue = upsample ? MMAudioVAEUpsample(channels: outputChannels) : nil
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = blocks.reduce(input) { value, block in
            MLX.clip(block(value), min: -256, max: 256)
        }
        return upsample?(hidden) ?? hidden
    }
}

private final class MMAudioVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var input: MMAudioMPConv1D
    @ModuleInfo(key: "mid") var middle: MMAudioVAEMiddle
    @ModuleInfo(key: "up") var up: [MMAudioVAEUpLevel]
    @ModuleInfo(key: "conv_out") var output: MMAudioMPConv1D
    @ParameterInfo(key: "learnable_gain") var learnableGain: MLXArray

    override init() {
        self._input.wrappedValue = MMAudioMPConv1D(inputChannels: 40, outputChannels: 2_048, kernelSize: 3)
        self._middle.wrappedValue = MMAudioVAEMiddle(channels: 2_048)
        self._up.wrappedValue = [
            MMAudioVAEUpLevel(inputChannels: 1_024, outputChannels: 512, upsample: false),
            MMAudioVAEUpLevel(inputChannels: 2_048, outputChannels: 1_024, upsample: true),
            MMAudioVAEUpLevel(inputChannels: 2_048, outputChannels: 2_048, upsample: false),
        ]
        self._output.wrappedValue = MMAudioMPConv1D(inputChannels: 512, outputChannels: 128, kernelSize: 3)
        self._learnableGain.wrappedValue = MLXArray(Float(0))
    }

    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var hidden = input(latent)
        hidden = MLX.clip(middle(hidden), min: -256, max: 256)
        for level in up.reversed() {
            hidden = level(hidden)
        }
        hidden = MMAudioVAETensorOps.magnitudePreservingSiLU(hidden)
        return output(hidden, gain: learnableGain + 1)
    }

    func parityStages(_ latent: MLXArray) -> MMAudioVAEParityStages {
        let inputProjection = input(latent)
        let middleFirst = middle.first(inputProjection)
        let middleAttention = middle.attention(middleFirst)
        let middleSecond = MLX.clip(middle.second(middleAttention), min: -256, max: 256)
        var hidden = middleSecond
        var upLevels: [MLXArray] = []
        for level in up.reversed() {
            hidden = level(hidden)
            upLevels.append(hidden)
        }
        let preOutput = MMAudioVAETensorOps.magnitudePreservingSiLU(hidden)
        let rawOutput = output(preOutput, gain: learnableGain + 1)
        return MMAudioVAEParityStages(
            rawInputWeight: input.weight,
            effectiveInputWeight: input.effectiveWeight(),
            inputProjection: inputProjection,
            middleFirst: middleFirst,
            middleAttention: middleAttention,
            middleSecond: middleSecond,
            upLevels: upLevels,
            preOutput: preOutput,
            rawOutput: rawOutput
        )
    }
}

struct MMAudioVAEParityStages {
    let rawInputWeight: MLXArray
    let effectiveInputWeight: MLXArray
    let inputProjection: MLXArray
    let middleFirst: MLXArray
    let middleAttention: MLXArray
    let middleSecond: MLXArray
    let upLevels: [MLXArray]
    let preOutput: MLXArray
    let rawOutput: MLXArray
}

public final class MMAudioVAE: Module {
    @ModuleInfo(key: "decoder") private var decoder: MMAudioVAEDecoder

    private let dataMean: MLXArray
    private let dataStandardDeviation: MLXArray

    public override init() {
        self._decoder.wrappedValue = MMAudioVAEDecoder()
        self.dataMean = MLXArray(Self.meanValues).reshaped(1, 1, 128)
        self.dataStandardDeviation = MLXArray(Self.standardDeviationValues).reshaped(1, 1, 128)
        super.init()
    }

    public static func load(resources: MMAudioModelResources) throws -> MMAudioVAE {
        let model = MMAudioVAE()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: model,
            dtype: .float16,
            verify: .none,
            mapper: mapWeights
        )
        return model
    }

    public func decode(_ latent: MLXArray) -> MLXArray {
        let spectrogram = decoder(latent.asType(.float16))
        return spectrogram * dataStandardDeviation.asType(spectrogram.dtype)
            + dataMean.asType(spectrogram.dtype)
    }

    func parityStages(_ latent: MLXArray) -> MMAudioVAEParityStages {
        decoder.parityStages(latent.asType(.float16))
    }

    private static func mapWeights(key: String, value: MLXArray) -> [(String, MLXArray)] {
        guard key.hasPrefix("decoder.") else { return [] }
        if key.hasSuffix(".weight"), value.ndim == 3 {
            let transposed = value.transposed(0, 2, 1)
            return [(key, transposed.reshaped(-1).reshaped(transposed.shape))]
        }
        return [(key, value)]
    }

    private static let meanValues: [Float] = [
        -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006,
        -2.2357, -2.4597, -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047,
        -2.7483, -2.5926, -2.7462, -2.7033, -2.7386, -2.8112, -2.7502, -2.9594,
        -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157, -3.1191, -2.9893,
        -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
        -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509,
        -3.5089, -3.4647, -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747,
        -3.7072, -3.7279, -3.7283, -3.7795, -3.8259, -3.8447, -3.8663, -3.9182,
        -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121, -4.1488, -4.1874,
        -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
        -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053,
        -5.4927, -5.5712, -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103,
        -6.0955, -6.1673, -6.2362, -6.3120, -6.3926, -6.4797, -6.5565, -6.6511,
        -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663, -7.6136, -7.7469,
        -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
        -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861,
    ]

    private static let standardDeviationValues: [Float] = [
        2.3804, 2.4368, 2.3772, 2.3145, 2.2803, 2.2510, 2.2316, 2.2083,
        2.1996, 2.1835, 2.1769, 2.1659, 2.1631, 2.1618, 2.1540, 2.1606,
        2.1571, 2.1567, 2.1612, 2.1579, 2.1679, 2.1683, 2.1634, 2.1557,
        2.1668, 2.1518, 2.1415, 2.1449, 2.1406, 2.1350, 2.1313, 2.1415,
        2.1281, 2.1352, 2.1219, 2.1182, 2.1327, 2.1195, 2.1137, 2.1080,
        2.1179, 2.1036, 2.1087, 2.1036, 2.1015, 2.1068, 2.0975, 2.0991,
        2.0902, 2.1015, 2.0857, 2.0920, 2.0893, 2.0897, 2.0910, 2.0881,
        2.0925, 2.0873, 2.0960, 2.0900, 2.0957, 2.0958, 2.0978, 2.0936,
        2.0886, 2.0905, 2.0845, 2.0855, 2.0796, 2.0840, 2.0813, 2.0817,
        2.0838, 2.0840, 2.0917, 2.1061, 2.1431, 2.1976, 2.2482, 2.3055,
        2.3700, 2.4088, 2.4372, 2.4609, 2.4731, 2.4847, 2.5072, 2.5451,
        2.5772, 2.6147, 2.6529, 2.6596, 2.6645, 2.6726, 2.6803, 2.6812,
        2.6899, 2.6916, 2.6931, 2.6998, 2.7062, 2.7262, 2.7222, 2.7158,
        2.7041, 2.7485, 2.7491, 2.7451, 2.7485, 2.7233, 2.7297, 2.7233,
        2.7145, 2.6958, 2.6788, 2.6439, 2.6007, 2.4786, 2.2469, 2.1877,
        2.1392, 2.0717, 2.0107, 1.9676, 1.9140, 1.7102, 0.9101, 0.7164,
    ]
}
