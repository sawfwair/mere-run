import Foundation
import MLX
import MLXNN
import MLXRandom

private final class MMAudioWeightNormParameters: Module {
    @ParameterInfo(key: "original0") var magnitude: MLXArray
    @ParameterInfo(key: "original1") var direction: MLXArray

    init(magnitudeShape: [Int], directionShape: [Int]) {
        self._magnitude.wrappedValue = MLXArray.ones(magnitudeShape)
        self._direction.wrappedValue = MLXRandom.normal(directionShape) * 0.02
    }
}

private final class MMAudioWeightParametrization: Module {
    @ModuleInfo(key: "weight") var weight: MMAudioWeightNormParameters

    init(magnitudeShape: [Int], directionShape: [Int]) {
        self._weight.wrappedValue = MMAudioWeightNormParameters(
            magnitudeShape: magnitudeShape,
            directionShape: directionShape
        )
    }
}

private final class MMAudioBigVGANConv1D: Module {
    @ModuleInfo(key: "parametrizations") var parametrizations: MMAudioWeightParametrization
    @ParameterInfo(key: "bias") var bias: MLXArray?

    private let stride: Int
    private let padding: Int
    private let dilation: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._parametrizations.wrappedValue = MMAudioWeightParametrization(
            magnitudeShape: [outputChannels, 1, 1],
            directionShape: [outputChannels, kernelSize, inputChannels]
        )
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parameters = parametrizations.weight
        let norm = MLX.sqrt(
            MLX.sum(parameters.direction * parameters.direction, axes: [1, 2], keepDims: true) + 1e-12
        )
        let weight = parameters.direction * (parameters.magnitude / norm)
        var output = MLX.conv1d(
            x,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation
        )
        if let bias {
            output = output + bias
        }
        return output
    }
}

private final class MMAudioBigVGANConvTranspose1D: Module {
    @ModuleInfo(key: "parametrizations") var parametrizations: MMAudioWeightParametrization
    @ParameterInfo(key: "bias") var bias: MLXArray?

    private let stride: Int
    private let padding: Int

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.stride = stride
        self.padding = padding
        self._parametrizations.wrappedValue = MMAudioWeightParametrization(
            magnitudeShape: [inputChannels, 1, 1],
            directionShape: [outputChannels, kernelSize, inputChannels]
        )
        self._bias.wrappedValue = MLXArray.zeros([outputChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parameters = parametrizations.weight
        let norm = MLX.sqrt(
            MLX.sum(parameters.direction * parameters.direction, axes: [0, 1], keepDims: true) + 1e-12
        )
        let magnitude = parameters.magnitude.reshaped(1, 1, -1)
        let weight = parameters.direction * (magnitude / norm)
        var output = MLX.convTransposed1d(x, weight, stride: stride, padding: padding)
        if let bias {
            output = output + bias
        }
        return output
    }
}

private final class MMAudioSnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let frequency = MLX.exp(alpha).reshaped(1, 1, -1)
        let magnitude = MLX.exp(beta).reshaped(1, 1, -1)
        let periodic = MLX.sin(x * frequency)
        return x + periodic * periodic / (magnitude + 1e-9)
    }
}

private enum MMAudioAliasFreeFilter {
    static let values: [Float] = [
        0.00202896645, 0.00938946394, -0.0255434641, -0.0576573755,
        0.128572609, 0.4432098, 0.4432098, 0.128572609,
        -0.0576573755, -0.0255434641, 0.00938946394, 0.00202896645,
    ]

    static func weight(channels: Int, dtype: DType) -> MLXArray {
        MLX.tiled(
            MLXArray(values).reshaped(1, values.count, 1),
            repetitions: [channels, 1, 1]
        ).asType(dtype)
    }
}

private final class MMAudioAliasFreeActivation: Module {
    @ModuleInfo(key: "act") var activation: MMAudioSnakeBeta

    init(channels: Int) {
        self._activation.wrappedValue = MMAudioSnakeBeta(channels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(2)
        let left = MLX.repeated(x[0..., 0..<1, 0...], count: 5, axis: 1)
        let right = MLX.repeated(x[0..., (x.dim(1) - 1)..., 0...], count: 5, axis: 1)
        let padded = MLX.concatenated([left, x, right], axis: 1)
        var upsampled = MLX.convTransposed1d(
            padded,
            MMAudioAliasFreeFilter.weight(channels: channels, dtype: x.dtype) * 2,
            stride: 2,
            padding: 0,
            groups: channels
        )
        upsampled = upsampled[0..., 15..<(upsampled.dim(1) - 15), 0...]
        let activated = activation(upsampled)
        let downLeft = MLX.repeated(activated[0..., 0..<1, 0...], count: 5, axis: 1)
        let downRight = MLX.repeated(activated[0..., (activated.dim(1) - 1)..., 0...], count: 6, axis: 1)
        return MLX.conv1d(
            MLX.concatenated([downLeft, activated, downRight], axis: 1),
            MMAudioAliasFreeFilter.weight(channels: channels, dtype: x.dtype),
            stride: 2,
            padding: 0,
            groups: channels
        )
    }
}

private final class MMAudioAMPBlock: Module {
    @ModuleInfo(key: "convs1") var firstConvolutions: [MMAudioBigVGANConv1D]
    @ModuleInfo(key: "convs2") var secondConvolutions: [MMAudioBigVGANConv1D]
    @ModuleInfo(key: "activations") var activations: [MMAudioAliasFreeActivation]

    init(channels: Int, kernelSize: Int) {
        let dilations = [1, 3, 5]
        self._firstConvolutions.wrappedValue = dilations.map { dilation in
            MMAudioBigVGANConv1D(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize * dilation - dilation) / 2,
                dilation: dilation
            )
        }
        self._secondConvolutions.wrappedValue = dilations.map { _ in
            MMAudioBigVGANConv1D(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize - 1) / 2
            )
        }
        self._activations.wrappedValue = (0..<6).map { _ in
            MMAudioAliasFreeActivation(channels: channels)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input
        for index in 0..<firstConvolutions.count {
            var update = activations[index * 2](hidden)
            update = firstConvolutions[index](update)
            update = activations[index * 2 + 1](update)
            update = secondConvolutions[index](update)
            hidden = hidden + update
        }
        return hidden
    }
}

private final class MMAudioBigVGANUpsample: Module {
    @ModuleInfo(key: "convolution") var convolution: MMAudioBigVGANConvTranspose1D

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int) {
        self._convolution.wrappedValue = MMAudioBigVGANConvTranspose1D(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: (kernelSize - stride) / 2
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convolution(x)
    }
}

public final class MMAudioBigVGAN: Module {
    @ModuleInfo(key: "conv_pre") private var preConvolution: MMAudioBigVGANConv1D
    @ModuleInfo(key: "ups") private var upsamplers: [MMAudioBigVGANUpsample]
    @ModuleInfo(key: "resblocks") private var residualBlocks: [MMAudioAMPBlock]
    @ModuleInfo(key: "activation_post") private var postActivation: MMAudioAliasFreeActivation
    @ModuleInfo(key: "conv_post") private var postConvolution: MMAudioBigVGANConv1D
    private let useFloat32: Bool

    public convenience override init() {
        self.init(
            inputChannels: 128,
            initialChannels: 1_536,
            upsampleRates: [8, 4, 2, 2, 2, 2],
            upsampleKernelSizes: [16, 8, 4, 4, 4, 4],
            useFloat32: false
        )
    }

    public init(
        inputChannels: Int,
        initialChannels: Int,
        upsampleRates: [Int],
        upsampleKernelSizes: [Int],
        useFloat32: Bool = false
    ) {
        self.useFloat32 = useFloat32
        precondition(!upsampleRates.isEmpty && upsampleRates.count == upsampleKernelSizes.count)
        var upsamplers: [MMAudioBigVGANUpsample] = []
        var residualBlocks: [MMAudioAMPBlock] = []
        for stage in upsampleRates.indices {
            let stageInputChannels = initialChannels / (1 << stage)
            let outputChannels = initialChannels / (1 << (stage + 1))
            upsamplers.append(MMAudioBigVGANUpsample(
                inputChannels: stageInputChannels,
                outputChannels: outputChannels,
                kernelSize: upsampleKernelSizes[stage],
                stride: upsampleRates[stage]
            ))
            for residualKernel in [3, 7, 11] {
                residualBlocks.append(MMAudioAMPBlock(
                    channels: outputChannels,
                    kernelSize: residualKernel
                ))
            }
        }

        self._preConvolution.wrappedValue = MMAudioBigVGANConv1D(
            inputChannels: inputChannels,
            outputChannels: initialChannels,
            kernelSize: 7,
            padding: 3
        )
        self._upsamplers.wrappedValue = upsamplers
        self._residualBlocks.wrappedValue = residualBlocks
        let finalChannels = initialChannels / (1 << upsampleRates.count)
        self._postActivation.wrappedValue = MMAudioAliasFreeActivation(channels: finalChannels)
        self._postConvolution.wrappedValue = MMAudioBigVGANConv1D(
            inputChannels: finalChannels,
            outputChannels: 1,
            kernelSize: 7,
            padding: 3,
            bias: false
        )
    }

    public func callAsFunction(_ spectrogram: MLXArray) -> MLXArray {
        var hidden = preConvolution(spectrogram.asType(useFloat32 ? .float32 : .float16))
        for stage in upsamplers.indices {
            hidden = upsamplers[stage](hidden)
            let start = stage * 3
            hidden = (
                residualBlocks[start](hidden)
                    + residualBlocks[start + 1](hidden)
                    + residualBlocks[start + 2](hidden)
            ) / 3
        }
        hidden = postConvolution(postActivation(hidden))
        return MLX.clip(hidden, min: -1, max: 1)
    }

    public static func load(resources: MMAudioModelResources) throws -> MMAudioBigVGAN {
        let model = MMAudioBigVGAN()
        let weightsURL = resources.bigVGANWeightsURL()
        if weightsURL.pathExtension == "safetensors" {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: weightsURL,
                to: model,
                dtype: .float16,
                verify: .none,
                mapper: mapSafetensorsWeights
            )
        } else {
            try MMAudioBigVGANCheckpointLoader.load(url: weightsURL, into: model)
        }
        return model
    }

    static func mapCheckpointWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasSuffix(".filter") {
            return []
        }
        let mappedKey = remapUpsampleKey(key)
        if mappedKey.hasSuffix(".weight_g") {
            let base = String(mappedKey.dropLast(".weight_g".count))
            return [("\(base).parametrizations.weight.original0", value)]
        }
        if mappedKey.hasSuffix(".weight_v"), value.ndim == 3 {
            let base = String(mappedKey.dropLast(".weight_v".count))
            let isTranspose = mappedKey.hasPrefix("ups.")
            let transposed = isTranspose ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            return [(
                "\(base).parametrizations.weight.original1",
                transposed.reshaped(-1).reshaped(transposed.shape)
            )]
        }
        return [(mappedKey, value)]
    }

    private static func remapUpsampleKey(_ key: String) -> String {
        var components = key.split(separator: ".").map(String.init)
        if components.count >= 4, components[0] == "ups", components[2] == "0" {
            components[2] = "convolution"
        }
        return components.joined(separator: ".")
    }

    static func mapSafetensorsWeights(key: String, value: MLXArray) -> [(String, MLXArray)] {
        let mappedKey = remapUpsampleKey(key)
        if mappedKey.hasSuffix(".weight"), value.ndim == 3 {
            let base = String(mappedKey.dropLast(".weight".count))
            let isTranspose = mappedKey.hasPrefix("ups.")
            let transposed = isTranspose ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            let magnitudeAxes = isTranspose ? [0, 1] : [1, 2]
            let magnitude = MLX.sqrt(MLX.sum(transposed * transposed, axes: magnitudeAxes, keepDims: true))
            let reshapedMagnitude = isTranspose
                ? magnitude.reshaped(-1, 1, 1)
                : magnitude
            return [
                ("\(base).parametrizations.weight.original0", reshapedMagnitude),
                ("\(base).parametrizations.weight.original1", transposed),
            ]
        }
        return mapCheckpointWeight(key: mappedKey, value: value)
    }
}
