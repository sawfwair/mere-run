import Foundation
import MLX
import MLXFast
import MLXNN

private final class HaruScalarEmbedding: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(width: Int) {
        _fc1.wrappedValue = Linear(256, width, bias: false)
        _fc2.wrappedValue = Linear(width, width, bias: false)
    }

    func callAsFunction(_ value: Float, dtype: DType) -> MLXArray {
        let frequencies = exp(MLXArray(0..<128).asType(.float32) * (-log(Float(10_000)) / 128))
        let angles = frequencies * MLXArray(value).asType(dtype).asType(.float32)
        let features = concatenated([cos(angles), sin(angles)]).reshaped(1, 256).asType(dtype)
        return fc2(silu(fc1(features)))
    }
}

private final class HaruHeadFFN: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(width: Int, intermediate: Int) {
        _gate.wrappedValue = Linear(width, intermediate, bias: false)
        _up.wrappedValue = Linear(width, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, width, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        down(silu(gate(input)) * up(input))
    }
}

final class HaruGain: Module {
    let weight: MLXArray

    init(width: Int) {
        weight = MLXArray.ones([width])
    }
}

private func headNorm(_ input: MLXArray, epsilon: Float) -> MLXArray {
    MLXFast.rmsNorm(
        input.asType(.float32), weight: MLXArray.ones([input.dim(-1)]), eps: epsilon
    ).asType(input.dtype)
}

private final class HaruHeadLayer: Module {
    let norm: HaruGain
    let ffn: HaruHeadFFN
    let ada: Linear
    let epsilon: Float

    init(width: Int, intermediate: Int, epsilon: Float) {
        self.norm = HaruGain(width: width)
        self.ffn = HaruHeadFFN(width: width, intermediate: intermediate)
        self.ada = Linear(width, 3 * width, bias: false)
        self.epsilon = epsilon
    }

    func callAsFunction(_ input: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = ada(conditioning).split(parts: 3, axis: -1)
        let hidden = headNorm(input, epsilon: epsilon) * norm.weight
        return input + modulation[2] * ffn(hidden * (1 + modulation[1]) + modulation[0])
    }
}

private final class HaruHeadOutput: Module {
    let proj: Linear
    let ada: Linear
    let epsilon: Float

    init(width: Int, latentDim: Int, epsilon: Float) {
        proj = Linear(width, latentDim, bias: false)
        ada = Linear(width, 2 * width, bias: false)
        self.epsilon = epsilon
    }

    func callAsFunction(_ input: MLXArray, conditioning: MLXArray) -> MLXArray {
        let (shift, scale) = ada(conditioning).split(axis: -1)
        return proj(headNorm(input, epsilon: epsilon) * (1 + scale) + shift)
    }
}

final class HaruMeanFlow: Module {
    @ModuleInfo(key: "latent_proj") var latentProjection: Linear
    @ModuleInfo(key: "cond_proj") var conditionProjection: Linear
    @ModuleInfo(key: "t_embed") private var time: HaruScalarEmbedding
    @ModuleInfo(key: "dt_embed") private var interval: HaruScalarEmbedding
    @ModuleInfo(key: "h_embed") private var step: HaruScalarEmbedding
    @ModuleInfo(key: "omega_embed") private var guidance: HaruScalarEmbedding
    private let layers: [HaruHeadLayer]
    private let final: HaruHeadOutput
    private let config: HaruConfiguration.Head

    init(config: HaruConfiguration.Head) {
        self.config = config
        let width = config.hiddenSize
        _latentProjection.wrappedValue = Linear(config.latentDim, width, bias: false)
        _conditionProjection.wrappedValue = Linear(width, width, bias: false)
        _time.wrappedValue = HaruScalarEmbedding(width: width)
        _interval.wrappedValue = HaruScalarEmbedding(width: width)
        _step.wrappedValue = HaruScalarEmbedding(width: width)
        _guidance.wrappedValue = HaruScalarEmbedding(width: width)
        layers = (0..<config.numLayers).map { _ in
            HaruHeadLayer(width: width, intermediate: Int(Float(width) * config.ffnRatio), epsilon: config.normEps)
        }
        final = HaruHeadOutput(width: width, latentDim: config.latentDim, epsilon: config.normEps)
    }

    func conditioning(dtype: DType) -> [MLXArray] {
        [
            time(0, dtype: dtype), interval(config.intervalEmbed, dtype: dtype), step(1, dtype: dtype),
            guidance(1 - 1 / config.guidanceScale, dtype: dtype),
        ]
    }

    func sample(hidden: MLXArray, noise: MLXArray, conditioning: [MLXArray]) -> MLXArray {
        // Each addition rounds in the checkpoint dtype.
        let condition = silu(conditioning.reduce(conditionProjection(hidden), +))
        let initial = latentProjection(noise)
        let velocity = layers.reduce(initial) { hidden, layer in layer(hidden, conditioning: condition) }
        return noise + final(velocity, conditioning: condition)
    }
}
