// Adapted from Gradium xn-ptts; MIT license, see THIRD_PARTY_NOTICES.md.
import MLX
import MLXNN

struct PhononFlow {
    struct TimeEmbedding {
        let first, second: PhononLinear
        let frequencies, alpha: MLXArray
        init(_ weights: PhononWeights, _ prefix: String, width: Int) throws {
            first = try PhononLinear(weights, prefix + ".mlp.0", 256, width, bias: true)
            second = try PhononLinear(weights, prefix + ".mlp.2", width, width, bias: true)
            frequencies = try weights.tensor(prefix + ".freqs", [128])
            alpha = try weights.tensor(prefix + ".mlp.3.alpha", [width])
        }
        func callAsFunction(_ time: Float) -> MLXArray {
            let phase = frequencies * time
            let x = second(silu(first(concatenated([cos(phase), sin(phase)]).reshaped(1, 256))))
            return x * rsqrt(mean(x * x, axis: -1, keepDims: true) + 1e-5) * alpha
        }
    }
    struct Residual {
        let norm: PhononNorm
        let modulation, first, second: PhononLinear
        init(_ weights: PhononWeights, _ prefix: String, width: Int) throws {
            norm = try PhononNorm(weights, prefix + ".in_ln", width, epsilon: 1e-6)
            modulation = try PhononLinear(weights, prefix + ".adaLN_modulation.1", width, 3 * width, bias: true)
            first = try PhononLinear(weights, prefix + ".mlp.0", width, width, bias: true)
            second = try PhononLinear(weights, prefix + ".mlp.2", width, width, bias: true)
        }
        func callAsFunction(_ x: MLXArray, _ condition: MLXArray) -> MLXArray {
            let parts = split(modulation(silu(condition)), parts: 3, axis: -1)
            let normalized = norm(x) * (1 + parts[1]) + parts[0]
            return x + parts[2] * second(silu(first(normalized)))
        }
    }
    let input, condition, finalModulation, output: PhononLinear
    let times: [TimeEmbedding]
    let residuals: [Residual]
    init(_ weights: PhononWeights, config: PhononConfig.Flow) throws {
        let prefix = "flow_lm.flow_net", width = config.flowDim
        input = try PhononLinear(weights, prefix + ".input_proj", config.ldim, width, bias: true)
        condition = try PhononLinear(weights, prefix + ".cond_embed", config.dModel, width, bias: true)
        times = try (0..<2).map { try TimeEmbedding(weights, prefix + ".time_embed.\($0)", width: width) }
        residuals = try (0..<config.flowDepth).map { try Residual(weights, prefix + ".res_blocks.\($0)", width: width) }
        finalModulation = try PhononLinear(weights, prefix + ".final_layer.adaLN_modulation.1", width, 2 * width, bias: true)
        output = try PhononLinear(weights, prefix + ".final_layer.linear", width, config.ldim, bias: true)
    }
    func direction(_ x: MLXArray, condition backbone: MLXArray, start: Float, end: Float) -> MLXArray {
        let condition = condition(backbone) + (times[0](start) + times[1](end)) * 0.5
        var hidden = input(x)
        for residual in residuals { hidden = residual(hidden, condition) }
        let parts = split(finalModulation(silu(condition)), parts: 2, axis: -1)
        let centered = hidden - mean(hidden, axis: -1, keepDims: true)
        let norm = centered * rsqrt(mean(centered * centered, axis: -1, keepDims: true) + 1e-6)
        return output(norm * (1 + parts[1]) + parts[0])
    }
    func sample(_ noise: MLXArray, condition: MLXArray, steps: Int) -> MLXArray {
        var latent = noise
        for step in 0..<steps {
            latent = latent + direction(latent, condition: condition,
                                        start: Float(step) / Float(steps), end: Float(step + 1) / Float(steps)) / Float(steps)
        }
        return latent
    }
}
