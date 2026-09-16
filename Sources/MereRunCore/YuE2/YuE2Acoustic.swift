import Foundation
import MLX
import MLXNN

/// Caches the causal branch once per original context chunk. Acoustic queries see
/// all prefix keys and all acoustic keys, including the two zero-content boundaries.
struct YuE2Acoustic {
    let model: YuE2Model
    let prefixLength: Int
    let cache: [(MLXArray, MLXArray)]

    init(model: YuE2Model, tokens: [Int]) throws {
        self.model = model
        prefixLength = tokens.count
        var hidden = model.embedding[MLXArray(tokens.map(Int32.init))].expandedDimensions(axis: 0)
        var cache: [(MLXArray, MLXArray)] = []
        for layer in model.autoregressive {
            try Task.checkCancellation()
            let (query, key, value) = layer.attention.project(layer.norm(hidden), offset: 0)
            cache.append((key, value))
            hidden = hidden + layer.attention.attend(query, key, value, causal: true)
            hidden = hidden + layer.mlp(layer.postNorm(hidden))
            MLX.eval(hidden, key, value)
        }
        self.cache = cache
    }

    func velocity(_ state: MLXArray, rawTime: Double) throws -> MLXArray {
        let count = state.dim(0) + 2
        guard state.ndim == 2, state.dim(1) == model.configuration.latentDim,
              prefixLength + count <= model.configuration.maxPositionEmbeddings,
              count <= model.configuration.maxLatentFrames else {
            throw YuE2Error.invalidRequest("Acoustic chunk exceeds the model context or latent dimensions.")
        }
        let zero = MLXArray.zeros([1, state.dim(1)], dtype: state.dtype)
        let padded = concatenated([zero, state, zero], axis: 0).expandedDimensions(axis: 0)
        let time = sigmoid(MLXArray(Float(rawTime)).asType(state.dtype))
        let shift = model.configuration.timestepShift
        let shifted = shift * time / (1 + (shift - 1) * time)
        let frequencies = exp(MLXArray(0..<128).asType(.float32) * (-log(Float(10000)) / 128))
        let angles = shifted.asType(.float32) * frequencies
        let timeFeatures = concatenated([cos(angles), sin(angles)]).asType(state.dtype)
        let timeEmbedding = model.timeOutput(silu(model.timeInput(timeFeatures)))
        var hidden = model.latentInput(padded) + timeEmbedding + model.positions[..<count]
        for (index, layer) in model.acoustic.enumerated() {
            try Task.checkCancellation()
            let (query, key, value) = layer.attention.project(layer.norm(hidden), offset: prefixLength)
            let keys = concatenated([cache[index].0, key], axis: 2)
            let values = concatenated([cache[index].1, value], axis: 2)
            hidden = hidden + layer.attention.attend(query, keys, values, causal: false)
            hidden = hidden + layer.mlp(layer.postNorm(hidden))
            MLX.eval(hidden)
        }
        return model.latentOutput(model.norm(hidden))[0, 1..<(count - 1), 0...]
    }

    func solve(noise: MLXArray, steps: Int, progress: (Int) -> Void) throws -> MLXArray {
        var state = noise.asType(model.embedding.dtype)
        let dt = 1 / Double(steps)
        for step in 0..<steps {
            try Task.checkCancellation()
            let time = 1 - Double(step) * dt
            let first = try velocity(state, rawTime: Self.rawTime(time))
            let midpoint = state - first * Float(dt / 2)
            let second = try velocity(midpoint, rawTime: Self.rawTime(time - dt / 2))
            state = state - second * Float(dt)
            MLX.eval(state)
            progress(step + 1)
        }
        let result = state.asType(.float32)
        guard all(isFinite(result)).item(Bool.self) else {
            throw YuE2Error.invalidAudio("Flow matching produced non-finite latents.")
        }
        return result
    }

    static func rawTime(_ time: Double) -> Double {
        min(20, max(-20, log(time / (1 - time))))
    }
}
