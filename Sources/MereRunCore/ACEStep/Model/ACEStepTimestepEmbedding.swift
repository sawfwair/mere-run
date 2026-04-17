import Foundation
import MLX
import MLXNN

final class ACEStepTimestepEmbedding: Module {
    let inChannels: Int
    let timeEmbedDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "time_proj") var timeProj: Linear

    init(inChannels: Int, timeEmbedDim: Int, scale: Float = 1000.0) {
        self.inChannels = inChannels
        self.timeEmbedDim = timeEmbedDim
        self.scale = scale

        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        self._timeProj.wrappedValue = Linear(timeEmbedDim, timeEmbedDim * 6, bias: true)
    }

    func callAsFunction(_ t: MLXArray) -> (temb: MLXArray, timestepProj: MLXArray) {
        let batch = t.dim(0)
        let tFreq = buildSinusoidalEmbedding(t: t, dim: inChannels).asType(t.dtype)

        var temb = linear1(tFreq)
        temb = silu(temb)
        temb = linear2(temb)

        var proj = timeProj(silu(temb))
        proj = proj.reshaped(batch, 6, timeEmbedDim)

        return (temb, proj)
    }

    private func buildSinusoidalEmbedding(t: MLXArray, dim: Int, maxPeriod: Float = 10000.0) -> MLXArray {
        // Matches `TimestepEmbedding.timestep_embedding` in ACE-Step (Diffusers-style embedding).
        // t: [B] (fractional timesteps)
        // returns: [B, dim]
        let half = dim / 2
        let indices = MLXArray((0..<half).map { Float($0) }).asType(.float32)
        let freqs = MLX.exp(-MLXArray(Float(log(maxPeriod))) * indices / Float(half))

        let tScaled = t.asType(.float32) * MLXArray(scale)
        let args = tScaled[0..., .newAxis] * freqs[.newAxis]

        var emb = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        if dim % 2 != 0 {
            emb = MLX.concatenated([emb, MLX.zeros([t.dim(0), 1], dtype: emb.dtype)], axis: -1)
        }
        return emb
    }
}
