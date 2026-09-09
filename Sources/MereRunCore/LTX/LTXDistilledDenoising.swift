import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func denoiseLoop(
    latents: MLXArray,
    rope: (cos: MLXArray, sin: MLXArray),
    context: MLXArray,
    transformer: LTXDistilledTransformer,
    label: String,
    sigmas: [Float],
    conditioning: LTXLatentConditioningState?
) -> MLXArray {
    var current = latents
    let dtype = latents.dtype
    let debugDenoise = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DENOISE"] == "1"
    let debugDumpPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DUMP_PREFIX"]
    let debugDumpAll = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DUMP_ALL"] == "1"

    for i in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[i]
        let nextSigma = sigmas[i + 1]

        let b = current.dim(0)
        let c = current.dim(1)
        let f = current.dim(2)
        let h = current.dim(3)
        let w = current.dim(4)
        let tokenCount = f * h * w

        let flat = current.transposed(0, 2, 3, 4, 1).reshaped(b, tokenCount, c)
        let timesteps: MLXArray
        if let conditioning {
            let mask = conditioning.denoiseMask.reshaped(b, 1, f, 1, 1)
            let broadcastMask = broadcast(mask, to: [b, 1, f, h, w]).reshaped(b, tokenCount)
            timesteps = MLXArray(sigma).asType(dtype) * broadcastMask
        } else {
            timesteps = MLX.full([b, tokenCount], values: MLXArray(sigma).asType(dtype))
        }

        let velocity = transformer.forward(latent: flat, timesteps: timesteps, context: context, rope: rope)
            .reshaped(b, f, h, w, c)
            .transposed(0, 4, 1, 2, 3)
        MLX.eval(velocity)

        if let debugDumpPrefix, !debugDumpPrefix.isEmpty, (debugDumpAll || i == 0) {
            let base = URL(fileURLWithPath: debugDumpPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            let step = i + 1
            let currentURL = parent.appendingPathComponent("\(stem)_\(label)_step\(step)_current.npy")
            let timestepsURL = parent.appendingPathComponent("\(stem)_\(label)_step\(step)_timesteps.npy")
            let velocityURL = parent.appendingPathComponent("\(stem)_\(label)_step\(step)_velocity.npy")
            let contextURL = parent.appendingPathComponent("\(stem)_\(label)_context.npy")
            let ropeCosURL = parent.appendingPathComponent("\(stem)_\(label)_rope_cos.npy")
            let ropeSinURL = parent.appendingPathComponent("\(stem)_\(label)_rope_sin.npy")
            try? MLX.save(array: current, url: currentURL)
            try? MLX.save(array: timesteps, url: timestepsURL)
            try? MLX.save(array: velocity, url: velocityURL)
            if i == 0 {
                try? MLX.save(array: context, url: contextURL)
                try? MLX.save(array: rope.cos, url: ropeCosURL)
                try? MLX.save(array: rope.sin, url: ropeSinURL)
            }
        }

        var denoised = toDenoised(noisy: current, velocity: velocity, sigma: sigma)
        MLX.eval(denoised)
        if debugDenoise {
            print("[LTX] \(label) step \(i + 1)/\(sigmas.count - 1) sigma=\(sigma) next=\(nextSigma)")
            print("[LTX] \(label) current \(tensorStatsString(current))")
            print("[LTX] \(label) velocity \(tensorStatsString(velocity))")
            print("[LTX] \(label) denoised \(tensorStatsString(denoised))")
        }
        if let conditioning {
            let one = MLXArray(1.0).asType(denoised.dtype)
            denoised = denoised * conditioning.denoiseMask + conditioning.cleanLatent * (one - conditioning.denoiseMask)
        }
        if nextSigma > 0 {
            let sigmaArr = MLXArray(sigma).asType(dtype)
            let nextArr = MLXArray(nextSigma).asType(dtype)
            current = denoised + nextArr * (current - denoised) / sigmaArr
        } else {
            current = denoised
        }
        MLX.eval(current)
    }

    return current
}
