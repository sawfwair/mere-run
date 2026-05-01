import Foundation
import MLX
import MLXNN
import MLXRandom

final class OobleckVAE: Module {
    let config: OobleckVAEConfig

    @ModuleInfo(key: "encoder") var encoder: OobleckEncoder
    @ModuleInfo(key: "decoder") var decoder: OobleckDecoder

    init(config: OobleckVAEConfig) {
        self.config = config
        self._encoder.wrappedValue = OobleckEncoder(config: config)
        self._decoder.wrappedValue = OobleckDecoder(config: config)
    }

    func encode(_ audio: MLXArray, sample: Bool = true, seed: UInt64? = nil) -> MLXArray {
        precondition(audio.ndim == 3, "audio must be [B, S, C].")
        precondition(audio.dim(2) == config.audioChannels, "audio channel dimension must be \(config.audioChannels).")

        let stats = encoder(audio)
        let latentChannels = config.decoderInputChannels
        precondition(stats.dim(2) >= latentChannels * 2, "encoder output must contain mean and logvar channels.")

        let mean = stats[0..., 0..., 0..<latentChannels]
        if !sample {
            return mean
        }

        let logvar = stats[0..., 0..., latentChannels..<(latentChannels * 2)]
        let clampedLogvar = MLX.clip(logvar, min: -30.0, max: 20.0)
        let std = MLX.exp(clampedLogvar * 0.5)

        let noise: MLXArray
        if let seed {
            noise = MLXRandom.normal(mean.shape, key: MLXRandom.key(seed))
        } else {
            noise = MLXRandom.normal(mean.shape)
        }
        return mean + std * noise.asType(mean.dtype)
    }

    func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents)
    }

    /// Encode long waveforms by chunking in waveform-space and discarding overlapped latents.
    ///
    /// This mirrors ACE-Step's overlap-discard strategy used by `tiled_encode`.
    /// - Parameters:
    ///   - audio: `[B, S, C]`
    ///   - chunkSize: Waveform samples per window
    ///   - overlap: Waveform samples of overlap on both sides
    func tiledEncode(
        _ audio: MLXArray,
        chunkSize: Int = 48_000 * 30,
        overlap: Int = 48_000 * 2,
        sample: Bool = true,
        seed: UInt64? = nil
    ) -> MLXArray {
        precondition(audio.ndim == 3, "audio must be [B, S, C].")
        precondition(audio.dim(2) == config.audioChannels, "audio channel dimension must be \(config.audioChannels).")

        let S = audio.dim(1)
        if S <= chunkSize {
            return encode(audio, sample: sample, seed: seed)
        }

        let stride = chunkSize - 2 * overlap
        precondition(stride > 0, "chunkSize \(chunkSize) must be > 2 * overlap \(overlap)")

        let numSteps = (S + stride - 1) / stride
        var encoded: [MLXArray] = []
        encoded.reserveCapacity(numSteps)

        var downsampleFactor: Double?
        for i in 0..<numSteps {
            let coreStart = i * stride
            let coreEnd = min(coreStart + stride, S)

            let winStart = max(0, coreStart - overlap)
            let winEnd = min(S, coreEnd + overlap)

            let chunk = audio[0..., winStart..<winEnd, 0...]
            let chunkSeed = seed.map { $0 &+ UInt64(i) }
            let latentChunk = encode(chunk, sample: sample, seed: chunkSeed)

            if downsampleFactor == nil {
                downsampleFactor = Double(chunk.dim(1)) / Double(latentChunk.dim(1))
            }

            let factor = downsampleFactor!
            let trimStart = Int(round(Double(coreStart - winStart) / factor))
            let trimEnd = Int(round(Double(winEnd - coreEnd) / factor))

            let latentLen = latentChunk.dim(1)
            let endIndex = trimEnd > 0 ? (latentLen - trimEnd) : latentLen
            encoded.append(latentChunk[0..., trimStart..<endIndex, 0...])
        }

        return MLX.concatenated(encoded, axis: 1)
    }

    /// Decode long sequences by chunking in latent-space and discarding overlapped audio.
    ///
    /// This follows ACE-Step's overlap-discard tiling strategy to avoid boundary artifacts.
    /// - Parameters:
    ///   - latents: `[B, T, C]`
    ///   - chunkSize: Latent frames per window
    ///   - overlap: Latent frames of overlap on both sides
    func tiledDecode(_ latents: MLXArray, chunkSize: Int = 512, overlap: Int = 64) -> MLXArray {
        let T = latents.dim(1)

        if T <= chunkSize {
            return decode(latents)
        }

        let stride = chunkSize - 2 * overlap
        precondition(stride > 0, "chunkSize \(chunkSize) must be > 2 * overlap \(overlap)")

        let upsampleFactor = config.downsamplingRatios.reduce(1, *)
        let numSteps = (T + stride - 1) / stride

        var decoded: [MLXArray] = []
        decoded.reserveCapacity(numSteps)

        for i in 0..<numSteps {
            let coreStart = i * stride
            let coreEnd = min(coreStart + stride, T)

            let winStart = max(0, coreStart - overlap)
            let winEnd = min(T, coreEnd + overlap)

            let latentChunk = latents[0..., winStart..<winEnd, 0...]
            let audioChunk = decode(latentChunk)

            let trimStart = (coreStart - winStart) * upsampleFactor
            let trimEnd = (winEnd - coreEnd) * upsampleFactor

            let audioLen = audioChunk.dim(1)
            let endIndex = trimEnd > 0 ? (audioLen - trimEnd) : audioLen

            decoded.append(audioChunk[0..., trimStart..<endIndex, 0...])
        }

        return MLX.concatenated(decoded, axis: 1)
    }
}
