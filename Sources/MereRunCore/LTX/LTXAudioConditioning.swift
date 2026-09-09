import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func validateLTXAudioSegment(
    metadata: MediaAudioMetadata,
    startTime: Double,
    duration: Double
) throws {
    guard metadata.channelCount == 1 || metadata.channelCount == 2 else {
        throw LTXUnifiedAVGeneratorError.unsupportedAudioChannels(metadata.channelCount)
    }
    let availableDuration = max(0, metadata.durationSeconds - startTime)
    guard availableDuration + 1e-6 >= duration else {
        throw LTXUnifiedAVGeneratorError.audioSegmentTooShort(
            required: duration,
            available: availableDuration
        )
    }
}

func decodeExactStereoAudioSegment(
    url: URL,
    startTime: Double,
    duration: Double,
    sampleRate: Int
) throws -> MediaAudioBuffer {
    let decoded = try MediaAudioIO.decodeSegment(
        url,
        startTime: startTime,
        duration: duration,
        targetSampleRate: sampleRate,
        channels: 2
    )
    let frameCount = Int((duration * Double(sampleRate)).rounded(.toNearestOrEven))
    let requiredSampleCount = frameCount * 2
    guard decoded.samples.count >= requiredSampleCount else {
        throw LTXUnifiedAVGeneratorError.audioDecodeReturnedTooFewSamples(
            required: requiredSampleCount,
            actual: decoded.samples.count
        )
    }
    return MediaAudioBuffer(
        samples: Array(decoded.samples.prefix(requiredSampleCount)),
        sampleRate: sampleRate,
        channelCount: 2,
        isInterleaved: true
    )
}

func planarAudioChannels(_ audio: MediaAudioBuffer) -> [[Float]] {
    let frameCount = audio.samples.count / audio.channelCount
    var channels = [[Float]](
        repeating: [Float](repeating: 0, count: frameCount),
        count: audio.channelCount
    )
    if audio.isInterleaved {
        for frame in 0..<frameCount {
            for channel in 0..<audio.channelCount {
                channels[channel][frame] = audio.samples[frame * audio.channelCount + channel]
            }
        }
    } else {
        for channel in 0..<audio.channelCount {
            let start = channel * frameCount
            channels[channel] = Array(audio.samples[start..<(start + frameCount)])
        }
    }
    return channels
}

func makeLTXAudioTemporalConditioning(
    cleanLatent: MLXArray,
    startTime: Double,
    endTime: Double,
    regenerate: Bool
) -> LTXLatentConditioningState {
    precondition(cleanLatent.ndim == 4, "audio latent must be BCTF")
    precondition(startTime >= 0 && startTime < endTime)
    let frameCount = cleanLatent.dim(2)
    let values = (0..<frameCount).map { frame -> Float in
        guard regenerate else { return 0 }
        let start = Double(max(0, frame * LTXAudioLatentDownsampleFactor + 1
            - LTXAudioLatentDownsampleFactor))
            * Double(LTXAudioHopLength) / Double(LTXAudioLatentSampleRate)
        let end = Double(max(0, (frame + 1) * LTXAudioLatentDownsampleFactor + 1
            - LTXAudioLatentDownsampleFactor))
            * Double(LTXAudioHopLength) / Double(LTXAudioLatentSampleRate)
        return end > startTime && start < endTime ? 1 : 0
    }
    let mask = MLXArray(values)
        .reshaped(1, 1, frameCount, 1)
        .asType(cleanLatent.dtype)
    return LTXLatentConditioningState(
        latent: cleanLatent,
        cleanLatent: cleanLatent,
        denoiseMask: mask
    )
}

struct LTXAudioReferenceConditioningState {
    let state: LTXLatentConditioningState
    let positions: MLXArray
    let targetFrameCount: Int

    func mainLatent(from latent: MLXArray) -> MLXArray {
        latent[0..., 0..., 0..<targetFrameCount, 0...]
    }
}

/// Mirrors upstream `AudioConditionByReferenceLatent`: reference tokens are
/// appended clean after the target sequence and placed immediately before time
/// zero, with a 40 ms gap. A frozen target is used by Dub-It's second stage.
func makeLTXAudioReferenceConditioning(
    targetLatent: MLXArray,
    referenceLatent: MLXArray,
    frozenTarget: Bool
) -> LTXAudioReferenceConditioningState {
    precondition(targetLatent.ndim == 4, "target audio latent must be BCTF")
    precondition(referenceLatent.ndim == 4, "reference audio latent must be BCTF")
    precondition(targetLatent.dim(0) == referenceLatent.dim(0), "audio batches must match")
    precondition(targetLatent.dim(1) == referenceLatent.dim(1), "audio channels must match")
    precondition(targetLatent.dim(3) == referenceLatent.dim(3), "audio mel bins must match")

    let targetFrames = targetLatent.dim(2)
    let referenceFrames = referenceLatent.dim(2)
    let dtype = targetLatent.dtype
    let targetMask = frozenTarget
        ? MLX.zeros([targetLatent.dim(0), 1, targetFrames, 1], dtype: dtype)
        : MLX.ones([targetLatent.dim(0), 1, targetFrames, 1], dtype: dtype)
    let referenceMask = MLX.zeros(
        [referenceLatent.dim(0), 1, referenceFrames, 1],
        dtype: dtype
    )
    let targetClean = frozenTarget
        ? targetLatent
        : MLX.zeros(targetLatent.shape, dtype: dtype)
    let typedReference = referenceLatent.asType(dtype)
    let state = LTXLatentConditioningState(
        latent: MLX.concatenated([targetLatent, typedReference], axis: 2),
        cleanLatent: MLX.concatenated([targetClean, typedReference], axis: 2),
        denoiseMask: MLX.concatenated([targetMask, referenceMask], axis: 2)
    )

    let targetPositions = createAudioPositionGrid(
        batchSize: targetLatent.dim(0),
        audioFrames: targetFrames
    )
    let referencePositions = createAudioPositionGrid(
        batchSize: referenceLatent.dim(0),
        audioFrames: referenceFrames
    )
    let lastReferenceEnd = Float(max(
        0,
        referenceFrames * LTXAudioLatentDownsampleFactor
            + 1 - LTXAudioLatentDownsampleFactor
    )) * Float(LTXAudioHopLength) / Float(LTXAudioLatentSampleRate)
    let shiftedReferencePositions = referencePositions - MLXArray(lastReferenceEnd + 0.04)
    let positions = MLX.concatenated(
        [targetPositions, shiftedReferencePositions],
        axis: 2
    ).asType(.float32)
    return LTXAudioReferenceConditioningState(
        state: state,
        positions: positions,
        targetFrameCount: targetFrames
    )
}

func encodeLTX23AudioLatents(
    spectrogram: MLXArray,
    requiredFrameCount: Int,
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> MLXArray {
    let encoder = LTXAudioEncoder()
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: encoder,
        dtype: dtype,
        verify: .none,
        include: { key in
            key.hasPrefix("audio_vae.encoder.")
                || key.hasPrefix("audio_vae.per_channel_statistics.")
        },
        mapper: { key, value in
            mapAudioVaeEncoderWeight(
                key: key,
                value: value,
                dtype: dtype,
                sourceLayout: sourceLayout
            )
        },
        batchSize: 24
    )
    let encoded = encoder.encode(spectrogram: spectrogram.asType(dtype))
    guard encoded.dim(2) >= requiredFrameCount else {
        throw LTXUnifiedAVGeneratorError.audioLatentTooShort(
            required: requiredFrameCount,
            actual: encoded.dim(2)
        )
    }
    let cropped = encoded[0..., 0..., 0..<requiredFrameCount, 0...]
    MLX.eval(cropped)
    return cropped
}
