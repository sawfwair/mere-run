import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func ltx25ImageConditionings(
    options: LTXUnifiedAVGenerationOptions
) -> [LTXVideoConditioningInput] {
    var values = options.imageConditionings
    if let sourceImageURL = options.sourceImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: sourceImageURL,
                pixelFrameIndex: options.imageFrameIndex,
                strength: options.imageStrength
            )
        )
    }
    if let endImageURL = options.endImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: endImageURL,
                pixelFrameIndex: options.numFrames - 1,
                strength: options.endImageStrength
            )
        )
    }
    return values
}

func ltx25ImageConditionings(
    options: LTXAudioToVideoGenerationOptions
) -> [LTXVideoConditioningInput] {
    var values = options.imageConditionings
    if let sourceImageURL = options.sourceImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: sourceImageURL,
                pixelFrameIndex: options.imageFrameIndex,
                strength: options.imageStrength
            )
        )
    }
    if let endImageURL = options.endImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: endImageURL,
                pixelFrameIndex: options.numFrames - 1,
                strength: options.endImageStrength
            )
        )
    }
    return values
}

func makeConditionedLTX25VideoTokenState(
    initialLatent: MLXArray,
    positions: MLXArray,
    imageConditionings: [LTXVideoConditioningInput],
    generatedKeyframeIndices: [Int],
    initialGeneratedKeyframes: MLXArray?,
    encoder: LTXVideoEncoder?,
    pixelWidth: Int,
    pixelHeight: Int,
    fps: Double,
    replaceFirstImage: Bool = true,
    hdrColorSpace: LTXHDRColorSpace? = nil
) throws -> LTX25VideoTokenState {
    var state = LTX25VideoTokenState(initialLatent: initialLatent, positions: positions)
    if !imageConditionings.isEmpty {
        guard let encoder else {
            throw LTXUnifiedAVGeneratorError.encoderNotLoaded
        }
        for input in imageConditionings {
            let image = try loadImageForEncoding(
                url: input.imageURL,
                width: pixelWidth,
                height: pixelHeight,
                dtype: initialLatent.dtype,
                hdrColorSpace: hdrColorSpace,
                crf: input.crf ?? 18
            )
            state.applyImageLatent(
                encoder.encode(image: image),
                pixelFrameIndex: input.pixelFrameIndex,
                strength: input.strength,
                fps: fps,
                replaceFirstFrame: replaceFirstImage
            )
        }
    }
    if !generatedKeyframeIndices.isEmpty {
        state.appendGeneratedKeyframeSlots(
            pixelFrameIndices: generatedKeyframeIndices,
            initialKeyframes: initialGeneratedKeyframes,
            fps: fps
        )
    }
    return state
}
