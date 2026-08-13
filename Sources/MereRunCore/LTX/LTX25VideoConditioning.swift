import Foundation
import MLX

/// One upstream-style image guide for LTX 2.5.
///
/// `pixelFrameIndex` is expressed in decoded video frames. Frame zero replaces
/// the causal first-frame tokens; later frames are appended as clean keyframe
/// tokens with an exact one-pixel-frame RoPE span.
public struct LTXVideoConditioningInput: Sendable, Equatable {
    public let imageURL: URL
    public let pixelFrameIndex: Int
    public let strength: Float
    public let crf: Int?

    public init(
        imageURL: URL,
        pixelFrameIndex: Int,
        strength: Float = 1,
        crf: Int? = nil
    ) {
        precondition(crf.map { (0...51).contains($0) } ?? true)
        self.imageURL = imageURL
        self.pixelFrameIndex = pixelFrameIndex
        self.strength = strength
        self.crf = crf
    }
}

public struct LTXReferenceVideoConditioningInput: Sendable, Equatable {
    public let videoURL: URL
    public let strength: Float
    public let attentionStrength: Float?
    public let attentionMaskVideoURL: URL?
    public let downscaleFactor: Int
    public let temporalScaleFactor: Int

    public init(
        videoURL: URL,
        strength: Float = 1,
        attentionStrength: Float? = nil,
        attentionMaskVideoURL: URL? = nil,
        downscaleFactor: Int = 1,
        temporalScaleFactor: Int = 1
    ) {
        precondition((0...1).contains(strength), "strength must be in [0, 1]")
        precondition(attentionStrength.map { (0...1).contains($0) } ?? true)
        precondition(downscaleFactor > 0 && temporalScaleFactor > 0)
        self.videoURL = videoURL.standardizedFileURL
        self.strength = strength
        self.attentionStrength = attentionStrength
        self.attentionMaskVideoURL = attentionMaskVideoURL?.standardizedFileURL
        self.downscaleFactor = downscaleFactor
        self.temporalScaleFactor = temporalScaleFactor
    }
}

public struct LTXRetakeOptions: Sendable, Equatable {
    public let sourceVideoURL: URL
    public let startTime: Double
    public let endTime: Double
    public let regenerateVideo: Bool
    public let regenerateAudio: Bool

    public init(
        sourceVideoURL: URL,
        startTime: Double,
        endTime: Double,
        regenerateVideo: Bool = true,
        regenerateAudio: Bool = true
    ) {
        precondition(startTime.isFinite && endTime.isFinite && startTime >= 0 && startTime < endTime)
        precondition(regenerateVideo || regenerateAudio, "Retake must regenerate at least one modality.")
        self.sourceVideoURL = sourceVideoURL.standardizedFileURL
        self.startTime = startTime
        self.endTime = endTime
        self.regenerateVideo = regenerateVideo
        self.regenerateAudio = regenerateAudio
    }
}

/// Upstream Dub-It conditioning: one synchronized reference container supplies
/// both IC-LoRA video tokens and clean, negative-time audio reference tokens.
public struct LTXDubItOptions: Sendable, Equatable {
    public let referenceVideoURL: URL
    public let referenceStrength: Float

    public init(referenceVideoURL: URL, referenceStrength: Float = 1) {
        precondition((0...1).contains(referenceStrength), "referenceStrength must be in [0, 1]")
        self.referenceVideoURL = referenceVideoURL.standardizedFileURL
        self.referenceStrength = referenceStrength
    }
}

public struct LTXGeneratedKeyframeLayout: Sendable, Equatable {
    public let pixelFrameIndices: [Int]
    public let tokensPerKeyframe: Int
    public let firstToken: Int

    public init(pixelFrameIndices: [Int], tokensPerKeyframe: Int, firstToken: Int) {
        self.pixelFrameIndices = pixelFrameIndices
        self.tokensPerKeyframe = tokensPerKeyframe
        self.firstToken = firstToken
    }

    public var tokenCount: Int {
        pixelFrameIndices.count * tokensPerKeyframe
    }
}

public enum LTXGeneratedKeyframePositionError: Error, Equatable, LocalizedError {
    case negativeCount(Int)
    case insufficientFrames(count: Int, numFrames: Int)

    public var errorDescription: String? {
        switch self {
        case .negativeCount(let count):
            "Generated keyframe count must be nonnegative, got \(count)."
        case .insufficientFrames(let count, let numFrames):
            "Generated keyframes need at least count + 2 target frames; got count=\(count), frames=\(numFrames)."
        }
    }
}

/// Returns the exact evenly spaced interior pixel-frame positions used by the
/// official LTX pipeline for an integer generated-keyframe request.
public func ltxEvenlySpacedGeneratedKeyframePositions(
    count: Int,
    numFrames: Int
) throws -> [Int] {
    guard count >= 0 else {
        throw LTXGeneratedKeyframePositionError.negativeCount(count)
    }
    guard count > 0 else { return [] }
    guard numFrames >= count + 2 else {
        throw LTXGeneratedKeyframePositionError.insufficientFrames(
            count: count,
            numFrames: numFrames
        )
    }
    let lastFrame = Double(numFrames - 1)
    let divisor = Double(count + 1)
    return (1...count).map { index in
        Int((lastFrame * Double(index) / divisor).rounded(.toNearestOrEven))
    }
}

struct LTXVideoLatentShape: Equatable {
    let batch: Int
    let channels: Int
    let frames: Int
    let height: Int
    let width: Int

    var tokenCount: Int { frames * height * width }
    var tokensPerFrame: Int { height * width }
}

struct LTX25VideoTokenState {
    var latent: MLXArray
    var denoiseMask: MLXArray
    var positions: MLXArray
    var cleanLatent: MLXArray
    var keyframesMask: MLXArray
    var attentionMask: MLXArray?
    let targetShape: LTXVideoLatentShape
    var generatedKeyframeLayout: LTXGeneratedKeyframeLayout?

    init(initialLatent: MLXArray, positions: MLXArray) {
        precondition(initialLatent.ndim == 5, "video latent must be BCFHW")
        let shape = LTXVideoLatentShape(
            batch: initialLatent.dim(0),
            channels: initialLatent.dim(1),
            frames: initialLatent.dim(2),
            height: initialLatent.dim(3),
            width: initialLatent.dim(4)
        )
        precondition(
            positions.shape == [shape.batch, 3, shape.tokenCount, 2],
            "positions must match the target token grid"
        )
        let tokens = ltxVideoPatchify(initialLatent)
        self.latent = tokens
        self.denoiseMask = MLX.ones([shape.batch, shape.tokenCount, 1], dtype: tokens.dtype)
        self.positions = positions
        self.cleanLatent = MLX.zeros(tokens.shape, dtype: tokens.dtype)
        self.keyframesMask = makeLTXVideoKeyframesMask(
            batchSize: shape.batch,
            tokenCount: shape.tokenCount,
            tokensPerFirstFrame: shape.tokensPerFrame,
            dtype: tokens.dtype
        )
        self.attentionMask = nil
        self.targetShape = shape
        self.generatedKeyframeLayout = nil
    }

    mutating func applyTemporalRetake(
        cleanVideoLatent: MLXArray,
        startTime: Double,
        endTime: Double,
        fps: Double,
        regenerate: Bool
    ) {
        precondition(cleanVideoLatent.shape == [
            targetShape.batch,
            targetShape.channels,
            targetShape.frames,
            targetShape.height,
            targetShape.width,
        ])
        precondition(startTime >= 0 && startTime < endTime && fps > 0)
        let cleanTokens = ltxVideoPatchify(cleanVideoLatent).asType(latent.dtype)
        cleanLatent = replacingLTXTokenPrefix(cleanLatent, with: cleanTokens)
        let frameValues = (0..<targetShape.frames).map { latentFrame -> Float in
            guard regenerate else { return 0 }
            let pixelStart = max(0, latentFrame * 8 + 1 - 8)
            let pixelEnd = max(0, (latentFrame + 1) * 8 + 1 - 8)
            return Double(pixelEnd) / fps > startTime
                && Double(pixelStart) / fps < endTime
                ? 1
                : 0
        }
        let frameMask = MLXArray(frameValues)
            .reshaped(1, targetShape.frames, 1, 1)
        let expanded = broadcast(
            frameMask,
            to: [targetShape.batch, targetShape.frames, targetShape.height, targetShape.width]
        ).reshaped(targetShape.batch, targetShape.tokenCount, 1)
        denoiseMask = replacingLTXTokenPrefix(denoiseMask, with: expanded.asType(denoiseMask.dtype))
    }

    mutating func applyImageLatent(
        _ imageLatent: MLXArray,
        pixelFrameIndex: Int,
        strength: Float,
        fps: Double,
        replaceFirstFrame: Bool = true
    ) {
        precondition(imageLatent.ndim == 5, "image latent must be BCFHW")
        precondition(imageLatent.dim(0) == targetShape.batch, "image batch must match target")
        precondition(imageLatent.dim(1) == targetShape.channels, "image channels must match target")
        precondition(imageLatent.dim(2) == 1, "image latent must contain one frame")
        precondition(imageLatent.dim(3) == targetShape.height, "image height must match target")
        precondition(imageLatent.dim(4) == targetShape.width, "image width must match target")
        precondition(pixelFrameIndex >= 0, "pixelFrameIndex must be nonnegative")
        precondition(strength >= 0 && strength <= 1, "strength must be in [0, 1]")
        precondition(fps > 0, "fps must be positive")

        let tokens = ltxVideoPatchify(imageLatent).asType(latent.dtype)
        let guideMask = MLX.full(
            [targetShape.batch, targetShape.tokensPerFrame, 1],
            values: MLXArray(1 - strength).asType(denoiseMask.dtype)
        )
        if pixelFrameIndex == 0, replaceFirstFrame {
            cleanLatent = replacingLTXTokenPrefix(cleanLatent, with: tokens)
            denoiseMask = replacingLTXTokenPrefix(denoiseMask, with: guideMask)
            return
        }

        latent = MLX.concatenated([latent, MLX.zeros(tokens.shape, dtype: latent.dtype)], axis: 1)
        cleanLatent = MLX.concatenated([cleanLatent, tokens], axis: 1)
        denoiseMask = MLX.concatenated([denoiseMask, guideMask], axis: 1)
        positions = MLX.concatenated(
            [
                positions,
                makeLTXSinglePixelFramePositions(
                    batchSize: targetShape.batch,
                    height: targetShape.height,
                    width: targetShape.width,
                    pixelFrameIndex: pixelFrameIndex,
                    fps: fps
                ),
            ],
            axis: 2
        )
        keyframesMask = MLX.concatenated(
            [
                keyframesMask,
                MLX.zeros(
                    [targetShape.batch, targetShape.tokensPerFrame, 1],
                    dtype: keyframesMask.dtype
                ),
            ],
            axis: 1
        )
    }

    mutating func appendGeneratedKeyframeSlots(
        pixelFrameIndices: [Int],
        initialKeyframes: MLXArray? = nil,
        fps: Double
    ) {
        precondition(!pixelFrameIndices.isEmpty, "generated keyframe positions must not be empty")
        precondition(
            zip(pixelFrameIndices, pixelFrameIndices.dropFirst()).allSatisfy(<),
            "generated keyframe positions must be strictly increasing"
        )
        precondition(pixelFrameIndices[0] >= 0, "generated keyframe positions must be nonnegative")
        precondition(generatedKeyframeLayout == nil, "generated keyframe slots may only be appended once")
        precondition(fps > 0, "fps must be positive")

        let firstToken = latent.dim(1)
        let slotTokens: MLXArray
        if let initialKeyframes {
            precondition(initialKeyframes.ndim == 5, "initial keyframes must be BCKHW")
            precondition(initialKeyframes.dim(0) == targetShape.batch, "keyframe batch must match target")
            precondition(initialKeyframes.dim(1) == targetShape.channels, "keyframe channels must match target")
            precondition(initialKeyframes.dim(2) == pixelFrameIndices.count, "keyframe count must match positions")
            precondition(initialKeyframes.dim(3) == targetShape.height, "keyframe height must match target")
            precondition(initialKeyframes.dim(4) == targetShape.width, "keyframe width must match target")
            let perKeyframe = (0..<pixelFrameIndices.count).map { index in
                ltxVideoPatchify(initialKeyframes[0..., 0..., index..<index + 1, 0..., 0...])
            }
            slotTokens = MLX.concatenated(perKeyframe, axis: 1).asType(latent.dtype)
        } else {
            slotTokens = MLX.zeros(
                [
                    targetShape.batch,
                    targetShape.tokensPerFrame * pixelFrameIndices.count,
                    targetShape.channels,
                ],
                dtype: latent.dtype
            )
        }

        let slotPositions = MLX.concatenated(
            pixelFrameIndices.map {
                makeLTXSinglePixelFramePositions(
                    batchSize: targetShape.batch,
                    height: targetShape.height,
                    width: targetShape.width,
                    pixelFrameIndex: $0,
                    fps: fps
                )
            },
            axis: 2
        )
        latent = MLX.concatenated([latent, slotTokens], axis: 1)
        cleanLatent = MLX.concatenated(
            [cleanLatent, MLX.zeros(slotTokens.shape, dtype: cleanLatent.dtype)],
            axis: 1
        )
        denoiseMask = MLX.concatenated(
            [
                denoiseMask,
                MLX.ones(
                    [targetShape.batch, slotTokens.dim(1), 1],
                    dtype: denoiseMask.dtype
                ),
            ],
            axis: 1
        )
        positions = MLX.concatenated([positions, slotPositions], axis: 2)
        keyframesMask = MLX.concatenated(
            [
                keyframesMask,
                MLX.ones(
                    [targetShape.batch, slotTokens.dim(1), 1],
                    dtype: keyframesMask.dtype
                ),
            ],
            axis: 1
        )
        generatedKeyframeLayout = LTXGeneratedKeyframeLayout(
            pixelFrameIndices: pixelFrameIndices,
            tokensPerKeyframe: targetShape.tokensPerFrame,
            firstToken: firstToken
        )
    }

    /// Appends a clean reference-video sequence using the exact IC-LoRA token
    /// contract: zero noisy tokens, clean encoded latents, independently scaled
    /// spatial/temporal RoPE coordinates, and unmarked keyframe state.
    mutating func appendReferenceLatent(
        _ referenceLatent: MLXArray,
        downscaleFactor: Int = 1,
        temporalScaleFactor: Int = 1,
        strength: Float = 1,
        attentionStrength: Float? = nil,
        attentionWeights: MLXArray? = nil,
        fps: Double
    ) {
        precondition(referenceLatent.ndim == 5, "reference latent must be BCFHW")
        precondition(referenceLatent.dim(0) == targetShape.batch, "reference batch must match target")
        precondition(referenceLatent.dim(1) == targetShape.channels, "reference channels must match target")
        precondition(downscaleFactor > 0, "reference downscale factor must be positive")
        precondition(temporalScaleFactor > 0, "reference temporal scale factor must be positive")
        precondition((0...1).contains(strength), "reference strength must be in [0, 1]")
        precondition(attentionStrength.map { (0...1).contains($0) } ?? true, "attention strength must be in [0, 1]")
        precondition(
            attentionWeights.map { $0.shape == [targetShape.batch, referenceLatent.dim(2) * referenceLatent.dim(3) * referenceLatent.dim(4)] } ?? true,
            "attention weights must match the reference token grid"
        )
        precondition(fps > 0, "fps must be positive")

        let tokens = ltxVideoPatchify(referenceLatent).asType(latent.dtype)
        let previousTokenCount = latent.dim(1)
        latent = MLX.concatenated([latent, MLX.zeros(tokens.shape, dtype: latent.dtype)], axis: 1)
        cleanLatent = MLX.concatenated([cleanLatent, tokens], axis: 1)
        denoiseMask = MLX.concatenated(
            [
                denoiseMask,
                MLX.full(
                    [targetShape.batch, tokens.dim(1), 1],
                    values: MLXArray(1 - strength).asType(denoiseMask.dtype)
                ),
            ],
            axis: 1
        )
        positions = MLX.concatenated(
            [
                positions,
                makeLTXReferenceVideoPositions(
                    batchSize: targetShape.batch,
                    latentFrames: referenceLatent.dim(2),
                    latentHeight: referenceLatent.dim(3),
                    latentWidth: referenceLatent.dim(4),
                    downscaleFactor: downscaleFactor,
                    temporalScaleFactor: temporalScaleFactor,
                    fps: fps
                ),
            ],
            axis: 2
        )
        keyframesMask = MLX.concatenated(
            [
                keyframesMask,
                MLX.zeros([targetShape.batch, tokens.dim(1), 1], dtype: keyframesMask.dtype),
            ],
            axis: 1
        )
        attentionMask = extendingLTXReferenceAttentionMask(
            attentionMask,
            batchSize: targetShape.batch,
            noisyTokenCount: targetShape.tokenCount,
            existingTokenCount: previousTokenCount,
            newTokenCount: tokens.dim(1),
            attentionStrength: attentionStrength,
            attentionWeights: attentionWeights,
            dtype: latent.dtype
        )
    }

    mutating func addNoise(scale: Float) {
        let noise = MLXRandom.normal(latent.shape).asType(latent.dtype)
        addNoise(noise, scale: scale)
    }

    mutating func addNoise(_ noise: MLXArray, scale: Float) {
        precondition(noise.shape == latent.shape, "noise must match the token sequence")
        let scaleArray = MLXArray(scale).asType(latent.dtype)
        let noised = latent * (MLXArray(1).asType(latent.dtype) - scaleArray)
            + noise.asType(latent.dtype) * scaleArray
        latent = cleanLatent * (MLXArray(1).asType(latent.dtype) - denoiseMask)
            + noised * denoiseMask
    }

    func mainLatent() -> MLXArray {
        ltxVideoUnpatchify(
            latent[0..., 0..<targetShape.tokenCount, 0...],
            shape: targetShape
        )
    }

    func generatedKeyframes() -> MLXArray? {
        guard let layout = generatedKeyframeLayout else { return nil }
        let stop = layout.firstToken + layout.tokenCount
        precondition(stop <= latent.dim(1), "generated keyframe layout exceeds token sequence")
        let tokens = latent[0..., layout.firstToken..<stop, 0...]
        return tokens
            .reshaped(
                targetShape.batch,
                layout.pixelFrameIndices.count,
                targetShape.height,
                targetShape.width,
                targetShape.channels
            )
            .transposed(0, 4, 1, 2, 3)
    }
}

func makeLTXReferenceVideoPositions(
    batchSize: Int,
    latentFrames: Int,
    latentHeight: Int,
    latentWidth: Int,
    downscaleFactor: Int,
    temporalScaleFactor: Int,
    fps: Double,
    temporalScale: Int = 8,
    spatialScale: Int = 32
) -> MLXArray {
    precondition(batchSize > 0 && latentFrames > 0 && latentHeight > 0 && latentWidth > 0)
    let tokenCount = latentFrames * latentHeight * latentWidth
    var values = [Float](repeating: 0, count: batchSize * 3 * tokenCount * 2)
    let targetFrameDuration = 1 / Float(fps)
    let temporalShift = Float(temporalScaleFactor - 1) * targetFrameDuration
    for batch in 0..<batchSize {
        var token = 0
        for frame in 0..<latentFrames {
            for row in 0..<latentHeight {
                for column in 0..<latentWidth {
                    let base = ((batch * 3 * tokenCount) + token) * 2
                    let causalShift = Float(1 - temporalScale)
                    let rawStart = max(0, Float(frame * temporalScale) + causalShift)
                    let rawEnd = max(0, Float((frame + 1) * temporalScale) + causalShift)
                    values[base] = max(
                        0,
                        rawStart * Float(temporalScaleFactor) / Float(fps) - temporalShift
                    )
                    values[base + 1] = max(
                        0,
                        rawEnd * Float(temporalScaleFactor) / Float(fps) - temporalShift
                    )

                    let heightBase = ((batch * 3 * tokenCount) + tokenCount + token) * 2
                    values[heightBase] = Float(row * spatialScale * downscaleFactor)
                    values[heightBase + 1] = Float((row + 1) * spatialScale * downscaleFactor)

                    let widthBase = ((batch * 3 * tokenCount) + (2 * tokenCount) + token) * 2
                    values[widthBase] = Float(column * spatialScale * downscaleFactor)
                    values[widthBase + 1] = Float((column + 1) * spatialScale * downscaleFactor)
                    token += 1
                }
            }
        }
    }
    return MLXArray(values).reshaped(batchSize, 3, tokenCount, 2)
}

private func extendingLTXReferenceAttentionMask(
    _ existingMask: MLXArray?,
    batchSize: Int,
    noisyTokenCount: Int,
    existingTokenCount: Int,
    newTokenCount: Int,
    attentionStrength: Float?,
    attentionWeights: MLXArray?,
    dtype: DType
) -> MLXArray? {
    guard attentionStrength != nil || attentionWeights != nil || existingMask != nil else { return nil }
    let topLeft = existingMask
        ?? MLX.ones([batchSize, existingTokenCount, existingTokenCount], dtype: dtype)
    let crossWeights = attentionWeights.map {
        $0.asType(dtype) * MLXArray(attentionStrength ?? 1).asType(dtype)
    } ?? MLX.full(
        [batchSize, newTokenCount],
        values: MLXArray(attentionStrength ?? 1).asType(dtype)
    )
    let noisyCross = broadcast(
        crossWeights.reshaped(batchSize, 1, newTokenCount),
        to: [batchSize, noisyTokenCount, newTokenCount]
    )
    let previousReferenceCount = existingTokenCount - noisyTokenCount
    let previousCross = MLX.zeros(
        [batchSize, previousReferenceCount, newTokenCount],
        dtype: dtype
    )
    let topRight = previousReferenceCount == 0
        ? noisyCross
        : MLX.concatenated([noisyCross, previousCross], axis: 1)
    let bottomLeft = topRight.transposed(0, 2, 1)
    let bottomRight = MLX.ones([batchSize, newTokenCount, newTokenCount], dtype: dtype)
    return MLX.concatenated(
        [
            MLX.concatenated([topLeft, topRight], axis: 2),
            MLX.concatenated([bottomLeft, bottomRight], axis: 2),
        ],
        axis: 1
    )
}

/// Matches upstream IC-LoRA mask downsampling: area-average spatial pixels,
/// retain the causal first frame, then average each remaining temporal group.
func downsampleLTXReferenceAttentionMask(
    _ maskVideo: MLXArray,
    targetLatentShape: LTXVideoLatentShape
) -> MLXArray {
    precondition(maskVideo.ndim == 5, "mask video must be BCFHW")
    precondition(maskVideo.dim(0) == targetLatentShape.batch, "mask batch must match reference")
    precondition(maskVideo.dim(1) == 1, "mask video must be single-channel")
    precondition(maskVideo.dim(2) >= 1, "mask video must contain at least one frame")
    precondition(maskVideo.dim(3).isMultiple(of: targetLatentShape.height))
    precondition(maskVideo.dim(4).isMultiple(of: targetLatentShape.width))

    let heightFactor = maskVideo.dim(3) / targetLatentShape.height
    let widthFactor = maskVideo.dim(4) / targetLatentShape.width
    let spatial = MLX.mean(
        maskVideo.reshaped(
            targetLatentShape.batch,
            1,
            maskVideo.dim(2),
            targetLatentShape.height,
            heightFactor,
            targetLatentShape.width,
            widthFactor
        ),
        axes: [4, 6]
    )
    let first = spatial[0..., 0..., 0..<1, 0..., 0...]
    let latentMask: MLXArray
    if maskVideo.dim(2) > 1, targetLatentShape.frames > 1 {
        let remainingPixelFrames = maskVideo.dim(2) - 1
        let remainingLatentFrames = targetLatentShape.frames - 1
        precondition(remainingPixelFrames.isMultiple(of: remainingLatentFrames))
        let temporalFactor = remainingPixelFrames / remainingLatentFrames
        let rest = MLX.mean(
            spatial[0..., 0..., 1..., 0..., 0...].reshaped(
                targetLatentShape.batch,
                1,
                remainingLatentFrames,
                temporalFactor,
                targetLatentShape.height,
                targetLatentShape.width
            ),
            axis: 3
        )
        latentMask = MLX.concatenated([first, rest], axis: 2)
    } else {
        latentMask = first
    }
    return latentMask.reshaped(targetLatentShape.batch, targetLatentShape.tokenCount)
}

func ltxVideoPatchify(_ latent: MLXArray) -> MLXArray {
    precondition(latent.ndim == 5, "video latent must be BCFHW")
    return latent
        .transposed(0, 2, 3, 4, 1)
        .reshaped(latent.dim(0), latent.dim(2) * latent.dim(3) * latent.dim(4), latent.dim(1))
}

func ltxVideoUnpatchify(_ tokens: MLXArray, shape: LTXVideoLatentShape) -> MLXArray {
    precondition(
        tokens.shape == [shape.batch, shape.tokenCount, shape.channels],
        "tokens must match target shape"
    )
    return tokens
        .reshaped(shape.batch, shape.frames, shape.height, shape.width, shape.channels)
        .transposed(0, 4, 1, 2, 3)
}

func makeLTXSinglePixelFramePositions(
    batchSize: Int,
    height: Int,
    width: Int,
    pixelFrameIndex: Int,
    fps: Double,
    spatialScale: Int = 32
) -> MLXArray {
    precondition(batchSize > 0, "batchSize must be positive")
    precondition(height > 0 && width > 0, "spatial dimensions must be positive")
    precondition(pixelFrameIndex >= 0, "pixelFrameIndex must be nonnegative")
    precondition(fps > 0, "fps must be positive")
    let tokenCount = height * width
    var values = [Float](repeating: 0, count: batchSize * 3 * tokenCount * 2)
    for batch in 0..<batchSize {
        for row in 0..<height {
            for column in 0..<width {
                let token = row * width + column
                let temporalBase = ((batch * 3 * tokenCount) + token) * 2
                values[temporalBase] = Float(pixelFrameIndex) / Float(fps)
                values[temporalBase + 1] = Float(pixelFrameIndex + 1) / Float(fps)

                let heightBase = ((batch * 3 * tokenCount) + tokenCount + token) * 2
                values[heightBase] = Float(row * spatialScale)
                values[heightBase + 1] = Float((row + 1) * spatialScale)

                let widthBase = ((batch * 3 * tokenCount) + (2 * tokenCount) + token) * 2
                values[widthBase] = Float(column * spatialScale)
                values[widthBase + 1] = Float((column + 1) * spatialScale)
            }
        }
    }
    return MLXArray(values).reshaped(batchSize, 3, tokenCount, 2)
}

private func replacingLTXTokenPrefix(_ array: MLXArray, with replacement: MLXArray) -> MLXArray {
    precondition(array.dim(0) == replacement.dim(0), "replacement batch must match")
    precondition(array.dim(2) == replacement.dim(2), "replacement width must match")
    precondition(replacement.dim(1) <= array.dim(1), "replacement must fit in token sequence")
    guard replacement.dim(1) < array.dim(1) else { return replacement }
    return MLX.concatenated(
        [replacement, array[0..., replacement.dim(1)..., 0...]],
        axis: 1
    )
}
