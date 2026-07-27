import MLX

enum ACEStepRepaint {
    struct Conditioning {
        var sourceLatents: MLXArray
        var chunkMasks: MLXArray
        var repaintMask: MLXArray
        var latentRange: Range<Int>
    }

    static func prepareConditioning(
        cleanSourceLatents: MLXArray,
        silenceLatents: MLXArray,
        chunkChannels: Int,
        configuration: ACEStepRepaintConfiguration,
        task: ACEStepTask
    ) -> Conditioning {
        precondition(cleanSourceLatents.ndim == 3, "cleanSourceLatents must be [B,T,D].")
        precondition(
            silenceLatents.shape == cleanSourceLatents.shape,
            "silenceLatents must match cleanSourceLatents."
        )
        precondition(chunkChannels > 0, "chunkChannels must be positive.")

        let batchSize = cleanSourceLatents.dim(0)
        let totalFrames = cleanSourceLatents.dim(1)
        let range = configuration.latentRange(totalFrames: totalFrames)
        let maskValues = (0..<(batchSize * totalFrames)).map { index -> Float in
            range.contains(index % totalFrames) ? 1 : 0
        }
        let repaintMask = MLXArray(maskValues, [batchSize, totalFrames]).asType(.bool)
        let expandedMask = repaintMask.reshaped(batchSize, totalFrames, 1)

        let sourceLatents: MLXArray
        if task == .lego {
            sourceLatents = cleanSourceLatents
        } else {
            sourceLatents = MLX.where(
                expandedMask,
                silenceLatents.asType(cleanSourceLatents.dtype),
                cleanSourceLatents
            )
        }

        let chunkMasks: MLXArray
        switch configuration.chunkMaskMode {
        case .auto:
            chunkMasks = MLXArray.ones(
                [batchSize, totalFrames, chunkChannels],
                dtype: .float32
            )
        case .explicit:
            chunkMasks = MLX.broadcast(
                repaintMask.asType(.float32).reshaped(batchSize, totalFrames, 1),
                to: [batchSize, totalFrames, chunkChannels]
            )
        }

        return Conditioning(
            sourceLatents: sourceLatents,
            chunkMasks: chunkMasks,
            repaintMask: repaintMask,
            latentRange: range
        )
    }

    static func injectPreservedSource(
        generatedLatents: MLXArray,
        cleanSourceLatents: MLXArray,
        repaintMask: MLXArray,
        nextTimestep: Float,
        noise: MLXArray
    ) -> MLXArray {
        precondition(generatedLatents.shape == cleanSourceLatents.shape)
        precondition(generatedLatents.shape == noise.shape)
        let batchSize = generatedLatents.dim(0)
        let totalFrames = generatedLatents.dim(1)
        precondition(repaintMask.shape == [batchSize, totalFrames])

        let t = MLXArray(nextTimestep).asType(generatedLatents.dtype)
        let noisedSource = t * noise
            + MLXArray(Float(1 - nextTimestep)).asType(generatedLatents.dtype)
                * cleanSourceLatents.asType(generatedLatents.dtype)
        return MLX.where(
            repaintMask.reshaped(batchSize, totalFrames, 1),
            generatedLatents,
            noisedSource
        )
    }

    static func blendLatentBoundaries(
        generatedLatents: MLXArray,
        cleanSourceLatents: MLXArray,
        repaintMask: MLXArray,
        crossfadeFrames: Int
    ) -> MLXArray {
        precondition(generatedLatents.shape == cleanSourceLatents.shape)
        let batchSize = generatedLatents.dim(0)
        let totalFrames = generatedLatents.dim(1)
        precondition(repaintMask.shape == [batchSize, totalFrames])

        MLX.eval(repaintMask)
        let mask = repaintMask.asArray(Bool.self)
        var soft = mask.map { $0 ? Float(1) : Float(0) }

        if crossfadeFrames > 0 {
            for batch in 0..<batchSize {
                let offset = batch * totalFrames
                let row = Array(mask[offset..<(offset + totalFrames)])
                guard let left = row.firstIndex(of: true),
                      let last = row.lastIndex(of: true)
                else {
                    continue
                }
                let right = last + 1
                let fadeStart = max(left - crossfadeFrames, 0)
                let leftLength = left - fadeStart
                if leftLength > 0 {
                    for index in 0..<leftLength {
                        soft[offset + fadeStart + index] = Float(index + 1) / Float(leftLength + 1)
                    }
                }
                let fadeEnd = min(right + crossfadeFrames, totalFrames)
                let rightLength = fadeEnd - right
                if rightLength > 0 {
                    for index in 0..<rightLength {
                        soft[offset + right + index] = Float(rightLength - index) / Float(rightLength + 1)
                    }
                }
            }
        }

        let blend = MLXArray(soft, [batchSize, totalFrames, 1])
            .asType(generatedLatents.dtype)
        return blend * generatedLatents
            + (MLXArray(Float(1)).asType(generatedLatents.dtype) - blend)
                * cleanSourceLatents.asType(generatedLatents.dtype)
    }

    static func spliceWaveform(
        generatedAudio: MLXArray,
        sourceAudio: MLXArray,
        startSeconds: Float,
        endSeconds: Float,
        crossfadeSeconds: Float,
        sampleRate: Int = 48_000
    ) -> MLXArray {
        precondition(generatedAudio.ndim == 3, "generatedAudio must be [B,S,C].")
        precondition(sourceAudio.ndim == 3, "sourceAudio must be [B,S,C].")
        precondition(generatedAudio.dim(0) == sourceAudio.dim(0))
        precondition(generatedAudio.dim(2) == sourceAudio.dim(2))

        let batchSize = generatedAudio.dim(0)
        let channels = generatedAudio.dim(2)
        let generatedSamples = generatedAudio.dim(1)
        let sourceSamples = sourceAudio.dim(1)
        let commonSamples = min(generatedSamples, sourceSamples)
        guard commonSamples > 0 else {
            return generatedAudio
        }

        let start = max(0, min(Int(startSeconds * Float(sampleRate)), commonSamples))
        let resolvedEndSeconds = endSeconds > startSeconds
            ? endSeconds
            : Float(commonSamples) / Float(sampleRate)
        let end = max(start, min(Int(resolvedEndSeconds * Float(sampleRate)), commonSamples))
        if start == 0 && end >= commonSamples {
            return generatedAudio
        }

        let crossfadeSamples = max(0, Int(crossfadeSeconds * Float(sampleRate)))
        var mask = Array(repeating: Float(0), count: commonSamples)
        if start < end {
            for sample in start..<end {
                mask[sample] = 1
            }
        }

        let fadeStart = max(start - crossfadeSamples, 0)
        let leftLength = start - fadeStart
        if leftLength > 0 {
            for index in 0..<leftLength {
                mask[fadeStart + index] = Float(index + 1) / Float(leftLength + 1)
            }
        }

        let fadeEnd = min(end + crossfadeSamples, commonSamples)
        let rightLength = fadeEnd - end
        if rightLength > 0 {
            for index in 0..<rightLength {
                mask[end + index] = Float(rightLength - index) / Float(rightLength + 1)
            }
        }

        let blend = MLX.broadcast(
            MLXArray(mask, [1, commonSamples, 1]),
            to: [batchSize, commonSamples, channels]
        ).asType(generatedAudio.dtype)
        let generatedHead = generatedAudio[0..., 0..<commonSamples, 0...]
        let sourceHead = sourceAudio[0..., 0..<commonSamples, 0...].asType(generatedAudio.dtype)
        let spliced = blend * generatedHead
            + (MLXArray(Float(1)).asType(generatedAudio.dtype) - blend) * sourceHead

        guard generatedSamples > commonSamples else {
            return spliced
        }
        return MLX.concatenated(
            [spliced, generatedAudio[0..., commonSamples..<generatedSamples, 0...]],
            axis: 1
        )
    }
}
