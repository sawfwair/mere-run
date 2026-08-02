import Foundation
@preconcurrency import MLX

public final class MelBandRoFormerSeparator {
    public typealias ProgressHandler = @Sendable (_ completedChunks: Int, _ totalChunks: Int) -> Void

    public let checkpoint: MelBandRoFormerCheckpoint
    public let dtype: DType
    private let model: MelBandRoFormer

    private init(
        checkpoint: MelBandRoFormerCheckpoint,
        dtype: DType,
        model: MelBandRoFormer
    ) {
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.model = model
    }

    public static func load(
        resources: MelBandRoFormerResources,
        dtype: DType = .float16
    ) throws -> MelBandRoFormerSeparator {
        let checkpoint = try resources.resolve()
        let model = try MelBandRoFormer.load(checkpoint: checkpoint, dtype: dtype)
        return MelBandRoFormerSeparator(checkpoint: checkpoint, dtype: dtype, model: model)
    }

    public func separate(
        interleavedSamples: [Float],
        sampleRate: Int,
        channels: Int,
        overlap: Int? = nil,
        progress: ProgressHandler? = nil
    ) throws -> RoFormerSeparationResult {
        let configuration = checkpoint.configuration
        guard sampleRate == configuration.sampleRate, channels == configuration.audioChannels else {
            throw RoFormerError.unsupportedAudio(sampleRate: sampleRate, channels: channels)
        }
        guard !interleavedSamples.isEmpty, interleavedSamples.count.isMultiple(of: channels) else {
            throw RoFormerError.invalidAudioBuffer
        }
        let started = ContinuousClock.now
        let frameCount = interleavedSamples.count / channels
        let resolvedOverlap = overlap ?? configuration.overlap
        let plan = try RoFormerChunkPlan.make(
            sampleCount: frameCount,
            chunkSize: configuration.chunkSize,
            overlap: resolvedOverlap
        )
        let original = RoFormerSeparator.deinterleave(interleavedSamples, channels: channels)
        let working = plan.usesBorderPadding
            ? original.map { RoFormerSeparator.reflectPad($0, amount: plan.border) }
            : original
        let workingCount = working[0].count
        var accumulated = Array(
            repeating: Array(
                repeating: [Float](repeating: 0, count: workingCount),
                count: channels
            ),
            count: configuration.numStems
        )
        var weights = [Float](repeating: 0, count: workingCount)

        for (chunkIndex, start) in plan.starts.enumerated() {
            let available = min(configuration.chunkSize, workingCount - start)
            var channelMajorChunk = [Float](
                repeating: 0,
                count: channels * configuration.chunkSize
            )
            for channel in 0..<channels {
                let source = working[channel]
                for offset in 0..<available {
                    channelMajorChunk[channel * configuration.chunkSize + offset] = source[start + offset]
                }
                if available > configuration.chunkSize / 2, available < configuration.chunkSize {
                    for offset in available..<configuration.chunkSize {
                        let reflected = 2 * available - offset - 2
                        channelMajorChunk[channel * configuration.chunkSize + offset] = source[start + reflected]
                    }
                }
            }
            let input = MLXArray(channelMajorChunk)
                .reshaped(1, channels, configuration.chunkSize)
                .asType(dtype)
            let output = model(input)[0].asType(.float32)
            MLX.eval(output)
            let separated = output.asArray(Float.self)
            let fade = plan.fadeWindow(chunkSize: configuration.chunkSize, chunkIndex: chunkIndex)
            for offset in 0..<available {
                let destination = start + offset
                let weight = fade[offset]
                weights[destination] += weight
                for stem in 0..<configuration.numStems {
                    for channel in 0..<channels {
                        let valueIndex = stem * channels * configuration.chunkSize
                            + channel * configuration.chunkSize
                            + offset
                        let value = separated[valueIndex]
                        accumulated[stem][channel][destination] += (value.isFinite ? value : 0) * weight
                    }
                }
            }
            MLX.Memory.clearCache()
            progress?(chunkIndex + 1, plan.starts.count)
        }

        for stem in 0..<configuration.numStems {
            for channel in 0..<channels {
                for index in 0..<workingCount {
                    accumulated[stem][channel][index] /= max(weights[index], 1e-8)
                }
            }
        }

        let stems = zip(configuration.stemNames, accumulated).map { name, channels in
            let cropped: [[Float]]
            if plan.usesBorderPadding {
                cropped = channels.map { channel in
                    Array(channel[plan.border..<(plan.border + frameCount)])
                }
            } else {
                cropped = channels.map { Array($0.prefix(frameCount)) }
            }
            return RoFormerStemResult(
                name: name,
                samples: RoFormerSeparator.interleave(cropped)
            )
        }
        let elapsed = started.duration(to: .now)
        return RoFormerSeparationResult(
            stems: stems,
            sampleRate: sampleRate,
            channels: channels,
            frameCount: frameCount,
            chunkCount: plan.starts.count,
            elapsedSeconds: Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        )
    }
}
