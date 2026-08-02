import Foundation
@preconcurrency import MLX

public struct RoFormerSeparationResult: Sendable {
    public let vocals: [Float]
    public let instrumental: [Float]
    public let sampleRate: Int
    public let channels: Int
    public let frameCount: Int
    public let chunkCount: Int
    public let elapsedSeconds: Double
}

struct RoFormerChunkPlan: Equatable, Sendable {
    let starts: [Int]
    let step: Int
    let border: Int
    let fadeSize: Int
    let usesBorderPadding: Bool

    static func make(sampleCount: Int, chunkSize: Int, overlap: Int) throws -> Self {
        guard overlap > 0, chunkSize.isMultiple(of: overlap) else {
            throw RoFormerError.invalidOverlap(overlap)
        }
        let step = chunkSize / overlap
        let border = chunkSize - step
        let usesBorderPadding = sampleCount > 2 * border
        let workingCount = sampleCount + (usesBorderPadding ? 2 * border : 0)
        return Self(
            starts: Array(stride(from: 0, to: max(workingCount, 1), by: step)),
            step: step,
            border: border,
            fadeSize: chunkSize / 10,
            usesBorderPadding: usesBorderPadding
        )
    }

    func fadeWindow(chunkSize: Int, chunkIndex: Int) -> [Float] {
        var window = [Float](repeating: 1, count: chunkSize)
        guard fadeSize > 0 else { return window }
        if chunkIndex > 0 {
            for index in 0..<fadeSize {
                window[index] = Float(index) / Float(fadeSize)
            }
        }
        if chunkIndex < starts.count - 1 {
            for index in 0..<fadeSize {
                window[chunkSize - fadeSize + index] = Float(fadeSize - index) / Float(fadeSize)
            }
        }
        return window
    }
}

public final class RoFormerSeparator {
    public typealias ProgressHandler = @Sendable (_ completedChunks: Int, _ totalChunks: Int) -> Void

    public let checkpoint: RoFormerCheckpoint
    public let dtype: DType
    private let model: BSRoFormer

    private init(checkpoint: RoFormerCheckpoint, dtype: DType, model: BSRoFormer) {
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.model = model
    }

    public static func load(
        resources: RoFormerResources,
        dtype: DType = .float16
    ) throws -> RoFormerSeparator {
        let checkpoint = try resources.resolve()
        let model = try BSRoFormer.load(checkpoint: checkpoint, dtype: dtype)
        return RoFormerSeparator(checkpoint: checkpoint, dtype: dtype, model: model)
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
        let original = Self.deinterleave(interleavedSamples, channels: channels)
        let working = plan.usesBorderPadding
            ? original.map { Self.reflectPad($0, amount: plan.border) }
            : original
        let workingCount = working[0].count
        var accumulated = Array(
            repeating: [Float](repeating: 0, count: workingCount),
            count: channels
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
            let output = model(input)[0, 0].asType(.float32)
            MLX.eval(output)
            let separated = output.asArray(Float.self)
            let fade = plan.fadeWindow(chunkSize: configuration.chunkSize, chunkIndex: chunkIndex)
            for offset in 0..<available {
                let destination = start + offset
                let weight = fade[offset]
                weights[destination] += weight
                for channel in 0..<channels {
                    let value = separated[channel * configuration.chunkSize + offset]
                    accumulated[channel][destination] += (value.isFinite ? value : 0) * weight
                }
            }
            MLX.Memory.clearCache()
            progress?(chunkIndex + 1, plan.starts.count)
        }

        for channel in 0..<channels {
            for index in 0..<workingCount {
                accumulated[channel][index] /= max(weights[index], 1e-8)
            }
        }
        let vocalsByChannel: [[Float]]
        if plan.usesBorderPadding {
            vocalsByChannel = accumulated.map { channel in
                Array(channel[plan.border..<(plan.border + frameCount)])
            }
        } else {
            vocalsByChannel = accumulated.map { Array($0.prefix(frameCount)) }
        }
        let vocals = Self.interleave(vocalsByChannel)
        let instrumental = zip(interleavedSamples, vocals).map { mixture, vocal in
            mixture - vocal
        }
        let elapsed = started.duration(to: .now)
        return RoFormerSeparationResult(
            vocals: vocals,
            instrumental: instrumental,
            sampleRate: sampleRate,
            channels: channels,
            frameCount: frameCount,
            chunkCount: plan.starts.count,
            elapsedSeconds: Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        )
    }

    static func deinterleave(_ samples: [Float], channels: Int) -> [[Float]] {
        let frames = samples.count / channels
        var result = Array(repeating: [Float](repeating: 0, count: frames), count: channels)
        for frame in 0..<frames {
            for channel in 0..<channels {
                result[channel][frame] = samples[frame * channels + channel]
            }
        }
        return result
    }

    static func interleave(_ channels: [[Float]]) -> [Float] {
        guard let frames = channels.first?.count else { return [] }
        var result = [Float](repeating: 0, count: frames * channels.count)
        for frame in 0..<frames {
            for channel in channels.indices {
                result[frame * channels.count + channel] = channels[channel][frame]
            }
        }
        return result
    }

    static func reflectPad(_ samples: [Float], amount: Int) -> [Float] {
        precondition(amount >= 0 && amount < samples.count)
        let left = stride(from: amount, through: 1, by: -1).map { samples[$0] }
        let right = stride(
            from: samples.count - 2,
            through: samples.count - amount - 1,
            by: -1
        ).map { samples[$0] }
        return left + samples + right
    }
}
