import Foundation
@preconcurrency import MLX

public struct APBWEEnhancementResult: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let channels: Int
    public let frameCount: Int
    public let chunkCount: Int
    public let elapsedSeconds: Double
}

public enum APBWEAudio {
    /// Band-limited 3x interpolation matching the pinned 16 kHz to 48 kHz profile.
    public static func upsample16kTo48k(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }
        let factor = 3
        let radius = 32
        var output = [Float](repeating: 0, count: samples.count * factor)
        for outputIndex in output.indices {
            let position = Double(outputIndex) / Double(factor)
            let center = Int(floor(position))
            var sum = 0.0
            var normalization = 0.0
            for sourceIndex in (center - radius + 1)...(center + radius) {
                let distance = position - Double(sourceIndex)
                guard abs(distance) < Double(radius) else { continue }
                let coefficient = sinc(distance) * sinc(distance / Double(radius))
                let reflected = reflectedIndex(sourceIndex, count: samples.count)
                sum += Double(samples[reflected]) * coefficient
                normalization += coefficient
            }
            output[outputIndex] = Float(sum / max(abs(normalization), 1e-12))
        }
        return output
    }

    static func reflectPad(_ samples: [Float], amount: Int) -> [Float] {
        precondition(amount >= 0)
        guard amount > 0, !samples.isEmpty else { return samples }
        return (0..<(samples.count + amount * 2)).map { outputIndex in
            samples[reflectedIndex(outputIndex - amount, count: samples.count)]
        }
    }

    private static func sinc(_ value: Double) -> Double {
        if abs(value) < 1e-12 { return 1 }
        let argument = Double.pi * value
        return sin(argument) / argument
    }

    private static func reflectedIndex(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        let upper = count - 1
        let period = upper * 2
        var value = index % period
        if value < 0 { value += period }
        return value > upper ? period - value : value
    }
}

public final class APBWEEnhancer {
    public typealias ProgressHandler = @Sendable (_ completedChunks: Int, _ totalChunks: Int) -> Void

    public let checkpoint: APBWECheckpoint
    public let dtype: DType
    private let model: APBWEModel

    private init(checkpoint: APBWECheckpoint, dtype: DType, model: APBWEModel) {
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.model = model
    }

    public static func load(
        resources: APBWEResources,
        dtype: DType = .float32
    ) throws -> APBWEEnhancer {
        let checkpoint = try resources.resolve()
        let model = try APBWEModel.load(checkpoint: checkpoint, dtype: dtype)
        return APBWEEnhancer(checkpoint: checkpoint, dtype: dtype, model: model)
    }

    public func enhance(
        narrowband48kSamples: [Float],
        sampleRate: Int,
        channels: Int,
        overlap: Int? = nil,
        progress: ProgressHandler? = nil
    ) throws -> APBWEEnhancementResult {
        let configuration = checkpoint.configuration
        guard sampleRate == configuration.highSampleRate, channels == 1 else {
            throw APBWEError.unsupportedAudio(sampleRate: sampleRate, channels: channels)
        }
        guard !narrowband48kSamples.isEmpty else {
            throw APBWEError.invalidAudioBuffer
        }
        let started = ContinuousClock.now
        let resolvedOverlap = overlap ?? configuration.overlap
        let plan = try RoFormerChunkPlan.make(
            sampleCount: narrowband48kSamples.count,
            chunkSize: configuration.chunkSize,
            overlap: resolvedOverlap
        )
        let working = plan.usesBorderPadding
            ? APBWEAudio.reflectPad(narrowband48kSamples, amount: plan.border)
            : narrowband48kSamples
        var accumulated = [Float](repeating: 0, count: working.count)
        var weights = [Float](repeating: 0, count: working.count)

        for (chunkIndex, start) in plan.starts.enumerated() {
            let available = min(configuration.chunkSize, working.count - start)
            var chunk = [Float](repeating: 0, count: configuration.chunkSize)
            for offset in 0..<available {
                chunk[offset] = working[start + offset]
            }
            if available > configuration.chunkSize / 2, available < configuration.chunkSize {
                for offset in available..<configuration.chunkSize {
                    chunk[offset] = working[start + 2 * available - offset - 2]
                }
            }
            let input = MLXArray(chunk)
                .reshaped(1, 1, configuration.chunkSize)
                .asType(dtype)
            let enhanced = model(input)
                .reshaped(configuration.chunkSize)
                .asType(.float32)
            MLX.eval(enhanced)
            let values = enhanced.asArray(Float.self)
            let fade = plan.fadeWindow(
                chunkSize: configuration.chunkSize,
                chunkIndex: chunkIndex
            )
            for offset in 0..<available {
                let destination = start + offset
                let weight = fade[offset]
                let value = values[offset]
                accumulated[destination] += (value.isFinite ? value : 0) * weight
                weights[destination] += weight
            }
            MLX.Memory.clearCache()
            progress?(chunkIndex + 1, plan.starts.count)
        }

        for index in accumulated.indices {
            accumulated[index] /= max(weights[index], 1e-8)
        }
        let output = plan.usesBorderPadding
            ? Array(accumulated[plan.border..<(plan.border + narrowband48kSamples.count)])
            : Array(accumulated.prefix(narrowband48kSamples.count))
        let elapsed = started.duration(to: .now)
        return APBWEEnhancementResult(
            samples: output,
            sampleRate: configuration.highSampleRate,
            channels: 1,
            frameCount: output.count,
            chunkCount: plan.starts.count,
            elapsedSeconds: Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        )
    }
}
