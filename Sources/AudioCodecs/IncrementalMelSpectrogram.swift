import Foundation
import MLX

/// Maintains streaming PCM audio and exposes mel extraction over the buffered signal.
public struct IncrementalMelSpectrogram: Sendable {
    public let sampleRate: Int
    private var bufferedSamples: [Float]
    private var stableLogMelFrames: [[Float]] = []

    public init(sampleRate: Int = 16_000, initialCapacity: Int = 0) {
        self.sampleRate = max(1, sampleRate)
        if initialCapacity > 0 {
            self.bufferedSamples = []
            self.bufferedSamples.reserveCapacity(initialCapacity)
        } else {
            self.bufferedSamples = []
        }
    }

    public var sampleCount: Int {
        bufferedSamples.count
    }

    public var totalAudioSeconds: Double {
        Double(bufferedSamples.count) / Double(sampleRate)
    }

    public mutating func append(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        bufferedSamples.append(contentsOf: samples)
    }

    public mutating func removeAll(keepingCapacity: Bool = true) {
        bufferedSamples.removeAll(keepingCapacity: keepingCapacity)
        stableLogMelFrames.removeAll(keepingCapacity: keepingCapacity)
    }

    public func snapshotSamples() -> [Float] {
        bufferedSamples
    }

    public func snapshotSamples(count: Int) -> [Float] {
        Array(bufferedSamples.prefix(max(0, min(count, bufferedSamples.count))))
    }

    public mutating func extract(
        using extractor: MelSpectrogram,
        sampleCount requestedSampleCount: Int? = nil
    ) -> MLXArray {
        let requestedSampleCount = requestedSampleCount ?? bufferedSamples.count
        precondition(requestedSampleCount >= 0 && requestedSampleCount <= bufferedSamples.count)

        let stableFrameCount = extractor.stableFrameCount(sampleCount: requestedSampleCount)
        if stableLogMelFrames.count < stableFrameCount {
            for frameIndex in stableLogMelFrames.count..<stableFrameCount {
                stableLogMelFrames.append(extractor.logMelFrame(
                    from: bufferedSamples,
                    sampleCount: requestedSampleCount,
                    frameIndex: frameIndex
                ))
            }
        }

        let outputFrameCount = extractor.outputFrameCount(sampleCount: requestedSampleCount)
        var frames = Array(stableLogMelFrames.prefix(min(stableFrameCount, outputFrameCount)))
        if frames.count < outputFrameCount {
            for frameIndex in frames.count..<outputFrameCount {
                frames.append(extractor.logMelFrame(
                    from: bufferedSamples,
                    sampleCount: requestedSampleCount,
                    frameIndex: frameIndex
                ))
            }
        }

        let maximum = frames.lazy.flatMap { $0 }.max() ?? -10
        let clampFloor = maximum - 8
        var flattened = [Float](repeating: 0, count: extractor.nMels * outputFrameCount)
        for frameIndex in 0..<outputFrameCount {
            for melIndex in 0..<extractor.nMels {
                let value = max(frames[frameIndex][melIndex], clampFloor)
                flattened[(melIndex * outputFrameCount) + frameIndex] = (value + 4) / 4
            }
        }
        return MLXArray(flattened).reshaped(1, extractor.nMels, outputFrameCount)
    }
}
