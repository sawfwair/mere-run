import Foundation
import MLX

/// Maintains streaming PCM audio and exposes mel extraction over the buffered signal.
public struct IncrementalMelSpectrogram: Sendable {
    public let sampleRate: Int
    private var bufferedSamples: [Float]

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
    }

    public func snapshotSamples() -> [Float] {
        bufferedSamples
    }

    public func extract(using extractor: MelSpectrogram) -> MLXArray {
        extractor.extract(from: bufferedSamples)
    }
}
