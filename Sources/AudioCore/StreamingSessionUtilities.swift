import Foundation

public struct StreamingDecodeCadence: Sendable, Hashable {
    public let minDecodeSamples: Int
    public let decodeIntervalSamples: Int
    public private(set) var lastDecodedSampleCount: Int

    public init(
        sampleRate: Int,
        decodeIntervalMs: Int,
        minDecodeAudioMs: Int,
        lastDecodedSampleCount: Int = 0
    ) {
        self.minDecodeSamples = max(0, sampleRate * max(0, minDecodeAudioMs) / 1_000)
        self.decodeIntervalSamples = max(1, sampleRate * max(1, decodeIntervalMs) / 1_000)
        self.lastDecodedSampleCount = max(0, lastDecodedSampleCount)
    }

    public func shouldDecode(bufferedSampleCount: Int, force: Bool = false) -> Bool {
        if force {
            return bufferedSampleCount > 0
        }
        guard bufferedSampleCount >= minDecodeSamples else {
            return false
        }
        return bufferedSampleCount - lastDecodedSampleCount >= decodeIntervalSamples
    }

    public mutating func markDecoded(sampleCount: Int) {
        lastDecodedSampleCount = max(0, sampleCount)
    }
}

public struct StreamingAudioTail: Sendable, Equatable {
    public let samples: [Float]
    public let updatedSampleCount: Int

    public init(samples: [Float], updatedSampleCount: Int) {
        self.samples = samples
        self.updatedSampleCount = updatedSampleCount
    }
}

public func streamingAudioNewTail(
    fullSamples: [Float],
    emittedSampleCount: Int
) -> StreamingAudioTail {
    let clampedStart = max(0, min(emittedSampleCount, fullSamples.count))
    let delta = Array(fullSamples[clampedStart..<fullSamples.count])
    return StreamingAudioTail(samples: delta, updatedSampleCount: fullSamples.count)
}
