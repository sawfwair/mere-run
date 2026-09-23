import Foundation
import MLX

/// A nonoverlapping portion of the speaker timeline emitted as soon as its
/// configured right context has arrived. Segment boundaries are local to this
/// chunk; consumers may join adjacent same-speaker segments for display.
public struct Nemotron3DiarizationStreamChunk: Equatable, Sendable {
    public let startFrame: Int
    public let endFrame: Int
    public let segments: [DiarizationSegment]

    public var startSeconds: Float { Float(startFrame) * 0.01 }
    public var endSeconds: Float { Float(endFrame) * 0.01 }
}

/// Incremental 16 kHz mono inference. Audio, the AOSC speaker cache, and FIFO
/// context remain in this session across calls to `feed`.
public final class Nemotron3DiarizationStreamingSession {
    private let runtime: Nemotron3DiarizationRuntime
    private let chunkLength: Int
    private let rightContext: Int
    private let fifoLength: Int
    private let speakerCacheLength: Int
    private let cacheUpdatePeriod: Int
    private let threshold: Float

    private var audio = [Float]()
    private var audioStartSample = 0
    private var receivedSamples = 0
    private var cursor = 0
    private var closed = false

    private var cache = MLXArray.zeros([1, 0, 512])
    private var cachePredictions = MLXArray.zeros([1, 0, 8])
    private var fifo = MLXArray.zeros([1, 0, 512])
    private var fifoPredictions = MLXArray.zeros([1, 0, 8])
    private var cacheCompressed = false

    public var bufferedSampleCount: Int { audio.count }
    public var processedFrameCount: Int { cursor }

    public init(
        runtime: Nemotron3DiarizationRuntime,
        threshold: Float = 0.5,
        chunkLength: Int = 9,
        rightContext: Int = 4,
        fifoLength: Int = 264,
        speakerCacheLength: Int = 264,
        cacheUpdatePeriod: Int = 222
    ) throws {
        guard threshold.isFinite, (0...1).contains(threshold),
              chunkLength > 0, rightContext >= 0, fifoLength >= 0,
              speakerCacheLength >= 16, speakerCacheLength.isMultiple(of: 8),
              cacheUpdatePeriod > 0 else {
            throw SortformerDiarizationError.invalidConfiguration
        }
        self.runtime = runtime
        self.threshold = threshold
        self.chunkLength = chunkLength
        self.rightContext = rightContext
        self.fifoLength = fifoLength
        self.speakerCacheLength = speakerCacheLength
        self.cacheUpdatePeriod = cacheUpdatePeriod
    }

    public func feed(samples: [Float]) throws -> [Nemotron3DiarizationStreamChunk] {
        guard !closed else { throw SortformerDiarizationError.invalidConfiguration }
        audio.append(contentsOf: samples)
        receivedSamples += samples.count
        return processAvailable(final: false)
    }

    public func finish() throws -> [Nemotron3DiarizationStreamChunk] {
        guard !closed else { throw SortformerDiarizationError.invalidConfiguration }
        closed = true
        guard receivedSamples > 0 else { throw SortformerDiarizationError.emptyAudio }
        return processAvailable(final: true)
    }

    private func processAvailable(final: Bool) -> [Nemotron3DiarizationStreamChunk] {
        let totalFrames = receivedSamples / 160 + 1
        var emitted = [Nemotron3DiarizationStreamChunk]()
        while cursor < totalFrames {
            let currentEnd = min(cursor + chunkLength * 8, totalFrames)
            let contextEnd = min(currentEnd + rightContext * 8, totalFrames)
            if !final {
                guard currentEnd - cursor == chunkLength * 8,
                      contextEnd - currentEnd == rightContext * 8,
                      receivedSamples >= (contextEnd - 1) * 160 + 200 else { break }
            }

            let localStartFrame = max(0, cursor - 2)
            let localStartSample = localStartFrame * 160
            let localEndSample = final
                ? receivedSamples
                : min(receivedSamples, (contextEnd - 1) * 160 + 200)
            let localAudio = Array(audio[(localStartSample - audioStartSample)..<(localEndSample - audioStartSample)])
            let mel = runtime.features(samples: localAudio)
            let fromFrame = cursor - localStartFrame
            let toFrame = contextEnd - localStartFrame
            let newFeatures = mel[0..., 0..., fromFrame..<toFrame]
            let embeddings = runtime.model.preEncode(newFeatures)
            let currentFrames = (currentEnd - cursor + 7) / 8
            let context = MLX.concatenated([cache, fifo, embeddings], axis: 1)
            let probabilities = runtime.model.probabilities(preEncoded: context)
            eval(probabilities, embeddings)

            let prefixFrames = cache.dim(1) + fifo.dim(1)
            let outputStart = prefixFrames * 8
            let validFrames = currentEnd - cursor
            let chunkPredictions = probabilities[0, outputStart..<(outputStart + validFrames), 0...]
            let localSegments = SortformerModel.predsToSegments(
                chunkPredictions, frameDuration: 0.01, threshold: threshold
            )
            let maximumEnd = Float(receivedSamples) / 16_000
            let offset = Float(cursor) * 0.01
            let segments = localSegments.compactMap { segment -> DiarizationSegment? in
                let end = min(segment.end + offset, maximumEnd)
                guard end > segment.start + offset else { return nil }
                return DiarizationSegment(
                    start: segment.start + offset, end: end, speaker: segment.speaker
                )
            }
            emitted.append(Nemotron3DiarizationStreamChunk(
                startFrame: cursor, endFrame: currentEnd, segments: segments
            ))

            let coarse = Nemotron3DiarizationRuntime.downsample(probabilities)
            let currentEmbeddings = embeddings[0..., 0..<currentFrames, 0...]
            let currentPredictions = coarse[0..., prefixFrames..<(prefixFrames + currentFrames), 0...]
            let updatedFIFO = MLX.concatenated([fifo, currentEmbeddings], axis: 1)
            let updatedFIFOPredictions = MLX.concatenated([
                coarse[0..., cache.dim(1)..<prefixFrames, 0...], currentPredictions,
            ], axis: 1)

            if updatedFIFO.dim(1) > fifoLength {
                let popCount = min(
                    updatedFIFO.dim(1),
                    max(cacheUpdatePeriod, updatedFIFO.dim(1) - fifoLength)
                )
                let popped = updatedFIFO[0..., 0..<popCount, 0...]
                let poppedPredictions = updatedFIFOPredictions[0..., 0..<popCount, 0...]
                let previousCachePredictions = cacheCompressed
                    ? cachePredictions
                    : coarse[0..., 0..<cache.dim(1), 0...]
                cache = MLX.concatenated([cache, popped], axis: 1)
                cachePredictions = MLX.concatenated([previousCachePredictions, poppedPredictions], axis: 1)
                fifo = updatedFIFO[0..., popCount..., 0...]
                fifoPredictions = updatedFIFOPredictions[0..., popCount..., 0...]
                if cache.dim(1) > speakerCacheLength {
                    (cache, cachePredictions) = runtime.compressCache(
                        embeddings: cache,
                        predictions: cachePredictions,
                        length: speakerCacheLength
                    )
                    cacheCompressed = true
                }
            } else {
                fifo = updatedFIFO
                fifoPredictions = updatedFIFOPredictions
            }
            eval(cache, cachePredictions, fifo, fifoPredictions)
            cursor = currentEnd
            let nextAudioStart = max(0, cursor - 2) * 160
            audio.removeFirst(nextAudioStart - audioStartSample)
            audioStartSample = nextAudioStart
        }
        return emitted
    }
}
