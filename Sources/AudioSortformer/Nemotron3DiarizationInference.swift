import Foundation
import MLX

/// Native NeMo feature extraction and cache-aware inference for the pinned
/// eight-speaker checkpoint. All model computation runs in MLX.
/// Streaming reference: NVIDIA/NeMo cf724ac337d1ebc7d0dda1e23fb80916f52927a5.
public final class Nemotron3DiarizationRuntime {
    private let model: Nemotron3DiarizationModel
    private let window: MLXArray
    private let filterbank: MLXArray

    public init(model: Nemotron3DiarizationModel, window: MLXArray, filterbank: MLXArray) {
        self.model = model
        self.window = window.asType(.float32)
        self.filterbank = filterbank.asType(.float32).reshaped(128, 257).transposed(1, 0)
    }

    public func features(samples: [Float]) -> MLXArray {
        let waveform = preemphasisFilter(MLXArray(samples))
        let centeredWindow = MLX.concatenated([
            MLXArray.zeros([56]), window, MLXArray.zeros([56]),
        ])
        let spectrum = stft(
            audio: waveform,
            window: centeredWindow,
            nFft: 512,
            hopLength: 160,
            padMode: .constant
        )
        let mel = MLX.matmul(MLX.abs(spectrum).square(), filterbank)
        return MLX.log(mel + Float(pow(2.0, -24))).transposed(1, 0).expandedDimensions(axis: 0)
    }

    public func diarize(
        samples: [Float],
        sampleRate: Int = 16_000,
        threshold: Float = 0.5,
        minDuration: Float = 0.25,
        mergeGap: Float = 0.25,
        chunkLength: Int = 340,
        rightContext: Int = 40,
        fifoLength: Int = 40,
        speakerCacheLength: Int = 264,
        cacheUpdatePeriod: Int = 300
    ) throws -> DiarizationOutput {
        guard sampleRate == 16_000 else {
            throw SortformerDiarizationError.unsupportedSampleRate(actual: sampleRate, expected: 16_000)
        }
        guard !samples.isEmpty else { throw SortformerDiarizationError.emptyAudio }
        guard chunkLength > 0, rightContext >= 0, fifoLength >= 0,
              speakerCacheLength >= 16, speakerCacheLength.isMultiple(of: 8),
              cacheUpdatePeriod > 0 else {
            throw SortformerDiarizationError.invalidConfiguration
        }

        let started = Date()
        let mel = features(samples: samples)
        eval(mel)
        let featureFrames = mel.dim(2)
        var cache = MLXArray.zeros([1, 0, 512])
        var cachePredictions = MLXArray.zeros([1, 0, 8])
        var fifo = MLXArray.zeros([1, 0, 512])
        var fifoPredictions = MLXArray.zeros([1, 0, 8])
        var cacheCompressed = false
        var chunks: [MLXArray] = []
        var cursor = 0

        while cursor < featureFrames {
            let currentEnd = min(cursor + chunkLength * 8, featureFrames)
            let contextEnd = min(currentEnd + rightContext * 8, featureFrames)
            let newFeatures = mel[0..., 0..., cursor..<contextEnd]
            let embeddings = model.preEncode(newFeatures)
            let currentFrames = (currentEnd - cursor + 7) / 8
            let context = MLX.concatenated([cache, fifo, embeddings], axis: 1)
            let probabilities = model.probabilities(preEncoded: context)
            eval(probabilities, embeddings)

            let prefixFrames = cache.dim(1) + fifo.dim(1)
            let highResolutionStart = prefixFrames * 8
            let validFeatureFrames = currentEnd - cursor
            chunks.append(probabilities[0..., highResolutionStart..<(highResolutionStart + validFeatureFrames), 0...])

            let coarse = Self.downsample(probabilities)
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
                    (cache, cachePredictions) = compressCache(
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
            eval(cache, cachePredictions, fifo, fifoPredictions, chunks[chunks.count - 1])
            cursor = currentEnd
        }

        let predictions = MLX.concatenated(chunks, axis: 1)[0, 0..<featureFrames, 0...]
        var segments = SortformerModel.predsToSegments(
            predictions,
            frameDuration: 0.01,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap
        )
        let duration = Float(samples.count) / 16_000
        segments = segments.compactMap { segment in
            let end = min(segment.end, duration)
            return end > segment.start
                ? DiarizationSegment(start: segment.start, end: end, speaker: segment.speaker)
                : nil
        }
        return DiarizationOutput(
            segments: segments,
            numSpeakers: Set(segments.map(\.speaker)).count,
            totalTime: Date().timeIntervalSince(started)
        )
    }

    private static func downsample(_ probabilities: MLXArray) -> MLXArray {
        let frames = probabilities.dim(1) / 8
        return MLX.mean(probabilities.reshaped(1, frames, 8, 8), axis: 2)
    }

    private func compressCache(
        embeddings: MLXArray,
        predictions: MLXArray,
        length: Int
    ) -> (MLXArray, MLXArray) {
        let frameCount = embeddings.dim(1)
        let probabilities = predictions.asType(.float32).reshaped([-1]).asArray(Float.self)
        let vectors = embeddings.asType(.float32).reshaped([-1]).asArray(Float.self)
        let silence = model.silenceEmbedding.asType(.float32).asArray(Float.self)
        let speakerSlots = length / 8 - 1
        let minimumPositive = Int(floor(Float(speakerSlots) * 0.5))
        let strongCount = Int(floor(Float(speakerSlots) * 0.75))
        let weakCount = Int(floor(Float(speakerSlots) * 1.5))
        var scores = [[Float]](repeating: [Float](repeating: -.infinity, count: frameCount + 1), count: 8)

        for frame in 0..<frameCount {
            var inactiveSum: Float = 0
            for speaker in 0..<8 {
                let probability = probabilities[frame * 8 + speaker]
                inactiveSum += log(max(1 - probability, 0.25))
            }
            for speaker in 0..<8 {
                let probability = probabilities[frame * 8 + speaker]
                guard probability > 0.5 else { continue }
                scores[speaker][frame] = log(max(probability, 0.25))
                    - log(max(1 - probability, 0.25))
                    + inactiveSum - log(Float(0.5))
            }
        }

        for speaker in 0..<8 {
            let positive = scores[speaker].filter { $0 > 0 }.count
            if positive >= minimumPositive {
                for frame in 0..<frameCount where scores[speaker][frame] <= 0 {
                    scores[speaker][frame] = -.infinity
                }
            }
            for frame in length..<frameCount where scores[speaker][frame].isFinite {
                scores[speaker][frame] += 0.05
            }
            for (count, boost) in [(strongCount, Float(2 * log(2.0))), (weakCount, Float(log(2.0)))] {
                let ranked = (0..<frameCount).sorted { scores[speaker][$0] > scores[speaker][$1] }
                for frame in ranked.prefix(count) where scores[speaker][frame].isFinite {
                    scores[speaker][frame] += boost
                }
            }
            scores[speaker][frameCount] = .infinity
        }

        let width = frameCount + 1
        let selected = (0..<(width * 8))
            .sorted { scores[$0 / width][$0 % width] > scores[$1 / width][$1 % width] }
            .prefix(length)
            .sorted()
        var selectedVectors = [Float]()
        var selectedPredictions = [Float]()
        selectedVectors.reserveCapacity(length * 512)
        selectedPredictions.reserveCapacity(length * 8)
        for index in selected {
            let frame = index % width
            if frame == frameCount || !scores[index / width][frame].isFinite {
                selectedVectors.append(contentsOf: silence)
                selectedPredictions.append(contentsOf: repeatElement(Float(0), count: 8))
            } else {
                selectedVectors.append(contentsOf: vectors[(frame * 512)..<((frame + 1) * 512)])
                selectedPredictions.append(contentsOf: probabilities[(frame * 8)..<((frame + 1) * 8)])
            }
        }
        return (
            MLXArray(selectedVectors).reshaped(1, length, 512),
            MLXArray(selectedPredictions).reshaped(1, length, 8)
        )
    }
}
