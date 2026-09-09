// Adapted from mlx-audio-swift at commit 4266f988d170a83017d1e82e2e4654602f277f1d.
// Copyright (c) 2025 Prince Canuma. Licensed under the MIT License.
import Foundation
import MLX
import MLXNN

// MARK: - FastConformer Encoder Components

extension SortformerModel {
    // MARK: - Offline Inference

    public func generate(
        audio: MLXArray,
        sampleRate: Int = 16000,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0
    ) throws -> DiarizationOutput {
        let startTime = Date()
        let processor = config.processorConfig
        guard sampleRate == processor.samplingRate else {
            throw SortformerDiarizationError.unsupportedSampleRate(
                actual: sampleRate,
                expected: processor.samplingRate
            )
        }
        guard audio.size > 0 else {
            throw SortformerDiarizationError.emptyAudio
        }

        var waveform = audio.asType(.float32)
        if waveform.ndim > 1 {
            waveform = MLX.mean(waveform, axis: -1)
        }

        let (trimmed, trimOffset) = trimSilence(waveform, sampleRate: processor.samplingRate)
        waveform = trimmed
        let trimOffsetSeconds = Float(trimOffset) / Float(processor.samplingRate)
        waveform = (1.0 / (MLX.abs(waveform).max() + 1e-3)) * waveform

        let features = extractMelFeatures(
            waveform,
            sampleRate: processor.samplingRate,
            nFft: processor.nFft,
            hopLength: processor.hopLength,
            winLength: processor.winLength,
            nMels: processor.featureSize,
            preemphasisCoeff: processor.preemphasis
        )
        let featureLengths = MLXArray([Int32(features.dim(2))])
        let predictions = self(features, audioSignalLength: featureLengths)
        eval(predictions)

        let subsamplingFactor = config.fcEncoderConfig.subsamplingFactor
        let frameDuration = Float(processor.hopLength * subsamplingFactor) / Float(processor.samplingRate)
        var segments = Self.predsToSegments(
            predictions[0],
            frameDuration: frameDuration,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap
        )

        if trimOffset > 0 {
            segments = segments.map {
                DiarizationSegment(
                    start: $0.start + trimOffsetSeconds,
                    end: $0.end + trimOffsetSeconds,
                    speaker: $0.speaker
                )
            }
        }

        return DiarizationOutput(
            segments: segments,
            numSpeakers: Set(segments.map(\.speaker)).count,
            totalTime: Date().timeIntervalSince(startTime)
        )
    }
    // MARK: - Postprocessing

    public static func predsToSegments(
        _ preds: MLXArray,
        frameDuration: Float,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0
    ) -> [DiarizationSegment] {
        let numFrames = preds.dim(0)
        let numSpeakers = preds.dim(1)
        var segments = [DiarizationSegment]()

        // Single bulk GPU->CPU readback (row-major [frame, speaker]); all change
        // detection runs in pure Swift to avoid per-frame .item() round-trips.
        let flat = preds.asType(.float32).reshaped([-1]).asArray(Float.self)

        for spk in 0..<numSpeakers {
            var spkSegments = [DiarizationSegment]()
            var segStart = -1
            for f in 0..<numFrames {
                let active = flat[f * numSpeakers + spk] > threshold
                if active {
                    if segStart < 0 { segStart = f }
                } else if segStart >= 0 {
                    let startTime = Float(segStart) * frameDuration
                    let endTime = Float(f) * frameDuration
                    if endTime - startTime >= minDuration {
                        spkSegments.append(DiarizationSegment(start: startTime, end: endTime, speaker: spk))
                    }
                    segStart = -1
                }
            }
            if segStart >= 0 {
                let startTime = Float(segStart) * frameDuration
                let endTime = Float(numFrames) * frameDuration
                if endTime - startTime >= minDuration {
                    spkSegments.append(DiarizationSegment(start: startTime, end: endTime, speaker: spk))
                }
            }

            if mergeGap > 0 && spkSegments.count > 1 {
                var merged = [spkSegments[0]]
                for seg in spkSegments.dropFirst() {
                    if seg.start - merged.last!.end <= mergeGap {
                        merged[merged.count - 1] = DiarizationSegment(
                            start: merged.last!.start, end: seg.end, speaker: seg.speaker
                        )
                    } else {
                        merged.append(seg)
                    }
                }
                spkSegments = merged
            }

            segments.append(contentsOf: spkSegments)
        }

        segments.sort { $0.start < $1.start }
        return segments
    }
}
