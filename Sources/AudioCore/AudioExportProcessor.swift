import Foundation

public struct AudioExportStatistics: Codable, Sendable, Equatable {
    public let sampleCount: Int
    public let replacedNonfiniteSamples: Int
    public let clippedSamples: Int
    public let fadeInFrames: Int
    public let fadeOutFrames: Int
    public let normalizationGain: Double
    public let inputPeak: Float
    public let outputPeak: Float
    public let outputRMS: Double
}

/// Only the processor constructs this value, so the encoder receives finite, bounded samples.
public struct ProcessedAudio: Sendable {
    public let waveform: AudioWaveform
    public let plan: AudioExportPlan
    public let statistics: AudioExportStatistics
}

public enum AudioExportProcessor {
    public static func process(_ waveform: AudioWaveform, plan: AudioExportPlan) throws -> ProcessedAudio {
        try Task.checkCancellation()
        _ = try AudioWAVLayout(sampleCount: waveform.samples.count, channels: waveform.channels,
                               sampleRate: waveform.sampleRate, encoding: plan.options.format)
        let options = plan.options
        var replacements = 0
        var inputPeak: Float = 0
        var samples = try waveform.samples.enumerated().map { index, sample -> Float in
            if index.isMultiple(of: 16_384) { try Task.checkCancellation() }
            if !sample.isFinite { replacements += 1; return 0 }
            inputPeak = max(inputPeak, abs(sample))
            return sample
        }
        let fadeIn = fadeFrames(options.fadeInMilliseconds, waveform: waveform)
        let fadeOut = fadeFrames(options.fadeOutMilliseconds, waveform: waveform)
        for frame in 0..<fadeIn {
            if frame.isMultiple(of: 16_384) { try Task.checkCancellation() }
            let gain = Float(frame) / Float(max(fadeIn - 1, 1))
            for channel in 0..<waveform.channels { samples[frame * waveform.channels + channel] *= gain }
        }
        for offset in 0..<fadeOut {
            if offset.isMultiple(of: 16_384) { try Task.checkCancellation() }
            let frame = waveform.frameCount - fadeOut + offset
            let gain = Float(fadeOut - offset - 1) / Float(max(fadeOut - 1, 1))
            for channel in 0..<waveform.channels { samples[frame * waveform.channels + channel] *= gain }
        }
        let peak = samples.reduce(Float.zero) { max($0, abs($1)) }
        var normalizationGain = 1.0
        if options.normalization == .peak && peak > 0 {
            let target = pow(Float(10), options.targetPeakDB / 20)
            let gain = target / peak
            normalizationGain = Double(target) / Double(peak)
            for index in samples.indices {
                if index.isMultiple(of: 16_384) { try Task.checkCancellation() }
                // Preserve existing rounding for ordinary signals. Division first
                // keeps subnormal input signals finite when their Float gain overflows.
                samples[index] = gain.isFinite ? samples[index] * gain : (samples[index] / peak) * target
            }
        }
        var clipped = 0
        var outputPeak: Float = 0
        var sumSquares = 0.0
        for index in samples.indices {
                if index.isMultiple(of: 16_384) { try Task.checkCancellation() }
            if abs(samples[index]) > 1 { clipped += 1 }
            samples[index] = max(-1, min(1, samples[index]))
            outputPeak = max(outputPeak, abs(samples[index]))
            sumSquares += Double(samples[index]) * Double(samples[index])
        }
        return ProcessedAudio(
            waveform: try AudioWaveform(interleaved: samples, channels: waveform.channels, sampleRate: waveform.sampleRate),
            plan: plan,
            statistics: .init(sampleCount: samples.count, replacedNonfiniteSamples: replacements,
                              clippedSamples: clipped, fadeInFrames: fadeIn, fadeOutFrames: fadeOut,
                              normalizationGain: normalizationGain, inputPeak: inputPeak, outputPeak: outputPeak,
                              outputRMS: samples.isEmpty ? 0 : sqrt(sumSquares / Double(samples.count)))
        )
    }

    private static func fadeFrames(_ milliseconds: Float, waveform: AudioWaveform) -> Int {
        let requested = Float(waveform.sampleRate) * milliseconds / 1_000
        return requested < Float(waveform.frameCount) ? Int(requested) : waveform.frameCount
    }
}
