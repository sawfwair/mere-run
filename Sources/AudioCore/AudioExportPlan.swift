import Foundation

public enum AudioWAVEncoding: String, CaseIterable, Codable, Sendable {
    case pcm16, pcm24, float32
    public var bitsPerSample: UInt16 {
        switch self { case .pcm16: 16; case .pcm24: 24; case .float32: 32 }
    }
}

public enum AudioNormalizationMode: String, CaseIterable, Codable, Sendable {
    case none, peak
}

/// Serializable options. Construct a plan to validate them before inference.
/// Coding keys retain the existing music recipe representation.
public struct AudioExportOptions: Codable, Hashable, Sendable {
    public var format: AudioWAVEncoding
    public var normalization: AudioNormalizationMode
    public var targetPeakDB: Float
    public var fadeInMilliseconds: Float
    public var fadeOutMilliseconds: Float
    public var dither: Bool

    public init(
        format: AudioWAVEncoding = .pcm24, normalization: AudioNormalizationMode = .peak,
        targetPeakDB: Float = -1, fadeInMilliseconds: Float = 5,
        fadeOutMilliseconds: Float = 20, dither: Bool = true
    ) {
        self.format = format
        self.normalization = normalization
        self.targetPeakDB = targetPeakDB
        self.fadeInMilliseconds = fadeInMilliseconds
        self.fadeOutMilliseconds = fadeOutMilliseconds
        self.dither = dither
    }

    public static let music = AudioExportOptions()
    public static let referencePCM16 = AudioExportOptions(
        format: .pcm16, normalization: .none, targetPeakDB: 0,
        fadeInMilliseconds: 0, fadeOutMilliseconds: 0, dither: false
    )
}

/// Optional wire settings resolve against the entry point's explicit defaults.
public struct AudioExportOverrides: Codable, Sendable {
    public var format: AudioWAVEncoding?
    public var normalization: AudioNormalizationMode?
    public var targetPeakDB: Float?
    public var fadeInMilliseconds: Float?
    public var fadeOutMilliseconds: Float?
    public var dither: Bool?

    public init(
        format: AudioWAVEncoding? = nil, normalization: AudioNormalizationMode? = nil,
        targetPeakDB: Float? = nil, fadeInMilliseconds: Float? = nil,
        fadeOutMilliseconds: Float? = nil, dither: Bool? = nil
    ) {
        self.format = format
        self.normalization = normalization
        self.targetPeakDB = targetPeakDB
        self.fadeInMilliseconds = fadeInMilliseconds
        self.fadeOutMilliseconds = fadeOutMilliseconds
        self.dither = dither
    }

    enum CodingKeys: String, CodingKey {
        case format, normalization, dither
        case targetPeakDB = "target_peak_db"
        case fadeInMilliseconds = "fade_in_ms"
        case fadeOutMilliseconds = "fade_out_ms"
    }

    public func resolve(defaults: AudioExportOptions) throws -> AudioExportPlan {
        try AudioExportPlan(options: .init(
            format: format ?? defaults.format, normalization: normalization ?? defaults.normalization,
            targetPeakDB: targetPeakDB ?? defaults.targetPeakDB,
            fadeInMilliseconds: fadeInMilliseconds ?? defaults.fadeInMilliseconds,
            fadeOutMilliseconds: fadeOutMilliseconds ?? defaults.fadeOutMilliseconds,
            dither: dither ?? defaults.dither
        ))
    }
}

public enum AudioExportError: Error, LocalizedError, Sendable {
    case invalidChannels(Int), invalidSampleRate(Int), incompleteFrame(samples: Int, channels: Int)
    case invalidPeak(Float), invalidFade(Float), wavSizeLimit

    public var errorDescription: String? {
        switch self {
        case .invalidChannels(let channels): "Invalid channel count \(channels). Expected 1...8."
        case .invalidSampleRate(let rate): "Invalid sample rate \(rate)."
        case .incompleteFrame(let samples, let channels): "The \(samples) interleaved samples do not contain complete \(channels)-channel frames."
        case .invalidPeak(let peak): "Target peak must be finite and at most 0 dBFS; got \(peak)."
        case .invalidFade(let milliseconds): "Output fade duration must be finite and nonnegative; got \(milliseconds) ms."
        case .wavSizeLimit: "Audio exceeds the size or byte-rate limits of a RIFF WAV file."
        }
    }
}

public struct AudioExportPlan: Sendable, Hashable {
    public let options: AudioExportOptions

    public init(options: AudioExportOptions) throws {
        guard options.targetPeakDB.isFinite, options.targetPeakDB <= 0 else {
            throw AudioExportError.invalidPeak(options.targetPeakDB)
        }
        for milliseconds in [options.fadeInMilliseconds, options.fadeOutMilliseconds] {
            guard milliseconds.isFinite, milliseconds >= 0 else { throw AudioExportError.invalidFade(milliseconds) }
        }
        self.options = options
    }
}

/// Explicit, interleaved frame data. Tensor layout conversion belongs to model adapters.
public struct AudioWaveform: Sendable {
    public let samples: [Float]
    public let channels: Int
    public let sampleRate: Int
    public var frameCount: Int { samples.count / channels }
    public var duration: Double { Double(frameCount) / Double(sampleRate) }

    public init(interleaved samples: [Float], channels: Int, sampleRate: Int) throws {
        guard (1...8).contains(channels) else { throw AudioExportError.invalidChannels(channels) }
        guard sampleRate > 0, sampleRate <= Int(UInt32.max) else { throw AudioExportError.invalidSampleRate(sampleRate) }
        guard samples.count % channels == 0 else { throw AudioExportError.incompleteFrame(samples: samples.count, channels: channels) }
        self.samples = samples
        self.channels = channels
        self.sampleRate = sampleRate
    }

    /// Retains the music exporter's linear interpolation and frame-count rounding.
    public func resampled(to targetRate: Int) throws -> AudioWaveform {
        try Task.checkCancellation()
        guard targetRate > 0, targetRate <= Int(UInt32.max) else { throw AudioExportError.invalidSampleRate(targetRate) }
        if targetRate == sampleRate { return self }
        if frameCount == 0 { return try AudioWaveform(interleaved: [], channels: channels, sampleRate: targetRate) }
        let requestedCount = (Double(frameCount) / Double(sampleRate) * Double(targetRate)).rounded(.toNearestOrAwayFromZero)
        // Bound the allocation and conversion using the widest supported WAV encoding.
        let maximumFrames = (Int(UInt32.max) - 36) / (channels * 4)
        guard requestedCount <= Double(maximumFrames) else { throw AudioExportError.wavSizeLimit }
        let count = max(1, Int(requestedCount))
        if count == 1 {
            return try AudioWaveform(interleaved: Array(samples.prefix(channels)), channels: channels, sampleRate: targetRate)
        }
        var output = [Float](repeating: 0, count: count * channels)
        let ratio = Double(sampleRate) / Double(targetRate)
        for frame in 0..<count {
            if frame.isMultiple(of: 16_384) { try Task.checkCancellation() }
            let position = Double(frame) * ratio
            let lower = min(frameCount - 1, Int(floor(position)))
            let upper = min(frameCount - 1, lower + 1)
            let blend = Float(position - Double(lower))
            for channel in 0..<channels {
                let low = samples[lower * channels + channel]
                output[frame * channels + channel] = low + (samples[upper * channels + channel] - low) * blend
            }
        }
        return try AudioWaveform(interleaved: output, channels: channels, sampleRate: targetRate)
    }
}
