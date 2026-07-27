import Foundation
import MLX

enum ACEStepAudioFormat: String, CaseIterable, Codable {
    case pcm16
    case pcm24
    case float32

    var bitsPerSample: UInt16 {
        switch self {
        case .pcm16:
            16
        case .pcm24:
            24
        case .float32:
            32
        }
    }

    var waveFormat: UInt16 {
        self == .float32 ? 3 : 1
    }
}

enum ACEStepNormalizationMode: String, CaseIterable, Codable {
    case none
    case peak
}

struct ACEStepAudioExportOptions: Codable {
    var format: ACEStepAudioFormat = .pcm24
    var normalization: ACEStepNormalizationMode = .peak
    var targetPeakDB: Float = -1
    var fadeInMilliseconds: Float = 5
    var fadeOutMilliseconds: Float = 20
    var dither: Bool = true
}

enum ACEStepWAVWriter {
    enum WriterError: LocalizedError {
        case invalidShape([Int])
        case invalidChannels(Int)
        case invalidSampleRate(Int)
        case invalidPeak(Float)

        var errorDescription: String? {
            switch self {
            case .invalidShape(let shape):
                return "Unsupported ACE-Step audio tensor shape: \(shape). "
                    + "Expected [1,S,C], [S,C], [1,S], or [S]."
            case .invalidChannels(let channels):
                return "Invalid channel count \(channels). Expected 1...8."
            case .invalidSampleRate(let sampleRate):
                return "Invalid sample rate \(sampleRate)."
            case .invalidPeak(let peak):
                return "Target peak must be at most 0 dBFS; got \(peak)."
            }
        }
    }

    static func writeWAV(
        _ audio: MLXArray,
        to url: URL,
        sampleRate: Int,
        options: ACEStepAudioExportOptions = .init()
    ) throws {
        try wavData(audio, sampleRate: sampleRate, options: options).write(
            to: url,
            options: .atomic
        )
    }

    static func wavData(
        _ audio: MLXArray,
        sampleRate: Int,
        options: ACEStepAudioExportOptions = .init()
    ) throws -> Data {
        guard sampleRate > 0 else {
            throw WriterError.invalidSampleRate(sampleRate)
        }
        guard options.targetPeakDB <= 0 else {
            throw WriterError.invalidPeak(options.targetPeakDB)
        }
        let (interleaved, channels) = try flattenToInterleaved(audio)
        guard (1...8).contains(channels) else {
            throw WriterError.invalidChannels(channels)
        }

        var samples = interleaved.map { $0.isFinite ? $0 : 0 }
        applyFades(
            to: &samples,
            channels: channels,
            sampleRate: sampleRate,
            fadeInMilliseconds: options.fadeInMilliseconds,
            fadeOutMilliseconds: options.fadeOutMilliseconds
        )
        normalize(
            &samples,
            mode: options.normalization,
            targetPeakDB: options.targetPeakDB
        )

        let bytesPerSample = Int(options.format.bitsPerSample / 8)
        let dataSize = UInt32(samples.count * bytesPerSample)
        let blockAlign = UInt16(channels * bytesPerSample)
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)

        var data = Data()
        data.append(Data("RIFF".utf8))
        append(UInt32(36) + dataSize, to: &data)
        data.append(Data("WAVE".utf8))
        data.append(Data("fmt ".utf8))
        append(UInt32(16), to: &data)
        append(options.format.waveFormat, to: &data)
        append(UInt16(channels), to: &data)
        append(UInt32(sampleRate), to: &data)
        append(byteRate, to: &data)
        append(blockAlign, to: &data)
        append(options.format.bitsPerSample, to: &data)
        data.append(Data("data".utf8))
        append(dataSize, to: &data)

        for (index, sample) in samples.enumerated() {
            let clamped = max(-1, min(1, sample))
            switch options.format {
            case .pcm16:
                let dither = options.dither ? triangularDither(index: index) / 32_768 : 0
                append(Int16(max(-1, min(1, clamped + dither)) * 32_767), to: &data)
            case .pcm24:
                let dither = options.dither ? triangularDither(index: index) / 8_388_608 : 0
                let value = Int32(max(-1, min(1, clamped + dither)) * 8_388_607)
                data.append(UInt8(truncatingIfNeeded: value))
                data.append(UInt8(truncatingIfNeeded: value >> 8))
                data.append(UInt8(truncatingIfNeeded: value >> 16))
            case .float32:
                appendFloat32(clamped, to: &data)
            }
        }
        return data
    }

    static func flattenToInterleaved(_ audio: MLXArray) throws -> ([Float], Int) {
        let sampleChannel: MLXArray
        if audio.ndim == 1 {
            sampleChannel = audio.reshaped(audio.dim(0), 1)
        } else if audio.ndim == 2 {
            if audio.dim(1) <= 8 {
                sampleChannel = audio
            } else if audio.dim(0) <= 8 {
                sampleChannel = audio.transposed(1, 0)
            } else {
                throw WriterError.invalidShape(audio.shape)
            }
        } else if audio.ndim == 3 {
            guard audio.dim(0) == 1 else {
                throw WriterError.invalidShape(audio.shape)
            }
            let squeezed = audio[0, 0..., 0...]
            if squeezed.dim(1) <= 8 {
                sampleChannel = squeezed
            } else if squeezed.dim(0) <= 8 {
                sampleChannel = squeezed.transposed(1, 0)
            } else {
                throw WriterError.invalidShape(audio.shape)
            }
        } else {
            throw WriterError.invalidShape(audio.shape)
        }

        MLX.eval(sampleChannel)
        return (
            sampleChannel.asType(.float32).reshaped(-1).asArray(Float.self),
            sampleChannel.dim(1)
        )
    }

    private static func normalize(
        _ samples: inout [Float],
        mode: ACEStepNormalizationMode,
        targetPeakDB: Float
    ) {
        guard mode == .peak else {
            return
        }
        let peak = samples.reduce(Float.zero) { max($0, abs($1)) }
        guard peak > 0 else {
            return
        }
        let target = pow(10, targetPeakDB / 20)
        let gain = target / peak
        for index in samples.indices {
            samples[index] *= gain
        }
    }

    private static func applyFades(
        to samples: inout [Float],
        channels: Int,
        sampleRate: Int,
        fadeInMilliseconds: Float,
        fadeOutMilliseconds: Float
    ) {
        let frameCount = samples.count / channels
        guard frameCount > 0 else {
            return
        }
        let fadeInFrames = min(
            frameCount,
            max(0, Int(Float(sampleRate) * fadeInMilliseconds / 1_000))
        )
        let fadeOutFrames = min(
            frameCount,
            max(0, Int(Float(sampleRate) * fadeOutMilliseconds / 1_000))
        )
        for frame in 0..<fadeInFrames {
            let gain = Float(frame) / Float(max(fadeInFrames - 1, 1))
            for channel in 0..<channels {
                samples[frame * channels + channel] *= gain
            }
        }
        for offset in 0..<fadeOutFrames {
            let frame = frameCount - fadeOutFrames + offset
            let gain = Float(fadeOutFrames - offset - 1)
                / Float(max(fadeOutFrames - 1, 1))
            for channel in 0..<channels {
                samples[frame * channels + channel] *= gain
            }
        }
    }

    private static func triangularDither(index: Int) -> Float {
        let first = pseudoRandom(UInt64(index) &* 2)
        let second = pseudoRandom(UInt64(index) &* 2 &+ 1)
        return first - second
    }

    private static func pseudoRandom(_ seed: UInt64) -> Float {
        var value = seed &+ 0x9E37_79B9_7F4A_7C15
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        return Float(value & 0x00FF_FFFF) / Float(0x0100_0000)
    }

    private static func append<T: FixedWidthInteger>(
        _ value: T,
        to data: inout Data
    ) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            data.append(contentsOf: bytes)
        }
    }

    private static func appendFloat32(_ value: Float, to data: inout Data) {
        append(value.bitPattern, to: &data)
    }
}
