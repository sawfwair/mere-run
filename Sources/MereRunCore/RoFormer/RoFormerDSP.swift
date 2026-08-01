import Foundation
@preconcurrency import MLX

enum RoFormerDSP {
    static func periodicHannWindow(length: Int, dtype: DType = .float32) -> MLXArray {
        let positions = MLXArray((0..<length).map(Float.init)).asType(dtype)
        return MLXArray(0.5).asType(dtype)
            - MLXArray(0.5).asType(dtype) * MLX.cos(positions * (2 * Float.pi / Float(length)))
    }

    /// PyTorch-compatible centered STFT, arranged as `[batch, frequency * channel, frame, complex]`.
    static func stft(
        _ audio: MLXArray,
        nFFT: Int,
        hopLength: Int,
        window: MLXArray
    ) -> MLXArray {
        precondition(audio.ndim == 3)
        let batch = audio.dim(0)
        let channels = audio.dim(1)
        let sampleCount = audio.dim(2)
        let flattened = audio.reshaped(batch * channels, sampleCount)
        let padding = nFFT / 2
        let leftIndices = MLXArray(Array(stride(from: padding, through: 1, by: -1)))
        let rightIndices = MLXArray(Array(stride(
            from: sampleCount - 2,
            through: sampleCount - padding - 1,
            by: -1
        )))
        let padded = MLX.concatenated([
            MLX.take(flattened, leftIndices, axis: 1),
            flattened,
            MLX.take(flattened, rightIndices, axis: 1),
        ], axis: 1)
        let paddedCount = sampleCount + 2 * padding
        let frameCount = 1 + (paddedCount - nFFT) / hopLength
        let frames = MLX.asStrided(
            padded,
            [batch * channels, frameCount, nFFT],
            strides: [paddedCount, hopLength, 1]
        )
        let spectrum = MLX.rfft(frames * window, n: nFFT, axis: -1)
        let frequencies = nFFT / 2 + 1
        let arranged = spectrum
            .reshaped(batch, channels, frameCount, frequencies)
            .transposed(0, 3, 1, 2)
            .reshaped(batch, frequencies * channels, frameCount)
        return MLX.stacked([arranged.realPart(), arranged.imaginaryPart()], axis: -1)
    }

    /// PyTorch-compatible centered ISTFT from `[batch, stem, frequency * channel, frame, complex]`.
    static func istft(
        _ representation: MLXArray,
        channels: Int,
        length: Int,
        nFFT: Int,
        hopLength: Int,
        window: MLXArray,
        zeroDC: Bool
    ) -> MLXArray {
        precondition(representation.ndim == 5)
        let batch = representation.dim(0)
        let stems = representation.dim(1)
        let frequencyChannels = representation.dim(2)
        let frameCount = representation.dim(3)
        let frequencies = frequencyChannels / channels
        let real = representation[.ellipsis, 0]
        let imaginary = representation[.ellipsis, 1]
        var spectrum = real.asType(.complex64)
            + MLXArray(real: 0, imaginary: 1) * imaginary.asType(.complex64)
        spectrum = spectrum
            .reshaped(batch, stems, frequencies, channels, frameCount)
            .transposed(0, 1, 3, 4, 2)
            .reshaped(batch * stems * channels, frameCount, frequencies)
        if zeroDC {
            let frequencyMask = MLX.concatenated([
                MLX.zeros([1], dtype: real.dtype),
                MLX.ones([frequencies - 1], dtype: real.dtype),
            ]).reshaped(1, 1, frequencies)
            spectrum = spectrum * frequencyMask.asType(.complex64)
        }

        let frames = MLX.irfft(spectrum, n: nFFT, axis: -1) * window
        let paddedLength = nFFT + hopLength * (frameCount - 1)
        let positions = (0..<frameCount).flatMap { frame in
            let start = frame * hopLength
            return Array(start..<(start + nFFT))
        }
        let positionArray = MLXArray(positions)
        let flattenedFrames = frames.reshaped(batch * stems * channels, frameCount * nFFT)
        let overlapAdded = MLX.zeros(
            [batch * stems * channels, paddedLength],
            dtype: frames.dtype
        ).at[0..., positionArray].add(flattenedFrames)
        let windowSquared = window.square()
        let repeatedWindow = MLX.tiled(windowSquared, repetitions: [frameCount])
        let denominator = MLX.zeros([paddedLength], dtype: window.dtype)
            .at[positionArray].add(repeatedWindow)
        let normalized = overlapAdded / MLX.maximum(
            denominator,
            MLXArray(1e-11).asType(denominator.dtype)
        )
        let padding = nFFT / 2
        return normalized[0..., padding..<(padding + length)]
            .reshaped(batch, stems, channels, length)
    }
}
