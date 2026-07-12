import Foundation
import MLX

enum Wan2DreamXColorStabilizer {
    static func process(_ frames: MLXArray, strength: Double = 0.3) -> MLXArray {
        precondition(frames.ndim == 5 && frames.dim(4) == 3)
        guard frames.dim(1) > 1, strength > 0 else { return frames }
        let batchCount = frames.dim(0)
        let frameCount = frames.dim(1)
        let height = frames.dim(2)
        let width = frames.dim(3)
        let pixelsPerFrame = height * width
        var values = frames.asArray(UInt8.self)
        let reference = statistics(values, offset: 0, pixelCount: pixelsPerFrame)

        for batch in 0..<batchCount {
            var target = reference
            for frame in 0..<frameCount {
                let frameOffset = (batch * frameCount + frame) * pixelsPerFrame * 3
                if frame == 9 {
                    target = statistics(values, offset: frameOffset - pixelsPerFrame * 3, pixelCount: pixelsPerFrame)
                } else if frame > 9, (frame - 9).isMultiple(of: 12) {
                    target = statistics(values, offset: frameOffset - pixelsPerFrame * 3, pixelCount: pixelsPerFrame)
                }
                stabilize(
                    &values,
                    offset: frameOffset,
                    pixelCount: pixelsPerFrame,
                    target: target,
                    strength: strength
                )
            }
        }
        return MLXArray(values).reshaped(frames.shape)
    }

    private struct Statistics {
        let mean: SIMD3<Double>
        let deviation: SIMD3<Double>
    }

    private static func statistics(
        _ values: [UInt8],
        offset: Int,
        pixelCount: Int
    ) -> Statistics {
        var sum = SIMD3<Double>(repeating: 0)
        var squared = SIMD3<Double>(repeating: 0)
        for pixel in 0..<pixelCount {
            let index = offset + pixel * 3
            let lab = rgbToLab(
                Double(values[index]) / 255,
                Double(values[index + 1]) / 255,
                Double(values[index + 2]) / 255
            )
            sum += lab
            squared += lab * lab
        }
        let count = Double(pixelCount)
        let mean = sum / count
        let variance = squared / count - mean * mean
        return Statistics(
            mean: mean,
            deviation: SIMD3(
                Foundation.sqrt(Swift.max(0, variance.x)),
                Foundation.sqrt(Swift.max(0, variance.y)),
                Foundation.sqrt(Swift.max(0, variance.z))
            )
        )
    }

    private static func stabilize(
        _ values: inout [UInt8],
        offset: Int,
        pixelCount: Int,
        target: Statistics,
        strength: Double
    ) {
        let source = statistics(values, offset: offset, pixelCount: pixelCount)
        for pixel in 0..<pixelCount {
            let index = offset + pixel * 3
            let original = SIMD3(
                Double(values[index]) / 255,
                Double(values[index + 1]) / 255,
                Double(values[index + 2]) / 255
            )
            var correctedLab = rgbToLab(original.x, original.y, original.z)
            for channel in 0..<3 {
                if source.deviation[channel] > 1e-6 {
                    correctedLab[channel] = (correctedLab[channel] - source.mean[channel])
                        * (target.deviation[channel] / source.deviation[channel])
                        + target.mean[channel]
                } else {
                    correctedLab[channel] = target.mean[channel]
                }
            }
            let corrected = labToRGB(correctedLab)
            let blended = (1 - strength) * original + strength * corrected
            values[index] = channel(blended.x)
            values[index + 1] = channel(blended.y)
            values[index + 2] = channel(blended.z)
        }
    }

    private static func rgbToLab(_ red: Double, _ green: Double, _ blue: Double) -> SIMD3<Double> {
        let r = linear(red)
        let g = linear(green)
        let b = linear(blue)
        let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
        let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        let z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
        let fx = labCurve(x)
        let fy = labCurve(y)
        let fz = labCurve(z)
        return SIMD3(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }

    private static func labToRGB(_ lab: SIMD3<Double>) -> SIMD3<Double> {
        let fy = (lab.x + 16) / 116
        let fx = fy + lab.y / 500
        let fz = fy - lab.z / 200
        let x = 0.95047 * inverseLabCurve(fx)
        let y = inverseLabCurve(fy)
        let z = 1.08883 * inverseLabCurve(fz)
        let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        return SIMD3(sRGB(r), sRGB(g), sRGB(b))
    }

    private static func linear(_ value: Double) -> Double {
        value <= 0.04045 ? value / 12.92 : Foundation.pow((value + 0.055) / 1.055, 2.4)
    }

    private static func sRGB(_ value: Double) -> Double {
        let encoded = value <= 0.0031308
            ? 12.92 * value
            : 1.055 * Foundation.pow(Swift.max(0, value), 1.0 / 2.4) - 0.055
        return Swift.min(1, Swift.max(0, encoded))
    }

    private static func labCurve(_ value: Double) -> Double {
        value > 216.0 / 24_389.0
            ? Foundation.pow(value, 1.0 / 3.0)
            : (24_389.0 / 27.0 * value + 16) / 116
    }

    private static func inverseLabCurve(_ value: Double) -> Double {
        let cube = value * value * value
        return cube > 216.0 / 24_389.0
            ? cube
            : (116 * value - 16) * 27.0 / 24_389.0
    }

    private static func channel(_ value: Double) -> UInt8 {
        UInt8(clamping: Int((Swift.min(1, Swift.max(0, value)) * 255).rounded()))
    }
}
