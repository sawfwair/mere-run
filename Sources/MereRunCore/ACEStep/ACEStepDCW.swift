import Foundation
import MLX

enum ACEStepDCW {
    static func applyHaar(
        xNext: MLXArray,
        denoised: MLXArray,
        tCurr: Float,
        enabled: Bool,
        mode: ACEStepDCWMode,
        scaler: Float,
        highScaler: Float
    ) -> MLXArray {
        guard enabled else {
            return xNext
        }

        let rawLow = scaler
        let rawHigh = highScaler
        let lowScale = tCurr * rawLow
        let highScale = (1.0 - tCurr) * rawLow
        let doubleHighScale = (1.0 - tCurr) * rawHigh

        switch mode {
        case .pix:
            guard rawLow != 0 else {
                return xNext
            }
            return xNext + MLXArray(rawLow).asType(xNext.dtype) * (xNext - denoised)
        case .low:
            guard lowScale != 0 else {
                return xNext
            }
        case .high:
            guard highScale != 0 else {
                return xNext
            }
        case .double:
            guard lowScale != 0 || doubleHighScale != 0 else {
                return xNext
            }
        }

        let targetFrames = xNext.dim(1)
        var (xLow, xHigh) = haarDWT1D(xNext)
        let (yLow, yHigh) = haarDWT1D(denoised)

        switch mode {
        case .low:
            let s = MLXArray(lowScale).asType(xLow.dtype)
            xLow = xLow + s * (xLow - yLow)
        case .high:
            let s = MLXArray(highScale).asType(xHigh.dtype)
            xHigh = xHigh + s * (xHigh - yHigh)
        case .double:
            if lowScale != 0 {
                let s = MLXArray(lowScale).asType(xLow.dtype)
                xLow = xLow + s * (xLow - yLow)
            }
            if doubleHighScale != 0 {
                let s = MLXArray(doubleHighScale).asType(xHigh.dtype)
                xHigh = xHigh + s * (xHigh - yHigh)
            }
        case .pix:
            break
        }

        return haarIDWT1D(low: xLow, high: xHigh, targetFrames: targetFrames)
    }

    static func haarDWT1D(_ x: MLXArray) -> (low: MLXArray, high: MLXArray) {
        precondition(x.ndim == 3, "DCW expects [B,T,C] latents.")

        var padded = x
        if x.dim(1) % 2 == 1 {
            let pad = MLXArray.zeros([x.dim(0), 1, x.dim(2)], dtype: x.dtype)
            padded = MLX.concatenated([x, pad], axis: 1)
        }

        let even = padded[0..., 0..<padded.dim(1), 0...][0..., .stride(by: 2), 0...]
        let odd = padded[0..., 1..<padded.dim(1), 0...][0..., .stride(by: 2), 0...]
        let invSqrt2 = MLXArray(Float(1.0 / sqrt(2.0))).asType(x.dtype)
        let low = (even + odd) * invSqrt2
        let high = (even - odd) * invSqrt2
        return (low, high)
    }

    static func haarIDWT1D(low: MLXArray, high: MLXArray, targetFrames: Int) -> MLXArray {
        precondition(low.ndim == 3 && high.ndim == 3, "DCW expects [B,T,C] wavelet bands.")

        let invSqrt2 = MLXArray(Float(1.0 / sqrt(2.0))).asType(low.dtype)
        let even = (low + high) * invSqrt2
        let odd = (low - high) * invSqrt2
        let interleaved = MLX.stacked([even, odd], axis: 2)
            .reshaped(even.dim(0), even.dim(1) * 2, even.dim(2))
        return interleaved[0..., 0..<targetFrames, 0...]
    }
}
