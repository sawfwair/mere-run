import Foundation

public struct Ideogram4Scheduler: Sendable, Hashable {
    public let steps: Int
    public let mean: Double
    public let std: Double
    public let guidanceSchedule: [Float]
    public let logSNRMin: Double
    public let logSNRMax: Double

    public init(
        steps: Int,
        width: Int,
        height: Int,
        guidanceScale: Double,
        mean: Double,
        std: Double,
        logSNRMin: Double = -15.0,
        logSNRMax: Double = 18.0
    ) {
        self.steps = steps
        self.mean = mean + 0.5 * Foundation.log(Double(width * height) / Double(512 * 512))
        self.std = std
        self.guidanceSchedule = Ideogram4Scheduler.guidanceSchedule(steps: steps, guidanceScale: Float(guidanceScale))
        self.logSNRMin = logSNRMin
        self.logSNRMax = logSNRMax
    }

    public static func preset(steps: Int, width: Int, height: Int, guidanceScale: Double) -> Ideogram4Scheduler {
        switch steps {
        case 48:
            return Ideogram4Scheduler(
                steps: steps,
                width: width,
                height: height,
                guidanceScale: guidanceScale,
                mean: 0.0,
                std: 1.5
            )
        case 20:
            return Ideogram4Scheduler(
                steps: steps,
                width: width,
                height: height,
                guidanceScale: guidanceScale,
                mean: 0.0,
                std: 1.75
            )
        case 12:
            return Ideogram4Scheduler(
                steps: steps,
                width: width,
                height: height,
                guidanceScale: guidanceScale,
                mean: 0.5,
                std: 1.75
            )
        default:
            return Ideogram4Scheduler(
                steps: steps,
                width: width,
                height: height,
                guidanceScale: guidanceScale,
                mean: 0.0,
                std: 1.75
            )
        }
    }

    public func interval(_ index: Int) -> Double {
        Double(index) / Double(steps)
    }

    public func value(at interval: Double) -> Float {
        let z = Self.normalQuantile(interval)
        let y = mean + std * z
        let t = 1.0 - Self.sigmoid(y)
        let tMin = 1.0 / (1.0 + Foundation.exp(0.5 * logSNRMax))
        let tMax = 1.0 / (1.0 + Foundation.exp(0.5 * logSNRMin))
        return Float(min(max(t, tMin), tMax))
    }

    private static func guidanceSchedule(steps: Int, guidanceScale: Float) -> [Float] {
        switch steps {
        case 48:
            return Array(repeating: 3.0, count: 3) + Array(repeating: 7.0, count: 45)
        case 20:
            return Array(repeating: 3.0, count: 2) + Array(repeating: 7.0, count: 18)
        case 12:
            return [3.0] + Array(repeating: 7.0, count: 11)
        default:
            return Array(repeating: guidanceScale, count: steps)
        }
    }

    private static func sigmoid(_ x: Double) -> Double {
        if x >= 0 {
            let expNeg = Foundation.exp(-x)
            return 1.0 / (1.0 + expNeg)
        } else {
            let expPos = Foundation.exp(x)
            return expPos / (1.0 + expPos)
        }
    }

    // Peter J. Acklam's rational approximation for the inverse normal CDF.
    private static func normalQuantile(_ p: Double) -> Double {
        if p <= 0 { return -Double.infinity }
        if p >= 1 { return Double.infinity }

        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ]
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ]
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ]
        let plow = 0.02425
        let phigh = 1.0 - plow

        if p < plow {
            let q = Foundation.sqrt(-2.0 * Foundation.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
        if p > phigh {
            let q = Foundation.sqrt(-2.0 * Foundation.log(1.0 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }

        let q = p - 0.5
        let r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    }
}
