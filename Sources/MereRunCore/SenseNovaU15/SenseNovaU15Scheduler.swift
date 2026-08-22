import Foundation

public enum SenseNovaU15Scheduler {
    public static func timesteps(steps: Int, shift: Float) -> [Float] {
        precondition(steps > 0)
        return (0...steps).map { index in
            shifted(Float(index) / Float(steps), shift: shift)
        }
    }

    public static func shifted(_ timestep: Float, shift: Float) -> Float {
        let sigma = 1 - timestep
        let shiftedSigma = shift * sigma / (1 + (shift - 1) * sigma)
        return 1 - shiftedSigma
    }

    public static func noiseScale(
        imageTokenCount: Int,
        baseImageTokenCount: Int,
        baseScale: Float,
        maximum: Float,
        mode: String
    ) -> Float {
        guard mode == "resolution" || mode == "dynamic" || mode == "dynamic_sqrt" else {
            return min(baseScale, maximum)
        }
        var scale = sqrt(Float(imageTokenCount) / Float(baseImageTokenCount)) * baseScale
        if mode == "dynamic_sqrt" { scale = sqrt(scale) }
        return min(scale, maximum)
    }
}
