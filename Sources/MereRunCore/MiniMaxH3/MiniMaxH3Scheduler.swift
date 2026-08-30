import Foundation
import MLX

public struct MiniMaxH3Schedule: Sendable {
    public let sigmas: [Float]
    public let timesteps: [Float]

    public init(pointCount: Int, shift: Float) throws {
        guard pointCount >= 2 else {
            throw MiniMaxH3LayoutError.invalidGeometry("schedule needs at least two points")
        }
        guard shift > 0 else {
            throw MiniMaxH3LayoutError.invalidGeometry("schedule shift must be positive")
        }
        var shifted: [Float] = []
        shifted.reserveCapacity(pointCount)
        for index in 0..<pointCount {
            let base = Float(pointCount - 1 - index) / Float(pointCount - 1)
            let sigma = shift * base / (1 + (shift - 1) * base)
            if shifted.last != sigma { shifted.append(sigma) }
        }
        self.sigmas = shifted
        self.timesteps = shifted.dropLast().map { 1 - $0 }
    }

    public init(baseSigmas: [Float], shift: Float) throws {
        guard baseSigmas.count >= 2 else {
            throw MiniMaxH3LayoutError.invalidGeometry("schedule needs at least two points")
        }
        guard shift > 0 else {
            throw MiniMaxH3LayoutError.invalidGeometry("schedule shift must be positive")
        }
        guard baseSigmas.first.map({ $0 > 0 && $0 <= 1 }) == true,
              baseSigmas.last == 0,
              zip(baseSigmas, baseSigmas.dropFirst()).allSatisfy({ $0 > $1 }) else {
            throw MiniMaxH3LayoutError.invalidGeometry(
                "base sigmas must strictly descend from (0, 1] to 0"
            )
        }
        self.sigmas = baseSigmas.map { base in
            shift * base / (1 + (shift - 1) * base)
        }
        self.timesteps = sigmas.dropLast().map { 1 - $0 }
    }

    public func step(sample: MLXArray, velocity: MLXArray, index: Int) -> MLXArray {
        precondition(index >= 0 && index + 1 < sigmas.count)
        let sigma = sigmas[index]
        let sigmaNext = sigmas[index + 1]
        let timestep = timesteps[index]
        let denoised = sample + (1 - timestep) * velocity
        let ratio = sigmaNext / sigma
        return (ratio * sample.asType(.float32) + (1 - ratio) * denoised.asType(.float32)).asType(sample.dtype)
    }

    public static func noise(clean: MLXArray, timestep: Float, noise: MLXArray) -> MLXArray {
        timestep * clean + (1 - timestep) * noise
    }
}
