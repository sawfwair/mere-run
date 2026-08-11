import Foundation
import MLX

struct MiniMaxH3ServingRNG: Sendable {
    private var state: UInt64 = 0
    private var increment: UInt64
    private var spare: Float = 0
    private var hasSpare = false

    init(seed: UInt64) {
        increment = (seed &<< 1) | 1
        _ = nextUInt32()
        state &+= seed ^ 0x9e37_79b9_7f4a_7c15
        _ = nextUInt32()
    }

    mutating func nextUInt32() -> UInt32 {
        let oldState = state
        state = oldState &* 6_364_136_223_846_793_005 &+ increment
        let shifted = UInt32(truncatingIfNeeded: ((oldState >> 18) ^ oldState) >> 27)
        let rotation = UInt32(oldState >> 59)
        return (shifted >> rotation) | (shifted << ((0 &- rotation) & 31))
    }

    mutating func nextNormal() -> Float {
        if hasSpare {
            hasSpare = false
            return spare
        }
        let first = (Double(nextUInt32()) + 1) / 4_294_967_297
        let second = (Double(nextUInt32()) + 0.5) / 4_294_967_296
        let radius = sqrt(-2 * log(first))
        let angle = 2 * 3.14159265358979323846 * second
        spare = Float(radius * sin(angle))
        hasSpare = true
        return Float(radius * cos(angle))
    }

    mutating func normalArray(shape: [Int]) -> MLXArray {
        precondition(shape.allSatisfy { $0 > 0 })
        let count = shape.reduce(1, *)
        var values: [Float] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            values.append(nextNormal())
        }
        return MLXArray(values).reshaped(shape)
    }
}

enum MiniMaxH3ServingContract {
    static func referenceImageCanvas(
        width: Int,
        height: Int,
        targetWidth: Int,
        targetHeight: Int
    ) throws -> (width: Int, height: Int) {
        guard width > 0,
              height > 0,
              targetWidth > 0,
              targetHeight > 0 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "reference image and target dimensions must be positive"
            )
        }
        let targetArea = Double(targetWidth) * Double(targetHeight)
        let sourceArea = Double(width) * Double(height)
        let scale = min(1, sqrt(targetArea / sourceArea))
        let resolvedWidth = Int((Double(width) * scale / 32).rounded(.toNearestOrEven)) * 32
        let resolvedHeight = Int((Double(height) * scale / 32).rounded(.toNearestOrEven)) * 32
        return (max(32, resolvedWidth), max(32, resolvedHeight))
    }

    static func extrapolationRatio(
        currentSigma: Float,
        lastSigma: Float,
        previousSigma: Float?
    ) -> Float {
        guard let previousSigma else { return 0 }
        let denominator = lastSigma - previousSigma
        let ratio = denominator == 0 ? 0 : (currentSigma - lastSigma) / denominator
        return min(2, max(-2, ratio))
    }

    static func noisedCondition(_ clean: MLXArray, seed: UInt64) -> MLXArray {
        var generator = MiniMaxH3ServingRNG(seed: seed)
        return MiniMaxH3Schedule.noise(
            clean: clean,
            timestep: 0.999,
            noise: generator.normalArray(shape: clean.shape)
        )
    }

    static func targetVideoNoise(
        seed: UInt64,
        latentFrames: Int,
        latentHeight: Int,
        latentWidth: Int
    ) -> MLXArray {
        var generator = MiniMaxH3ServingRNG(seed: seed)
        return generator.normalArray(shape: [1, 24, latentFrames, latentHeight, latentWidth])
    }

    static func targetAudioNoise(seed: UInt64, latentFrames: Int) -> MLXArray {
        var generator = MiniMaxH3ServingRNG(seed: seed)
        let latent = generator.normalArray(shape: [1, 32, 2, latentFrames])
        return MiniMaxH3Geometry.packAudio(latent).expandedDimensions(axis: 0)
    }
}
