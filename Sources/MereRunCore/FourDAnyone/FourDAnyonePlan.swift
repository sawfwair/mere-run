import Foundation
import MLX

public enum FourDAnyoneError: LocalizedError {
    case invalidInput(String)
    case invalidCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let reason): return "Invalid 4DAnyone input: \(reason)"
        case .invalidCheckpoint(let reason): return "Invalid 4DAnyone checkpoint: \(reason)"
        }
    }
}

/// Canonical camera identities and the released cyclic denoising order.
public struct FourDAnyoneViewPlan: Equatable, Sendable {
    public let viewsPerLayer: Int
    public let layerPitches: [Int]
    public let groupSize: Int
    public let referencePacking: Bool
    public let targetContextRouting: Bool

    public var viewCount: Int { viewsPerLayer * layerPitches.count }
    public var packedSourceCount: Int { referencePacking ? 2 : 1 }

    public init(
        viewsPerLayer: Int = 6,
        layerPitches: [Int] = [15],
        groupSize: Int? = nil,
        referencePacking: Bool = true,
        targetContextRouting: Bool = true
    ) throws {
        guard viewsPerLayer > 0, !layerPitches.isEmpty,
              !viewsPerLayer.multipliedReportingOverflow(by: layerPitches.count).overflow,
              Set(layerPitches).count == layerPitches.count,
              layerPitches.allSatisfy({ (-15...45).contains($0) }) else {
            throw FourDAnyoneError.invalidInput("Use positive view counts and distinct pitch layers from -15 to 45.")
        }
        let resolvedGroup = groupSize ?? (viewsPerLayer.isMultiple(of: 6) ? 6 : 4)
        guard [4, 6].contains(resolvedGroup), viewsPerLayer.isMultiple(of: resolvedGroup) else {
            throw FourDAnyoneError.invalidInput("Each layer must be divisible by the group size, four or six.")
        }
        self.viewsPerLayer = viewsPerLayer
        self.layerPitches = layerPitches
        self.groupSize = resolvedGroup
        self.referencePacking = referencePacking && viewsPerLayer * layerPitches.count > 6
        self.targetContextRouting = targetContextRouting
    }

    public var cameraOrder: [Int] {
        guard layerPitches.count > 1 else { return Array(0..<viewsPerLayer) }
        let layers = layerPitches.indices.sorted { layerPitches[$0] < layerPitches[$1] }
        var order = layers.map { $0 * viewsPerLayer }
        for yaw in 1..<viewsPerLayer {
            let ranks = yaw.isMultiple(of: 2)
                ? Array(1..<layers.count)
                : Array((1..<layers.count).reversed())
            order += ranks.map { layers[$0] * viewsPerLayer + yaw }
        }
        order += (1..<viewsPerLayer).reversed().map { layers[0] * viewsPerLayer + $0 }
        return order
    }

    /// Returns a disjoint partition of canonical IDs for one Base denoising step.
    public func groups(step: Int) -> [[Int]] {
        precondition(step >= 0)
        let order = cameraOrder
        let offset = targetContextRouting ? step % viewCount : 0
        return stride(from: 0, to: viewCount, by: groupSize).map { start in
            (0..<groupSize).map { order[(start + offset + $0) % viewCount] }
        }
    }
}

/// The released flow-matching schedule, including the terminal zero sigma.
public struct FourDAnyoneSchedule: Equatable, Sendable {
    public let sigmas: [Float]
    public var timesteps: [Float] { sigmas.dropLast().map { $0 * 1_000 } }
    public var stepCount: Int { sigmas.count - 1 }

    public init(steps: Int = 24) throws {
        guard steps > 0 else {
            throw FourDAnyoneError.invalidInput("The denoising step count must be positive.")
        }
        self.sigmas = (0..<steps).map { index in
            let sigma = Float(steps - index) / Float(steps)
            return 5 * sigma / (1 + 4 * sigma)
        } + [0]
    }

    public func step(prediction: MLXArray, sample: MLXArray, index: Int) -> MLXArray {
        precondition((0..<stepCount).contains(index))
        // A scalar tensor preserves the reference sample's arithmetic dtype.
        let delta = MLXArray(sigmas[index + 1] - sigmas[index]).asType(sample.dtype)
        return sample + prediction * delta
    }
}
