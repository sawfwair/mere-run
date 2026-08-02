import Foundation
import MLX

/// A global camera pose tracked separately from DreamX's chunk-relative model conditioning.
/// The matrix maps world coordinates into the camera coordinate system.
public struct Wan2DreamXWorldPose: Codable, Hashable, Sendable {
    public let worldToCamera: [Float]

    public init(worldToCamera: [Float]) {
        precondition(worldToCamera.count == 16)
        precondition(worldToCamera.allSatisfy(\.isFinite))
        self.worldToCamera = worldToCamera
    }

    enum CodingKeys: String, CodingKey {
        case worldToCamera = "world_to_camera"
    }

    public static let identity = Wan2DreamXWorldPose(worldToCamera: Wan2PoseMatrix.identity.values)

    public var position: [Float] {
        let cameraToWorld = Wan2PoseMatrix(worldToCamera).invertedRigid()
        return [
            cameraToWorld.values[3],
            cameraToWorld.values[7],
            cameraToWorld.values[11],
        ]
    }

    public var yawDegrees: Float {
        let cameraToWorld = Wan2PoseMatrix(worldToCamera).invertedRigid()
        let forward = cameraToWorld.transformDirection(SIMD3<Float>(0, 0, 1))
        return atan2(forward.x, forward.z) * 180 / .pi
    }

    public func translationDistance(to other: Self) -> Float {
        let lhs = position
        let rhs = other.position
        let x = lhs[0] - rhs[0]
        let y = lhs[1] - rhs[1]
        let z = lhs[2] - rhs[2]
        return sqrt(x * x + y * y + z * z)
    }

    public func yawDistanceDegrees(to other: Self) -> Float {
        let raw = abs(yawDegrees - other.yawDegrees).truncatingRemainder(dividingBy: 360)
        return min(raw, 360 - raw)
    }

    public func viewOverlapScore(with other: Self) -> Float {
        let yawRadians = Double(yawDistanceDegrees(to: other) * .pi / 180)
        let orientation = max(0, cos(yawRadians))
        let proximity = exp(-Double(translationDistance(to: other)))
        return Float(orientation * proximity)
    }
}

/// Builds a persistent world-space pose path while leaving the model-facing AR path untouched.
public enum Wan2DreamXWorldTrajectory {
    public static func compile(
        segments: [Wan2DreamXARTrajectorySegment],
        pixelFrameCount: Int,
        speed: Float = Wan2DreamXARTrajectory.defaultSpeed,
        startingAt start: Wan2DreamXWorldPose = .identity
    ) -> [Wan2DreamXWorldPose] {
        let local = Wan2DreamXARTrajectory.compile(
            segments: segments,
            pixelFrameCount: pixelFrameCount,
            speed: speed,
            chunkRelative: false
        )
        let startMatrix = Wan2PoseMatrix(start.worldToCamera)
        return stride(from: 0, to: local.viewMatrices.count, by: 16).map { offset in
            let relative = Wan2PoseMatrix(Array(local.viewMatrices[offset..<(offset + 16)]))
            return Wan2DreamXWorldPose(worldToCamera: (relative * startMatrix).values)
        }
    }
}

public enum Wan2DreamXSceneMemoryMode: String, Codable, Hashable, Sendable {
    case disabled
    /// Reconstructs the paper's geometry retrieval and uses a conservative clean-latent anchor.
    /// This is not the unreleased memory-trained DreamX pathway.
    case paperReconstructedRevisitAnchor = "paper_reconstructed_revisit_anchor"
}

public struct Wan2DreamXSceneMemoryPolicy: Codable, Hashable, Sendable {
    public let mode: Wan2DreamXSceneMemoryMode
    public let maximumFrameCount: Int
    public let minimumFrameGap: Int
    public let maximumYawDistanceDegrees: Float
    public let maximumTranslationDistance: Float
    public let recyclingStrength: Float

    public init(
        mode: Wan2DreamXSceneMemoryMode = .paperReconstructedRevisitAnchor,
        maximumFrameCount: Int = 96,
        minimumFrameGap: Int = 3,
        maximumYawDistanceDegrees: Float = 2,
        maximumTranslationDistance: Float = 0.1,
        recyclingStrength: Float = 0.08
    ) {
        precondition(maximumFrameCount >= 0)
        precondition(minimumFrameGap >= 0)
        precondition(maximumYawDistanceDegrees.isFinite && maximumYawDistanceDegrees >= 0)
        precondition(maximumTranslationDistance.isFinite && maximumTranslationDistance >= 0)
        precondition(recyclingStrength.isFinite && recyclingStrength >= 0 && recyclingStrength <= 1)
        self.mode = mode
        self.maximumFrameCount = maximumFrameCount
        self.minimumFrameGap = minimumFrameGap
        self.maximumYawDistanceDegrees = maximumYawDistanceDegrees
        self.maximumTranslationDistance = maximumTranslationDistance
        self.recyclingStrength = recyclingStrength
    }

    public static let disabled = Self(
        mode: .disabled,
        maximumFrameCount: 0,
        recyclingStrength: 0
    )
}

public struct Wan2DreamXSceneMemoryMatchMetadata: Codable, Hashable, Sendable {
    public let memoryFrameIndex: Int
    public let targetFrameIndex: Int
    public let temporalGap: Int
    public let yawDistanceDegrees: Float
    public let translationDistance: Float
    public let viewOverlapScore: Float
}

struct Wan2DreamXSceneMemoryEntry: @unchecked Sendable {
    let frameIndex: Int
    let pose: Wan2DreamXWorldPose
    let cleanLatent: MLXArray
}

struct Wan2DreamXSceneMemoryMatch: @unchecked Sendable {
    let metadata: Wan2DreamXSceneMemoryMatchMetadata
    let cleanLatent: MLXArray
}

struct Wan2DreamXSceneMemoryCheckpoint: @unchecked Sendable {
    let entries: [Wan2DreamXSceneMemoryEntry]
}

struct Wan2DreamXSceneMemory: @unchecked Sendable {
    let policy: Wan2DreamXSceneMemoryPolicy
    private(set) var entries: [Wan2DreamXSceneMemoryEntry] = []

    init(policy: Wan2DreamXSceneMemoryPolicy) {
        self.policy = policy
    }

    var frameCount: Int { entries.count }

    mutating func reset() {
        entries.removeAll(keepingCapacity: true)
    }

    func checkpoint() -> Wan2DreamXSceneMemoryCheckpoint {
        Wan2DreamXSceneMemoryCheckpoint(entries: entries)
    }

    mutating func restore(_ checkpoint: Wan2DreamXSceneMemoryCheckpoint) {
        entries = checkpoint.entries
        trimToCapacity()
    }

    func retrieve(
        for targetPose: Wan2DreamXWorldPose,
        targetFrameIndex: Int
    ) -> Wan2DreamXSceneMemoryMatch? {
        guard policy.mode != .disabled, policy.maximumFrameCount > 0 else { return nil }
        let paperGap = Int(floor(Double(targetFrameIndex) * 0.2))
        let requiredGap = max(policy.minimumFrameGap, paperGap)
        return entries.compactMap { entry -> Wan2DreamXSceneMemoryMatch? in
            let gap = targetFrameIndex - entry.frameIndex
            guard gap >= requiredGap else { return nil }
            let yaw = targetPose.yawDistanceDegrees(to: entry.pose)
            let translation = targetPose.translationDistance(to: entry.pose)
            guard yaw <= policy.maximumYawDistanceDegrees,
                  translation <= policy.maximumTranslationDistance else {
                return nil
            }
            return Wan2DreamXSceneMemoryMatch(
                metadata: Wan2DreamXSceneMemoryMatchMetadata(
                    memoryFrameIndex: entry.frameIndex,
                    targetFrameIndex: targetFrameIndex,
                    temporalGap: gap,
                    yawDistanceDegrees: yaw,
                    translationDistance: translation,
                    viewOverlapScore: targetPose.viewOverlapScore(with: entry.pose)
                ),
                cleanLatent: entry.cleanLatent
            )
        }.max { lhs, rhs in
            let lhsCost = lhs.metadata.yawDistanceDegrees + 10 * lhs.metadata.translationDistance
            let rhsCost = rhs.metadata.yawDistanceDegrees + 10 * rhs.metadata.translationDistance
            if lhsCost == rhsCost {
                return lhs.metadata.temporalGap < rhs.metadata.temporalGap
            }
            return lhsCost > rhsCost
        }
    }

    mutating func record(
        cleanLatents: MLXArray,
        poses: [Wan2DreamXWorldPose],
        startingAt frameIndex: Int
    ) {
        guard policy.mode != .disabled, policy.maximumFrameCount > 0 else { return }
        precondition(cleanLatents.ndim == 4)
        precondition(cleanLatents.dim(1) == poses.count)
        for index in poses.indices {
            let latent = cleanLatents[0..., index..<(index + 1), 0..., 0...]
            eval(latent)
            entries.append(Wan2DreamXSceneMemoryEntry(
                frameIndex: frameIndex + index,
                pose: poses[index],
                cleanLatent: latent
            ))
        }
        trimToCapacity()
    }

    private mutating func trimToCapacity() {
        if entries.count > policy.maximumFrameCount {
            entries.removeFirst(entries.count - policy.maximumFrameCount)
        }
    }
}

private struct Wan2PoseMatrix: Hashable {
    let values: [Float]

    init(_ values: [Float]) {
        precondition(values.count == 16)
        self.values = values
    }

    static let identity = Wan2PoseMatrix([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ])

    static func * (left: Self, right: Self) -> Self {
        var output = Array(repeating: Float(0), count: 16)
        for row in 0..<4 {
            for column in 0..<4 {
                output[row * 4 + column] = (0..<4).reduce(Float(0)) { partial, index in
                    partial + left.values[row * 4 + index] * right.values[index * 4 + column]
                }
            }
        }
        return Self(output)
    }

    func withTranslation(_ translation: SIMD3<Float>) -> Self {
        var output = values
        output[3] = translation.x
        output[7] = translation.y
        output[11] = translation.z
        return Self(output)
    }

    func transformDirection(_ direction: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            values[0] * direction.x + values[1] * direction.y + values[2] * direction.z,
            values[4] * direction.x + values[5] * direction.y + values[6] * direction.z,
            values[8] * direction.x + values[9] * direction.y + values[10] * direction.z
        )
    }

    func invertedRigid() -> Self {
        let rotationTranspose = Self([
            values[0], values[4], values[8], 0,
            values[1], values[5], values[9], 0,
            values[2], values[6], values[10], 0,
            0, 0, 0, 1,
        ])
        let translation = SIMD3(values[3], values[7], values[11])
        return rotationTranspose.withTranslation(-rotationTranspose.transformDirection(translation))
    }
}
