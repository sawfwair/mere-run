import Foundation
import MLX
import MLXFast
import MLXNN

public struct Wan2DreamXTrajectorySegment: Hashable, Sendable {
    public let action: String
    public let speed: Float

    public init(action: String, speed: Float = 1) {
        precondition(speed > 0)
        self.action = action.lowercased()
        self.speed = speed
    }
}

public struct Wan2ProjectiveCameraConditioning: Hashable, Sendable {
    public let frameCount: Int
    public let viewMatrices: [Float]
    public let intrinsics: [Float]

    public init(frameCount: Int, viewMatrices: [Float], intrinsics: [Float]) {
        precondition(frameCount > 0)
        precondition(viewMatrices.count == frameCount * 16)
        precondition(intrinsics.count == frameCount * 9)
        self.frameCount = frameCount
        self.viewMatrices = viewMatrices
        self.intrinsics = intrinsics
    }
}

public enum Wan2DreamXCameraTrajectory {
    public static let defaultFocalX: Float = 969.696_96 / (960 * 2)
    public static let defaultFocalY: Float = 969.696_96 / (540 * 2)

    public static func segments(for control: Wan2WorldCameraControl) -> [Wan2DreamXTrajectorySegment] {
        switch control.motion {
        case .hold:
            return [Wan2DreamXTrajectorySegment(action: " ")]
        case .forward:
            return [Wan2DreamXTrajectorySegment(action: "w", speed: max(abs(control.translationMeters[2]), 0.001))]
        case .backward:
            return [Wan2DreamXTrajectorySegment(action: "s", speed: max(abs(control.translationMeters[2]), 0.001))]
        case .strafeLeft:
            return [Wan2DreamXTrajectorySegment(action: "a", speed: max(abs(control.translationMeters[0]), 0.001))]
        case .strafeRight:
            return [Wan2DreamXTrajectorySegment(action: "d", speed: max(abs(control.translationMeters[0]), 0.001))]
        case .yawLeft:
            return [Wan2DreamXTrajectorySegment(action: "j", speed: max(abs(control.rotationDegrees[1]) / 10, 0.001))]
        case .yawRight:
            return [Wan2DreamXTrajectorySegment(action: "l", speed: max(abs(control.rotationDegrees[1]) / 10, 0.001))]
        case .custom:
            var action = ""
            if control.translationMeters[2] > 0 { action.append("w") }
            if control.translationMeters[2] < 0 { action.append("s") }
            if control.translationMeters[0] < 0 { action.append("a") }
            if control.translationMeters[0] > 0 { action.append("d") }
            if control.rotationDegrees[0] > 0 { action.append("i") }
            if control.rotationDegrees[0] < 0 { action.append("k") }
            if control.rotationDegrees[1] < 0 { action.append("j") }
            if control.rotationDegrees[1] > 0 { action.append("l") }
            let translationSpeed = control.translationMeters.map(abs).max() ?? 0
            let rotationSpeed = (control.rotationDegrees.map(abs).max() ?? 0) / 10
            return [Wan2DreamXTrajectorySegment(
                action: action.isEmpty ? " " : action,
                speed: max(max(translationSpeed, rotationSpeed), 0.001)
            )]
        }
    }

    public static func compile(
        control: Wan2WorldCameraControl,
        pixelFrameCount: Int
    ) -> Wan2ProjectiveCameraConditioning {
        compile(segments: segments(for: control), pixelFrameCount: pixelFrameCount)
    }

    public static func compile(
        segments: [Wan2DreamXTrajectorySegment],
        pixelFrameCount: Int,
        focalX: Float = defaultFocalX,
        focalY: Float = defaultFocalY,
        principalX: Float = 0,
        principalY: Float = 0
    ) -> Wan2ProjectiveCameraConditioning {
        precondition(!segments.isEmpty)
        precondition(pixelFrameCount > 0)
        let duration = Int(ceil(Double(pixelFrameCount) / Double(segments.count)))
        var position = SIMD3<Float>(repeating: 0)
        var rotationDegrees = SIMD3<Float>(repeating: 0)
        var worldToCamera: [Wan2Matrix4] = [.identity]
        worldToCamera.reserveCapacity(pixelFrameCount + 1)

        for segment in segments {
            let action = segment.action
            let translationStep = translationStep(
                action: action,
                speed: segment.speed,
                rotationDegrees: rotationDegrees,
                duration: duration
            )
            let rotationStep = rotationStep(action: action, speed: segment.speed, duration: duration)
            for frame in 1...duration {
                let nextPosition = position + translationStep * Float(frame)
                let nextRotation = rotationDegrees + rotationStep * Float(frame)
                let rotation = Wan2Matrix4.cameraRotation(
                    pitchDegrees: nextRotation.x,
                    yawDegrees: nextRotation.y,
                    rollDegrees: nextRotation.z
                )
                worldToCamera.append(rotation.withTranslation(-rotation.transformDirection(nextPosition)))
            }
            position += translationStep * Float(duration)
            rotationDegrees += rotationStep * Float(duration)
        }

        worldToCamera = Array(worldToCamera.prefix(pixelFrameCount))
        let aligned = latentFrameIndices(pixelFrameCount: pixelFrameCount)
        let views = aligned.flatMap { worldToCamera[$0].values }
        let intrinsic = [
            focalX, 0, principalX,
            0, focalY, principalY,
            0, 0, 1
        ]
        return Wan2ProjectiveCameraConditioning(
            frameCount: aligned.count,
            viewMatrices: views,
            intrinsics: aligned.flatMap { _ in intrinsic }
        )
    }

    public static func latentFrameIndices(pixelFrameCount: Int) -> [Int] {
        precondition(pixelFrameCount > 0 && (pixelFrameCount - 1).isMultiple(of: 4))
        return Array(stride(from: 0, through: pixelFrameCount - 1, by: 4))
    }

    private static func translationStep(
        action: String,
        speed: Float,
        rotationDegrees: SIMD3<Float>,
        duration: Int
    ) -> SIMD3<Float> {
        let yaw = rotationDegrees.y * .pi / 180
        let pitch = rotationDegrees.x * .pi / 180
        var translation = SIMD3<Float>(repeating: 0)
        if action.contains("w") || action.contains("s") {
            let direction: Float = action.contains("w") ? 1 : -1
            translation += SIMD3(
                -sin(yaw) * cos(pitch),
                sin(pitch),
                cos(yaw) * cos(pitch)
            ) * speed * direction
        }
        if action.contains("a") || action.contains("d") {
            let direction: Float = action.contains("d") ? 1 : -1
            translation += SIMD3(cos(yaw), 0, sin(yaw)) * speed * direction
        }
        return translation / Float(duration)
    }

    private static func rotationStep(action: String, speed: Float, duration: Int) -> SIMD3<Float> {
        var rotation = SIMD3<Float>(repeating: 0)
        if action.contains("j") { rotation.y += speed * 10 }
        if action.contains("l") { rotation.y -= speed * 10 }
        if action.contains("i") { rotation.x -= speed * 10 }
        if action.contains("k") { rotation.x += speed * 10 }
        return rotation / Float(duration)
    }
}

public struct Wan2DreamXARTrajectorySegment: Hashable, Sendable {
    public let action: String
    public let weight: Float

    public init(action: String, weight: Float = 1) {
        precondition(weight > 0)
        self.action = action.lowercased()
        self.weight = weight
    }
}

public enum Wan2DreamXARTrajectory {
    public static func segments(for control: Wan2WorldCameraControl) -> [Wan2DreamXARTrajectorySegment] {
        let action: String
        switch control.motion {
        case .hold: action = " "
        case .forward: action = "w"
        case .backward: action = "s"
        case .strafeLeft: action = "a"
        case .strafeRight: action = "d"
        case .yawLeft: action = "j"
        case .yawRight: action = "l"
        case .custom:
            var composite = ""
            if control.translationMeters[2] > 0 { composite.append("w") }
            if control.translationMeters[2] < 0 { composite.append("s") }
            if control.translationMeters[0] < 0 { composite.append("a") }
            if control.translationMeters[0] > 0 { composite.append("d") }
            if control.rotationDegrees[0] > 0 { composite.append("i") }
            if control.rotationDegrees[0] < 0 { composite.append("k") }
            if control.rotationDegrees[1] < 0 { composite.append("j") }
            if control.rotationDegrees[1] > 0 { composite.append("l") }
            action = composite.isEmpty ? " " : composite
        }
        return [Wan2DreamXARTrajectorySegment(action: action)]
    }

    public static func compile(
        control: Wan2WorldCameraControl,
        pixelFrameCount: Int,
        speed: Float? = nil,
        chunkRelative: Bool = true
    ) -> Wan2ProjectiveCameraConditioning {
        compile(
            segments: segments(for: control),
            pixelFrameCount: pixelFrameCount,
            speed: speed ?? physicalSpeed(for: control, pixelFrameCount: pixelFrameCount),
            chunkRelative: chunkRelative
        )
    }

    private static func physicalSpeed(
        for control: Wan2WorldCameraControl,
        pixelFrameCount: Int
    ) -> Float {
        precondition(pixelFrameCount > 0)
        let translation = control.translationMeters.map(abs).max() ?? 0
        let rotation = control.rotationDegrees.map(abs).max() ?? 0
        switch control.motion {
        case .hold:
            return 1.5
        case .forward, .backward, .strafeLeft, .strafeRight:
            return max(translation / (0.05 * Float(pixelFrameCount)), 0.001)
        case .yawLeft, .yawRight:
            return max(rotation / Float(pixelFrameCount), 0.001)
        case .custom:
            return max(
                max(translation / (0.05 * Float(pixelFrameCount)), rotation / Float(pixelFrameCount)),
                0.001
            )
        }
    }

    public static func compile(
        segments: [Wan2DreamXARTrajectorySegment],
        pixelFrameCount: Int,
        speed: Float = 1.5,
        chunkRelative: Bool = true
    ) -> Wan2ProjectiveCameraConditioning {
        precondition(!segments.isEmpty)
        precondition(pixelFrameCount > 0 && (pixelFrameCount - 1).isMultiple(of: 4))
        let totalWeight = segments.reduce(Float(0)) { $0 + $1.weight }
        var allocated = 0
        var weightedSegments: [(String, Int)] = []
        for (index, segment) in segments.enumerated() {
            let frames: Int
            if index == segments.count - 1 {
                frames = pixelFrameCount - allocated
            } else {
                frames = max(1, Int((Float(pixelFrameCount) * segment.weight / totalWeight).rounded(.toNearestOrEven)))
            }
            let clamped = min(frames, pixelFrameCount - allocated)
            if clamped > 0 {
                weightedSegments.append((segment.action, clamped))
                allocated += clamped
            }
        }

        var position = SIMD3<Float>(repeating: 0)
        var cameraToWorldRotation = Wan2Matrix4.identity
        var absoluteViews: [Wan2Matrix4] = []
        absoluteViews.reserveCapacity(pixelFrameCount)
        let movementStep = speed * 0.05
        let rotationStep = speed * .pi / 180
        for (action, frameCount) in weightedSegments {
            for _ in 0..<frameCount {
                var pitchDelta: Float = 0
                var yawDelta: Float = 0
                if action.contains("i") { pitchDelta += rotationStep }
                if action.contains("k") { pitchDelta -= rotationStep }
                if action.contains("j") { yawDelta -= rotationStep }
                if action.contains("l") { yawDelta += rotationStep }
                if pitchDelta != 0 || yawDelta != 0 {
                    cameraToWorldRotation = cameraToWorldRotation
                        * Wan2Matrix4.euler(roll: pitchDelta, pitch: yawDelta, yaw: 0)
                }
                var localMovement = SIMD3<Float>(repeating: 0)
                if action.contains("w") { localMovement.z += movementStep }
                if action.contains("s") { localMovement.z -= movementStep }
                if action.contains("a") { localMovement.x -= movementStep }
                if action.contains("d") { localMovement.x += movementStep }
                position += cameraToWorldRotation.transformDirection(localMovement)
                let worldToCameraRotation = cameraToWorldRotation.transposed()
                absoluteViews.append(
                    worldToCameraRotation.withTranslation(-worldToCameraRotation.transformDirection(position))
                )
            }
        }

        let alignedIndices = [0] + Array(stride(from: 1, to: pixelFrameCount, by: 4))
        let alignedViews = alignedIndices.map { absoluteViews[$0] }
        var relativeViews: [Wan2Matrix4] = []
        relativeViews.reserveCapacity(alignedViews.count)
        if chunkRelative {
            for chunkStart in stride(from: 0, to: alignedViews.count, by: 3) {
                let reference = alignedViews[chunkStart == 0 ? 0 : chunkStart - 1].invertedRigid()
                let chunkEnd = min(chunkStart + 3, alignedViews.count)
                relativeViews.append(contentsOf: (chunkStart..<chunkEnd).map { alignedViews[$0] * reference })
            }
        } else {
            let reference = alignedViews[0].invertedRigid()
            relativeViews = alignedViews.map { $0 * reference }
        }
        let intrinsic: [Float] = [
            Wan2DreamXCameraTrajectory.defaultFocalX, 0, 0.5,
            0, Wan2DreamXCameraTrajectory.defaultFocalY, 0.5,
            0, 0, 1
        ]
        return Wan2ProjectiveCameraConditioning(
            frameCount: relativeViews.count,
            viewMatrices: relativeViews.flatMap(\.values),
            intrinsics: relativeViews.flatMap { _ in intrinsic }
        )
    }
}

struct Wan2ProjectiveTransforms {
    let query: MLXArray
    let keyValue: MLXArray
    let output: MLXArray
}

enum Wan2ProjectivePositionEncoding {
    static func prepare(
        conditioning: Wan2ProjectiveCameraConditioning,
        batchSize: Int,
        dtype: DType
    ) -> Wan2ProjectiveTransforms {
        var query: [Float] = []
        var keyValue: [Float] = []
        var output: [Float] = []
        query.reserveCapacity(conditioning.frameCount * 16)
        keyValue.reserveCapacity(conditioning.frameCount * 16)
        output.reserveCapacity(conditioning.frameCount * 16)

        for frame in 0..<conditioning.frameCount {
            let view = Wan2Matrix4(Array(conditioning.viewMatrices[(frame * 16)..<((frame + 1) * 16)]))
            let intrinsicOffset = frame * 9
            let focalX = conditioning.intrinsics[intrinsicOffset]
            let focalY = conditioning.intrinsics[intrinsicOffset + 4]
            let lifted = Wan2Matrix4.diagonal(focalX, focalY, 1, 1)
            let liftedInverse = Wan2Matrix4.diagonal(1 / focalX, 1 / focalY, 1, 1)
            let projection = lifted * view
            query.append(contentsOf: projection.transposed().values)
            keyValue.append(contentsOf: (view.invertedRigid() * liftedInverse).values)
            output.append(contentsOf: projection.values)
        }

        func tensor(_ values: [Float]) -> MLXArray {
            let single = MLXArray(values).reshaped(1, conditioning.frameCount, 4, 4).asType(dtype)
            return batchSize == 1
                ? single
                : MLX.broadcast(single, to: [batchSize, conditioning.frameCount, 4, 4])
        }
        return Wan2ProjectiveTransforms(
            query: tensor(query),
            keyValue: tensor(keyValue),
            output: tensor(output)
        )
    }

    static func apply(_ input: MLXArray, matrices: MLXArray, cameraFrames: Int) -> MLXArray {
        precondition(input.ndim == 4)
        precondition(matrices.shape == [input.dim(0), cameraFrames, 4, 4])
        precondition(input.dim(2).isMultiple(of: cameraFrames))
        precondition(input.dim(3).isMultiple(of: 4))
        let batch = input.dim(0)
        let heads = input.dim(1)
        let patchesPerFrame = input.dim(2) / cameraFrames
        let vectorGroups = input.dim(3) / 4
        let vectors = input
            .reshaped(batch, heads, cameraFrames, patchesPerFrame, vectorGroups, 4)
            .expandedDimensions(axis: -1)
        let tiledMatrices = matrices
            .reshaped(batch, 1, cameraFrames, 1, 1, 4, 4)
        return MLX.matmul(tiledMatrices, vectors)
            .squeezed(axis: -1)
            .reshaped(input.shape)
    }
}

final class Wan2ProjectiveSelfAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: Wan2RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: Wan2RMSNorm

    init(
        dimensions: Int,
        attentionDimensions: Int? = nil,
        heads: Int,
        epsilon: Float
    ) {
        let attentionDimensions = attentionDimensions ?? dimensions
        precondition(attentionDimensions.isMultiple(of: heads))
        self.heads = heads
        self.headDimension = attentionDimensions / heads
        self.scale = 1 / Float(headDimension).squareRoot()
        self._query.wrappedValue = Linear(dimensions, attentionDimensions, bias: true)
        self._key.wrappedValue = Linear(dimensions, attentionDimensions, bias: true)
        self._value.wrappedValue = Linear(dimensions, attentionDimensions, bias: true)
        self._output.wrappedValue = Linear(attentionDimensions, dimensions, bias: true)
        self._queryNorm.wrappedValue = Wan2RMSNorm(dimensions: attentionDimensions, epsilon: epsilon)
        self._keyNorm.wrappedValue = Wan2RMSNorm(dimensions: attentionDimensions, epsilon: epsilon)
    }

    func callAsFunction(
        _ input: MLXArray,
        conditioning: Wan2ProjectiveCameraConditioning
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let dtype = query.weight.dtype
        let typed = input.asType(dtype)
        var q = queryNorm(query(typed)).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        var k = keyNorm(key(typed)).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        var v = value(typed).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        let transforms = Wan2ProjectivePositionEncoding.prepare(
            conditioning: conditioning,
            batchSize: batch,
            dtype: dtype
        )
        q = Wan2ProjectivePositionEncoding.apply(q, matrices: transforms.query, cameraFrames: conditioning.frameCount)
        k = Wan2ProjectivePositionEncoding.apply(k, matrices: transforms.keyValue, cameraFrames: conditioning.frameCount)
        v = Wan2ProjectivePositionEncoding.apply(v, matrices: transforms.keyValue, cameraFrames: conditioning.frameCount)
        var attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        attended = Wan2ProjectivePositionEncoding.apply(
            attended,
            matrices: transforms.output,
            cameraFrames: conditioning.frameCount
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDimension))
    }
}

private struct Wan2Matrix4: Hashable {
    let values: [Float]

    init(_ values: [Float]) {
        precondition(values.count == 16)
        self.values = values
    }

    static let identity = Wan2Matrix4([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])

    static func diagonal(_ x: Float, _ y: Float, _ z: Float, _ w: Float) -> Wan2Matrix4 {
        Wan2Matrix4([
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, w
        ])
    }

    static func euler(roll: Float, pitch: Float, yaw: Float) -> Wan2Matrix4 {
        let sineX = sin(roll)
        let cosineX = cos(roll)
        let sineY = sin(pitch)
        let cosineY = cos(pitch)
        let sineZ = sin(yaw)
        let cosineZ = cos(yaw)
        let rotationX = Wan2Matrix4([
            1, 0, 0, 0,
            0, cosineX, -sineX, 0,
            0, sineX, cosineX, 0,
            0, 0, 0, 1
        ])
        let rotationY = Wan2Matrix4([
            cosineY, 0, sineY, 0,
            0, 1, 0, 0,
            -sineY, 0, cosineY, 0,
            0, 0, 0, 1
        ])
        let rotationZ = Wan2Matrix4([
            cosineZ, -sineZ, 0, 0,
            sineZ, cosineZ, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ])
        return rotationZ * rotationY * rotationX
    }

    static func cameraRotation(
        pitchDegrees: Float,
        yawDegrees: Float,
        rollDegrees: Float
    ) -> Wan2Matrix4 {
        let pitch = pitchDegrees * .pi / 180
        let yaw = yawDegrees * .pi / 180
        let roll = rollDegrees * .pi / 180
        let cosineYaw = cos(yaw * 0.5)
        let sineYaw = sin(yaw * 0.5)
        let cosinePitch = cos(pitch * 0.5)
        let sinePitch = sin(pitch * 0.5)
        let cosineRoll = cos(roll * 0.5)
        let sineRoll = sin(roll * 0.5)
        let quaternionW = cosineYaw * cosinePitch * cosineRoll + sineYaw * sinePitch * sineRoll
        let quaternionX = cosineYaw * sinePitch * cosineRoll + sineYaw * cosinePitch * sineRoll
        let quaternionY = sineYaw * cosinePitch * cosineRoll - cosineYaw * sinePitch * sineRoll
        let quaternionZ = cosineYaw * cosinePitch * sineRoll - sineYaw * sinePitch * cosineRoll
        return Wan2Matrix4([
            1 - 2 * (quaternionY * quaternionY + quaternionZ * quaternionZ),
            2 * (quaternionX * quaternionY - quaternionW * quaternionZ),
            2 * (quaternionX * quaternionZ + quaternionW * quaternionY),
            0,
            2 * (quaternionX * quaternionY + quaternionW * quaternionZ),
            1 - 2 * (quaternionX * quaternionX + quaternionZ * quaternionZ),
            2 * (quaternionY * quaternionZ - quaternionW * quaternionX),
            0,
            2 * (quaternionX * quaternionZ - quaternionW * quaternionY),
            2 * (quaternionY * quaternionZ + quaternionW * quaternionX),
            1 - 2 * (quaternionX * quaternionX + quaternionY * quaternionY),
            0,
            0, 0, 0, 1
        ])
    }

    static func * (left: Wan2Matrix4, right: Wan2Matrix4) -> Wan2Matrix4 {
        var output = Array(repeating: Float(0), count: 16)
        for row in 0..<4 {
            for column in 0..<4 {
                var value = Float(0)
                for index in 0..<4 {
                    value += left.values[row * 4 + index] * right.values[index * 4 + column]
                }
                output[row * 4 + column] = value
            }
        }
        return Wan2Matrix4(output)
    }

    func transposed() -> Wan2Matrix4 {
        Wan2Matrix4((0..<16).map { values[($0 % 4) * 4 + ($0 / 4)] })
    }

    func withTranslation(_ translation: SIMD3<Float>) -> Wan2Matrix4 {
        var output = values
        output[3] = translation.x
        output[7] = translation.y
        output[11] = translation.z
        return Wan2Matrix4(output)
    }

    func transformDirection(_ direction: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            values[0] * direction.x + values[1] * direction.y + values[2] * direction.z,
            values[4] * direction.x + values[5] * direction.y + values[6] * direction.z,
            values[8] * direction.x + values[9] * direction.y + values[10] * direction.z
        )
    }

    func invertedRigid() -> Wan2Matrix4 {
        let rotationTranspose = Wan2Matrix4([
            values[0], values[4], values[8], 0,
            values[1], values[5], values[9], 0,
            values[2], values[6], values[10], 0,
            0, 0, 0, 1
        ])
        let translation = SIMD3(values[3], values[7], values[11])
        return rotationTranspose.withTranslation(-rotationTranspose.transformDirection(translation))
    }
}
