import Foundation

/// Normalized `camera_pose` actions consumed directly by Cosmos3-Edge.
///
/// NVIDIA defines each row as the relative 9D pose delta between consecutive
/// visual states: translation plus column-based rotation-6D. The bundled
/// 60x9 trajectory is an exact upstream parity sample, not a canonical forward
/// path. Semantic controls preserve its per-step translation magnitudes while
/// compiling explicit camera-relative axes.
public enum Cosmos3CameraModelSpaceTrajectory {
    public static let nvidiaFrameworkRevision =
        "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
    public static let modelRevision =
        "6f58f6b4c91288838e60b6bcb2cc45d997e961de"
    public static let referenceSHA256 =
        "547b977ccd6d435d579ba83e82e58618d0bebf2e8e7d8d577bf2e9c8883c5595"
    public static let actionDimension = 9
    public static let referenceFrameRate = 30

    public static let forwardReference: [[Float]] = {
        guard let url = Bundle.module.url(
            forResource: "camera-pose-forward-60",
            withExtension: "json"
        ) else {
            preconditionFailure("Missing bundled Cosmos3 camera-pose reference trajectory.")
        }
        do {
            let data = try Data(contentsOf: url)
            let actions = try JSONDecoder().decode([[Float]].self, from: data)
            precondition(actions.count == 60)
            precondition(actions.allSatisfy { $0.count == actionDimension })
            return actions
        } catch {
            preconditionFailure("Invalid bundled Cosmos3 camera-pose trajectory: \(error)")
        }
    }()

    public static let identityRotation6D: [Float] = [1, 0, 0, 0, 1, 0]

    /// Compiles the semantic forward axis while retaining the exact magnitude
    /// envelope of NVIDIA's published camera-pose sample.
    public static func forward(actionCount: Int) -> [[Float]] {
        translated(direction: [0, 0, 1], scale: 1, actionCount: actionCount)
    }

    public static func translated(
        direction: [Float],
        scale: Float,
        actionCount: Int
    ) -> [[Float]] {
        precondition(direction.count == 3)
        precondition(actionCount > 0)
        precondition(scale.isFinite && scale >= 0)
        let length = sqrt(direction.reduce(Float.zero) { $0 + $1 * $1 })
        precondition(length > 0)
        let unit = direction.map { $0 / length }
        let reference = fitted(forwardReference, actionCount: actionCount)
        return reference.map { action in
            let magnitude = sqrt(action[0] * action[0] + action[1] * action[1] + action[2] * action[2])
            return unit.map { $0 * magnitude * scale } + identityRotation6D
        }
    }

    public static func stationary(actionCount: Int) -> [[Float]] {
        precondition(actionCount > 0)
        return Array(
            repeating: [0, 0, 0] + identityRotation6D,
            count: actionCount
        )
    }

    public static func fitted(_ actions: [[Float]], actionCount: Int) -> [[Float]] {
        precondition(actionCount > 0)
        precondition(!actions.isEmpty)
        precondition(actions.allSatisfy { $0.count == actionDimension })
        var fitted = Array(actions.prefix(actionCount))
        while fitted.count < actionCount {
            fitted.append(fitted[fitted.count - 1])
        }
        return fitted
    }
}
