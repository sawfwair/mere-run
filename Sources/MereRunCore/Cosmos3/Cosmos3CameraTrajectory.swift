import Foundation

/// Normalized `camera_pose` actions consumed directly by Cosmos3-Edge.
///
/// These values are model-space poses, not meters or per-frame physical deltas.
/// The reference path is NVIDIA's released 60x9 camera trajectory at the pinned
/// Cosmos3-Edge revision. Keeping the fixture intact gives the native Swift
/// runtime an exact parity input and avoids inventing a camera normalization.
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

    /// NVIDIA's autoregressive camera canary continues translation by the
    /// average displacement of the complete published path. The divisor is
    /// the action count (60), because the next chunk starts one action after
    /// the terminal pose and spans another 60 actions.
    public static let forwardContinuationTranslation: [Float] = {
        let first = forwardReference[0]
        let last = forwardReference[forwardReference.count - 1]
        return (0..<3).map {
            (last[$0] - first[$0]) / Float(forwardReference.count)
        }
    }()

    /// Returns the published path at its original cadence. Short chunks use a
    /// prefix, matching NVIDIA's 16-action camera canary. Longer chunks continue
    /// the final translation trend while keeping the terminal rotation.
    public static func forward(actionCount: Int) -> [[Float]] {
        precondition(actionCount > 0)
        let reference = forwardReference
        guard actionCount > reference.count else {
            return Array(reference.prefix(actionCount))
        }

        var result = reference
        let first = reference[0]
        let last = reference[reference.count - 1]
        let denominator = Float(reference.count - 1)
        let delta = (0..<3).map { (last[$0] - first[$0]) / denominator }
        while result.count < actionCount {
            var next = result[result.count - 1]
            for axis in 0..<3 {
                next[axis] += delta[axis]
            }
            result.append(next)
        }
        return result
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
