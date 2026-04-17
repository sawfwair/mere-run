import MLX
import MLXNN

public enum SharpMath {
    public static func inverseSigmoid(_ x: MLXArray) -> MLXArray {
        MLX.log(x / (1.0 - x))
    }

    public static func inverseSoftplus(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let clamped = MLX.clip(x, min: eps)
        let sigmoidNeg = sigmoid(-clamped)
        let expTerm = sigmoidNeg / (1.0 - sigmoidNeg)
        return clamped + MLX.log(1.0 - expTerm)
    }

    public static func activationForward(_ type: SharpActivationType, _ x: MLXArray) -> MLXArray {
        switch type {
        case .linear:
            return x
        case .exp:
            return MLX.exp(x)
        case .sigmoid:
            return sigmoid(x)
        case .softplus:
            return softplus(x)
        case .reluWithPushback:
            return MLX.maximum(x, 0.0)
        case .hardSigmoidWithPushback:
            return MLX.clip((x / 6.0) + 0.5, min: 0.0, max: 1.0)
        }
    }

    public static func activationInverse(_ type: SharpActivationType, _ x: MLXArray) -> MLXArray {
        switch type {
        case .linear:
            return x
        case .exp:
            return MLX.log(x)
        case .sigmoid:
            return inverseSigmoid(x)
        case .softplus:
            return inverseSoftplus(x)
        case .reluWithPushback:
            return x
        case .hardSigmoidWithPushback:
            return (6.0 * x) - 3.0
        }
    }
}
