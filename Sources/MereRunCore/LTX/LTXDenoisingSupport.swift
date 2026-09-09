import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func toDenoised(
    noisy: MLXArray,
    velocity: MLXArray,
    sigma: Float
) -> MLXArray {
    noisy - MLXArray(sigma).asType(velocity.dtype) * velocity
}

func tensorStatsString(_ x: MLXArray) -> String {
    let x32 = x.asType(.float32)
    let mean = MLX.mean(x32).item(Float.self)
    let std = MLX.std(x32).item(Float.self)
    let minVal = MLX.min(x32).item(Float.self)
    let maxVal = MLX.max(x32).item(Float.self)
    return String(
        format: "shape=%@ mean=%.6f std=%.6f min=%.6f max=%.6f",
        x.shape.description,
        mean,
        std,
        minVal,
        maxVal
    )
}
