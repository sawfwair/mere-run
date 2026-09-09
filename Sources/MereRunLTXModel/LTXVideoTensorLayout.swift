import Foundation
import MLX
import MLXFast
import MLXNN

package func blockIndex<T: AnyObject>(of object: T, in array: [T]) -> Int? {
    for (i, element) in array.enumerated() {
        if object === element {
            return i
        }
    }
    return nil
}

package func depthToSpace3D(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
    let b = x.dim(0)
    let packedC = x.dim(1)
    let d = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    let st = stride.0
    let sh = stride.1
    let sw = stride.2

    let c = packedC / (st * sh * sw)
    var out = x.reshaped(b, c, st, sh, sw, d, h, w)
    out = out.transposed(0, 1, 5, 2, 6, 3, 7, 4)
    return out.reshaped(b, c, d * st, h * sh, w * sw)
}

package func spaceToDepth3D(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let d = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    let st = stride.0
    let sh = stride.1
    let sw = stride.2

    var out = x.reshaped(b, c, d / st, st, h / sh, sh, w / sw, sw)
    out = out.transposed(0, 1, 3, 5, 7, 2, 4, 6)
    return out.reshaped(b, c * st * sh * sw, d / st, h / sh, w / sw)
}

package func patchify3D(_ x: MLXArray, patchSizeHW: Int, patchSizeT: Int) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let f = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let newF = f / patchSizeT
    let newH = h / patchSizeHW
    let newW = w / patchSizeHW
    var out = x.reshaped(b, c, newF, patchSizeT, newH, patchSizeHW, newW, patchSizeHW)
    out = out.transposed(0, 1, 3, 7, 5, 2, 4, 6)
    return out.reshaped(b, c * patchSizeT * patchSizeHW * patchSizeHW, newF, newH, newW)
}

package func unpatchify3D(_ x: MLXArray, patchSizeHW: Int, patchSizeT: Int) -> MLXArray {
    let b = x.dim(0)
    let packedC = x.dim(1)
    let f = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let c = packedC / (patchSizeHW * patchSizeHW * patchSizeT)
    var out = x.reshaped(b, c, patchSizeT, patchSizeHW, patchSizeHW, f, h, w)
    out = out.transposed(0, 1, 5, 2, 6, 4, 7, 3)
    return out.reshaped(b, c, f * patchSizeT, h * patchSizeHW, w * patchSizeHW)
}

package func pixelNormChannels(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    let denom = MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + MLXArray(eps))
    return x / denom
}

package func postprocessDecodedVideo(_ decoded: MLXArray) -> MLXArray {
    var video = decoded[0, 0..., 0..., 0..., 0...]
    video = video.transposed(1, 2, 3, 0)
    video = MLX.clip((video + MLXArray(1.0)) / MLXArray(2.0), min: MLXArray(0.0), max: MLXArray(1.0))
    return (video * MLXArray(255.0)).asType(.uint8)
}
