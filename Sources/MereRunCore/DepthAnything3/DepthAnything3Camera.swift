import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

private final class DA3CameraAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var projection: Linear

    init(hiddenSize: Int, headCount: Int) {
        self.headCount = headCount
        self.headDimension = hiddenSize / headCount
        self.scale = 1 / sqrt(Float(headDimension))
        self._qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: true)
        self._projection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let projected = qkv(input).reshaped(batch, sequence, 3, headCount, headDimension)
        let query = projected[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        let key = projected[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let value = projected[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: .none
        )
        return projection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, headCount * headDimension)
        )
    }
}

private final class DA3CameraMLP: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear

    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        self._first.wrappedValue = Linear(inputSize, hiddenSize, bias: true)
        self._second.wrappedValue = Linear(hiddenSize, outputSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(gelu(first(input)))
    }
}

private final class DA3CameraLayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(hiddenSize: Int, initialValue: Float) {
        self._gamma.wrappedValue = MLX.ones([hiddenSize], dtype: .float32) * initialValue
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * gamma.asType(input.dtype)
    }
}

private final class DA3CameraBlock: Module {
    @ModuleInfo(key: "norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: DA3CameraAttention
    @ModuleInfo(key: "ls1") var firstScale: DA3CameraLayerScale
    @ModuleInfo(key: "norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: DA3CameraMLP
    @ModuleInfo(key: "ls2") var secondScale: DA3CameraLayerScale

    init(configuration: DepthAnything3Configuration) {
        self._firstNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.headLayerNormEpsilon
        )
        self._attention.wrappedValue = DA3CameraAttention(
            hiddenSize: configuration.hiddenSize,
            headCount: configuration.cameraEncoderHeadCount
        )
        self._firstScale.wrappedValue = DA3CameraLayerScale(
            hiddenSize: configuration.hiddenSize,
            initialValue: configuration.cameraEncoderLayerScale
        )
        self._secondNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.headLayerNormEpsilon
        )
        self._mlp.wrappedValue = DA3CameraMLP(
            inputSize: configuration.hiddenSize,
            hiddenSize: configuration.hiddenSize * 4,
            outputSize: configuration.hiddenSize
        )
        self._secondScale.wrappedValue = DA3CameraLayerScale(
            hiddenSize: configuration.hiddenSize,
            initialValue: configuration.cameraEncoderLayerScale
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + firstScale(attention(firstNorm(input)))
        return attended + secondScale(mlp(secondNorm(attended)))
    }
}

final class DepthAnything3CameraEncoder: Module {
    let configuration: DepthAnything3Configuration

    @ModuleInfo(key: "trunk") private var trunk: [DA3CameraBlock]
    @ModuleInfo(key: "token_norm") var tokenNorm: LayerNorm
    @ModuleInfo(key: "trunk_norm") var trunkNorm: LayerNorm
    @ModuleInfo(key: "pose_branch") private var poseBranch: DA3CameraMLP

    init(configuration: DepthAnything3Configuration) {
        self.configuration = configuration
        self._trunk.wrappedValue = (0..<configuration.cameraEncoderDepth).map { _ in
            DA3CameraBlock(configuration: configuration)
        }
        self._tokenNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.headLayerNormEpsilon
        )
        self._trunkNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.headLayerNormEpsilon
        )
        self._poseBranch.wrappedValue = DA3CameraMLP(
            inputSize: 9,
            hiddenSize: configuration.hiddenSize / 2,
            outputSize: configuration.hiddenSize
        )
        super.init()
    }

    func callAsFunction(
        _ conditioning: DepthAnything3CameraConditioning,
        imageHeight: Int,
        imageWidth: Int
    ) -> MLXArray {
        let pose = poseEncoding(
            extrinsics: conditioning.extrinsics,
            intrinsics: conditioning.intrinsics,
            imageHeight: imageHeight,
            imageWidth: imageWidth
        )
        var hidden = tokenNorm(poseBranch(pose))
        for block in trunk { hidden = block(hidden) }
        return trunkNorm(hidden)
    }

    /// Camera conditioning is tiny compared with image features. Performing
    /// this rigid-transform conversion eagerly on CPU avoids a large branchy
    /// quaternion graph while preserving the authoritative float32 formulas.
    private func poseEncoding(
        extrinsics: MLXArray,
        intrinsics: MLXArray,
        imageHeight: Int,
        imageWidth: Int
    ) -> MLXArray {
        let ext = extrinsics.asType(.float32)
        let ixt = intrinsics.asType(.float32)
        MLX.eval(ext, ixt)
        let extValues = ext.asArray(Float.self)
        let ixtValues = ixt.asArray(Float.self)
        let batch = ext.dim(0)
        let views = ext.dim(1)
        var output: [Float] = []
        output.reserveCapacity(batch * views * 9)
        for index in 0..<(batch * views) {
            let extBase = index * 16
            let r = [
                extValues[extBase], extValues[extBase + 1], extValues[extBase + 2],
                extValues[extBase + 4], extValues[extBase + 5], extValues[extBase + 6],
                extValues[extBase + 8], extValues[extBase + 9], extValues[extBase + 10],
            ]
            let t = [extValues[extBase + 3], extValues[extBase + 7], extValues[extBase + 11]]
            let c2wRotation = [
                r[0], r[3], r[6],
                r[1], r[4], r[7],
                r[2], r[5], r[8],
            ]
            let c2wTranslation = [
                -(c2wRotation[0] * t[0] + c2wRotation[1] * t[1] + c2wRotation[2] * t[2]),
                -(c2wRotation[3] * t[0] + c2wRotation[4] * t[1] + c2wRotation[5] * t[2]),
                -(c2wRotation[6] * t[0] + c2wRotation[7] * t[1] + c2wRotation[8] * t[2]),
            ]
            let quaternion = matrixToQuaternionXYZW(c2wRotation)
            let ixtBase = index * 9
            let fy = ixtValues[ixtBase + 4]
            let fx = ixtValues[ixtBase]
            let fovHeight = 2 * atan((Float(imageHeight) / 2) / fy)
            let fovWidth = 2 * atan((Float(imageWidth) / 2) / fx)
            output.append(contentsOf: c2wTranslation)
            output.append(contentsOf: quaternion)
            output.append(contentsOf: [fovHeight, fovWidth])
        }
        return MLXArray(output).reshaped(batch, views, 9)
    }

    private func matrixToQuaternionXYZW(_ matrix: [Float]) -> [Float] {
        let m00 = matrix[0], m01 = matrix[1], m02 = matrix[2]
        let m10 = matrix[3], m11 = matrix[4], m12 = matrix[5]
        let m20 = matrix[6], m21 = matrix[7], m22 = matrix[8]
        let qAbs = [
            sqrt(max(0, 1 + m00 + m11 + m22)),
            sqrt(max(0, 1 + m00 - m11 - m22)),
            sqrt(max(0, 1 - m00 + m11 - m22)),
            sqrt(max(0, 1 - m00 - m11 + m22)),
        ]
        let candidates = [
            [qAbs[0] * qAbs[0], m21 - m12, m02 - m20, m10 - m01],
            [m21 - m12, qAbs[1] * qAbs[1], m10 + m01, m02 + m20],
            [m02 - m20, m10 + m01, qAbs[2] * qAbs[2], m12 + m21],
            [m10 - m01, m20 + m02, m21 + m12, qAbs[3] * qAbs[3]],
        ]
        let selected = qAbs.enumerated().max { $0.element < $1.element }!.offset
        let denominator = 2 * max(qAbs[selected], 0.1)
        let rijk = candidates[selected].map { $0 / denominator }
        var xyzw = [rijk[1], rijk[2], rijk[3], rijk[0]]
        if xyzw[3] < 0 { xyzw = xyzw.map(-) }
        return xyzw
    }
}

private final class DA3CameraDecoderBackbone: Module {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "second") var second: Linear

    init(channels: Int) {
        self._first.wrappedValue = Linear(channels, channels, bias: true)
        self._second.wrappedValue = Linear(channels, channels, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        relu(second(relu(first(input))))
    }
}

private final class DA3CameraFOVHead: Module {
    @ModuleInfo(key: "projection") var projection: Linear

    init(channels: Int) {
        self._projection.wrappedValue = Linear(channels, 2, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        relu(projection(input))
    }
}

struct DepthAnything3CameraPrediction {
    let extrinsics: MLXArray
    let intrinsics: MLXArray
}

final class DepthAnything3CameraDecoder: Module {
    let inputChannels: Int

    @ModuleInfo(key: "backbone") private var backbone: DA3CameraDecoderBackbone
    @ModuleInfo(key: "fc_t") var translation: Linear
    @ModuleInfo(key: "fc_qvec") var quaternion: Linear
    @ModuleInfo(key: "fc_fov") private var fieldOfView: DA3CameraFOVHead

    init(configuration: DepthAnything3Configuration) {
        self.inputChannels = configuration.hiddenSize * 2
        self._backbone.wrappedValue = DA3CameraDecoderBackbone(channels: inputChannels)
        self._translation.wrappedValue = Linear(inputChannels, 3, bias: true)
        self._quaternion.wrappedValue = Linear(inputChannels, 4, bias: true)
        self._fieldOfView.wrappedValue = DA3CameraFOVHead(channels: inputChannels)
        super.init()
    }

    func callAsFunction(
        _ cameraToken: MLXArray,
        imageHeight: Int,
        imageWidth: Int
    ) -> DepthAnything3CameraPrediction {
        precondition(cameraToken.ndim == 3 && cameraToken.dim(2) == inputChannels)
        let batch = cameraToken.dim(0)
        let views = cameraToken.dim(1)
        let hidden = backbone(cameraToken.reshaped(batch * views, inputChannels)).asType(.float32)
        let translation = translation(hidden).reshaped(batch, views, 3)
        let quaternion = quaternion(hidden).reshaped(batch, views, 4)
        let fov = fieldOfView(hidden).reshaped(batch, views, 2)
        let cameraToWorldRotation = quaternionToMatrix(quaternion)
        let worldToCameraRotation = cameraToWorldRotation.transposed(0, 1, 3, 2)
        let worldToCameraTranslation = -MLX.matmul(
            worldToCameraRotation,
            translation.expandedDimensions(axis: -1)
        )
        let extrinsics = MLX.concatenated(
            [worldToCameraRotation, worldToCameraTranslation],
            axis: -1
        )

        let fovHeight = fov[0..., 0..., 0]
        let fovWidth = fov[0..., 0..., 1]
        let fy = (Float(imageHeight) / 2)
            / MLX.maximum(MLX.tan(fovHeight / 2), MLXArray(Float(1e-6)))
        let fx = (Float(imageWidth) / 2)
            / MLX.maximum(MLX.tan(fovWidth / 2), MLXArray(Float(1e-6)))
        let zero = MLX.zeros(like: fx)
        let one = MLX.ones(like: fx)
        let cx = MLX.ones(like: fx) * (Float(imageWidth) / 2)
        let cy = MLX.ones(like: fy) * (Float(imageHeight) / 2)
        let row0 = MLX.stacked([fx, zero, cx], axis: -1)
        let row1 = MLX.stacked([zero, fy, cy], axis: -1)
        let row2 = MLX.stacked([zero, zero, one], axis: -1)
        let intrinsics = MLX.stacked([row0, row1, row2], axis: -2)
        return DepthAnything3CameraPrediction(extrinsics: extrinsics, intrinsics: intrinsics)
    }

    private func quaternionToMatrix(_ quaternion: MLXArray) -> MLXArray {
        let i = quaternion[0..., 0..., 0]
        let j = quaternion[0..., 0..., 1]
        let k = quaternion[0..., 0..., 2]
        let r = quaternion[0..., 0..., 3]
        let twoScale: MLXArray = 2 / MLX.sum(quaternion * quaternion, axis: -1)
        let one = MLX.ones(like: twoScale)
        let values = [
            one - twoScale * (j * j + k * k),
            twoScale * (i * j - k * r),
            twoScale * (i * k + j * r),
            twoScale * (i * j + k * r),
            one - twoScale * (i * i + k * k),
            twoScale * (j * k - i * r),
            twoScale * (i * k - j * r),
            twoScale * (j * k + i * r),
            one - twoScale * (i * i + j * j),
        ]
        return MLX.stacked(values, axis: -1)
            .reshaped(quaternion.dim(0), quaternion.dim(1), 3, 3)
    }
}
