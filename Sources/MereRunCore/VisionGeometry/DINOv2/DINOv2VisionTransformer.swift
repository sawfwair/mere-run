import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// PyTorch uses the caller-supplied scale factor for half-pixel sampling even
/// when flooring that scale to an integer output size. MLX's general-purpose
/// `Upsample` centers its grid from the floored size instead, which loses
/// DINOv2's historical +0.1 interpolation offset.
func dinoV2PyTorchInterpolationCoordinates(
    sourceSize: Int,
    outputSize: Int,
    offset: Float
) -> MLXArray {
    precondition(sourceSize > 0 && outputSize > 0)
    let scale = (Float(outputSize) + offset) / Float(sourceSize)
    return (MLX.arange(outputSize, dtype: .float32) + 0.5) / scale - 0.5
}

private func dinoV2CubicWeight(_ distance: MLXArray) -> MLXArray {
    // Matches PyTorch bicubic interpolation with antialias=false.
    let coefficient = Float(-0.75)
    let absoluteDistance = abs(distance)
    let near = ((coefficient + 2) * absoluteDistance - (coefficient + 3))
        * absoluteDistance * absoluteDistance + 1
    let far = (((absoluteDistance - 5) * absoluteDistance + 8) * absoluteDistance - 4)
        * coefficient
    return MLX.where(absoluteDistance .<= MLXArray(Float(1)), near, far)
}

func dinoV2PyTorchBicubicResize(
    _ input: MLXArray,
    outputHeight: Int,
    outputWidth: Int,
    offset: Float
) -> MLXArray {
    precondition(input.ndim == 4)
    let sourceHeight = input.dim(1)
    let sourceWidth = input.dim(2)
    let heightCoordinates = dinoV2PyTorchInterpolationCoordinates(
        sourceSize: sourceHeight,
        outputSize: outputHeight,
        offset: offset
    )
    let widthCoordinates = dinoV2PyTorchInterpolationCoordinates(
        sourceSize: sourceWidth,
        outputSize: outputWidth,
        offset: offset
    )

    var vertical = MLX.zeros(
        [input.dim(0), outputHeight, sourceWidth, input.dim(3)],
        dtype: input.dtype
    )
    let heightBase = floor(heightCoordinates)
    for neighborOffset in -1...2 {
        let neighbor = heightBase + Float(neighborOffset)
        let index = MLX.clip(neighbor, min: 0, max: Float(sourceHeight - 1)).asType(.int32)
        let weight = dinoV2CubicWeight(heightCoordinates - neighbor).reshaped(1, outputHeight, 1, 1)
        vertical = vertical + input[0..., index, 0..., 0...] * weight
    }

    var output = MLX.zeros(
        [input.dim(0), outputHeight, outputWidth, input.dim(3)],
        dtype: input.dtype
    )
    let widthBase = floor(widthCoordinates)
    for neighborOffset in -1...2 {
        let neighbor = widthBase + Float(neighborOffset)
        let index = MLX.clip(neighbor, min: 0, max: Float(sourceWidth - 1)).asType(.int32)
        let weight = dinoV2CubicWeight(widthCoordinates - neighbor).reshaped(1, 1, outputWidth, 1)
        output = output + vertical[0..., 0..., index, 0...] * weight
    }
    return output
}

public struct DINOv2Configuration: Equatable, Sendable {
    public let hiddenSize: Int
    public let layerCount: Int
    public let headCount: Int
    public let intermediateSize: Int
    public let patchSize: Int
    public let positionGridSize: Int
    public let layerNormEpsilon: Float
    public let positionInterpolationOffset: Float

    public init(
        hiddenSize: Int = 384,
        layerCount: Int = 12,
        headCount: Int = 6,
        intermediateSize: Int = 1_536,
        patchSize: Int = 14,
        positionGridSize: Int = 37,
        layerNormEpsilon: Float = 1e-6,
        positionInterpolationOffset: Float = 0.1
    ) {
        precondition(hiddenSize.isMultiple(of: headCount))
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.positionGridSize = positionGridSize
        self.layerNormEpsilon = layerNormEpsilon
        self.positionInterpolationOffset = positionInterpolationOffset
    }

    public static let smallPatch14 = DINOv2Configuration()
}

public struct DINOv2VisionOutput {
    public let featuresByLayer: [Int: MLXArray]
    public let classToken: MLXArray
    public let gridHeight: Int
    public let gridWidth: Int
}

final class DINOv2Attention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var projection: Linear

    init(configuration: DINOv2Configuration) {
        self.headCount = configuration.headCount
        self.headDimension = configuration.hiddenSize / configuration.headCount
        self.scale = 1 / sqrt(Float(headDimension))
        self._qkv.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize * 3, bias: true)
        self._projection.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
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
        return projection(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, headCount * headDimension))
    }
}

final class DINOv2MLP: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear

    init(configuration: DINOv2Configuration) {
        self._first.wrappedValue = Linear(configuration.hiddenSize, configuration.intermediateSize, bias: true)
        self._second.wrappedValue = Linear(configuration.intermediateSize, configuration.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(gelu(first(input)))
    }
}

final class DINOv2LayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(hiddenSize: Int) {
        self._gamma.wrappedValue = MLX.ones([hiddenSize], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * gamma.asType(input.dtype)
    }
}

final class DINOv2Block: Module {
    @ModuleInfo(key: "norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: DINOv2Attention
    @ModuleInfo(key: "ls1") var firstScale: DINOv2LayerScale
    @ModuleInfo(key: "norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: DINOv2MLP
    @ModuleInfo(key: "ls2") var secondScale: DINOv2LayerScale

    init(configuration: DINOv2Configuration) {
        self._firstNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
        self._attention.wrappedValue = DINOv2Attention(configuration: configuration)
        self._firstScale.wrappedValue = DINOv2LayerScale(hiddenSize: configuration.hiddenSize)
        self._secondNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
        self._mlp.wrappedValue = DINOv2MLP(configuration: configuration)
        self._secondScale.wrappedValue = DINOv2LayerScale(hiddenSize: configuration.hiddenSize)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + firstScale(attention(firstNorm(input)))
        return attended + secondScale(mlp(secondNorm(attended)))
    }
}

final class DINOv2PatchEmbed: Module {
    @ModuleInfo(key: "proj") var projection: Conv2d

    init(configuration: DINOv2Configuration) {
        self._projection.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: configuration.hiddenSize,
            kernelSize: IntOrPair(configuration.patchSize),
            stride: IntOrPair(configuration.patchSize),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        projection(input)
    }
}

public final class DINOv2VisionTransformer: Module {
    public let configuration: DINOv2Configuration

    @ParameterInfo(key: "cls_token") var classToken: MLXArray
    @ParameterInfo(key: "class_position") var classPosition: MLXArray
    @ParameterInfo(key: "patch_position") var patchPosition: MLXArray
    @ModuleInfo(key: "patch_embed") var patchEmbed: DINOv2PatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [DINOv2Block]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    public init(configuration: DINOv2Configuration = .smallPatch14) {
        self.configuration = configuration
        self._classToken.wrappedValue = MLX.zeros([1, 1, configuration.hiddenSize], dtype: .float32)
        self._classPosition.wrappedValue = MLX.zeros([1, 1, configuration.hiddenSize], dtype: .float32)
        self._patchPosition.wrappedValue = MLX.zeros(
            [1, configuration.positionGridSize * configuration.positionGridSize, configuration.hiddenSize],
            dtype: .float32
        )
        self._patchEmbed.wrappedValue = DINOv2PatchEmbed(configuration: configuration)
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            DINOv2Block(configuration: configuration)
        }
        self._norm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
        super.init()
    }

    /// Input is NHWC RGB, already ImageNet-normalized.
    public func callAsFunction(
        _ input: MLXArray,
        intermediateLayers: Set<Int>
    ) -> DINOv2VisionOutput {
        precondition(input.ndim == 4 && input.dim(3) == 3)
        precondition(intermediateLayers.allSatisfy { $0 >= 0 && $0 < configuration.layerCount })
        let batch = input.dim(0)
        let patches = patchEmbed(input)
        let gridHeight = patches.dim(1)
        let gridWidth = patches.dim(2)
        let flattened = patches.reshaped(batch, gridHeight * gridWidth, configuration.hiddenSize)
        let cls = MLX.broadcast(classToken.asType(input.dtype), to: [batch, 1, configuration.hiddenSize])
        var hidden = MLX.concatenated([cls, flattened], axis: 1)
        hidden = hidden + interpolatedPosition(
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            batch: batch,
            dtype: hidden.dtype
        )

        var features: [Int: MLXArray] = [:]
        var latestClassToken = hidden[0..., 0, 0...]
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden)
            if intermediateLayers.contains(index) {
                let normalized = norm(hidden)
                latestClassToken = normalized[0..., 0, 0...]
                features[index] = normalized[0..., 1..., 0...]
                    .reshaped(batch, gridHeight, gridWidth, configuration.hiddenSize)
            }
        }
        return DINOv2VisionOutput(
            featuresByLayer: features,
            classToken: latestClassToken,
            gridHeight: gridHeight,
            gridWidth: gridWidth
        )
    }

    private func interpolatedPosition(
        gridHeight: Int,
        gridWidth: Int,
        batch: Int,
        dtype: DType
    ) -> MLXArray {
        let source = configuration.positionGridSize
        let patchGrid = patchPosition
            .reshaped(1, source, source, configuration.hiddenSize)
            .asType(dtype)
        let resized: MLXArray
        if gridHeight == source && gridWidth == source {
            resized = patchGrid
        } else {
            resized = dinoV2PyTorchBicubicResize(
                patchGrid.asType(.float32),
                outputHeight: gridHeight,
                outputWidth: gridWidth,
                offset: configuration.positionInterpolationOffset
            ).asType(dtype)
        }
        precondition(resized.dim(1) == gridHeight && resized.dim(2) == gridWidth)
        let patch = resized.reshaped(1, gridHeight * gridWidth, configuration.hiddenSize)
        let position = MLX.concatenated([classPosition.asType(dtype), patch], axis: 1)
        return MLX.broadcast(position, to: [batch, 1 + gridHeight * gridWidth, configuration.hiddenSize])
    }
}
