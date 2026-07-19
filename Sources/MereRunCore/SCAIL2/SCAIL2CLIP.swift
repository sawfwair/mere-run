import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public struct SCAIL2CLIPVisionConfiguration: Hashable, Sendable {
    public let imageSize: Int
    public let patchSize: Int
    public let hiddenSize: Int
    public let feedForwardSize: Int
    public let headCount: Int
    public let layerCount: Int
    public let epsilon: Float

    public init(
        imageSize: Int = 224,
        patchSize: Int = 14,
        hiddenSize: Int = 1_280,
        feedForwardSize: Int = 5_120,
        headCount: Int = 16,
        layerCount: Int = 31,
        epsilon: Float = 1e-5
    ) {
        precondition(imageSize.isMultiple(of: patchSize))
        precondition(hiddenSize.isMultiple(of: headCount))
        precondition(layerCount > 0 && layerCount <= 32)
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hiddenSize = hiddenSize
        self.feedForwardSize = feedForwardSize
        self.headCount = headCount
        self.layerCount = layerCount
        self.epsilon = epsilon
    }

    public var tokenCount: Int {
        let side = imageSize / patchSize
        return side * side + 1
    }
}

final class SCAIL2CLIPAttention: Module {
    let dimensions: Int
    let heads: Int
    let headDimensions: Int
    @ModuleInfo(key: "to_qkv") var inputProjection: Linear
    @ModuleInfo(key: "proj") var outputProjection: Linear

    init(dimensions: Int, heads: Int) {
        self.dimensions = dimensions
        self.heads = heads
        self.headDimensions = dimensions / heads
        self._inputProjection.wrappedValue = Linear(dimensions, dimensions * 3, bias: true)
        self._outputProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let qkv = inputProjection(input)
            .reshaped(batch, sequence, 3, heads, headDimensions)
        let query = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        let key = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let value = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / Float(headDimensions).squareRoot(),
            mask: .none
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, dimensions)
        )
    }
}

final class SCAIL2CLIPMLP: Module {
    @ModuleInfo(key: "layer_0") var input: Linear
    @ModuleInfo(key: "layer_2") var output: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._input.wrappedValue = Linear(dimensions, hiddenDimensions, bias: true)
        self._output.wrappedValue = Linear(hiddenDimensions, dimensions, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        output(MLXNN.gelu(input(hidden)))
    }
}

final class SCAIL2CLIPBlock: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: SCAIL2CLIPAttention
    @ModuleInfo(key: "norm2") var feedForwardNorm: LayerNorm
    @ModuleInfo(key: "mlp") var feedForward: SCAIL2CLIPMLP

    init(configuration: SCAIL2CLIPVisionConfiguration) {
        self._attentionNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.epsilon
        )
        self._attention.wrappedValue = SCAIL2CLIPAttention(
            dimensions: configuration.hiddenSize,
            heads: configuration.headCount
        )
        self._feedForwardNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.epsilon
        )
        self._feedForward.wrappedValue = SCAIL2CLIPMLP(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.feedForwardSize
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(attentionNorm(input))
        return attended + feedForward(feedForwardNorm(attended))
    }
}

public final class SCAIL2CLIPVisionModel: Module {
    public let configuration: SCAIL2CLIPVisionConfiguration
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ParameterInfo(key: "cls_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "pos_embedding") var positionEmbedding: MLXArray
    @ModuleInfo(key: "pre_norm") var inputNorm: LayerNorm
    @ModuleInfo(key: "transformer") var blocks: [SCAIL2CLIPBlock]

    public init(configuration: SCAIL2CLIPVisionConfiguration = SCAIL2CLIPVisionConfiguration()) {
        self.configuration = configuration
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: configuration.hiddenSize,
            kernelSize: IntOrPair(configuration.patchSize),
            stride: IntOrPair(configuration.patchSize),
            bias: false
        )
        self._classEmbedding.wrappedValue = MLX.zeros([1, 1, configuration.hiddenSize])
        self._positionEmbedding.wrappedValue = MLX.zeros([
            1, configuration.tokenCount, configuration.hiddenSize,
        ])
        self._inputNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.epsilon
        )
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            SCAIL2CLIPBlock(configuration: configuration)
        }
    }

    public func callAsFunction(_ normalizedImagesNHWC: MLXArray) -> MLXArray {
        precondition(normalizedImagesNHWC.ndim == 4)
        precondition(normalizedImagesNHWC.dim(1) == configuration.imageSize)
        precondition(normalizedImagesNHWC.dim(2) == configuration.imageSize)
        precondition(normalizedImagesNHWC.dim(3) == 3)
        var hidden = patchEmbedding(normalizedImagesNHWC.asType(patchEmbedding.weight.dtype))
        let batch = hidden.dim(0)
        hidden = hidden.reshaped(batch, -1, configuration.hiddenSize)
        let classTokens = MLX.broadcast(
            classEmbedding.asType(hidden.dtype),
            to: [batch, 1, configuration.hiddenSize]
        )
        hidden = inputNorm(
            MLX.concatenated([classTokens, hidden], axis: 1)
                + positionEmbedding.asType(hidden.dtype)
        )
        for block in blocks {
            hidden = block(hidden)
        }
        return hidden
    }
}

public enum SCAIL2CLIPPreprocessor {
    public static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    public static let standardDeviation: [Float] = [0.26862954, 0.26130258, 0.27577711]

    public static func normalizedNHWC(
        _ image: MediaImage,
        configuration: SCAIL2CLIPVisionConfiguration = SCAIL2CLIPVisionConfiguration()
    ) -> MLXArray {
        var rgb: [Float] = []
        rgb.reserveCapacity(image.width * image.height * 3)
        for pixel in 0..<(image.width * image.height) {
            let offset = pixel * 4
            rgb.append(Float(image.rgba8[offset]) / 255)
            rgb.append(Float(image.rgba8[offset + 1]) / 255)
            rgb.append(Float(image.rgba8[offset + 2]) / 255)
        }
        let resized = bicubicResize(
            rgb,
            sourceWidth: image.width,
            sourceHeight: image.height,
            targetWidth: configuration.imageSize,
            targetHeight: configuration.imageSize
        )
        var normalized = resized
        for pixel in 0..<(configuration.imageSize * configuration.imageSize) {
            for channel in 0..<3 {
                let index = pixel * 3 + channel
                normalized[index] = (normalized[index] - mean[channel]) / standardDeviation[channel]
            }
        }
        return MLXArray(normalized).reshaped(
            1, configuration.imageSize, configuration.imageSize, 3
        )
    }

    /// Preprocesses the already aspect-filled SCAIL reference tensor. This is
    /// the path used by upstream `CLIPModel.visual`: resize each `[-1, 1]`
    /// frame with non-antialiased bicubic interpolation, then OpenCLIP-normalize.
    public static func normalizedNHWC(
        croppedImages: MLXArray,
        configuration: SCAIL2CLIPVisionConfiguration = SCAIL2CLIPVisionConfiguration()
    ) -> MLXArray {
        precondition(croppedImages.ndim == 4 && croppedImages.dim(3) == 3)
        let batch = croppedImages.dim(0)
        let sourceHeight = croppedImages.dim(1)
        let sourceWidth = croppedImages.dim(2)
        let values = ((croppedImages.asType(.float32) + 1) * 0.5).asArray(Float.self)
        let sourceFrameSize = sourceHeight * sourceWidth * 3
        var output: [Float] = []
        output.reserveCapacity(batch * configuration.imageSize * configuration.imageSize * 3)
        for frame in 0..<batch {
            let start = frame * sourceFrameSize
            let resized = bicubicResize(
                Array(values[start..<(start + sourceFrameSize)]),
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                targetWidth: configuration.imageSize,
                targetHeight: configuration.imageSize
            )
            for pixel in 0..<(configuration.imageSize * configuration.imageSize) {
                for channel in 0..<3 {
                    let index = pixel * 3 + channel
                    output.append((resized[index] - mean[channel]) / standardDeviation[channel])
                }
            }
        }
        return MLXArray(output).reshaped(
            batch, configuration.imageSize, configuration.imageSize, 3
        )
    }

    /// CPU port of `torch.nn.functional.interpolate` with `mode="bicubic"`,
    /// `align_corners=false`, and its default `antialias=false` setting.
    static func bicubicResize(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        precondition(source.count == sourceWidth * sourceHeight * 3)
        let horizontal = interpolationWeights(sourceSize: sourceWidth, targetSize: targetWidth)
        var intermediate = [Float](repeating: 0, count: sourceHeight * targetWidth * 3)
        for y in 0..<sourceHeight {
            for x in 0..<targetWidth {
                let destination = (y * targetWidth + x) * 3
                for (sourceX, weight) in horizontal[x] {
                    let sourceOffset = (y * sourceWidth + sourceX) * 3
                    for channel in 0..<3 {
                        intermediate[destination + channel] += source[sourceOffset + channel] * weight
                    }
                }
            }
        }
        let vertical = interpolationWeights(sourceSize: sourceHeight, targetSize: targetHeight)
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * 3)
        for y in 0..<targetHeight {
            for (sourceY, weight) in vertical[y] {
                for x in 0..<targetWidth {
                    let sourceOffset = (sourceY * targetWidth + x) * 3
                    let destination = (y * targetWidth + x) * 3
                    for channel in 0..<3 {
                        output[destination + channel] += intermediate[sourceOffset + channel] * weight
                    }
                }
            }
        }
        return output
    }

    private static func interpolationWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        return (0..<targetSize).map { destination in
            let sourcePosition = scale * (Float(destination) + 0.5) - 0.5
            let base = Int(Foundation.floor(Double(sourcePosition)))
            var combined: [Int: Float] = [:]
            for offset in -1...2 {
                let unclamped = base + offset
                let index = min(sourceSize - 1, max(0, unclamped))
                combined[index, default: 0] += cubicKernel(sourcePosition - Float(unclamped))
            }
            return combined.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        }
    }

    private static func cubicKernel(_ value: Float) -> Float {
        let distance = abs(value)
        let coefficient: Float = -0.75
        if distance <= 1 {
            return ((coefficient + 2) * distance - (coefficient + 3)) * distance * distance + 1
        }
        if distance < 2 {
            return ((coefficient * distance - 5 * coefficient) * distance + 8 * coefficient) * distance
                - 4 * coefficient
        }
        return 0
    }
}
