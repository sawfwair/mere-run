import Foundation
import MLX
import MLXFast
import MLXNN

/// Arithmetic precision is independent of checkpoint storage precision.
public enum FourDAnyoneComputePrecision: String, Sendable {
    case model
    case float32

    func dtype(for storage: DType) -> DType {
        switch self {
        case .model: storage
        case .float32: .float32
        }
    }
}

/// Match the reference activation's FP32 arithmetic before its output cast.
enum FourDAnyoneActivation {
    static func silu(_ input: MLXArray) -> MLXArray {
        MLXNN.silu(input.asType(.float32)).asType(input.dtype)
    }

    static func gelu(_ input: MLXArray) -> MLXArray {
        MLXNN.geluApproximate(input.asType(.float32)).asType(input.dtype)
    }
}

final class FourDAnyoneRMSNorm: Module {
    @ModuleInfo var weight: MLXArray
    let epsilon: Float

    init(_ dimensions: Int, epsilon: Float) {
        self._weight.wrappedValue = MLX.ones([dimensions])
        self.epsilon = epsilon
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let value = input.asType(.float32)
        let normalized = value * MLX.rsqrt(MLX.mean(value * value, axis: -1, keepDims: true) + epsilon)
        return normalized.asType(input.dtype) * weight
    }
}

/// FP64 host trigonometry followed by FP32 tensor rotation and the model's cast.
enum FourDAnyoneRoPE {
    static func prepare(grid: Wan2GridSize, headDimension: Int) -> Wan2RoPE.Cache {
        let spatial = headDimension / 3
        let dimensions = [headDimension - 2 * spatial, spatial, spatial]
        precondition(dimensions.allSatisfy { $0 > 0 && $0.isMultiple(of: 2) })
        var cosine: [Float] = []
        var sine: [Float] = []
        cosine.reserveCapacity(grid.sequenceLength * headDimension / 2)
        sine.reserveCapacity(cosine.capacity)
        for frame in 0..<grid.frames {
            for row in 0..<grid.height {
                for column in 0..<grid.width {
                    for (position, dimension) in zip([frame, row, column], dimensions) {
                        for index in stride(from: 0, to: dimension, by: 2) {
                            let angle = Double(position) / pow(10_000, Double(index) / Double(dimension))
                            cosine.append(Float(cos(angle)))
                            sine.append(Float(sin(angle)))
                        }
                    }
                }
            }
        }
        let shape = [grid.sequenceLength, 1, headDimension / 2]
        return Wan2RoPE.Cache(cosine: MLXArray(cosine, shape), sine: MLXArray(sine, shape))
    }

    static func apply(_ input: MLXArray, cache: Wan2RoPE.Cache) -> MLXArray {
        let shape = input.shape
        let pairs = input.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let real = pairs[0..., 0..., 0..., 0..., 0]
        let imaginary = pairs[0..., 0..., 0..., 0..., 1]
        let cosine = cache.cosine.expandedDimensions(axis: 0)
        let sine = cache.sine.expandedDimensions(axis: 0)
        return MLX.stacked(
            [real * cosine - imaginary * sine, real * sine + imaginary * cosine], axis: -1
        ).reshaped(shape).asType(input.dtype)
    }
}

final class FourDAnyoneAttention: Module {
    @ModuleInfo(key: "q") var query: Linear
    @ModuleInfo(key: "k") var key: Linear
    @ModuleInfo(key: "v") var value: Linear
    @ModuleInfo(key: "o") var output: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: FourDAnyoneRMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: FourDAnyoneRMSNorm
    let heads: Int
    let computePrecision: FourDAnyoneComputePrecision

    init(_ configuration: Wan2TransformerConfiguration, computePrecision: FourDAnyoneComputePrecision) {
        let width = configuration.hiddenSize
        self.heads = configuration.headCount
        self.computePrecision = computePrecision
        self._query.wrappedValue = Linear(width, width)
        self._key.wrappedValue = Linear(width, width)
        self._value.wrappedValue = Linear(width, width)
        self._output.wrappedValue = Linear(width, width)
        self._queryNorm.wrappedValue = FourDAnyoneRMSNorm(width, epsilon: configuration.epsilon)
        self._keyNorm.wrappedValue = FourDAnyoneRMSNorm(width, epsilon: configuration.epsilon)
    }

    func callAsFunction(
        _ input: MLXArray,
        context: MLXArray? = nil,
        rope: Wan2RoPE.Cache? = nil,
        maximumQueryTokens: Int
    ) -> MLXArray {
        let typed = input.asType(computePrecision.dtype(for: query.weight.dtype))
        let source = context.map { MLX.repeated($0, count: input.dim(0), axis: 0) }
            .map { $0.asType(computePrecision.dtype(for: key.weight.dtype)) } ?? typed
        let headWidth = query.weight.dim(0) / heads
        var q = queryNorm(query(typed)).reshaped(input.dim(0), -1, heads, headWidth)
        var k = keyNorm(key(source)).reshaped(source.dim(0), -1, heads, headWidth)
        let v = value(source).reshaped(source.dim(0), -1, heads, headWidth).transposed(0, 2, 1, 3)
        if let rope {
            q = FourDAnyoneRoPE.apply(q, cache: rope)
            k = FourDAnyoneRoPE.apply(k, cache: rope)
        }
        let attended = SCAIL2SelfAttention.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v,
            scale: 1 / Float(headWidth).squareRoot(),
            maximumQueryTokens: maximumQueryTokens
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(input.dim(0), -1, heads * headWidth))
    }
}

final class FourDAnyoneProjection: Module {
    @ModuleInfo var input: Linear
    @ModuleInfo var output: Linear
    let usesSiLU: Bool
    let computePrecision: FourDAnyoneComputePrecision

    init(
        input: Int, hidden: Int, output: Int, usesSiLU: Bool = false,
        computePrecision: FourDAnyoneComputePrecision
    ) {
        self._input.wrappedValue = Linear(input, hidden)
        self._output.wrappedValue = Linear(hidden, output)
        self.usesSiLU = usesSiLU
        self.computePrecision = computePrecision
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let projected = input(value.asType(computePrecision.dtype(for: input.weight.dtype)))
        return output(usesSiLU ? FourDAnyoneActivation.silu(projected) : FourDAnyoneActivation.gelu(projected))
    }
}

final class FourDAnyoneTimeProjection: Module {
    @ModuleInfo var linear: Linear

    init(_ width: Int) { self._linear.wrappedValue = Linear(width, width * 6) }
    func callAsFunction(_ input: MLXArray) -> MLXArray { linear(FourDAnyoneActivation.silu(input)) }
}

final class FourDAnyoneBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: FourDAnyoneAttention
    @ModuleInfo(key: "self_attn_mvs") var multiviewAttention: FourDAnyoneAttention
    @ModuleInfo(key: "cross_attn") var crossAttention: FourDAnyoneAttention
    @ModuleInfo(key: "norm1") var selfNorm: Wan2LayerNorm
    @ModuleInfo(key: "norm1_mvs") var multiviewNorm: Wan2LayerNorm
    @ModuleInfo(key: "norm2") var feedForwardNorm: Wan2LayerNorm
    @ModuleInfo(key: "norm3") var crossNorm: Wan2LayerNorm
    @ModuleInfo(key: "ffn") var feedForward: FourDAnyoneProjection
    @ModuleInfo var modulation: MLXArray
    @ModuleInfo(key: "modulation_mvs") var multiviewModulation: MLXArray

    init(_ configuration: Wan2TransformerConfiguration, computePrecision: FourDAnyoneComputePrecision) {
        let width = configuration.hiddenSize
        self._selfAttention.wrappedValue = FourDAnyoneAttention(configuration, computePrecision: computePrecision)
        self._multiviewAttention.wrappedValue = FourDAnyoneAttention(configuration, computePrecision: computePrecision)
        self._crossAttention.wrappedValue = FourDAnyoneAttention(configuration, computePrecision: computePrecision)
        self._selfNorm.wrappedValue = Wan2LayerNorm(dimensions: width, epsilon: configuration.epsilon)
        self._multiviewNorm.wrappedValue = Wan2LayerNorm(dimensions: width, epsilon: configuration.epsilon)
        self._feedForwardNorm.wrappedValue = Wan2LayerNorm(dimensions: width, epsilon: configuration.epsilon)
        self._crossNorm.wrappedValue = Wan2LayerNorm(
            dimensions: width, epsilon: configuration.epsilon, affine: true
        )
        self._feedForward.wrappedValue = FourDAnyoneProjection(
            input: width, hidden: configuration.feedForwardSize, output: width, computePrecision: computePrecision
        )
        self._modulation.wrappedValue = MLX.zeros([1, 6, width])
        self._multiviewModulation.wrappedValue = MLX.zeros([1, 3, width])
    }

    func callAsFunction(
        _ input: MLXArray,
        context: MLXArray,
        time: MLXArray,
        grid: Wan2GridSize,
        spatialRoPE: Wan2RoPE.Cache,
        multiviewRoPE: Wan2RoPE.Cache,
        maximumQueryTokens: Int,
        observe: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let parts = MLX.split(modulation.asType(time.dtype) + time, parts: 6, axis: 1)
        var x = input
        let normalized = selfNorm(x.asType(.float32)) * (1 + parts[1]) + parts[0]
        x = x + parts[2] * selfAttention(
            normalized, rope: spatialRoPE, maximumQueryTokens: maximumQueryTokens
        )
        observe?("video_attention", x)
        let viewParts = MLX.split(
            multiviewModulation.asType(time.dtype) + time[0..., 0..<3, 0...], parts: 3, axis: 1
        )
        let views = x.dim(0)
        let width = x.dim(2)
        let multiview = (multiviewNorm(x.asType(.float32)) * (1 + viewParts[1]) + viewParts[0])
            .reshaped(views, grid.frames, grid.height, grid.width, width)
            .transposed(1, 0, 2, 3, 4)
            .reshaped(grid.frames, views * grid.height * grid.width, width)
        let attended = multiviewAttention(
            multiview, rope: multiviewRoPE, maximumQueryTokens: maximumQueryTokens
        ).reshaped(grid.frames, views, grid.height, grid.width, width)
            .transposed(1, 0, 2, 3, 4).reshaped(views, grid.sequenceLength, width)
        x = x + viewParts[2] * attended
        observe?("multiview_attention", x)
        x = x + crossAttention(
            crossNorm(x.asType(.float32)), context: context, maximumQueryTokens: maximumQueryTokens
        )
        observe?("text_attention", x)
        let feedInput = feedForwardNorm(x.asType(.float32)) * (1 + parts[4]) + parts[3]
        let output = x + parts[5] * feedForward(feedInput)
        observe?("feed_forward", output)
        return output
    }
}
