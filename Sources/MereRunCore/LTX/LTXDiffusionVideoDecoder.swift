import Foundation
import MLX
import MLXFast
import MLXNN

public enum LTXVideoDecoderKind: String, Sendable, CaseIterable {
    case convolutional
    case diffusion
}

public enum LTXDiffusionVideoDecoderError: LocalizedError {
    case missingWeights(URL)
    case invalidLatentShape([Int])

    public var errorDescription: String? {
        switch self {
        case .missingWeights(let url):
            return "Missing LTX 2.5 DiffVAE weights at \(url.path)."
        case .invalidLatentShape(let shape):
            return "LTX 2.5 DiffVAE expects [B, 128, F, H, W] latents, got \(shape)."
        }
    }
}

/// Native Swift/MLX port of the official LTX 2.5 `NADiffusionDecoder`.
///
/// The decoder uses the checkpoint's deterministic neighborhood-attention
/// stages followed by its one-step x0 diffusion stage. Attention is evaluated
/// in bounded query tiles with the same shifted boundary windows as NATTEN.
public final class LTXDiffusionVideoDecoder: Module {
    public static let patchSize = 4

    private let stageKernels = [
        (3, 7, 7),
        (3, 7, 7),
        (3, 5, 5),
        (3, 5, 5),
    ]
    private let stageStrides = [
        (1, 2, 2),
        (2, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
    ]
    private let stage5Kernel = (11, 11, 11)
    private let trailingLatentFrames = 2

    @ModuleInfo(key: "conv_in") private var convIn: Linear
    @ModuleInfo(key: "det_stages") private var deterministicStages: [LTXDiffVAEDeterministicStage]
    @ModuleInfo(key: "upsamples") private var upsamplers: [LTXDiffVAELinearUpsampler]
    @ModuleInfo(key: "conv_in_x_t") private var pixelInputProjection: Linear
    @ModuleInfo(key: "shared_adaln") private var sharedAdaLN: LTXDiffVAESharedAdaLN
    @ModuleInfo(key: "diff_blocks") private var diffusionBlocks: [LTXDiffVAEDiffusionBlock]
    @ModuleInfo(key: "norm_out") private var outputNorm: RMSNorm
    @ModuleInfo(key: "conv_out") private var outputProjection: Linear
    @ModuleInfo(key: "t_embedder") private var timestepEmbedder: LTXDiffVAETimestepEmbedder

    public var latentsMean: MLXArray = MLX.zeros([128], dtype: .float32)
    public var latentsStd: MLXArray = MLX.ones([128], dtype: .float32)

    public override init() {
        let channels = [2_048, 1_024, 512, 512, 256]
        let depths = [4, 6, 4, 2]
        let kernels = [(3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)]
        let diffusionKernel = (11, 11, 11)
        self._convIn.wrappedValue = Linear(128, channels[0], bias: true)
        self._deterministicStages.wrappedValue = (0..<4).map { index in
            LTXDiffVAEDeterministicStage(
                channels: channels[index],
                depth: depths[index],
                kernel: kernels[index]
            )
        }
        self._upsamplers.wrappedValue = [
            LTXDiffVAELinearUpsampler(channels: 2_048, stride: (1, 2, 2), reduction: 2),
            LTXDiffVAELinearUpsampler(channels: 1_024, stride: (2, 1, 1), reduction: 2),
            LTXDiffVAELinearUpsampler(channels: 512, stride: (2, 2, 2), reduction: 1),
            LTXDiffVAELinearUpsampler(channels: 512, stride: (2, 2, 2), reduction: 2),
        ]
        self._pixelInputProjection.wrappedValue = Linear(48, 256, bias: true)
        self._sharedAdaLN.wrappedValue = LTXDiffVAESharedAdaLN()
        self._diffusionBlocks.wrappedValue = (0..<8).map { _ in
            LTXDiffVAEDiffusionBlock(channels: 256, contextChannels: 256, kernel: diffusionKernel)
        }
        self._outputNorm.wrappedValue = RMSNorm(dimensions: 256, eps: 1e-6)
        self._outputProjection.wrappedValue = Linear(256, 48, bias: true)
        self._timestepEmbedder.wrappedValue = LTXDiffVAETimestepEmbedder()
        super.init()
    }

    public static func load(
        weightsURL: URL,
        dtype: DType = .bfloat16,
        fileManager: FileManager = .default
    ) throws -> LTXDiffusionVideoDecoder {
        guard fileManager.fileExists(atPath: weightsURL.path) else {
            throw LTXDiffusionVideoDecoderError.missingWeights(weightsURL)
        }
        let decoder = LTXDiffusionVideoDecoder()
        let statistics = try SafetensorsStreamingLoader.loadArrays(
            url: weightsURL,
            where: {
                $0 == "per_channel_statistics.mean-of-means"
                    || $0 == "per_channel_statistics.std-of-means"
            },
            dtype: .float32
        )
        if let mean = statistics["per_channel_statistics.mean-of-means"] {
            decoder.latentsMean = mean
        }
        if let standardDeviation = statistics["per_channel_statistics.std-of-means"] {
            decoder.latentsStd = standardDeviation
        }
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: weightsURL,
            to: decoder,
            dtype: dtype,
            verify: [.noUnusedKeys, .shapeMismatch],
            include: { $0.hasPrefix("decoder.") && $0 != "decoder.type_emb" },
            mapper: { key, value in
                mapLTXDiffusionVideoDecoderWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 16
        )
        return decoder
    }

    public func decode(sample: MLXArray, seed: Int) throws -> MLXArray {
        guard sample.ndim == 5, sample.dim(1) == 128 else {
            throw LTXDiffusionVideoDecoderError.invalidLatentShape(sample.shape)
        }
        let contentFrames = 1 + (sample.dim(2) - 1) * 8
        let contentHeight = sample.dim(3) * 32
        let contentWidth = sample.dim(4) * 32
        let minimum = ltxDiffVAEMinimumLatentShape(
            stageKernels: stageKernels,
            stageStrides: stageStrides,
            stage5Kernel: stage5Kernel
        )
        let resized = ltxDiffVAEResizeLatentToMinimum(sample, minimum: minimum)
        var latent = resized.array
        let lastFrame = latent[0..., 0..., (latent.dim(2) - 1)..<latent.dim(2), 0..., 0...]
        latent = MLX.concatenated(
            [latent, MLX.repeated(lastFrame, count: trailingLatentFrames, axis: 2)],
            axis: 2
        )
        let mean = latentsMean.asType(.float32).reshaped(1, 128, 1, 1, 1)
        let standardDeviation = latentsStd.asType(.float32).reshaped(1, 128, 1, 1, 1)
        latent = (latent.asType(.float32) * standardDeviation + mean)
            .asType(sample.dtype)
            .transposed(0, 2, 3, 4, 1)
        var hidden = convIn(latent)
        MLX.eval(hidden)
        for index in 0..<deterministicStages.count {
            hidden = deterministicStages[index](hidden)
            hidden = upsamplers[index](hidden, dropLeadingFrame: true)
            MLX.eval(hidden)
            Memory.clearCache()
        }

        let ghostFrames = trailingLatentFrames * 8
        let contextFrames = max(hidden.dim(1) - ghostFrames, 1)
        let contextKeep = min(hidden.dim(1), max(contextFrames, stage5Kernel.0))
        let context = hidden[0..., 0..<contextKeep, 0..., 0..., 0...]
        let pixelFrames = context.dim(1)
        let pixelHeight = context.dim(2) * Self.patchSize
        let pixelWidth = context.dim(3) * Self.patchSize

        MLXRandom.seed(UInt64(bitPattern: Int64(seed)))
        let initialNoise = MLXRandom.normal([
            sample.dim(0),
            3,
            pixelFrames,
            pixelHeight,
            pixelWidth,
        ]).asType(sample.dtype)
        var hiddenPixels = pixelInputProjection(ltxDiffVAEPatchifyPixels(initialNoise))
        let timestep = MLX.ones([sample.dim(0)], dtype: sample.dtype)
        let modulation = sharedAdaLN(timestepEmbedder(timestep))
        for block in diffusionBlocks {
            hiddenPixels = block(hiddenPixels, context: context, modulation: modulation)
            MLX.eval(hiddenPixels)
            Memory.clearCache()
        }
        let patched = outputProjection(outputNorm(hiddenPixels))
            .transposed(0, 4, 1, 2, 3)
        var pixels = ltxDiffVAEUnpatchifyPixels(patched)
        let heightStart = resized.heightBefore * 32
        let widthStart = resized.widthBefore * 32
        pixels = pixels[
            0...,
            0...,
            0..<contentFrames,
            heightStart..<(heightStart + contentHeight),
            widthStart..<(widthStart + contentWidth)
        ]
        return pixels
    }
}

func mapLTXDiffusionVideoDecoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    let targets = ltxDiffusionVideoDecoderWeightTargets(key: key, shape: value.shape)
    guard !targets.isEmpty else { return [] }
    let casted = value.dtype.isFloatingPoint && value.dtype != dtype
        ? value.asType(dtype)
        : value
    if targets.count == 3 {
        let dimension = casted.dim(0) / 3
        return [
            (targets[0].name, casted[0..<dimension]),
            (targets[1].name, casted[dimension..<(dimension * 2)]),
            (targets[2].name, casted[(dimension * 2)...]),
        ]
    }
    return [(targets[0].name, casted)]
}

func ltxDiffusionVideoDecoderWeightTargets(
    key: String,
    shape: [Int]
) -> [(name: String, shape: [Int])] {
    guard key.hasPrefix("decoder."), key != "decoder.type_emb" else { return [] }
    var mapped = String(key.dropFirst("decoder.".count))
    if mapped.hasPrefix("t_embedder.mlp.0.") {
        mapped = mapped.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.linear_1.")
    } else if mapped.hasPrefix("t_embedder.mlp.2.") {
        mapped = mapped.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.linear_2.")
    }
    for stage in 0..<4 {
        mapped = mapped.replacingOccurrences(
            of: "det_stages.\(stage).",
            with: "det_stages.\(stage).blocks."
        )
    }
    if mapped.contains(".attn.qkv.weight") || mapped.contains(".attn.qkv.bias") {
        guard let firstDimension = shape.first, firstDimension % 3 == 0 else { return [] }
        let dimension = firstDimension / 3
        let qkvSuffix = mapped.hasSuffix("weight") ? "qkv.weight" : "qkv.bias"
        let prefix = String(mapped.dropLast(qkvSuffix.count))
        let suffix = mapped.hasSuffix("weight") ? "weight" : "bias"
        var splitShape = shape
        splitShape[0] = dimension
        return [
            ("\(prefix)to_q.\(suffix)", splitShape),
            ("\(prefix)to_k.\(suffix)", splitShape),
            ("\(prefix)to_v.\(suffix)", splitShape),
        ]
    }
    return [(mapped, shape)]
}

private final class LTXDiffVAEDeterministicStage: Module {
    @ModuleInfo(key: "blocks") var blocks: [LTXDiffVAEDeterministicBlock]

    init(channels: Int, depth: Int, kernel: (Int, Int, Int)) {
        self._blocks.wrappedValue = (0..<depth).map { _ in
            LTXDiffVAEDeterministicBlock(channels: channels, kernel: kernel)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        blocks.reduce(input) { hidden, block in block(hidden) }
    }
}

private final class LTXDiffVAEDeterministicBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attention: LTXDiffVAENeighborhoodAttention3D
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: LTXDiffVAESwiGLU

    init(channels: Int, kernel: (Int, Int, Int)) {
        self._norm1.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._attention.wrappedValue = LTXDiffVAENeighborhoodAttention3D(
            channels: channels,
            kernel: kernel
        )
        self._norm2.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._mlp.wrappedValue = LTXDiffVAESwiGLU(channels: channels)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input + attention(norm1(input))
        hidden = hidden + mlp(norm2(hidden))
        return hidden
    }
}

private final class LTXDiffVAELinearUpsampler: Module {
    let stride: (Int, Int, Int)
    @ModuleInfo(key: "proj") var projection: Linear

    init(channels: Int, stride: (Int, Int, Int), reduction: Int) {
        self.stride = stride
        self._projection.wrappedValue = Linear(
            channels,
            channels * stride.0 * stride.1 * stride.2 / reduction,
            bias: true
        )
    }

    func callAsFunction(_ input: MLXArray, dropLeadingFrame: Bool) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        let expanded = projection(input)
        let channelsAfterShuffle = expanded.dim(4) / (stride.0 * stride.1 * stride.2)
        var output = expanded
            .reshaped(batch, frames, height, width, channelsAfterShuffle, stride.0, stride.1, stride.2)
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(
                batch,
                frames * stride.0,
                height * stride.1,
                width * stride.2,
                channelsAfterShuffle
            )
        if stride.0 == 2, dropLeadingFrame {
            output = output[0..., 1..., 0..., 0..., 0...]
        }
        return output
    }
}

private final class LTXDiffVAESharedAdaLN: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    override init() {
        self._projection.wrappedValue = Linear(384, 7 * 256, bias: true)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        projection(ltxDiffVAESiLU(timestep)).reshaped(timestep.dim(0), 7, 256)
    }
}

private final class LTXDiffVAETimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    override init() {
        self._first.wrappedValue = Linear(256, 384, bias: true)
        self._second.wrappedValue = Linear(384, 384, bias: true)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let projected = ltxDiffVAETimestepEmbedding(timestep * MLXArray(Float(1_000)))
            .asType(timestep.dtype)
        return second(ltxDiffVAESiLU(first(projected)))
    }
}

private final class LTXDiffVAEDiffusionBlock: Module {
    @ModuleInfo(key: "context_proj") var contextProjection: Linear
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attention: LTXDiffVAENeighborhoodAttention3D
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: LTXDiffVAESwiGLU

    init(channels: Int, contextChannels: Int, kernel: (Int, Int, Int)) {
        self._contextProjection.wrappedValue = Linear(contextChannels, channels, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([7, channels], dtype: .float32)
        self._norm1.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._attention.wrappedValue = LTXDiffVAENeighborhoodAttention3D(
            channels: channels,
            kernel: kernel
        )
        self._norm2.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._mlp.wrappedValue = LTXDiffVAESwiGLU(channels: channels)
    }

    func callAsFunction(
        _ input: MLXArray,
        context: MLXArray,
        modulation: MLXArray
    ) -> MLXArray {
        let batch = input.dim(0)
        let combined = modulation + scaleShiftTable
            .asType(modulation.dtype)
            .reshaped(1, 7, input.dim(4))
        let scaleAttention = combined[0..., 0, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        let shiftAttention = combined[0..., 1, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        let scaleMLP = combined[0..., 3, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        let shiftMLP = combined[0..., 4, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        var hidden = input + contextProjection(context)
        let normalizedAttention = norm1(hidden) * (MLXArray(1).asType(hidden.dtype) + scaleAttention)
            + shiftAttention
        hidden = hidden + attention(normalizedAttention)
        let normalizedMLP = norm2(hidden) * (MLXArray(1).asType(hidden.dtype) + scaleMLP)
            + shiftMLP
        return hidden + mlp(normalizedMLP)
    }
}

private final class LTXDiffVAESwiGLU: Module {
    @ModuleInfo(key: "w_up") var up: Linear
    @ModuleInfo(key: "w_gate") var gate: Linear
    @ModuleInfo(key: "w_down") var down: Linear

    init(channels: Int) {
        let hidden = ((channels * 4 + 15) / 16) * 16
        self._up.wrappedValue = Linear(channels, hidden, bias: false)
        self._gate.wrappedValue = Linear(channels, hidden, bias: false)
        self._down.wrappedValue = Linear(hidden, channels, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let tokenCount = input.dim(0) * input.dim(1) * input.dim(2) * input.dim(3)
        let flat = input.reshaped(tokenCount, input.dim(4))
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((tokenCount + 16_383) / 16_384)
        for start in stride(from: 0, to: tokenCount, by: 16_384) {
            let end = min(start + 16_384, tokenCount)
            let tile = flat[start..<end]
            chunks.append(down(ltxDiffVAESiLU(gate(tile)) * up(tile)))
        }
        return MLX.concatenated(chunks, axis: 0).reshaped(input.shape)
    }
}

final class LTXDiffVAENeighborhoodAttention3D: Module {
    let channels: Int
    let headDimension = 64
    let kernel: (Int, Int, Int)
    let scoreBudget: Int

    @ModuleInfo(key: "to_q") var queryProjection: Linear
    @ModuleInfo(key: "to_k") var keyProjection: Linear
    @ModuleInfo(key: "to_v") var valueProjection: Linear
    @ModuleInfo(key: "proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(
        channels: Int,
        kernel: (Int, Int, Int),
        scoreBudget: Int = 1 << 25
    ) {
        precondition(channels % headDimension == 0)
        self.channels = channels
        self.kernel = kernel
        self.scoreBudget = scoreBudget
        self._queryProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._keyProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._valueProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._outputProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._queryNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: 1e-6)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: 1e-6)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        precondition(frames >= kernel.0 && height >= kernel.1 && width >= kernel.2)
        let heads = channels / headDimension
        let headShape = [batch, frames, height, width, heads, headDimension]
        var query = queryNorm(queryProjection(input).reshaped(headShape))
            * MLXArray(1 / Float(headDimension).squareRoot()).asType(input.dtype)
        var key = keyNorm(keyProjection(input).reshaped(headShape))
        let value = valueProjection(input).reshaped(headShape)
        query = ltxDiffVAEAbsoluteRoPE(query)
        key = ltxDiffVAEAbsoluteRoPE(key)
        let attended = LTXDiffVAEMetalNeighborhoodAttention.apply(
            query: query,
            key: key,
            value: value,
            kernel: kernel
        ) ?? ltxDiffVAENeighborhoodAttention(
            query: query,
            key: key,
            value: value,
            kernel: kernel,
            scoreBudget: scoreBudget
        )
        return outputProjection(attended.reshaped(batch, frames, height, width, channels))
    }
}

func ltxDiffVAEWindowBounds(length: Int, kernel: Int) -> (starts: [Int], ends: [Int]) {
    let effectiveKernel = min(length, kernel)
    let maximumStart = length - effectiveKernel
    let half = effectiveKernel / 2
    let starts = (0..<length).map { min(max($0 - half, 0), maximumStart) }
    return (starts, starts.map { $0 + effectiveKernel })
}

func ltxDiffVAENeighborhoodAttention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    kernel: (Int, Int, Int),
    scoreBudget: Int
) -> MLXArray {
    let batch = query.dim(0)
    let dims = [query.dim(1), query.dim(2), query.dim(3)]
    let kernels = [kernel.0, kernel.1, kernel.2]
    let bounds = zip(dims, kernels).map { ltxDiffVAEWindowBounds(length: $0, kernel: $1) }
    var tile = dims
    func scoreCost(_ candidate: [Int]) -> Int {
        let queries = candidate.reduce(1, *)
        let references = zip(zip(candidate, kernels), dims)
            .map { min($1, $0.0 + $0.1 - 1) }
            .reduce(1, *)
        return queries * references
    }
    while scoreCost(tile) > scoreBudget, tile.max()! > 1 {
        let axis = (0..<3).max { lhs, rhs in
            Double(tile[lhs]) / Double(kernels[lhs]) < Double(tile[rhs]) / Double(kernels[rhs])
        }!
        tile[axis] = max(1, (tile[axis] + 1) / 2)
    }

    var temporalParts: [MLXArray] = []
    for temporalStart in stride(from: 0, to: dims[0], by: tile[0]) {
        let temporalEnd = min(temporalStart + tile[0], dims[0])
        var heightParts: [MLXArray] = []
        for heightStart in stride(from: 0, to: dims[1], by: tile[1]) {
            let heightEnd = min(heightStart + tile[1], dims[1])
            var widthParts: [MLXArray] = []
            for widthStart in stride(from: 0, to: dims[2], by: tile[2]) {
                let widthEnd = min(widthStart + tile[2], dims[2])
                let queryRanges = [
                    temporalStart..<temporalEnd,
                    heightStart..<heightEnd,
                    widthStart..<widthEnd,
                ]
                let referenceRanges: [Range<Int>] = (0..<3).map { axis in
                    let lower = bounds[axis].starts[queryRanges[axis].lowerBound]
                    let upper = bounds[axis].ends[queryRanges[axis].upperBound - 1]
                    return lower..<upper
                }
                let queryTile = query[
                    0...,
                    queryRanges[0],
                    queryRanges[1],
                    queryRanges[2],
                    0...,
                    0...
                ]
                let keyTile = key[
                    0...,
                    referenceRanges[0],
                    referenceRanges[1],
                    referenceRanges[2],
                    0...,
                    0...
                ]
                let valueTile = value[
                    0...,
                    referenceRanges[0],
                    referenceRanges[1],
                    referenceRanges[2],
                    0...,
                    0...
                ]
                let queryCount = queryRanges.map(\.count).reduce(1, *)
                let referenceCount = referenceRanges.map(\.count).reduce(1, *)
                let heads = query.dim(4)
                let headDimension = query.dim(5)
                let flattenedQuery = queryTile
                    .transposed(0, 4, 1, 2, 3, 5)
                    .reshaped(batch, heads, queryCount, headDimension)
                let flattenedKey = keyTile
                    .transposed(0, 4, 1, 2, 3, 5)
                    .reshaped(batch, heads, referenceCount, headDimension)
                let flattenedValue = valueTile
                    .transposed(0, 4, 1, 2, 3, 5)
                    .reshaped(batch, heads, referenceCount, headDimension)
                let mask = ltxDiffVAENeighborhoodMask(
                    queryRanges: queryRanges,
                    referenceRanges: referenceRanges,
                    bounds: bounds,
                    dtype: query.dtype
                )
                let attended = MLXFast.scaledDotProductAttention(
                    queries: flattenedQuery,
                    keys: flattenedKey,
                    values: flattenedValue,
                    scale: 1,
                    mask: .array(mask)
                )
                widthParts.append(
                    attended
                        .reshaped(
                            batch,
                            heads,
                            queryRanges[0].count,
                            queryRanges[1].count,
                            queryRanges[2].count,
                            headDimension
                        )
                        .transposed(0, 2, 3, 4, 1, 5)
                )
            }
            heightParts.append(MLX.concatenated(widthParts, axis: 3))
        }
        temporalParts.append(MLX.concatenated(heightParts, axis: 2))
    }
    return MLX.concatenated(temporalParts, axis: 1)
}

enum LTXDiffVAEMetalNeighborhoodAttention {
    static func apply(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        kernel: (Int, Int, Int)
    ) -> MLXArray? {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        guard ProcessInfo.processInfo.environment["MERERUN_LTX_DIFFVAE_ATTENTION"] != "mlx",
              Device.defaultDevice().deviceType == .gpu,
              query.shape == key.shape,
              query.shape == value.shape,
              query.ndim == 6,
              query.dim(5) == 64,
              query.dtype == key.dtype,
              query.dtype == value.dtype,
              query.dtype == .bfloat16 || query.dtype == .float16 || query.dtype == .float32,
              query.dim(1) >= kernel.0,
              query.dim(2) >= kernel.1,
              query.dim(3) >= kernel.2 else {
            return nil
        }
        let tokenCount = query.dim(1) * query.dim(2) * query.dim(3)
        let batchHeads = query.dim(0) * query.dim(4)
        return neighborhoodKernel(
            [query, key, value],
            template: [
                ("Frames", query.dim(1)),
                ("Height", query.dim(2)),
                ("Width", query.dim(3)),
                ("Heads", query.dim(4)),
                ("KernelT", kernel.0),
                ("KernelH", kernel.1),
                ("KernelW", kernel.2),
                ("OutputT", query.dtype),
            ],
            grid: (32, tokenCount, batchHeads),
            threadGroup: (32, 1, 1),
            outputShapes: [query.shape],
            outputDTypes: [query.dtype]
        )[0]
        #else
        return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let neighborhoodKernel = MLXFast.metalKernel(
        name: "mere_ltx_diffvae_neighborhood_attention_3d_d64_v1",
        inputNames: ["queries", "keys", "values"],
        outputNames: ["output"],
        source: """
            constexpr uint HeadDimension = 64;
            uint dimension = thread_position_in_threadgroup.x;
            uint upper_dimension = dimension + 32;
            uint query_token = threadgroup_position_in_grid.y;
            uint batch_head = threadgroup_position_in_grid.z;
            uint head = batch_head % Heads;
            uint batch = batch_head / Heads;
            uint query_width = query_token % Width;
            uint query_plane = query_token / Width;
            uint query_height = query_plane % Height;
            uint query_frame = query_plane / Height;

            uint temporal_start = metal::min(
                query_frame > KernelT / 2 ? query_frame - KernelT / 2 : 0,
                uint(Frames - KernelT)
            );
            uint height_start = metal::min(
                query_height > KernelH / 2 ? query_height - KernelH / 2 : 0,
                uint(Height - KernelH)
            );
            uint width_start = metal::min(
                query_width > KernelW / 2 ? query_width - KernelW / 2 : 0,
                uint(Width - KernelW)
            );
            uint query_offset = (
                (batch * Frames * Height * Width + query_token) * Heads + head
            ) * HeadDimension;
            float query_value = float(queries[query_offset + dimension]);
            float upper_query_value = float(queries[query_offset + upper_dimension]);
            float row_maximum = -INFINITY;
            float row_sum = 0.0f;
            float accumulated = 0.0f;
            float upper_accumulated = 0.0f;

            for (uint temporal = 0; temporal < KernelT; ++temporal) {
                uint key_frame = temporal_start + temporal;
                for (uint y = 0; y < KernelH; ++y) {
                    uint key_height = height_start + y;
                    for (uint x = 0; x < KernelW; ++x) {
                        uint key_width = width_start + x;
                        uint key_token = (key_frame * Height + key_height) * Width + key_width;
                        uint key_offset = (
                            (batch * Frames * Height * Width + key_token) * Heads + head
                        ) * HeadDimension;
                        float score = simd_sum(
                            query_value * float(keys[key_offset + dimension])
                                + upper_query_value * float(keys[key_offset + upper_dimension])
                        );
                        float next_maximum = metal::max(row_maximum, score);
                        float previous_scale = metal::fast::exp(row_maximum - next_maximum);
                        float current_scale = metal::fast::exp(score - next_maximum);
                        accumulated = accumulated * previous_scale
                            + current_scale * float(values[key_offset + dimension]);
                        upper_accumulated = upper_accumulated * previous_scale
                            + current_scale * float(values[key_offset + upper_dimension]);
                        row_sum = row_sum * previous_scale + current_scale;
                        row_maximum = next_maximum;
                    }
                }
            }
            output[query_offset + dimension] = OutputT(accumulated / row_sum);
            output[query_offset + upper_dimension] = OutputT(upper_accumulated / row_sum);
        """,
        ensureRowContiguous: true
    )
    #endif
}

private func ltxDiffVAENeighborhoodMask(
    queryRanges: [Range<Int>],
    referenceRanges: [Range<Int>],
    bounds: [(starts: [Int], ends: [Int])],
    dtype: DType
) -> MLXArray {
    var visibility: MLXArray?
    for axis in 0..<3 {
        let queryIndices = MLXArray(queryRanges[axis].map(Int32.init)).reshaped(
            axis == 0 ? queryRanges[axis].count : 1,
            axis == 1 ? queryRanges[axis].count : 1,
            axis == 2 ? queryRanges[axis].count : 1,
            1,
            1,
            1
        )
        let keyIndices = MLXArray(referenceRanges[axis].map(Int32.init)).reshaped(
            1,
            1,
            1,
            axis == 0 ? referenceRanges[axis].count : 1,
            axis == 1 ? referenceRanges[axis].count : 1,
            axis == 2 ? referenceRanges[axis].count : 1
        )
        let startValues = queryRanges[axis].map { Int32(bounds[axis].starts[$0]) }
        let endValues = queryRanges[axis].map { Int32(bounds[axis].ends[$0]) }
        let starts = MLXArray(startValues).reshaped(queryIndices.shape)
        let ends = MLXArray(endValues).reshaped(queryIndices.shape)
        let current = (keyIndices .>= starts) .&& (keyIndices .< ends)
        visibility = visibility.map { $0 .&& current } ?? current
    }
    let queryCount = queryRanges.map(\.count).reduce(1, *)
    let referenceCount = referenceRanges.map(\.count).reduce(1, *)
    let visible = visibility!.reshaped(queryCount, referenceCount)
    let zero = MLX.zeros([queryCount, referenceCount], dtype: dtype)
    let negativeInfinity = MLX.full(
        [queryCount, referenceCount],
        values: MLXArray(-Float.infinity).asType(dtype)
    )
    return MLX.where(visible, zero, negativeInfinity).reshaped(1, 1, queryCount, referenceCount)
}

private func ltxDiffVAEAbsoluteRoPE(_ input: MLXArray) -> MLXArray {
    let split = [16, 24, 24]
    let axes = [1, 2, 3]
    var chunks: [MLXArray] = []
    var offset = 0
    for index in 0..<3 {
        let dimension = split[index]
        let positions = MLXArray(0..<input.dim(axes[index])).asType(.float32)
        let exponents = MLXArray(Array(stride(from: 0, to: dimension, by: 2))).asType(.float32)
            / MLXArray(Float(dimension))
        let inverse = exp(-MLXArray(Float(Foundation.log(Double(10_000)))) * exponents)
        var angleShape = [1, 1, 1, 1, 1, dimension / 2]
        angleShape[axes[index]] = positions.dim(0)
        let angles = (positions.reshaped(-1, 1) * inverse.reshaped(1, -1)).reshaped(angleShape)
        let source = input[0..., 0..., 0..., 0..., 0..., offset..<(offset + dimension)]
        let pairs = source.reshaped(source.dim(0), source.dim(1), source.dim(2), source.dim(3), source.dim(4), dimension / 2, 2)
        let even = pairs[0..., 0..., 0..., 0..., 0..., 0..., 0].asType(.float32)
        let odd = pairs[0..., 0..., 0..., 0..., 0..., 0..., 1].asType(.float32)
        let cosine = MLX.cos(angles)
        let sine = MLX.sin(angles)
        chunks.append(
            MLX.stacked(
                [even * cosine - odd * sine, even * sine + odd * cosine],
                axis: -1
            ).reshaped(source.shape).asType(input.dtype)
        )
        offset += dimension
    }
    return MLX.concatenated(chunks, axis: -1)
}

private struct LTXDiffVAEResizedLatent {
    let array: MLXArray
    let heightBefore: Int
    let widthBefore: Int
}

private func ltxDiffVAEResizeLatentToMinimum(
    _ input: MLXArray,
    minimum: (Int, Int, Int)
) -> LTXDiffVAEResizedLatent {
    var output = input
    if output.dim(2) < minimum.0 {
        let last = output[0..., 0..., (output.dim(2) - 1)..<output.dim(2), 0..., 0...]
        output = MLX.concatenated(
            [output, MLX.repeated(last, count: minimum.0 - output.dim(2), axis: 2)],
            axis: 2
        )
    }
    var heightBefore = 0
    if output.dim(3) < minimum.1 {
        let needed = minimum.1 - output.dim(3)
        heightBefore = needed / 2
        let first = output[0..., 0..., 0..., 0..<1, 0...]
        let last = output[0..., 0..., 0..., (output.dim(3) - 1)..<output.dim(3), 0...]
        output = MLX.concatenated([
            MLX.repeated(first, count: heightBefore, axis: 3),
            output,
            MLX.repeated(last, count: needed - heightBefore, axis: 3),
        ], axis: 3)
    }
    var widthBefore = 0
    if output.dim(4) < minimum.2 {
        let needed = minimum.2 - output.dim(4)
        widthBefore = needed / 2
        let first = output[0..., 0..., 0..., 0..., 0..<1]
        let last = output[0..., 0..., 0..., 0..., (output.dim(4) - 1)..<output.dim(4)]
        output = MLX.concatenated([
            MLX.repeated(first, count: widthBefore, axis: 4),
            output,
            MLX.repeated(last, count: needed - widthBefore, axis: 4),
        ], axis: 4)
    }
    return LTXDiffVAEResizedLatent(
        array: output,
        heightBefore: heightBefore,
        widthBefore: widthBefore
    )
}

private func ltxDiffVAEMinimumLatentShape(
    stageKernels: [(Int, Int, Int)],
    stageStrides: [(Int, Int, Int)],
    stage5Kernel: (Int, Int, Int)
) -> (Int, Int, Int) {
    var minimum = [1, 1, 1]
    var cumulative = [1, 1, 1]
    for index in 0..<stageStrides.count {
        let kernel = [stageKernels[index].0, stageKernels[index].1, stageKernels[index].2]
        for axis in 0..<3 {
            minimum[axis] = max(minimum[axis], (kernel[axis] + cumulative[axis] - 1) / cumulative[axis])
        }
        let stride = [stageStrides[index].0, stageStrides[index].1, stageStrides[index].2]
        for axis in 0..<3 { cumulative[axis] *= stride[axis] }
    }
    let finalKernel = [stage5Kernel.0, stage5Kernel.1, stage5Kernel.2]
    for axis in 0..<3 {
        minimum[axis] = max(minimum[axis], (finalKernel[axis] + cumulative[axis] - 1) / cumulative[axis])
    }
    return (minimum[0], minimum[1], minimum[2])
}

private func ltxDiffVAEPatchifyPixels(_ input: MLXArray) -> MLXArray {
    let batch = input.dim(0)
    let channels = input.dim(1)
    let frames = input.dim(2)
    let height = input.dim(3)
    let width = input.dim(4)
    return input
        .reshaped(batch, channels, frames, height / 4, 4, width / 4, 4)
        .transposed(0, 2, 3, 5, 1, 6, 4)
        .reshaped(batch, frames, height / 4, width / 4, channels * 16)
}

private func ltxDiffVAEUnpatchifyPixels(_ input: MLXArray) -> MLXArray {
    let batch = input.dim(0)
    let channels = input.dim(1) / 16
    let frames = input.dim(2)
    let height = input.dim(3)
    let width = input.dim(4)
    return input
        .reshaped(batch, channels, 4, 4, frames, height, width)
        .transposed(0, 1, 4, 5, 3, 6, 2)
        .reshaped(batch, channels, frames, height * 4, width * 4)
}

private func ltxDiffVAETimestepEmbedding(_ timestep: MLXArray) -> MLXArray {
    let halfDimension = 128
    let exponent = -Foundation.log(Double(10_000)) * MLXArray(0..<halfDimension).asType(.float32)
        / MLXArray(Float(halfDimension))
    let frequencies = exp(exponent)
    let angles = timestep.asType(.float32).reshaped(-1, 1) * frequencies.reshaped(1, -1)
    return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
}

private func ltxDiffVAESiLU(_ input: MLXArray) -> MLXArray {
    input * MLX.sigmoid(input)
}
