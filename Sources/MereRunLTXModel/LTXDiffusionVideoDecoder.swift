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
