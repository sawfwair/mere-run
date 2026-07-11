import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public struct LTXDistilledLatentGenerationOptions: Sendable {
    public let prompt: String
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Int
    public let seed: Int
    public let maxTextLength: Int
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float

    public init(
        prompt: String,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Int = 24,
        seed: Int,
        maxTextLength: Int = 1024,
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1.0,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1.0
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.maxTextLength = maxTextLength
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
    }
}

public struct LTXDistilledLatentGenerationResult: @unchecked Sendable {
    public let latents: MLXArray
    public let stage1Latents: MLXArray

    public init(latents: MLXArray, stage1Latents: MLXArray) {
        self.latents = latents
        self.stage1Latents = stage1Latents
    }
}

public struct LTXDistilledVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let latents: MLXArray

    public init(frames: MLXArray, latents: MLXArray) {
        self.frames = frames
        self.latents = latents
    }
}

public enum LTXDistilledLatentGeneratorError: LocalizedError {
    case transformerWeightsMissing(URL)
    case upsamplerWeightsMissing(URL)
    case unsupportedLTX23SplitModel(URL)
    case generatorNotLoaded
    case invalidResolution(width: Int, height: Int)
    case invalidFrameCount(Int)
    case invalidImageStrength(Float)
    case invalidImageFrameIndex(Int)
    case imageNotFound(URL)
    case imageDecodeFailed(URL)
    case emptyPrompt
    case decoderNotLoaded
    case encoderNotLoaded
    case upsamplerNotLoaded

    public var errorDescription: String? {
        switch self {
        case .transformerWeightsMissing(let url):
            return "Missing LTX transformer weights at \(url.path)"
        case .upsamplerWeightsMissing(let url):
            return "Missing LTX upsampler weights at \(url.path)"
        case .unsupportedLTX23SplitModel(let url):
            return """
            Detected an LTX 2.3 split MLX model at \(url.path). This native loader still supports the older \
            merged LTX layout; port the LTX 2.3 V2 connector, transformer, and split component loader before \
            generation.
            """
        case .generatorNotLoaded:
            return "LTX distilled latent generator is not loaded."
        case .invalidResolution(let width, let height):
            return "Resolution must be divisible by 64 (got \(width)x\(height))."
        case .invalidFrameCount(let value):
            return "numFrames must satisfy 8n+1 and be >= 9 (got \(value))."
        case .invalidImageStrength(let value):
            return "imageStrength must be in [0, 1] (got \(value))."
        case .invalidImageFrameIndex(let value):
            return "imageFrameIndex must be >= 0 (got \(value))."
        case .imageNotFound(let url):
            return "Source image not found: \(url.path)"
        case .imageDecodeFailed(let url):
            return "Could not decode source image: \(url.path)"
        case .emptyPrompt:
            return "Prompt cannot be empty."
        case .decoderNotLoaded:
            return "LTX decoder is not loaded."
        case .encoderNotLoaded:
            return "LTX encoder is not loaded."
        case .upsamplerNotLoaded:
            return "LTX latent upsampler is not loaded."
        }
    }
}

public func isLTX23SplitModelRoot(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
    let root = rootURL.standardizedFileURL
    let splitModel = root.appendingPathComponent("split_model.json", isDirectory: false)
    let transformer = root.appendingPathComponent("transformer-distilled.safetensors", isDirectory: false)
    guard fileManager.fileExists(atPath: splitModel.path),
          fileManager.fileExists(atPath: transformer.path) else {
        return false
    }

    let config = root.appendingPathComponent("config.json", isDirectory: false)
    guard let data = try? Data(contentsOf: config),
          let text = String(data: data, encoding: .utf8) else {
        return true
    }
    return text.contains(#""model_version""#) && text.contains("2.3")
}

public actor LTXDistilledLatentGenerator {
    private var textEncoder: LTXGemmaTextEncoder?
    private var transformer: LTXDistilledTransformer?
    private var decoder: LTXVideoDecoder?
    private var encoder: LTXVideoEncoder?
    private var upsampler: LTXLatentUpsampler?
    private var modelWeightsURL: URL?
    private var loadedDType: DType = .bfloat16
    private var loadedRoot: URL?

    public init() {}

    public func load(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        maxTextLength _: Int = 1024
    ) async throws {
        let root = modelRoot.standardizedFileURL
        if isLTX23SplitModelRoot(root) {
            throw LTXDistilledLatentGeneratorError.unsupportedLTX23SplitModel(root)
        }
        let transformerURL = root.appendingPathComponent("ltx-2-19b-distilled.safetensors", isDirectory: false)
        let upsamplerURL = root.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors", isDirectory: false)
        guard FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw LTXDistilledLatentGeneratorError.transformerWeightsMissing(transformerURL)
        }
        guard FileManager.default.fileExists(atPath: upsamplerURL.path) else {
            throw LTXDistilledLatentGeneratorError.upsamplerWeightsMissing(upsamplerURL)
        }

        let text = LTXGemmaTextEncoder()
        try await text.load(modelRoot: root, dtype: dtype, loadConnectorWeights: true)

        let model = LTXDistilledTransformer()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: transformerURL,
            to: model,
            dtype: dtype,
            verify: .none,
            include: { key in
                key.hasPrefix("model.diffusion_model.")
            },
            mapper: { key, value in
                mapDistilledTransformerWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 24
        )

        let vaeDecoder = LTXVideoDecoder(timestepConditioning: false)
        let decoderStats = try SafetensorsStreamingLoader.loadArrays(
            url: transformerURL,
            where: { key in
                key == "latents_mean"
                    || key == "latents_std"
                    || key == "vae.per_channel_statistics.mean-of-means"
                    || key == "vae.per_channel_statistics.std-of-means"
            },
            dtype: .float32
        )

        if let mean = decoderStats["latents_mean"] ?? decoderStats["vae.per_channel_statistics.mean-of-means"] {
            vaeDecoder.latentsMean = mean.asType(.float32)
        }
        if let std = decoderStats["latents_std"] ?? decoderStats["vae.per_channel_statistics.std-of-means"] {
            vaeDecoder.latentsStd = std.asType(.float32)
        }

        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: transformerURL,
            to: vaeDecoder,
            dtype: dtype,
            verify: .none,
            include: { key in
                key.hasPrefix("decoder.") || key.hasPrefix("vae.decoder.")
            },
            mapper: { key, value in
                mapLTXDecoderWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 24
        )

        let latentUpsampler = LTXLatentUpsampler(inChannels: 128, midChannels: 1024, numBlocksPerStage: 4)
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: upsamplerURL,
            to: latentUpsampler,
            dtype: dtype,
            verify: .none,
            include: { _ in true },
            mapper: { key, value in
                mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype)
            },
            batchSize: 24
        )

        self.textEncoder = text
        self.transformer = model
        self.decoder = vaeDecoder
        self.upsampler = latentUpsampler
        self.encoder = nil
        self.modelWeightsURL = transformerURL
        self.loadedDType = dtype
        self.loadedRoot = root
    }

    public func unload() async {
        if let textEncoder {
            await textEncoder.unload()
        }
        textEncoder = nil
        transformer = nil
        decoder = nil
        encoder = nil
        upsampler = nil
        modelWeightsURL = nil
        loadedRoot = nil
        Memory.clearCache()
    }

    private func loadEncoderIfNeeded() throws {
        if encoder != nil {
            return
        }
        guard let modelWeightsURL else {
            throw LTXDistilledLatentGeneratorError.encoderNotLoaded
        }

        let vaeEncoder = LTXVideoEncoder()
        if let decoder {
            vaeEncoder.latentsMean = decoder.latentsMean.asType(.float32)
            vaeEncoder.latentsStd = decoder.latentsStd.asType(.float32)
        } else {
            let stats = try SafetensorsStreamingLoader.loadArrays(
                url: modelWeightsURL,
                where: { key in
                    key == "latents_mean"
                        || key == "latents_std"
                        || key == "vae.per_channel_statistics.mean-of-means"
                        || key == "vae.per_channel_statistics.std-of-means"
                },
                dtype: .float32
            )
            if let mean = stats["latents_mean"] ?? stats["vae.per_channel_statistics.mean-of-means"] {
                vaeEncoder.latentsMean = mean.asType(.float32)
            }
            if let std = stats["latents_std"] ?? stats["vae.per_channel_statistics.std-of-means"] {
                vaeEncoder.latentsStd = std.asType(.float32)
            }
        }

        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: modelWeightsURL,
            to: vaeEncoder,
            dtype: loadedDType,
            verify: .none,
            include: { key in
                key.hasPrefix("encoder.") || key.hasPrefix("vae.encoder.")
            },
            mapper: { key, value in
                mapLTXEncoderWeight(key: key, value: value, dtype: loadedDType)
            },
            batchSize: 24
        )

        encoder = vaeEncoder
    }

    public func generate(
        options: LTXDistilledLatentGenerationOptions
    ) async throws -> LTXDistilledLatentGenerationResult {
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw LTXDistilledLatentGeneratorError.emptyPrompt
        }

        guard options.width % 64 == 0, options.height % 64 == 0 else {
            throw LTXDistilledLatentGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXDistilledLatentGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXDistilledLatentGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXDistilledLatentGeneratorError.invalidImageStrength(options.imageStrength)
        }

        guard let textEncoder, let transformer else {
            throw LTXDistilledLatentGeneratorError.generatorNotLoaded
        }
        guard let decoder else {
            throw LTXDistilledLatentGeneratorError.decoderNotLoaded
        }
        guard let upsampler else {
            throw LTXDistilledLatentGeneratorError.upsamplerNotLoaded
        }

        let encoding = try await textEncoder.encode(prompt: prompt, maxLength: options.maxTextLength)
        let context = encoding.videoEmbeddings
        let debugDenoise = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DENOISE"] == "1"
        if debugDenoise {
            print("[LTX] context shape=\(context.shape) \(tensorStatsString(context))")
        }

        let latentFrames = 1 + ((options.numFrames - 1) / 8)
        let stage1H = options.height / 2 / 32
        let stage1W = options.width / 2 / 32
        let stage2H = options.height / 32
        let stage2W = options.width / 32

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))

        let modelDType = context.dtype
        let isImageToVideo = options.sourceImageURL != nil
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?

        var latents: MLXArray
        if isImageToVideo {
            let sourceImageURL = options.sourceImageURL!
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw LTXDistilledLatentGeneratorError.imageNotFound(sourceImageURL)
            }
            try loadEncoderIfNeeded()
            guard let encoder else {
                throw LTXDistilledLatentGeneratorError.encoderNotLoaded
            }
            if options.imageFrameIndex >= latentFrames {
                throw LTXDistilledLatentGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
            }

            let stage1Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: options.width / 2,
                height: options.height / 2,
                dtype: modelDType
            )
            let stage1ImageLatent = encoder.encode(image: stage1Image)

            let stage2Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: options.width,
                height: options.height,
                dtype: modelDType
            )
            let stage2ImageLatent = encoder.encode(image: stage2Image)
            stage2ConditioningLatent = stage2ImageLatent

            // Optional end keyframe -> conditions the tail latent frame so the clip
            // interpolates a directed start->end motion.
            var stage1EndImageLatent: MLXArray?
            if let endImageURL = options.endImageURL {
                let stage1EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: options.width / 2,
                    height: options.height / 2,
                    dtype: modelDType
                )
                stage1EndImageLatent = encoder.encode(image: stage1EndImage)
                let stage2EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: options.width,
                    height: options.height,
                    dtype: modelDType
                )
                stage2EndConditioningLatent = encoder.encode(image: stage2EndImage)
            }

            var state1 = applyLatentConditioning(
                baseLatent: MLX.zeros([1, 128, latentFrames, stage1H, stage1W], dtype: modelDType),
                conditionedLatent: stage1ImageLatent,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage1EndImageLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
            let stage1Noise = MLXRandom.normal(state1.latent.shape).asType(modelDType)
            let stage1Sigma = MLXArray(STAGE1Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = state1.denoiseMask * stage1Sigma
            state1.latent = stage1Noise * scaledMask + state1.latent * (one - scaledMask)
            latents = state1.latent
            MLX.eval(latents)
            stage1ConditioningState = state1

        } else {
            latents = MLXRandom.normal([1, 128, latentFrames, stage1H, stage1W]).asType(modelDType)
            MLX.eval(latents)
        }

        let stage1Positions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1H,
            width: stage1W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        let stage1Rope = precomputeSplitRope(
            positions: stage1Positions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )

        latents = denoiseLoop(
            latents: latents,
            rope: stage1Rope,
            context: context,
            transformer: transformer,
            label: "stage1",
            sigmas: STAGE1Sigmas,
            conditioning: stage1ConditioningState
        )
        MLX.eval(latents)

        let stage1Latents = latents
        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            try? MLX.save(array: stage1Latents, url: parent.appendingPathComponent("\(stem)_stage1_latents.npy"))
        }

        // Python reference initializes and loads stage-2 models after stage 1.
        // Replay that init/load flow to keep stage-2 RNG stream aligned.
        guard let modelWeightsURL else {
            throw LTXDistilledLatentGeneratorError.generatorNotLoaded
        }
        guard let loadedRoot else {
            throw LTXDistilledLatentGeneratorError.generatorNotLoaded
        }
        let upsamplerWeightsURL = loadedRoot.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors", isDirectory: false)
        try advanceRandomStreamForPythonParityAfterStage1(
            modelWeightsURL: modelWeightsURL,
            upsamplerWeightsURL: upsamplerWeightsURL,
            dtype: loadedDType
        )

        latents = upsampleLatents(
            latents,
            upsampler: upsampler,
            latentMean: decoder.latentsMean,
            latentStd: decoder.latentsStd
        )
        MLX.eval(latents)
        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            try? MLX.save(array: latents, url: parent.appendingPathComponent("\(stem)_upsampled_latents.npy"))
        }

        if let stage2State = stage2ConditioningLatent.map({
            applyLatentConditioning(
                baseLatent: latents,
                conditionedLatent: $0,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage2EndConditioningLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
        }) {
            let noise = MLXRandom.normal(latents.shape).asType(modelDType)
            let noiseScale = MLXArray(STAGE2Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = stage2State.denoiseMask * noiseScale
            latents = noise * scaledMask + stage2State.latent * (one - scaledMask)
            MLX.eval(latents)
            stage2ConditioningState = stage2State
        } else {
            let noiseScale = MLXArray(STAGE2Sigmas[0]).asType(modelDType)
            let oneMinusScale = MLXArray(1.0 - STAGE2Sigmas[0]).asType(modelDType)
            let noise = MLXRandom.normal(latents.shape).asType(modelDType)
            latents = noise * noiseScale + latents * oneMinusScale
            MLX.eval(latents)
        }
        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            try? MLX.save(array: latents, url: parent.appendingPathComponent("\(stem)_stage2_init_latents.npy"))
        }

        let stage2Positions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2H,
            width: stage2W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        let stage2Rope = precomputeSplitRope(
            positions: stage2Positions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )

        latents = denoiseLoop(
            latents: latents,
            rope: stage2Rope,
            context: context,
            transformer: transformer,
            label: "stage2",
            sigmas: STAGE2Sigmas,
            conditioning: stage2ConditioningState
        )
        MLX.eval(latents)

        return LTXDistilledLatentGenerationResult(latents: latents, stage1Latents: stage1Latents)
    }

    public func generateVideo(
        options: LTXDistilledLatentGenerationOptions
    ) async throws -> LTXDistilledVideoGenerationResult {
        guard let decoder else {
            throw LTXDistilledLatentGeneratorError.decoderNotLoaded
        }

        let latentResult = try await generate(options: options)
        let decoded: MLXArray?
        let frames: MLXArray
        if let tiling = selectDecodeTilingConfig(
            width: options.width,
            height: options.height,
            numFrames: options.numFrames,
            fps: options.fps
        ) {
            decoded = nil
            frames = decodeWithTiling(
                decoder: decoder,
                latents: latentResult.latents,
                spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                spatialScale: 32,
                temporalScale: 8
            )
        } else {
            let fullDecoded = decoder.decode(sample: latentResult.latents, timestep: nil)
            decoded = fullDecoded
            frames = postprocessDecodedVideo(fullDecoded)
        }
        MLX.eval(frames)

        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            if let decoded {
                try? MLX.save(array: decoded, url: parent.appendingPathComponent("\(stem)_decoded.npy"))
            }
            try? MLX.save(array: frames, url: parent.appendingPathComponent("\(stem)_frames_postprocess.npy"))
        }
        return LTXDistilledVideoGenerationResult(frames: frames, latents: latentResult.latents)
    }
}

private func advanceRandomStreamForPythonParityAfterStage1(
    modelWeightsURL: URL,
    upsamplerWeightsURL: URL,
    dtype: DType
) throws {
    let dummyUpsampler = LTXLatentUpsampler(inChannels: 128, midChannels: 1024, numBlocksPerStage: 4)
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: upsamplerWeightsURL,
        to: dummyUpsampler,
        dtype: dtype,
        verify: .none,
        include: { _ in true },
        mapper: { key, value in
            mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype)
        },
        batchSize: 24
    )
    MLX.eval(dummyUpsampler)

    _ = modelWeightsURL
}

private let STAGE1Sigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
private let STAGE2Sigmas: [Float] = [0.909375, 0.725, 0.421875, 0.0]

private struct LTXLatentConditioningState {
    var latent: MLXArray
    var cleanLatent: MLXArray
    var denoiseMask: MLXArray
}

private final class LTXDistilledTransformer: Module {
    let hiddenSize = 4096
    let heads = 32
    let headDim = 128
    let outChannels = 128
    let timestepScaleMultiplier: Float = 1000.0

    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "adaln_single") var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "caption_projection") var captionProjection: LTXPixArtTextProjection
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTXDistilledTransformerBlock]

    override init() {
        self._patchifyProj.wrappedValue = Linear(128, 4096, bias: true)
        self._adalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 6)
        self._captionProjection.wrappedValue = LTXPixArtTextProjection(inFeatures: 3840, hiddenSize: 4096, outFeatures: 4096, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([2, 4096], dtype: .float32)
        self._normOut.wrappedValue = LayerNorm(dimensions: 4096, eps: 1e-6, affine: false)
        self._projOut.wrappedValue = Linear(4096, 128, bias: true)
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXDistilledTransformerBlock(dim: 4096, heads: 32, headDim: 128)
        }
        super.init()
    }

    func forward(
        latent: MLXArray,
        timesteps: MLXArray,
        context: MLXArray,
        rope: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let batch = latent.dim(0)
        let tokenCount = latent.dim(1)

        var x = patchifyProj(latent)

        let scaledTimesteps = timesteps.asType(x.dtype) * MLXArray(timestepScaleMultiplier).asType(x.dtype)
        let (timeEmb, embeddedTime) = adalnSingle(timestep: scaledTimesteps.reshaped(-1), hiddenDType: x.dtype)
        let reshapedTimeEmb = timeEmb.reshaped(batch, tokenCount, -1)
        let reshapedEmbedded = embeddedTime.reshaped(batch, tokenCount, -1)

        let projectedContext = captionProjection(context).reshaped(batch, context.dim(1), hiddenSize)

        for block in transformerBlocks {
            x = block(
                x,
                context: projectedContext,
                timestepEmb: reshapedTimeEmb,
                rope: rope,
                contextMask: nil
            )
        }

        let timePairs = scaleShiftTable.reshaped(1, 1, 2, hiddenSize)
            + reshapedEmbedded.reshaped(batch, tokenCount, 1, hiddenSize)
        let shift = timePairs[0..., 0..., 0, 0...]
        let scale = timePairs[0..., 0..., 1, 0...]

        x = normOut(x)
        x = x * (MLXArray(1.0).asType(x.dtype) + scale) + shift
        return projOut(x)
    }
}

private final class LTXDistilledTransformerBlock: Module {
    let dim: Int

    @ModuleInfo(key: "attn1") var attn1: LTXDistilledAttention
    @ModuleInfo(key: "attn2") var attn2: LTXDistilledAttention
    @ModuleInfo(key: "ff") var ff: LTXDistilledFeedForward
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(dim: Int, heads: Int, headDim: Int) {
        self.dim = dim
        self._attn1.wrappedValue = LTXDistilledAttention(queryDim: dim, contextDim: nil, heads: heads, headDim: headDim, normEps: 1e-6)
        self._attn2.wrappedValue = LTXDistilledAttention(queryDim: dim, contextDim: dim, heads: heads, headDim: headDim, normEps: 1e-6)
        self._ff.wrappedValue = LTXDistilledFeedForward(dim: dim, dimOut: dim, mult: 4)
        self._scaleShiftTable.wrappedValue = MLX.zeros([6, dim], dtype: .float32)
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        timestepEmb: MLXArray,
        rope: (cos: MLXArray, sin: MLXArray),
        contextMask: MLXArray?
    ) -> MLXArray {
        let batch = x.dim(0)
        let tokens = x.dim(1)

        let ada = scaleShiftTable.reshaped(1, 1, 6, dim) + timestepEmb.reshaped(batch, tokens, 6, dim)

        let shiftMSA = ada[0..., 0..., 0, 0...]
        let scaleMSA = ada[0..., 0..., 1, 0...]
        let gateMSA = ada[0..., 0..., 2, 0...]

        var h = rmsNormNoWeight(x)
        h = h * (MLXArray(1.0).asType(h.dtype) + scaleMSA) + shiftMSA
        h = attn1(h, context: nil, mask: nil, rope: rope)

        var out = x + h * gateMSA
        out = out + attn2(rmsNormNoWeight(out), context: context, mask: contextMask, rope: nil)

        let shiftMLP = ada[0..., 0..., 3, 0...]
        let scaleMLP = ada[0..., 0..., 4, 0...]
        let gateMLP = ada[0..., 0..., 5, 0...]

        var mlpInput = rmsNormNoWeight(out)
        mlpInput = mlpInput * (MLXArray(1.0).asType(mlpInput.dtype) + scaleMLP) + shiftMLP
        let mlpOut = ff(mlpInput)
        out = out + mlpOut * gateMLP

        return out
    }
}

private final class LTXDistilledAttention: Module {
    let heads: Int
    let headDim: Int
    let innerDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear?

    init(
        queryDim: Int,
        contextDim: Int?,
        heads: Int,
        headDim: Int,
        normEps: Float,
        applyGatedAttention: Bool = false
    ) {
        self.heads = heads
        self.headDim = headDim
        self.innerDim = heads * headDim

        let effectiveContextDim = contextDim ?? queryDim

        self._toQ.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(effectiveContextDim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(effectiveContextDim, innerDim, bias: true)
        self._qNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: normEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: normEps)
        self._toOut.wrappedValue = Linear(innerDim, queryDim, bias: true)
        self._toGateLogits.wrappedValue = applyGatedAttention ? Linear(queryDim, heads, bias: true) : nil
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray?,
        mask: MLXArray?,
        rope: (cos: MLXArray, sin: MLXArray)?,
        keyRope: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        let ctx = context ?? x

        let q = qNorm(toQ(x))
        let k = kNorm(toK(ctx))
        let v = toV(ctx)

        var qHeads = q.reshaped(q.dim(0), q.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        var kHeads = k.reshaped(k.dim(0), k.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        let vHeads = v.reshaped(v.dim(0), v.dim(1), heads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            qHeads = applySplitRoPEHeads(qHeads, cosFreq: rope.cos, sinFreq: rope.sin)
            let kRope = keyRope ?? rope
            kHeads = applySplitRoPEHeads(kHeads, cosFreq: kRope.cos, sinFreq: kRope.sin)
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: qHeads,
            keys: kHeads,
            values: vHeads,
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: mask.map { .array($0) } ?? .none
        )

        if let toGateLogits {
            let gate = MLXArray(2.0).asType(out.dtype) * MLX.sigmoid(toGateLogits(x).asType(out.dtype))
            out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: 3)
        }

        let merged = out.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), innerDim)
        return toOut(merged)
    }
}

private final class LTXDistilledFeedForward: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, dimOut: Int, mult: Int = 4, bias: Bool = true) {
        let inner = dim * mult
        self._projIn.wrappedValue = Linear(dim, inner, bias: bias)
        self._projOut.wrappedValue = Linear(inner, dimOut, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(geluApproximate(projIn(x)))
    }
}

private final class LTXPixArtTextProjection: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(inFeatures: Int, hiddenSize: Int, outFeatures: Int?, bias: Bool = true) {
        let out = outFeatures ?? hiddenSize
        self._linear1.wrappedValue = Linear(inFeatures, hiddenSize, bias: bias)
        self._linear2.wrappedValue = Linear(hiddenSize, out, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(geluApproximate(linear1(x)))
    }
}

private final class LTXAdaLayerNormSingle: Module {
    @ModuleInfo(key: "emb") var emb: LTXPixArtTimestepSizeEmbeddings
    @ModuleInfo(key: "linear") var linear: Linear

    init(embeddingDim: Int, embeddingCoefficient: Int) {
        self._emb.wrappedValue = LTXPixArtTimestepSizeEmbeddings(embeddingDim: embeddingDim)
        self._linear.wrappedValue = Linear(embeddingDim, embeddingCoefficient * embeddingDim, bias: true)
    }

    func callAsFunction(timestep: MLXArray, hiddenDType: DType? = nil) -> (MLXArray, MLXArray) {
        let embedded = emb(timestep: timestep, hiddenDType: hiddenDType)
        let params = linear(silu(embedded))
        return (params, embedded)
    }
}

private final class LTXPixArtTimestepSizeEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: LTXTimestepEmbedding

    init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue = LTXTimestepEmbedding(inChannels: 256, timeEmbedDim: embeddingDim, outDim: embeddingDim)
    }

    func callAsFunction(timestep: MLXArray, hiddenDType: DType?) -> MLXArray {
        var projected = getTimestepEmbedding(
            timesteps: timestep,
            embeddingDim: 256,
            flipSinToCos: true,
            downscaleFreqShift: 0,
            scale: 1,
            maxPeriod: 10_000
        )
        if let hiddenDType {
            projected = projected.asType(hiddenDType)
        }
        return timestepEmbedder(projected)
    }
}

private final class LTXTimestepEmbedding: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int, outDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, outDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

private func mapDistilledTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("model.diffusion_model.") else {
        return []
    }

    var mapped = String(key.dropFirst("model.diffusion_model.".count))
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    if mapped.hasPrefix("video_embeddings_connector") || mapped.hasPrefix("audio_embeddings_connector") {
        return []
    }
    if mapped.hasPrefix("text_embedding_projection") {
        return []
    }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

private func denoiseLoop(
    latents: MLXArray,
    rope: (cos: MLXArray, sin: MLXArray),
    context: MLXArray,
    transformer: LTXDistilledTransformer,
    label: String,
    sigmas: [Float],
    conditioning: LTXLatentConditioningState?
) -> MLXArray {
    var current = latents
    let dtype = latents.dtype
    let debugDenoise = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DENOISE"] == "1"
    let debugDumpPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DUMP_PREFIX"]
    let debugDumpAll = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DUMP_ALL"] == "1"

    for i in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[i]
        let nextSigma = sigmas[i + 1]

        let b = current.dim(0)
        let c = current.dim(1)
        let f = current.dim(2)
        let h = current.dim(3)
        let w = current.dim(4)
        let tokenCount = f * h * w

        let flat = current.transposed(0, 2, 3, 4, 1).reshaped(b, tokenCount, c)
        let timesteps: MLXArray
        if let conditioning {
            let mask = conditioning.denoiseMask.reshaped(b, 1, f, 1, 1)
            let broadcastMask = broadcast(mask, to: [b, 1, f, h, w]).reshaped(b, tokenCount)
            timesteps = MLXArray(sigma).asType(dtype) * broadcastMask
        } else {
            timesteps = MLX.full([b, tokenCount], values: MLXArray(sigma).asType(dtype))
        }

        let velocity = transformer.forward(latent: flat, timesteps: timesteps, context: context, rope: rope)
            .reshaped(b, f, h, w, c)
            .transposed(0, 4, 1, 2, 3)
        MLX.eval(velocity)

        if let debugDumpPrefix, !debugDumpPrefix.isEmpty, (debugDumpAll || i == 0) {
            let base = URL(fileURLWithPath: debugDumpPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            let step = i + 1
            let currentURL = parent.appendingPathComponent("\(stem)_\(label)_step\(step)_current.npy")
            let timestepsURL = parent.appendingPathComponent("\(stem)_\(label)_step\(step)_timesteps.npy")
            let velocityURL = parent.appendingPathComponent("\(stem)_\(label)_step\(step)_velocity.npy")
            let contextURL = parent.appendingPathComponent("\(stem)_\(label)_context.npy")
            let ropeCosURL = parent.appendingPathComponent("\(stem)_\(label)_rope_cos.npy")
            let ropeSinURL = parent.appendingPathComponent("\(stem)_\(label)_rope_sin.npy")
            try? MLX.save(array: current, url: currentURL)
            try? MLX.save(array: timesteps, url: timestepsURL)
            try? MLX.save(array: velocity, url: velocityURL)
            if i == 0 {
                try? MLX.save(array: context, url: contextURL)
                try? MLX.save(array: rope.cos, url: ropeCosURL)
                try? MLX.save(array: rope.sin, url: ropeSinURL)
            }
        }

        var denoised = toDenoised(noisy: current, velocity: velocity, sigma: sigma)
        MLX.eval(denoised)
        if debugDenoise {
            print("[LTX] \(label) step \(i + 1)/\(sigmas.count - 1) sigma=\(sigma) next=\(nextSigma)")
            print("[LTX] \(label) current \(tensorStatsString(current))")
            print("[LTX] \(label) velocity \(tensorStatsString(velocity))")
            print("[LTX] \(label) denoised \(tensorStatsString(denoised))")
        }
        if let conditioning {
            let one = MLXArray(1.0).asType(denoised.dtype)
            denoised = denoised * conditioning.denoiseMask + conditioning.cleanLatent * (one - conditioning.denoiseMask)
        }
        if nextSigma > 0 {
            let sigmaArr = MLXArray(sigma).asType(dtype)
            let nextArr = MLXArray(nextSigma).asType(dtype)
            current = denoised + nextArr * (current - denoised) / sigmaArr
        } else {
            current = denoised
        }
        MLX.eval(current)
    }

    return current
}

private func upsampleLatents(
    _ latents: MLXArray,
    upsampler: LTXLatentUpsampler,
    latentMean: MLXArray,
    latentStd: MLXArray
) -> MLXArray {
    let dtype = latents.dtype
    let mean = latentMean.asType(dtype).reshaped(1, -1, 1, 1, 1)
    let std = latentStd.asType(dtype).reshaped(1, -1, 1, 1, 1)

    var x = latents
    x = x * std + mean
    x = upsampler(x)
    x = (x - mean) / std
    return x
}

private func applyLatentConditioning(
    baseLatent: MLXArray,
    conditionedLatent: MLXArray,
    frameIndex: Int,
    strength: Float,
    endConditionedLatent: MLXArray? = nil,
    endFrameIndex: Int = -1,
    endStrength: Float = 1.0
) -> LTXLatentConditioningState {
    let b = baseLatent.dim(0)
    let c = baseLatent.dim(1)
    let f = baseLatent.dim(2)
    let h = baseLatent.dim(3)
    let w = baseLatent.dim(4)
    let condFrames = conditionedLatent.dim(2)
    let dtype = baseLatent.dtype

    let condEnd = min(frameIndex + condFrames, f)
    let oneMinusStrength = MLXArray(1.0 - strength).asType(dtype)

    // Optional end keyframe: condition a second image at the tail of the clip so
    // LTX interpolates a directed start->end motion. Defaults to the last latent
    // frame(s). Start conditioning takes precedence on any overlap.
    let endCondFrames = endConditionedLatent?.dim(2) ?? 0
    let endStart = endConditionedLatent != nil
        ? (endFrameIndex >= 0 ? endFrameIndex : max(0, f - endCondFrames))
        : f
    let endStop = min(endStart + endCondFrames, f)
    let oneMinusEndStrength = MLXArray(1.0 - endStrength).asType(dtype)

    var latentFrames: [MLXArray] = []
    var cleanFrames: [MLXArray] = []
    var maskFrames: [MLXArray] = []
    latentFrames.reserveCapacity(f)
    cleanFrames.reserveCapacity(f)
    maskFrames.reserveCapacity(f)

    for frame in 0..<f {
        if frame >= frameIndex, frame < condEnd {
            let condIdx = frame - frameIndex
            let condSlice = conditionedLatent[0..., 0..., condIdx..<condIdx + 1, 0..., 0...]
            latentFrames.append(condSlice)
            cleanFrames.append(condSlice)
            maskFrames.append(MLX.full([b, 1, 1, 1, 1], values: oneMinusStrength))
        } else if let endLatent = endConditionedLatent, frame >= endStart, frame < endStop {
            let condIdx = frame - endStart
            let condSlice = endLatent[0..., 0..., condIdx..<condIdx + 1, 0..., 0...]
            latentFrames.append(condSlice)
            cleanFrames.append(condSlice)
            maskFrames.append(MLX.full([b, 1, 1, 1, 1], values: oneMinusEndStrength))
        } else {
            latentFrames.append(baseLatent[0..., 0..., frame..<frame + 1, 0..., 0...])
            cleanFrames.append(MLX.zeros([b, c, 1, h, w], dtype: dtype))
            maskFrames.append(MLX.ones([b, 1, 1, 1, 1], dtype: dtype))
        }
    }

    return LTXLatentConditioningState(
        latent: MLX.concatenated(latentFrames, axis: 2),
        cleanLatent: MLX.concatenated(cleanFrames, axis: 2),
        denoiseMask: MLX.concatenated(maskFrames, axis: 2)
    )
}

private func loadImageForEncoding(
    url: URL,
    width: Int,
    height: Int,
    dtype: DType
) throws -> MLXArray {
    let image: MediaImage
    do {
        image = try MediaImageIO.resized(try MediaImageIO.decode(url), width: width, height: height)
    } catch {
        throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
    }

    let channels = MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: true)

    let chw = MLXArray(channels).reshaped(1, 3, height, width).asType(dtype)
    return chw.reshaped(1, 3, 1, height, width)
}

private func toDenoised(
    noisy: MLXArray,
    velocity: MLXArray,
    sigma: Float
) -> MLXArray {
    noisy - MLXArray(sigma).asType(velocity.dtype) * velocity
}

private func tensorStatsString(_ x: MLXArray) -> String {
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

private func rmsNormNoWeight(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    let normalized = x32 * rsqrt(variance + MLXArray(eps))
    return normalized.asType(dtype)
}

private func applySplitRoPEHeads(
    _ x: MLXArray,
    cosFreq: MLXArray,
    sinFreq: MLXArray
) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let cos32 = cosFreq.asType(.float32)
    let sin32 = sinFreq.asType(.float32)

    let halfDim = x32.dim(3) / 2
    let x1 = x32[0..., 0..., 0..., 0..<halfDim]
    let x2 = x32[0..., 0..., 0..., halfDim...]

    let out1 = x1 * cos32 - sin32 * x2
    let out2 = x2 * cos32 + sin32 * x1

    return MLX.concatenated([out1, out2], axis: 3).asType(dtype)
}

private func precomputeSplitRope(
    positions: MLXArray,
    dim: Int,
    theta: Float,
    maxPos: [Int],
    numHeads: Int
) -> (cos: MLXArray, sin: MLXArray) {
    let batch = positions.dim(0)
    let positionDims = positions.dim(1)
    let tokenCount = positions.dim(2)

    let nElem = 2 * positionDims
    var indexCount = dim / nElem
    if indexCount == 0 {
        indexCount = 1
    }

    var indices = [Double](repeating: 0.0, count: indexCount)
    if indexCount == 1 {
        indices[0] = Double.pi / 2.0
    } else {
        for i in 0..<indexCount {
            let fraction = Double(i) / Double(indexCount - 1)
            indices[i] = pow(Double(theta), fraction) * (Double.pi / 2.0)
        }
    }

    let rawPositions = positions.asType(.float32).asArray(Float.self)

    let expectedFreqs = dim / 2
    let currentFreqs = positionDims * indexCount
    let padSize = max(0, expectedFreqs - currentFreqs)

    let perHeadFreq = expectedFreqs / numHeads
    let halfHead = perHeadFreq

    var cosValues = [Float](repeating: 1.0, count: batch * numHeads * tokenCount * halfHead)
    var sinValues = [Float](repeating: 0.0, count: batch * numHeads * tokenCount * halfHead)

    for b in 0..<batch {
        for t in 0..<tokenCount {
            var scaledPositions = [Float](repeating: 0.0, count: positionDims)
            for p in 0..<positionDims {
                let startIndex = ((b * positionDims + p) * tokenCount + t) * 2
                let startValue = rawPositions[startIndex]
                let endValue = rawPositions[startIndex + 1]
                let middle = (startValue + endValue) * 0.5
                scaledPositions[p] = (middle / Float(maxPos[p])) * 2.0 - 1.0
            }

            var freqVector = [Float](repeating: 0.0, count: currentFreqs)
            var cursor = 0
            // Match Python ordering after swapaxes(freqs, -1, -2): for each index, iterate position dims.
            for i in 0..<indexCount {
                let indexValue = indices[i]
                for p in 0..<positionDims {
                    freqVector[cursor] = Float(Double(scaledPositions[p]) * indexValue)
                    cursor += 1
                }
            }

            var paddedFreq = [Float](repeating: 0.0, count: expectedFreqs)
            if padSize > 0 {
                for i in 0..<min(currentFreqs, expectedFreqs - padSize) {
                    paddedFreq[padSize + i] = freqVector[i]
                }
            } else {
                for i in 0..<min(currentFreqs, expectedFreqs) {
                    paddedFreq[i] = freqVector[i]
                }
            }

            for h in 0..<numHeads {
                for d in 0..<halfHead {
                    let src = h * halfHead + d
                    let phase = src < paddedFreq.count ? paddedFreq[src] : 0
                    let dst = ((b * numHeads + h) * tokenCount + t) * halfHead + d
                    cosValues[dst] = Foundation.cos(phase)
                    sinValues[dst] = Foundation.sin(phase)
                }
            }
        }
    }

    let cos = MLXArray(cosValues).reshaped(batch, numHeads, tokenCount, halfHead)
    let sin = MLXArray(sinValues).reshaped(batch, numHeads, tokenCount, halfHead)
    return (cos, sin)
}

private final class LTXCausalConv3d: Module {
    enum SpatialPaddingMode {
        case zeros
        case reflect
    }

    @ModuleInfo(key: "conv") var conv: Conv3d

    let kernelSize: (Int, Int, Int)
    let stride: (Int, Int, Int)
    let temporalPadding: Int
    let spatialPadding: (Int, Int)
    let spatialPaddingMode: SpatialPaddingMode

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        spatialPadding: (Int, Int) = (1, 1),
        spatialPaddingMode: SpatialPaddingMode = .zeros
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.temporalPadding = kernelSize.0 - 1
        self.spatialPadding = spatialPadding
        self.spatialPaddingMode = spatialPaddingMode

        self._conv.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init([kernelSize.0, kernelSize.1, kernelSize.2]),
            stride: .init([stride.0, stride.1, stride.2]),
            padding: .init(0),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        var hidden = x

        if kernelSize.0 > 1 {
            if causal {
                if temporalPadding > 0 {
                    let firstFrame = hidden[0..., 0..., 0..<1, 0..., 0...]
                    let repeated = tiled(firstFrame, repetitions: [1, 1, temporalPadding, 1, 1])
                    hidden = MLX.concatenated([repeated, hidden], axis: 2)
                }
            } else {
                let padSize = (kernelSize.0 - 1) / 2
                if padSize > 0 {
                    let firstFrame = hidden[0..., 0..., 0..<1, 0..., 0...]
                    let lastFrame = hidden[0..., 0..., (hidden.dim(2) - 1)..., 0..., 0...]
                    let front = tiled(firstFrame, repetitions: [1, 1, padSize, 1, 1])
                    let back = tiled(lastFrame, repetitions: [1, 1, padSize, 1, 1])
                    hidden = MLX.concatenated([front, hidden, back], axis: 2)
                }
            }
        }

        hidden = hidden.transposed(0, 2, 3, 4, 1)
        if spatialPadding.0 > 0 || spatialPadding.1 > 0 {
            switch spatialPaddingMode {
            case .zeros:
                hidden = padded(hidden, widths: [
                    [0, 0],
                    [0, 0],
                    [spatialPadding.0, spatialPadding.0],
                    [spatialPadding.1, spatialPadding.1],
                    [0, 0],
                ])
            case .reflect:
                hidden = reflectPad2DNDHWC(hidden, padH: spatialPadding.0, padW: spatialPadding.1)
            }
        }

        hidden = conv(hidden)
        return hidden.transposed(0, 4, 1, 2, 3)
    }
}

private func reflectPad2DNDHWC(_ x: MLXArray, padH: Int, padW: Int) -> MLXArray {
    var y = x

    if padH > 0 {
        let top = y[0..., 0..., 1..<(padH + 1), 0..., 0...]
        let bottomStart = max(0, y.dim(2) - padH - 1)
        let bottomEnd = max(bottomStart, y.dim(2) - 1)
        let bottom = y[0..., 0..., bottomStart..<bottomEnd, 0..., 0...]
        let reverseH = MLXArray(Array(stride(from: padH - 1, through: 0, by: -1)).map(Int32.init))
        let topReflected = top.take(reverseH, axis: 2)
        let bottomReflected = bottom.take(reverseH, axis: 2)
        y = MLX.concatenated([topReflected, y, bottomReflected], axis: 2)
    }

    if padW > 0 {
        let left = y[0..., 0..., 0..., 1..<(padW + 1), 0...]
        let rightStart = max(0, y.dim(3) - padW - 1)
        let rightEnd = max(rightStart, y.dim(3) - 1)
        let right = y[0..., 0..., 0..., rightStart..<rightEnd, 0...]
        let reverseW = MLXArray(Array(stride(from: padW - 1, through: 0, by: -1)).map(Int32.init))
        let leftReflected = left.take(reverseW, axis: 3)
        let rightReflected = right.take(reverseW, axis: 3)
        y = MLX.concatenated([leftReflected, y, rightReflected], axis: 3)
    }

    return y
}

private final class LTXResnetBlock3DSimple: Module {
    @ModuleInfo(key: "conv1") var conv1: LTXCausalConv3d
    @ModuleInfo(key: "conv2") var conv2: LTXCausalConv3d
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    let channels: Int
    let timestepConditioning: Bool

    init(
        channels: Int,
        timestepConditioning: Bool,
        spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = .zeros
    ) {
        self.channels = channels
        self.timestepConditioning = timestepConditioning
        self._conv1.wrappedValue = LTXCausalConv3d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._conv2.wrappedValue = LTXCausalConv3d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._scaleShiftTable.wrappedValue = MLX.zeros([4, channels], dtype: .float32)
    }

    func callAsFunction(
        _ x: MLXArray,
        causal: Bool,
        timestepEmbedding: MLXArray?
    ) -> MLXArray {
        let residual = x
        var h = pixelNormChannels(x)

        if timestepConditioning, let timestepEmbedding {
            let batch = x.dim(0)
            let ada = scaleShiftTable.reshaped(1, 4, channels, 1, 1)
                + timestepEmbedding.reshaped(batch, 4, channels, 1, 1)
            let shift1 = ada[0..., 0, 0..., 0..., 0...]
            let scale1 = ada[0..., 1, 0..., 0..., 0...]
            h = h * (MLXArray(1.0).asType(h.dtype) + scale1) + shift1
        }

        h = silu(h)
        h = conv1(h, causal: causal)
        h = pixelNormChannels(h)

        if timestepConditioning, let timestepEmbedding {
            let batch = x.dim(0)
            let ada = scaleShiftTable.reshaped(1, 4, channels, 1, 1)
                + timestepEmbedding.reshaped(batch, 4, channels, 1, 1)
            let shift2 = ada[0..., 2, 0..., 0..., 0...]
            let scale2 = ada[0..., 3, 0..., 0..., 0...]
            h = h * (MLXArray(1.0).asType(h.dtype) + scale2) + shift2
        }

        h = silu(h)
        h = conv2(h, causal: causal)
        return residual + h
    }
}

enum LTXVideoVAEArchitecture: Equatable {
    case legacy
    case ltx23Split
}

private final class LTXVideoEncoder: Module {
    let patchSize: Int = 4
    let latentChannels: Int = 128

    @ModuleInfo(key: "conv_in") var convIn: LTXCausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [LTXEncoderBlock]
    @ModuleInfo(key: "conv_out") var convOut: LTXCausalConv3d

    var latentsMean: MLXArray = MLX.zeros([128], dtype: .float32)
    var latentsStd: MLXArray = MLX.ones([128], dtype: .float32)

    init(architecture: LTXVideoVAEArchitecture = .legacy) {
        self._convIn.wrappedValue = LTXCausalConv3d(
            inChannels: 3 * 4 * 4,
            outChannels: 128,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1)
        )

        switch architecture {
        case .legacy:
            self._downBlocks.wrappedValue = [
                LTXEncoderBlock(channels: 128, numLayers: 4),
                LTXEncoderBlock(inChannels: 128, outChannels: 256, stride: (1, 2, 2)),
                LTXEncoderBlock(channels: 256, numLayers: 6),
                LTXEncoderBlock(inChannels: 256, outChannels: 512, stride: (2, 1, 1)),
                LTXEncoderBlock(channels: 512, numLayers: 6),
                LTXEncoderBlock(inChannels: 512, outChannels: 1024, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 1024, numLayers: 2),
                LTXEncoderBlock(inChannels: 1024, outChannels: 2048, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 2048, numLayers: 2),
            ]

            self._convOut.wrappedValue = LTXCausalConv3d(
                inChannels: 2048,
                outChannels: 129,
                kernelSize: (3, 3, 3),
                stride: (1, 1, 1),
                spatialPadding: (1, 1)
            )

        case .ltx23Split:
            self._downBlocks.wrappedValue = [
                LTXEncoderBlock(channels: 128, numLayers: 4),
                LTXEncoderBlock(inChannels: 128, outChannels: 256, stride: (1, 2, 2)),
                LTXEncoderBlock(channels: 256, numLayers: 6),
                LTXEncoderBlock(inChannels: 256, outChannels: 512, stride: (2, 1, 1)),
                LTXEncoderBlock(channels: 512, numLayers: 4),
                LTXEncoderBlock(inChannels: 512, outChannels: 1024, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 1024, numLayers: 2),
                LTXEncoderBlock(inChannels: 1024, outChannels: 1024, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 1024, numLayers: 2),
            ]

            self._convOut.wrappedValue = LTXCausalConv3d(
                inChannels: 1024,
                outChannels: 129,
                kernelSize: (3, 3, 3),
                stride: (1, 1, 1),
                spatialPadding: (1, 1)
            )
        }

        super.init()
    }

    func encode(image: MLXArray) -> MLXArray {
        var sample = patchify3D(image, patchSizeHW: patchSize, patchSizeT: 1)
        sample = convIn(sample, causal: true)

        for block in downBlocks {
            sample = block(sample, causal: true)
        }

        sample = pixelNormChannels(sample)
        sample = silu(sample)
        sample = convOut(sample, causal: true)

        let means = sample[0..., 0..<latentChannels, 0..., 0..., 0...]
        return normalizeLatents(means)
    }

    private func normalizeLatents(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let mean = latentsMean.asType(.float32).reshaped(1, -1, 1, 1, 1)
        let std = latentsStd.asType(.float32).reshaped(1, -1, 1, 1, 1)
        return ((x.asType(.float32) - mean) / std).asType(dtype)
    }
}

private final class LTXEncoderBlock: Module {
    enum Kind {
        case resnetGroup
        case downsample
    }

    let kind: Kind
    let stride: (Int, Int, Int)
    let outChannels: Int
    let groupSize: Int

    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXResnetBlock3DSimple]
    @ModuleInfo(key: "conv") var conv: LTXCausalConv3d?

    init(channels: Int, numLayers: Int) {
        self.kind = .resnetGroup
        self.stride = (1, 1, 1)
        self.outChannels = channels
        self.groupSize = 1
        self._resBlocks.wrappedValue = (0..<numLayers).map { _ in
            LTXResnetBlock3DSimple(channels: channels, timestepConditioning: false)
        }
        self._conv.wrappedValue = nil
    }

    init(inChannels: Int, outChannels: Int, stride: (Int, Int, Int)) {
        self.kind = .downsample
        self.stride = stride
        self.outChannels = outChannels
        let multiplier = stride.0 * stride.1 * stride.2
        self.groupSize = max(1, inChannels * multiplier / outChannels)
        let convOutChannels = max(1, outChannels / multiplier)
        self._conv.wrappedValue = LTXCausalConv3d(
            inChannels: inChannels,
            outChannels: convOutChannels,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1)
        )
        self._resBlocks.wrappedValue = []
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        switch kind {
        case .resnetGroup:
            var h = x
            for block in resBlocks {
                h = block(h, causal: causal, timestepEmbedding: nil)
            }
            return h

        case .downsample:
            guard let conv else { return x }
            let st = stride.0
            let sh = stride.1
            let sw = stride.2

            var h = x
            if st == 2 {
                h = MLX.concatenated([h[0..., 0..., 0..<1, 0..., 0...], h], axis: 2)
            }

            let padD = (st - (h.dim(2) % st)) % st
            let padH = (sh - (h.dim(3) % sh)) % sh
            let padW = (sw - (h.dim(4) % sw)) % sw
            if padD > 0 || padH > 0 || padW > 0 {
                h = padded(h, widths: [
                    [0, 0],
                    [0, 0],
                    [0, padD],
                    [0, padH],
                    [0, padW],
                ])
            }

            let depthInput = spaceToDepth3D(h, stride: stride)
            let b = depthInput.dim(0)
            let d = depthInput.dim(2)
            let hh = depthInput.dim(3)
            let ww = depthInput.dim(4)

            let reduced = MLX.mean(
                depthInput.reshaped(b, outChannels, groupSize, d, hh, ww),
                axis: 2
            )
            let convOut = spaceToDepth3D(conv(h, causal: causal), stride: stride)
            return convOut + reduced
        }
    }
}

private final class LTXUpsamplerGroupNorm3d: Module {
    let numGroups: Int
    let numChannels: Int
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    init(numGroups: Int, numChannels: Int, eps: Float = 1e-5) {
        self.numGroups = numGroups
        self.numChannels = numChannels
        self.eps = eps
        self._weight.wrappedValue = MLX.ones([numChannels], dtype: .float32)
        self._bias.wrappedValue = MLX.zeros([numChannels], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 5, "Expected NDHWC tensor for GroupNorm3d")

        let n = x.dim(0)
        let d = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let c = x.dim(4)
        precondition(c == numChannels, "Channel mismatch for GroupNorm3d")
        precondition(c % numGroups == 0, "GroupNorm3d requires channels divisible by groups")

        let inputDType = x.dtype
        var y = x.asType(.float32).reshaped(n, d * h * w, numGroups, c / numGroups)
        let mean = MLX.mean(y, axes: [1, 3], keepDims: true)
        let variance = MLX.mean((y - mean) * (y - mean), axes: [1, 3], keepDims: true)
        y = (y - mean) / MLX.sqrt(variance + MLXArray(eps))
        y = y.reshaped(n, d, h, w, c)

        let weight = self.weight.asType(.float32).reshaped(1, 1, 1, 1, c)
        let bias = self.bias.asType(.float32).reshaped(1, 1, 1, 1, c)
        y = y * weight + bias
        return y.asType(inputDType)
    }
}

private final class LTXUpsamplerResBlock3D: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv3d
    @ModuleInfo(key: "norm1") var norm1: LTXUpsamplerGroupNorm3d
    @ModuleInfo(key: "conv2") var conv2: Conv3d
    @ModuleInfo(key: "norm2") var norm2: LTXUpsamplerGroupNorm3d

    init(channels: Int) {
        self._conv1.wrappedValue = Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._norm1.wrappedValue = LTXUpsamplerGroupNorm3d(numGroups: 32, numChannels: channels, eps: 1e-5)
        self._conv2.wrappedValue = Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._norm2.wrappedValue = LTXUpsamplerGroupNorm3d(numGroups: 32, numChannels: channels, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = conv1(x)
        h = norm1(h)
        h = silu(h)
        h = conv2(h)
        h = norm2(h)
        return silu(h + residual)
    }
}

private func pixelShuffle2D(_ x: MLXArray, upscaleFactor: Int = 2) -> MLXArray {
    let n = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    let r = upscaleFactor
    precondition(c % (r * r) == 0, "PixelShuffle2D channel count must be divisible by scale^2")

    let outC = c / (r * r)
    var y = x.reshaped(n, h, w, outC, r, r)
    y = y.transposed(0, 1, 4, 2, 5, 3)
    return y.reshaped(n, h * r, w * r, outC)
}

private final class LTXSpatialRationalResampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(midChannels: Int, scale _: Float = 2.0) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: midChannels,
            outputChannels: 4 * midChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 5, "Expected NDHWC tensor")

        let n = x.dim(0)
        let d = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let c = x.dim(4)

        var y = x.reshaped(n * d, h, w, c)
        y = conv(y)
        y = pixelShuffle2D(y, upscaleFactor: 2)
        return y.reshaped(n, d, h * 2, w * 2, c)
    }
}

private final class LTXLatentUpsampler: Module {
    let inChannels: Int
    let midChannels: Int
    let numBlocksPerStage: Int

    @ModuleInfo(key: "initial_conv") var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") var initialNorm: LTXUpsamplerGroupNorm3d
    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "upsampler") var upsampler: LTXSpatialRationalResampler
    @ModuleInfo(key: "post_upsample_res_blocks") var postUpsampleResBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "final_conv") var finalConv: Conv3d

    init(inChannels: Int = 128, midChannels: Int = 1024, numBlocksPerStage: Int = 4) {
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.numBlocksPerStage = numBlocksPerStage

        self._initialConv.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: midChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._initialNorm.wrappedValue = LTXUpsamplerGroupNorm3d(numGroups: 32, numChannels: midChannels, eps: 1e-5)
        self._resBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._upsampler.wrappedValue = LTXSpatialRationalResampler(midChannels: midChannels, scale: 2.0)
        self._postUpsampleResBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._finalConv.wrappedValue = Conv3d(
            inputChannels: midChannels,
            outputChannels: inChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
    }

    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        precondition(latent.ndim == 5, "Expected NCDHW latent tensor")

        var x = latent.transposed(0, 2, 3, 4, 1)
        x = initialConv(x)
        x = initialNorm(x)
        x = silu(x)

        for block in resBlocks {
            x = block(x)
        }

        x = upsampler(x)

        for block in postUpsampleResBlocks {
            x = block(x)
        }

        x = finalConv(x)
        return x.transposed(0, 4, 1, 2, 3)
    }
}

private final class LTXPixArtAlphaTimestepEmbedder: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: LTXTimestepEmbedding

    init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue = LTXTimestepEmbedding(
            inChannels: 256,
            timeEmbedDim: embeddingDim,
            outDim: embeddingDim
        )
    }

    func callAsFunction(_ timestep: MLXArray, hiddenDType: DType) -> MLXArray {
        let projected = getTimestepEmbedding(
            timesteps: timestep,
            embeddingDim: 256,
            flipSinToCos: true,
            downscaleFreqShift: 0,
            scale: 1,
            maxPeriod: 10_000
        ).asType(hiddenDType)
        return timestepEmbedder(projected)
    }
}

private final class LTXDecoderBlock: Module {
    enum Kind {
        case resnetGroup
        case upsample
    }

    let kind: Kind
    let channels: Int
    let timestepConditioning: Bool
    let stride: (Int, Int, Int)
    let residualEnabled: Bool
    let outChannelsReductionFactor: Int

    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXResnetBlock3DSimple]
    @ModuleInfo(key: "time_embedder") var timeEmbedder: LTXPixArtAlphaTimestepEmbedder?
    @ModuleInfo(key: "conv") var conv: LTXCausalConv3d?

    init(
        channels: Int,
        numLayers: Int,
        timestepConditioning: Bool,
        spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = .reflect
    ) {
        self.kind = .resnetGroup
        self.channels = channels
        self.timestepConditioning = timestepConditioning
        self.stride = (1, 1, 1)
        self.residualEnabled = false
        self.outChannelsReductionFactor = 1

        self._resBlocks.wrappedValue = (0..<numLayers).map { _ in
            LTXResnetBlock3DSimple(
                channels: channels,
                timestepConditioning: timestepConditioning,
                spatialPaddingMode: spatialPaddingMode
            )
        }
        self._timeEmbedder.wrappedValue = timestepConditioning
            ? LTXPixArtAlphaTimestepEmbedder(embeddingDim: channels * 4)
            : nil
        self._conv.wrappedValue = nil
    }

    init(
        inChannels: Int,
        stride: (Int, Int, Int),
        residual: Bool,
        outChannelsReductionFactor: Int,
        spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = .reflect
    ) {
        self.kind = .upsample
        self.channels = inChannels
        self.timestepConditioning = false
        self.stride = stride
        self.residualEnabled = residual
        self.outChannelsReductionFactor = outChannelsReductionFactor

        let multiplier = stride.0 * stride.1 * stride.2
        let outChannels = inChannels / outChannelsReductionFactor
        self._conv.wrappedValue = LTXCausalConv3d(
            inChannels: inChannels,
            outChannels: outChannels * multiplier,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._resBlocks.wrappedValue = []
        self._timeEmbedder.wrappedValue = nil
    }

    func callAsFunction(
        _ x: MLXArray,
        causal: Bool,
        timestep: MLXArray?
    ) -> MLXArray {
        switch kind {
        case .resnetGroup:
            var h = x
            let timestepEmbedding: MLXArray?
            if timestepConditioning, let timestep, let timeEmbedder {
                timestepEmbedding = timeEmbedder(timestep, hiddenDType: x.dtype)
            } else {
                timestepEmbedding = nil
            }
            for block in resBlocks {
                h = block(h, causal: causal, timestepEmbedding: timestepEmbedding)
            }
            return h

        case .upsample:
            guard let conv else { return x }
            let st = stride.0
            let sh = stride.1
            let sw = stride.2

            var residual: MLXArray?
            if residualEnabled {
                var up = depthToSpace3D(x, stride: stride)
                let repeats = max(1, (st * sh * sw) / outChannelsReductionFactor)
                up = tiled(up, repetitions: [1, repeats, 1, 1, 1])
                if st > 1 {
                    up = up[0..., 0..., 1..., 0..., 0...]
                }
                residual = up
            }

            var h = conv(x, causal: causal)
            h = depthToSpace3D(h, stride: stride)
            if st > 1 {
                h = h[0..., 0..., 1..., 0..., 0...]
            }
            if let residual {
                h = h + residual
            }
            return h
        }
    }
}

private final class LTXVideoDecoder: Module {
    let patchSize: Int = 4
    let timestepConditioning: Bool
    let decodeNoiseScale: Float = 0.025
    let decodeTimestep: Float = 0.05

    @ModuleInfo(key: "conv_in") var convIn: LTXCausalConv3d
    @ModuleInfo(key: "up_blocks") var upBlocks: [LTXDecoderBlock]
    @ModuleInfo(key: "conv_out") var convOut: LTXCausalConv3d
    @ModuleInfo(key: "last_time_embedder") var lastTimeEmbedder: LTXPixArtAlphaTimestepEmbedder
    @ModuleInfo(key: "last_scale_shift_table") var lastScaleShiftTable: MLXArray

    var latentsMean: MLXArray = MLX.zeros([128], dtype: .float32)
    var latentsStd: MLXArray = MLX.ones([128], dtype: .float32)

    init(timestepConditioning: Bool, architecture: LTXVideoVAEArchitecture = .legacy) {
        self.timestepConditioning = timestepConditioning
        let spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = architecture == .ltx23Split ? .zeros : .reflect
        self._convIn.wrappedValue = LTXCausalConv3d(
            inChannels: 128,
            outChannels: 1024,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        switch architecture {
        case .legacy:
            self._upBlocks.wrappedValue = [
                LTXDecoderBlock(channels: 1024, numLayers: 5, timestepConditioning: timestepConditioning),
                LTXDecoderBlock(inChannels: 1024, stride: (2, 2, 2), residual: true, outChannelsReductionFactor: 2),
                LTXDecoderBlock(channels: 512, numLayers: 5, timestepConditioning: timestepConditioning),
                LTXDecoderBlock(inChannels: 512, stride: (2, 2, 2), residual: true, outChannelsReductionFactor: 2),
                LTXDecoderBlock(channels: 256, numLayers: 5, timestepConditioning: timestepConditioning),
                LTXDecoderBlock(inChannels: 256, stride: (2, 2, 2), residual: true, outChannelsReductionFactor: 2),
                LTXDecoderBlock(channels: 128, numLayers: 5, timestepConditioning: timestepConditioning),
            ]

        case .ltx23Split:
            self._upBlocks.wrappedValue = [
                LTXDecoderBlock(
                    channels: 1024,
                    numLayers: 2,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 1024,
                    stride: (2, 2, 2),
                    residual: false,
                    outChannelsReductionFactor: 2,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 512,
                    numLayers: 2,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 512,
                    stride: (2, 2, 2),
                    residual: false,
                    outChannelsReductionFactor: 1,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 512,
                    numLayers: 4,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 512,
                    stride: (2, 1, 1),
                    residual: false,
                    outChannelsReductionFactor: 2,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 256,
                    numLayers: 6,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 256,
                    stride: (1, 2, 2),
                    residual: false,
                    outChannelsReductionFactor: 2,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 128,
                    numLayers: 4,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
            ]
        }
        self._convOut.wrappedValue = LTXCausalConv3d(
            inChannels: 128,
            outChannels: 3 * 4 * 4,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._lastTimeEmbedder.wrappedValue = LTXPixArtAlphaTimestepEmbedder(embeddingDim: 128 * 2)
        self._lastScaleShiftTable.wrappedValue = MLX.zeros([2, 128], dtype: .float32)
    }

    func decode(sample: MLXArray, timestep: MLXArray?) -> MLXArray {
        let batch = sample.dim(0)
        var h = sample
        let debugDecoderPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DECODER_PREFIX"]
        let debugDecoder = debugDecoderPrefix?.isEmpty == false
        var debugRoot: URL?
        var debugStem = ""

        if debugDecoder, let debugDecoderPrefix {
            let base = URL(fileURLWithPath: debugDecoderPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            debugRoot = parent
            debugStem = base.lastPathComponent
        }

        func dumpDecoder(_ name: String, _ array: MLXArray) {
            guard debugDecoder, let debugRoot else { return }
            let path = debugRoot.appendingPathComponent("\(debugStem)_\(name).npy")
            try? MLX.save(array: array, url: path)
        }

        dumpDecoder("input", h)

        if timestepConditioning {
            let noise = MLXRandom.normal(h.shape).asType(h.dtype) * MLXArray(decodeNoiseScale).asType(h.dtype)
            h = noise + (MLXArray(1.0 - decodeNoiseScale).asType(h.dtype) * h)
        }
        dumpDecoder("after_noise", h)

        h = denormalize(h)
        dumpDecoder("after_denormalize", h)

        var currentTimestep = timestep
        if currentTimestep == nil, timestepConditioning {
            currentTimestep = MLX.full([batch], values: MLXArray(decodeTimestep).asType(h.dtype))
        }

        var scaledTimestep: MLXArray?
        if timestepConditioning, let currentTimestep {
            scaledTimestep = currentTimestep * MLXArray(1000.0).asType(currentTimestep.dtype)
        }
        if let scaledTimestep {
            dumpDecoder("scaled_timestep", scaledTimestep)
        }

        h = convIn(h, causal: false)
        MLX.eval(h)
        dumpDecoder("after_conv_in", h)
        for block in upBlocks {
            h = block(h, causal: false, timestep: scaledTimestep)
            MLX.eval(h)
            Memory.clearCache()
            dumpDecoder("after_up_block_\(blockIndex(of: block, in: upBlocks) ?? -1)", h)
        }

        h = pixelNormChannels(h)
        MLX.eval(h)
        dumpDecoder("after_pixel_norm", h)
        if timestepConditioning, let scaledTimestep {
            let embedded = lastTimeEmbedder(scaledTimestep.reshaped(-1), hiddenDType: h.dtype)
            let ada = lastScaleShiftTable.reshaped(1, 2, 128, 1, 1)
                + embedded.reshaped(batch, 2, 128, 1, 1)
            let shift = ada[0..., 0, 0..., 0..., 0...]
            let scale = ada[0..., 1, 0..., 0..., 0...]
            h = h * (MLXArray(1.0).asType(h.dtype) + scale) + shift
            MLX.eval(h)
            dumpDecoder("after_last_ada", h)
        }

        h = silu(h)
        MLX.eval(h)
        dumpDecoder("after_silu", h)
        h = convOut(h, causal: false)
        MLX.eval(h)
        dumpDecoder("after_conv_out", h)
        let out = unpatchify3D(h, patchSizeHW: patchSize, patchSizeT: 1)
        MLX.eval(out)
        dumpDecoder("after_unpatchify", out)
        return out
    }

    private func denormalize(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let mean = latentsMean.asType(.float32).reshaped(1, -1, 1, 1, 1)
        let std = latentsStd.asType(.float32).reshaped(1, -1, 1, 1, 1)
        return (x.asType(.float32) * std + mean).asType(dtype)
    }
}

private func blockIndex<T: AnyObject>(of object: T, in array: [T]) -> Int? {
    for (i, element) in array.enumerated() {
        if object === element {
            return i
        }
    }
    return nil
}

private func depthToSpace3D(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
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

private func spaceToDepth3D(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
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

private func patchify3D(_ x: MLXArray, patchSizeHW: Int, patchSizeT: Int) -> MLXArray {
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

private func unpatchify3D(_ x: MLXArray, patchSizeHW: Int, patchSizeT: Int) -> MLXArray {
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

private func pixelNormChannels(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    let denom = MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + MLXArray(eps))
    return x / denom
}

private func postprocessDecodedVideo(_ decoded: MLXArray) -> MLXArray {
    var video = decoded[0, 0..., 0..., 0..., 0...]
    video = video.transposed(1, 2, 3, 0)
    video = MLX.clip((video + MLXArray(1.0)) / MLXArray(2.0), min: MLXArray(0.0), max: MLXArray(1.0))
    return (video * MLXArray(255.0)).asType(.uint8)
}

private struct LTXIntervals {
    let starts: [Int]
    let ends: [Int]
    let leftRamps: [Int]
    let rightRamps: [Int]
}

struct LTXDecodeTilingConfig {
    let spatialTileSizeInPixels: Int?
    let spatialTileOverlapInPixels: Int
    let temporalTileSizeInFrames: Int?
    let temporalTileOverlapInFrames: Int
}

func selectDecodeTilingConfig(
    width: Int,
    height: Int,
    numFrames: Int,
    fps: Int,
    decodeBudgetGiB: Double? = nil
) -> LTXDecodeTilingConfig? {
    let outputPixelBudget = 135_000_000.0
    let framePixels = Double(max(1, width)) * Double(max(1, height))
    let totalOutputPixels = framePixels * Double(max(1, numFrames))

    let latentFrames = 1 + ((numFrames - 1) / 8)
    let latentH = max(1, height / 32)
    let latentW = max(1, width / 32)
    let budgetGiB = decodeBudgetGiB ?? decodeTilingBudgetGiB()
    let budgetBytes = max(1.0, budgetGiB) * 1024.0 * 1024.0 * 1024.0

    let bytesPerLatentFrame = Double(512 * 4) * Double(latentH * 4) * Double(latentW * 4) * 2.0
    let totalActivationBytes = bytesPerLatentFrame * Double(latentFrames)
    let activationLimitedTileFrames: Int?
    if totalActivationBytes > budgetBytes {
        let maxLatentFrames = max(2, Int(budgetBytes / max(bytesPerLatentFrame, 1.0)))
        activationLimitedTileFrames = max(16, maxLatentFrames * 8)
    } else {
        activationLimitedTileFrames = nil
    }

    let outputLimitedTileFrames: Int?
    if totalOutputPixels > outputPixelBudget {
        let rawTileFrames = max(16, Int(outputPixelBudget / max(framePixels, 1.0)))
        outputLimitedTileFrames = max(16, (rawTileFrames / 8) * 8)
    } else {
        outputLimitedTileFrames = nil
    }

    let tileFramesCandidates = [
        activationLimitedTileFrames,
        outputLimitedTileFrames,
    ].compactMap { $0 }

    guard let tileFrames = tileFramesCandidates.min() else {
        return nil
    }

    let oneSecondFrames = max(8, (max(1, fps) / 8) * 8)
    let overlapFrames = min(oneSecondFrames, (tileFrames / 32) * 8)

    return LTXDecodeTilingConfig(
        spatialTileSizeInPixels: nil,
        spatialTileOverlapInPixels: 0,
        temporalTileSizeInFrames: tileFrames,
        temporalTileOverlapInFrames: min(max(0, overlapFrames), max(0, tileFrames - 8))
    )
}

private func decodeTilingBudgetGiB() -> Double {
    let environment = ProcessInfo.processInfo.environment
    for key in ["LTX2_VAE_DECODE_BUDGET_GB", "MERERUN_VIDEO_LTX_VAE_DECODE_BUDGET_GB"] {
        if let value = environment[key], let parsed = Double(value), parsed.isFinite, parsed > 0 {
            return parsed
        }
    }
    return 8.0
}

private func computeTrapezoidalMask1D(
    length: Int,
    rampLeft: Int,
    rampRight: Int,
    leftStartsFromZero: Bool
) -> [Float] {
    precondition(length > 0, "Mask length must be positive.")
    let left = max(0, min(rampLeft, length))
    let right = max(0, min(rampRight, length))

    var mask = [Float](repeating: 1.0, count: length)

    if left > 0 {
        let intervalLength = leftStartsFromZero ? (left + 1) : (left + 2)
        var fade = [Float](repeating: 0, count: left)
        if leftStartsFromZero {
            for i in 0..<left {
                fade[i] = Float(i) / Float(intervalLength - 1)
            }
        } else {
            for i in 0..<left {
                fade[i] = Float(i + 1) / Float(intervalLength - 1)
            }
        }
        for i in 0..<left {
            mask[i] *= fade[i]
        }
    }

    if right > 0 {
        for i in 0..<right {
            let fade = Float(right - i) / Float(right + 1)
            mask[length - right + i] *= fade
        }
    }

    for i in 0..<mask.count {
        mask[i] = min(1.0, max(0.0, mask[i]))
    }
    return mask
}

private func splitInSpatial(size: Int, overlap: Int, dimensionSize: Int) -> LTXIntervals {
    if dimensionSize <= size {
        return LTXIntervals(starts: [0], ends: [dimensionSize], leftRamps: [0], rightRamps: [0])
    }

    let amount = (dimensionSize + size - 2 * overlap - 1) / (size - overlap)
    let starts = (0..<amount).map { $0 * (size - overlap) }
    var ends = starts.map { $0 + size }
    if !ends.isEmpty {
        ends[ends.count - 1] = dimensionSize
    }
    let leftRamps = [0] + Array(repeating: overlap, count: max(0, amount - 1))
    let rightRamps = Array(repeating: overlap, count: max(0, amount - 1)) + [0]
    return LTXIntervals(starts: starts, ends: ends, leftRamps: leftRamps, rightRamps: rightRamps)
}

private func splitInTemporal(size: Int, overlap: Int, dimensionSize: Int) -> LTXIntervals {
    if dimensionSize <= size {
        return LTXIntervals(starts: [0], ends: [dimensionSize], leftRamps: [0], rightRamps: [0])
    }

    let intervals = splitInSpatial(size: size, overlap: overlap, dimensionSize: dimensionSize)
    var starts = intervals.starts
    var leftRamps = intervals.leftRamps

    if starts.count > 1 {
        for i in 1..<starts.count {
            starts[i] -= 1
            leftRamps[i] += 1
        }
    }

    return LTXIntervals(
        starts: starts,
        ends: intervals.ends,
        leftRamps: leftRamps,
        rightRamps: intervals.rightRamps
    )
}

private func mapSpatialOutputSlice(
    begin: Int,
    end: Int,
    leftRamp: Int,
    rightRamp: Int,
    scale: Int
) -> (start: Int, length: Int, mask: [Float]) {
    let start = begin * scale
    let stop = end * scale
    let length = max(1, stop - start)
    let mask = computeTrapezoidalMask1D(
        length: length,
        rampLeft: leftRamp * scale,
        rampRight: rightRamp * scale,
        leftStartsFromZero: false
    )
    return (start, length, mask)
}

private func mapTemporalOutputSlice(
    begin: Int,
    end: Int,
    leftRamp: Int,
    rightRamp: Int,
    scale: Int
) -> (start: Int, length: Int, mask: [Float]) {
    let start = begin * scale
    let stop = 1 + (end - 1) * scale
    let length = max(1, stop - start)
    let leftScaled = leftRamp > 0 ? (1 + (leftRamp - 1) * scale) : 0
    let rightScaled = rightRamp * scale
    let mask = computeTrapezoidalMask1D(
        length: length,
        rampLeft: leftScaled,
        rampRight: rightScaled,
        leftStartsFromZero: true
    )
    return (start, length, mask)
}

func accumulateLTXDecodedTile(
    output: MLXArray,
    weights: MLXArray,
    tileDecoded: MLXArray,
    outputFrameStart: Int,
    outputHeightStart: Int,
    outputWidthStart: Int,
    temporalMask: [Float],
    heightMask: [Float],
    widthMask: [Float]
) {
    let tileFrames = min(tileDecoded.dim(2), min(temporalMask.count, output.dim(2) - outputFrameStart))
    let tileHeight = min(tileDecoded.dim(3), min(heightMask.count, output.dim(3) - outputHeightStart))
    let tileWidth = min(tileDecoded.dim(4), min(widthMask.count, output.dim(4) - outputWidthStart))
    guard tileFrames > 0, tileHeight > 0, tileWidth > 0 else { return }

    let temporal = MLXArray(Array(temporalMask.prefix(tileFrames)))
        .asType(output.dtype)
        .reshaped(1, 1, tileFrames, 1, 1)
    let vertical = MLXArray(Array(heightMask.prefix(tileHeight)))
        .asType(output.dtype)
        .reshaped(1, 1, 1, tileHeight, 1)
    let horizontal = MLXArray(Array(widthMask.prefix(tileWidth)))
        .asType(output.dtype)
        .reshaped(1, 1, 1, 1, tileWidth)
    let blend = temporal * vertical * horizontal
    let frameRange = outputFrameStart..<(outputFrameStart + tileFrames)
    let heightRange = outputHeightStart..<(outputHeightStart + tileHeight)
    let widthRange = outputWidthStart..<(outputWidthStart + tileWidth)
    let tile = tileDecoded[0..., 0..<3, 0..<tileFrames, 0..<tileHeight, 0..<tileWidth]
        .asType(output.dtype)

    output[0..., 0..., frameRange, heightRange, widthRange] =
        output[0..., 0..., frameRange, heightRange, widthRange] + (tile * blend)
    weights[0..., 0..., frameRange, heightRange, widthRange] =
        weights[0..., 0..., frameRange, heightRange, widthRange] + blend
}

func finalizeLTXDecodedTiles(output: MLXArray, weights: MLXArray) -> MLXArray {
    let epsilon: Float = weights.dtype == .float16 ? 1e-4 : 1e-8
    let denominator = MLX.maximum(weights, MLXArray(epsilon).asType(weights.dtype))
    var video = (output / denominator)[0]
    video = video.transposed(1, 2, 3, 0)
    let zero = MLXArray(Float(0)).asType(video.dtype)
    let one = MLXArray(Float(1)).asType(video.dtype)
    video = MLX.clip(
        (video + one) / MLXArray(Float(2)).asType(video.dtype),
        min: zero,
        max: one
    )
    return (video * MLXArray(Float(255)).asType(video.dtype)).asType(.uint8)
}

private func decodeWithTiling(
    decoder: LTXVideoDecoder,
    latents: MLXArray,
    spatialTileSizeInPixels: Int?,
    spatialOverlapInPixels: Int,
    temporalTileSizeInFrames: Int?,
    temporalOverlapInFrames: Int,
    spatialScale: Int,
    temporalScale: Int
) -> MLXArray {
    precondition(latents.ndim == 5, "Expected latent tensor [B, C, F, H, W]")
    let batch = latents.dim(0)
    precondition(batch == 1, "Tiled decode currently supports batch=1.")

    let latentFrames = latents.dim(2)
    let latentH = latents.dim(3)
    let latentW = latents.dim(4)

    let outFrames = 1 + (latentFrames - 1) * temporalScale
    let outH = latentH * spatialScale
    let outW = latentW * spatialScale

    let temporalIntervals: LTXIntervals
    if let temporalTileSizeInFrames {
        let tileSize = max(1, temporalTileSizeInFrames / temporalScale)
        let overlap = max(0, temporalOverlapInFrames / temporalScale)
        temporalIntervals = splitInTemporal(size: tileSize, overlap: overlap, dimensionSize: latentFrames)
    } else {
        temporalIntervals = LTXIntervals(starts: [0], ends: [latentFrames], leftRamps: [0], rightRamps: [0])
    }

    let heightIntervals: LTXIntervals
    let widthIntervals: LTXIntervals
    if let spatialTileSizeInPixels {
        let tileSize = max(1, spatialTileSizeInPixels / spatialScale)
        let overlap = max(0, spatialOverlapInPixels / spatialScale)
        heightIntervals = splitInSpatial(size: tileSize, overlap: overlap, dimensionSize: latentH)
        widthIntervals = splitInSpatial(size: tileSize, overlap: overlap, dimensionSize: latentW)
    } else {
        heightIntervals = LTXIntervals(starts: [0], ends: [latentH], leftRamps: [0], rightRamps: [0])
        widthIntervals = LTXIntervals(starts: [0], ends: [latentW], leftRamps: [0], rightRamps: [0])
    }

    let totalRGB = 3 * outFrames * outH * outW
    let accumulatorDType: DType = totalRGB >= 128_000_000 ? .float16 : .float32
    let output = MLX.zeros([1, 3, outFrames, outH, outW], dtype: accumulatorDType)
    let weights = MLX.zeros([1, 1, outFrames, outH, outW], dtype: accumulatorDType)

    for tIndex in 0..<temporalIntervals.starts.count {
        let tStart = temporalIntervals.starts[tIndex]
        let tEnd = temporalIntervals.ends[tIndex]
        let temporalOutput = mapTemporalOutputSlice(
            begin: tStart,
            end: tEnd,
            leftRamp: temporalIntervals.leftRamps[tIndex],
            rightRamp: temporalIntervals.rightRamps[tIndex],
            scale: temporalScale
        )
        let outTStart = temporalOutput.start
        let expectedOutT = temporalOutput.length
        let tMask = temporalOutput.mask

        for hIndex in 0..<heightIntervals.starts.count {
            let hStart = heightIntervals.starts[hIndex]
            let hEnd = heightIntervals.ends[hIndex]
            let heightOutput = mapSpatialOutputSlice(
                begin: hStart,
                end: hEnd,
                leftRamp: heightIntervals.leftRamps[hIndex],
                rightRamp: heightIntervals.rightRamps[hIndex],
                scale: spatialScale
            )
            let outHStart = heightOutput.start
            let expectedOutH = heightOutput.length
            let hMask = heightOutput.mask

            for wIndex in 0..<widthIntervals.starts.count {
                let wStart = widthIntervals.starts[wIndex]
                let wEnd = widthIntervals.ends[wIndex]
                let widthOutput = mapSpatialOutputSlice(
                    begin: wStart,
                    end: wEnd,
                    leftRamp: widthIntervals.leftRamps[wIndex],
                    rightRamp: widthIntervals.rightRamps[wIndex],
                    scale: spatialScale
                )
                let outWStart = widthOutput.start
                let expectedOutW = widthOutput.length
                let wMask = widthOutput.mask

                let tileLatents = latents[0..., 0..., tStart..<tEnd, hStart..<hEnd, wStart..<wEnd]
                let tileDecoded = decoder.decode(sample: tileLatents, timestep: nil).asType(.float32)
                accumulateLTXDecodedTile(
                    output: output,
                    weights: weights,
                    tileDecoded: tileDecoded,
                    outputFrameStart: outTStart,
                    outputHeightStart: outHStart,
                    outputWidthStart: outWStart,
                    temporalMask: Array(tMask.prefix(expectedOutT)),
                    heightMask: Array(hMask.prefix(expectedOutH)),
                    widthMask: Array(wMask.prefix(expectedOutW))
                )
                MLX.eval(output, weights)

                Memory.clearCache()
            }
        }
    }

    let frames = finalizeLTXDecodedTiles(output: output, weights: weights)
    MLX.eval(frames)
    return frames
}

func mapLTXDecoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("vae.decoder.") {
        mapped = String(key.dropFirst("vae.decoder.".count))
    } else if key.hasPrefix("vae_decoder.") {
        mapped = String(key.dropFirst("vae_decoder.".count))
    } else if key.hasPrefix("decoder.") {
        mapped = String(key.dropFirst("decoder.".count))
    } else {
        return []
    }

    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    var casted = value
    if sourceLayout == .pytorch, mapped.contains(".conv.weight"), casted.ndim == 5 {
        casted = casted.transposed(0, 2, 3, 4, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }

    return [(mapped, casted)]
}

func mapLTXEncoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("vae.encoder.") {
        mapped = String(key.dropFirst("vae.encoder.".count))
    } else if key.hasPrefix("vae_encoder.") {
        mapped = String(key.dropFirst("vae_encoder.".count))
    } else if key.hasPrefix("encoder.") {
        mapped = String(key.dropFirst("encoder.".count))
    } else {
        return []
    }

    var casted = value
    if sourceLayout == .pytorch, mapped.contains(".conv.weight"), casted.ndim == 5 {
        casted = casted.transposed(0, 2, 3, 4, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }

    return [(mapped, casted)]
}

func mapLTXUpsamplerWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped = key
    for prefix in [
        "spatial_upscaler_x2_v1_1.",
        "spatial_upscaler_x1_5_v1_0.",
        "temporal_upscaler_x2_v1_0.",
    ] where mapped.hasPrefix(prefix) {
        mapped = String(mapped.dropFirst(prefix.count))
        break
    }
    mapped = mapped.replacingOccurrences(of: "upsampler.0.", with: "upsampler.conv.")

    var casted = value
    if sourceLayout == .pytorch, mapped.contains("conv"), mapped.contains("weight") {
        if casted.ndim == 5 {
            casted = casted.transposed(0, 2, 3, 4, 1)
        } else if casted.ndim == 4 {
            casted = casted.transposed(0, 2, 3, 1)
        }
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

enum LTXTensorWeightLayout: Equatable {
    case pytorch
    case mlx
}

private func createPositionGrid(
    batchSize: Int,
    numFrames: Int,
    height: Int,
    width: Int,
    temporalScale: Int,
    spatialScale: Int,
    fps: Float,
    causalFix: Bool
) -> MLXArray {
    let tokenCount = numFrames * height * width
    var data = [Float](repeating: 0, count: batchSize * 3 * tokenCount * 2)

    for b in 0..<batchSize {
        var token = 0
        for t in 0..<numFrames {
            for h in 0..<height {
                for w in 0..<width {
                    let pixelT0 = Float(t * temporalScale)
                    let pixelT1 = Float((t + 1) * temporalScale)
                    let pixelH0 = Float(h * spatialScale)
                    let pixelH1 = Float((h + 1) * spatialScale)
                    let pixelW0 = Float(w * spatialScale)
                    let pixelW1 = Float((w + 1) * spatialScale)

                    let base = ((b * 3 * tokenCount) + token) * 2

                    var t0 = pixelT0
                    var t1 = pixelT1
                    if causalFix {
                        let shift = Float(1 - temporalScale)
                        t0 = max(0, t0 + shift)
                        t1 = max(0, t1 + shift)
                    }
                    t0 /= fps
                    t1 /= fps

                    data[base] = t0
                    data[base + 1] = t1

                    let hBase = ((b * 3 * tokenCount) + tokenCount + token) * 2
                    data[hBase] = pixelH0
                    data[hBase + 1] = pixelH1

                    let wBase = ((b * 3 * tokenCount) + (2 * tokenCount) + token) * 2
                    data[wBase] = pixelW0
                    data[wBase + 1] = pixelW1

                    token += 1
                }
            }
        }
    }

    return MLXArray(data).reshaped(batchSize, 3, tokenCount, 2)
}

private func getTimestepEmbedding(
    timesteps: MLXArray,
    embeddingDim: Int,
    flipSinToCos: Bool,
    downscaleFreqShift: Float,
    scale: Float,
    maxPeriod: Float
) -> MLXArray {
    let halfDim = embeddingDim / 2
    let exponent = -Foundation.log(maxPeriod) * MLXArray(0..<halfDim).asType(.float32)
        / MLXArray(Float(halfDim) - downscaleFreqShift)
    let emb = exp(exponent)
    let timestepExpanded = timesteps.asType(.float32).reshaped(-1, 1)
    let args = timestepExpanded * emb.reshaped(1, halfDim) * MLXArray(scale)

    let sinPart = MLX.sin(args)
    let cosPart = MLX.cos(args)
    var output = flipSinToCos
        ? MLX.concatenated([cosPart, sinPart], axis: -1)
        : MLX.concatenated([sinPart, cosPart], axis: -1)

    if embeddingDim % 2 == 1 {
        output = padded(output, widths: [[0, 0], [0, 1]])
    }
    return output
}

// MARK: - Unified AV Generator (Native Swift/MLX)

private let LTXAudioSampleRate = 24_000
private let LTXAudioLatentSampleRate = 16_000
private let LTXAudioHopLength = 160
private let LTXAudioLatentDownsampleFactor = 4
private let LTXAudioLatentChannels = 8
private let LTXAudioLatentMelBins = 16

public struct LTXUnifiedAVGenerationOptions: Sendable {
    public let prompt: String
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Int
    public let seed: Int
    public let maxTextLength: Int
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float

    public init(
        prompt: String,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Int = 24,
        seed: Int,
        maxTextLength: Int = 1024,
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1.0,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1.0
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.maxTextLength = maxTextLength
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
    }
}

public struct LTXUnifiedAVGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let videoLatents: MLXArray
    public let audioLatents: MLXArray
    public let audioWaveform: MLXArray
    public let audioSampleRate: Int

    public init(
        frames: MLXArray,
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        audioWaveform: MLXArray,
        audioSampleRate: Int
    ) {
        self.frames = frames
        self.videoLatents = videoLatents
        self.audioLatents = audioLatents
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
    }
}

public enum LTXUnifiedAVGeneratorError: LocalizedError {
    case transformerWeightsMissing(URL)
    case upsamplerWeightsMissing(URL)
    case unsupportedLTX23SplitModel(URL)
    case ltx23TextEncoderMissing(String)
    case generatorNotLoaded
    case invalidResolution(width: Int, height: Int)
    case invalidFrameCount(Int)
    case invalidImageStrength(Float)
    case invalidImageFrameIndex(Int)
    case imageNotFound(URL)
    case imageDecodeFailed(URL)
    case emptyPrompt
    case decoderNotLoaded
    case encoderNotLoaded
    case upsamplerNotLoaded
    case audioEmbeddingsMissing
    case audioDecoderNotLoaded
    case vocoderNotLoaded
    case bweVocoderConfigMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .transformerWeightsMissing(let url):
            return "Missing LTX transformer weights at \(url.path)"
        case .upsamplerWeightsMissing(let url):
            return "Missing LTX upsampler weights at \(url.path)"
        case .unsupportedLTX23SplitModel(let url):
            return """
            Detected an LTX 2.3 split MLX model at \(url.path). This native loader still supports the older \
            merged LTX layout; port the LTX 2.3 V2 connector, transformer, and split component loader before \
            generation.
            """
        case .ltx23TextEncoderMissing(let id):
            return """
            LTX 2.3 requires the companion Gemma 3 text encoder `\(id)`. Install it with \
            `mere.run model pull video-ltx23-av-mlx`, or set MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT to a local \
            mlx-community/gemma-3-12b-it-4bit checkout.
            """
        case .generatorNotLoaded:
            return "LTX unified AV generator is not loaded."
        case .invalidResolution(let width, let height):
            return "Resolution must be divisible by 64 (got \(width)x\(height))."
        case .invalidFrameCount(let value):
            return "numFrames must satisfy 8n+1 and be >= 9 (got \(value))."
        case .invalidImageStrength(let value):
            return "imageStrength must be in [0, 1] (got \(value))."
        case .invalidImageFrameIndex(let value):
            return "imageFrameIndex must be >= 0 (got \(value))."
        case .imageNotFound(let url):
            return "Source image not found: \(url.path)"
        case .imageDecodeFailed(let url):
            return "Could not decode source image: \(url.path)"
        case .emptyPrompt:
            return "Prompt cannot be empty."
        case .decoderNotLoaded:
            return "LTX video decoder is not loaded."
        case .encoderNotLoaded:
            return "LTX video encoder is not loaded."
        case .upsamplerNotLoaded:
            return "LTX upsampler is not loaded."
        case .audioEmbeddingsMissing:
            return "Audio text embeddings are unavailable in this model."
        case .audioDecoderNotLoaded:
            return "Audio decoder is not loaded."
        case .vocoderNotLoaded:
            return "Audio vocoder is not loaded."
        case .bweVocoderConfigMissing(let url):
            return "LTX BWE vocoder weights require a vocoder BWE config under \(url.path)."
        }
    }
}

public actor LTXUnifiedAVGenerator {
    private var textEncoder: LTXGemmaTextEncoder?
    private var transformer: (any LTXUnifiedAVTransformerRuntime)?
    private var decoder: LTXVideoDecoder?
    private var encoder: LTXVideoEncoder?
    private var upsampler: LTXLatentUpsampler?
    private var audioDecoder: LTXAudioDecoder?
    private var vocoder: LTXAudioVocoderBase?
    private var modelWeightsURL: URL?
    private var videoEncoderWeightsURL: URL?
    private var videoVAEWeightLayout: LTXTensorWeightLayout = .pytorch
    private var loadedDType: DType = .bfloat16
    private var loadedRoot: URL?

    public init() {}

    public func load(
        modelRoot: URL,
        dtype: DType = .bfloat16
    ) async throws {
        let root = modelRoot.standardizedFileURL
        let isLTX23 = isLTX23SplitModelRoot(root)
        let splitTensorLayout: LTXTensorWeightLayout = isLTX23 ? .mlx : .pytorch
        let transformerURL = isLTX23
            ? root.appendingPathComponent("transformer-distilled.safetensors", isDirectory: false)
            : root.appendingPathComponent("ltx-2-19b-distilled.safetensors", isDirectory: false)
        let upsamplerURL = isLTX23
            ? root.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors", isDirectory: false)
            : root.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors", isDirectory: false)
        guard FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw LTXUnifiedAVGeneratorError.transformerWeightsMissing(transformerURL)
        }
        guard FileManager.default.fileExists(atPath: upsamplerURL.path) else {
            throw LTXUnifiedAVGeneratorError.upsamplerWeightsMissing(upsamplerURL)
        }

        let text = LTXGemmaTextEncoder()
        let textEncoderRoot = try isLTX23 ? resolveLTX23TextEncoderRoot(modelRoot: root) : nil
        try await text.load(
            modelRoot: root,
            textEncoderRoot: textEncoderRoot,
            dtype: dtype,
            loadConnectorWeights: true
        )

        let model: any LTXUnifiedAVTransformerRuntime
        if isLTX23 {
            let splitModel = LTXUnifiedAVTransformerV2()
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: transformerURL,
                to: splitModel,
                dtype: dtype,
                verify: .none,
                include: { key in
                    key.hasPrefix("transformer.")
                },
                mapper: { key, value in
                    mapLTX23UnifiedTransformerWeight(key: key, value: value, dtype: dtype)
                },
                batchSize: 24
            )
            model = splitModel
        } else {
            let mergedModel = LTXUnifiedAVTransformer()
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: transformerURL,
                to: mergedModel,
                dtype: dtype,
                verify: .none,
                include: { key in
                    key.hasPrefix("model.diffusion_model.")
                },
                mapper: { key, value in
                    mapUnifiedTransformerWeight(key: key, value: value, dtype: dtype)
                },
                batchSize: 24
            )
            model = mergedModel
        }

        let vaeDecoder = LTXVideoDecoder(
            timestepConditioning: false,
            architecture: isLTX23 ? .ltx23Split : .legacy
        )
        let videoDecoderURL = isLTX23
            ? root.appendingPathComponent("vae_decoder.safetensors", isDirectory: false)
            : transformerURL
        let decoderStats = try SafetensorsStreamingLoader.loadArrays(
            url: videoDecoderURL,
            where: { key in
                key == "latents_mean"
                    || key == "latents_std"
                    || key == "vae.per_channel_statistics.mean-of-means"
                    || key == "vae.per_channel_statistics.std-of-means"
                    || key == "vae_decoder.per_channel_statistics.mean-of-means"
                    || key == "vae_decoder.per_channel_statistics.std-of-means"
                    || key == "vae_decoder.per_channel_statistics.mean"
                    || key == "vae_decoder.per_channel_statistics.std"
            },
            dtype: .float32
        )

        if let mean = decoderStats["latents_mean"]
            ?? decoderStats["vae.per_channel_statistics.mean-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.mean-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.mean"] {
            vaeDecoder.latentsMean = mean.asType(.float32)
        }
        if let std = decoderStats["latents_std"]
            ?? decoderStats["vae.per_channel_statistics.std-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.std-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.std"] {
            vaeDecoder.latentsStd = std.asType(.float32)
        }

        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: videoDecoderURL,
            to: vaeDecoder,
            dtype: dtype,
            verify: .none,
            include: { key in
                key.hasPrefix("decoder.")
                    || key.hasPrefix("vae.decoder.")
                    || key.hasPrefix("vae_decoder.")
            },
            mapper: { key, value in
                mapLTXDecoderWeight(key: key, value: value, dtype: dtype, sourceLayout: splitTensorLayout)
            },
            batchSize: 24
        )

        let latentUpsampler = LTXLatentUpsampler(inChannels: 128, midChannels: 1024, numBlocksPerStage: 4)
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: upsamplerURL,
            to: latentUpsampler,
            dtype: dtype,
            verify: .none,
            include: { _ in true },
            mapper: { key, value in
                mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype, sourceLayout: splitTensorLayout)
            },
            batchSize: 24
        )

        let audioDecoder = LTXAudioDecoder()
        let audioDecoderURL = isLTX23
            ? root.appendingPathComponent("audio_vae.safetensors", isDirectory: false)
            : transformerURL
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: audioDecoderURL,
            to: audioDecoder,
            dtype: .float32,
            verify: .none,
            include: { key in
                key.hasPrefix("audio_vae.decoder.")
                    || key.hasPrefix("audio_vae.per_channel_statistics.")
            },
            mapper: { key, value in
                mapAudioVaeDecoderWeight(key: key, value: value, dtype: .float32, sourceLayout: splitTensorLayout)
            },
            batchSize: 24
        )

        let vocoderURL = isLTX23
            ? root.appendingPathComponent("vocoder.safetensors", isDirectory: false)
            : transformerURL
        let vocoderMetadata = try SafetensorsStreamingLoader.metadata(url: vocoderURL)
        let vocoderFlavor = detectLTXVocoderFlavor(keys: vocoderMetadata.keys)
        let vocoder: LTXAudioVocoderBase = switch vocoderFlavor {
        case .legacy:
            LTXVocoder()
        case .bandwidthExtension:
            if let config = try loadLTXBWEVocoderConfig(modelRoot: root) {
                LTXVocoderWithBWE(config: config)
            } else {
                throw LTXUnifiedAVGeneratorError.bweVocoderConfigMissing(root)
            }
        }
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: vocoderURL,
            to: vocoder,
            dtype: .float32,
            verify: .none,
            include: { key in
                key.hasPrefix("vocoder.")
            },
            mapper: { key, value in
                mapVocoderWeight(
                    key: key,
                    value: value,
                    dtype: .float32,
                    sourceLayout: isLTX23 ? .mlx : .pytorch,
                    targetFlavor: vocoderFlavor
                )
            },
            batchSize: 24
        )

        self.textEncoder = text
        self.transformer = model
        self.decoder = vaeDecoder
        self.upsampler = latentUpsampler
        self.audioDecoder = audioDecoder
        self.vocoder = vocoder
        self.encoder = nil
        self.modelWeightsURL = transformerURL
        self.videoEncoderWeightsURL = isLTX23
            ? root.appendingPathComponent("vae_encoder.safetensors", isDirectory: false)
            : transformerURL
        self.videoVAEWeightLayout = splitTensorLayout
        self.loadedDType = dtype
        self.loadedRoot = root
    }

    public func unload() async {
        if let textEncoder {
            await textEncoder.unload()
        }
        textEncoder = nil
        transformer = nil
        decoder = nil
        encoder = nil
        upsampler = nil
        audioDecoder = nil
        vocoder = nil
        modelWeightsURL = nil
        videoEncoderWeightsURL = nil
        videoVAEWeightLayout = .pytorch
        loadedRoot = nil
        Memory.clearCache()
    }

    private func loadEncoderIfNeeded() throws {
        if encoder != nil {
            return
        }
        guard let modelWeightsURL else {
            throw LTXUnifiedAVGeneratorError.encoderNotLoaded
        }
        let encoderURL = videoEncoderWeightsURL ?? modelWeightsURL

        let vaeEncoder = LTXVideoEncoder(
            architecture: videoVAEWeightLayout == .mlx ? .ltx23Split : .legacy
        )
        if let decoder {
            vaeEncoder.latentsMean = decoder.latentsMean.asType(.float32)
            vaeEncoder.latentsStd = decoder.latentsStd.asType(.float32)
        } else {
            let stats = try SafetensorsStreamingLoader.loadArrays(
                url: encoderURL,
                where: { key in
                    key == "latents_mean"
                        || key == "latents_std"
                        || key == "vae.per_channel_statistics.mean-of-means"
                        || key == "vae.per_channel_statistics.std-of-means"
                        || key == "vae_encoder.per_channel_statistics.mean-of-means"
                        || key == "vae_encoder.per_channel_statistics.std-of-means"
                        || key == "vae_encoder.per_channel_statistics._mean_of_means"
                        || key == "vae_encoder.per_channel_statistics._std_of_means"
                },
                dtype: .float32
            )
            if let mean = stats["latents_mean"]
                ?? stats["vae.per_channel_statistics.mean-of-means"]
                ?? stats["vae_encoder.per_channel_statistics.mean-of-means"]
                ?? stats["vae_encoder.per_channel_statistics._mean_of_means"] {
                vaeEncoder.latentsMean = mean.asType(.float32)
            }
            if let std = stats["latents_std"]
                ?? stats["vae.per_channel_statistics.std-of-means"]
                ?? stats["vae_encoder.per_channel_statistics.std-of-means"]
                ?? stats["vae_encoder.per_channel_statistics._std_of_means"] {
                vaeEncoder.latentsStd = std.asType(.float32)
            }
        }

        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: encoderURL,
            to: vaeEncoder,
            dtype: loadedDType,
            verify: .none,
            include: { key in
                key.hasPrefix("encoder.")
                    || key.hasPrefix("vae.encoder.")
                    || key.hasPrefix("vae_encoder.")
            },
            mapper: { key, value in
                mapLTXEncoderWeight(
                    key: key,
                    value: value,
                    dtype: loadedDType,
                    sourceLayout: videoVAEWeightLayout
                )
            },
            batchSize: 24
        )

        encoder = vaeEncoder
    }

    public func generate(
        options: LTXUnifiedAVGenerationOptions
    ) async throws -> LTXUnifiedAVGenerationResult {
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        guard options.width % 64 == 0, options.height % 64 == 0 else {
            throw LTXUnifiedAVGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.imageStrength)
        }

        guard let textEncoder, let transformer else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        guard let decoder else {
            throw LTXUnifiedAVGeneratorError.decoderNotLoaded
        }
        guard let upsampler else {
            throw LTXUnifiedAVGeneratorError.upsamplerNotLoaded
        }
        guard let audioDecoder else {
            throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
        }
        guard let vocoder else {
            throw LTXUnifiedAVGeneratorError.vocoderNotLoaded
        }

        let encoding = try await textEncoder.encode(prompt: prompt, maxLength: options.maxTextLength)
        let videoContext = encoding.videoEmbeddings
        guard let audioContext = encoding.audioEmbeddings else {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }

        let latentFrames = 1 + ((options.numFrames - 1) / 8)
        let stage1H = options.height / 2 / 32
        let stage1W = options.width / 2 / 32
        let stage2H = options.height / 32
        let stage2W = options.width / 32
        let audioFrames = computeAudioLatentFrameCount(videoFrames: options.numFrames, fps: max(1, options.fps))

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        let modelDType = videoContext.dtype

        let isImageToVideo = options.sourceImageURL != nil
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?

        var videoLatents: MLXArray
        if isImageToVideo {
            let sourceImageURL = options.sourceImageURL!
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw LTXUnifiedAVGeneratorError.imageNotFound(sourceImageURL)
            }
            try loadEncoderIfNeeded()
            guard let encoder else {
                throw LTXUnifiedAVGeneratorError.encoderNotLoaded
            }
            if options.imageFrameIndex >= latentFrames {
                throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
            }

            let stage1Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: options.width / 2,
                height: options.height / 2,
                dtype: modelDType
            )
            let stage1ImageLatent = encoder.encode(image: stage1Image)

            let stage2Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: options.width,
                height: options.height,
                dtype: modelDType
            )
            let stage2ImageLatent = encoder.encode(image: stage2Image)
            stage2ConditioningLatent = stage2ImageLatent

            // Optional end keyframe -> conditions the tail latent frame so the clip
            // interpolates a directed start->end motion.
            var stage1EndImageLatent: MLXArray?
            if let endImageURL = options.endImageURL {
                let stage1EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: options.width / 2,
                    height: options.height / 2,
                    dtype: modelDType
                )
                stage1EndImageLatent = encoder.encode(image: stage1EndImage)
                let stage2EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: options.width,
                    height: options.height,
                    dtype: modelDType
                )
                stage2EndConditioningLatent = encoder.encode(image: stage2EndImage)
            }

            var state1 = applyLatentConditioning(
                baseLatent: MLX.zeros([1, 128, latentFrames, stage1H, stage1W], dtype: modelDType),
                conditionedLatent: stage1ImageLatent,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage1EndImageLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
            let stage1Noise = MLXRandom.normal(state1.latent.shape).asType(modelDType)
            let stage1Sigma = MLXArray(STAGE1Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = state1.denoiseMask * stage1Sigma
            state1.latent = stage1Noise * scaledMask + state1.latent * (one - scaledMask)
            videoLatents = state1.latent
            MLX.eval(videoLatents)
            stage1ConditioningState = state1
        } else {
            videoLatents = MLXRandom.normal([1, 128, latentFrames, stage1H, stage1W]).asType(modelDType)
            MLX.eval(videoLatents)
        }

        var audioLatents = MLXRandom.normal([1, LTXAudioLatentChannels, audioFrames, LTXAudioLatentMelBins]).asType(modelDType)
        MLX.eval(audioLatents)

        let stage1VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1H,
            width: stage1W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        let stage1VideoRope = precomputeSplitRope(
            positions: stage1VideoPositions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )
        let stage1VideoCrossPositions = stage1VideoPositions[0..., 0..<1, 0..., 0...]
        let stage1VideoCrossRope = precomputeSplitRope(
            positions: stage1VideoCrossPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )
        let stage1AudioPositions = createAudioPositionGrid(batchSize: 1, audioFrames: audioFrames)
        let stage1AudioRope = precomputeSplitRope(
            positions: stage1AudioPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )

        (videoLatents, audioLatents) = denoiseAVLoop(
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            videoRope: stage1VideoRope,
            audioRope: stage1AudioRope,
            videoCrossRope: stage1VideoCrossRope,
            audioCrossRope: stage1AudioRope,
            videoContext: videoContext,
            audioContext: audioContext,
            transformer: transformer,
            sigmas: STAGE1Sigmas,
            videoConditioning: stage1ConditioningState
        )
        MLX.eval(videoLatents, audioLatents)

        videoLatents = upsampleLatents(
            videoLatents,
            upsampler: upsampler,
            latentMean: decoder.latentsMean,
            latentStd: decoder.latentsStd
        )
        MLX.eval(videoLatents)

        if let stage2State = stage2ConditioningLatent.map({
            applyLatentConditioning(
                baseLatent: videoLatents,
                conditionedLatent: $0,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage2EndConditioningLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
        }) {
            let noise = MLXRandom.normal(videoLatents.shape).asType(modelDType)
            let noiseScale = MLXArray(STAGE2Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = stage2State.denoiseMask * noiseScale
            videoLatents = noise * scaledMask + stage2State.latent * (one - scaledMask)
            MLX.eval(videoLatents)
            stage2ConditioningState = stage2State

            let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
            let oneMinusScale = MLXArray(1.0).asType(modelDType) - noiseScale
            audioLatents = audioNoise * noiseScale + audioLatents * oneMinusScale
            MLX.eval(audioLatents)
        } else {
            let noiseScale = MLXArray(STAGE2Sigmas[0]).asType(modelDType)
            let oneMinusScale = MLXArray(1.0 - STAGE2Sigmas[0]).asType(modelDType)
            let videoNoise = MLXRandom.normal(videoLatents.shape).asType(modelDType)
            let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
            videoLatents = videoNoise * noiseScale + videoLatents * oneMinusScale
            audioLatents = audioNoise * noiseScale + audioLatents * oneMinusScale
            MLX.eval(videoLatents, audioLatents)
        }

        let stage2VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2H,
            width: stage2W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        let stage2VideoRope = precomputeSplitRope(
            positions: stage2VideoPositions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )
        let stage2VideoCrossPositions = stage2VideoPositions[0..., 0..<1, 0..., 0...]
        let stage2VideoCrossRope = precomputeSplitRope(
            positions: stage2VideoCrossPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )

        (videoLatents, audioLatents) = denoiseAVLoop(
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            videoRope: stage2VideoRope,
            audioRope: stage1AudioRope,
            videoCrossRope: stage2VideoCrossRope,
            audioCrossRope: stage1AudioRope,
            videoContext: videoContext,
            audioContext: audioContext,
            transformer: transformer,
            sigmas: STAGE2Sigmas,
            videoConditioning: stage2ConditioningState
        )
        MLX.eval(videoLatents, audioLatents)

        let decodedVideo: MLXArray?
        let frames: MLXArray
        if let tiling = selectDecodeTilingConfig(
            width: options.width,
            height: options.height,
            numFrames: options.numFrames,
            fps: options.fps
        ) {
            decodedVideo = nil
            frames = decodeWithTiling(
                decoder: decoder,
                latents: videoLatents,
                spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                spatialScale: 32,
                temporalScale: 8
            )
        } else {
            let fullDecoded = decoder.decode(sample: videoLatents, timestep: nil)
            decodedVideo = fullDecoded
            frames = postprocessDecodedVideo(fullDecoded)
        }
        MLX.eval(frames)

        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            if let decodedVideo {
                try? MLX.save(array: decodedVideo, url: parent.appendingPathComponent("\(stem)_decoded.npy"))
            }
            try? MLX.save(array: frames, url: parent.appendingPathComponent("\(stem)_frames_postprocess.npy"))
            try? MLX.save(array: videoLatents, url: parent.appendingPathComponent("\(stem)_video_latents.npy"))
            try? MLX.save(array: audioLatents, url: parent.appendingPathComponent("\(stem)_audio_latents.npy"))
        }

        _ = decodedVideo
        let mel = audioDecoder.decode(latents: audioLatents.asType(.float32))
        saveLTXAVDebugArray(mel, suffix: "audio_mel")
        let vocodedAudio = vocoder(mel)
        saveLTXAVDebugAudio(vocodedAudio, suffix: "audio_vocoded_raw", sampleRate: vocoder.outputSamplingRate)
        let audioWaveform = matchLTXAudioWaveformDuration(
            vocodedAudio,
            videoFrames: options.numFrames,
            fps: options.fps,
            sampleRate: vocoder.outputSamplingRate
        )
        MLX.eval(audioWaveform)
        saveLTXAVDebugAudio(audioWaveform, suffix: "audio_waveform_matched", sampleRate: vocoder.outputSamplingRate)

        return LTXUnifiedAVGenerationResult(
            frames: frames,
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            audioWaveform: audioWaveform,
            audioSampleRate: vocoder.outputSamplingRate
        )
    }
}

func matchLTXAudioWaveformDuration(
    _ waveform: MLXArray,
    videoFrames: Int,
    fps: Int,
    sampleRate: Int
) -> MLXArray {
    guard waveform.ndim == 3,
          videoFrames > 0,
          fps > 0,
          sampleRate > 0 else {
        return waveform
    }

    let targetSamples = max(1, Int((Double(videoFrames) * Double(sampleRate) / Double(fps)).rounded()))
    let currentSamples = waveform.dim(2)
    if currentSamples == targetSamples {
        return waveform
    }
    if currentSamples > targetSamples {
        return waveform[0..., 0..., 0..<targetSamples]
    }

    let padding = MLX.zeros(
        [waveform.dim(0), waveform.dim(1), targetSamples - currentSamples],
        dtype: waveform.dtype
    )
    return MLX.concatenated([waveform, padding], axis: 2)
}

private func ltxAVDebugBaseURL() -> URL? {
    guard let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"],
          !debugPrefix.isEmpty else {
        return nil
    }
    return URL(fileURLWithPath: debugPrefix).standardizedFileURL
}

private func ltxAVDebugURL(suffix: String, fileExtension: String) -> URL? {
    guard let base = ltxAVDebugBaseURL() else { return nil }
    let parent = base.deletingLastPathComponent()
    try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    return parent.appendingPathComponent("\(base.lastPathComponent)_\(suffix).\(fileExtension)", isDirectory: false)
}

private func saveLTXAVDebugArray(_ array: MLXArray, suffix: String) {
    guard let url = ltxAVDebugURL(suffix: suffix, fileExtension: "npy") else { return }
    try? MLX.save(array: array, url: url)
}

private func saveLTXAVDebugAudio(_ waveform: MLXArray, suffix: String, sampleRate: Int) {
    guard let url = ltxAVDebugURL(suffix: suffix, fileExtension: "wav") else { return }
    guard let prepared = try? LTXVideoMP4Writer.prepareAudio(waveform) else { return }
    try? MediaAudioIO.writeFloatWAV(
        samples: prepared.interleaved,
        sampleRate: sampleRate,
        channels: prepared.channels,
        to: url
    )
}

private func resolveLTX23TextEncoderRoot(modelRoot: URL) throws -> URL {
    let fm = FileManager.default
    let localTextEncoder = modelRoot.appendingPathComponent("text_encoder", isDirectory: true)
    if fm.fileExists(atPath: localTextEncoder.appendingPathComponent("config.json").path) {
        return localTextEncoder
    }

    if let override = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT"],
       !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return URL(fileURLWithPath: override).standardizedFileURL
    }

    let companionID = ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue
    if let installed = ManagedModelResolver.resolveInstalledModel(id: companionID) {
        return installed
    }

    throw LTXUnifiedAVGeneratorError.ltx23TextEncoderMissing(companionID)
}

private func denoiseAVLoop(
    videoLatents: MLXArray,
    audioLatents: MLXArray,
    videoRope: (cos: MLXArray, sin: MLXArray),
    audioRope: (cos: MLXArray, sin: MLXArray),
    videoCrossRope: (cos: MLXArray, sin: MLXArray),
    audioCrossRope: (cos: MLXArray, sin: MLXArray),
    videoContext: MLXArray,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoConditioning: LTXLatentConditioningState?
) -> (MLXArray, MLXArray) {
    var currentVideo = videoLatents
    var currentAudio = audioLatents
    let dtype = videoLatents.dtype

    for i in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[i]
        let nextSigma = sigmas[i + 1]

        let b = currentVideo.dim(0)
        let c = currentVideo.dim(1)
        let f = currentVideo.dim(2)
        let h = currentVideo.dim(3)
        let w = currentVideo.dim(4)
        let videoTokenCount = f * h * w
        let flatVideo = currentVideo.transposed(0, 2, 3, 4, 1).reshaped(b, videoTokenCount, c)

        let ab = currentAudio.dim(0)
        let ac = currentAudio.dim(1)
        let at = currentAudio.dim(2)
        let af = currentAudio.dim(3)
        let flatAudio = currentAudio.transposed(0, 2, 1, 3).reshaped(ab, at, ac * af)

        let videoTimesteps: MLXArray
        if let videoConditioning {
            let mask = videoConditioning.denoiseMask.reshaped(b, 1, f, 1, 1)
            let broadcastMask = broadcast(mask, to: [b, 1, f, h, w]).reshaped(b, videoTokenCount)
            videoTimesteps = MLXArray(sigma).asType(dtype) * broadcastMask
        } else {
            videoTimesteps = MLX.full([b, videoTokenCount], values: MLXArray(sigma).asType(dtype))
        }
        let audioTimesteps = MLX.full([ab, at], values: MLXArray(sigma).asType(dtype))
        let globalTimestep = MLX.full([b], values: MLXArray(sigma).asType(dtype))

        let velocity = transformer.forward(
            videoLatent: flatVideo,
            audioLatent: flatAudio,
            timestep: globalTimestep,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope
        )

        let videoVelocity = velocity.videoVelocity
            .reshaped(b, f, h, w, c)
            .transposed(0, 4, 1, 2, 3)
        let audioVelocity = velocity.audioVelocity
            .reshaped(ab, at, ac, af)
            .transposed(0, 2, 1, 3)
        MLX.eval(videoVelocity, audioVelocity)

        var denoisedVideo = toDenoised(noisy: currentVideo, velocity: videoVelocity, sigma: sigma)
        if let videoConditioning {
            let one = MLXArray(1.0).asType(denoisedVideo.dtype)
            denoisedVideo = denoisedVideo * videoConditioning.denoiseMask + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
        }
        let denoisedAudio = toDenoised(noisy: currentAudio, velocity: audioVelocity, sigma: sigma)
        MLX.eval(denoisedVideo, denoisedAudio)

        if nextSigma > 0 {
            let sigmaArr = MLXArray(sigma).asType(dtype)
            let nextArr = MLXArray(nextSigma).asType(dtype)
            currentVideo = denoisedVideo + nextArr * (currentVideo - denoisedVideo) / sigmaArr
            currentAudio = denoisedAudio + nextArr * (currentAudio - denoisedAudio) / sigmaArr
        } else {
            currentVideo = denoisedVideo
            currentAudio = denoisedAudio
        }
        MLX.eval(currentVideo, currentAudio)
    }

    return (currentVideo, currentAudio)
}

private func computeAudioLatentFrameCount(videoFrames: Int, fps: Int) -> Int {
    let duration = Double(videoFrames) / Double(max(1, fps))
    let latentsPerSecond = Double(LTXAudioLatentSampleRate) / Double(LTXAudioHopLength) / Double(LTXAudioLatentDownsampleFactor)
    return max(1, Int((duration * latentsPerSecond).rounded()))
}

private func createAudioPositionGrid(
    batchSize: Int,
    audioFrames: Int,
    sampleRate: Float = Float(LTXAudioLatentSampleRate),
    hopLength: Float = Float(LTXAudioHopLength),
    downsampleFactor: Int = LTXAudioLatentDownsampleFactor,
    causalFix: Bool = true
) -> MLXArray {
    var values = [Float](repeating: 0, count: batchSize * audioFrames * 2)
    for b in 0..<batchSize {
        for t in 0..<audioFrames {
            var startFrame = Float(t * downsampleFactor)
            var endFrame = Float((t + 1) * downsampleFactor)
            if causalFix {
                let shift = Float(1 - downsampleFactor)
                startFrame = max(0, startFrame + shift)
                endFrame = max(0, endFrame + shift)
            }

            let startSeconds = (startFrame * hopLength) / sampleRate
            let endSeconds = (endFrame * hopLength) / sampleRate
            let idx = (b * audioFrames + t) * 2
            values[idx] = startSeconds
            values[idx + 1] = endSeconds
        }
    }
    return MLXArray(values).reshaped(batchSize, 1, audioFrames, 2).asType(.float32)
}

private protocol LTXUnifiedAVTransformerRuntime: Module {
    func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray)
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray)
}

private final class LTXUnifiedAVTransformer: Module, LTXUnifiedAVTransformerRuntime {
    let videoDim = 4096
    let audioDim = 2048
    let videoHeads = 32
    let videoHeadDim = 128
    let audioHeads = 32
    let audioHeadDim = 64
    let timestepScaleMultiplier: Float = 1000.0

    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "adaln_single") var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "caption_projection") var captionProjection: LTXPixArtTextProjection
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear

    @ModuleInfo(key: "audio_patchify_proj") var audioPatchifyProj: Linear
    @ModuleInfo(key: "audio_adaln_single") var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_caption_projection") var audioCaptionProjection: LTXPixArtTextProjection
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_norm_out") var audioNormOut: LayerNorm
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear

    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") var avCaVideoScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") var avCaAudioScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") var avCaA2VGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") var avCaV2AGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTXUnifiedAVTransformerBlock]

    override init() {
        self._patchifyProj.wrappedValue = Linear(128, 4096, bias: true)
        self._adalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 6)
        self._captionProjection.wrappedValue = LTXPixArtTextProjection(inFeatures: 3840, hiddenSize: 4096, outFeatures: 4096, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([2, 4096], dtype: .float32)
        self._normOut.wrappedValue = LayerNorm(dimensions: 4096, eps: 1e-6, affine: false)
        self._projOut.wrappedValue = Linear(4096, 128, bias: true)

        self._audioPatchifyProj.wrappedValue = Linear(128, 2048, bias: true)
        self._audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 2048, embeddingCoefficient: 6)
        self._audioCaptionProjection.wrappedValue = LTXPixArtTextProjection(inFeatures: 3840, hiddenSize: 2048, outFeatures: 2048, bias: true)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([2, 2048], dtype: .float32)
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: 2048, eps: 1e-6, affine: false)
        self._audioProjOut.wrappedValue = Linear(2048, 128, bias: true)

        self._avCaVideoScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 4)
        self._avCaAudioScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 2048, embeddingCoefficient: 4)
        self._avCaA2VGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 1)
        self._avCaV2AGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 2048, embeddingCoefficient: 1)
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXUnifiedAVTransformerBlock()
        }
        super.init()
    }

    func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timestep _: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray)
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let videoSteps = videoTimesteps ?? MLX.zeros([videoLatent.dim(0), videoLatent.dim(1)], dtype: videoLatent.dtype)
        let audioSteps = audioTimesteps ?? MLX.zeros([audioLatent.dim(0), audioLatent.dim(1)], dtype: audioLatent.dtype)
        return forward(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            videoTimesteps: videoSteps,
            audioTimesteps: audioSteps,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope
        )
    }

    func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        videoTimesteps: MLXArray,
        audioTimesteps: MLXArray,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray)
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let videoBatch = videoLatent.dim(0)
        let videoTokens = videoLatent.dim(1)
        let audioBatch = audioLatent.dim(0)
        let audioTokens = audioLatent.dim(1)

        var videoX = patchifyProj(videoLatent)
        var audioX = audioPatchifyProj(audioLatent)

        let scaledVideoTimesteps = videoTimesteps.asType(videoX.dtype) * MLXArray(timestepScaleMultiplier).asType(videoX.dtype)
        let scaledAudioTimesteps = audioTimesteps.asType(audioX.dtype) * MLXArray(timestepScaleMultiplier).asType(audioX.dtype)

        let (videoTimeParamsFlat, videoEmbeddedFlat) = adalnSingle(timestep: scaledVideoTimesteps.reshaped(-1), hiddenDType: videoX.dtype)
        let (audioTimeParamsFlat, audioEmbeddedFlat) = audioAdalnSingle(timestep: scaledAudioTimesteps.reshaped(-1), hiddenDType: audioX.dtype)

        let videoTimeParams = videoTimeParamsFlat.reshaped(videoBatch, videoTokens, -1)
        let audioTimeParams = audioTimeParamsFlat.reshaped(audioBatch, audioTokens, -1)
        let videoEmbedded = videoEmbeddedFlat.reshaped(videoBatch, videoTokens, -1)
        let audioEmbedded = audioEmbeddedFlat.reshaped(audioBatch, audioTokens, -1)

        let projectedVideoContext = captionProjection(videoContext).reshaped(videoBatch, videoContext.dim(1), videoDim)
        let projectedAudioContext = audioCaptionProjection(audioContext).reshaped(audioBatch, audioContext.dim(1), audioDim)

        let (videoCrossScaleShiftFlat, _) = avCaVideoScaleShiftAdalnSingle(
            timestep: scaledVideoTimesteps.reshaped(-1),
            hiddenDType: videoX.dtype
        )
        let (audioCrossScaleShiftFlat, _) = avCaAudioScaleShiftAdalnSingle(
            timestep: scaledAudioTimesteps.reshaped(-1),
            hiddenDType: audioX.dtype
        )
        let (videoCrossGateFlat, _) = avCaA2VGateAdalnSingle(
            timestep: scaledVideoTimesteps.reshaped(-1),
            hiddenDType: videoX.dtype
        )
        let (audioCrossGateFlat, _) = avCaV2AGateAdalnSingle(
            timestep: scaledAudioTimesteps.reshaped(-1),
            hiddenDType: audioX.dtype
        )

        let videoCrossScaleShift = videoCrossScaleShiftFlat.reshaped(videoBatch, videoTokens, -1)
        let audioCrossScaleShift = audioCrossScaleShiftFlat.reshaped(audioBatch, audioTokens, -1)
        let videoCrossGate = videoCrossGateFlat.reshaped(videoBatch, videoTokens, -1)
        let audioCrossGate = audioCrossGateFlat.reshaped(audioBatch, audioTokens, -1)

        for block in transformerBlocks {
            let out = block(
                videoX: videoX,
                audioX: audioX,
                videoContext: projectedVideoContext,
                audioContext: projectedAudioContext,
                videoTimestepEmb: videoTimeParams,
                audioTimestepEmb: audioTimeParams,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                videoCrossScaleShiftTimestep: videoCrossScaleShift,
                audioCrossScaleShiftTimestep: audioCrossScaleShift,
                videoCrossGateTimestep: videoCrossGate,
                audioCrossGateTimestep: audioCrossGate
            )
            videoX = out.videoX
            audioX = out.audioX
        }

        let videoScaleShift = scaleShiftTable.reshaped(1, 1, 2, videoDim) + videoEmbedded.reshaped(videoBatch, videoTokens, 1, videoDim)
        let videoShift = videoScaleShift[0..., 0..., 0, 0...]
        let videoScale = videoScaleShift[0..., 0..., 1, 0...]
        videoX = normOut(videoX)
        videoX = videoX * (MLXArray(1.0).asType(videoX.dtype) + videoScale) + videoShift
        let videoVelocity = projOut(videoX)

        let audioScaleShift = audioScaleShiftTable.reshaped(1, 1, 2, audioDim) + audioEmbedded.reshaped(audioBatch, audioTokens, 1, audioDim)
        let audioShift = audioScaleShift[0..., 0..., 0, 0...]
        let audioScale = audioScaleShift[0..., 0..., 1, 0...]
        audioX = audioNormOut(audioX)
        audioX = audioX * (MLXArray(1.0).asType(audioX.dtype) + audioScale) + audioShift
        let audioVelocity = audioProjOut(audioX)

        return (videoVelocity, audioVelocity)
    }
}

private final class LTXUnifiedAVTransformerBlock: Module {
    let videoDim = 4096
    let audioDim = 2048

    @ModuleInfo(key: "attn1") var attn1: LTXDistilledAttention
    @ModuleInfo(key: "attn2") var attn2: LTXDistilledAttention
    @ModuleInfo(key: "ff") var ff: LTXDistilledFeedForward
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    @ModuleInfo(key: "audio_attn1") var audioAttn1: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn2") var audioAttn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_ff") var audioFF: LTXDistilledFeedForward
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray

    @ModuleInfo(key: "audio_to_video_attn") var audioToVideoAttn: LTXDistilledAttention
    @ModuleInfo(key: "video_to_audio_attn") var videoToAudioAttn: LTXDistilledAttention
    @ModuleInfo(key: "scale_shift_table_a2v_ca_audio") var scaleShiftTableA2VCAAudio: MLXArray
    @ModuleInfo(key: "scale_shift_table_a2v_ca_video") var scaleShiftTableA2VCAVideo: MLXArray

    override init() {
        self._attn1.wrappedValue = LTXDistilledAttention(queryDim: 4096, contextDim: nil, heads: 32, headDim: 128, normEps: 1e-6)
        self._attn2.wrappedValue = LTXDistilledAttention(queryDim: 4096, contextDim: 4096, heads: 32, headDim: 128, normEps: 1e-6)
        self._ff.wrappedValue = LTXDistilledFeedForward(dim: 4096, dimOut: 4096, mult: 4)
        self._scaleShiftTable.wrappedValue = MLX.zeros([6, 4096], dtype: .float32)

        self._audioAttn1.wrappedValue = LTXDistilledAttention(queryDim: 2048, contextDim: nil, heads: 32, headDim: 64, normEps: 1e-6)
        self._audioAttn2.wrappedValue = LTXDistilledAttention(queryDim: 2048, contextDim: 2048, heads: 32, headDim: 64, normEps: 1e-6)
        self._audioFF.wrappedValue = LTXDistilledFeedForward(dim: 2048, dimOut: 2048, mult: 4)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([6, 2048], dtype: .float32)

        self._audioToVideoAttn.wrappedValue = LTXDistilledAttention(queryDim: 4096, contextDim: 2048, heads: 32, headDim: 64, normEps: 1e-6)
        self._videoToAudioAttn.wrappedValue = LTXDistilledAttention(queryDim: 2048, contextDim: 4096, heads: 32, headDim: 64, normEps: 1e-6)
        self._scaleShiftTableA2VCAAudio.wrappedValue = MLX.zeros([5, 2048], dtype: .float32)
        self._scaleShiftTableA2VCAVideo.wrappedValue = MLX.zeros([5, 4096], dtype: .float32)
        super.init()
    }

    func callAsFunction(
        videoX: MLXArray,
        audioX: MLXArray,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoTimestepEmb: MLXArray,
        audioTimestepEmb: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        videoCrossScaleShiftTimestep: MLXArray,
        audioCrossScaleShiftTimestep: MLXArray,
        videoCrossGateTimestep: MLXArray,
        audioCrossGateTimestep: MLXArray
    ) -> (videoX: MLXArray, audioX: MLXArray) {
        var vx = videoX
        var ax = audioX

        let videoAda = scaleShiftTable.reshaped(1, 1, 6, videoDim) + videoTimestepEmb.reshaped(vx.dim(0), vx.dim(1), 6, videoDim)
        let vShiftMSA = videoAda[0..., 0..., 0, 0...]
        let vScaleMSA = videoAda[0..., 0..., 1, 0...]
        let vGateMSA = videoAda[0..., 0..., 2, 0...]

        var normVX = rmsNormNoWeight(vx)
        normVX = normVX * (MLXArray(1.0).asType(normVX.dtype) + vScaleMSA) + vShiftMSA
        vx = vx + attn1(normVX, context: nil, mask: nil, rope: videoRope) * vGateMSA
        vx = vx + attn2(rmsNormNoWeight(vx), context: videoContext, mask: nil, rope: nil)

        let audioAda = audioScaleShiftTable.reshaped(1, 1, 6, audioDim) + audioTimestepEmb.reshaped(ax.dim(0), ax.dim(1), 6, audioDim)
        let aShiftMSA = audioAda[0..., 0..., 0, 0...]
        let aScaleMSA = audioAda[0..., 0..., 1, 0...]
        let aGateMSA = audioAda[0..., 0..., 2, 0...]

        var normAX = rmsNormNoWeight(ax)
        normAX = normAX * (MLXArray(1.0).asType(normAX.dtype) + aScaleMSA) + aShiftMSA
        ax = ax + audioAttn1(normAX, context: nil, mask: nil, rope: audioRope) * aGateMSA
        ax = ax + audioAttn2(rmsNormNoWeight(ax), context: audioContext, mask: nil, rope: nil)

        let a2vAudio = crossAdaValues(
            table: scaleShiftTableA2VCAAudio,
            scaleShiftTimestep: audioCrossScaleShiftTimestep,
            gateTimestep: audioCrossGateTimestep,
            dim: audioDim
        )
        let a2vVideo = crossAdaValues(
            table: scaleShiftTableA2VCAVideo,
            scaleShiftTimestep: videoCrossScaleShiftTimestep,
            gateTimestep: videoCrossGateTimestep,
            dim: videoDim
        )

        let normVX3 = rmsNormNoWeight(vx)
        let normAX3 = rmsNormNoWeight(ax)

        let vxScaledA2V = normVX3 * (MLXArray(1.0).asType(normVX3.dtype) + a2vVideo.scaleA2V) + a2vVideo.shiftA2V
        let axScaledA2V = normAX3 * (MLXArray(1.0).asType(normAX3.dtype) + a2vAudio.scaleA2V) + a2vAudio.shiftA2V
        vx = vx + audioToVideoAttn(
            vxScaledA2V,
            context: axScaledA2V,
            mask: nil,
            rope: videoCrossRope,
            keyRope: audioCrossRope
        ) * a2vVideo.gate

        let axScaledV2A = normAX3 * (MLXArray(1.0).asType(normAX3.dtype) + a2vAudio.scaleV2A) + a2vAudio.shiftV2A
        let vxScaledV2A = normVX3 * (MLXArray(1.0).asType(normVX3.dtype) + a2vVideo.scaleV2A) + a2vVideo.shiftV2A
        ax = ax + videoToAudioAttn(
            axScaledV2A,
            context: vxScaledV2A,
            mask: nil,
            rope: audioCrossRope,
            keyRope: videoCrossRope
        ) * a2vAudio.gate

        let vShiftMLP = videoAda[0..., 0..., 3, 0...]
        let vScaleMLP = videoAda[0..., 0..., 4, 0...]
        let vGateMLP = videoAda[0..., 0..., 5, 0...]
        var vMLPInput = rmsNormNoWeight(vx)
        vMLPInput = vMLPInput * (MLXArray(1.0).asType(vMLPInput.dtype) + vScaleMLP) + vShiftMLP
        vx = vx + ff(vMLPInput) * vGateMLP

        let aShiftMLP = audioAda[0..., 0..., 3, 0...]
        let aScaleMLP = audioAda[0..., 0..., 4, 0...]
        let aGateMLP = audioAda[0..., 0..., 5, 0...]
        var aMLPInput = rmsNormNoWeight(ax)
        aMLPInput = aMLPInput * (MLXArray(1.0).asType(aMLPInput.dtype) + aScaleMLP) + aShiftMLP
        ax = ax + audioFF(aMLPInput) * aGateMLP

        return (vx, ax)
    }

    private func crossAdaValues(
        table: MLXArray,
        scaleShiftTimestep: MLXArray,
        gateTimestep: MLXArray,
        dim: Int
    ) -> (scaleA2V: MLXArray, shiftA2V: MLXArray, scaleV2A: MLXArray, shiftV2A: MLXArray, gate: MLXArray) {
        let batch = scaleShiftTimestep.dim(0)
        let tokens = scaleShiftTimestep.dim(1)

        let scaleShiftValues = table[0..<4, 0...].reshaped(1, 1, 4, dim)
            + scaleShiftTimestep.reshaped(batch, tokens, 4, dim)
        let gateValues = table[4..<5, 0...].reshaped(1, 1, 1, dim)
            + gateTimestep.reshaped(batch, tokens, 1, dim)

        return (
            scaleShiftValues[0..., 0..., 0, 0...],
            scaleShiftValues[0..., 0..., 1, 0...],
            scaleShiftValues[0..., 0..., 2, 0...],
            scaleShiftValues[0..., 0..., 3, 0...],
            gateValues[0..., 0..., 0, 0...]
        )
    }
}

private final class LTXUnifiedAVTransformerV2: Module, LTXUnifiedAVTransformerRuntime {
    let videoDim = 4096
    let audioDim = 2048
    let videoHeads = 32
    let videoHeadDim = 128
    let audioHeads = 32
    let audioHeadDim = 64
    let avCrossHeads = 32
    let avCrossHeadDim = 64
    let timestepScaleMultiplier: Float = 1000.0
    let avCaTimestepScaleMultiplier: Float = 1000.0

    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "audio_patchify_proj") var audioPatchifyProj: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "adaln_single") var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_adaln_single") var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "prompt_adaln_single") var promptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") var audioPromptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") var avCaVideoScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") var avCaAudioScaleShiftAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") var avCaA2VGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") var avCaV2AGateAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTXUnifiedAVTransformerV2Block]

    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "audio_norm_out") var audioNormOut: LayerNorm

    override init() {
        self._patchifyProj.wrappedValue = Linear(128, videoDim, bias: true)
        self._audioPatchifyProj.wrappedValue = Linear(128, audioDim, bias: true)
        self._projOut.wrappedValue = Linear(videoDim, 128, bias: true)
        self._audioProjOut.wrappedValue = Linear(audioDim, 128, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([2, videoDim], dtype: .float32)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        self._adalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: videoDim, embeddingCoefficient: 9)
        self._audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: audioDim, embeddingCoefficient: 9)
        self._promptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: videoDim, embeddingCoefficient: 2)
        self._audioPromptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 2
        )
        self._avCaVideoScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: videoDim,
            embeddingCoefficient: 4
        )
        self._avCaAudioScaleShiftAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 4
        )
        self._avCaA2VGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: videoDim,
            embeddingCoefficient: 1
        )
        self._avCaV2AGateAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 1
        )
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXUnifiedAVTransformerV2Block()
        }
        self._normOut.wrappedValue = LayerNorm(dimensions: videoDim, eps: 1e-6, affine: false)
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: audioDim, eps: 1e-6, affine: false)
        super.init()
    }

    func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray)
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let videoBatch = videoLatent.dim(0)
        let videoTokens = videoLatent.dim(1)
        let audioBatch = audioLatent.dim(0)
        let audioTokens = audioLatent.dim(1)

        var videoX = patchifyProj(videoLatent.asType(.bfloat16))
        var audioX = audioPatchifyProj(audioLatent.asType(.bfloat16))
        let globalTimestep = timestep.asType(videoX.dtype).reshaped(-1)
        let scaledGlobal = globalTimestep * MLXArray(timestepScaleMultiplier).asType(globalTimestep.dtype)
        let scaledAVGate = globalTimestep * MLXArray(avCaTimestepScaleMultiplier).asType(globalTimestep.dtype)

        let (videoAdalnParams, videoEmbedded): (MLXArray, MLXArray)
        let (avCaVideoParams, _): (MLXArray, MLXArray)
        if let videoTimesteps {
            let scaled = videoTimesteps.asType(videoX.dtype) * MLXArray(timestepScaleMultiplier).asType(videoX.dtype)
            let videoFlat = scaled.reshaped(-1)
            let adaln = adalnSingle(timestep: videoFlat, hiddenDType: videoX.dtype)
            videoAdalnParams = adaln.0.reshaped(videoBatch, videoTokens, -1)
            videoEmbedded = adaln.1.reshaped(videoBatch, videoTokens, -1)
            let avCa = avCaVideoScaleShiftAdalnSingle(timestep: videoFlat, hiddenDType: videoX.dtype)
            avCaVideoParams = avCa.0.reshaped(videoBatch, videoTokens, -1)
        } else {
            (videoAdalnParams, videoEmbedded) = adalnSingle(timestep: scaledGlobal, hiddenDType: videoX.dtype)
            (avCaVideoParams, _) = avCaVideoScaleShiftAdalnSingle(timestep: scaledGlobal, hiddenDType: videoX.dtype)
        }

        let (audioAdalnParams, audioEmbedded): (MLXArray, MLXArray)
        let (avCaAudioParams, _): (MLXArray, MLXArray)
        if let audioTimesteps {
            let scaled = audioTimesteps.asType(audioX.dtype) * MLXArray(timestepScaleMultiplier).asType(audioX.dtype)
            let audioFlat = scaled.reshaped(-1)
            let adaln = audioAdalnSingle(timestep: audioFlat, hiddenDType: audioX.dtype)
            audioAdalnParams = adaln.0.reshaped(audioBatch, audioTokens, -1)
            audioEmbedded = adaln.1.reshaped(audioBatch, audioTokens, -1)
            let avCa = avCaAudioScaleShiftAdalnSingle(timestep: audioFlat, hiddenDType: audioX.dtype)
            avCaAudioParams = avCa.0.reshaped(audioBatch, audioTokens, -1)
        } else {
            (audioAdalnParams, audioEmbedded) = audioAdalnSingle(timestep: scaledGlobal, hiddenDType: audioX.dtype)
            (avCaAudioParams, _) = avCaAudioScaleShiftAdalnSingle(timestep: scaledGlobal, hiddenDType: audioX.dtype)
        }

        let (avCaA2VGateParams, _) = avCaA2VGateAdalnSingle(timestep: scaledAVGate, hiddenDType: videoX.dtype)
        let (avCaV2AGateParams, _) = avCaV2AGateAdalnSingle(timestep: scaledAVGate, hiddenDType: audioX.dtype)
        let (videoPromptParams, _) = promptAdalnSingle(timestep: scaledGlobal, hiddenDType: videoX.dtype)
        let (audioPromptParams, _) = audioPromptAdalnSingle(timestep: scaledGlobal, hiddenDType: audioX.dtype)

        let evalEvery = Int(ProcessInfo.processInfo.environment["LTX2_DIT_EVAL_EVERY"] ?? "8") ?? 8
        for (index, block) in transformerBlocks.enumerated() {
            let out = block(
                videoHidden: videoX,
                audioHidden: audioX,
                videoAdalnParams: videoAdalnParams,
                audioAdalnParams: audioAdalnParams,
                videoPromptAdalnParams: videoPromptParams,
                audioPromptAdalnParams: audioPromptParams,
                avCaVideoParams: avCaVideoParams,
                avCaAudioParams: avCaAudioParams,
                avCaA2VGateParams: avCaA2VGateParams,
                avCaV2AGateParams: avCaV2AGateParams,
                videoTextEmbeds: videoContext.asType(videoX.dtype),
                audioTextEmbeds: audioContext.asType(audioX.dtype),
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope
            )
            videoX = out.video
            audioX = out.audio
            if evalEvery > 0 && (index + 1).isMultiple(of: evalEvery) {
                MLX.eval(videoX, audioX)
            }
        }

        let videoVelocity = outputBlock(
            videoX,
            embeddedTimestep: videoEmbedded,
            table: scaleShiftTable,
            norm: normOut,
            projection: projOut,
            dim: videoDim
        )
        let audioVelocity = outputBlock(
            audioX,
            embeddedTimestep: audioEmbedded,
            table: audioScaleShiftTable,
            norm: audioNormOut,
            projection: audioProjOut,
            dim: audioDim
        )
        return (videoVelocity, audioVelocity)
    }

    private func outputBlock(
        _ x: MLXArray,
        embeddedTimestep: MLXArray,
        table: MLXArray,
        norm: LayerNorm,
        projection: Linear,
        dim: Int
    ) -> MLXArray {
        let embedded = embeddedTimestep.ndim == 2
            ? embeddedTimestep.expandedDimensions(axis: 1)
            : embeddedTimestep
        let scaleShift = table.reshaped(1, 1, 2, dim) + embedded.expandedDimensions(axis: 2)
        let shift = scaleShift[0..., 0..., 0, 0...]
        let scale = scaleShift[0..., 0..., 1, 0...]
        var y = norm(x)
        y = y * (MLXArray(1.0).asType(y.dtype) + scale) + shift
        return projection(y)
    }
}

private final class LTXUnifiedAVTransformerV2Block: Module {
    let videoDim = 4096
    let audioDim = 2048

    @ModuleInfo(key: "attn1") var attn1: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn1") var audioAttn1: LTXDistilledAttention
    @ModuleInfo(key: "attn2") var attn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn2") var audioAttn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_to_video_attn") var audioToVideoAttn: LTXDistilledAttention
    @ModuleInfo(key: "video_to_audio_attn") var videoToAudioAttn: LTXDistilledAttention
    @ModuleInfo(key: "ff") var ff: LTXDistilledFeedForward
    @ModuleInfo(key: "audio_ff") var audioFF: LTXDistilledFeedForward
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "prompt_scale_shift_table") var promptScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_prompt_scale_shift_table") var audioPromptScaleShiftTable: MLXArray
    @ModuleInfo(key: "scale_shift_table_a2v_ca_video") var scaleShiftTableA2VCAVideo: MLXArray
    @ModuleInfo(key: "scale_shift_table_a2v_ca_audio") var scaleShiftTableA2VCAAudio: MLXArray

    override init() {
        self._attn1.wrappedValue = LTXDistilledAttention(
            queryDim: videoDim,
            contextDim: nil,
            heads: 32,
            headDim: 128,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._audioAttn1.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: nil,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._attn2.wrappedValue = LTXDistilledAttention(
            queryDim: videoDim,
            contextDim: videoDim,
            heads: 32,
            headDim: 128,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._audioAttn2.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: audioDim,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._audioToVideoAttn.wrappedValue = LTXDistilledAttention(
            queryDim: videoDim,
            contextDim: audioDim,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._videoToAudioAttn.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: videoDim,
            heads: 32,
            headDim: 64,
            normEps: 1e-6,
            applyGatedAttention: true
        )
        self._ff.wrappedValue = LTXDistilledFeedForward(dim: videoDim, dimOut: videoDim, mult: 4)
        self._audioFF.wrappedValue = LTXDistilledFeedForward(dim: audioDim, dimOut: audioDim, mult: 4)
        self._scaleShiftTable.wrappedValue = MLX.zeros([9, videoDim], dtype: .float32)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([9, audioDim], dtype: .float32)
        self._promptScaleShiftTable.wrappedValue = MLX.zeros([2, videoDim], dtype: .float32)
        self._audioPromptScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        self._scaleShiftTableA2VCAVideo.wrappedValue = MLX.zeros([5, videoDim], dtype: .float32)
        self._scaleShiftTableA2VCAAudio.wrappedValue = MLX.zeros([5, audioDim], dtype: .float32)
        super.init()
    }

    func callAsFunction(
        videoHidden: MLXArray,
        audioHidden: MLXArray,
        videoAdalnParams: MLXArray,
        audioAdalnParams: MLXArray,
        videoPromptAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray,
        avCaVideoParams: MLXArray,
        avCaAudioParams: MLXArray,
        avCaA2VGateParams: MLXArray,
        avCaV2AGateParams: MLXArray,
        videoTextEmbeds: MLXArray,
        audioTextEmbeds: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray)
    ) -> (video: MLXArray, audio: MLXArray) {
        var video = videoHidden
        var audio = audioHidden

        let vAda = unpackAdaln(videoAdalnParams, table: scaleShiftTable, count: 9, dim: videoDim)
        let aAda = unpackAdaln(audioAdalnParams, table: audioScaleShiftTable, count: 9, dim: audioDim)

        var videoNorm = rmsNormNoWeight(video)
        videoNorm = videoNorm * (MLXArray(1.0).asType(videoNorm.dtype) + vAda[1]) + vAda[0]
        video = video + attn1(videoNorm, context: nil, mask: nil, rope: videoRope) * vAda[2]

        var audioNorm = rmsNormNoWeight(audio)
        audioNorm = audioNorm * (MLXArray(1.0).asType(audioNorm.dtype) + aAda[1]) + aAda[0]
        audio = audio + audioAttn1(audioNorm, context: nil, mask: nil, rope: audioRope) * aAda[2]

        let vPromptAda = unpackAdaln(videoPromptAdalnParams, table: promptScaleShiftTable, count: 2, dim: videoDim)
        let videoText = videoTextEmbeds * (MLXArray(1.0).asType(videoTextEmbeds.dtype) + vPromptAda[1]) + vPromptAda[0]
        var videoCrossNorm = rmsNormNoWeight(video)
        videoCrossNorm = videoCrossNorm * (MLXArray(1.0).asType(videoCrossNorm.dtype) + vAda[7]) + vAda[6]
        video = video + attn2(videoCrossNorm, context: videoText, mask: nil, rope: nil) * vAda[8]

        let aPromptAda = unpackAdaln(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2,
            dim: audioDim
        )
        let audioText = audioTextEmbeds * (MLXArray(1.0).asType(audioTextEmbeds.dtype) + aPromptAda[1]) + aPromptAda[0]
        var audioCrossNorm = rmsNormNoWeight(audio)
        audioCrossNorm = audioCrossNorm * (MLXArray(1.0).asType(audioCrossNorm.dtype) + aAda[7]) + aAda[6]
        audio = audio + audioAttn2(audioCrossNorm, context: audioText, mask: nil, rope: nil) * aAda[8]

        let videoNorm3 = rmsNormNoWeight(video)
        let audioNorm3 = rmsNormNoWeight(audio)
        let vAV = unpackAdaln(avCaVideoParams, table: scaleShiftTableA2VCAVideo, count: 4, dim: videoDim)
        let aAV = unpackAdaln(avCaAudioParams, table: scaleShiftTableA2VCAAudio, count: 4, dim: audioDim)
        let a2vGate = unpackAVGate(avCaA2VGateParams, table: scaleShiftTableA2VCAVideo, dim: videoDim)
        let v2aGate = unpackAVGate(avCaV2AGateParams, table: scaleShiftTableA2VCAAudio, dim: audioDim)

        let videoQA2V = videoNorm3 * (MLXArray(1.0).asType(videoNorm3.dtype) + vAV[0]) + vAV[1]
        let audioKVA2V = audioNorm3 * (MLXArray(1.0).asType(audioNorm3.dtype) + aAV[0]) + aAV[1]
        video = video + audioToVideoAttn(
            videoQA2V,
            context: audioKVA2V,
            mask: nil,
            rope: videoCrossRope,
            keyRope: audioCrossRope
        ) * a2vGate

        let audioQV2A = audioNorm3 * (MLXArray(1.0).asType(audioNorm3.dtype) + aAV[2]) + aAV[3]
        let videoKVV2A = videoNorm3 * (MLXArray(1.0).asType(videoNorm3.dtype) + vAV[2]) + vAV[3]
        audio = audio + videoToAudioAttn(
            audioQV2A,
            context: videoKVV2A,
            mask: nil,
            rope: audioCrossRope,
            keyRope: videoCrossRope
        ) * v2aGate

        var videoFFNorm = rmsNormNoWeight(video)
        videoFFNorm = videoFFNorm * (MLXArray(1.0).asType(videoFFNorm.dtype) + vAda[4]) + vAda[3]
        video = video + ff(videoFFNorm) * vAda[5]

        var audioFFNorm = rmsNormNoWeight(audio)
        audioFFNorm = audioFFNorm * (MLXArray(1.0).asType(audioFFNorm.dtype) + aAda[4]) + aAda[3]
        audio = audio + audioFF(audioFFNorm) * aAda[5]

        return (video, audio)
    }

    private func unpackAdaln(_ params: MLXArray, table: MLXArray, count: Int, dim: Int) -> [MLXArray] {
        if params.ndim == 2 {
            let values = params.reshaped(-1, count, dim) + table[0..<count, 0...].reshaped(1, count, dim)
            return (0..<count).map { values[0..., $0, 0...].expandedDimensions(axis: 1) }
        }

        let batch = params.dim(0)
        let tokens = params.dim(1)
        let values = params.reshaped(batch, tokens, count, dim)
            + table[0..<count, 0...].reshaped(1, 1, count, dim)
        return (0..<count).map { values[0..., 0..., $0, 0...] }
    }

    private func unpackAVGate(_ params: MLXArray, table: MLXArray, dim: Int) -> MLXArray {
        if params.ndim == 2 {
            return (params + table[4, 0...]).expandedDimensions(axis: 1)
        }
        return params + table[4, 0...].reshaped(1, 1, dim)
    }
}

private func mapUnifiedTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("model.diffusion_model.") else {
        return []
    }

    var mapped = String(key.dropFirst("model.diffusion_model.".count))
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.0.proj.", with: ".audio_ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.2.", with: ".audio_ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    if mapped.hasPrefix("video_embeddings_connector")
        || mapped.hasPrefix("audio_embeddings_connector")
        || mapped.hasPrefix("text_embedding_projection")
    {
        return []
    }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

func mapLTX23UnifiedTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("transformer.") else {
        return []
    }

    var mapped = String(key.dropFirst("transformer.".count))
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.0.proj.", with: ".audio_ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.2.", with: ".audio_ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    let ignoredPrefixes = [
        "text_embedding_projection",
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "caption_projection",
        "audio_caption_projection",
    ]
    if ignoredPrefixes.contains(where: { mapped.hasPrefix($0) }) {
        return []
    }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

private enum LTXAudioCausalityAxis {
    case none
    case height
}

private final class LTXAudioPixelNorm: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanSq = MLX.mean(x * x, axis: -1, keepDims: true)
        return x / MLX.sqrt(meanSq + MLXArray(eps).asType(x.dtype))
    }
}

private final class LTXAudioCausalConv2d: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    let causalityAxis: LTXAudioCausalityAxis
    let padTop: Int
    let padBottom: Int
    let padLeft: Int
    let padRight: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        causalityAxis: LTXAudioCausalityAxis = .height
    ) {
        self.causalityAxis = causalityAxis
        let pad = kernelSize - 1
        switch causalityAxis {
        case .none:
            self.padTop = pad / 2
            self.padBottom = pad - (pad / 2)
            self.padLeft = pad / 2
            self.padRight = pad - (pad / 2)
        case .height:
            self.padTop = pad
            self.padBottom = 0
            self.padLeft = pad / 2
            self.padRight = pad - (pad / 2)
        }
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init([kernelSize, kernelSize]),
            stride: .init([stride, stride]),
            padding: .init(0),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddedInput = padded(
            x,
            widths: [
                [0, 0],
                [padTop, padBottom],
                [padLeft, padRight],
                [0, 0],
            ]
        )
        return conv(paddedInput)
    }
}

private final class LTXAudioResnetBlock2D: Module {
    let inChannels: Int
    let outChannels: Int

    @ModuleInfo(key: "norm1") var norm1: LTXAudioPixelNorm
    @ModuleInfo(key: "conv1") var conv1: LTXAudioCausalConv2d
    @ModuleInfo(key: "norm2") var norm2: LTXAudioPixelNorm
    @ModuleInfo(key: "conv2") var conv2: LTXAudioCausalConv2d
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: LTXAudioCausalConv2d?

    init(inChannels: Int, outChannels: Int) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self._norm1.wrappedValue = LTXAudioPixelNorm()
        self._conv1.wrappedValue = LTXAudioCausalConv2d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        self._norm2.wrappedValue = LTXAudioPixelNorm()
        self._conv2.wrappedValue = LTXAudioCausalConv2d(
            inChannels: outChannels,
            outChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        if inChannels != outChannels {
            self._ninShortcut.wrappedValue = LTXAudioCausalConv2d(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: 1,
                stride: 1,
                causalityAxis: .height
            )
        } else {
            self._ninShortcut.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        let residual = ninShortcut?(x) ?? x
        return residual + h
    }
}

private final class LTXAudioUpsample2d: Module {
    @ModuleInfo(key: "conv") var conv: LTXAudioCausalConv2d

    init(channels: Int) {
        self._conv.wrappedValue = LTXAudioCausalConv2d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.repeated(x, count: 2, axis: 1)
        y = MLX.repeated(y, count: 2, axis: 2)
        y = conv(y)
        y = y[0..., 1..., 0..., 0...]
        return y
    }
}

private final class LTXAudioDecoderStage: Module {
    @ModuleInfo(key: "block") var blocks: [LTXAudioResnetBlock2D]
    @ModuleInfo(key: "upsample") var upsample: LTXAudioUpsample2d?

    init(blocks: [LTXAudioResnetBlock2D], upsampleChannels: Int?) {
        self._blocks.wrappedValue = blocks
        if let upsampleChannels {
            self._upsample.wrappedValue = LTXAudioUpsample2d(channels: upsampleChannels)
        } else {
            self._upsample.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        if let upsample {
            h = upsample(h)
        }
        return h
    }
}

private final class LTXAudioMidBlock: Module {
    @ModuleInfo(key: "block_1") var block1: LTXAudioResnetBlock2D
    @ModuleInfo(key: "block_2") var block2: LTXAudioResnetBlock2D

    init(channels: Int) {
        self._block1.wrappedValue = LTXAudioResnetBlock2D(inChannels: channels, outChannels: channels)
        self._block2.wrappedValue = LTXAudioResnetBlock2D(inChannels: channels, outChannels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        block2(block1(x))
    }
}

private final class LTXAudioPerChannelStatistics: Module {
    @ParameterInfo(key: "_std_of_means") var stdOfMeans: MLXArray
    @ParameterInfo(key: "_mean_of_means") var meanOfMeans: MLXArray

    init(latentChannels: Int = 128) {
        self._stdOfMeans.wrappedValue = MLX.ones([latentChannels], dtype: .float32)
        self._meanOfMeans.wrappedValue = MLX.zeros([latentChannels], dtype: .float32)
    }

    func unNormalize(_ x: MLXArray) -> MLXArray {
        let std = stdOfMeans.asType(x.dtype)
        let mean = meanOfMeans.asType(x.dtype)
        return x * std + mean
    }
}

private struct LTXAudioPatchifier {
    func patchify(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let f = x.dim(2)
        let c = x.dim(3)
        return x.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
    }

    func unpatchify(_ x: MLXArray, channels: Int, melBins: Int) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        return x.reshaped(b, t, channels, melBins).transposed(0, 1, 3, 2)
    }
}

private final class LTXAudioDecoder: Module {
    @ModuleInfo(key: "per_channel_statistics") var perChannelStatistics: LTXAudioPerChannelStatistics
    @ModuleInfo(key: "conv_in") var convIn: LTXAudioCausalConv2d
    @ModuleInfo(key: "mid") var mid: LTXAudioMidBlock
    @ModuleInfo(key: "up") var up: [LTXAudioDecoderStage]
    @ModuleInfo(key: "norm_out") var normOut: LTXAudioPixelNorm
    @ModuleInfo(key: "conv_out") var convOut: LTXAudioCausalConv2d

    let latentChannels = LTXAudioLatentChannels
    let outputChannels = 2
    let outputMelBins = 64
    let patchifier = LTXAudioPatchifier()

    override init() {
        self._perChannelStatistics.wrappedValue = LTXAudioPerChannelStatistics(latentChannels: 128)
        self._convIn.wrappedValue = LTXAudioCausalConv2d(
            inChannels: LTXAudioLatentChannels,
            outChannels: 512,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        self._mid.wrappedValue = LTXAudioMidBlock(channels: 512)
        self._up.wrappedValue = [
            LTXAudioDecoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 128),
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                ],
                upsampleChannels: nil
            ),
            LTXAudioDecoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 256),
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 256),
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 256),
                ],
                upsampleChannels: 256
            ),
            LTXAudioDecoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                ],
                upsampleChannels: 512
            ),
        ]
        self._normOut.wrappedValue = LTXAudioPixelNorm()
        self._convOut.wrappedValue = LTXAudioCausalConv2d(
            inChannels: 128,
            outChannels: 2,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        super.init()
    }

    func decode(latents: MLXArray) -> MLXArray {
        var sample = latents
        if sample.ndim == 4, sample.dim(1) == latentChannels {
            sample = sample.transposed(0, 2, 3, 1)
        }

        let originalFrames = sample.dim(1)
        let originalMelBins = sample.dim(2)
        let originalChannels = sample.dim(3)

        var patched = patchifier.patchify(sample)
        patched = perChannelStatistics.unNormalize(patched)
        sample = patchifier.unpatchify(patched, channels: originalChannels, melBins: originalMelBins)
        saveLTXAVDebugArray(sample, suffix: "audio_decoder_denormalized")

        var targetFrames = originalFrames * LTXAudioLatentDownsampleFactor
        targetFrames = max(1, targetFrames - (LTXAudioLatentDownsampleFactor - 1))
        let targetMelBins = outputMelBins

        var h = convIn(sample)
        saveLTXAVDebugArray(h, suffix: "audio_decoder_conv_in")
        h = mid(h)
        saveLTXAVDebugArray(h, suffix: "audio_decoder_mid")

        for level in stride(from: up.count - 1, through: 0, by: -1) {
            h = up[level](h)
            saveLTXAVDebugArray(h, suffix: "audio_decoder_up_\(level)")
        }

        h = normOut(h)
        h = silu(h)
        h = convOut(h)
        saveLTXAVDebugArray(h, suffix: "audio_decoder_conv_out")

        let croppedFrames = min(h.dim(1), targetFrames)
        let croppedMels = min(h.dim(2), targetMelBins)
        var output = h[0..., 0..<croppedFrames, 0..<croppedMels, 0..<outputChannels]

        let framePad = max(0, targetFrames - output.dim(1))
        let melPad = max(0, targetMelBins - output.dim(2))
        if framePad > 0 || melPad > 0 {
            output = padded(
                output,
                widths: [
                    [0, 0],
                    [0, framePad],
                    [0, melPad],
                    [0, 0],
                ]
            )
        }

        output = output[0..., 0..<targetFrames, 0..<targetMelBins, 0..<outputChannels]
        return output.transposed(0, 3, 1, 2)
    }
}

func mapAudioVaeDecoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("audio_vae.decoder.") {
        mapped = String(key.dropFirst("audio_vae.decoder.".count))
    } else if key == "audio_vae.per_channel_statistics.mean-of-means"
        || key == "audio_vae.per_channel_statistics._mean_of_means" {
        mapped = "per_channel_statistics._mean_of_means"
    } else if key == "audio_vae.per_channel_statistics.std-of-means"
        || key == "audio_vae.per_channel_statistics._std_of_means" {
        mapped = "per_channel_statistics._std_of_means"
    } else {
        return []
    }

    var casted = value
    if sourceLayout == .pytorch,
       mapped.lowercased().contains("conv"),
       mapped.hasSuffix("weight"),
       casted.ndim == 4 {
        casted = casted.transposed(0, 2, 3, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

private func ltxLeakyRelu(_ x: MLXArray, slope: Float) -> MLXArray {
    MLX.maximum(x, x * MLXArray(slope).asType(x.dtype))
}

enum LTXVocoderFlavor: Equatable {
    case legacy
    case bandwidthExtension
}

func detectLTXVocoderFlavor<S: Sequence>(keys: S) -> LTXVocoderFlavor where S.Element == String {
    for key in keys {
        if key.hasPrefix("vocoder.bwe_generator.")
            || key.hasPrefix("vocoder.mel_stft.")
            || key.hasPrefix("vocoder.vocoder.") {
            return .bandwidthExtension
        }
    }
    return .legacy
}

enum LTXVocoderWeightLayout: Equatable {
    case pytorch
    case mlx
}

enum LTXVocoderResBlockKind: Equatable {
    case legacy
    case amp
}

enum LTXVocoderActivationKind: Equatable {
    case leaky
    case snake
    case snakeBeta
}

struct LTXVocoderArchitectureConfig: Equatable {
    let inputChannels: Int
    let outputChannels: Int
    let upsampleInitialChannels: Int
    let upsampleRates: [Int]
    let upsampleKernelSizes: [Int]
    let resblockKernelSizes: [Int]
    let resblockDilationSizes: [[Int]]
    let blockKind: LTXVocoderResBlockKind
    let activation: LTXVocoderActivationKind
    let applyFinalActivation: Bool
    let useTanhAtFinal: Bool
    let useBiasAtFinal: Bool

    static let legacy = LTXVocoderArchitectureConfig(
        inputChannels: 128,
        outputChannels: 2,
        upsampleInitialChannels: 1024,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [16, 15, 8, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        blockKind: .legacy,
        activation: .leaky,
        applyFinalActivation: true,
        useTanhAtFinal: true,
        useBiasAtFinal: true
    )

    static let defaultBWEBase = LTXVocoderArchitectureConfig(
        inputChannels: 128,
        outputChannels: 2,
        upsampleInitialChannels: 1024,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [16, 15, 8, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        blockKind: .amp,
        activation: .snakeBeta,
        applyFinalActivation: true,
        useTanhAtFinal: true,
        useBiasAtFinal: true
    )

    static let defaultBWEGenerator = LTXVocoderArchitectureConfig(
        inputChannels: 128,
        outputChannels: 2,
        upsampleInitialChannels: 1024,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [16, 15, 8, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        blockKind: .amp,
        activation: .snakeBeta,
        applyFinalActivation: false,
        useTanhAtFinal: true,
        useBiasAtFinal: true
    )
}

struct LTXBWEVocoderRuntimeConfig {
    let inputSamplingRate: Int
    let outputSamplingRate: Int
    let hopLength: Int
    let filterLength: Int
    let melChannels: Int
    let baseVocoder: LTXVocoderArchitectureConfig
    let bandwidthExtensionVocoder: LTXVocoderArchitectureConfig
}

private struct LTXVocoderModelConfig: Decodable {
    let upsampleInitialChannels: Int?
    let resblock: String?
    let upsampleRates: [Int]?
    let resblockKernelSizes: [Int]?
    let upsampleKernelSizes: [Int]?
    let resblockDilationSizes: [[Int]]?
    let useTanhAtFinal: Bool?
    let activation: String?
    let useBiasAtFinal: Bool?
    let applyFinalActivation: Bool?

    private enum CodingKeys: String, CodingKey {
        case upsampleInitialChannels = "upsample_initial_channel"
        case resblock
        case upsampleRates = "upsample_rates"
        case resblockKernelSizes = "resblock_kernel_sizes"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case useTanhAtFinal = "use_tanh_at_final"
        case activation
        case useBiasAtFinal = "use_bias_at_final"
        case applyFinalActivation = "apply_final_activation"
    }

    func runtimeArchitecture(
        defaultArchitecture: LTXVocoderArchitectureConfig
    ) -> LTXVocoderArchitectureConfig {
        let blockKind: LTXVocoderResBlockKind = switch resblock?.lowercased() {
        case "amp1", "amp":
            .amp
        default:
            defaultArchitecture.blockKind
        }
        let activationKind: LTXVocoderActivationKind = switch activation?.lowercased() {
        case "snake":
            .snake
        case "snakebeta", "snake_beta":
            .snakeBeta
        default:
            defaultArchitecture.activation
        }
        return LTXVocoderArchitectureConfig(
            inputChannels: defaultArchitecture.inputChannels,
            outputChannels: defaultArchitecture.outputChannels,
            upsampleInitialChannels: upsampleInitialChannels ?? defaultArchitecture.upsampleInitialChannels,
            upsampleRates: upsampleRates ?? defaultArchitecture.upsampleRates,
            upsampleKernelSizes: upsampleKernelSizes ?? defaultArchitecture.upsampleKernelSizes,
            resblockKernelSizes: resblockKernelSizes ?? defaultArchitecture.resblockKernelSizes,
            resblockDilationSizes: resblockDilationSizes ?? defaultArchitecture.resblockDilationSizes,
            blockKind: blockKind,
            activation: activationKind,
            applyFinalActivation: applyFinalActivation ?? defaultArchitecture.applyFinalActivation,
            useTanhAtFinal: useTanhAtFinal ?? defaultArchitecture.useTanhAtFinal,
            useBiasAtFinal: useBiasAtFinal ?? defaultArchitecture.useBiasAtFinal
        )
    }

    var hasArchitectureFields: Bool {
        upsampleInitialChannels != nil
            || resblock != nil
            || upsampleRates != nil
            || resblockKernelSizes != nil
            || upsampleKernelSizes != nil
            || resblockDilationSizes != nil
            || useTanhAtFinal != nil
            || activation != nil
            || useBiasAtFinal != nil
            || applyFinalActivation != nil
    }
}

private struct LTXBWEVocoderConfig: Decodable {
    let inputSamplingRate: Int
    let outputSamplingRate: Int
    let hopLength: Int
    let filterLength: Int
    let winLength: Int?
    let melChannels: Int
    let modelConfig: LTXVocoderModelConfig

    private enum CodingKeys: String, CodingKey {
        case inputSamplingRate = "input_sampling_rate"
        case outputSamplingRate = "output_sampling_rate"
        case hopLength = "hop_length"
        case filterLength = "n_fft"
        case winLength = "win_size"
        case melChannels = "num_mels"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        inputSamplingRate = try container.decode(Int.self, forKey: .inputSamplingRate)
        outputSamplingRate = try container.decode(Int.self, forKey: .outputSamplingRate)
        hopLength = try container.decode(Int.self, forKey: .hopLength)
        filterLength = try container.decode(Int.self, forKey: .filterLength)
        winLength = try container.decodeIfPresent(Int.self, forKey: .winLength)
        melChannels = try container.decode(Int.self, forKey: .melChannels)
        modelConfig = try LTXVocoderModelConfig(from: decoder)
    }

    func runtimeConfig(baseVocoder: LTXVocoderArchitectureConfig?) -> LTXBWEVocoderRuntimeConfig {
        LTXBWEVocoderRuntimeConfig(
            inputSamplingRate: inputSamplingRate,
            outputSamplingRate: outputSamplingRate,
            hopLength: hopLength,
            filterLength: winLength ?? filterLength,
            melChannels: melChannels,
            baseVocoder: baseVocoder ?? .defaultBWEBase,
            bandwidthExtensionVocoder: modelConfig.runtimeArchitecture(defaultArchitecture: .defaultBWEGenerator)
        )
    }
}

private struct LTXVocoderConfigEnvelope: Decodable {
    let baseVocoder: LTXVocoderModelConfig?
    let bwe: LTXBWEVocoderConfig?

    private enum CodingKeys: String, CodingKey {
        case vocoder
        case bwe
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let topLevelBWE = try container.decodeIfPresent(LTXBWEVocoderConfig.self, forKey: .bwe)

        var resolvedBase: LTXVocoderModelConfig?
        var resolvedBWE = topLevelBWE

        if let directBase = try? container.decodeIfPresent(LTXVocoderModelConfig.self, forKey: .vocoder),
           directBase.hasArchitectureFields {
            resolvedBase = directBase
        }

        if let nested = try? container.decodeIfPresent(LTXVocoderConfigNode.self, forKey: .vocoder) {
            if let nestedBase = nested.vocoder, nestedBase.hasArchitectureFields {
                resolvedBase = nestedBase
            }
            if resolvedBWE == nil {
                resolvedBWE = nested.bwe
            }
        }

        self.baseVocoder = resolvedBase
        self.bwe = resolvedBWE
    }
}

private struct LTXVocoderConfigNode: Decodable {
    let vocoder: LTXVocoderModelConfig?
    let bwe: LTXBWEVocoderConfig?
}

func loadLTXBWEVocoderConfig(modelRoot: URL) throws -> LTXBWEVocoderRuntimeConfig? {
    let candidates = [
        modelRoot.appendingPathComponent("vocoder/config.json", isDirectory: false),
        modelRoot.appendingPathComponent("embedded_config.json", isDirectory: false),
        modelRoot.appendingPathComponent("config.json", isDirectory: false),
        modelRoot.appendingPathComponent("audio_vae/config.json", isDirectory: false),
    ]
    let decoder = JSONDecoder()

    for url in candidates where FileManager.default.fileExists(atPath: url.path) {
        let data = try Data(contentsOf: url)
        if let direct = try? decoder.decode(LTXBWEVocoderConfig.self, from: data) {
            return direct.runtimeConfig(baseVocoder: nil)
        }
        if let envelope = try? decoder.decode(LTXVocoderConfigEnvelope.self, from: data) {
            if let bwe = envelope.bwe {
                let base = envelope.baseVocoder?.runtimeArchitecture(defaultArchitecture: .defaultBWEBase)
                return bwe.runtimeConfig(baseVocoder: base)
            }
        }
    }

    return nil
}

private class LTXAudioVocoderBase: Module {
    let outputSamplingRate: Int

    init(outputSamplingRate: Int) {
        self.outputSamplingRate = outputSamplingRate
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        preconditionFailure("LTXAudioVocoderBase must be subclassed.")
    }
}

private class LTXVocoderResidualBlock: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        preconditionFailure("LTXVocoderResidualBlock must be subclassed.")
    }
}

private final class LTXVocoderSnake: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray
    @ModuleInfo(key: "beta") var beta: MLXArray?

    init(channels: Int, hasBeta: Bool) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = hasBeta ? MLXArray.zeros([channels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaValue = MLX.exp(alpha.asType(x.dtype)).reshaped(1, 1, -1)
        let betaValue = (beta.map { MLX.exp($0.asType(x.dtype)).reshaped(1, 1, -1) } ?? alphaValue)
        let sine = MLX.sin(x * alphaValue)
        return x + (sine * sine) / (betaValue + MLXArray(1e-9).asType(x.dtype))
    }
}

private func ltxBesselI0(_ value: Double) -> Double {
    var sum = 1.0
    var term = 1.0
    let scaled = (value * value) / 4.0
    for index in 1...24 {
        term *= scaled / Double(index * index)
        sum += term
        if term < 1e-12 {
            break
        }
    }
    return sum
}

private func ltxSinc(_ value: Float) -> Float {
    if abs(value) < 1e-8 {
        return 1.0
    }
    return sin(Float.pi * value) / (Float.pi * value)
}

private func ltxKaiserWindow(kernelSize: Int, beta: Float) -> [Float] {
    guard kernelSize > 1 else { return [1.0] }
    let denominator = ltxBesselI0(Double(beta))
    return (0..<kernelSize).map { index in
        let ratio = (2.0 * Double(index)) / Double(kernelSize - 1) - 1.0
        let value = Double(beta) * sqrt(max(0.0, 1.0 - ratio * ratio))
        return Float(ltxBesselI0(value) / denominator)
    }
}

private func ltxKaiserSincFilter1d(cutoff: Float, halfWidth: Float, kernelSize: Int) -> [Float] {
    let even = kernelSize.isMultiple(of: 2)
    let halfSize = kernelSize / 2
    let deltaF = 4.0 * halfWidth
    let amplitude = 2.285 * Float(halfSize - 1) * Float.pi * deltaF + 7.95
    let beta: Float
    if amplitude > 50.0 {
        beta = 0.1102 * (amplitude - 8.7)
    } else if amplitude >= 21.0 {
        beta = 0.5842 * pow(amplitude - 21.0, 0.4) + 0.07886 * (amplitude - 21.0)
    } else {
        beta = 0.0
    }

    let window = ltxKaiserWindow(kernelSize: kernelSize, beta: beta)
    guard cutoff != 0 else {
        return [Float](repeating: 0, count: kernelSize)
    }

    var filter = [Float](repeating: 0, count: kernelSize)
    for index in 0..<kernelSize {
        let time = even ? Float(index - halfSize) + 0.5 : Float(index - halfSize)
        filter[index] = 2.0 * cutoff * window[index] * ltxSinc(2.0 * cutoff * time)
    }

    let total = filter.reduce(0, +)
    if total != 0 {
        filter = filter.map { $0 / total }
    }
    return filter
}

private func ltxHannSincFilter1d(ratio: Int) -> [Float] {
    let rolloff: Float = 0.99
    let lowpassFilterWidth: Float = 6.0
    let width = Int(ceil(lowpassFilterWidth / rolloff))
    let kernelSize = 2 * width * ratio + 1
    return (0..<kernelSize).map { index in
        let timeAxis = (Float(index) / Float(ratio) - Float(width)) * rolloff
        let clamped = min(lowpassFilterWidth, max(-lowpassFilterWidth, timeAxis))
        let window = pow(cos(clamped * Float.pi / lowpassFilterWidth / 2.0), 2.0)
        return ltxSinc(timeAxis) * window * rolloff / Float(ratio)
    }
}

private final class LTXLowPassFilter1d: Module {
    @ModuleInfo(key: "filter") var filter: MLXArray
    let kernelSize: Int
    let padLeft: Int
    let padRight: Int
    let stride: Int

    init(ratio: Int) {
        self.kernelSize = Int(6 * ratio / 2) * 2
        self.padLeft = kernelSize / 2 - (kernelSize.isMultiple(of: 2) ? 1 : 0)
        self.padRight = kernelSize / 2
        self.stride = ratio
        let values = ltxKaiserSincFilter1d(
            cutoff: 0.5 / Float(ratio),
            halfWidth: 0.6 / Float(ratio),
            kernelSize: kernelSize
        )
        self._filter.wrappedValue = MLXArray(values).reshaped(1, kernelSize, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(2)
        let paddedInput = padded(
            x,
            widths: [[0, 0], [padLeft, padRight], [0, 0]],
            mode: .edge
        )
        let weights = broadcast(filter.asType(x.dtype), to: [channels, kernelSize, 1])
        return MLX.conv1d(paddedInput, weights, stride: stride, padding: 0, groups: channels)
    }
}

final class LTXSincUpsample1d: Module {
    @ModuleInfo(key: "filter") var filter: MLXArray
    let ratio: Int
    let kernelSize: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int

    init(ratio: Int, windowType: String = "kaiser") {
        self.ratio = ratio
        let values: [Float]
        if windowType == "hann" {
            values = ltxHannSincFilter1d(ratio: ratio)
            self.kernelSize = values.count
            let width = Int(ceil(6.0 / 0.99))
            self.pad = width
            self.padLeft = 2 * width * ratio
            self.padRight = kernelSize - ratio
        } else {
            self.kernelSize = Int(6 * ratio / 2) * 2
            self.pad = kernelSize / ratio - 1
            self.padLeft = pad * ratio + (kernelSize - ratio) / 2
            self.padRight = pad * ratio + (kernelSize - ratio + 1) / 2
            values = ltxKaiserSincFilter1d(
                cutoff: 0.5 / Float(ratio),
                halfWidth: 0.6 / Float(ratio),
                kernelSize: kernelSize
            )
        }
        self._filter.wrappedValue = MLXArray(values).reshaped(1, kernelSize, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(2)
        let paddedInput = padded(
            x,
            widths: [[0, 0], [pad, pad], [0, 0]],
            mode: .edge
        )
        let paddedLength = paddedInput.dim(1)
        let upsampledLength = (paddedLength - 1) * ratio + 1
        let zeroTail = MLX.zeros([paddedInput.dim(0), paddedLength, max(0, ratio - 1), channels], dtype: x.dtype)
        let expanded = MLX.concatenated([paddedInput.expandedDimensions(axis: 2), zeroTail], axis: 2)
            .reshaped(paddedInput.dim(0), paddedLength * ratio, channels)
        let upsampled = expanded[0..., 0..<upsampledLength, 0...]

        let convInput = padded(
            upsampled,
            widths: [[0, 0], [kernelSize - 1, kernelSize - 1], [0, 0]]
        )
        let weights = broadcast(filter.asType(x.dtype), to: [channels, kernelSize, 1])
        let filtered = MLX.conv1d(
            convInput,
            weights,
            stride: 1,
            padding: 0,
            groups: channels
        ) * MLXArray(Float(ratio)).asType(x.dtype)
        let end = max(padLeft, filtered.dim(1) - padRight)
        return filtered[0..., padLeft..<end, 0...]
    }
}

private final class LTXVocoderActivation1d: Module {
    @ModuleInfo(key: "act") var act: LTXVocoderSnake?
    @ModuleInfo(key: "upsample") var upsample: LTXSincUpsample1d
    @ModuleInfo(key: "downsample") var downsample: LTXLowPassFilter1d
    let kind: LTXVocoderActivationKind

    init(channels: Int, kind: LTXVocoderActivationKind) {
        self.kind = kind
        switch kind {
        case .leaky:
            self._act.wrappedValue = nil
        case .snake:
            self._act.wrappedValue = LTXVocoderSnake(channels: channels, hasBeta: false)
        case .snakeBeta:
            self._act.wrappedValue = LTXVocoderSnake(channels: channels, hasBeta: true)
        }
        self._upsample.wrappedValue = LTXSincUpsample1d(ratio: 2)
        self._downsample.wrappedValue = LTXLowPassFilter1d(ratio: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch kind {
        case .leaky:
            return ltxLeakyRelu(x, slope: 0.1)
        case .snake, .snakeBeta:
            guard let act else { return x }
            var h = upsample(x)
            h = act(h)
            return downsample(h)
        }
    }
}

private final class LTXVocoderResBlock1: LTXVocoderResidualBlock {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        self._convs1.wrappedValue = dilations.map { dilation in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: ((kernelSize - 1) * dilation) / 2,
                dilation: dilation,
                groups: 1,
                bias: true
            )
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for idx in 0..<convs1.count {
            var h = ltxLeakyRelu(out, slope: 0.1)
            h = convs1[idx](h)
            h = ltxLeakyRelu(h, slope: 0.1)
            h = convs2[idx](h)
            out = out + h
        }
        return out
    }
}

private final class LTXVocoderAMPBlock1: LTXVocoderResidualBlock {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "acts1") var acts1: [LTXVocoderActivation1d]
    @ModuleInfo(key: "acts2") var acts2: [LTXVocoderActivation1d]

    init(channels: Int, kernelSize: Int, dilations: [Int], activation: LTXVocoderActivationKind) {
        self._convs1.wrappedValue = dilations.map { dilation in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: ((kernelSize - 1) * dilation) / 2,
                dilation: dilation,
                groups: 1,
                bias: true
            )
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }
        self._acts1.wrappedValue = dilations.map { _ in
            LTXVocoderActivation1d(channels: channels, kind: activation)
        }
        self._acts2.wrappedValue = dilations.map { _ in
            LTXVocoderActivation1d(channels: channels, kind: activation)
        }
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for idx in 0..<convs1.count {
            var h = acts1[idx](out)
            h = convs1[idx](h)
            h = acts2[idx](h)
            h = convs2[idx](h)
            out = out + h
        }
        return out
    }
}

private final class LTXVocoder: LTXAudioVocoderBase {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resBlocks: [LTXVocoderResidualBlock]
    @ModuleInfo(key: "act_post") var actPost: LTXVocoderActivation1d?
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    let numKernels: Int
    let blockKind: LTXVocoderResBlockKind
    let applyFinalActivation: Bool
    let useTanhAtFinal: Bool

    init(
        blockKind: LTXVocoderResBlockKind = .legacy,
        outputSamplingRate: Int = LTXAudioSampleRate,
        activation: LTXVocoderActivationKind = .leaky,
        applyFinalActivation: Bool = true,
        useTanhAtFinal: Bool = true,
        useBiasAtFinal: Bool = true,
        inputChannels: Int = 128,
        outputChannels: Int = 2,
        upsampleInitialChannels: Int = 1024,
        upsampleRates: [Int] = [6, 5, 2, 2, 2],
        upsampleKernelSizes: [Int] = [16, 15, 8, 4, 4],
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ) {
        self.blockKind = blockKind
        self.applyFinalActivation = applyFinalActivation
        self.useTanhAtFinal = useTanhAtFinal
        self.numKernels = resblockKernelSizes.count

        self._convPre.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: upsampleInitialChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: true
        )

        let upLayers = upsampleRates.enumerated().map { idx, stride in
            let kernel = upsampleKernelSizes[idx]
            let inCh = upsampleInitialChannels / (1 << idx)
            let outCh = upsampleInitialChannels / (1 << (idx + 1))
            return ConvTransposed1d(
                inputChannels: inCh,
                outputChannels: outCh,
                kernelSize: kernel,
                stride: stride,
                padding: (kernel - stride) / 2,
                dilation: 1,
                groups: 1,
                bias: true
            )
        }
        self._ups.wrappedValue = upLayers

        var blocks: [LTXVocoderResidualBlock] = []
        for i in 0..<upLayers.count {
            let channels = upsampleInitialChannels / (1 << (i + 1))
            for (kernelIndex, kernel) in resblockKernelSizes.enumerated() {
                let dilations = kernelIndex < resblockDilationSizes.count
                    ? resblockDilationSizes[kernelIndex]
                    : [1, 3, 5]
                switch blockKind {
                case .legacy:
                    blocks.append(LTXVocoderResBlock1(channels: channels, kernelSize: kernel, dilations: dilations))
                case .amp:
                    blocks.append(LTXVocoderAMPBlock1(
                        channels: channels,
                        kernelSize: kernel,
                        dilations: dilations,
                        activation: activation
                    ))
                }
            }
        }
        self._resBlocks.wrappedValue = blocks

        let finalChannels = upsampleInitialChannels / (1 << upLayers.count)
        self._actPost.wrappedValue = blockKind == .amp
            ? LTXVocoderActivation1d(channels: finalChannels, kind: activation)
            : nil
        self._convPost.wrappedValue = Conv1d(
            inputChannels: finalChannels,
            outputChannels: outputChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            groups: 1,
            bias: useBiasAtFinal
        )
        super.init(outputSamplingRate: outputSamplingRate)
    }

    convenience init(
        architecture: LTXVocoderArchitectureConfig,
        outputSamplingRate: Int
    ) {
        self.init(
            blockKind: architecture.blockKind,
            outputSamplingRate: outputSamplingRate,
            activation: architecture.activation,
            applyFinalActivation: architecture.applyFinalActivation,
            useTanhAtFinal: architecture.useTanhAtFinal,
            useBiasAtFinal: architecture.useBiasAtFinal,
            inputChannels: architecture.inputChannels,
            outputChannels: architecture.outputChannels,
            upsampleInitialChannels: architecture.upsampleInitialChannels,
            upsampleRates: architecture.upsampleRates,
            upsampleKernelSizes: architecture.upsampleKernelSizes,
            resblockKernelSizes: architecture.resblockKernelSizes,
            resblockDilationSizes: architecture.resblockDilationSizes
        )
    }

    override func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed(0, 1, 3, 2) // [B, 2, 64, T]
        let b = x.dim(0)
        let stereo = x.dim(1)
        let melBins = x.dim(2)
        let time = x.dim(3)
        x = x.reshaped(b, stereo * melBins, time).transposed(0, 2, 1) // [B, T, 128]

        x = convPre(x)

        for i in 0..<ups.count {
            if blockKind == .legacy {
                x = ltxLeakyRelu(x, slope: 0.1)
            }
            x = ups[i](x)
            var blockOutputs: [MLXArray] = []
            blockOutputs.reserveCapacity(numKernels)
            let start = i * numKernels
            for j in 0..<numKernels {
                blockOutputs.append(resBlocks[start + j](x))
            }
            x = MLX.mean(MLX.stacked(blockOutputs, axis: 0), axis: 0)
        }

        if let actPost {
            x = actPost(x)
        } else {
            x = ltxLeakyRelu(x, slope: 0.01)
        }
        x = convPost(x)
        if applyFinalActivation {
            x = useTanhAtFinal
                ? tanh(x)
                : MLX.clip(x, min: MLXArray(-1.0).asType(x.dtype), max: MLXArray(1.0).asType(x.dtype))
        }
        return x.transposed(0, 2, 1) // [B, 2, samples]
    }
}

private final class LTXSTFTFn: Module {
    @ModuleInfo(key: "forward_basis") var forwardBasis: MLXArray
    @ModuleInfo(key: "inverse_basis") var inverseBasis: MLXArray
    let hopLength: Int
    let winLength: Int

    init(filterLength: Int, hopLength: Int, winLength: Int) {
        let frequencies = filterLength / 2 + 1
        self.hopLength = hopLength
        self.winLength = winLength
        self._forwardBasis.wrappedValue = MLXArray.zeros([frequencies * 2, filterLength, 1])
        self._inverseBasis.wrappedValue = MLXArray.zeros([frequencies * 2, filterLength, 1])
    }

    func magnitude(_ y: MLXArray) -> MLXArray {
        var input = y
        if input.ndim == 2 {
            input = input.expandedDimensions(axis: 2)
        }
        let leftPad = max(0, winLength - hopLength)
        input = padded(input, widths: [[0, 0], [leftPad, 0], [0, 0]])
        let spec = MLX.conv1d(input, forwardBasis.asType(input.dtype), stride: hopLength, padding: 0)
        let frequencies = spec.dim(2) / 2
        let real = spec[0..., 0..., 0..<frequencies]
        let imag = spec[0..., 0..., frequencies..<(frequencies * 2)]
        return MLX.sqrt(real * real + imag * imag)
    }
}

private final class LTXMelSTFT: Module {
    @ModuleInfo(key: "stft_fn") var stftFn: LTXSTFTFn
    @ModuleInfo(key: "mel_basis") var melBasis: MLXArray

    init(filterLength: Int = 1024, hopLength: Int = 60, winLength: Int = 1024, melChannels: Int = 128) {
        self._stftFn.wrappedValue = LTXSTFTFn(
            filterLength: filterLength,
            hopLength: hopLength,
            winLength: winLength
        )
        self._melBasis.wrappedValue = MLXArray.zeros([melChannels, filterLength / 2 + 1])
    }

    func melSpectrogram(_ y: MLXArray) -> MLXArray {
        let magnitude = stftFn.magnitude(y)
        let basis = melBasis.asType(magnitude.dtype)
        let mel = MLX.matmul(magnitude, basis.transposed())
        return MLX.log(MLX.maximum(mel, MLXArray(1e-5).asType(mel.dtype)))
    }
}

private final class LTXVocoderWithBWE: LTXAudioVocoderBase {
    @ModuleInfo(key: "vocoder") var vocoder: LTXVocoder
    @ModuleInfo(key: "bwe_generator") var bweGenerator: LTXVocoder
    @ModuleInfo(key: "mel_stft") var melSTFT: LTXMelSTFT
    let inputSamplingRate: Int
    let hopLength: Int
    let resampler: LTXSincUpsample1d

    convenience init(config: LTXBWEVocoderRuntimeConfig) {
        self.init(
            inputSamplingRate: config.inputSamplingRate,
            outputSamplingRate: config.outputSamplingRate,
            hopLength: config.hopLength,
            filterLength: config.filterLength,
            melChannels: config.melChannels,
            baseVocoder: config.baseVocoder,
            bandwidthExtensionVocoder: config.bandwidthExtensionVocoder
        )
    }

    init(
        inputSamplingRate: Int = LTXAudioSampleRate,
        outputSamplingRate: Int = 48_000,
        hopLength: Int = 60,
        filterLength: Int = 1024,
        melChannels: Int = 128,
        baseVocoder: LTXVocoderArchitectureConfig = .defaultBWEBase,
        bandwidthExtensionVocoder: LTXVocoderArchitectureConfig = .defaultBWEGenerator
    ) {
        self.inputSamplingRate = inputSamplingRate
        self.hopLength = hopLength
        self._vocoder.wrappedValue = LTXVocoder(architecture: baseVocoder, outputSamplingRate: inputSamplingRate)
        self._bweGenerator.wrappedValue = LTXVocoder(
            architecture: bandwidthExtensionVocoder,
            outputSamplingRate: outputSamplingRate
        )
        self._melSTFT.wrappedValue = LTXMelSTFT(
            filterLength: filterLength,
            hopLength: hopLength,
            winLength: filterLength,
            melChannels: melChannels
        )
        self.resampler = LTXSincUpsample1d(
            ratio: max(1, outputSamplingRate / inputSamplingRate),
            windowType: "hann"
        )
        super.init(outputSamplingRate: outputSamplingRate)
    }

    override func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let inputDType = mel.dtype
        var lowRate = vocoder(mel.asType(.float32))
        let lowRateLength = lowRate.dim(2)
        let outputLength = lowRateLength * outputSamplingRate / inputSamplingRate
        saveLTXAVDebugAudio(lowRate, suffix: "bwe_low_rate", sampleRate: inputSamplingRate)

        let remainder = lowRateLength % hopLength
        if remainder != 0 {
            lowRate = padded(lowRate, widths: [[0, 0], [0, 0], [0, hopLength - remainder]])
        }

        let batch = lowRate.dim(0)
        let channels = lowRate.dim(1)
        let flattened = lowRate.reshaped(batch * channels, lowRate.dim(2))
        let computedMel = melSTFT.melSpectrogram(flattened)
            .reshaped(batch, channels, -1, melSTFT.melBasis.dim(0))
        saveLTXAVDebugArray(computedMel, suffix: "bwe_computed_mel")
        let residual = bweGenerator(computedMel)
        saveLTXAVDebugAudio(residual, suffix: "bwe_residual", sampleRate: outputSamplingRate)
        let skip = resampler(lowRate.transposed(0, 2, 1)).transposed(0, 2, 1)
        saveLTXAVDebugAudio(skip, suffix: "bwe_skip", sampleRate: outputSamplingRate)

        let mixedLength = min(residual.dim(2), skip.dim(2))
        var mixed = residual[0..., 0..., 0..<mixedLength] + skip[0..., 0..., 0..<mixedLength]
        mixed = MLX.clip(mixed, min: MLXArray(-1.0).asType(mixed.dtype), max: MLXArray(1.0).asType(mixed.dtype))
        saveLTXAVDebugAudio(mixed, suffix: "bwe_mixed", sampleRate: outputSamplingRate)

        let cropLength = min(outputLength, mixed.dim(2))
        mixed = mixed[0..., 0..., 0..<cropLength]
        if cropLength < outputLength {
            mixed = padded(mixed, widths: [[0, 0], [0, 0], [0, outputLength - cropLength]])
        }
        return mixed.asType(inputDType)
    }
}

func mapVocoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXVocoderWeightLayout = .pytorch,
    targetFlavor: LTXVocoderFlavor = .legacy
) -> [(String, MLXArray)] {
    guard key.hasPrefix("vocoder.") else { return [] }
    var mapped = String(key.dropFirst("vocoder.".count))
    if targetFlavor == .bandwidthExtension {
        if mapped.hasPrefix("vocoder.") {
            mapped = String(mapped.dropFirst("vocoder.".count))
        }
        if !mapped.hasPrefix("bwe_generator.") && !mapped.hasPrefix("mel_stft.") {
            mapped = "vocoder." + mapped
        }
    }
    mapped = mapped.replacingOccurrences(of: ".downsample.lowpass.filter", with: ".downsample.filter")

    var casted = value
    if sourceLayout == .pytorch {
        if mapped.hasSuffix("weight"), casted.ndim == 3 {
            if mapped.contains("ups.") {
                casted = casted.transposed(1, 2, 0)
            } else {
                casted = casted.transposed(0, 2, 1)
            }
        } else if casted.ndim == 3,
                  mapped.hasSuffix("filter")
                    || mapped.hasSuffix("forward_basis")
                    || mapped.hasSuffix("inverse_basis") {
            casted = casted.transposed(0, 2, 1)
        }
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}
