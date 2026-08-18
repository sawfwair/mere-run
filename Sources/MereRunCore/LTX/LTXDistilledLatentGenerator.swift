import AudioCodecs
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
    public let fps: Double
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
        fps: Double = 24,
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
    case ltx25ConditioningRequiresLTX25
    case invalidGeneratedKeyframes([Int])
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
            Detected an LTX 2.3 split MLX model at \(url.path). This legacy loader only supports the older \
            merged LTX layout. Use `mere.run video generate --variant distilled` for video-only output from \
            split models, or `--variant unified-av` for synchronized audio and video.
            """
        case .generatorNotLoaded:
            return "LTX distilled latent generator is not loaded."
        case .invalidResolution(let width, let height):
            return "Resolution does not meet the selected LTX pipeline's 32- or 64-pixel alignment (got \(width)x\(height))."
        case .invalidFrameCount(let value):
            return "numFrames must satisfy 8n+1 and be >= 9 (got \(value))."
        case .invalidImageStrength(let value):
            return "imageStrength must be in [0, 1] (got \(value))."
        case .invalidImageFrameIndex(let value):
            return "imageFrameIndex must be >= 0 (got \(value))."
        case .ltx25ConditioningRequiresLTX25:
            return "Arbitrary timed image conditioning and generated keyframe slots require an LTX 2.5 checkpoint."
        case .invalidGeneratedKeyframes(let values):
            return "Generated keyframe indices must be strictly increasing pixel-frame positions inside the output (got \(values))."
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

public func isLTX23AudioToVideoModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    let root = rootURL.standardizedFileURL
    let required = [
        "split_model.json",
        "config.json",
        "connector.safetensors",
        "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
    ]
    return required.allSatisfy {
        fileManager.fileExists(atPath: root.appendingPathComponent($0).path)
    }
}

public func isLTX23FullModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    isLTX23AudioToVideoModelRoot(rootURL, fileManager: fileManager)
        && fileManager.fileExists(
            atPath: rootURL.standardizedFileURL
                .appendingPathComponent("vocoder.safetensors", isDirectory: false).path
        )
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

struct LTXLatentConditioningState {
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

private struct LTXAttentionProjectedContext {
    let keys: MLXArray
    let values: MLXArray
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
        keyRope: (cos: MLXArray, sin: MLXArray)? = nil,
        projectedContext: LTXAttentionProjectedContext? = nil
    ) -> MLXArray {
        let q = qNorm(toQ(x))
        var qHeads = q.reshaped(q.dim(0), q.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        var kHeads: MLXArray
        let vHeads: MLXArray
        if let projectedContext {
            kHeads = projectedContext.keys
            vHeads = projectedContext.values
        } else {
            let ctx = context ?? x
            let projection = projectContext(ctx, keyRope: keyRope ?? rope)
            kHeads = projection.keys
            vHeads = projection.values
        }

        if let rope {
            qHeads = applySplitRoPEHeads(qHeads, cosFreq: rope.cos, sinFreq: rope.sin)
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

    func projectContext(
        _ context: MLXArray,
        keyRope: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> LTXAttentionProjectedContext {
        let k = kNorm(toK(context))
        let v = toV(context)
        var keys = k.reshaped(k.dim(0), k.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        if let keyRope {
            keys = applySplitRoPEHeads(keys, cosFreq: keyRope.cos, sinFreq: keyRope.sin)
        }
        let values = v.reshaped(v.dim(0), v.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        return LTXAttentionProjectedContext(keys: keys, values: values)
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

private func upsampleLatentsTemporally(
    _ latents: MLXArray,
    upsampler: LTXTemporalLatentUpsampler,
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

func applyLatentConditioning(
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
    dtype: DType,
    hdrColorSpace: LTXHDRColorSpace? = nil,
    crf: Int = 0
) throws -> MLXArray {
    if MediaHDRImageIO.isEXR(url) {
        guard let hdrColorSpace else {
            throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
        }
        do {
            let image = try MediaHDRImageIO.centerCropped(
                MediaHDRImageIO.decodeEXR(url),
                width: width,
                height: height
            )
            return LTXHDRColorPipeline.makeConditioningImage(
                image,
                colorSpace: hdrColorSpace,
                dtype: dtype
            )
        } catch {
            throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
        }
    }
    let image: MediaImage
    do {
        let decoded = try MediaImageIO.decode(url)
        image = try MediaImageIO.h264RoundTrip(decoded, crf: crf)
    } catch {
        throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
    }

    let channels: [Float]
    do {
        channels = try MediaImageIO.bilinearCenterCroppedRGBCHWFloat(
            image,
            width: width,
            height: height,
            normalizedToMinusOneToOne: true
        )
    } catch {
        throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
    }

    let chw = MLXArray(channels).reshaped(1, 3, height, width).asType(dtype)
    return chw.reshaped(1, 3, 1, height, width)
}

private func loadVideoForEncoding(
    url: URL,
    width: Int,
    height: Int,
    frameCap: Int,
    temporalScaleFactor: Int,
    dtype: DType,
    hdrColorSpace: LTXHDRColorSpace? = nil,
    duplicateEachFrame: Bool = false,
    hdrICLoRAReference: Bool = false
) throws -> MLXArray {
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(url)
    }
    if MediaHDRImageIO.isEXRDirectory(url) {
        guard let hdrColorSpace else {
            throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(
                url,
                "EXR input requires an explicit HDR color space"
            )
        }
        do {
            let sourceFrameCap = duplicateEachFrame ? (frameCap + 1) / 2 : frameCap
            let urls = Array(try MediaHDRImageIO.exrFrameURLs(in: url).prefix(sourceFrameCap))
            guard !urls.isEmpty else {
                throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(url, "no EXR frames")
            }
            var indices = [0]
            if urls.count > 1 {
                indices.append(contentsOf: stride(from: 1, to: urls.count, by: temporalScaleFactor))
            }
            var frames = try indices.map { index in
                let image = try MediaHDRImageIO.reflectPadded(
                    MediaHDRImageIO.decodeEXR(urls[index]),
                    width: width,
                    height: height
                )
                return LTXHDRColorPipeline.makeConditioningImage(
                    image,
                    colorSpace: hdrColorSpace,
                    dtype: dtype
                )
            }
            if duplicateEachFrame {
                frames = Array(frames.flatMap { [$0, $0] }.prefix(frameCap))
            }
            return MLX.concatenated(frames, axis: 2)
        } catch let error as LTXUnifiedAVGeneratorError {
            throw error
        } catch {
            throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(url, error.localizedDescription)
        }
    }
    let temporaryDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("mererun-ltx-reference-\(UUID().uuidString)", isDirectory: true)
    do {
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let sourceFrameCap = duplicateEachFrame ? (frameCap + 1) / 2 : frameCap
        let sequence = try MediaVideoIO.extractFrames(
            from: url,
            into: temporaryDirectory,
            endFrame: max(0, sourceFrameCap - 1)
        )
        guard !sequence.frameURLs.isEmpty else {
            throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(url, "no frames")
        }
        var indices = [0]
        if sequence.frameURLs.count > 1 {
            indices.append(contentsOf: stride(
                from: 1,
                to: sequence.frameURLs.count,
                by: temporalScaleFactor
            ))
        }
        var frames = try indices.map { index in
            if hdrICLoRAReference {
                let image = try MediaImageIO.decode(sequence.frameURLs[index])
                var rgb = [Float](repeating: 0, count: image.width * image.height * 3)
                for pixel in 0..<(image.width * image.height) {
                    rgb[pixel * 3] = Float(image.rgba8[pixel * 4]) / 255
                    rgb[pixel * 3 + 1] = Float(image.rgba8[pixel * 4 + 1]) / 255
                    rgb[pixel * 3 + 2] = Float(image.rgba8[pixel * 4 + 2]) / 255
                }
                let padded = try MediaHDRImageIO.reflectPadded(
                    MediaFloatImage(width: image.width, height: image.height, rgb: rgb),
                    width: width,
                    height: height
                )
                return (MLXArray(padded.rgb).reshaped(1, height, width, 3) * MLXArray(Float(2))
                    - MLXArray(Float(1)))
                    .transposed(0, 3, 1, 2)
                    .reshaped(1, 3, 1, height, width)
                    .asType(dtype)
            }
            return try loadImageForEncoding(
                url: sequence.frameURLs[index],
                width: width,
                height: height,
                dtype: dtype
            )
        }
        if duplicateEachFrame {
            frames = Array(frames.flatMap { [$0, $0] }.prefix(frameCap))
        }
        return MLX.concatenated(frames, axis: 2)
    } catch let error as LTXUnifiedAVGeneratorError {
        throw error
    } catch {
        throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(
            url,
            error.localizedDescription
        )
    }
}

private func loadLTXReferenceAttentionWeights(
    reference: LTXReferenceVideoConditioningInput,
    width: Int,
    height: Int,
    frameCap: Int,
    targetLatent: MLXArray,
    dtype: DType
) throws -> MLXArray? {
    guard let maskURL = reference.attentionMaskVideoURL else { return nil }
    let pixelMask = try loadVideoForEncoding(
        url: maskURL,
        width: width,
        height: height,
        frameCap: frameCap,
        temporalScaleFactor: 1,
        dtype: dtype
    )
    let targetShape = LTXVideoLatentShape(
        batch: targetLatent.dim(0),
        channels: targetLatent.dim(1),
        frames: targetLatent.dim(2),
        height: targetLatent.dim(3),
        width: targetLatent.dim(4)
    )
    guard pixelMask.dim(0) == targetShape.batch,
          pixelMask.dim(3).isMultiple(of: targetShape.height),
          pixelMask.dim(4).isMultiple(of: targetShape.width),
          targetShape.frames == 1
            || (pixelMask.dim(2) - 1).isMultiple(of: targetShape.frames - 1) else {
        throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(
            maskURL,
            "mask video frames or dimensions are incompatible with the encoded reference"
        )
    }
    let grayscale = MLX.clip(
        (MLX.mean(pixelMask, axis: 1, keepDims: true) + MLXArray(Float(1)))
            / MLXArray(Float(2)),
        min: MLXArray(Float(0)),
        max: MLXArray(Float(1))
    )
    let weights = downsampleLTXReferenceAttentionMask(
        grayscale,
        targetLatentShape: targetShape
    )
    MLX.eval(weights)
    return weights
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
    let cos = cosFreq.asType(dtype)
    let sin = sinFreq.asType(dtype)

    let halfDim = x.dim(3) / 2
    let x1 = x[0..., 0..., 0..., 0..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]

    let out1 = x1 * cos - sin * x2
    let out2 = x1 * sin + x2 * cos

    return MLX.concatenated([out1, out2], axis: 3)
}

func precomputeSplitRope(
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
    let indexCount = max(1, dim / nElem)
    let expectedFreqs = dim / 2
    let currentFreqs = positionDims * indexCount
    let padSize = max(0, expectedFreqs - currentFreqs)
    let halfHead = expectedFreqs / numHeads

    let grid = indexCount == 1
        ? MLXArray([Float(1)])
        : linspace(Float(0), Float(1), count: indexCount).asType(.float32)
    let indices = MLX.pow(MLXArray(theta), grid) * MLXArray(Float.pi / 2)

    let position32 = positions.asType(.float32)
    let middle = (
        position32[0..., 0..., 0..., 0]
            + position32[0..., 0..., 0..., 1]
    ) * MLXArray(Float(0.5))
    let tokenMajor = middle.transposed(0, 2, 1)
    let maxima = MLXArray(maxPos.map(Float.init)).reshaped(1, 1, positionDims)
    let fractional = tokenMajor / maxima
    var freqs = (
        indices.reshaped(1, 1, 1, indexCount)
            * (fractional.expandedDimensions(axis: 3) * MLXArray(Float(2)) - MLXArray(Float(1)))
    ).transposed(0, 1, 3, 2).reshaped(batch, tokenCount, currentFreqs)
    if padSize > 0 {
        freqs = MLX.concatenated([
            MLX.zeros([batch, tokenCount, padSize], dtype: .float32),
            freqs,
        ], axis: 2)
    }

    let cos = MLX.cos(freqs).reshaped(batch, tokenCount, numHeads, halfHead).transposed(0, 2, 1, 3)
    let sin = MLX.sin(freqs).reshaped(batch, tokenCount, numHeads, halfHead).transposed(0, 2, 1, 3)
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

func ltxTemporalPixelShuffle(_ x: MLXArray, upscaleFactor: Int = 2) -> MLXArray {
    precondition(x.ndim == 5, "Expected NDHWC tensor")
    let n = x.dim(0)
    let d = x.dim(1)
    let h = x.dim(2)
    let w = x.dim(3)
    let c = x.dim(4)
    precondition(c % upscaleFactor == 0, "Temporal pixel-shuffle channels must divide by the scale")

    let outC = c / upscaleFactor
    return x.reshaped(n, d, h, w, outC, upscaleFactor)
        .transposed(0, 1, 5, 2, 3, 4)
        .reshaped(n, d * upscaleFactor, h, w, outC)
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

private final class LTXTemporalPixelShuffleUpsampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d

    init(midChannels: Int) {
        self._conv.wrappedValue = Conv3d(
            inputChannels: midChannels,
            outputChannels: 2 * midChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        ltxTemporalPixelShuffle(conv(x), upscaleFactor: 2)
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

private final class LTXTemporalLatentUpsampler: Module {
    let inChannels: Int
    let midChannels: Int
    let numBlocksPerStage: Int

    @ModuleInfo(key: "initial_conv") var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") var initialNorm: LTXUpsamplerGroupNorm3d
    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "upsampler") var upsampler: LTXTemporalPixelShuffleUpsampler
    @ModuleInfo(key: "post_upsample_res_blocks") var postUpsampleResBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "final_conv") var finalConv: Conv3d

    init(inChannels: Int = 128, midChannels: Int = 512, numBlocksPerStage: Int = 4) {
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
        self._initialNorm.wrappedValue = LTXUpsamplerGroupNorm3d(
            numGroups: 32,
            numChannels: midChannels,
            eps: 1e-5
        )
        self._resBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._upsampler.wrappedValue = LTXTemporalPixelShuffleUpsampler(midChannels: midChannels)
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
        x = silu(initialNorm(initialConv(x)))
        for block in resBlocks {
            x = block(x)
        }
        x = upsampler(x)
        x = x[0..., 1..., 0..., 0..., 0...]
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
    fps: Double,
    decodeBudgetGiB: Double? = nil,
    spatialTileSizeInPixels: Int? = nil,
    spatialTileOverlapInPixels: Int = 0
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

    let tileFrames = tileFramesCandidates.min()
    guard tileFrames != nil || spatialTileSizeInPixels != nil else {
        return nil
    }

    let overlapFrames: Int
    if let tileFrames {
        let oneSecondFrames = max(8, (Int(max(1, fps).rounded()) / 8) * 8)
        overlapFrames = min(oneSecondFrames, (tileFrames / 32) * 8)
    } else {
        overlapFrames = 0
    }

    return LTXDecodeTilingConfig(
        spatialTileSizeInPixels: spatialTileSizeInPixels,
        spatialTileOverlapInPixels: spatialTileOverlapInPixels,
        temporalTileSizeInFrames: tileFrames,
        temporalTileOverlapInFrames: min(
            max(0, overlapFrames),
            max(0, (tileFrames ?? 8) - 8)
        )
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
    var video = finalizeLTXDecodedTilesRaw(output: output, weights: weights)[0]
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

func finalizeLTXDecodedTilesRaw(output: MLXArray, weights: MLXArray) -> MLXArray {
    let epsilon: Float = weights.dtype == .float16 ? 1e-4 : 1e-8
    let denominator = MLX.maximum(weights, MLXArray(epsilon).asType(weights.dtype))
    return output / denominator
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
    let decoded = decodeWithTilingRaw(
        decoder: decoder,
        latents: latents,
        spatialTileSizeInPixels: spatialTileSizeInPixels,
        spatialOverlapInPixels: spatialOverlapInPixels,
        temporalTileSizeInFrames: temporalTileSizeInFrames,
        temporalOverlapInFrames: temporalOverlapInFrames,
        spatialScale: spatialScale,
        temporalScale: temporalScale
    )
    var video = decoded[0].transposed(1, 2, 3, 0)
    let zero = MLXArray(Float(0)).asType(video.dtype)
    let one = MLXArray(Float(1)).asType(video.dtype)
    video = MLX.clip(
        (video + one) / MLXArray(Float(2)).asType(video.dtype),
        min: zero,
        max: one
    )
    let frames = (video * MLXArray(Float(255)).asType(video.dtype)).asType(.uint8)
    MLX.eval(frames)
    return frames
}

private func decodeWithTilingRaw(
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

    let decoded = finalizeLTXDecodedTilesRaw(output: output, weights: weights)
    MLX.eval(decoded)
    return decoded
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

func makeLTXVideoKeyframesMask(
    batchSize: Int,
    tokenCount: Int,
    tokensPerFirstFrame: Int,
    dtype: DType
) -> MLXArray {
    precondition(batchSize > 0, "batchSize must be positive")
    precondition(tokenCount > 0, "tokenCount must be positive")
    precondition(
        tokensPerFirstFrame > 0 && tokensPerFirstFrame <= tokenCount,
        "tokensPerFirstFrame must fit in tokenCount"
    )
    let firstFrame = MLX.ones([batchSize, tokensPerFirstFrame, 1], dtype: dtype)
    guard tokensPerFirstFrame < tokenCount else { return firstFrame }
    let remainder = MLX.zeros(
        [batchSize, tokenCount - tokensPerFirstFrame, 1],
        dtype: dtype
    )
    return MLX.concatenated([firstFrame, remainder], axis: 1)
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

func ltx25UsesDistilledAncestralStage1(
    isLTX25: Bool,
    isFullTwoStage: Bool,
    usesDFR: Bool,
    usesHDRICLoRA: Bool,
    usesRetake: Bool,
    usesDubIt: Bool,
    hasReferenceVideos: Bool
) -> Bool {
    isLTX25
        && !isFullTwoStage
        && !usesDFR
        && !usesHDRICLoRA
        && !usesRetake
        && !usesDubIt
        && !hasReferenceVideos
}

public enum LTXTransformerExecution: String, Codable, CaseIterable, Sendable {
    case eager
    case compiled
}

public struct LTXUnifiedAVGenerationOptions: Sendable {
    public static let defaultNegativePrompt = LTXAudioToVideoGenerationOptions.defaultNegativePrompt

    public let prompt: String
    public let negativePrompt: String
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let inferenceSteps: Int
    public let maxTextLength: Int
    public let videoGuidance: LTXMultiModalGuidance
    public let audioGuidance: LTXMultiModalGuidance
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float
    public let imageConditionings: [LTXVideoConditioningInput]
    public let generatedKeyframeCount: Int
    public let generatedKeyframeIndices: [Int]
    public let referenceVideos: [LTXReferenceVideoConditioningInput]
    public let loras: [LTXLoRAConfiguration]
    public let dfr: LTX25DFROptions?
    public let sigmas: [Float]?
    public let stage2Sigmas: [Float]?
    public let sampler: LTXSamplerConfiguration
    public let pipeline: LTXGenerationPipeline
    public let distilledLoRAStrengthStage1: Float
    public let distilledLoRAStrengthStage2: Float
    public let hdrColorSpace: LTXHDRColorSpace?
    public let hdrTransfer: LTXHDRTransfer
    public let hdrICLoRA: LTXHDRICLoRAOptions?
    public let vaeSpatialTileSize: Int?
    public let vaeSpatialTileOverlap: Int
    public let skipStage2: Bool
    public let precomputedTextEmbeddingsURL: URL?
    public let retake: LTXRetakeOptions?
    public let dubIt: LTXDubItOptions?
    public let transformerExecution: LTXTransformerExecution
    public let guidanceProjectionCache: LTXGuidanceProjectionCacheMode
    public let teaCache: LTXTeaCacheConfiguration?

    public init(
        prompt: String,
        negativePrompt: String = Self.defaultNegativePrompt,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        inferenceSteps: Int = 30,
        maxTextLength: Int = 1024,
        videoGuidance: LTXMultiModalGuidance = LTXMultiModalGuidance(classifierFreeScale: 3),
        audioGuidance: LTXMultiModalGuidance = LTXMultiModalGuidance(classifierFreeScale: 7),
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1.0,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1.0,
        imageConditionings: [LTXVideoConditioningInput] = [],
        generatedKeyframeCount: Int = 0,
        generatedKeyframeIndices: [Int] = [],
        referenceVideos: [LTXReferenceVideoConditioningInput] = [],
        loras: [LTXLoRAConfiguration] = [],
        dfr: LTX25DFROptions? = nil,
        sigmas: [Float]? = nil,
        stage2Sigmas: [Float]? = nil,
        sampler: LTXSamplerConfiguration = LTXSamplerConfiguration(),
        pipeline: LTXGenerationPipeline = .twoStage,
        distilledLoRAStrengthStage1: Float = 0,
        distilledLoRAStrengthStage2: Float = 1,
        hdrColorSpace: LTXHDRColorSpace? = nil,
        hdrTransfer: LTXHDRTransfer = .acesCCT,
        hdrICLoRA: LTXHDRICLoRAOptions? = nil,
        vaeSpatialTileSize: Int? = nil,
        vaeSpatialTileOverlap: Int = 256,
        skipStage2: Bool = false,
        precomputedTextEmbeddingsURL: URL? = nil,
        retake: LTXRetakeOptions? = nil,
        dubIt: LTXDubItOptions? = nil,
        transformerExecution: LTXTransformerExecution = .eager,
        guidanceProjectionCache: LTXGuidanceProjectionCacheMode = .disabled,
        teaCache: LTXTeaCacheConfiguration? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.inferenceSteps = inferenceSteps
        self.maxTextLength = maxTextLength
        self.videoGuidance = videoGuidance
        self.audioGuidance = audioGuidance
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
        self.imageConditionings = imageConditionings
        self.generatedKeyframeCount = generatedKeyframeCount
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.referenceVideos = referenceVideos
        self.loras = loras
        self.dfr = dfr
        self.sigmas = sigmas
        self.stage2Sigmas = stage2Sigmas
        self.sampler = sampler
        self.pipeline = pipeline
        self.distilledLoRAStrengthStage1 = distilledLoRAStrengthStage1
        self.distilledLoRAStrengthStage2 = distilledLoRAStrengthStage2
        self.hdrColorSpace = hdrColorSpace
        self.hdrTransfer = hdrTransfer
        self.hdrICLoRA = hdrICLoRA
        self.vaeSpatialTileSize = vaeSpatialTileSize
        self.vaeSpatialTileOverlap = vaeSpatialTileOverlap
        self.skipStage2 = skipStage2
        self.precomputedTextEmbeddingsURL = precomputedTextEmbeddingsURL?.standardizedFileURL
        self.retake = retake
        self.dubIt = dubIt
        self.transformerExecution = transformerExecution
        self.guidanceProjectionCache = guidanceProjectionCache
        self.teaCache = teaCache
    }
}

public struct LTXUnifiedAVGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let videoLatents: MLXArray
    public let audioLatents: MLXArray
    public let audioWaveform: MLXArray
    public let audioSampleRate: Int
    public let hdrOutput: LTXHDROutputFrames?
    public let generatedKeyframeLatents: MLXArray?
    public let generatedKeyframeIndices: [Int]
    public let playbackFPS: Double
    public let timings: LTXGenerationTimings

    public init(
        frames: MLXArray,
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        audioWaveform: MLXArray,
        audioSampleRate: Int,
        hdrOutput: LTXHDROutputFrames? = nil,
        generatedKeyframeLatents: MLXArray? = nil,
        generatedKeyframeIndices: [Int] = [],
        playbackFPS: Double = 24,
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.frames = frames
        self.videoLatents = videoLatents
        self.audioLatents = audioLatents
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
        self.hdrOutput = hdrOutput
        self.generatedKeyframeLatents = generatedKeyframeLatents
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.playbackFPS = playbackFPS
        self.timings = timings
    }
}

public struct LTXUnifiedVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let hdrOutput: LTXHDROutputFrames?
    public let videoLatents: MLXArray
    public let generatedKeyframeLatents: MLXArray?
    public let generatedKeyframeIndices: [Int]
    public let playbackFPS: Double
    public let timings: LTXGenerationTimings

    public init(
        frames: MLXArray,
        hdrOutput: LTXHDROutputFrames? = nil,
        videoLatents: MLXArray,
        generatedKeyframeLatents: MLXArray? = nil,
        generatedKeyframeIndices: [Int] = [],
        playbackFPS: Double = 24,
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.frames = frames
        self.hdrOutput = hdrOutput
        self.videoLatents = videoLatents
        self.generatedKeyframeLatents = generatedKeyframeLatents
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.playbackFPS = playbackFPS
        self.timings = timings
    }
}

public struct LTXTextToAudioGuidance: Sendable, Hashable {
    public let classifierFreeScale: Float
    public let spatioTemporalScale: Float
    public let rescale: Float
    public let spatioTemporalBlocks: Set<Int>
    public let skipStep: Int

    public init(
        classifierFreeScale: Float = 7,
        spatioTemporalScale: Float = 1,
        rescale: Float = 0.7,
        spatioTemporalBlocks: Set<Int> = [28],
        skipStep: Int = 0
    ) {
        self.classifierFreeScale = classifierFreeScale
        self.spatioTemporalScale = spatioTemporalScale
        self.rescale = rescale
        self.spatioTemporalBlocks = spatioTemporalBlocks
        self.skipStep = skipStep
    }

    func combine(
        conditioned: MLXArray,
        negativeText: MLXArray,
        perturbed: MLXArray
    ) -> MLXArray {
        let dtype = conditioned.dtype
        let conditioned32 = conditioned.asType(.float32)
        var prediction = conditioned32
            + MLXArray(classifierFreeScale - 1)
                * (conditioned32 - negativeText.asType(.float32))
            + MLXArray(spatioTemporalScale)
                * (conditioned32 - perturbed.asType(.float32))
        if rescale != 0 {
            let factor = sampleStandardDeviation(conditioned32)
                / sampleStandardDeviation(prediction)
            prediction = prediction
                * (MLXArray(rescale) * factor + MLXArray(1 - rescale))
        }
        return prediction.asType(dtype)
    }

    func shouldSkip(step: Int) -> Bool {
        skipStep > 0 && !step.isMultiple(of: skipStep + 1)
    }
}

public struct LTXTextToAudioGenerationOptions: Sendable {
    public let prompt: String
    public let negativePrompt: String
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let inferenceSteps: Int
    public let maxTextLength: Int
    public let guidance: LTXTextToAudioGuidance
    public let sigmas: [Float]?
    public let loras: [LTXLoRAConfiguration]

    public init(
        prompt: String,
        negativePrompt: String = LTXUnifiedAVGenerationOptions.defaultNegativePrompt,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        inferenceSteps: Int = 30,
        maxTextLength: Int = 1_024,
        guidance: LTXTextToAudioGuidance = LTXTextToAudioGuidance(),
        sigmas: [Float]? = nil,
        loras: [LTXLoRAConfiguration] = []
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.inferenceSteps = inferenceSteps
        self.maxTextLength = maxTextLength
        self.guidance = guidance
        self.sigmas = sigmas
        self.loras = loras
    }
}

public struct LTXTextToAudioGenerationResult: @unchecked Sendable {
    public let audioLatents: MLXArray
    public let audioWaveform: MLXArray
    public let audioSampleRate: Int
    public let timings: LTXGenerationTimings

    public init(
        audioLatents: MLXArray,
        audioWaveform: MLXArray,
        audioSampleRate: Int,
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.audioLatents = audioLatents
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
        self.timings = timings
    }
}

private struct LTXUnifiedGenerationOutput {
    let frames: MLXArray
    let hdrOutput: LTXHDROutputFrames?
    let videoLatents: MLXArray
    let audioLatents: MLXArray
    let audioWaveform: MLXArray?
    let audioSampleRate: Int?
    let generatedKeyframeLatents: MLXArray?
    let generatedKeyframeIndices: [Int]
    let playbackFPS: Double
    let timings: LTXGenerationTimings
}

public struct LTXAudioToVideoGenerationOptions: Sendable {
    public static let defaultNegativePrompt = """
    blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
    grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
    deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
    wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
    field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
    lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
    valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
    mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
    off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
    pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
    inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.
    """

    public let prompt: String
    public let negativePrompt: String
    public let audioURL: URL
    public let audioStartTime: Double
    public let audioMaxDuration: Double?
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let inferenceSteps: Int
    public let maxTextLength: Int
    public let guidance: LTXAudioToVideoGuidance
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float
    public let imageConditionings: [LTXVideoConditioningInput]
    public let generatedKeyframeCount: Int
    public let generatedKeyframeIndices: [Int]
    public let hdrColorSpace: LTXHDRColorSpace?
    public let hdrTransfer: LTXHDRTransfer
    public let transformerExecution: LTXTransformerExecution

    public init(
        prompt: String,
        negativePrompt: String = Self.defaultNegativePrompt,
        audioURL: URL,
        audioStartTime: Double = 0,
        audioMaxDuration: Double? = nil,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        inferenceSteps: Int = 30,
        maxTextLength: Int = 1_024,
        guidance: LTXAudioToVideoGuidance = LTXAudioToVideoGuidance(),
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1,
        imageConditionings: [LTXVideoConditioningInput] = [],
        generatedKeyframeCount: Int = 0,
        generatedKeyframeIndices: [Int] = [],
        hdrColorSpace: LTXHDRColorSpace? = nil,
        hdrTransfer: LTXHDRTransfer = .acesCCT,
        transformerExecution: LTXTransformerExecution = .eager
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.audioURL = audioURL
        self.audioStartTime = audioStartTime
        self.audioMaxDuration = audioMaxDuration
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.inferenceSteps = inferenceSteps
        self.maxTextLength = maxTextLength
        self.guidance = guidance
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
        self.imageConditionings = imageConditionings
        self.generatedKeyframeCount = generatedKeyframeCount
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.hdrColorSpace = hdrColorSpace
        self.hdrTransfer = hdrTransfer
        self.transformerExecution = transformerExecution
    }
}

public struct LTXAudioToVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let hdrOutput: LTXHDROutputFrames?
    public let videoLatents: MLXArray
    public let audioLatents: MLXArray
    public let sourceAudio: MediaAudioBuffer
    public let generatedKeyframeLatents: MLXArray?
    public let generatedKeyframeIndices: [Int]
    public let timings: LTXGenerationTimings

    public init(
        frames: MLXArray,
        hdrOutput: LTXHDROutputFrames? = nil,
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        sourceAudio: MediaAudioBuffer,
        generatedKeyframeLatents: MLXArray? = nil,
        generatedKeyframeIndices: [Int] = [],
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.frames = frames
        self.hdrOutput = hdrOutput
        self.videoLatents = videoLatents
        self.audioLatents = audioLatents
        self.sourceAudio = sourceAudio
        self.generatedKeyframeLatents = generatedKeyframeLatents
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.timings = timings
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
    case ltx25ConditioningRequiresLTX25
    case invalidGeneratedKeyframes([Int])
    case imageNotFound(URL)
    case imageDecodeFailed(URL)
    case referenceVideoNotFound(URL)
    case referenceVideoDecodeFailed(URL, String)
    case emptyPrompt
    case decoderNotLoaded
    case encoderNotLoaded
    case upsamplerNotLoaded
    case audioEmbeddingsMissing
    case audioDecoderNotLoaded
    case vocoderNotLoaded
    case bweVocoderConfigMissing(URL)
    case fullGenerationRequiresCompatibleModel(URL)
    case audioToVideoRequiresCompatibleModel(URL)
    case textToAudioRequiresLTX25Full(URL)
    case durationPredictionRequiresLTX25(URL?)
    case distilledLoRAMissing(URL)
    case loraMissing(URL)
    case audioVAEWeightsMissing(URL)
    case audioSourceNotFound(URL)
    case unsupportedAudioChannels(Int)
    case audioSegmentTooShort(required: Double, available: Double)
    case invalidAudioStartTime(Double)
    case invalidInferenceSteps(Int)
    case invalidSigmaSchedule([Float])
    case invalidFrameRate(Double)
    case audioDecodeReturnedTooFewSamples(required: Int, actual: Int)
    case audioLatentTooShort(required: Int, actual: Int)
    case audioToVideoGeneratorNotLoaded
    case textToAudioGeneratorNotLoaded
    case audioToVideoRequiresReload
    case fullGenerationRequiresReload
    case dfrRequiresLTX25Full(URL)
    case retakeRequiresLTX25(URL?)
    case invalidRetakeRange(start: Double, end: Double)
    case dubItRequiresLTX25(URL?)
    case dubItRequiresOneICLoRA(Int)
    case dubItReferenceAudioMissing(URL)
    case incompatibleLTX25Workflows(String)

    public var errorDescription: String? {
        switch self {
        case .transformerWeightsMissing(let url):
            return "Missing LTX transformer weights at \(url.path)"
        case .upsamplerWeightsMissing(let url):
            return "Missing LTX upsampler weights at \(url.path)"
        case .unsupportedLTX23SplitModel(let url):
            return """
            Detected an LTX 2.3 split MLX model at \(url.path). Use \
            `mere.run video generate --variant distilled` for video-only output, or \
            `--variant unified-av` for synchronized audio and video.
            """
        case .ltx23TextEncoderMissing(let id):
            return """
            LTX 2.3 requires the companion Gemma 3 text encoder `\(id)`. Install it with \
            `mere.run model pull video-ltx23-full-mlx --accept-model-license` for unified AV and A2Vid, or \
            `mere.run model pull video-ltx23-av-mlx --accept-model-license` for the distilled lane, or set \
            MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT to a local mlx-community/gemma-3-12b-it-4bit checkout.
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
        case .ltx25ConditioningRequiresLTX25:
            return "Arbitrary timed image conditioning and generated keyframe slots require an LTX 2.5 checkpoint."
        case .invalidGeneratedKeyframes(let values):
            return "Generated keyframe indices must be strictly increasing pixel-frame positions inside the output (got \(values))."
        case .imageNotFound(let url):
            return "Source image not found: \(url.path)"
        case .imageDecodeFailed(let url):
            return "Could not decode source image: \(url.path)"
        case .referenceVideoNotFound(let url):
            return "Reference video not found: \(url.path)"
        case .referenceVideoDecodeFailed(let url, let details):
            return "Could not decode reference video \(url.path): \(details)"
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
        case .fullGenerationRequiresCompatibleModel(let url):
            return "Native full LTX generation requires a compatible LTX 2.3 or 2.5 dev, distilled LoRA, VAE, and vocoder bundle at \(url.path)."
        case .audioToVideoRequiresCompatibleModel(let url):
            return "Native LTX audio-to-video requires a compatible full LTX 2.3 or 2.5 model at \(url.path)."
        case .textToAudioRequiresLTX25Full(let url):
            return "Native LTX text-to-audio requires the full LTX 2.5 checkpoint at \(url.path)."
        case .durationPredictionRequiresLTX25(let url):
            let location = url.map { " at \($0.path)" } ?? ""
            return "Automatic duration prediction requires an LTX 2.5 checkpoint\(location)."
        case .distilledLoRAMissing(let url):
            return "Missing the official LTX 2.3 distilled LoRA at \(url.path)."
        case .loraMissing(let url):
            return "Missing LTX LoRA at \(url.path)."
        case .audioVAEWeightsMissing(let url):
            return "Missing LTX audio VAE weights at \(url.path)."
        case .audioSourceNotFound(let url):
            return "Audio source not found: \(url.path)"
        case .unsupportedAudioChannels(let channels):
            return "LTX audio-to-video supports mono or stereo source audio, not \(channels) channels."
        case .audioSegmentTooShort(let required, let available):
            return "The selected audio segment is too short: generation requires \(required) seconds, but only \(available) seconds remain."
        case .invalidAudioStartTime(let value):
            return "audioStartTime must be finite and nonnegative (got \(value))."
        case .invalidInferenceSteps(let value):
            return "inferenceSteps must be positive (got \(value))."
        case .invalidSigmaSchedule(let values):
            return "LTX sigmas must be finite, nonincreasing values in [0, 1] ending at zero (got \(values))."
        case .invalidFrameRate(let value):
            return "fps must be positive (got \(value))."
        case .audioDecodeReturnedTooFewSamples(let required, let actual):
            return "Audio decoding returned \(actual) samples; the requested segment requires \(required) without padding."
        case .audioLatentTooShort(let required, let actual):
            return "The encoded audio contains \(actual) latent frames; generation requires \(required)."
        case .audioToVideoGeneratorNotLoaded:
            return "LTX audio-to-video generator is not loaded."
        case .textToAudioGeneratorNotLoaded:
            return "LTX text-to-audio generator is not loaded."
        case .audioToVideoRequiresReload:
            return "Reload the LTX audio-to-video generator before starting another generation."
        case .fullGenerationRequiresReload:
            return "Reload the full LTX generator before starting another two-stage generation."
        case .dfrRequiresLTX25Full(let url):
            return "LTX 2.5 DFR requires the full checkpoint, spatial/temporal upsamplers, and distilled LoRA at \(url.path)."
        case .retakeRequiresLTX25(let url):
            let location = url.map { " at \($0.path)" } ?? ""
            return "LTX Retake requires an official LTX 2.5 checkpoint\(location)."
        case .invalidRetakeRange(let start, let end):
            return "Retake requires 0 <= start-time < end-time <= output duration (got \(start)...\(end))."
        case .dubItRequiresLTX25(let url):
            let location = url.map { " at \($0.path)" } ?? ""
            return "LTX Dub-It requires an official LTX 2.5 checkpoint\(location)."
        case .dubItRequiresOneICLoRA(let count):
            return "LTX Dub-It requires exactly one IC-LoRA (got \(count))."
        case .dubItReferenceAudioMissing(let url):
            return "LTX Dub-It reference video has no audio track: \(url.path)"
        case .incompatibleLTX25Workflows(let details):
            return "Incompatible LTX 2.5 workflow options: \(details)"
        }
    }
}

private func loadLTXVideoDecoder(
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXVideoDecoder {
    let decoder = LTXVideoDecoder(timestepConditioning: false, architecture: .ltx23Split)
    let stats = try SafetensorsStreamingLoader.loadArrays(
        url: weightsURL,
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

    if let mean = stats["latents_mean"]
        ?? stats["vae.per_channel_statistics.mean-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.mean-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.mean"] {
        decoder.latentsMean = mean.asType(.float32)
    }
    if let std = stats["latents_std"]
        ?? stats["vae.per_channel_statistics.std-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.std-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.std"] {
        decoder.latentsStd = std.asType(.float32)
    }

    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: decoder,
        dtype: dtype,
        verify: .none,
        include: { key in
            key.hasPrefix("decoder.")
                || key.hasPrefix("vae.decoder.")
                || key.hasPrefix("vae_decoder.")
        },
        mapper: { key, value in
            mapLTXDecoderWeight(key: key, value: value, dtype: dtype, sourceLayout: sourceLayout)
        },
        batchSize: 24
    )
    return decoder
}

private func loadLTXVideoUpsampler(
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXLatentUpsampler {
    let upsampler = LTXLatentUpsampler(inChannels: 128, midChannels: 1_024, numBlocksPerStage: 4)
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: upsampler,
        dtype: dtype,
        verify: .none,
        include: { _ in true },
        mapper: { key, value in
            mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype, sourceLayout: sourceLayout)
        },
        batchSize: 24
    )
    return upsampler
}

private func loadLTXTemporalVideoUpsampler(
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXTemporalLatentUpsampler {
    let upsampler = LTXTemporalLatentUpsampler(inChannels: 128, midChannels: 512, numBlocksPerStage: 4)
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: upsampler,
        dtype: dtype,
        verify: .none,
        include: { _ in true },
        mapper: { key, value in
            mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype, sourceLayout: sourceLayout)
        },
        batchSize: 24
    )
    return upsampler
}

private func loadLTXAudioDecoder(
    weightsURL: URL,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXAudioDecoder {
    let decoder = LTXAudioDecoder()
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: decoder,
        dtype: .float32,
        verify: .none,
        include: { key in
            key.hasPrefix("audio_vae.decoder.")
                || key.hasPrefix("audio_vae.per_channel_statistics.")
        },
        mapper: { key, value in
            mapAudioVaeDecoderWeight(
                key: key,
                value: value,
                dtype: .float32,
                sourceLayout: sourceLayout
            )
        },
        batchSize: 24
    )
    return decoder
}

private func loadLTXVocoder(
    weightsURL: URL,
    sourceLayout: LTXVocoderWeightLayout,
    configurationRoot: URL,
    usesPackedConfiguration: Bool
) throws -> LTXAudioVocoderBase {
    let metadata = try SafetensorsStreamingLoader.metadata(url: weightsURL)
    let flavor = detectLTXVocoderFlavor(keys: metadata.keys)
    let vocoder: LTXAudioVocoderBase
    switch flavor {
    case .legacy:
        vocoder = LTXVocoder()
    case .bandwidthExtension:
        let config = usesPackedConfiguration
            ? try loadLTXPackedBWEVocoderConfig(weightsURL: weightsURL)
            : try loadLTXBWEVocoderConfig(modelRoot: configurationRoot)
        guard let config else {
            throw LTXUnifiedAVGeneratorError.bweVocoderConfigMissing(configurationRoot)
        }
        vocoder = LTXVocoderWithBWE(config: config)
    }
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: vocoder,
        dtype: .float32,
        verify: .none,
        include: { $0.hasPrefix("vocoder.") },
        mapper: { key, value in
            mapVocoderWeight(
                key: key,
                value: value,
                dtype: .float32,
                sourceLayout: sourceLayout,
                targetFlavor: flavor
            )
        },
        batchSize: 24
    )
    return vocoder
}

private func ltx25ImageConditionings(
    options: LTXUnifiedAVGenerationOptions
) -> [LTXVideoConditioningInput] {
    var values = options.imageConditionings
    if let sourceImageURL = options.sourceImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: sourceImageURL,
                pixelFrameIndex: options.imageFrameIndex,
                strength: options.imageStrength
            )
        )
    }
    if let endImageURL = options.endImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: endImageURL,
                pixelFrameIndex: options.numFrames - 1,
                strength: options.endImageStrength
            )
        )
    }
    return values
}

private func ltx25ImageConditionings(
    options: LTXAudioToVideoGenerationOptions
) -> [LTXVideoConditioningInput] {
    var values = options.imageConditionings
    if let sourceImageURL = options.sourceImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: sourceImageURL,
                pixelFrameIndex: options.imageFrameIndex,
                strength: options.imageStrength
            )
        )
    }
    if let endImageURL = options.endImageURL {
        values.append(
            LTXVideoConditioningInput(
                imageURL: endImageURL,
                pixelFrameIndex: options.numFrames - 1,
                strength: options.endImageStrength
            )
        )
    }
    return values
}

private func makeConditionedLTX25VideoTokenState(
    initialLatent: MLXArray,
    positions: MLXArray,
    imageConditionings: [LTXVideoConditioningInput],
    generatedKeyframeIndices: [Int],
    initialGeneratedKeyframes: MLXArray?,
    encoder: LTXVideoEncoder?,
    pixelWidth: Int,
    pixelHeight: Int,
    fps: Double,
    replaceFirstImage: Bool = true,
    hdrColorSpace: LTXHDRColorSpace? = nil
) throws -> LTX25VideoTokenState {
    var state = LTX25VideoTokenState(initialLatent: initialLatent, positions: positions)
    if !imageConditionings.isEmpty {
        guard let encoder else {
            throw LTXUnifiedAVGeneratorError.encoderNotLoaded
        }
        for input in imageConditionings {
            let image = try loadImageForEncoding(
                url: input.imageURL,
                width: pixelWidth,
                height: pixelHeight,
                dtype: initialLatent.dtype,
                hdrColorSpace: hdrColorSpace,
                crf: input.crf ?? 18
            )
            state.applyImageLatent(
                encoder.encode(image: image),
                pixelFrameIndex: input.pixelFrameIndex,
                strength: input.strength,
                fps: fps,
                replaceFirstFrame: replaceFirstImage
            )
        }
    }
    if !generatedKeyframeIndices.isEmpty {
        state.appendGeneratedKeyframeSlots(
            pixelFrameIndices: generatedKeyframeIndices,
            initialKeyframes: initialGeneratedKeyframes,
            fps: fps
        )
    }
    return state
}

public actor LTXUnifiedAVGenerator {
    private var textEncoder: LTXGemmaTextEncoder?
    private var transformer: (any LTXUnifiedAVTransformerRuntime)?
    private var audioOnlyTransformer: LTXAudioOnlyTransformerV2?
    private var decoder: LTXVideoDecoder?
    private var diffusionDecoder: LTXDiffusionVideoDecoder?
    private var encoder: LTXVideoEncoder?
    private var upsampler: LTXLatentUpsampler?
    private var temporalUpsampler: LTXTemporalLatentUpsampler?
    private var audioDecoder: LTXAudioDecoder?
    private var vocoder: LTXAudioVocoderBase?
    private var modelWeightsURL: URL?
    private var videoEncoderWeightsURL: URL?
    private var videoVAEWeightLayout: LTXTensorWeightLayout = .pytorch
    private var videoVAEArchitecture: LTXVideoVAEArchitecture = .legacy
    private var loadedDType: DType = .bfloat16
    private var loadedRoot: URL?
    private var audioVAEWeightsURL: URL?
    private var distilledLoRAURL: URL?
    private var runtimeLoRAAdapter: LTXRuntimeLoRAAdapter?
    private var runtimeUserLoRAAdapters: [LTXRuntimeLoRAAdapter] = []
    private var runtimeDetailingLoRAAdapters: [LTXRuntimeLoRAAdapter] = []
    private var runtimeAudioLoRAAdapters: [LTXRuntimeLoRAAdapter] = []
    private var loadedForAudioToVideo = false
    private var loadedForFullTwoStage = false
    private var loadedForReusableFullTwoStage = false
    private var loadedForVideoOnlyOutput = false
    private var loadedForLTX25 = false
    private var loadedForDFR = false
    private var twoStageGenerationConsumed = false
    private var promptEmbeddingCache = LTXPromptEmbeddingCache()

    public init() {}

    public func configurePromptCache(capacity: Int) {
        promptEmbeddingCache = LTXPromptEmbeddingCache(capacity: capacity)
    }

    public func clearPromptCache() {
        promptEmbeddingCache.removeAll()
    }

    public func promptCacheStatistics() -> LTXPromptCacheStatistics {
        promptEmbeddingCache.statistics()
    }

    @discardableResult
    public func loadTextToAudio(
        modelRoot: URL,
        dtype: DType = .bfloat16
    ) async throws -> LTXLoadTimings {
        let totalStart = ltxMonotonicSeconds()
        await unload()
        let root = modelRoot.standardizedFileURL
        guard isLTX25FullModelRoot(root) else {
            throw LTXUnifiedAVGeneratorError.textToAudioRequiresLTX25Full(root)
        }
        let resources = LTX25Resources(rootURL: root)
        let transformerURL = resolvedLTX25TransformerURL(resources: resources, kind: .dev)
        let audioWeightsURL = resources.audioVAEURL
        guard FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw LTXUnifiedAVGeneratorError.transformerWeightsMissing(transformerURL)
        }
        guard FileManager.default.fileExists(atPath: audioWeightsURL.path) else {
            throw LTXUnifiedAVGeneratorError.audioVAEWeightsMissing(audioWeightsURL)
        }

        let textStart = ltxMonotonicSeconds()
        let text = LTXGemmaTextEncoder()
        try await text.load(
            modelRoot: root,
            textEncoderRoot: nil,
            dtype: dtype,
            loadConnectorWeights: true
        )
        let textSeconds = ltxMonotonicSeconds() - textStart

        let transformerStart = ltxMonotonicSeconds()
        let model = LTXAudioOnlyTransformerV2()
        try loadLTX25TransformerWeights(
            url: transformerURL,
            model: model,
            dtype: dtype,
            sourceInclude: isLTXAudioOnlyTransformerWeight,
            nativeInclude: {
                isLTXAudioOnlyTransformerWeight("model.diffusion_model.\($0)")
            }
        )
        let transformerSeconds = ltxMonotonicSeconds() - transformerStart

        let decoderStart = ltxMonotonicSeconds()
        let activeAudioDecoder = try loadLTXAudioDecoder(
            weightsURL: audioWeightsURL,
            sourceLayout: .pytorch
        )
        let activeVocoder = try loadLTXVocoder(
            weightsURL: audioWeightsURL,
            sourceLayout: .pytorch,
            configurationRoot: root,
            usesPackedConfiguration: true
        )
        let decoderSeconds = ltxMonotonicSeconds() - decoderStart

        textEncoder = text
        audioOnlyTransformer = model
        audioDecoder = activeAudioDecoder
        vocoder = activeVocoder
        loadedRoot = root
        loadedDType = dtype
        loadedForLTX25 = true
        return LTXLoadTimings(
            textEncoderSeconds: textSeconds,
            transformerSeconds: transformerSeconds,
            audioDecoderSeconds: decoderSeconds,
            totalSeconds: ltxMonotonicSeconds() - totalStart
        )
    }

    public func generateTextToAudio(
        options: LTXTextToAudioGenerationOptions
    ) async throws -> LTXTextToAudioGenerationResult {
        let totalStart = ltxMonotonicSeconds()
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let negativePrompt = options.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !negativePrompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.fps.isFinite, options.fps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameRate(options.fps)
        }
        guard options.inferenceSteps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidInferenceSteps(options.inferenceSteps)
        }
        guard let textEncoder,
              let audioOnlyTransformer,
              let audioDecoder,
              let vocoder else {
            throw LTXUnifiedAVGeneratorError.textToAudioGeneratorNotLoaded
        }
        let sigmas = try validatedLTXSigmaSchedule(
            options.sigmas ?? LTX2DiffusionScheduler.sigmas(steps: options.inferenceSteps)
        )

        if runtimeAudioLoRAAdapters.isEmpty, !options.loras.isEmpty {
            runtimeAudioLoRAAdapters = try options.loras.map { configuration in
                guard FileManager.default.fileExists(atPath: configuration.url.path) else {
                    throw LTXUnifiedAVGeneratorError.loraMissing(configuration.url)
                }
                return try LTXRuntimeLoRAAdapter.install(
                    url: configuration.url,
                    into: audioOnlyTransformer,
                    strength: configuration.strength,
                    expectedPairCount: nil,
                    ignoreMissingTargets: true
                )
            }
        }
        runtimeAudioLoRAAdapters.forEach { $0.setActive(true) }
        defer { runtimeAudioLoRAAdapters.forEach { $0.setActive(false) } }

        let textStart = ltxMonotonicSeconds()
        let positiveEncoding = try await textEncoder.encode(
            prompt: prompt,
            maxLength: options.maxTextLength
        )
        let negativeEncoding = try await textEncoder.encode(
            prompt: negativePrompt,
            maxLength: options.maxTextLength
        )
        guard let positiveAudioContext = positiveEncoding.audioEmbeddings,
              let negativeAudioContext = negativeEncoding.audioEmbeddings else {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }
        MLX.eval(positiveAudioContext, negativeAudioContext)
        await textEncoder.unload()
        self.textEncoder = nil
        Memory.clearCache()
        let textSeconds = ltxMonotonicSeconds() - textStart

        let preparationStart = ltxMonotonicSeconds()
        let audioFrameCount = computeAudioLatentFrameCount(
            videoFrames: options.numFrames,
            fps: options.fps
        )
        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        let initialAudio = MLXRandom.normal(
            [1, LTXAudioLatentChannels, audioFrameCount, LTXAudioLatentMelBins]
        ).asType(positiveAudioContext.dtype)
        let positions = createAudioPositionGrid(batchSize: 1, audioFrames: audioFrameCount)
        let rope = precomputeSplitRope(
            positions: positions,
            dim: 2_048,
            theta: 10_000,
            maxPos: [20],
            numHeads: 32
        )
        MLX.eval(initialAudio, rope.cos, rope.sin)
        let preparationSeconds = ltxMonotonicSeconds() - preparationStart

        let denoiseStart = ltxMonotonicSeconds()
        let audioLatents = denoiseLTX25AudioOnlyLoop(
            audioLatents: initialAudio,
            audioRope: rope,
            positiveContext: positiveAudioContext,
            negativeContext: negativeAudioContext,
            transformer: audioOnlyTransformer,
            sigmas: sigmas,
            guidance: options.guidance
        )
        MLX.eval(audioLatents)
        let denoiseSeconds = ltxMonotonicSeconds() - denoiseStart

        let decodeStart = ltxMonotonicSeconds()
        let mel = audioDecoder.decode(latents: audioLatents.asType(.float32))
        let vocoded = vocoder(mel)
        let waveform = matchLTXAudioWaveformDuration(
            vocoded,
            videoFrames: options.numFrames,
            fps: options.fps,
            sampleRate: vocoder.outputSamplingRate
        )
        MLX.eval(waveform)
        let decodeSeconds = ltxMonotonicSeconds() - decodeStart
        return LTXTextToAudioGenerationResult(
            audioLatents: audioLatents,
            audioWaveform: waveform,
            audioSampleRate: vocoder.outputSamplingRate,
            timings: LTXGenerationTimings(
                textEncodingSeconds: textSeconds,
                preparationSeconds: preparationSeconds,
                stage1DenoiseSeconds: denoiseSeconds,
                audioDecodeSeconds: decodeSeconds,
                totalSeconds: ltxMonotonicSeconds() - totalStart
            )
        )
    }

    public func predictFrameCount(
        prompt: String,
        frameRate: Double,
        range: LTX25AutoDuration = LTX25AutoDuration(),
        conditioning: LTX25DurationConditioning = .audioVideo,
        maxTextLength: Int = 1_024
    ) async throws -> Int {
        guard loadedForLTX25, let loadedRoot else {
            throw LTXUnifiedAVGeneratorError.durationPredictionRequiresLTX25(self.loadedRoot)
        }
        guard let textEncoder else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        let encoding = try await textEncoder.encode(
            prompt: trimmedPrompt,
            maxLength: maxTextLength
        )
        let head = try LTX25DurationHead.load(
            weightsURL: LTX25Resources(rootURL: loadedRoot).durationHeadURL,
            dtype: loadedDType
        )
        let videoTokens: MLXArray? = conditioning == .audioVideo
            ? encoding.videoEmbeddings
            : nil
        let audioTokens = encoding.audioEmbeddings
        if conditioning == .audioOnly, audioTokens == nil {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }
        let frameCount = try head.predictFrameCount(
            videoTokens: videoTokens,
            audioTokens: audioTokens,
            frameRate: frameRate,
            range: range
        )
        Memory.clearCache()
        return frameCount
    }

    @discardableResult
    public func loadDFR(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder: LTXVideoDecoderKind = .diffusion,
        videoDecoderDType: DType? = nil
    ) async throws -> LTXLoadTimings {
        let totalStart = ltxMonotonicSeconds()
        let root = modelRoot.standardizedFileURL
        guard isLTX25FullModelRoot(root) else {
            throw LTXUnifiedAVGeneratorError.dfrRequiresLTX25Full(root)
        }
        let base = try await loadFullReusable(
            modelRoot: root,
            dtype: dtype,
            videoDecoder: videoDecoder,
            videoDecoderDType: videoDecoderDType
        )
        let temporalStart = ltxMonotonicSeconds()
        do {
            temporalUpsampler = try loadLTXTemporalVideoUpsampler(
                weightsURL: LTX25Resources(rootURL: root).temporalUpsamplerURL,
                dtype: dtype,
                sourceLayout: .pytorch
            )
        } catch {
            await unload()
            throw error
        }
        let temporalSeconds = ltxMonotonicSeconds() - temporalStart
        loadedForDFR = true
        return LTXLoadTimings(
            textEncoderSeconds: base.textEncoderSeconds,
            transformerSeconds: base.transformerSeconds,
            videoDecoderSeconds: base.videoDecoderSeconds,
            upsamplerSeconds: base.upsamplerSeconds + temporalSeconds,
            audioDecoderSeconds: base.audioDecoderSeconds,
            loraAdapterSeconds: base.loraAdapterSeconds,
            totalSeconds: ltxMonotonicSeconds() - totalStart
        )
    }

    @discardableResult
    public func loadFull(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder: LTXVideoDecoderKind = .diffusion,
        videoDecoderDType: DType? = nil
    ) async throws -> LTXLoadTimings {
        let root = modelRoot.standardizedFileURL
        guard isLTX23FullModelRoot(root) || isLTX25FullModelRoot(root) else {
            throw LTXUnifiedAVGeneratorError.fullGenerationRequiresCompatibleModel(root)
        }
        let timings = try await loadAudioToVideo(
            modelRoot: root,
            dtype: dtype,
            videoDecoder: videoDecoder,
            videoDecoderDType: videoDecoderDType
        )
        loadedForFullTwoStage = true
        loadedForReusableFullTwoStage = false
        return timings
    }

    /// Loads the full dev + distilled-LoRA quality pipeline without requiring
    /// audio decoder or vocoder output components.
    @discardableResult
    public func loadFullVideoOnly(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder: LTXVideoDecoderKind = .diffusion,
        videoDecoderDType: DType? = nil
    ) async throws -> LTXLoadTimings {
        let root = modelRoot.standardizedFileURL
        guard isLTX23AudioToVideoModelRoot(root) || isLTX25FullModelRoot(root) else {
            throw LTXUnifiedAVGeneratorError.fullGenerationRequiresCompatibleModel(root)
        }
        let timings = try await loadAudioToVideo(
            modelRoot: root,
            dtype: dtype,
            videoDecoder: videoDecoder,
            videoDecoderDType: videoDecoderDType
        )
        loadedForFullTwoStage = true
        loadedForReusableFullTwoStage = false
        loadedForVideoOnlyOutput = true
        return timings
    }

    /// Loads the full dev transformer once and installs the distilled adapter as
    /// a reversible runtime path. Stage 1 uses the untouched dev weights; Stage 2
    /// enables the adapter without permanently fusing into the base checkpoint.
    @discardableResult
    public func loadFullReusable(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder: LTXVideoDecoderKind = .diffusion,
        videoDecoderDType: DType? = nil
    ) async throws -> LTXLoadTimings {
        let totalStart = ltxMonotonicSeconds()
        let root = modelRoot.standardizedFileURL
        guard isLTX23FullModelRoot(root) || isLTX25FullModelRoot(root) else {
            throw LTXUnifiedAVGeneratorError.fullGenerationRequiresCompatibleModel(root)
        }
        let baseTimings = try await loadAudioToVideo(
            modelRoot: root,
            dtype: dtype,
            videoDecoder: videoDecoder,
            videoDecoderDType: videoDecoderDType
        )
        guard let transformer, let distilledLoRAURL else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }

        let adapterStart = ltxMonotonicSeconds()
        do {
            runtimeLoRAAdapter = try LTXRuntimeLoRAAdapter.install(
                url: distilledLoRAURL,
                into: transformer
            )
        } catch {
            await unload()
            throw error
        }
        let loraAdapterSeconds = ltxMonotonicSeconds() - adapterStart
        loadedForFullTwoStage = true
        loadedForReusableFullTwoStage = true
        return LTXLoadTimings(
            textEncoderSeconds: baseTimings.textEncoderSeconds,
            transformerSeconds: baseTimings.transformerSeconds,
            videoDecoderSeconds: baseTimings.videoDecoderSeconds,
            upsamplerSeconds: baseTimings.upsamplerSeconds,
            audioDecoderSeconds: baseTimings.audioDecoderSeconds,
            loraAdapterSeconds: loraAdapterSeconds,
            totalSeconds: ltxMonotonicSeconds() - totalStart
        )
    }

    @discardableResult
    public func loadAudioToVideo(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder decoderKind: LTXVideoDecoderKind = .diffusion,
        videoDecoderDType: DType? = nil
    ) async throws -> LTXLoadTimings {
        let totalStart = ltxMonotonicSeconds()
        await unload()
        let root = modelRoot.standardizedFileURL
        let isLTX25 = isLTX25FullModelRoot(root)
        guard isLTX23AudioToVideoModelRoot(root) || isLTX25 else {
            throw LTXUnifiedAVGeneratorError.audioToVideoRequiresCompatibleModel(root)
        }
        let ltx25Resources = LTX25Resources(rootURL: root)
        let transformerURL = isLTX25
            ? resolvedLTX25TransformerURL(resources: ltx25Resources, kind: .dev)
            : root.appendingPathComponent("transformer-dev.safetensors", isDirectory: false)
        let upsamplerURL = isLTX25
            ? ltx25Resources.spatialUpsamplerURL
            : root.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors", isDirectory: false)
        let audioVAEURL = isLTX25
            ? ltx25Resources.audioVAEURL
            : root.appendingPathComponent("audio_vae.safetensors", isDirectory: false)
        let loraURL = isLTX25
            ? ltx25Resources.distilledLoRAURL
            : root.appendingPathComponent(
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                isDirectory: false
            )
        for requiredURL in [transformerURL, upsamplerURL] where !FileManager.default.fileExists(atPath: requiredURL.path) {
            if requiredURL == transformerURL {
                throw LTXUnifiedAVGeneratorError.transformerWeightsMissing(requiredURL)
            }
            throw LTXUnifiedAVGeneratorError.upsamplerWeightsMissing(requiredURL)
        }
        guard FileManager.default.fileExists(atPath: audioVAEURL.path) else {
            throw LTXUnifiedAVGeneratorError.audioVAEWeightsMissing(audioVAEURL)
        }
        guard FileManager.default.fileExists(atPath: loraURL.path) else {
            throw LTXUnifiedAVGeneratorError.distilledLoRAMissing(loraURL)
        }

        let textStart = ltxMonotonicSeconds()
        let text = LTXGemmaTextEncoder()
        try await text.load(
            modelRoot: root,
            textEncoderRoot: isLTX25 ? nil : try resolveLTX23TextEncoderRoot(modelRoot: root),
            dtype: dtype,
            loadConnectorWeights: true
        )
        let textEncoderSeconds = ltxMonotonicSeconds() - textStart

        let transformerStart = ltxMonotonicSeconds()
        let model = LTXUnifiedAVTransformerV2()
        if isLTX25 {
            try loadLTX25TransformerWeights(url: transformerURL, model: model, dtype: dtype)
        } else {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: transformerURL,
                to: model,
                dtype: dtype,
                verify: .none,
                include: { $0.hasPrefix("transformer.") },
                mapper: { key, value in
                    mapLTX23UnifiedTransformerWeight(key: key, value: value, dtype: dtype)
                },
                batchSize: 24
            )
        }
        let transformerSeconds = ltxMonotonicSeconds() - transformerStart

        let videoDecoderStart = ltxMonotonicSeconds()
        let videoVAEURL = isLTX25
            ? ltx25Resources.videoVAEURL
            : root.appendingPathComponent("vae_decoder.safetensors", isDirectory: false)
        let sourceLayout: LTXTensorWeightLayout = isLTX25 ? .pytorch : .mlx
        let resolvedVideoDecoderDType = videoDecoderDType ?? dtype
        let vaeDecoder = try loadLTXVideoDecoder(
            weightsURL: videoVAEURL,
            dtype: resolvedVideoDecoderDType,
            sourceLayout: sourceLayout
        )
        let loadedDiffusionDecoder = isLTX25 && decoderKind == .diffusion
            ? try LTXDiffusionVideoDecoder.load(
                weightsURL: ltx25Resources.diffusionVideoVAEURL,
                dtype: resolvedVideoDecoderDType
            )
            : nil
        let videoDecoderSeconds = ltxMonotonicSeconds() - videoDecoderStart
        let upsamplerStart = ltxMonotonicSeconds()
        let latentUpsampler = try loadLTXVideoUpsampler(
            weightsURL: upsamplerURL,
            dtype: dtype,
            sourceLayout: sourceLayout
        )
        let upsamplerSeconds = ltxMonotonicSeconds() - upsamplerStart

        textEncoder = text
        transformer = model
        decoder = vaeDecoder
        diffusionDecoder = loadedDiffusionDecoder
        upsampler = latentUpsampler
        audioDecoder = nil
        vocoder = nil
        encoder = nil
        modelWeightsURL = transformerURL
        videoEncoderWeightsURL = isLTX25
            ? ltx25Resources.videoVAEURL
            : root.appendingPathComponent("vae_encoder.safetensors", isDirectory: false)
        videoVAEWeightLayout = sourceLayout
        videoVAEArchitecture = .ltx23Split
        loadedDType = dtype
        loadedRoot = root
        audioVAEWeightsURL = audioVAEURL
        distilledLoRAURL = loraURL
        runtimeLoRAAdapter = nil
        loadedForAudioToVideo = true
        loadedForFullTwoStage = false
        loadedForReusableFullTwoStage = false
        loadedForVideoOnlyOutput = false
        loadedForLTX25 = isLTX25
        twoStageGenerationConsumed = false
        return LTXLoadTimings(
            textEncoderSeconds: textEncoderSeconds,
            transformerSeconds: transformerSeconds,
            videoDecoderSeconds: videoDecoderSeconds,
            upsamplerSeconds: upsamplerSeconds,
            totalSeconds: ltxMonotonicSeconds() - totalStart
        )
    }

    @discardableResult
    public func load(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder: LTXVideoDecoderKind = .convolutional,
        videoDecoderDType: DType? = nil
    ) async throws -> LTXLoadTimings {
        try await loadStandalone(
            modelRoot: modelRoot,
            dtype: dtype,
            loadAudioOutput: true,
            videoDecoder: videoDecoder,
            videoDecoderDType: videoDecoderDType
        )
    }

    /// Loads the standalone distilled transformer for video-only output.
    ///
    /// Audio latents remain part of the joint AV denoising contract because
    /// audio-to-video cross attention influences every video block. This lane
    /// skips only the audio VAE and vocoder, which are not needed when callers
    /// do not request an audio waveform.
    @discardableResult
    public func loadVideoOnly(
        modelRoot: URL,
        dtype: DType = .bfloat16,
        videoDecoder: LTXVideoDecoderKind = .convolutional,
        videoDecoderDType: DType? = nil,
        loadTextEncoder: Bool = true
    ) async throws -> LTXLoadTimings {
        try await loadStandalone(
            modelRoot: modelRoot,
            dtype: dtype,
            loadAudioOutput: false,
            loadTextEncoder: loadTextEncoder,
            videoDecoder: videoDecoder,
            videoDecoderDType: videoDecoderDType
        )
    }

    private func loadStandalone(
        modelRoot: URL,
        dtype: DType,
        loadAudioOutput: Bool,
        loadTextEncoder: Bool = true,
        videoDecoder decoderKind: LTXVideoDecoderKind,
        videoDecoderDType: DType?
    ) async throws -> LTXLoadTimings {
        let totalStart = ltxMonotonicSeconds()
        await unload()
        let root = modelRoot.standardizedFileURL
        let isLTX23 = isLTX23SplitModelRoot(root)
        let isLTX25 = isLTX25ModelRoot(root)
        let ltx25Resources = LTX25Resources(rootURL: root)
        let splitTensorLayout: LTXTensorWeightLayout = isLTX23 ? .mlx : .pytorch
        let transformerURL: URL
        let upsamplerURL: URL
        if isLTX25 {
            transformerURL = resolvedLTX25TransformerURL(
                resources: ltx25Resources,
                kind: .distilled
            )
            upsamplerURL = ltx25Resources.spatialUpsamplerURL
        } else if isLTX23 {
            transformerURL = root.appendingPathComponent("transformer-distilled.safetensors", isDirectory: false)
            upsamplerURL = root.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors", isDirectory: false)
        } else {
            transformerURL = root.appendingPathComponent("ltx-2-19b-distilled.safetensors", isDirectory: false)
            upsamplerURL = root.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors", isDirectory: false)
        }
        let standaloneAudioVAEURL = ltxStandaloneAudioVAEWeightsURL(
            modelRoot: root,
            isLTX23: isLTX23,
            isLTX25: isLTX25,
            transformerURL: transformerURL
        )
        guard FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw LTXUnifiedAVGeneratorError.transformerWeightsMissing(transformerURL)
        }
        guard FileManager.default.fileExists(atPath: upsamplerURL.path) else {
            throw LTXUnifiedAVGeneratorError.upsamplerWeightsMissing(upsamplerURL)
        }

        let text: LTXGemmaTextEncoder?
        let textEncoderSeconds: Double
        if loadTextEncoder {
            let textStart = ltxMonotonicSeconds()
            let loadedText = LTXGemmaTextEncoder()
            let textEncoderRoot = try isLTX23 ? resolveLTX23TextEncoderRoot(modelRoot: root) : nil
            try await loadedText.load(
                modelRoot: root,
                textEncoderRoot: textEncoderRoot,
                dtype: dtype,
                loadConnectorWeights: true
            )
            text = loadedText
            textEncoderSeconds = ltxMonotonicSeconds() - textStart
        } else {
            text = nil
            textEncoderSeconds = 0
        }

        let transformerStart = ltxMonotonicSeconds()
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
        } else if isLTX25 {
            let packedModel = LTXUnifiedAVTransformerV2()
            try loadLTX25TransformerWeights(
                url: transformerURL,
                model: packedModel,
                dtype: dtype
            )
            model = packedModel
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
        let transformerSeconds = ltxMonotonicSeconds() - transformerStart

        let videoDecoderStart = ltxMonotonicSeconds()
        let vaeDecoder = LTXVideoDecoder(
            timestepConditioning: false,
            architecture: isLTX23 || isLTX25 ? .ltx23Split : .legacy
        )
        let videoDecoderURL: URL
        if isLTX25 {
            videoDecoderURL = ltx25Resources.videoVAEURL
        } else if isLTX23 {
            videoDecoderURL = root.appendingPathComponent("vae_decoder.safetensors", isDirectory: false)
        } else {
            videoDecoderURL = transformerURL
        }
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
                    || key == "per_channel_statistics.mean-of-means"
                    || key == "per_channel_statistics.std-of-means"
            },
            dtype: .float32
        )

        if let mean = decoderStats["latents_mean"]
            ?? decoderStats["vae.per_channel_statistics.mean-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.mean-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.mean"]
            ?? decoderStats["per_channel_statistics.mean-of-means"] {
            vaeDecoder.latentsMean = mean.asType(.float32)
        }
        if let std = decoderStats["latents_std"]
            ?? decoderStats["vae.per_channel_statistics.std-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.std-of-means"]
            ?? decoderStats["vae_decoder.per_channel_statistics.std"]
            ?? decoderStats["per_channel_statistics.std-of-means"] {
            vaeDecoder.latentsStd = std.asType(.float32)
        }

        let resolvedVideoDecoderDType = videoDecoderDType ?? dtype
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: videoDecoderURL,
            to: vaeDecoder,
            dtype: resolvedVideoDecoderDType,
            verify: .none,
            include: { key in
                key.hasPrefix("decoder.")
                    || key.hasPrefix("vae.decoder.")
                    || key.hasPrefix("vae_decoder.")
            },
            mapper: { key, value in
                mapLTXDecoderWeight(
                    key: key,
                    value: value,
                    dtype: resolvedVideoDecoderDType,
                    sourceLayout: splitTensorLayout
                )
            },
            batchSize: 24
        )
        let loadedDiffusionDecoder = isLTX25
            && decoderKind == .diffusion
            && FileManager.default.fileExists(atPath: ltx25Resources.diffusionVideoVAEURL.path)
            ? try LTXDiffusionVideoDecoder.load(
                weightsURL: ltx25Resources.diffusionVideoVAEURL,
                dtype: resolvedVideoDecoderDType
            )
            : nil
        let videoDecoderSeconds = ltxMonotonicSeconds() - videoDecoderStart

        let upsamplerStart = ltxMonotonicSeconds()
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
        let upsamplerSeconds = ltxMonotonicSeconds() - upsamplerStart

        var loadedAudioDecoder: LTXAudioDecoder?
        var loadedVocoder: LTXAudioVocoderBase?
        var audioDecoderSeconds = 0.0
        if loadAudioOutput {
            let audioDecoderStart = ltxMonotonicSeconds()
            let audioDecoder = LTXAudioDecoder()
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: standaloneAudioVAEURL,
                to: audioDecoder,
                dtype: .float32,
                verify: .none,
                include: { key in
                    key.hasPrefix("audio_vae.decoder.")
                        || key.hasPrefix("audio_vae.per_channel_statistics.")
                },
                mapper: { key, value in
                    mapAudioVaeDecoderWeight(
                        key: key,
                        value: value,
                        dtype: .float32,
                        sourceLayout: splitTensorLayout
                    )
                },
                batchSize: 24
            )

            let vocoderURL: URL
            if isLTX25 {
                vocoderURL = ltx25Resources.audioVAEURL
            } else if isLTX23 {
                vocoderURL = root.appendingPathComponent("vocoder.safetensors", isDirectory: false)
            } else {
                vocoderURL = transformerURL
            }
            let vocoderMetadata = try SafetensorsStreamingLoader.metadata(url: vocoderURL)
            let vocoderFlavor = detectLTXVocoderFlavor(keys: vocoderMetadata.keys)
            let vocoder: LTXAudioVocoderBase
            switch vocoderFlavor {
            case .legacy:
                vocoder = LTXVocoder()
            case .bandwidthExtension:
                let config = isLTX25
                    ? try loadLTXPackedBWEVocoderConfig(weightsURL: vocoderURL)
                    : try loadLTXBWEVocoderConfig(modelRoot: root)
                if let config {
                    vocoder = LTXVocoderWithBWE(config: config)
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
            loadedAudioDecoder = audioDecoder
            loadedVocoder = vocoder
            audioDecoderSeconds = ltxMonotonicSeconds() - audioDecoderStart
        }

        self.textEncoder = text
        self.transformer = model
        self.decoder = vaeDecoder
        self.diffusionDecoder = loadedDiffusionDecoder
        self.upsampler = latentUpsampler
        self.audioDecoder = loadedAudioDecoder
        self.vocoder = loadedVocoder
        self.encoder = nil
        self.modelWeightsURL = transformerURL
        self.videoEncoderWeightsURL = isLTX23
            ? root.appendingPathComponent("vae_encoder.safetensors", isDirectory: false)
            : (isLTX25 ? ltx25Resources.videoVAEURL : transformerURL)
        self.videoVAEWeightLayout = splitTensorLayout
        self.videoVAEArchitecture = isLTX23 || isLTX25 ? .ltx23Split : .legacy
        self.loadedDType = dtype
        self.loadedRoot = root
        self.audioVAEWeightsURL = standaloneAudioVAEURL
        self.loadedForVideoOnlyOutput = !loadAudioOutput
        self.loadedForLTX25 = isLTX25
        return LTXLoadTimings(
            textEncoderSeconds: textEncoderSeconds,
            transformerSeconds: transformerSeconds,
            videoDecoderSeconds: videoDecoderSeconds,
            upsamplerSeconds: upsamplerSeconds,
            audioDecoderSeconds: audioDecoderSeconds,
            totalSeconds: ltxMonotonicSeconds() - totalStart
        )
    }

    public func unload() async {
        runtimeLoRAAdapter?.setActive(false)
        runtimeUserLoRAAdapters.forEach { $0.setActive(false) }
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
        runtimeAudioLoRAAdapters.forEach { $0.setActive(false) }
        if let textEncoder {
            await textEncoder.unload()
        }
        textEncoder = nil
        transformer = nil
        audioOnlyTransformer = nil
        decoder = nil
        diffusionDecoder = nil
        encoder = nil
        upsampler = nil
        temporalUpsampler = nil
        audioDecoder = nil
        vocoder = nil
        modelWeightsURL = nil
        videoEncoderWeightsURL = nil
        videoVAEWeightLayout = .pytorch
        videoVAEArchitecture = .legacy
        loadedRoot = nil
        audioVAEWeightsURL = nil
        distilledLoRAURL = nil
        runtimeLoRAAdapter = nil
        runtimeUserLoRAAdapters = []
        runtimeDetailingLoRAAdapters = []
        runtimeAudioLoRAAdapters = []
        loadedForAudioToVideo = false
        loadedForFullTwoStage = false
        loadedForReusableFullTwoStage = false
        loadedForVideoOnlyOutput = false
        loadedForLTX25 = false
        loadedForDFR = false
        twoStageGenerationConsumed = false
        promptEmbeddingCache.removeAll(keepingCapacity: false)
        Memory.clearCache()
    }

    private func installRuntimeLoRAsIfNeeded(
        user: [LTXLoRAConfiguration],
        detailing: [LTXLoRAConfiguration]
    ) throws {
        guard !user.isEmpty || !detailing.isEmpty else { return }
        guard let transformer else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        if runtimeUserLoRAAdapters.isEmpty {
            runtimeUserLoRAAdapters = try user.map { configuration in
                guard FileManager.default.fileExists(atPath: configuration.url.path) else {
                    throw LTXUnifiedAVGeneratorError.loraMissing(configuration.url)
                }
                return try LTXRuntimeLoRAAdapter.install(
                    url: configuration.url,
                    into: transformer,
                    strength: configuration.strength,
                    expectedPairCount: nil
                )
            }
        }
        if runtimeDetailingLoRAAdapters.isEmpty {
            runtimeDetailingLoRAAdapters = try detailing.map { configuration in
                guard FileManager.default.fileExists(atPath: configuration.url.path) else {
                    throw LTXUnifiedAVGeneratorError.loraMissing(configuration.url)
                }
                return try LTXRuntimeLoRAAdapter.install(
                    url: configuration.url,
                    into: transformer,
                    strength: configuration.strength,
                    expectedPairCount: nil
                )
            }
        }
    }

    private func loadFullTextEncoderIfNeeded() async throws {
        guard textEncoder == nil else { return }
        guard let loadedRoot else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let text = LTXGemmaTextEncoder()
        try await text.load(
            modelRoot: loadedRoot,
            textEncoderRoot: loadedForLTX25 ? nil : try resolveLTX23TextEncoderRoot(modelRoot: loadedRoot),
            dtype: loadedDType,
            loadConnectorWeights: true
        )
        textEncoder = text
    }

    private func cachedPromptEmbeddings(
        prompt: String,
        maxLength: Int
    ) async throws -> (embeddings: LTXCachedPromptEmbeddings, cacheHit: Bool) {
        let key = LTXPromptEmbeddingCacheKey(prompt: prompt, maxLength: maxLength)
        if let cached = promptEmbeddingCache.value(for: key) {
            return (cached, true)
        }
        guard let textEncoder else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let encoding = try await textEncoder.encode(prompt: prompt, maxLength: maxLength)
        let cached = LTXCachedPromptEmbeddings(
            video: encoding.videoEmbeddings,
            audio: encoding.audioEmbeddings
        )
        if let audio = cached.audio {
            MLX.eval(cached.video, audio)
        } else {
            MLX.eval(cached.video)
        }
        promptEmbeddingCache.insert(cached, for: key)
        return (cached, false)
    }

    private func loadEncoderIfNeeded() throws {
        if encoder != nil {
            return
        }
        guard let modelWeightsURL else {
            throw LTXUnifiedAVGeneratorError.encoderNotLoaded
        }
        let encoderURL = videoEncoderWeightsURL ?? modelWeightsURL

        let vaeEncoder = LTXVideoEncoder(architecture: videoVAEArchitecture)
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

    public func generateAudioToVideo(
        options: LTXAudioToVideoGenerationOptions
    ) async throws -> LTXAudioToVideoGenerationResult {
        let totalStart = ltxMonotonicSeconds()
        var preparationSeconds = 0.0
        var textEncodingSeconds = 0.0
        var textEncoderReloadSeconds = 0.0
        var stage1DenoiseSeconds = 0.0
        var loraFusionSeconds = 0.0
        var upsampleSeconds = 0.0
        var stage2DenoiseSeconds = 0.0
        var videoDecodeSeconds = 0.0
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let negativePrompt = options.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !negativePrompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        guard options.width > 0, options.height > 0,
              options.width % 64 == 0, options.height % 64 == 0 else {
            throw LTXUnifiedAVGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.fps.isFinite, options.fps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameRate(options.fps)
        }
        guard options.inferenceSteps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidInferenceSteps(options.inferenceSteps)
        }
        guard options.audioStartTime.isFinite, options.audioStartTime >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidAudioStartTime(options.audioStartTime)
        }
        if let audioMaxDuration = options.audioMaxDuration,
           !audioMaxDuration.isFinite || audioMaxDuration <= 0 {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "audioMaxDuration must be finite and positive."
            )
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.imageStrength)
        }
        guard options.endImageStrength >= 0, options.endImageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.endImageStrength)
        }
        let imageConditionings = ltx25ImageConditionings(options: options)
        let hasEXRInput = imageConditionings.contains { MediaHDRImageIO.isEXR($0.imageURL) }
        if hasEXRInput, options.hdrColorSpace == nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "EXR conditioning requires an explicit HDR color space."
            )
        }
        if options.hdrColorSpace != nil, !loadedForLTX25 {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "HDR audio-to-video requires an official LTX-2.5 checkpoint."
            )
        }
        guard options.generatedKeyframeCount >= 0,
              options.generatedKeyframeCount == 0 || options.generatedKeyframeIndices.isEmpty else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Use a generated-keyframe count or explicit indices, not both."
            )
        }
        let generatedIndices = options.generatedKeyframeCount > 0
            ? try ltxEvenlySpacedGeneratedKeyframePositions(
                count: options.generatedKeyframeCount,
                numFrames: options.numFrames
            )
            : options.generatedKeyframeIndices
        let requestsLTX25Conditioning = !options.imageConditionings.isEmpty
            || !generatedIndices.isEmpty
        guard loadedForLTX25 || !requestsLTX25Conditioning else {
            throw LTXUnifiedAVGeneratorError.ltx25ConditioningRequiresLTX25
        }
        if loadedForLTX25 {
            for input in imageConditionings {
                guard input.pixelFrameIndex >= 0, input.pixelFrameIndex < options.numFrames else {
                    throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(input.pixelFrameIndex)
                }
                guard input.strength >= 0, input.strength <= 1 else {
                    throw LTXUnifiedAVGeneratorError.invalidImageStrength(input.strength)
                }
                guard FileManager.default.fileExists(atPath: input.imageURL.path) else {
                    throw LTXUnifiedAVGeneratorError.imageNotFound(input.imageURL)
                }
            }
        }
        let generatedIndicesAreOrdered = zip(
            generatedIndices,
            generatedIndices.dropFirst()
        ).allSatisfy(<)
        guard generatedIndicesAreOrdered,
              generatedIndices.allSatisfy({ $0 >= 0 && $0 < options.numFrames }) else {
            throw LTXUnifiedAVGeneratorError.invalidGeneratedKeyframes(generatedIndices)
        }
        let usesLTX25TokenState = loadedForLTX25
            && (!imageConditionings.isEmpty || !generatedIndices.isEmpty)
        guard FileManager.default.fileExists(atPath: options.audioURL.path) else {
            throw LTXUnifiedAVGeneratorError.audioSourceNotFound(options.audioURL)
        }
        let usesReusableFullTwoStage = loadedForReusableFullTwoStage
        if usesReusableFullTwoStage {
            let reloadStart = ltxMonotonicSeconds()
            try await loadFullTextEncoderIfNeeded()
            textEncoderReloadSeconds = ltxMonotonicSeconds() - reloadStart
        }
        guard loadedForAudioToVideo,
              let textEncoder,
              let transformer,
              let decoder,
              let upsampler,
              let audioVAEWeightsURL,
              let distilledLoRAURL else {
            throw LTXUnifiedAVGeneratorError.audioToVideoGeneratorNotLoaded
        }
        if let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 {
            transformerV2.execution = options.transformerExecution
        }
        guard !twoStageGenerationConsumed else {
            throw LTXUnifiedAVGeneratorError.audioToVideoRequiresReload
        }
        if usesReusableFullTwoStage, runtimeLoRAAdapter == nil {
            throw LTXUnifiedAVGeneratorError.audioToVideoGeneratorNotLoaded
        }
        let parityIO = LTXAudioToVideoParityIO()

        let inputPreparationStart = ltxMonotonicSeconds()
        let duration = Double(options.numFrames) / Double(options.fps)
        let audioDecodeDuration = options.audioMaxDuration ?? duration
        let audioMetadata = try MediaAudioIO.probe(options.audioURL)
        try validateLTXAudioSegment(
            metadata: audioMetadata,
            startTime: options.audioStartTime,
            duration: audioDecodeDuration
        )

        let sourceAudio = try decodeExactStereoAudioSegment(
            url: options.audioURL,
            startTime: options.audioStartTime,
            duration: audioDecodeDuration,
            sampleRate: audioMetadata.sampleRate
        )
        let conditioningAudio = try decodeExactStereoAudioSegment(
            url: options.audioURL,
            startTime: options.audioStartTime,
            duration: audioDecodeDuration,
            sampleRate: LTXAudioMelProcessor.sampleRate
        )
        preparationSeconds += ltxMonotonicSeconds() - inputPreparationStart

        twoStageGenerationConsumed = true
        defer {
            if usesReusableFullTwoStage {
                runtimeLoRAAdapter?.setActive(false)
                twoStageGenerationConsumed = false
            }
        }
        let textEncodingStart = ltxMonotonicSeconds()
        let positiveEncoding = try await textEncoder.encode(
            prompt: prompt,
            maxLength: options.maxTextLength
        )
        let negativeEncoding = try await textEncoder.encode(
            prompt: negativePrompt,
            maxLength: options.maxTextLength
        )
        let positiveVideoContext = positiveEncoding.videoEmbeddings
        let negativeVideoContext = negativeEncoding.videoEmbeddings
        guard let positiveAudioContext = positiveEncoding.audioEmbeddings else {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }
        MLX.eval(positiveVideoContext, negativeVideoContext, positiveAudioContext)
        try parityIO.save(positiveVideoContext, suffix: "a2vid_positive_video_context")
        try parityIO.save(negativeVideoContext, suffix: "a2vid_negative_video_context")
        try parityIO.save(positiveAudioContext, suffix: "a2vid_positive_audio_context")
        await textEncoder.unload()
        self.textEncoder = nil
        Memory.clearCache()
        textEncodingSeconds = textEncoderReloadSeconds + ltxMonotonicSeconds() - textEncodingStart

        let latentPreparationStart = ltxMonotonicSeconds()
        let audioFrameCount = computeAudioLatentFrameCount(
            videoFrames: options.numFrames,
            fps: options.fps
        )
        let spectrogram = LTXAudioMelProcessor().extract(
            channels: planarAudioChannels(conditioningAudio)
        )
        try parityIO.save(spectrogram, suffix: "a2vid_audio_mel")
        var audioLatents = try encodeLTX23AudioLatents(
            spectrogram: spectrogram,
            requiredFrameCount: audioFrameCount,
            weightsURL: audioVAEWeightsURL,
            dtype: loadedDType,
            sourceLayout: loadedForLTX25 ? .pytorch : .mlx
        )
        audioLatents = audioLatents.asType(positiveVideoContext.dtype)
        MLX.eval(audioLatents)
        Memory.clearCache()

        let latentFrames = 1 + ((options.numFrames - 1) / 8)
        let stage1Height = options.height / 2 / 32
        let stage1Width = options.width / 2 / 32
        let stage2Height = options.height / 32
        let stage2Width = options.width / 32
        let modelDType = positiveVideoContext.dtype
        try parityIO.save(audioLatents, suffix: "a2vid_audio_latents")

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?
        var stage1TokenState: LTX25VideoTokenState?
        var stage2TokenState: LTX25VideoTokenState?
        var generatedKeyframeLatents: MLXArray?
        var stage2LTX25ImageLatents: [MLXArray] = []
        var videoLatents: MLXArray

        let baseStage1VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1Height,
            width: stage1Width,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(options.fps),
            causalFix: true
        )
        if usesLTX25TokenState {
            if !imageConditionings.isEmpty {
                try loadEncoderIfNeeded()
            }
            videoLatents = MLX.zeros(
                [1, 128, latentFrames, stage1Height, stage1Width],
                dtype: modelDType
            )
            var state = try makeConditionedLTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage1VideoPositions,
                imageConditionings: imageConditionings,
                generatedKeyframeIndices: generatedIndices,
                initialGeneratedKeyframes: nil,
                encoder: encoder,
                pixelWidth: options.width / 2,
                pixelHeight: options.height / 2,
                fps: options.fps,
                hdrColorSpace: options.hdrColorSpace
            )
            let noise = try parityIO.resolveNoise(
                stage: .stage1,
                generated: MLXRandom.normal(state.latent.shape).asType(modelDType)
            )
            state.addNoise(noise, scale: 1)
            MLX.eval(state.latent)
            stage1TokenState = state
            if !imageConditionings.isEmpty {
                guard let imageEncoder = encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                for input in imageConditionings {
                    let image = try loadImageForEncoding(
                        url: input.imageURL,
                        width: options.width,
                        height: options.height,
                        dtype: modelDType,
                        hdrColorSpace: options.hdrColorSpace,
                        crf: input.crf ?? 18
                    )
                    let encoded = imageEncoder.encode(image: image)
                    MLX.eval(encoded)
                    stage2LTX25ImageLatents.append(encoded)
                }
            }
            encoder = nil
            Memory.clearCache()
        } else if let sourceImageURL = options.sourceImageURL {
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw LTXUnifiedAVGeneratorError.imageNotFound(sourceImageURL)
            }
            if let endImageURL = options.endImageURL,
               !FileManager.default.fileExists(atPath: endImageURL.path) {
                throw LTXUnifiedAVGeneratorError.imageNotFound(endImageURL)
            }
            guard options.imageFrameIndex < latentFrames else {
                throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
            }

            try loadEncoderIfNeeded()
            do {
                guard let imageEncoder = encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                let stage1Image = try loadImageForEncoding(
                    url: sourceImageURL,
                    width: options.width / 2,
                    height: options.height / 2,
                    dtype: modelDType
                )
                let stage1ImageLatent = imageEncoder.encode(image: stage1Image)
                let stage2Image = try loadImageForEncoding(
                    url: sourceImageURL,
                    width: options.width,
                    height: options.height,
                    dtype: modelDType
                )
                let stage2ImageLatent = imageEncoder.encode(image: stage2Image)
                stage2ConditioningLatent = stage2ImageLatent

                var stage1EndImageLatent: MLXArray?
                if let endImageURL = options.endImageURL {
                    let stage1EndImage = try loadImageForEncoding(
                        url: endImageURL,
                        width: options.width / 2,
                        height: options.height / 2,
                        dtype: modelDType
                    )
                    stage1EndImageLatent = imageEncoder.encode(image: stage1EndImage)
                    let stage2EndImage = try loadImageForEncoding(
                        url: endImageURL,
                        width: options.width,
                        height: options.height,
                        dtype: modelDType
                    )
                    stage2EndConditioningLatent = imageEncoder.encode(image: stage2EndImage)
                }

                var state = applyLatentConditioning(
                    baseLatent: MLX.zeros(
                        [1, 128, latentFrames, stage1Height, stage1Width],
                        dtype: modelDType
                    ),
                    conditionedLatent: stage1ImageLatent,
                    frameIndex: options.imageFrameIndex,
                    strength: options.imageStrength,
                    endConditionedLatent: stage1EndImageLatent,
                    endFrameIndex: -1,
                    endStrength: options.endImageStrength
                )
                let noise = try parityIO.resolveNoise(
                    stage: .stage1,
                    generated: MLXRandom.normal(state.latent.shape).asType(modelDType)
                )
                try parityIO.save(noise, suffix: "a2vid_stage1_noise")
                let sigma = MLXArray(1).asType(modelDType)
                let scaledMask = state.denoiseMask * sigma
                state.latent = noise * scaledMask + state.latent * (MLXArray(1).asType(modelDType) - scaledMask)
                videoLatents = state.latent
                stage1ConditioningState = state
                MLX.eval(videoLatents, stage2ImageLatent)
                if let stage2EndConditioningLatent {
                    MLX.eval(stage2EndConditioningLatent)
                }
            }
            encoder = nil
            Memory.clearCache()
        } else {
            videoLatents = try parityIO.resolveNoise(
                stage: .stage1,
                generated: MLXRandom.normal(
                    [1, 128, latentFrames, stage1Height, stage1Width]
                ).asType(modelDType)
            )
            try parityIO.save(videoLatents, suffix: "a2vid_stage1_noise")
            MLX.eval(videoLatents)
        }

        let audioPositions = createAudioPositionGrid(
            batchSize: 1,
            audioFrames: audioFrameCount
        )
        let audioRope = precomputeSplitRope(
            positions: audioPositions,
            dim: 2_048,
            theta: 10_000,
            maxPos: [20],
            numHeads: 32
        )
        let stage1Ropes = stage1TokenState.map {
            makeLTXAudioToVideoVideoRopes(positions: $0.positions)
        } ?? makeLTXAudioToVideoVideoRopes(
            latentFrames: latentFrames,
            height: stage1Height,
            width: stage1Width,
            fps: options.fps
        )
        let stage1Sigmas = LTX2DiffusionScheduler.sigmas(steps: options.inferenceSteps)
        preparationSeconds += ltxMonotonicSeconds() - latentPreparationStart
        let stage1DenoiseStart = ltxMonotonicSeconds()
        if let tokenState = stage1TokenState {
            let result = denoiseFrozenLTX25AudioVideoTokenLoop(
                videoState: tokenState,
                audioLatents: audioLatents,
                videoRope: stage1Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage1Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: negativeVideoContext,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                guidance: options.guidance
            )
            stage1TokenState = result
            videoLatents = result.mainLatent()
            generatedKeyframeLatents = result.generatedKeyframes()
        } else {
            videoLatents = try denoiseFrozenAudioVideoLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage1Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage1Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: negativeVideoContext,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                videoConditioning: stage1ConditioningState,
                guidance: options.guidance,
                debugLabel: "a2vid_stage1"
            ).video
        }
        MLX.eval(videoLatents)
        stage1DenoiseSeconds = ltxMonotonicSeconds() - stage1DenoiseStart
        try parityIO.save(videoLatents, suffix: "a2vid_stage1_output")

        let upsampleStart = ltxMonotonicSeconds()
        videoLatents = upsampleLatents(
            videoLatents,
            upsampler: upsampler,
            latentMean: decoder.latentsMean,
            latentStd: decoder.latentsStd
        )
        MLX.eval(videoLatents)
        let stage2InitialGeneratedKeyframes = generatedKeyframeLatents.map {
            upsampleLatents(
                $0,
                upsampler: upsampler,
                latentMean: decoder.latentsMean,
                latentStd: decoder.latentsStd
            )
        }
        if let stage2InitialGeneratedKeyframes {
            MLX.eval(stage2InitialGeneratedKeyframes)
        }
        upsampleSeconds = ltxMonotonicSeconds() - upsampleStart
        try parityIO.save(videoLatents, suffix: "a2vid_upsampled_latents")

        let stage2Sigma = STAGE2Sigmas[0]
        let baseStage2VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2Height,
            width: stage2Width,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(options.fps),
            causalFix: true
        )
        if usesLTX25TokenState {
            var state = LTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage2VideoPositions
            )
            for (input, encoded) in zip(imageConditionings, stage2LTX25ImageLatents) {
                state.applyImageLatent(
                    encoded,
                    pixelFrameIndex: input.pixelFrameIndex,
                    strength: input.strength,
                    fps: options.fps
                )
            }
            if !generatedIndices.isEmpty {
                state.appendGeneratedKeyframeSlots(
                    pixelFrameIndices: generatedIndices,
                    initialKeyframes: stage2InitialGeneratedKeyframes,
                    fps: options.fps
                )
            }
            let noise = try parityIO.resolveNoise(
                stage: .stage2,
                generated: MLXRandom.normal(state.latent.shape).asType(modelDType)
            )
            state.addNoise(noise, scale: stage2Sigma)
            MLX.eval(state.latent)
            stage2TokenState = state
            videoLatents = state.mainLatent()
        } else if let conditionedLatent = stage2ConditioningLatent {
            var state = applyLatentConditioning(
                baseLatent: videoLatents,
                conditionedLatent: conditionedLatent,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage2EndConditioningLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
            let noise = try parityIO.resolveNoise(
                stage: .stage2,
                generated: MLXRandom.normal(videoLatents.shape).asType(modelDType)
            )
            try parityIO.save(noise, suffix: "a2vid_stage2_noise")
            let scaledMask = state.denoiseMask * MLXArray(stage2Sigma).asType(modelDType)
            state.latent = noise * scaledMask + state.latent * (MLXArray(1).asType(modelDType) - scaledMask)
            videoLatents = state.latent
            stage2ConditioningState = state
        } else {
            let noise = try parityIO.resolveNoise(
                stage: .stage2,
                generated: MLXRandom.normal(videoLatents.shape).asType(modelDType)
            )
            try parityIO.save(noise, suffix: "a2vid_stage2_noise")
            videoLatents = MLXArray(stage2Sigma).asType(modelDType) * noise
                + MLXArray(1 - stage2Sigma).asType(modelDType) * videoLatents
        }
        MLX.eval(videoLatents)
        try parityIO.save(videoLatents, suffix: "a2vid_stage2_input")

        let loraFusionStart = ltxMonotonicSeconds()
        if usesReusableFullTwoStage {
            runtimeLoRAAdapter?.setActive(true)
        } else {
            try LTXStreamingLoRAFuser.fuse(
                url: distilledLoRAURL,
                into: transformer,
                debugOutputPrefix: parityIO.outputPrefix
            )
        }
        loraFusionSeconds = ltxMonotonicSeconds() - loraFusionStart

        let stage2Ropes = stage2TokenState.map {
            makeLTXAudioToVideoVideoRopes(positions: $0.positions)
        } ?? makeLTXAudioToVideoVideoRopes(
            latentFrames: latentFrames,
            height: stage2Height,
            width: stage2Width,
            fps: options.fps
        )
        let stage2DenoiseStart = ltxMonotonicSeconds()
        if let tokenState = stage2TokenState {
            let result = denoiseFrozenLTX25AudioVideoTokenLoop(
                videoState: tokenState,
                audioLatents: audioLatents,
                videoRope: stage2Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage2Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: nil,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: STAGE2Sigmas,
                guidance: nil
            )
            stage2TokenState = result
            videoLatents = result.mainLatent()
            generatedKeyframeLatents = result.generatedKeyframes()
        } else {
            videoLatents = try denoiseFrozenAudioVideoLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage2Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage2Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: nil,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: STAGE2Sigmas,
                videoConditioning: stage2ConditioningState,
                guidance: nil,
                debugLabel: "a2vid_stage2"
            ).video
        }
        MLX.eval(videoLatents)
        if usesReusableFullTwoStage {
            runtimeLoRAAdapter?.setActive(false)
        }
        stage2DenoiseSeconds = ltxMonotonicSeconds() - stage2DenoiseStart
        try parityIO.save(videoLatents, suffix: "a2vid_stage2_output")

        let videoDecodeStart = ltxMonotonicSeconds()
        let frames: MLXArray
        let hdrOutput: LTXHDROutputFrames?
        if let diffusionDecoder {
            let decoded = try diffusionDecoder.decode(sample: videoLatents, seed: options.seed)
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    decoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(decoded)
            }
        } else if let tiling = selectDecodeTilingConfig(
            width: options.width,
            height: options.height,
            numFrames: options.numFrames,
            fps: options.fps
        ) {
            if let colorSpace = options.hdrColorSpace {
                let decoded = decodeWithTilingRaw(
                    decoder: decoder,
                    latents: videoLatents,
                    spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                    spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                    temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                    temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                    spatialScale: 32,
                    temporalScale: 8
                )
                let output = LTXHDRColorPipeline.decode(
                    decoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
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
            }
        } else {
            let decoded = decoder.decode(sample: videoLatents, timestep: nil)
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    decoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(decoded)
            }
        }
        MLX.eval(frames)
        videoDecodeSeconds = ltxMonotonicSeconds() - videoDecodeStart

        return LTXAudioToVideoGenerationResult(
            frames: frames,
            hdrOutput: hdrOutput,
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            sourceAudio: sourceAudio,
            generatedKeyframeLatents: generatedKeyframeLatents,
            generatedKeyframeIndices: generatedKeyframeLatents == nil ? [] : generatedIndices,
            timings: LTXGenerationTimings(
                textEncodingSeconds: textEncodingSeconds,
                preparationSeconds: preparationSeconds,
                stage1DenoiseSeconds: stage1DenoiseSeconds,
                loraFusionSeconds: loraFusionSeconds,
                upsampleSeconds: upsampleSeconds,
                stage2DenoiseSeconds: stage2DenoiseSeconds,
                videoDecodeSeconds: videoDecodeSeconds,
                totalSeconds: ltxMonotonicSeconds() - totalStart
            )
        )
    }

    public func generate(
        options: LTXUnifiedAVGenerationOptions
    ) async throws -> LTXUnifiedAVGenerationResult {
        guard !loadedForVideoOnlyOutput else {
            throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
        }
        let output = try await generate(options: options, decodeAudio: true)
        guard let audioWaveform = output.audioWaveform,
              let audioSampleRate = output.audioSampleRate else {
            throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
        }
        return LTXUnifiedAVGenerationResult(
            frames: output.frames,
            videoLatents: output.videoLatents,
            audioLatents: output.audioLatents,
            audioWaveform: audioWaveform,
            audioSampleRate: audioSampleRate,
            hdrOutput: output.hdrOutput,
            generatedKeyframeLatents: output.generatedKeyframeLatents,
            generatedKeyframeIndices: output.generatedKeyframeIndices,
            playbackFPS: output.playbackFPS,
            timings: output.timings
        )
    }

    public func generateVideoOnly(
        options: LTXUnifiedAVGenerationOptions
    ) async throws -> LTXUnifiedVideoGenerationResult {
        let output = try await generate(options: options, decodeAudio: false)
        return LTXUnifiedVideoGenerationResult(
            frames: output.frames,
            hdrOutput: output.hdrOutput,
            videoLatents: output.videoLatents,
            generatedKeyframeLatents: output.generatedKeyframeLatents,
            generatedKeyframeIndices: output.generatedKeyframeIndices,
            playbackFPS: output.playbackFPS,
            timings: output.timings
        )
    }

    private func generate(
        options: LTXUnifiedAVGenerationOptions,
        decodeAudio: Bool
    ) async throws -> LTXUnifiedGenerationOutput {
        let totalStart = ltxMonotonicSeconds()
        var textEncodingSeconds = 0.0
        var textEncoderReloadSeconds = 0.0
        var promptCacheHits = 0
        var promptCacheMisses = 0
        let guidanceProjectionCacheMetrics = LTXGuidanceProjectionCacheMetrics()
        let teaCacheController = options.teaCache.map {
            LTXTeaCacheController(configuration: $0, sampler: options.sampler.mode)
        }
        var preparationSeconds = 0.0
        var stage1DenoiseSeconds = 0.0
        var loraFusionSeconds = 0.0
        var upsampleSeconds = 0.0
        var stage2DenoiseSeconds = 0.0
        var videoDecodeSeconds = 0.0
        var audioDecodeSeconds = 0.0
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        let usesDFR = options.dfr != nil
        let stage1DistilledLoRAStrength = usesDFR && options.distilledLoRAStrengthStage1 == 0
            ? 1
            : options.distilledLoRAStrengthStage1
        let usesHDRICLoRA = options.hdrICLoRA != nil
        let resolutionMultiple = options.retake == nil ? 64 : 32
        let generationWidth = usesHDRICLoRA
            ? max(64, ((options.width + 63) / 64) * 64)
            : options.width
        let generationHeight = usesHDRICLoRA
            ? max(64, ((options.height + 63) / 64) * 64)
            : options.height
        guard options.width > 0,
              options.height > 0,
              usesHDRICLoRA || (
                  options.width.isMultiple(of: resolutionMultiple)
                      && options.height.isMultiple(of: resolutionMultiple)
              ) else {
            throw LTXUnifiedAVGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.vaeSpatialTileOverlap >= 0,
              options.vaeSpatialTileSize.map({ $0 > options.vaeSpatialTileOverlap }) ?? true else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "The VAE spatial tile must be positive and larger than its overlap."
            )
        }
        if usesDFR, options.retake != nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Retake and DFR cannot run in the same generation."
            )
        }
        if usesDFR,
           options.generatedKeyframeCount > 0 || !options.generatedKeyframeIndices.isEmpty {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "DFR derives its generated-keyframe slots from the official segment grid."
            )
        }
        if usesDFR, !options.referenceVideos.isEmpty {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "DFR detailing references are supplied through detailing LoRAs, not IC-LoRA reference videos."
            )
        }
        if options.dubIt != nil, options.retake != nil || usesDFR {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Dub-It cannot be combined with Retake or DFR."
            )
        }
        if usesHDRICLoRA {
            guard options.hdrColorSpace != nil, loadedForLTX25, !decodeAudio else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "HDR IC-LoRA is an LTX-2.5 video-only HDR pipeline."
                )
            }
            guard !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  options.sourceImageURL == nil,
                  options.endImageURL == nil,
                  options.imageConditionings.isEmpty,
                  options.generatedKeyframeCount == 0,
                  options.generatedKeyframeIndices.isEmpty,
                  !options.referenceVideos.isEmpty,
                  !options.loras.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "HDR IC-LoRA accepts one or more reference videos and HDR LoRA adapters only."
                )
            }
            for phase in options.hdrICLoRA!.stage2Phases {
                _ = try validatedLTXSigmaSchedule(phase.sigmas)
            }
        }
        if let embeddingsURL = options.precomputedTextEmbeddingsURL {
            guard usesHDRICLoRA else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Precomputed text embeddings are currently accepted by the dedicated HDR IC-LoRA pipeline."
                )
            }
            guard FileManager.default.fileExists(atPath: embeddingsURL.path) else {
                throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(embeddingsURL)
            }
        }
        if options.skipStage2 {
            guard !usesHDRICLoRA,
                  !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  !options.referenceVideos.isEmpty,
                  !loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Skipping stage two is available for distilled IC-LoRA reference generation."
                )
            }
        }
        if usesDFR, !loadedForDFR {
            if let loadedRoot {
                throw LTXUnifiedAVGeneratorError.dfrRequiresLTX25Full(loadedRoot)
            }
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let dfrCanvas = try options.dfr.map { _ in
            try LTX25DFRLayout.resolveCanvas(frameCount: options.numFrames)
        }
        let generationFrameCount = dfrCanvas?.frameCount
            ?? (options.hdrICLoRA?.highQuality == true ? 2 * options.numFrames - 1 : options.numFrames)
        guard options.fps.isFinite, options.fps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameRate(options.fps)
        }
        guard options.inferenceSteps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidInferenceSteps(options.inferenceSteps)
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard stage1DistilledLoRAStrength.isFinite,
              options.distilledLoRAStrengthStage2.isFinite else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Distilled LoRA strengths must be finite."
            )
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.imageStrength)
        }
        guard options.endImageStrength >= 0, options.endImageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.endImageStrength)
        }
        let imageConditionings = ltx25ImageConditionings(options: options)
        var referenceVideos = options.referenceVideos
        if !referenceVideos.isEmpty {
            _ = try ltxLoRAReferenceScaleConfiguration(options.loras)
        }
        let hasEXRInput = imageConditionings.contains { MediaHDRImageIO.isEXR($0.imageURL) }
            || referenceVideos.contains { MediaHDRImageIO.isEXRDirectory($0.videoURL) }
            || options.retake.map { MediaHDRImageIO.isEXRDirectory($0.sourceVideoURL) } == true
        if hasEXRInput, options.hdrColorSpace == nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "EXR conditioning requires an explicit HDR color space."
            )
        }
        if options.hdrColorSpace != nil, !loadedForLTX25 {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "HDR generation requires an official LTX-2.5 checkpoint."
            )
        }
        if options.hdrColorSpace != nil, options.dubIt != nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Dub-It does not support HDR output."
            )
        }
        if let dubIt = options.dubIt {
            guard loadedForLTX25, !loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.dubItRequiresLTX25(loadedRoot)
            }
            guard options.referenceVideos.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Dub-It owns its single reference video; do not also supply referenceVideos."
                )
            }
            guard options.loras.count == 1 else {
                throw LTXUnifiedAVGeneratorError.dubItRequiresOneICLoRA(options.loras.count)
            }
            let icLoRA = options.loras[0]
            guard FileManager.default.fileExists(atPath: icLoRA.url.path) else {
                throw LTXUnifiedAVGeneratorError.loraMissing(icLoRA.url)
            }
            guard FileManager.default.fileExists(atPath: dubIt.referenceVideoURL.path) else {
                throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(dubIt.referenceVideoURL)
            }
            guard MediaVideoIO.hasAudioTrack(dubIt.referenceVideoURL) else {
                throw LTXUnifiedAVGeneratorError.dubItReferenceAudioMissing(dubIt.referenceVideoURL)
            }
            referenceVideos = [
                LTXReferenceVideoConditioningInput(
                    videoURL: dubIt.referenceVideoURL,
                    strength: dubIt.referenceStrength,
                    attentionStrength: 1,
                    downscaleFactor: ltxLoRAReferenceDownscaleFactor(icLoRA),
                    temporalScaleFactor: ltxLoRAReferenceTemporalScaleFactor(icLoRA)
                ),
            ]
        }
        guard options.generatedKeyframeCount >= 0,
              options.generatedKeyframeCount == 0 || options.generatedKeyframeIndices.isEmpty else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Use a generated-keyframe count or explicit indices, not both."
            )
        }
        let requestedGeneratedIndices = options.generatedKeyframeCount > 0
            ? try ltxEvenlySpacedGeneratedKeyframePositions(
                count: options.generatedKeyframeCount,
                numFrames: generationFrameCount
            )
            : options.generatedKeyframeIndices
        let requestsLTX25Conditioning = !options.imageConditionings.isEmpty
            || !requestedGeneratedIndices.isEmpty
            || !referenceVideos.isEmpty
            || options.retake != nil
            || options.dubIt != nil
            || usesDFR
        guard loadedForLTX25 || !requestsLTX25Conditioning else {
            throw LTXUnifiedAVGeneratorError.ltx25ConditioningRequiresLTX25
        }
        if loadedForLTX25 {
            for input in imageConditionings {
                guard input.pixelFrameIndex >= 0, input.pixelFrameIndex < generationFrameCount else {
                    throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(input.pixelFrameIndex)
                }
                guard input.strength >= 0, input.strength <= 1 else {
                    throw LTXUnifiedAVGeneratorError.invalidImageStrength(input.strength)
                }
                guard FileManager.default.fileExists(atPath: input.imageURL.path) else {
                    throw LTXUnifiedAVGeneratorError.imageNotFound(input.imageURL)
                }
            }
            for reference in referenceVideos {
                guard FileManager.default.fileExists(atPath: reference.videoURL.path) else {
                    throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(reference.videoURL)
                }
                if let maskURL = reference.attentionMaskVideoURL,
                   !FileManager.default.fileExists(atPath: maskURL.path) {
                    throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(maskURL)
                }
                guard (generationWidth / 2).isMultiple(of: reference.downscaleFactor),
                      (generationHeight / 2).isMultiple(of: reference.downscaleFactor) else {
                    throw LTXUnifiedAVGeneratorError.invalidResolution(
                        width: options.width,
                        height: options.height
                    )
                }
            }
        }
        if let retake = options.retake {
            guard loadedForLTX25 else {
                throw LTXUnifiedAVGeneratorError.retakeRequiresLTX25(loadedRoot)
            }
            guard FileManager.default.fileExists(atPath: retake.sourceVideoURL.path) else {
                throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(retake.sourceVideoURL)
            }
            let duration = Double(generationFrameCount) / Double(options.fps)
            guard retake.startTime >= 0,
                  retake.startTime < retake.endTime,
                  retake.endTime <= duration else {
                throw LTXUnifiedAVGeneratorError.invalidRetakeRange(
                    start: retake.startTime,
                    end: retake.endTime
                )
            }
        }
        let generatedIndices = Array(
            Set(requestedGeneratedIndices + (dfrCanvas?.keyframePositions ?? []))
        ).sorted()
        let generatedIndicesAreOrdered = zip(
            generatedIndices,
            generatedIndices.dropFirst()
        ).allSatisfy(<)
        guard generatedIndicesAreOrdered,
              generatedIndices.allSatisfy({ $0 >= 0 && $0 < generationFrameCount }) else {
            throw LTXUnifiedAVGeneratorError.invalidGeneratedKeyframes(generatedIndices)
        }
        // LTX-2.5 is natively token-state based even when no conditioning tokens are appended.
        // Keeping the plain latent path for it would bypass generated-slot masks, attention masks,
        // and the upstream sampler/guidance implementation for ordinary text-to-video requests.
        let usesLTX25TokenState = loadedForLTX25

        let usesDevOneStage = options.pipeline == .devOneStage
        let usesKeyframeInterpolation = options.pipeline == .keyframeInterpolation
        let usesRetakeOneStage = options.retake != nil
        if usesDevOneStage {
            guard loadedForLTX25, loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The dev one-stage pipeline requires the full LTX-2.5 checkpoint."
                )
            }
            guard !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  referenceVideos.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The dev one-stage pipeline supports text, image, and generated-keyframe conditioning only."
                )
            }
            guard stage1DistilledLoRAStrength == 0,
                  options.distilledLoRAStrengthStage2 == 0 else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The dev one-stage pipeline runs the full transformer without the distilled LoRA."
                )
            }
        }
        if usesKeyframeInterpolation {
            guard loadedForLTX25, loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The keyframe-interpolation pipeline requires the full LTX-2.5 checkpoint."
                )
            }
            guard !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  referenceVideos.isEmpty,
                  requestedGeneratedIndices.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The keyframe-interpolation pipeline accepts timed image guides but not DFR, retake, Dub-It, IC-LoRA references, or generated-keyframe slots."
                )
            }
        }
        let usesFullTwoStage = loadedForFullTwoStage
        let usesReusableFullTwoStage = loadedForReusableFullTwoStage
        let usesGuidedFullTwoStage = usesFullTwoStage && !usesDFR && options.dubIt == nil
        if let teaCache = options.teaCache {
            guard usesGuidedFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache is calibrated only for full LTX-2.5 two-stage generation."
                )
            }
            guard options.transformerExecution == .eager else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache requires eager transformer execution so its calibrated gate remains stable."
                )
            }
            guard options.sampler.mode == .euler || options.sampler.mode == .res2s else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache supports the calibrated Euler and Res2S full-generation paths."
                )
            }
            guard options.inferenceSteps >= 8 else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache requires at least 8 stage-one inference steps."
                )
            }
            if let threshold = teaCache.threshold,
               !threshold.isFinite || threshold <= 0 {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache threshold must be finite and positive."
                )
            }
        }
        let usesPlainLTX25DistilledPipeline = ltx25UsesDistilledAncestralStage1(
            isLTX25: loadedForLTX25,
            isFullTwoStage: usesFullTwoStage,
            usesDFR: usesDFR,
            usesHDRICLoRA: usesHDRICLoRA,
            usesRetake: options.retake != nil,
            usesDubIt: options.dubIt != nil,
            hasReferenceVideos: !referenceVideos.isEmpty
        )
        let positivePromptCacheKey = LTXPromptEmbeddingCacheKey(
            prompt: prompt,
            maxLength: options.maxTextLength
        )
        let negativePromptCacheKey = LTXPromptEmbeddingCacheKey(
            prompt: options.negativePrompt,
            maxLength: options.maxTextLength
        )
        let needsPositiveTextEncoder = options.precomputedTextEmbeddingsURL == nil
            && !promptEmbeddingCache.contains(positivePromptCacheKey)
        let needsNegativeTextEncoder = usesGuidedFullTwoStage
            && !promptEmbeddingCache.contains(negativePromptCacheKey)
        let needsTextEncoder = needsPositiveTextEncoder || needsNegativeTextEncoder
        if usesReusableFullTwoStage, needsTextEncoder {
            let reloadStart = ltxMonotonicSeconds()
            try await loadFullTextEncoderIfNeeded()
            textEncoderReloadSeconds = ltxMonotonicSeconds() - reloadStart
        }
        guard let transformer else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        if let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 {
            transformerV2.execution = options.transformerExecution
        }
        if needsTextEncoder, textEncoder == nil {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        guard let decoder else {
            throw LTXUnifiedAVGeneratorError.decoderNotLoaded
        }
        guard let upsampler else {
            throw LTXUnifiedAVGeneratorError.upsamplerNotLoaded
        }
        try installRuntimeLoRAsIfNeeded(
            user: options.loras,
            detailing: options.dfr?.detailingLoRAs ?? []
        )
        runtimeUserLoRAAdapters.forEach { $0.setActive(true) }
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
        let fullLoRAURL: URL?
        if usesFullTwoStage, !usesDevOneStage {
            guard !twoStageGenerationConsumed else {
                throw LTXUnifiedAVGeneratorError.fullGenerationRequiresReload
            }
            if usesReusableFullTwoStage {
                guard runtimeLoRAAdapter != nil else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                fullLoRAURL = nil
            } else {
                guard let distilledLoRAURL else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                fullLoRAURL = distilledLoRAURL
            }
            twoStageGenerationConsumed = true
        } else {
            fullLoRAURL = nil
        }
        if stage1DistilledLoRAStrength != 0 {
            guard usesReusableFullTwoStage, let runtimeLoRAAdapter else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "A non-zero stage-one distilled LoRA strength requires the reusable full-model load path."
                )
            }
            runtimeLoRAAdapter.setStrength(stage1DistilledLoRAStrength)
            runtimeLoRAAdapter.setActive(true)
        }
        defer {
            runtimeUserLoRAAdapters.forEach { $0.setActive(false) }
            runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
            if usesReusableFullTwoStage {
                runtimeLoRAAdapter?.setActive(false)
                twoStageGenerationConsumed = false
            }
        }

        let textEncodingStart = ltxMonotonicSeconds()
        let videoContext: MLXArray
        let audioContext: MLXArray
        if let embeddingsURL = options.precomputedTextEmbeddingsURL {
            let embeddings = try MLX.loadArrays(url: embeddingsURL)
            guard let loadedVideoContext = embeddings["video_context"],
                  let loadedAudioContext = embeddings["audio_context"] else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Precomputed embeddings must contain video_context and audio_context tensors."
                )
            }
            videoContext = loadedVideoContext.asType(loadedDType)
            audioContext = loadedAudioContext.asType(loadedDType)
            MLX.eval(videoContext, audioContext)
            if let textEncoder {
                await textEncoder.unload()
                self.textEncoder = nil
                Memory.clearCache()
            }
        } else {
            let result = try await cachedPromptEmbeddings(
                prompt: prompt,
                maxLength: options.maxTextLength
            )
            promptCacheHits += result.cacheHit ? 1 : 0
            promptCacheMisses += result.cacheHit ? 0 : 1
            videoContext = result.embeddings.video
            guard let loadedAudioContext = result.embeddings.audio else {
                throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
            }
            audioContext = loadedAudioContext
        }
        var negativeVideoContext: MLXArray?
        var negativeAudioContext: MLXArray?
        if usesGuidedFullTwoStage {
            let result = try await cachedPromptEmbeddings(
                prompt: options.negativePrompt,
                maxLength: options.maxTextLength
            )
            promptCacheHits += result.cacheHit ? 1 : 0
            promptCacheMisses += result.cacheHit ? 0 : 1
            negativeVideoContext = result.embeddings.video
            guard let audioEmbeddings = result.embeddings.audio else {
                throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
            }
            negativeAudioContext = audioEmbeddings
            MLX.eval(videoContext, audioContext, result.embeddings.video, audioEmbeddings)
            if let textEncoder {
                await textEncoder.unload()
                self.textEncoder = nil
                Memory.clearCache()
            }
        } else if usesDFR {
            MLX.eval(videoContext, audioContext)
            if let textEncoder {
                await textEncoder.unload()
                self.textEncoder = nil
                Memory.clearCache()
            }
        }
        textEncodingSeconds = textEncoderReloadSeconds + ltxMonotonicSeconds() - textEncodingStart
        let preparationStart = ltxMonotonicSeconds()

        let latentFrames = 1 + ((generationFrameCount - 1) / 8)
        let usesTargetResolutionStageOne = usesDevOneStage || usesRetakeOneStage
        let stage1H = generationHeight / (usesTargetResolutionStageOne ? 32 : 64)
        let stage1W = generationWidth / (usesTargetResolutionStageOne ? 32 : 64)
        let stage2H = generationHeight / 32
        let stage2W = generationWidth / 32
        let audioFrames = computeAudioLatentFrameCount(
            videoFrames: generationFrameCount,
            fps: max(1, options.fps)
        )
        let stage1Sigmas = try validatedLTXSigmaSchedule(
            options.sigmas ?? (usesGuidedFullTwoStage
                ? LTX2DiffusionScheduler.sigmas(
                    steps: options.inferenceSteps,
                    tokenCount: latentFrames * stage1H * stage1W
                )
                : STAGE1Sigmas)
        )
        let stage2Sigmas = try validatedLTXSigmaSchedule(options.stage2Sigmas ?? STAGE2Sigmas)

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        let modelDType = videoContext.dtype

        let isImageToVideo = options.sourceImageURL != nil
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?
        var stage1TokenState: LTX25VideoTokenState?
        var stage2TokenState: LTX25VideoTokenState?
        var generatedKeyframeLatents: MLXArray?
        var stage2LTX25ImageLatents: [MLXArray] = []
        var stage2LTX25ReferenceLatents: [MLXArray] = []
        var stage2LTX25ReferenceAttentionWeights: [MLXArray?] = []
        var stage1RetakeLatent: MLXArray?

        var videoLatents: MLXArray
        if usesLTX25TokenState {
            if !imageConditionings.isEmpty || !referenceVideos.isEmpty || options.retake != nil {
                try loadEncoderIfNeeded()
            }
            if let retake = options.retake {
                guard let encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                let sourceVideo = try loadVideoForEncoding(
                    url: retake.sourceVideoURL,
                    width: generationWidth,
                    height: generationHeight,
                    frameCap: generationFrameCount,
                    temporalScaleFactor: 1,
                    dtype: modelDType,
                    hdrColorSpace: options.hdrColorSpace
                )
                stage1RetakeLatent = encoder.encode(image: sourceVideo)
                if let stage1RetakeLatent {
                    MLX.eval(stage1RetakeLatent)
                }
            }
            videoLatents = stage1RetakeLatent ?? MLX.zeros(
                [1, 128, latentFrames, stage1H, stage1W],
                dtype: modelDType
            )
        } else if isImageToVideo {
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
                width: generationWidth / 2,
                height: generationHeight / 2,
                dtype: modelDType,
                hdrColorSpace: options.hdrColorSpace
            )
            let stage1ImageLatent = encoder.encode(image: stage1Image)

            let stage2Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: generationWidth,
                height: generationHeight,
                dtype: modelDType,
                hdrColorSpace: options.hdrColorSpace
            )
            let stage2ImageLatent = encoder.encode(image: stage2Image)
            stage2ConditioningLatent = stage2ImageLatent

            // Optional end keyframe -> conditions the tail latent frame so the clip
            // interpolates a directed start->end motion.
            var stage1EndImageLatent: MLXArray?
            if let endImageURL = options.endImageURL {
                let stage1EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: generationWidth / 2,
                    height: generationHeight / 2,
                    dtype: modelDType,
                    hdrColorSpace: options.hdrColorSpace
                )
                stage1EndImageLatent = encoder.encode(image: stage1EndImage)
                let stage2EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: generationWidth,
                    height: generationHeight,
                    dtype: modelDType,
                    hdrColorSpace: options.hdrColorSpace
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
            let stage1Sigma = MLXArray(stage1Sigmas[0]).asType(modelDType)
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
        let baseStage1VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1H,
            width: stage1W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        if usesLTX25TokenState {
            var state = try makeConditionedLTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage1VideoPositions,
                imageConditionings: imageConditionings,
                generatedKeyframeIndices: generatedIndices,
                initialGeneratedKeyframes: nil,
                encoder: encoder,
                pixelWidth: generationWidth / (usesTargetResolutionStageOne ? 1 : 2),
                pixelHeight: generationHeight / (usesTargetResolutionStageOne ? 1 : 2),
                fps: options.fps,
                replaceFirstImage: !usesKeyframeInterpolation,
                hdrColorSpace: options.hdrColorSpace
            )
            if let retake = options.retake, let stage1RetakeLatent {
                state.applyTemporalRetake(
                    cleanVideoLatent: stage1RetakeLatent,
                    startTime: retake.startTime,
                    endTime: retake.endTime,
                    fps: options.fps,
                    regenerate: retake.regenerateVideo
                )
            }
            if !referenceVideos.isEmpty {
                guard let encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                for reference in referenceVideos {
                    let pixelVideo = try loadVideoForEncoding(
                        url: reference.videoURL,
                        width: (generationWidth / 2) / reference.downscaleFactor,
                        height: (generationHeight / 2) / reference.downscaleFactor,
                        frameCap: generationFrameCount,
                        temporalScaleFactor: reference.temporalScaleFactor,
                        dtype: modelDType,
                        hdrColorSpace: options.hdrColorSpace,
                        duplicateEachFrame: options.hdrICLoRA?.highQuality == true,
                        hdrICLoRAReference: usesHDRICLoRA
                    )
                    let encoded = encoder.encode(image: pixelVideo)
                    MLX.eval(encoded)
                    let attentionWeights = try loadLTXReferenceAttentionWeights(
                        reference: reference,
                        width: (generationWidth / 2) / reference.downscaleFactor,
                        height: (generationHeight / 2) / reference.downscaleFactor,
                        frameCap: generationFrameCount,
                        targetLatent: encoded,
                        dtype: modelDType
                    )
                    state.appendReferenceLatent(
                        encoded,
                        downscaleFactor: reference.downscaleFactor,
                        temporalScaleFactor: reference.temporalScaleFactor,
                        strength: reference.strength,
                        attentionStrength: reference.attentionStrength,
                        attentionWeights: attentionWeights,
                        fps: options.fps
                    )
                    if !options.skipStage2 {
                        let stage2PixelVideo = try loadVideoForEncoding(
                            url: reference.videoURL,
                            width: generationWidth / reference.downscaleFactor,
                            height: generationHeight / reference.downscaleFactor,
                            frameCap: generationFrameCount,
                            temporalScaleFactor: reference.temporalScaleFactor,
                            dtype: modelDType,
                            hdrColorSpace: options.hdrColorSpace,
                            duplicateEachFrame: options.hdrICLoRA?.highQuality == true,
                            hdrICLoRAReference: usesHDRICLoRA
                        )
                        let stage2Encoded = encoder.encode(image: stage2PixelVideo)
                        MLX.eval(stage2Encoded)
                        stage2LTX25ReferenceLatents.append(stage2Encoded)
                        stage2LTX25ReferenceAttentionWeights.append(
                            try loadLTXReferenceAttentionWeights(
                                reference: reference,
                                width: generationWidth / reference.downscaleFactor,
                                height: generationHeight / reference.downscaleFactor,
                                frameCap: generationFrameCount,
                                targetLatent: stage2Encoded,
                                dtype: modelDType
                            )
                        )
                    }
                }
            }
            state.addNoise(scale: stage1Sigmas[0])
            MLX.eval(state.latent)
            stage1TokenState = state
            if !usesDevOneStage, !imageConditionings.isEmpty {
                guard let encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                for input in imageConditionings {
                    let image = try loadImageForEncoding(
                        url: input.imageURL,
                        width: generationWidth,
                        height: generationHeight,
                        dtype: modelDType,
                        hdrColorSpace: options.hdrColorSpace,
                        crf: input.crf ?? 18
                    )
                    let encoded = encoder.encode(image: image)
                    MLX.eval(encoded)
                    stage2LTX25ImageLatents.append(encoded)
                }
            }
        }
        if usesFullTwoStage {
            encoder = nil
            Memory.clearCache()
        }

        if usesGuidedFullTwoStage {
            MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 1)))
        }
        var audioLatents = MLXRandom.normal(
            [1, LTXAudioLatentChannels, audioFrames, LTXAudioLatentMelBins]
        ).asType(modelDType)
        var stage1AudioConditioning: LTXLatentConditioningState?
        var stage1DubItAudioReference: LTXAudioReferenceConditioningState?
        var stage1DubItAudioLatent: MLXArray?
        if let retake = options.retake,
           !MediaHDRImageIO.isEXRDirectory(retake.sourceVideoURL),
           MediaVideoIO.hasAudioTrack(retake.sourceVideoURL),
           let audioVAEWeightsURL {
            let duration = Double(generationFrameCount) / Double(options.fps)
            let sourceAudio = try MediaAudioIO.decodeSegment(
                retake.sourceVideoURL,
                startTime: 0,
                duration: duration,
                targetSampleRate: LTXAudioMelProcessor.sampleRate,
                channels: 2
            )
            let spectrogram = LTXAudioMelProcessor().extract(
                channels: planarAudioChannels(sourceAudio)
            )
            let cleanAudioLatent = try encodeLTX23AudioLatents(
                spectrogram: spectrogram,
                requiredFrameCount: audioFrames,
                weightsURL: audioVAEWeightsURL,
                dtype: loadedDType,
                sourceLayout: loadedForLTX25 ? .pytorch : .mlx
            ).asType(modelDType)
            let conditioning = makeLTXAudioTemporalConditioning(
                cleanLatent: cleanAudioLatent,
                startTime: retake.startTime,
                endTime: retake.endTime,
                regenerate: retake.regenerateAudio
            )
            let one = MLXArray(1).asType(modelDType)
            let noiseScale = MLXArray(stage1Sigmas[0]).asType(modelDType)
            let noisedAudio = audioLatents * noiseScale + cleanAudioLatent * (one - noiseScale)
            audioLatents = noisedAudio * conditioning.denoiseMask
                + cleanAudioLatent * (one - conditioning.denoiseMask)
            stage1AudioConditioning = conditioning
        } else if let dubIt = options.dubIt {
            guard let audioVAEWeightsURL else {
                throw LTXUnifiedAVGeneratorError.audioVAEWeightsMissing(
                    loadedRoot ?? dubIt.referenceVideoURL.deletingLastPathComponent()
                )
            }
            let duration = Double(generationFrameCount) / Double(options.fps)
            let referenceAudio = try MediaAudioIO.decodeSegment(
                dubIt.referenceVideoURL,
                startTime: 0,
                duration: duration,
                targetSampleRate: LTXAudioMelProcessor.sampleRate,
                channels: 2
            )
            let referenceSpectrogram = LTXAudioMelProcessor().extract(
                channels: planarAudioChannels(referenceAudio)
            )
            let referenceAudioLatent = try encodeLTX23AudioLatents(
                spectrogram: referenceSpectrogram,
                requiredFrameCount: audioFrames,
                weightsURL: audioVAEWeightsURL,
                dtype: loadedDType,
                sourceLayout: loadedForLTX25 ? .pytorch : .mlx
            ).asType(modelDType)
            let conditioning = makeLTXAudioReferenceConditioning(
                targetLatent: audioLatents,
                referenceLatent: referenceAudioLatent,
                frozenTarget: false
            )
            audioLatents = conditioning.state.latent
            stage1AudioConditioning = conditioning.state
            stage1DubItAudioReference = conditioning
        }
        MLX.eval(audioLatents)

        let stage1VideoPositions = stage1TokenState?.positions ?? baseStage1VideoPositions
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
        let stage1AudioPositions = stage1DubItAudioReference?.positions
            ?? createAudioPositionGrid(batchSize: 1, audioFrames: audioFrames)
        let stage1AudioRope = precomputeSplitRope(
            positions: stage1AudioPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )

        preparationSeconds = ltxMonotonicSeconds() - preparationStart
        let stage1DenoiseStart = ltxMonotonicSeconds()
        if usesHDRICLoRA, let tokenState = stage1TokenState {
            guard let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 else {
                throw LTXUnifiedAVGeneratorError.generatorNotLoaded
            }
            let result = denoiseLTX25VideoTokenLoop(
                videoState: tokenState,
                videoRope: stage1VideoRope,
                videoContext: videoContext,
                transformer: transformerV2,
                sigmas: stage1Sigmas,
                ancestralNoiseSeed: options.seed,
                ancestralEta: 0
            )
            stage1TokenState = result
            videoLatents = result.mainLatent()
            generatedKeyframeLatents = result.generatedKeyframes()
        } else if let tokenState = stage1TokenState {
            let result: (video: LTX25VideoTokenState, audio: MLXArray)
            if usesGuidedFullTwoStage {
                guard let negativeVideoContext, let negativeAudioContext else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                result = denoiseGuidedLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage1VideoRope,
                    audioRope: stage1AudioRope,
                    videoCrossRope: stage1VideoCrossRope,
                    audioCrossRope: stage1AudioRope,
                    positiveVideoContext: videoContext,
                    negativeVideoContext: negativeVideoContext,
                    positiveAudioContext: audioContext,
                    negativeAudioContext: negativeAudioContext,
                    transformer: transformer,
                    sigmas: stage1Sigmas,
                    videoGuidance: options.videoGuidance,
                    audioGuidance: options.audioGuidance,
                    sampler: options.sampler,
                    seed: options.seed,
                    guidanceProjectionCache: options.teaCache == nil
                        ? options.guidanceProjectionCache
                        : .disabled,
                    guidanceProjectionCacheMetrics: guidanceProjectionCacheMetrics,
                    teaCacheController: teaCacheController,
                    teaCachePipelineStage: .coarse,
                    audioConditioning: stage1AudioConditioning
                )
            } else {
                result = denoiseLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage1VideoRope,
                    audioRope: stage1AudioRope,
                    videoCrossRope: stage1VideoCrossRope,
                    audioCrossRope: stage1AudioRope,
                    videoContext: videoContext,
                    audioContext: audioContext,
                    transformer: transformer,
                    sigmas: stage1Sigmas,
                    ancestralNoiseSeed: usesPlainLTX25DistilledPipeline
                        ? options.seed &+ 10_000
                        : nil,
                    audioConditioning: stage1AudioConditioning
                )
            }
            stage1TokenState = result.video
            videoLatents = result.video.mainLatent()
            generatedKeyframeLatents = result.video.generatedKeyframes()
            audioLatents = result.audio
        } else if usesGuidedFullTwoStage {
            guard let negativeVideoContext, let negativeAudioContext else {
                throw LTXUnifiedAVGeneratorError.generatorNotLoaded
            }
            (videoLatents, audioLatents) = denoiseGuidedAVLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage1VideoRope,
                audioRope: stage1AudioRope,
                videoCrossRope: stage1VideoCrossRope,
                audioCrossRope: stage1AudioRope,
                positiveVideoContext: videoContext,
                negativeVideoContext: negativeVideoContext,
                positiveAudioContext: audioContext,
                negativeAudioContext: negativeAudioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                videoConditioning: stage1ConditioningState,
                videoGuidance: options.videoGuidance,
                audioGuidance: options.audioGuidance
            )
        } else {
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
                sigmas: stage1Sigmas,
                videoConditioning: stage1ConditioningState,
                ancestralNoiseSeed: usesPlainLTX25DistilledPipeline
                    ? options.seed &+ 10_000
                    : nil
            )
        }
        if let stage1DubItAudioReference {
            let mainAudio = stage1DubItAudioReference.mainLatent(from: audioLatents)
            MLX.eval(mainAudio)
            stage1DubItAudioLatent = mainAudio
            audioLatents = mainAudio
        }
        MLX.eval(videoLatents, audioLatents)
        // The official full/HQ two-stage pipelines refine stage-two video only
        // and decode the full-context stage-one audio. DFR follows the same rule.
        let preservedStage1AudioLatents = (usesDFR || usesFullTwoStage) ? audioLatents : nil
        let dfrDetailingReferenceLatent = options.dfr?.detailingLoRAs.isEmpty == false
            ? videoLatents[0..<1, 0..., 0..., 0..., 0...]
            : nil
        if let dfrDetailingReferenceLatent {
            MLX.eval(dfrDetailingReferenceLatent)
        }
        stage1DenoiseSeconds = ltxMonotonicSeconds() - stage1DenoiseStart

        if !usesDevOneStage, !usesRetakeOneStage, !options.skipStage2 {
            let loraFusionStart = ltxMonotonicSeconds()
            if usesReusableFullTwoStage {
                runtimeLoRAAdapter?.setStrength(options.distilledLoRAStrengthStage2)
                runtimeLoRAAdapter?.setActive(options.distilledLoRAStrengthStage2 != 0)
                loraFusionSeconds = ltxMonotonicSeconds() - loraFusionStart
            } else if let fullLoRAURL, options.distilledLoRAStrengthStage2 != 0 {
                try LTXStreamingLoRAFuser.fuse(
                    url: fullLoRAURL,
                    into: transformer,
                    strength: options.distilledLoRAStrengthStage2
                )
                loraFusionSeconds = ltxMonotonicSeconds() - loraFusionStart
            }

        let upsampleStart = ltxMonotonicSeconds()
        videoLatents = upsampleLatents(
            videoLatents,
            upsampler: upsampler,
            latentMean: decoder.latentsMean,
            latentStd: decoder.latentsStd
        )
        MLX.eval(videoLatents)
        let stage2InitialGeneratedKeyframes = generatedKeyframeLatents.map {
            upsampleLatents(
                $0,
                upsampler: upsampler,
                latentMean: decoder.latentsMean,
                latentStd: decoder.latentsStd
            )
        }
        if let stage2InitialGeneratedKeyframes {
            MLX.eval(stage2InitialGeneratedKeyframes)
        }
        upsampleSeconds = ltxMonotonicSeconds() - upsampleStart

        let baseStage2VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2H,
            width: stage2W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        var stage2AudioConditioning = stage1AudioConditioning
        var stage2DubItAudioReference: LTXAudioReferenceConditioningState?
        if usesHDRICLoRA {
            // HDR IC-LoRA runs video-only stage-two phases below. Each phase
            // applies its own initial noise and optional reference tokens.
        } else if usesLTX25TokenState {
            var state = LTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage2VideoPositions
            )
            for (input, encoded) in zip(imageConditionings, stage2LTX25ImageLatents) {
                state.applyImageLatent(
                    encoded,
                    pixelFrameIndex: input.pixelFrameIndex,
                    strength: input.strength,
                    fps: options.fps,
                    replaceFirstFrame: !usesKeyframeInterpolation
                )
            }
            for index in referenceVideos.indices {
                let reference = referenceVideos[index]
                let encoded = stage2LTX25ReferenceLatents[index]
                state.appendReferenceLatent(
                    encoded,
                    downscaleFactor: reference.downscaleFactor,
                    temporalScaleFactor: reference.temporalScaleFactor,
                    strength: reference.strength,
                    attentionStrength: reference.attentionStrength,
                    attentionWeights: stage2LTX25ReferenceAttentionWeights[index],
                    fps: options.fps
                )
            }
            if !generatedIndices.isEmpty {
                state.appendGeneratedKeyframeSlots(
                    pixelFrameIndices: generatedIndices,
                    initialKeyframes: stage2InitialGeneratedKeyframes,
                    fps: options.fps
                )
            }
            if let dfrDetailingReferenceLatent, let dfr = options.dfr {
                state.appendReferenceLatent(
                    dfrDetailingReferenceLatent,
                    downscaleFactor: dfr.resolvedDetailingReferenceDownscaleFactor,
                    strength: 1,
                    fps: options.fps
                )
            }
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            state.addNoise(scale: stage2Sigmas[0])
            MLX.eval(state.latent)
            stage2TokenState = state

            if stage1DubItAudioLatent == nil {
                if usesFullTwoStage {
                    MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
                }
                let noiseScale = MLXArray(stage2Sigmas[0]).asType(modelDType)
                let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
                audioLatents = audioNoise * noiseScale
                    + audioLatents * (MLXArray(1).asType(modelDType) - noiseScale)
                MLX.eval(audioLatents)
            }
        } else if let stage2State = stage2ConditioningLatent.map({
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
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let noise = MLXRandom.normal(videoLatents.shape).asType(modelDType)
            let noiseScale = MLXArray(stage2Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = stage2State.denoiseMask * noiseScale
            videoLatents = noise * scaledMask + stage2State.latent * (one - scaledMask)
            MLX.eval(videoLatents)
            stage2ConditioningState = stage2State

            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
            let oneMinusScale = MLXArray(1.0).asType(modelDType) - noiseScale
            audioLatents = audioNoise * noiseScale + audioLatents * oneMinusScale
            MLX.eval(audioLatents)
        } else {
            let noiseScale = MLXArray(stage2Sigmas[0]).asType(modelDType)
            let oneMinusScale = MLXArray(1.0 - stage2Sigmas[0]).asType(modelDType)
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let videoNoise = MLXRandom.normal(videoLatents.shape).asType(modelDType)
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
            videoLatents = videoNoise * noiseScale + videoLatents * oneMinusScale
            audioLatents = audioNoise * noiseScale + audioLatents * oneMinusScale
            MLX.eval(videoLatents, audioLatents)
        }
        if let stage1DubItAudioLatent {
            let conditioning = makeLTXAudioReferenceConditioning(
                targetLatent: stage1DubItAudioLatent,
                referenceLatent: stage1DubItAudioLatent,
                frozenTarget: true
            )
            audioLatents = conditioning.state.latent
            stage2AudioConditioning = conditioning.state
            stage2DubItAudioReference = conditioning
            MLX.eval(audioLatents)
        } else if let stage2AudioConditioning {
            let one = MLXArray(1).asType(modelDType)
            audioLatents = audioLatents * stage2AudioConditioning.denoiseMask
                + stage2AudioConditioning.cleanLatent
                    * (one - stage2AudioConditioning.denoiseMask)
            MLX.eval(audioLatents)
        }

        let stage2VideoPositions = stage2TokenState?.positions ?? baseStage2VideoPositions
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
        let stage2AudioPositions = stage2DubItAudioReference?.positions ?? stage1AudioPositions
        let stage2AudioRope = precomputeSplitRope(
            positions: stage2AudioPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )

        let stage2DenoiseStart = ltxMonotonicSeconds()
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(true) }
        if let hdrICLoRA = options.hdrICLoRA {
            guard let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 else {
                throw LTXUnifiedAVGeneratorError.generatorNotLoaded
            }
            videoLatents = try denoiseLTXHDRICLoRAStage2(
                initialLatent: videoLatents,
                phases: hdrICLoRA.stage2Phases,
                referenceVideos: referenceVideos,
                referenceLatents: stage2LTX25ReferenceLatents,
                videoContext: videoContext,
                transformer: transformerV2,
                fps: options.fps,
                seed: options.seed
            )
        } else if let tokenState = stage2TokenState {
            let result: (video: LTX25VideoTokenState, audio: MLXArray)
            if usesGuidedFullTwoStage, options.sampler.mode == .res2s {
                guard let negativeVideoContext, let negativeAudioContext else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                let neutralGuidance = LTXMultiModalGuidance(
                    classifierFreeScale: 1,
                    spatioTemporalScale: 0,
                    rescale: 0,
                    modalityScale: 1,
                    spatioTemporalBlocks: []
                )
                result = denoiseGuidedLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage2VideoRope,
                    audioRope: stage2AudioRope,
                    videoCrossRope: stage2VideoCrossRope,
                    audioCrossRope: stage2AudioRope,
                    positiveVideoContext: videoContext,
                    negativeVideoContext: negativeVideoContext,
                    positiveAudioContext: audioContext,
                    negativeAudioContext: negativeAudioContext,
                    transformer: transformer,
                    sigmas: stage2Sigmas,
                    videoGuidance: neutralGuidance,
                    audioGuidance: neutralGuidance,
                    sampler: options.sampler,
                    seed: options.seed &+ 2,
                    guidanceProjectionCache: options.teaCache == nil
                        ? options.guidanceProjectionCache
                        : .disabled,
                    guidanceProjectionCacheMetrics: guidanceProjectionCacheMetrics,
                    teaCacheController: nil,
                    teaCachePipelineStage: .detail,
                    audioConditioning: stage2AudioConditioning
                )
            } else {
                result = denoiseLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage2VideoRope,
                    audioRope: stage2AudioRope,
                    videoCrossRope: stage2VideoCrossRope,
                    audioCrossRope: stage2AudioRope,
                    videoContext: videoContext,
                    audioContext: audioContext,
                    transformer: transformer,
                    sigmas: stage2Sigmas,
                    audioConditioning: stage2AudioConditioning
                )
            }
            stage2TokenState = result.video
            videoLatents = result.video.mainLatent()
            generatedKeyframeLatents = result.video.generatedKeyframes()
            audioLatents = result.audio
            if let stage2DubItAudioReference {
                audioLatents = stage2DubItAudioReference.mainLatent(from: audioLatents)
            }
        } else {
            (videoLatents, audioLatents) = denoiseAVLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage2VideoRope,
                audioRope: stage2AudioRope,
                videoCrossRope: stage2VideoCrossRope,
                audioCrossRope: stage2AudioRope,
                videoContext: videoContext,
                audioContext: audioContext,
                transformer: transformer,
                sigmas: stage2Sigmas,
                videoConditioning: stage2ConditioningState
            )
        }
        MLX.eval(videoLatents, audioLatents)
        stage2DenoiseSeconds = ltxMonotonicSeconds() - stage2DenoiseStart
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
        }

        var finalFrameCount = generationFrameCount
        var finalFPS = options.fps
        var outputGeneratedIndices = generatedIndices
        if let dfr = options.dfr, dfr.temporalUpsampleRounds > 0 {
            guard let temporalUpsampler,
                  let transformerV2 = transformer as? LTXUnifiedAVTransformerV2,
                  var carryKeyframes = generatedKeyframeLatents,
                  !generatedIndices.isEmpty else {
                throw LTXUnifiedAVGeneratorError.dfrRequiresLTX25Full(
                    loadedRoot ?? URL(fileURLWithPath: "", isDirectory: true)
                )
            }
            var carryPositions = generatedIndices
            let temporalSigmas = Array(STAGE1Sigmas.dropFirst(4))

            for roundIndex in 1...dfr.temporalUpsampleRounds {
                videoLatents = upsampleLatentsTemporally(
                    videoLatents,
                    upsampler: temporalUpsampler,
                    latentMean: decoder.latentsMean,
                    latentStd: decoder.latentsStd
                )
                MLX.eval(videoLatents)
                finalFrameCount = 2 * (finalFrameCount - 1) + 1
                finalFPS *= 2
                let seamPositions = carryPositions.map { 2 * $0 }
                let seamIndex = Dictionary(
                    uniqueKeysWithValues: seamPositions.enumerated().map { ($0.element, $0.offset) }
                )
                let ranges = try LTX25DFRLayout.tileRanges(
                    seamPositions: seamPositions,
                    frameCount: finalFrameCount,
                    tileCount: 1 << roundIndex
                )
                var tileLatents: [MLXArray] = []
                var slotPositions: [Int] = []
                var slotLatents: [MLXArray] = []

                for (tileIndex, range) in ranges.enumerated() {
                    let tileVideo = videoLatents[
                        0...,
                        0...,
                        range.latentStart..<range.latentEndExclusive,
                        0...,
                        0...
                    ]
                    let localLatentFrames = range.latentEndExclusive - range.latentStart
                    let positions = createPositionGrid(
                        batchSize: 1,
                        numFrames: localLatentFrames,
                        height: stage2H,
                        width: stage2W,
                        temporalScale: 8,
                        spatialScale: 32,
                        fps: Float(min(finalFPS, 60)),
                        causalFix: true
                    )
                    var state = LTX25VideoTokenState(
                        initialLatent: tileVideo,
                        positions: positions
                    )

                    for (input, encoded) in zip(imageConditionings, stage2LTX25ImageLatents) {
                        if range.pixelStart == 0
                            || (input.pixelFrameIndex >= range.pixelStart
                                && input.pixelFrameIndex <= range.pixelEnd) {
                            state.applyImageLatent(
                                encoded,
                                pixelFrameIndex: input.pixelFrameIndex - range.pixelStart,
                                strength: input.strength,
                                fps: min(finalFPS, 60)
                            )
                        }
                    }

                    for globalPosition in range.anchorKeyframes {
                        guard let index = seamIndex[globalPosition] else {
                            throw LTX25DFRLayoutError.invalidSeams(seamPositions)
                        }
                        state.applyImageLatent(
                            carryKeyframes[0..., 0..., index..<index + 1, 0..., 0...],
                            pixelFrameIndex: globalPosition - range.pixelStart,
                            strength: 0.95,
                            fps: min(finalFPS, 60),
                            replaceFirstFrame: false
                        )
                    }

                    let localSlots = LTX25DFRLayout.remapPositionsToLocal(
                        range.slotKeyframes,
                        pixelStart: range.pixelStart
                    )
                    if !localSlots.isEmpty {
                        let initials = MLX.concatenated(
                            localSlots.map { localPosition in
                                let index = min(
                                    max((localPosition + 4) / 8, 0),
                                    tileVideo.dim(2) - 1
                                )
                                return tileVideo[0..., 0..., index..<index + 1, 0..., 0...]
                            },
                            axis: 2
                        )
                        state.appendGeneratedKeyframeSlots(
                            pixelFrameIndices: localSlots,
                            initialKeyframes: initials,
                            fps: min(finalFPS, 60)
                        )
                    }
                    state.addNoise(scale: temporalSigmas[0])
                    let rope = precomputeSplitRope(
                        positions: state.positions,
                        dim: 4096,
                        theta: 10_000,
                        maxPos: [20, 2048, 2048],
                        numHeads: 32
                    )
                    state = denoiseLTX25VideoTokenLoop(
                        videoState: state,
                        videoRope: rope,
                        videoContext: videoContext,
                        transformer: transformerV2,
                        sigmas: temporalSigmas,
                        ancestralNoiseSeed: options.seed + 1_000 * roundIndex + tileIndex,
                        ancestralEta: 0.5
                    )
                    tileLatents.append(state.mainLatent())
                    if let generated = state.generatedKeyframes() {
                        slotPositions.append(contentsOf: range.slotKeyframes)
                        slotLatents.append(generated)
                    }
                }

                videoLatents = try LTX25DFRLayout.stitchTileLatents(
                    tileLatents,
                    ranges: ranges
                )
                MLX.eval(videoLatents)

                var latentsByPosition: [Int: MLXArray] = [:]
                for (index, position) in seamPositions.enumerated() {
                    latentsByPosition[position] = carryKeyframes[
                        0...,
                        0...,
                        index..<index + 1,
                        0...,
                        0...
                    ]
                }
                var flatSlotIndex = 0
                for group in slotLatents {
                    for index in 0..<group.dim(2) {
                        let position = slotPositions[flatSlotIndex]
                        if latentsByPosition[position] == nil {
                            latentsByPosition[position] = group[
                                0...,
                                0...,
                                index..<index + 1,
                                0...,
                                0...
                            ]
                        }
                        flatSlotIndex += 1
                    }
                }
                carryPositions = latentsByPosition.keys.sorted()
                carryKeyframes = MLX.concatenated(
                    carryPositions.map { latentsByPosition[$0]! },
                    axis: 2
                )
                MLX.eval(carryKeyframes)
            }

            let targetFrameCount = (options.numFrames - 1) * dfr.playbackRateMultiplier + 1
            if targetFrameCount != finalFrameCount {
                let keepLatents = (targetFrameCount - 1) / 8 + 1
                videoLatents = videoLatents[0..., 0..., 0..<keepLatents, 0..., 0...]
                finalFrameCount = targetFrameCount
                MLX.eval(videoLatents)
            }
            let retainedCarry = carryPositions.enumerated().filter { $0.element < targetFrameCount }
            outputGeneratedIndices = retainedCarry.map(\.element)
            if retainedCarry.isEmpty {
                generatedKeyframeLatents = nil
            } else {
                generatedKeyframeLatents = MLX.concatenated(
                    retainedCarry.map { index, _ in
                        carryKeyframes[0..., 0..., index..<index + 1, 0..., 0...]
                    },
                    axis: 2
                )
                MLX.eval(generatedKeyframeLatents!)
            }
        } else if options.dfr != nil {
            let targetFrameCount = options.numFrames
            if targetFrameCount != finalFrameCount {
                let keepLatents = (targetFrameCount - 1) / 8 + 1
                videoLatents = videoLatents[0..., 0..., 0..<keepLatents, 0..., 0...]
                finalFrameCount = targetFrameCount
                MLX.eval(videoLatents)
            }
            let retained = generatedIndices.enumerated().filter { $0.element < targetFrameCount }
            outputGeneratedIndices = retained.map(\.element)
            if let currentGeneratedKeyframes = generatedKeyframeLatents, !retained.isEmpty {
                let filtered = MLX.concatenated(
                    retained.map { index, _ in
                        currentGeneratedKeyframes[0..., 0..., index..<index + 1, 0..., 0...]
                    },
                    axis: 2
                )
                MLX.eval(filtered)
                generatedKeyframeLatents = filtered
            } else if retained.isEmpty {
                generatedKeyframeLatents = nil
            }
        }
        if let preservedStage1AudioLatents {
            audioLatents = preservedStage1AudioLatents
            MLX.eval(audioLatents)
        }
        if usesReusableFullTwoStage {
            runtimeLoRAAdapter?.setActive(false)
        }

        let videoDecodeStart = ltxMonotonicSeconds()
        let decodedVideo: MLXArray?
        var frames: MLXArray
        var hdrOutput: LTXHDROutputFrames?
        if let diffusionDecoder {
            let fullDecoded = try diffusionDecoder.decode(
                sample: videoLatents,
                seed: options.seed
            )
            decodedVideo = fullDecoded
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    fullDecoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(fullDecoded)
            }
        } else if let tiling = selectDecodeTilingConfig(
            width: generationWidth,
            height: generationHeight,
            numFrames: finalFrameCount,
            fps: finalFPS,
            spatialTileSizeInPixels: options.vaeSpatialTileSize
                ?? (usesHDRICLoRA ? 1_280 : nil),
            spatialTileOverlapInPixels: options.vaeSpatialTileSize != nil || usesHDRICLoRA
                ? options.vaeSpatialTileOverlap
                : 0
        ) {
            if let colorSpace = options.hdrColorSpace {
                let fullDecoded = decodeWithTilingRaw(
                    decoder: decoder,
                    latents: videoLatents,
                    spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                    spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                    temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                    temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                    spatialScale: 32,
                    temporalScale: 8
                )
                decodedVideo = fullDecoded
                let output = LTXHDRColorPipeline.decode(
                    fullDecoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                decodedVideo = nil
                hdrOutput = nil
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
            }
        } else {
            let fullDecoded = decoder.decode(sample: videoLatents, timestep: nil)
            decodedVideo = fullDecoded
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    fullDecoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(fullDecoded)
            }
        }
        if options.hdrICLoRA?.highQuality == true {
            let indices = MLXArray(
                Array(stride(from: 0, to: frames.dim(0), by: 2)).map(Int32.init)
            )
            frames = MLX.take(frames, indices, axis: 0)
            hdrOutput = hdrOutput?.selectingFrames(indices)
            finalFrameCount = options.numFrames
        }
        if usesHDRICLoRA,
           generationWidth != options.width || generationHeight != options.height {
            frames = frames[0..., 0..<options.height, 0..<options.width, 0...]
            hdrOutput = hdrOutput?.cropped(width: options.width, height: options.height)
        }
        MLX.eval(frames)
        videoDecodeSeconds = ltxMonotonicSeconds() - videoDecodeStart

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
        var audioWaveform: MLXArray?
        var audioSampleRate: Int?
        if decodeAudio {
            let audioDecodeStart = ltxMonotonicSeconds()
            let activeAudioDecoder: LTXAudioDecoder
            let activeVocoder: LTXAudioVocoderBase
            if usesFullTwoStage {
                guard let loadedRoot else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                if !usesReusableFullTwoStage {
                    self.transformer = nil
                    self.upsampler = nil
                    Memory.clearCache()
                }
                let audioWeightsURL = loadedForLTX25
                    ? LTX25Resources(rootURL: loadedRoot).audioVAEURL
                    : loadedRoot.appendingPathComponent("audio_vae.safetensors", isDirectory: false)
                let vocoderWeightsURL = loadedForLTX25
                    ? audioWeightsURL
                    : loadedRoot.appendingPathComponent("vocoder.safetensors", isDirectory: false)
                let sourceLayout: LTXTensorWeightLayout = loadedForLTX25 ? .pytorch : .mlx
                activeAudioDecoder = try loadLTXAudioDecoder(
                    weightsURL: audioWeightsURL,
                    sourceLayout: sourceLayout
                )
                activeVocoder = try loadLTXVocoder(
                    weightsURL: vocoderWeightsURL,
                    sourceLayout: loadedForLTX25 ? .pytorch : .mlx,
                    configurationRoot: loadedRoot,
                    usesPackedConfiguration: loadedForLTX25
                )
                if !usesReusableFullTwoStage {
                    self.audioDecoder = activeAudioDecoder
                    self.vocoder = activeVocoder
                }
            } else {
                guard let audioDecoder else {
                    throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
                }
                guard let vocoder else {
                    throw LTXUnifiedAVGeneratorError.vocoderNotLoaded
                }
                activeAudioDecoder = audioDecoder
                activeVocoder = vocoder
            }
            let mel = activeAudioDecoder.decode(latents: audioLatents.asType(.float32))
            saveLTXAVDebugArray(mel, suffix: "audio_mel")
            let vocodedAudio = activeVocoder(mel)
            saveLTXAVDebugAudio(
                vocodedAudio,
                suffix: "audio_vocoded_raw",
                sampleRate: activeVocoder.outputSamplingRate
            )
            let waveform = matchLTXAudioWaveformDuration(
                vocodedAudio,
                videoFrames: finalFrameCount,
                fps: finalFPS,
                sampleRate: activeVocoder.outputSamplingRate
            )
            MLX.eval(waveform)
            audioDecodeSeconds = ltxMonotonicSeconds() - audioDecodeStart
            saveLTXAVDebugAudio(
                waveform,
                suffix: "audio_waveform_matched",
                sampleRate: activeVocoder.outputSamplingRate
            )
            audioWaveform = waveform
            audioSampleRate = activeVocoder.outputSamplingRate
        }

        try teaCacheController?.writeCalibrationReport()

        return LTXUnifiedGenerationOutput(
            frames: frames,
            hdrOutput: hdrOutput,
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            audioWaveform: audioWaveform,
            audioSampleRate: audioSampleRate,
            generatedKeyframeLatents: generatedKeyframeLatents,
            generatedKeyframeIndices: generatedKeyframeLatents == nil ? [] : outputGeneratedIndices,
            playbackFPS: finalFPS,
            timings: LTXGenerationTimings(
                textEncodingSeconds: textEncodingSeconds,
                promptCacheHits: promptCacheHits,
                promptCacheMisses: promptCacheMisses,
                guidanceProjectionCacheBuildSeconds: guidanceProjectionCacheMetrics.buildSeconds,
                guidanceProjectionCacheBuilds: guidanceProjectionCacheMetrics.buildCount,
                guidanceProjectionCacheReuses: guidanceProjectionCacheMetrics.reuseCount,
                guidanceProjectionCacheFallbacks: guidanceProjectionCacheMetrics.fallbackCount,
                teaCacheDecisionSeconds: teaCacheController?.metrics.decisionSeconds ?? 0,
                teaCacheComputedBlockStacks: teaCacheController?.metrics.computedBlockStacks ?? 0,
                teaCacheReusedBlockStacks: teaCacheController?.metrics.reusedBlockStacks ?? 0,
                preparationSeconds: preparationSeconds,
                stage1DenoiseSeconds: stage1DenoiseSeconds,
                loraFusionSeconds: loraFusionSeconds,
                upsampleSeconds: upsampleSeconds,
                stage2DenoiseSeconds: stage2DenoiseSeconds,
                videoDecodeSeconds: videoDecodeSeconds,
                audioDecodeSeconds: audioDecodeSeconds,
                totalSeconds: ltxMonotonicSeconds() - totalStart
            )
        )
    }
}

func validateLTXAudioSegment(
    metadata: MediaAudioMetadata,
    startTime: Double,
    duration: Double
) throws {
    guard metadata.channelCount == 1 || metadata.channelCount == 2 else {
        throw LTXUnifiedAVGeneratorError.unsupportedAudioChannels(metadata.channelCount)
    }
    let availableDuration = max(0, metadata.durationSeconds - startTime)
    guard availableDuration + 1e-6 >= duration else {
        throw LTXUnifiedAVGeneratorError.audioSegmentTooShort(
            required: duration,
            available: availableDuration
        )
    }
}

private func decodeExactStereoAudioSegment(
    url: URL,
    startTime: Double,
    duration: Double,
    sampleRate: Int
) throws -> MediaAudioBuffer {
    let decoded = try MediaAudioIO.decodeSegment(
        url,
        startTime: startTime,
        duration: duration,
        targetSampleRate: sampleRate,
        channels: 2
    )
    let frameCount = Int((duration * Double(sampleRate)).rounded(.toNearestOrEven))
    let requiredSampleCount = frameCount * 2
    guard decoded.samples.count >= requiredSampleCount else {
        throw LTXUnifiedAVGeneratorError.audioDecodeReturnedTooFewSamples(
            required: requiredSampleCount,
            actual: decoded.samples.count
        )
    }
    return MediaAudioBuffer(
        samples: Array(decoded.samples.prefix(requiredSampleCount)),
        sampleRate: sampleRate,
        channelCount: 2,
        isInterleaved: true
    )
}

private func planarAudioChannels(_ audio: MediaAudioBuffer) -> [[Float]] {
    let frameCount = audio.samples.count / audio.channelCount
    var channels = [[Float]](
        repeating: [Float](repeating: 0, count: frameCount),
        count: audio.channelCount
    )
    if audio.isInterleaved {
        for frame in 0..<frameCount {
            for channel in 0..<audio.channelCount {
                channels[channel][frame] = audio.samples[frame * audio.channelCount + channel]
            }
        }
    } else {
        for channel in 0..<audio.channelCount {
            let start = channel * frameCount
            channels[channel] = Array(audio.samples[start..<(start + frameCount)])
        }
    }
    return channels
}

func makeLTXAudioTemporalConditioning(
    cleanLatent: MLXArray,
    startTime: Double,
    endTime: Double,
    regenerate: Bool
) -> LTXLatentConditioningState {
    precondition(cleanLatent.ndim == 4, "audio latent must be BCTF")
    precondition(startTime >= 0 && startTime < endTime)
    let frameCount = cleanLatent.dim(2)
    let values = (0..<frameCount).map { frame -> Float in
        guard regenerate else { return 0 }
        let start = Double(max(0, frame * LTXAudioLatentDownsampleFactor + 1
            - LTXAudioLatentDownsampleFactor))
            * Double(LTXAudioHopLength) / Double(LTXAudioLatentSampleRate)
        let end = Double(max(0, (frame + 1) * LTXAudioLatentDownsampleFactor + 1
            - LTXAudioLatentDownsampleFactor))
            * Double(LTXAudioHopLength) / Double(LTXAudioLatentSampleRate)
        return end > startTime && start < endTime ? 1 : 0
    }
    let mask = MLXArray(values)
        .reshaped(1, 1, frameCount, 1)
        .asType(cleanLatent.dtype)
    return LTXLatentConditioningState(
        latent: cleanLatent,
        cleanLatent: cleanLatent,
        denoiseMask: mask
    )
}

struct LTXAudioReferenceConditioningState {
    let state: LTXLatentConditioningState
    let positions: MLXArray
    let targetFrameCount: Int

    func mainLatent(from latent: MLXArray) -> MLXArray {
        latent[0..., 0..., 0..<targetFrameCount, 0...]
    }
}

/// Mirrors upstream `AudioConditionByReferenceLatent`: reference tokens are
/// appended clean after the target sequence and placed immediately before time
/// zero, with a 40 ms gap. A frozen target is used by Dub-It's second stage.
func makeLTXAudioReferenceConditioning(
    targetLatent: MLXArray,
    referenceLatent: MLXArray,
    frozenTarget: Bool
) -> LTXAudioReferenceConditioningState {
    precondition(targetLatent.ndim == 4, "target audio latent must be BCTF")
    precondition(referenceLatent.ndim == 4, "reference audio latent must be BCTF")
    precondition(targetLatent.dim(0) == referenceLatent.dim(0), "audio batches must match")
    precondition(targetLatent.dim(1) == referenceLatent.dim(1), "audio channels must match")
    precondition(targetLatent.dim(3) == referenceLatent.dim(3), "audio mel bins must match")

    let targetFrames = targetLatent.dim(2)
    let referenceFrames = referenceLatent.dim(2)
    let dtype = targetLatent.dtype
    let targetMask = frozenTarget
        ? MLX.zeros([targetLatent.dim(0), 1, targetFrames, 1], dtype: dtype)
        : MLX.ones([targetLatent.dim(0), 1, targetFrames, 1], dtype: dtype)
    let referenceMask = MLX.zeros(
        [referenceLatent.dim(0), 1, referenceFrames, 1],
        dtype: dtype
    )
    let targetClean = frozenTarget
        ? targetLatent
        : MLX.zeros(targetLatent.shape, dtype: dtype)
    let typedReference = referenceLatent.asType(dtype)
    let state = LTXLatentConditioningState(
        latent: MLX.concatenated([targetLatent, typedReference], axis: 2),
        cleanLatent: MLX.concatenated([targetClean, typedReference], axis: 2),
        denoiseMask: MLX.concatenated([targetMask, referenceMask], axis: 2)
    )

    let targetPositions = createAudioPositionGrid(
        batchSize: targetLatent.dim(0),
        audioFrames: targetFrames
    )
    let referencePositions = createAudioPositionGrid(
        batchSize: referenceLatent.dim(0),
        audioFrames: referenceFrames
    )
    let lastReferenceEnd = Float(max(
        0,
        referenceFrames * LTXAudioLatentDownsampleFactor
            + 1 - LTXAudioLatentDownsampleFactor
    )) * Float(LTXAudioHopLength) / Float(LTXAudioLatentSampleRate)
    let shiftedReferencePositions = referencePositions - MLXArray(lastReferenceEnd + 0.04)
    let positions = MLX.concatenated(
        [targetPositions, shiftedReferencePositions],
        axis: 2
    ).asType(.float32)
    return LTXAudioReferenceConditioningState(
        state: state,
        positions: positions,
        targetFrameCount: targetFrames
    )
}

func encodeLTX23AudioLatents(
    spectrogram: MLXArray,
    requiredFrameCount: Int,
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> MLXArray {
    let encoder = LTXAudioEncoder()
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: encoder,
        dtype: dtype,
        verify: .none,
        include: { key in
            key.hasPrefix("audio_vae.encoder.")
                || key.hasPrefix("audio_vae.per_channel_statistics.")
        },
        mapper: { key, value in
            mapAudioVaeEncoderWeight(
                key: key,
                value: value,
                dtype: dtype,
                sourceLayout: sourceLayout
            )
        },
        batchSize: 24
    )
    let encoded = encoder.encode(spectrogram: spectrogram.asType(dtype))
    guard encoded.dim(2) >= requiredFrameCount else {
        throw LTXUnifiedAVGeneratorError.audioLatentTooShort(
            required: requiredFrameCount,
            actual: encoded.dim(2)
        )
    }
    let cropped = encoded[0..., 0..., 0..<requiredFrameCount, 0...]
    MLX.eval(cropped)
    return cropped
}

typealias LTXRope = (cos: MLXArray, sin: MLXArray)

private func makeLTXAudioToVideoVideoRopes(
    latentFrames: Int,
    height: Int,
    width: Int,
    fps: Double
) -> (selfAttention: LTXRope, crossAttention: LTXRope) {
    let positions = createPositionGrid(
        batchSize: 1,
        numFrames: latentFrames,
        height: height,
        width: width,
        temporalScale: 8,
        spatialScale: 32,
        fps: Float(fps),
        causalFix: true
    )
    return makeLTXAudioToVideoVideoRopes(positions: positions)
}

private func makeLTXAudioToVideoVideoRopes(
    positions: MLXArray
) -> (selfAttention: LTXRope, crossAttention: LTXRope) {
    let selfAttention = precomputeSplitRope(
        positions: positions,
        dim: 4_096,
        theta: 10_000,
        maxPos: [20, 2_048, 2_048],
        numHeads: 32
    )
    let crossPositions = positions[0..., 0..<1, 0..., 0...]
    let crossAttention = precomputeSplitRope(
        positions: crossPositions,
        dim: 2_048,
        theta: 10_000,
        maxPos: [20],
        numHeads: 32
    )
    return (selfAttention, crossAttention)
}

private func predictFrozenAudioVideoDenoised(
    flatVideo: MLXArray,
    flatAudio: MLXArray,
    videoTimesteps: MLXArray,
    audioTimesteps: MLXArray,
    videoSigma: MLXArray,
    audioSigma: MLXArray,
    videoContext: MLXArray,
    audioContext: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    transformer: any LTXUnifiedAVTransformerRuntime,
    perturbation: LTXAudioToVideoPerturbation,
    outputShape: (batch: Int, channels: Int, frames: Int, height: Int, width: Int)
) -> MLXArray {
    let output = transformer.forward(
        videoLatent: flatVideo,
        videoKeyframesMask: makeLTXVideoKeyframesMask(
            batchSize: outputShape.batch,
            tokenCount: flatVideo.dim(1),
            tokensPerFirstFrame: outputShape.height * outputShape.width,
            dtype: flatVideo.dtype
        ),
        videoAttentionMask: nil,
        audioLatent: flatAudio,
        timestep: videoSigma,
        videoTimesteps: videoTimesteps,
        audioTimesteps: audioTimesteps,
        videoContext: videoContext,
        audioContext: audioContext,
        videoRope: videoRope,
        audioRope: audioRope,
        videoCrossRope: videoCrossRope,
        audioCrossRope: audioCrossRope,
        audioSigma: audioSigma,
        perturbation: perturbation
    )
    let denoisedFlat = (
        flatVideo.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(flatVideo.dtype)
    let denoised = denoisedFlat
        .reshaped(
            outputShape.batch,
            outputShape.frames,
            outputShape.height,
            outputShape.width,
            outputShape.channels
        )
        .transposed(0, 4, 1, 2, 3)
    MLX.eval(denoised)
    return denoised
}

func denoiseFrozenAudioVideoLoop(
    videoLatents: MLXArray,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray?,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoConditioning: LTXLatentConditioningState?,
    guidance: LTXAudioToVideoGuidance?,
    debugLabel: String? = nil
) throws -> (video: MLXArray, audio: MLXArray) {
    var currentVideo = videoLatents
    let dtype = videoLatents.dtype
    let audioBatch = audioLatents.dim(0)
    let audioChannels = audioLatents.dim(1)
    let audioFrames = audioLatents.dim(2)
    let audioMelBins = audioLatents.dim(3)
    let flatAudio = audioLatents
        .transposed(0, 2, 1, 3)
        .reshaped(audioBatch, audioFrames, audioChannels * audioMelBins)
    let audioTimesteps = MLX.zeros([audioBatch, audioFrames], dtype: dtype)
    let audioSigma = MLX.zeros([audioBatch], dtype: dtype)

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let batch = currentVideo.dim(0)
        let channels = currentVideo.dim(1)
        let frames = currentVideo.dim(2)
        let height = currentVideo.dim(3)
        let width = currentVideo.dim(4)
        let tokenCount = frames * height * width
        let flatVideo = currentVideo
            .transposed(0, 2, 3, 4, 1)
            .reshaped(batch, tokenCount, channels)

        let videoTimesteps: MLXArray
        if let videoConditioning {
            let mask = videoConditioning.denoiseMask.reshaped(batch, 1, frames, 1, 1)
            let flattenedMask = broadcast(
                mask,
                to: [batch, 1, frames, height, width]
            ).reshaped(batch, tokenCount)
            videoTimesteps = MLXArray(sigma).asType(dtype) * flattenedMask
        } else {
            videoTimesteps = MLX.full(
                [batch, tokenCount],
                values: MLXArray(sigma).asType(dtype)
            )
        }
        let videoSigma = MLX.full(
            [batch],
            values: MLXArray(sigma).asType(dtype)
        )
        let outputShape = (batch, channels, frames, height, width)
        if index == 0, let debugLabel {
            let parityIO = LTXAudioToVideoParityIO()
            try parityIO.save(flatVideo, suffix: "\(debugLabel)_step0_video_input")
            try parityIO.save(flatAudio, suffix: "\(debugLabel)_step0_audio_input")
            try parityIO.save(videoTimesteps, suffix: "\(debugLabel)_step0_video_timesteps")
            try parityIO.save(videoRope.cos, suffix: "\(debugLabel)_video_rope_cos")
            try parityIO.save(videoRope.sin, suffix: "\(debugLabel)_video_rope_sin")
            try parityIO.save(audioRope.cos, suffix: "\(debugLabel)_audio_rope_cos")
            try parityIO.save(audioRope.sin, suffix: "\(debugLabel)_audio_rope_sin")
        }

        let conditioned = predictFrozenAudioVideoDenoised(
            flatVideo: flatVideo,
            flatAudio: flatAudio,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoSigma: videoSigma,
            audioSigma: audioSigma,
            videoContext: positiveVideoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            transformer: transformer,
            perturbation: .none,
            outputShape: outputShape
        )
        if index == 0, let debugLabel {
            let conditionedTokens = conditioned
                .transposed(0, 2, 3, 4, 1)
                .reshaped(batch, tokenCount, channels)
            try LTXAudioToVideoParityIO().save(
                conditionedTokens,
                suffix: "\(debugLabel)_step0_conditioned_x0"
            )
        }

        var denoised = conditioned
        if let guidance {
            let negativeText: MLXArray
            if guidance.classifierFreeScale == 1 {
                negativeText = conditioned
            } else if let negativeVideoContext {
                negativeText = predictFrozenAudioVideoDenoised(
                    flatVideo: flatVideo,
                    flatAudio: flatAudio,
                    videoTimesteps: videoTimesteps,
                    audioTimesteps: audioTimesteps,
                    videoSigma: videoSigma,
                    audioSigma: audioSigma,
                    videoContext: negativeVideoContext,
                    audioContext: audioContext,
                    videoRope: videoRope,
                    audioRope: audioRope,
                    videoCrossRope: videoCrossRope,
                    audioCrossRope: audioCrossRope,
                    transformer: transformer,
                    perturbation: .none,
                    outputShape: outputShape
                )
            } else {
                preconditionFailure("A negative video context is required for classifier-free guidance.")
            }

            let perturbed = guidance.spatioTemporalScale == 0 ? conditioned : predictFrozenAudioVideoDenoised(
                flatVideo: flatVideo,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                videoSigma: videoSigma,
                audioSigma: audioSigma,
                videoContext: positiveVideoContext,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: .spatioTemporal(blocks: guidance.spatioTemporalBlocks),
                outputShape: outputShape
            )
            let isolated = guidance.audioToVideoScale == 1 ? conditioned : predictFrozenAudioVideoDenoised(
                flatVideo: flatVideo,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                videoSigma: videoSigma,
                audioSigma: audioSigma,
                videoContext: positiveVideoContext,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: .isolatedModalities,
                outputShape: outputShape
            )
            denoised = guidance.combine(
                conditioned: conditioned,
                negativeText: negativeText,
                perturbed: perturbed,
                isolatedAudio: isolated
            )
            MLX.eval(denoised)
            if index == 0, let debugLabel {
                let parityIO = LTXAudioToVideoParityIO()
                let variants = [
                    ("negative_x0", negativeText),
                    ("perturbed_x0", perturbed),
                    ("isolated_x0", isolated),
                    ("guided_x0", denoised),
                ]
                for (suffix, array) in variants {
                    let tokens = array
                        .transposed(0, 2, 3, 4, 1)
                        .reshaped(batch, tokenCount, channels)
                    try parityIO.save(tokens, suffix: "\(debugLabel)_step0_\(suffix)")
                }
            }
        }

        if let videoConditioning {
            let one = MLXArray(1).asType(denoised.dtype)
            denoised = denoised * videoConditioning.denoiseMask
                + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
        }
        let sigma32 = MLXArray(sigma)
        let delta32 = MLXArray(nextSigma - sigma)
        let velocity32 = (currentVideo.asType(.float32) - denoised.asType(.float32)) / sigma32
        currentVideo = (currentVideo.asType(.float32) + velocity32 * delta32).asType(dtype)
        MLX.eval(currentVideo)
    }
    return (currentVideo, audioLatents)
}

private func predictFrozenLTX25AudioVideoDenoised(
    videoState: LTX25VideoTokenState,
    flatAudio: MLXArray,
    videoTimesteps: MLXArray,
    audioTimesteps: MLXArray,
    videoSigma: MLXArray,
    audioSigma: MLXArray,
    videoContext: MLXArray,
    audioContext: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    transformer: any LTXUnifiedAVTransformerRuntime,
    perturbation: LTXAudioToVideoPerturbation
) -> MLXArray {
    let output = transformer.forward(
        videoLatent: videoState.latent,
        videoKeyframesMask: videoState.keyframesMask,
        videoAttentionMask: videoState.attentionMask,
        audioLatent: flatAudio,
        timestep: videoSigma,
        videoTimesteps: videoTimesteps,
        audioTimesteps: audioTimesteps,
        videoContext: videoContext,
        audioContext: audioContext,
        videoRope: videoRope,
        audioRope: audioRope,
        videoCrossRope: videoCrossRope,
        audioCrossRope: audioCrossRope,
        audioSigma: audioSigma,
        perturbation: perturbation
    )
    let denoised = (
        videoState.latent.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(videoState.latent.dtype)
    MLX.eval(denoised)
    return denoised
}

private func denoiseFrozenLTX25AudioVideoTokenLoop(
    videoState: LTX25VideoTokenState,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray?,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    guidance: LTXAudioToVideoGuidance?
) -> LTX25VideoTokenState {
    var current = videoState
    let dtype = current.latent.dtype
    let audioBatch = audioLatents.dim(0)
    let audioChannels = audioLatents.dim(1)
    let audioFrames = audioLatents.dim(2)
    let audioMelBins = audioLatents.dim(3)
    let flatAudio = audioLatents
        .transposed(0, 2, 1, 3)
        .reshaped(audioBatch, audioFrames, audioChannels * audioMelBins)
    let audioTimesteps = MLX.zeros([audioBatch, audioFrames], dtype: dtype)
    let audioSigma = MLX.zeros([audioBatch], dtype: dtype)

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let videoTimesteps = current.denoiseMask.squeezed(axis: -1)
            * MLXArray(sigma).asType(dtype)
        let videoSigma = MLX.full(
            [current.targetShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )

        func predict(
            context: MLXArray,
            perturbation: LTXAudioToVideoPerturbation
        ) -> MLXArray {
            predictFrozenLTX25AudioVideoDenoised(
                videoState: current,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                videoSigma: videoSigma,
                audioSigma: audioSigma,
                videoContext: context,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: perturbation
            )
        }

        let conditioned = predict(context: positiveVideoContext, perturbation: .none)
        var denoised = conditioned
        if let guidance {
            let negativeText: MLXArray
            if guidance.classifierFreeScale == 1 {
                negativeText = conditioned
            } else if let negativeVideoContext {
                negativeText = predict(context: negativeVideoContext, perturbation: .none)
            } else {
                preconditionFailure("A negative video context is required for classifier-free guidance.")
            }
            let perturbed = guidance.spatioTemporalScale == 0
                ? conditioned
                : predict(
                    context: positiveVideoContext,
                    perturbation: .spatioTemporal(blocks: guidance.spatioTemporalBlocks)
                )
            let isolated = guidance.audioToVideoScale == 1
                ? conditioned
                : predict(context: positiveVideoContext, perturbation: .isolatedModalities)
            denoised = guidance.combine(
                conditioned: conditioned,
                negativeText: negativeText,
                perturbed: perturbed,
                isolatedAudio: isolated
            )
        }
        let one = MLXArray(1).asType(dtype)
        denoised = denoised * current.denoiseMask
            + current.cleanLatent * (one - current.denoiseMask)
        let velocity = (
            current.latent.asType(.float32) - denoised.asType(.float32)
        ) / MLXArray(sigma)
        current.latent = (
            current.latent.asType(.float32) + velocity * MLXArray(nextSigma - sigma)
        ).asType(dtype)
        MLX.eval(current.latent)
    }
    return current
}

func matchLTXAudioWaveformDuration(
    _ waveform: MLXArray,
    videoFrames: Int,
    fps: Double,
    sampleRate: Int
) -> MLXArray {
    guard waveform.ndim == 3,
          videoFrames > 0,
          fps > 0,
          sampleRate > 0 else {
        return waveform
    }

    let targetSamples = max(1, Int((Double(videoFrames) * Double(sampleRate) / fps).rounded()))
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
    videoConditioning: LTXLatentConditioningState?,
    ancestralNoiseSeed: Int? = nil
) -> (MLXArray, MLXArray) {
    var currentVideo = videoLatents
    var currentAudio = audioLatents
    let dtype = videoLatents.dtype
    if let ancestralNoiseSeed {
        MLXRandom.seed(UInt64(bitPattern: Int64(ancestralNoiseSeed)))
    }

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
            videoKeyframesMask: makeLTXVideoKeyframesMask(
                batchSize: b,
                tokenCount: videoTokenCount,
                tokensPerFirstFrame: h * w,
                dtype: flatVideo.dtype
            ),
            videoAttentionMask: nil,
            audioLatent: flatAudio,
            timestep: globalTimestep,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            audioSigma: globalTimestep,
            perturbation: .none
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

        if nextSigma > 0, ancestralNoiseSeed != nil {
            let videoNoise = MLXRandom.normal(currentVideo.shape).asType(dtype)
            let audioNoise = MLXRandom.normal(currentAudio.shape).asType(dtype)
            currentVideo = ltxAncestralEulerStep(
                sample: currentVideo,
                denoised: denoisedVideo,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: videoNoise
            )
            currentAudio = ltxAncestralEulerStep(
                sample: currentAudio,
                denoised: denoisedAudio,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: audioNoise
            )
            if let videoConditioning {
                let one = MLXArray(1.0).asType(currentVideo.dtype)
                currentVideo = currentVideo * videoConditioning.denoiseMask
                    + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
            }
        } else if nextSigma > 0 {
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

private func ltxAncestralEulerStep(
    sample: MLXArray,
    denoised: MLXArray,
    sigma: Float,
    nextSigma: Float,
    noise: MLXArray,
    eta: Float = 1
) -> MLXArray {
    guard nextSigma > 0 else {
        return denoised.asType(sample.dtype)
    }

    let downstepRatio = 1 + (nextSigma / sigma - 1) * eta
    let sigmaDown = nextSigma * downstepRatio
    let sigmaDownRatio = sigmaDown / sigma
    let alphaNext = 1.0 - nextSigma
    let alphaDown = 1.0 - sigmaDown
    let alphaRatio = alphaNext / alphaDown
    let variance = max(0, nextSigma * nextSigma - sigmaDown * sigmaDown * alphaRatio * alphaRatio)
    let renoiseCoefficient = sqrt(variance)

    let sample32 = sample.asType(.float32)
    let denoised32 = denoised.asType(.float32)
    var next = MLXArray(sigmaDownRatio) * sample32
        + MLXArray(1.0 - sigmaDownRatio) * denoised32
    next = MLXArray(alphaRatio) * next + MLXArray(renoiseCoefficient) * noise.asType(.float32)
    return next.asType(sample.dtype)
}

private func denoiseLTX25VideoTokenLoop(
    videoState: LTX25VideoTokenState,
    videoRope: LTXRope,
    videoContext: MLXArray,
    transformer: LTXUnifiedAVTransformerV2,
    sigmas: [Float],
    ancestralNoiseSeed: Int,
    ancestralEta: Float
) -> LTX25VideoTokenState {
    var current = videoState
    let dtype = videoState.latent.dtype
    MLXRandom.seed(UInt64(bitPattern: Int64(ancestralNoiseSeed)))

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let timesteps = current.denoiseMask.squeezed(axis: -1)
            * MLXArray(sigma).asType(dtype)
        let globalTimestep = MLX.full(
            [current.targetShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )
        let velocity = transformer.forwardVideoOnly(
            videoLatent: current.latent,
            videoKeyframesMask: current.keyframesMask,
            videoAttentionMask: current.attentionMask,
            timestep: globalTimestep,
            videoTimesteps: timesteps,
            videoContext: videoContext,
            videoRope: videoRope
        )
        var denoised = toDenoised(noisy: current.latent, velocity: velocity, sigma: sigma)
        let one = MLXArray(1).asType(dtype)
        denoised = denoised * current.denoiseMask + current.cleanLatent * (one - current.denoiseMask)
        if nextSigma > 0 {
            current.latent = ltxAncestralEulerStep(
                sample: current.latent,
                denoised: denoised,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(current.latent.shape).asType(dtype),
                eta: ancestralEta
            )
            current.latent = current.latent * current.denoiseMask
                + current.cleanLatent * (one - current.denoiseMask)
        } else {
            current.latent = denoised
        }
        MLX.eval(current.latent)
    }
    return current
}

private func splitLTXByCount(
    numTiles requestedTiles: Int,
    overlap requestedOverlap: Int,
    dimensionSize: Int
) -> LTXIntervals {
    guard requestedTiles > 1, dimensionSize > 1 else {
        return LTXIntervals(
            starts: [0],
            ends: [dimensionSize],
            leftRamps: [0],
            rightRamps: [0]
        )
    }
    let numTiles = min(requestedTiles, dimensionSize)
    let overlap = min(requestedOverlap, max(0, dimensionSize - numTiles))
    let total = dimensionSize + overlap * (numTiles - 1)
    let tileSize = total / numTiles
    guard tileSize > overlap else {
        return LTXIntervals(
            starts: [0],
            ends: [dimensionSize],
            leftRamps: [0],
            rightRamps: [0]
        )
    }
    let remainder = total % numTiles
    let base = splitInSpatial(
        size: tileSize,
        overlap: overlap,
        dimensionSize: dimensionSize - remainder
    )
    var starts: [Int] = []
    var ends: [Int] = []
    var leftRamps: [Int] = []
    var rightRamps: [Int] = []
    for index in base.starts.indices {
        let shift = min(index, remainder)
        let grow = index < remainder ? 1 : 0
        starts.append(base.starts[index] + shift)
        ends.append(base.ends[index] + shift + grow)
        leftRamps.append(base.leftRamps[index])
        rightRamps.append(base.rightRamps[index])
    }
    return LTXIntervals(
        starts: starts,
        ends: ends,
        leftRamps: leftRamps,
        rightRamps: rightRamps
    )
}

private func denoiseLTXHDRICLoRAStage2(
    initialLatent: MLXArray,
    phases: [LTXHDRICLoRAStage2Phase],
    referenceVideos: [LTXReferenceVideoConditioningInput],
    referenceLatents: [MLXArray],
    videoContext: MLXArray,
    transformer: LTXUnifiedAVTransformerV2,
    fps: Double,
    seed: Int
) throws -> MLXArray {
    precondition(referenceVideos.count == referenceLatents.count)
    var phaseLatent = initialLatent
    for (phaseIndex, phase) in phases.enumerated() {
        let sigmas = try validatedLTXSigmaSchedule(phase.sigmas)
        let tiling = phase.tiling
        let temporal = splitLTXByCount(
            numTiles: tiling.frameTiles,
            overlap: tiling.frameOverlap,
            dimensionSize: phaseLatent.dim(2)
        )
        let vertical = splitLTXByCount(
            numTiles: tiling.heightTiles,
            overlap: tiling.heightOverlap,
            dimensionSize: phaseLatent.dim(3)
        )
        let horizontal = splitLTXByCount(
            numTiles: tiling.widthTiles,
            overlap: tiling.widthOverlap,
            dimensionSize: phaseLatent.dim(4)
        )
        let output = MLX.zeros(phaseLatent.shape, dtype: phaseLatent.dtype)
        var tileIndex = 0

        for tIndex in temporal.starts.indices {
            let tStart = temporal.starts[tIndex]
            let tEnd = temporal.ends[tIndex]
            let tMask = computeTrapezoidalMask1D(
                length: tEnd - tStart,
                rampLeft: temporal.leftRamps[tIndex],
                rampRight: temporal.rightRamps[tIndex],
                leftStartsFromZero: false
            )
            for hIndex in vertical.starts.indices {
                let hStart = vertical.starts[hIndex]
                let hEnd = vertical.ends[hIndex]
                let hMask = computeTrapezoidalMask1D(
                    length: hEnd - hStart,
                    rampLeft: vertical.leftRamps[hIndex],
                    rampRight: vertical.rightRamps[hIndex],
                    leftStartsFromZero: false
                )
                for wIndex in horizontal.starts.indices {
                    let wStart = horizontal.starts[wIndex]
                    let wEnd = horizontal.ends[wIndex]
                    let wMask = computeTrapezoidalMask1D(
                        length: wEnd - wStart,
                        rampLeft: horizontal.leftRamps[wIndex],
                        rampRight: horizontal.rightRamps[wIndex],
                        leftStartsFromZero: false
                    )
                    let tileLatent = phaseLatent[
                        0...,
                        0...,
                        tStart..<tEnd,
                        hStart..<hEnd,
                        wStart..<wEnd
                    ]
                    let positions = createPositionGrid(
                        batchSize: 1,
                        numFrames: tEnd - tStart,
                        height: hEnd - hStart,
                        width: wEnd - wStart,
                        temporalScale: 8,
                        spatialScale: 32,
                        fps: Float(fps),
                        causalFix: true
                    )
                    var state = LTX25VideoTokenState(
                        initialLatent: tileLatent,
                        positions: positions
                    )
                    if phase.usesICLoRAConditioning {
                        for (reference, latent) in zip(referenceVideos, referenceLatents) {
                            let downscale = reference.downscaleFactor
                            let refTStart = min(tStart, latent.dim(2) - 1)
                            let refTEnd = min(max(refTStart + 1, tEnd), latent.dim(2))
                            let refHStart = min(hStart / downscale, latent.dim(3) - 1)
                            let refHEnd = min(max(refHStart + 1, hEnd / downscale), latent.dim(3))
                            let refWStart = min(wStart / downscale, latent.dim(4) - 1)
                            let refWEnd = min(max(refWStart + 1, wEnd / downscale), latent.dim(4))
                            state.appendReferenceLatent(
                                latent[
                                    0...,
                                    0...,
                                    refTStart..<refTEnd,
                                    refHStart..<refHEnd,
                                    refWStart..<refWEnd
                                ],
                                downscaleFactor: downscale,
                                temporalScaleFactor: reference.temporalScaleFactor,
                                strength: reference.strength,
                                attentionStrength: reference.attentionStrength,
                                fps: fps
                            )
                        }
                    }
                    MLXRandom.seed(UInt64(bitPattern: Int64(seed &+ tileIndex)))
                    state.addNoise(scale: sigmas[0])
                    let rope = precomputeSplitRope(
                        positions: state.positions,
                        dim: 4096,
                        theta: 10_000,
                        maxPos: [20, 2048, 2048],
                        numHeads: 32
                    )
                    state = denoiseLTX25VideoTokenLoop(
                        videoState: state,
                        videoRope: rope,
                        videoContext: videoContext,
                        transformer: transformer,
                        sigmas: sigmas,
                        ancestralNoiseSeed: seed &+ tileIndex,
                        ancestralEta: 0
                    )
                    let blend = MLXArray(tMask).asType(output.dtype).reshaped(1, 1, tMask.count, 1, 1)
                        * MLXArray(hMask).asType(output.dtype).reshaped(1, 1, 1, hMask.count, 1)
                        * MLXArray(wMask).asType(output.dtype).reshaped(1, 1, 1, 1, wMask.count)
                    output[0..., 0..., tStart..<tEnd, hStart..<hEnd, wStart..<wEnd] =
                        output[0..., 0..., tStart..<tEnd, hStart..<hEnd, wStart..<wEnd]
                        + state.mainLatent() * blend
                    MLX.eval(output)
                    Memory.clearCache()
                    tileIndex += 1
                }
            }
        }
        phaseLatent = output
        MLX.eval(phaseLatent)
        _ = phaseIndex
    }
    return phaseLatent
}

private func denoiseLTX25AVTokenLoop(
    videoState: LTX25VideoTokenState,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    videoContext: MLXArray,
    audioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    ancestralNoiseSeed: Int? = nil,
    audioConditioning: LTXLatentConditioningState? = nil
) -> (video: LTX25VideoTokenState, audio: MLXArray) {
    var currentVideo = videoState
    var currentAudio = audioLatents
    let dtype = videoState.latent.dtype
    if let ancestralNoiseSeed {
        MLXRandom.seed(UInt64(bitPattern: Int64(ancestralNoiseSeed)))
    }

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let audioShape = (
            batch: currentAudio.dim(0),
            channels: currentAudio.dim(1),
            frames: currentAudio.dim(2),
            melBins: currentAudio.dim(3)
        )
        let flatAudio = currentAudio
            .transposed(0, 2, 1, 3)
            .reshaped(audioShape.batch, audioShape.frames, audioShape.channels * audioShape.melBins)
        let videoTimesteps = currentVideo.denoiseMask.squeezed(axis: -1)
            * MLXArray(sigma).asType(dtype)
        let audioTimesteps = audioConditioning.map {
            $0.denoiseMask[0..., 0, 0..., 0] * MLXArray(sigma).asType(dtype)
        } ?? MLX.full(
            [audioShape.batch, audioShape.frames],
            values: MLXArray(sigma).asType(dtype)
        )
        let globalTimestep = MLX.full(
            [currentVideo.targetShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )
        let velocity = transformer.forward(
            videoLatent: currentVideo.latent,
            videoKeyframesMask: currentVideo.keyframesMask,
            videoAttentionMask: currentVideo.attentionMask,
            audioLatent: flatAudio,
            timestep: globalTimestep,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            audioSigma: globalTimestep,
            perturbation: .none
        )
        var denoisedVideo = toDenoised(
            noisy: currentVideo.latent,
            velocity: velocity.videoVelocity,
            sigma: sigma
        )
        let one = MLXArray(1).asType(dtype)
        denoisedVideo = denoisedVideo * currentVideo.denoiseMask
            + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
        let audioVelocity = velocity.audioVelocity
            .reshaped(audioShape.batch, audioShape.frames, audioShape.channels, audioShape.melBins)
            .transposed(0, 2, 1, 3)
        var denoisedAudio = toDenoised(
            noisy: currentAudio,
            velocity: audioVelocity,
            sigma: sigma
        )
        if let audioConditioning {
            denoisedAudio = denoisedAudio * audioConditioning.denoiseMask
                + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
        }

        if nextSigma > 0, ancestralNoiseSeed != nil {
            currentVideo.latent = ltxAncestralEulerStep(
                sample: currentVideo.latent,
                denoised: denoisedVideo,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentVideo.latent.shape).asType(dtype)
            )
            currentAudio = ltxAncestralEulerStep(
                sample: currentAudio,
                denoised: denoisedAudio,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentAudio.shape).asType(dtype)
            )
            currentVideo.latent = currentVideo.latent * currentVideo.denoiseMask
                + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
            if let audioConditioning {
                currentAudio = currentAudio * audioConditioning.denoiseMask
                    + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
            }
        } else if nextSigma > 0 {
            let sigmaArray = MLXArray(sigma).asType(dtype)
            let nextArray = MLXArray(nextSigma).asType(dtype)
            currentVideo.latent = denoisedVideo
                + nextArray * (currentVideo.latent - denoisedVideo) / sigmaArray
            currentAudio = denoisedAudio
                + nextArray * (currentAudio - denoisedAudio) / sigmaArray
        } else {
            currentVideo.latent = denoisedVideo
            currentAudio = denoisedAudio
        }
        if let audioConditioning {
            currentAudio = currentAudio * audioConditioning.denoiseMask
                + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
        }
        MLX.eval(currentVideo.latent, currentAudio)
    }
    return (currentVideo, currentAudio)
}

private func predictLTX25JointAVTokenDenoised(
    videoState: LTX25VideoTokenState,
    flatAudio: MLXArray,
    videoTimesteps: MLXArray,
    audioTimesteps: MLXArray,
    globalTimestep: MLXArray,
    videoContext: MLXArray,
    audioContext: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    transformer: any LTXUnifiedAVTransformerRuntime,
    perturbation: LTXAudioToVideoPerturbation,
    audioShape: (batch: Int, channels: Int, frames: Int, melBins: Int)
) -> (video: MLXArray, audio: MLXArray) {
    let output = transformer.forward(
        videoLatent: videoState.latent,
        videoKeyframesMask: videoState.keyframesMask,
        videoAttentionMask: videoState.attentionMask,
        audioLatent: flatAudio,
        timestep: globalTimestep,
        videoTimesteps: videoTimesteps,
        audioTimesteps: audioTimesteps,
        videoContext: videoContext,
        audioContext: audioContext,
        videoRope: videoRope,
        audioRope: audioRope,
        videoCrossRope: videoCrossRope,
        audioCrossRope: audioCrossRope,
        audioSigma: globalTimestep,
        perturbation: perturbation
    )
    let video = (
        videoState.latent.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(videoState.latent.dtype)
    let audio = (
        flatAudio.asType(.float32)
            - audioTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.audioVelocity.asType(.float32)
    ).asType(flatAudio.dtype)
        .reshaped(audioShape.batch, audioShape.frames, audioShape.channels, audioShape.melBins)
        .transposed(0, 2, 1, 3)
    MLX.eval(video, audio)
    return (video, audio)
}

private struct LTXGuidedAVPrediction {
    let video: MLXArray
    let audio: MLXArray
    let unconditionalVideo: MLXArray
    let unconditionalAudio: MLXArray
}

private final class LTXGuidanceProjectionCacheMetrics {
    var buildSeconds = 0.0
    var buildCount = 0
    var reuseCount = 0
    var fallbackCount = 0
}

private func predictGuidedLTX25AV(
    videoState: LTX25VideoTokenState,
    audioLatents: MLXArray,
    sigma: Float,
    stepIndex: Int,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray,
    positiveAudioContext: MLXArray,
    negativeAudioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    videoGuidance: LTXMultiModalGuidance,
    audioGuidance: LTXMultiModalGuidance,
    guidanceProjectionCache: LTXGuidanceProjectionCacheMode,
    guidanceProjectionCacheMetrics: LTXGuidanceProjectionCacheMetrics,
    teaCacheController: LTXTeaCacheController?,
    teaCacheStage: LTXTeaCacheStage,
    teaCachePipelineStage: LTXTeaCachePipelineStage,
    teaCacheStepCount: Int,
    audioConditioning: LTXLatentConditioningState?,
    forceUnconditional: Bool,
    lastVideo: MLXArray?,
    lastAudio: MLXArray?
) -> LTXGuidedAVPrediction {
    let dtype = videoState.latent.dtype
    let audioShape = (
        batch: audioLatents.dim(0),
        channels: audioLatents.dim(1),
        frames: audioLatents.dim(2),
        melBins: audioLatents.dim(3)
    )
    let flatAudio = audioLatents
        .transposed(0, 2, 1, 3)
        .reshaped(audioShape.batch, audioShape.frames, audioShape.channels * audioShape.melBins)
    let videoTimesteps = videoState.denoiseMask.squeezed(axis: -1)
        * MLXArray(sigma).asType(dtype)
    let audioTimesteps = audioConditioning.map {
        $0.denoiseMask[0..., 0, 0..., 0] * MLXArray(sigma).asType(dtype)
    } ?? MLX.full(
        [audioShape.batch, audioShape.frames],
        values: MLXArray(sigma).asType(dtype)
    )
    let globalTimestep = MLX.full(
        [videoState.targetShape.batch],
        values: MLXArray(sigma).asType(dtype)
    )

    let needsUnconditional = forceUnconditional
        || videoGuidance.classifierFreeScale != 1
        || audioGuidance.classifierFreeScale != 1
    let needsPerturbed = videoGuidance.spatioTemporalScale != 0
        || audioGuidance.spatioTemporalScale != 0
    let needsIsolated = videoGuidance.modalityScale != 1 || audioGuidance.modalityScale != 1
    let positivePredictionCount = 1 + (needsPerturbed ? 1 : 0) + (needsIsolated ? 1 : 0)
    let transformerV2 = transformer as? LTXUnifiedAVTransformerV2
    let cacheDecision = ltxGuidanceProjectionCacheDecision(
        mode: guidanceProjectionCache,
        positivePredictionCount: positivePredictionCount,
        batchSize: videoState.targetShape.batch,
        videoTextTokens: positiveVideoContext.dim(1),
        audioTextTokens: positiveAudioContext.dim(1),
        blockCount: 48,
        bytesPerElement: 2,
        activeMemoryBytes: UInt64(Memory.activeMemory),
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )
    let textProjectionCache: LTXV2TextProjectionCache?
    if cacheDecision.shouldCache, let transformerV2 {
        let buildStart = ltxMonotonicSeconds()
        textProjectionCache = transformerV2.prepareTextProjectionCache(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            timestep: globalTimestep,
            audioSigma: globalTimestep
        )
        guidanceProjectionCacheMetrics.buildSeconds += ltxMonotonicSeconds() - buildStart
        guidanceProjectionCacheMetrics.buildCount += 1
        guidanceProjectionCacheMetrics.reuseCount += positivePredictionCount - 1
    } else {
        textProjectionCache = nil
        if guidanceProjectionCache != .disabled, positivePredictionCount > 1 {
            guidanceProjectionCacheMetrics.fallbackCount += 1
        }
    }
    defer {
        transformerV2?.useTextProjectionCache(nil)
        transformerV2?.useTeaCache(controller: nil, request: nil)
    }

    func predict(
        videoContext: MLXArray,
        audioContext: MLXArray,
        perturbation: LTXAudioToVideoPerturbation,
        usePositiveTextProjectionCache: Bool,
        teaCacheBranch: LTXTeaCacheBranch
    ) -> (video: MLXArray, audio: MLXArray) {
        transformerV2?.useTextProjectionCache(
            usePositiveTextProjectionCache ? textProjectionCache : nil
        )
        transformerV2?.useTeaCache(
            controller: teaCacheController,
            request: teaCacheController.map { _ in
                LTXTeaCacheRequest(
                    key: LTXTeaCacheKey(
                        branch: teaCacheBranch,
                        stage: teaCacheStage,
                        pipelineStage: teaCachePipelineStage
                    ),
                    stepIndex: stepIndex,
                    stepCount: teaCacheStepCount
                )
            }
        )
        return predictLTX25JointAVTokenDenoised(
            videoState: videoState,
            flatAudio: flatAudio,
            videoTimesteps: videoTimesteps,
            audioTimesteps: audioTimesteps,
            globalTimestep: globalTimestep,
            videoContext: videoContext,
            audioContext: audioContext,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            transformer: transformer,
            perturbation: perturbation,
            audioShape: audioShape
        )
    }

    let conditioned = predict(
        videoContext: positiveVideoContext,
        audioContext: positiveAudioContext,
        perturbation: .none,
        usePositiveTextProjectionCache: true,
        teaCacheBranch: .conditioned
    )
    let negative = needsUnconditional ? predict(
        videoContext: negativeVideoContext,
        audioContext: negativeAudioContext,
        perturbation: .none,
        usePositiveTextProjectionCache: false,
        teaCacheBranch: .unconditional
    ) : conditioned
    let perturbed = needsPerturbed ? predict(
        videoContext: positiveVideoContext,
        audioContext: positiveAudioContext,
        perturbation: .spatioTemporal(
            videoBlocks: videoGuidance.spatioTemporalBlocks,
            audioBlocks: audioGuidance.spatioTemporalBlocks
        ),
        usePositiveTextProjectionCache: true,
        teaCacheBranch: .perturbed
    ) : conditioned
    let isolated = needsIsolated ? predict(
        videoContext: positiveVideoContext,
        audioContext: positiveAudioContext,
        perturbation: .isolatedModalities,
        usePositiveTextProjectionCache: true,
        teaCacheBranch: .isolated
    ) : conditioned

    var guidedVideo = videoGuidance.shouldSkip(step: stepIndex)
        ? (lastVideo ?? conditioned.video)
        : videoGuidance.combine(
            conditioned: conditioned.video,
            negativeText: negative.video,
            perturbed: perturbed.video,
            isolatedModality: isolated.video
        )
    var guidedAudio = audioGuidance.shouldSkip(step: stepIndex)
        ? (lastAudio ?? conditioned.audio)
        : audioGuidance.combine(
            conditioned: conditioned.audio,
            negativeText: negative.audio,
            perturbed: perturbed.audio,
            isolatedModality: isolated.audio
        )
    let one = MLXArray(1).asType(dtype)
    guidedVideo = guidedVideo * videoState.denoiseMask
        + videoState.cleanLatent * (one - videoState.denoiseMask)
    if let audioConditioning {
        guidedAudio = guidedAudio * audioConditioning.denoiseMask
            + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
    }
    MLX.eval(guidedVideo, guidedAudio, negative.video, negative.audio)
    return LTXGuidedAVPrediction(
        video: guidedVideo,
        audio: guidedAudio,
        unconditionalVideo: negative.video,
        unconditionalAudio: negative.audio
    )
}

private struct LTXRandomKeyStream {
    private var key: MLXArray

    init(seed: Int) {
        key = MLXRandom.key(UInt64(bitPattern: Int64(seed)))
    }

    mutating func normal(shape: [Int]) -> MLXArray {
        let (nextKey, drawKey) = MLXRandom.split(key: key)
        key = nextKey
        return MLXRandom.normal(shape, key: drawKey)
    }
}

private func ltxNormalizedRes2sNoise(
    shape: [Int],
    dtype: DType,
    stream: inout LTXRandomKeyStream
) -> MLXArray {
    var noise = stream.normal(shape: shape).asType(.float32)
    let globalMean = MLX.mean(noise)
    let centered = noise - globalMean
    let elementCount = max(1, shape.reduce(1, *))
    let globalCorrection = elementCount > 1
        ? Float(elementCount) / Float(elementCount - 1)
        : 1
    let globalVariance = MLX.mean(centered * centered) * MLXArray(globalCorrection)
    noise = centered / MLX.sqrt(globalVariance + MLXArray(1e-12))
    let axes = [max(0, shape.count - 2), max(0, shape.count - 1)]
    let channelMean = MLX.mean(noise, axes: axes, keepDims: true)
    let channelCentered = noise - channelMean
    let channelElementCount = max(1, shape[axes[0]] * shape[axes[1]])
    let channelCorrection = channelElementCount > 1
        ? Float(channelElementCount) / Float(channelElementCount - 1)
        : 1
    let channelVariance = MLX.mean(
        channelCentered * channelCentered,
        axes: axes,
        keepDims: true
    ) * MLXArray(channelCorrection)
    return (channelCentered / MLX.sqrt(channelVariance + MLXArray(1e-12))).asType(dtype)
}

private func denoiseGuidedLTX25AVTokenLoop(
    videoState: LTX25VideoTokenState,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray,
    positiveAudioContext: MLXArray,
    negativeAudioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoGuidance: LTXMultiModalGuidance,
    audioGuidance: LTXMultiModalGuidance,
    sampler: LTXSamplerConfiguration,
    seed: Int,
    guidanceProjectionCache: LTXGuidanceProjectionCacheMode,
    guidanceProjectionCacheMetrics: LTXGuidanceProjectionCacheMetrics,
    teaCacheController: LTXTeaCacheController?,
    teaCachePipelineStage: LTXTeaCachePipelineStage,
    audioConditioning: LTXLatentConditioningState? = nil
) -> (video: LTX25VideoTokenState, audio: MLXArray) {
    var currentVideo = videoState
    var currentAudio = audioLatents
    let dtype = videoState.latent.dtype

    MLXRandom.seed(UInt64(bitPattern: Int64(seed &+ sampler.noiseSeedOffset)))
    var lastVideo: MLXArray?
    var lastAudio: MLXArray?
    var previousVideoVelocity: MLXArray?
    var previousAudioVelocity: MLXArray?

    func prediction(
        video: LTX25VideoTokenState,
        audio: MLXArray,
        sigma: Float,
        stepIndex: Int,
        teaCacheStage: LTXTeaCacheStage = .primary,
        teaCacheStepCount: Int,
        forceUnconditional: Bool = false,
        cachedVideo: MLXArray? = nil,
        cachedAudio: MLXArray? = nil
    ) -> LTXGuidedAVPrediction {
        predictGuidedLTX25AV(
            videoState: video,
            audioLatents: audio,
            sigma: sigma,
            stepIndex: stepIndex,
            videoRope: videoRope,
            audioRope: audioRope,
            videoCrossRope: videoCrossRope,
            audioCrossRope: audioCrossRope,
            positiveVideoContext: positiveVideoContext,
            negativeVideoContext: negativeVideoContext,
            positiveAudioContext: positiveAudioContext,
            negativeAudioContext: negativeAudioContext,
            transformer: transformer,
            videoGuidance: videoGuidance,
            audioGuidance: audioGuidance,
            guidanceProjectionCache: guidanceProjectionCache,
            guidanceProjectionCacheMetrics: guidanceProjectionCacheMetrics,
            teaCacheController: teaCacheController,
            teaCacheStage: teaCacheStage,
            teaCachePipelineStage: teaCachePipelineStage,
            teaCacheStepCount: teaCacheStepCount,
            audioConditioning: audioConditioning,
            forceUnconditional: forceUnconditional,
            lastVideo: cachedVideo,
            lastAudio: cachedAudio
        )
    }

    if sampler.mode == .res2s {
        let fullStepCount = sigmas.count - 1
        var workingSigmas = sigmas
        if workingSigmas.last == 0 {
            workingSigmas.removeLast()
            workingSigmas.append(contentsOf: [0.0011, 0])
        }
        var stepNoiseStream = LTXRandomKeyStream(seed: sampler.noiseSeedOffset)
        var substepNoiseStream = LTXRandomKeyStream(seed: sampler.substepNoiseSeedOffset)
        for index in 0..<fullStepCount {
            let sigma = workingSigmas[index]
            let nextSigma = workingSigmas[index + 1]
            var anchorVideo = currentVideo.latent.asType(.float32)
            var anchorAudio = currentAudio.asType(.float32)
            let first = prediction(
                video: currentVideo,
                audio: currentAudio,
                sigma: sigma,
                stepIndex: index,
                teaCacheStepCount: fullStepCount + 1,
                cachedVideo: lastVideo,
                cachedAudio: lastAudio
            )
            lastVideo = first.video
            lastAudio = first.audio
            let step = -log(Double(nextSigma) / Double(sigma))
            let coefficients = LTXRes2s.coefficients(step: step)
            var epsilonVideo = first.video.asType(.float32) - anchorVideo
            var epsilonAudio = first.audio.asType(.float32) - anchorAudio
            let midpointSigma = sqrt(sigma * nextSigma)
            var midpointVideo = anchorVideo
                + MLXArray(Float(step * coefficients.a21)) * epsilonVideo
            var midpointAudio = anchorAudio
                + MLXArray(Float(step * coefficients.a21)) * epsilonAudio
            midpointVideo = ltxRes2sSDEStep(
                sample: anchorVideo,
                denoised: midpointVideo,
                sigma: sigma,
                nextSigma: midpointSigma,
                eta: 0.5,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorVideo.shape,
                    dtype: dtype,
                    stream: &substepNoiseStream
                )
            )
            midpointAudio = ltxRes2sSDEStep(
                sample: anchorAudio,
                denoised: midpointAudio,
                sigma: sigma,
                nextSigma: midpointSigma,
                eta: 0.5,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorAudio.shape,
                    dtype: dtype,
                    stream: &substepNoiseStream
                )
            )
            if sampler.res2sBongMath, step < 0.5, sigma > 0.03 {
                for _ in 0..<sampler.res2sBongMathMaxIterations {
                    anchorVideo = midpointVideo.asType(.float32)
                        - MLXArray(Float(step * coefficients.a21)) * epsilonVideo
                    anchorAudio = midpointAudio.asType(.float32)
                        - MLXArray(Float(step * coefficients.a21)) * epsilonAudio
                    epsilonVideo = first.video.asType(.float32) - anchorVideo
                    epsilonAudio = first.audio.asType(.float32) - anchorAudio
                }
            }
            var midpointState = currentVideo
            midpointState.latent = midpointVideo.asType(dtype)
            let second = prediction(
                video: midpointState,
                audio: midpointAudio.asType(dtype),
                sigma: midpointSigma,
                stepIndex: index,
                teaCacheStage: .midpoint,
                teaCacheStepCount: fullStepCount
            )
            let nextVideoEstimate = anchorVideo + MLXArray(Float(step))
                * (MLXArray(Float(coefficients.b1)) * epsilonVideo
                    + MLXArray(Float(coefficients.b2))
                        * (second.video.asType(.float32) - anchorVideo))
            let nextAudioEstimate = anchorAudio + MLXArray(Float(step))
                * (MLXArray(Float(coefficients.b1)) * epsilonAudio
                    + MLXArray(Float(coefficients.b2))
                        * (second.audio.asType(.float32) - anchorAudio))
            currentVideo.latent = ltxRes2sSDEStep(
                sample: anchorVideo,
                denoised: nextVideoEstimate,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorVideo.shape,
                    dtype: dtype,
                    stream: &stepNoiseStream
                )
            )
            currentAudio = ltxRes2sSDEStep(
                sample: anchorAudio,
                denoised: nextAudioEstimate,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: ltxNormalizedRes2sNoise(
                    shape: anchorAudio.shape,
                    dtype: dtype,
                    stream: &stepNoiseStream
                )
            )
            let one = MLXArray(1).asType(dtype)
            currentVideo.latent = currentVideo.latent * currentVideo.denoiseMask
                + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
            if let audioConditioning {
                currentAudio = currentAudio * audioConditioning.denoiseMask
                    + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
            }
            MLX.eval(currentVideo.latent, currentAudio)
        }
        if workingSigmas.last == 0 {
            let finalSigma = workingSigmas[fullStepCount]
            let final = prediction(
                video: currentVideo,
                audio: currentAudio,
                sigma: finalSigma,
                stepIndex: fullStepCount,
                teaCacheStepCount: fullStepCount + 1,
                cachedVideo: lastVideo,
                cachedAudio: lastAudio
            )
            currentVideo.latent = final.video
            currentAudio = final.audio
            MLX.eval(currentVideo.latent, currentAudio)
        }
        return (currentVideo, currentAudio)
    }

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        var result = prediction(
            video: currentVideo,
            audio: currentAudio,
            sigma: sigma,
            stepIndex: index,
            teaCacheStepCount: max(1, sigmas.count - 1),
            forceUnconditional: sampler.mode == .cfgPlusPlus,
            cachedVideo: lastVideo,
            cachedAudio: lastAudio
        )
        lastVideo = result.video
        lastAudio = result.audio

        if sampler.mode == .gradientEstimatingEuler {
            let videoVelocity = (
                currentVideo.latent.asType(.float32) - result.video.asType(.float32)
            ) / MLXArray(sigma)
            let audioVelocity = (
                currentAudio.asType(.float32) - result.audio.asType(.float32)
            ) / MLXArray(sigma)
            if let previousVideoVelocity, let previousAudioVelocity {
                let correctedVideo = previousVideoVelocity
                    + MLXArray(sampler.gradientEstimationGamma)
                        * (videoVelocity - previousVideoVelocity)
                let correctedAudio = previousAudioVelocity
                    + MLXArray(sampler.gradientEstimationGamma)
                        * (audioVelocity - previousAudioVelocity)
                result = LTXGuidedAVPrediction(
                    video: currentVideo.latent.asType(.float32)
                        - MLXArray(sigma) * correctedVideo,
                    audio: currentAudio.asType(.float32)
                        - MLXArray(sigma) * correctedAudio,
                    unconditionalVideo: result.unconditionalVideo,
                    unconditionalAudio: result.unconditionalAudio
                )
            }
            previousVideoVelocity = videoVelocity
            previousAudioVelocity = audioVelocity
        }

        switch sampler.mode {
        case .euler, .gradientEstimatingEuler:
            currentVideo.latent = ltxEulerStep(
                sample: currentVideo.latent,
                denoised: result.video,
                sigma: sigma,
                nextSigma: nextSigma
            )
            currentAudio = ltxEulerStep(
                sample: currentAudio,
                denoised: result.audio,
                sigma: sigma,
                nextSigma: nextSigma
            )
        case .eulerAncestral:
            currentVideo.latent = ltxAncestralEulerStep(
                sample: currentVideo.latent,
                denoised: result.video,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentVideo.latent.shape).asType(dtype),
                eta: sampler.eta
            )
            currentAudio = ltxAncestralEulerStep(
                sample: currentAudio,
                denoised: result.audio,
                sigma: sigma,
                nextSigma: nextSigma,
                noise: MLXRandom.normal(currentAudio.shape).asType(dtype),
                eta: sampler.eta
            )
        case .cfgPlusPlus:
            currentVideo.latent = ltxCfgPlusPlusStep(
                sample: currentVideo.latent,
                denoised: result.video,
                unconditionalDenoised: result.unconditionalVideo,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: MLXRandom.normal(currentVideo.latent.shape).asType(dtype)
            )
            currentAudio = ltxCfgPlusPlusStep(
                sample: currentAudio,
                denoised: result.audio,
                unconditionalDenoised: result.unconditionalAudio,
                sigma: sigma,
                nextSigma: nextSigma,
                eta: sampler.eta,
                noise: MLXRandom.normal(currentAudio.shape).asType(dtype)
            )
        case .res2s:
            preconditionFailure("Res2s is handled by the dedicated second-order loop.")
        }
        let one = MLXArray(1).asType(dtype)
        currentVideo.latent = currentVideo.latent * currentVideo.denoiseMask
            + currentVideo.cleanLatent * (one - currentVideo.denoiseMask)
        if let audioConditioning {
            currentAudio = currentAudio * audioConditioning.denoiseMask
                + audioConditioning.cleanLatent * (one - audioConditioning.denoiseMask)
        }
        MLX.eval(currentVideo.latent, currentAudio)
    }
    return (currentVideo, currentAudio)
}

private func predictJointAVDenoised(
    flatVideo: MLXArray,
    flatAudio: MLXArray,
    videoTimesteps: MLXArray,
    audioTimesteps: MLXArray,
    globalTimestep: MLXArray,
    videoContext: MLXArray,
    audioContext: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    transformer: any LTXUnifiedAVTransformerRuntime,
    perturbation: LTXAudioToVideoPerturbation,
    videoShape: (batch: Int, channels: Int, frames: Int, height: Int, width: Int),
    audioShape: (batch: Int, channels: Int, frames: Int, melBins: Int)
) -> (video: MLXArray, audio: MLXArray) {
    let output = transformer.forward(
        videoLatent: flatVideo,
        videoKeyframesMask: makeLTXVideoKeyframesMask(
            batchSize: videoShape.batch,
            tokenCount: flatVideo.dim(1),
            tokensPerFirstFrame: videoShape.height * videoShape.width,
            dtype: flatVideo.dtype
        ),
        videoAttentionMask: nil,
        audioLatent: flatAudio,
        timestep: globalTimestep,
        videoTimesteps: videoTimesteps,
        audioTimesteps: audioTimesteps,
        videoContext: videoContext,
        audioContext: audioContext,
        videoRope: videoRope,
        audioRope: audioRope,
        videoCrossRope: videoCrossRope,
        audioCrossRope: audioCrossRope,
        audioSigma: globalTimestep,
        perturbation: perturbation
    )
    let videoFlat = (
        flatVideo.asType(.float32)
            - videoTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.videoVelocity.asType(.float32)
    ).asType(flatVideo.dtype)
    let audioFlat = (
        flatAudio.asType(.float32)
            - audioTimesteps.expandedDimensions(axis: 2).asType(.float32)
                * output.audioVelocity.asType(.float32)
    ).asType(flatAudio.dtype)
    let video = videoFlat
        .reshaped(
            videoShape.batch,
            videoShape.frames,
            videoShape.height,
            videoShape.width,
            videoShape.channels
        )
        .transposed(0, 4, 1, 2, 3)
    let audio = audioFlat
        .reshaped(
            audioShape.batch,
            audioShape.frames,
            audioShape.channels,
            audioShape.melBins
        )
        .transposed(0, 2, 1, 3)
    MLX.eval(video, audio)
    return (video, audio)
}

private func denoiseGuidedAVLoop(
    videoLatents: MLXArray,
    audioLatents: MLXArray,
    videoRope: LTXRope,
    audioRope: LTXRope,
    videoCrossRope: LTXRope,
    audioCrossRope: LTXRope,
    positiveVideoContext: MLXArray,
    negativeVideoContext: MLXArray,
    positiveAudioContext: MLXArray,
    negativeAudioContext: MLXArray,
    transformer: any LTXUnifiedAVTransformerRuntime,
    sigmas: [Float],
    videoConditioning: LTXLatentConditioningState?,
    videoGuidance: LTXMultiModalGuidance,
    audioGuidance: LTXMultiModalGuidance
) -> (video: MLXArray, audio: MLXArray) {
    var currentVideo = videoLatents
    var currentAudio = audioLatents
    let dtype = videoLatents.dtype

    for index in 0..<(max(0, sigmas.count - 1)) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let videoShape = (
            batch: currentVideo.dim(0),
            channels: currentVideo.dim(1),
            frames: currentVideo.dim(2),
            height: currentVideo.dim(3),
            width: currentVideo.dim(4)
        )
        let audioShape = (
            batch: currentAudio.dim(0),
            channels: currentAudio.dim(1),
            frames: currentAudio.dim(2),
            melBins: currentAudio.dim(3)
        )
        let videoTokenCount = videoShape.frames * videoShape.height * videoShape.width
        let flatVideo = currentVideo
            .transposed(0, 2, 3, 4, 1)
            .reshaped(videoShape.batch, videoTokenCount, videoShape.channels)
        let flatAudio = currentAudio
            .transposed(0, 2, 1, 3)
            .reshaped(audioShape.batch, audioShape.frames, audioShape.channels * audioShape.melBins)

        let videoTimesteps: MLXArray
        if let videoConditioning {
            let mask = videoConditioning.denoiseMask
                .reshaped(videoShape.batch, 1, videoShape.frames, 1, 1)
            let flattenedMask = broadcast(
                mask,
                to: [
                    videoShape.batch,
                    1,
                    videoShape.frames,
                    videoShape.height,
                    videoShape.width,
                ]
            ).reshaped(videoShape.batch, videoTokenCount)
            videoTimesteps = MLXArray(sigma).asType(dtype) * flattenedMask
        } else {
            videoTimesteps = MLX.full(
                [videoShape.batch, videoTokenCount],
                values: MLXArray(sigma).asType(dtype)
            )
        }
        let audioTimesteps = MLX.full(
            [audioShape.batch, audioShape.frames],
            values: MLXArray(sigma).asType(dtype)
        )
        let globalTimestep = MLX.full(
            [videoShape.batch],
            values: MLXArray(sigma).asType(dtype)
        )

        func predict(
            videoContext: MLXArray,
            audioContext: MLXArray,
            perturbation: LTXAudioToVideoPerturbation
        ) -> (video: MLXArray, audio: MLXArray) {
            predictJointAVDenoised(
                flatVideo: flatVideo,
                flatAudio: flatAudio,
                videoTimesteps: videoTimesteps,
                audioTimesteps: audioTimesteps,
                globalTimestep: globalTimestep,
                videoContext: videoContext,
                audioContext: audioContext,
                videoRope: videoRope,
                audioRope: audioRope,
                videoCrossRope: videoCrossRope,
                audioCrossRope: audioCrossRope,
                transformer: transformer,
                perturbation: perturbation,
                videoShape: videoShape,
                audioShape: audioShape
            )
        }

        let conditioned = predict(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            perturbation: .none
        )
        let negative = predict(
            videoContext: negativeVideoContext,
            audioContext: negativeAudioContext,
            perturbation: .none
        )
        let perturbed = predict(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            perturbation: .spatioTemporal(
                videoBlocks: videoGuidance.spatioTemporalBlocks,
                audioBlocks: audioGuidance.spatioTemporalBlocks
            )
        )
        let isolated = predict(
            videoContext: positiveVideoContext,
            audioContext: positiveAudioContext,
            perturbation: .isolatedModalities
        )

        var denoisedVideo = videoGuidance.combine(
            conditioned: conditioned.video,
            negativeText: negative.video,
            perturbed: perturbed.video,
            isolatedModality: isolated.video
        )
        let denoisedAudio = audioGuidance.combine(
            conditioned: conditioned.audio,
            negativeText: negative.audio,
            perturbed: perturbed.audio,
            isolatedModality: isolated.audio
        )
        if let videoConditioning {
            let one = MLXArray(1).asType(denoisedVideo.dtype)
            denoisedVideo = denoisedVideo * videoConditioning.denoiseMask
                + videoConditioning.cleanLatent * (one - videoConditioning.denoiseMask)
        }

        let sigma32 = MLXArray(sigma)
        let delta32 = MLXArray(nextSigma - sigma)
        let videoVelocity = (currentVideo.asType(.float32) - denoisedVideo.asType(.float32)) / sigma32
        let audioVelocity = (currentAudio.asType(.float32) - denoisedAudio.asType(.float32)) / sigma32
        currentVideo = (currentVideo.asType(.float32) + videoVelocity * delta32).asType(dtype)
        currentAudio = (currentAudio.asType(.float32) + audioVelocity * delta32).asType(dtype)
        MLX.eval(currentVideo, currentAudio)
    }

    return (currentVideo, currentAudio)
}

private func computeAudioLatentFrameCount(videoFrames: Int, fps: Double) -> Int {
    let duration = Double(videoFrames) / max(1, fps)
    let latentsPerSecond = Double(LTXAudioLatentSampleRate) / Double(LTXAudioHopLength) / Double(LTXAudioLatentDownsampleFactor)
    return max(1, Int((duration * latentsPerSecond).rounded(.toNearestOrEven)))
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

protocol LTXUnifiedAVTransformerRuntime: Module {
    func forward(
        videoLatent: MLXArray,
        videoKeyframesMask: MLXArray?,
        videoAttentionMask: MLXArray?,
        audioLatent: MLXArray,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        audioSigma: MLXArray,
        perturbation: LTXAudioToVideoPerturbation
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray)
}

private func prepareLTXSelfAttentionMask(
    _ mask: MLXArray?,
    dtype: DType
) -> MLXArray? {
    guard let mask else { return nil }
    let typed = mask.asType(dtype)
    let epsilon = MLXArray(Float(1e-7)).asType(dtype)
    let negative = MLX.full(
        typed.shape,
        values: MLXArray(Float(-1e9)).asType(dtype)
    )
    let bias = MLX.where(
        typed .> MLXArray(0).asType(dtype),
        MLX.log(MLX.maximum(typed, epsilon)),
        negative
    )
    return bias.expandedDimensions(axis: 1)
}

public func validatedLTXSigmaSchedule(_ values: [Float]) throws -> [Float] {
    guard values.count >= 2,
          values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
          zip(values, values.dropFirst()).allSatisfy({ $0 >= $1 }),
          values.first! > 0,
          values.last == 0 else {
        throw LTXUnifiedAVGeneratorError.invalidSigmaSchedule(values)
    }
    return values
}

func isLTXAudioOnlyTransformerWeight(_ key: String) -> Bool {
    guard key.hasPrefix("model.diffusion_model.") else { return false }
    let mapped = String(key.dropFirst("model.diffusion_model.".count))
    if mapped.hasPrefix("audio_patchify_proj.")
        || mapped.hasPrefix("audio_adaln_single.")
        || mapped.hasPrefix("audio_prompt_adaln_single.")
        || mapped.hasPrefix("audio_scale_shift_table")
        || mapped.hasPrefix("audio_norm_out.")
        || mapped.hasPrefix("audio_proj_out.") {
        return true
    }
    guard mapped.hasPrefix("transformer_blocks.") else { return false }
    return mapped.contains(".audio_attn1.")
        || mapped.contains(".audio_attn2.")
        || mapped.contains(".audio_ff.")
        || mapped.hasSuffix(".audio_scale_shift_table")
        || mapped.hasSuffix(".audio_prompt_scale_shift_table")
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
        videoKeyframesMask _: MLXArray?,
        videoAttentionMask _: MLXArray?,
        audioLatent: MLXArray,
        timestep _: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        audioSigma _: MLXArray,
        perturbation _: LTXAudioToVideoPerturbation
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

private typealias LTXV2CompiledBlockForward = @Sendable ([MLXArray]) -> [MLXArray]

private struct LTXV2CompiledBlockVariant: Hashable {
    let hasVideoSelfAttentionMask: Bool
    let hasTextProjectionCache: Bool
    let skipsVideoSelfAttention: Bool
    let skipsAudioSelfAttention: Bool
    let skipsAudioToVideoCrossAttention: Bool
    let skipsVideoToAudioCrossAttention: Bool
}

private struct LTXV2TextProjectionCache {
    let video: [LTXAttentionProjectedContext]
    let audio: [LTXAttentionProjectedContext]
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
    var execution: LTXTransformerExecution = .eager
    private var parityForwardCount = 0
    private var compiledBlockRunner: LTXUnifiedAVTransformerV2Block?
    private var compiledBlockForwards: [LTXV2CompiledBlockVariant: LTXV2CompiledBlockForward] = [:]
    private var activeTextProjectionCache: LTXV2TextProjectionCache?
    private var activeTeaCacheController: LTXTeaCacheController?
    private var activeTeaCacheRequest: LTXTeaCacheRequest?

    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "keyframes_abs_pos_embedding") var keyframesAbsPosEmbedding: MLXArray
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
        self._keyframesAbsPosEmbedding.wrappedValue = MLX.zeros([1, videoDim], dtype: .float32)
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

    func prepareTextProjectionCache(
        videoContext: MLXArray,
        audioContext: MLXArray,
        timestep: MLXArray,
        audioSigma: MLXArray
    ) -> LTXV2TextProjectionCache {
        let videoSigma = timestep.asType(.bfloat16).reshaped(-1)
        let typedAudioSigma = audioSigma.asType(.bfloat16).reshaped(-1)
        let scaledVideoSigma = videoSigma * MLXArray(timestepScaleMultiplier).asType(.bfloat16)
        let scaledAudioSigma = typedAudioSigma * MLXArray(timestepScaleMultiplier).asType(.bfloat16)
        let (videoPromptParams, _) = promptAdalnSingle(
            timestep: scaledVideoSigma,
            hiddenDType: .bfloat16
        )
        let (audioPromptParams, _) = audioPromptAdalnSingle(
            timestep: scaledAudioSigma,
            hiddenDType: .bfloat16
        )
        let videoText = videoContext.asType(.bfloat16)
        let audioText = audioContext.asType(.bfloat16)
        var videoProjections: [LTXAttentionProjectedContext] = []
        var audioProjections: [LTXAttentionProjectedContext] = []
        videoProjections.reserveCapacity(transformerBlocks.count)
        audioProjections.reserveCapacity(transformerBlocks.count)
        for block in transformerBlocks {
            let projections = block.projectTextContexts(
                videoTextEmbeds: videoText,
                audioTextEmbeds: audioText,
                videoPromptAdalnParams: videoPromptParams,
                audioPromptAdalnParams: audioPromptParams
            )
            videoProjections.append(projections.video)
            audioProjections.append(projections.audio)
        }
        MLX.eval(
            videoProjections.flatMap { [$0.keys, $0.values] }
                + audioProjections.flatMap { [$0.keys, $0.values] }
        )
        return LTXV2TextProjectionCache(video: videoProjections, audio: audioProjections)
    }

    func useTextProjectionCache(_ cache: LTXV2TextProjectionCache?) {
        activeTextProjectionCache = cache
    }

    func useTeaCache(
        controller: LTXTeaCacheController?,
        request: LTXTeaCacheRequest?
    ) {
        activeTeaCacheController = controller
        activeTeaCacheRequest = request
    }

    func forward(
        videoLatent: MLXArray,
        videoKeyframesMask: MLXArray?,
        videoAttentionMask: MLXArray?,
        audioLatent: MLXArray,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        audioTimesteps: MLXArray?,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoRope: (cos: MLXArray, sin: MLXArray),
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        audioSigma: MLXArray,
        perturbation: LTXAudioToVideoPerturbation
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let parityIO = LTXAudioToVideoParityIO()
        let parityForwardIndex = parityForwardCount
        parityForwardCount += 1
        func paritySave(_ array: MLXArray, _ name: String) {
            try? parityIO.save(array, suffix: "forward\(parityForwardIndex)_\(name)")
        }

        let videoBatch = videoLatent.dim(0)
        let videoTokens = videoLatent.dim(1)
        let audioBatch = audioLatent.dim(0)
        let audioTokens = audioLatent.dim(1)

        var videoX = patchifyProj(videoLatent.asType(.bfloat16))
        if let videoKeyframesMask {
            let marker = videoKeyframesMask.asType(videoX.dtype)
            videoX = videoX + marker * keyframesAbsPosEmbedding.asType(videoX.dtype)
        }
        var audioX = audioPatchifyProj(audioLatent.asType(.bfloat16))
        paritySave(videoX, "patchified_video")
        paritySave(audioX, "patchified_audio")
        let videoSigma = timestep.asType(videoX.dtype).reshaped(-1)
        let audioSigma = audioSigma.asType(audioX.dtype).reshaped(-1)
        let scaledVideoSigma = videoSigma * MLXArray(timestepScaleMultiplier).asType(videoSigma.dtype)
        let scaledAudioSigma = audioSigma * MLXArray(timestepScaleMultiplier).asType(audioSigma.dtype)
        let scaledA2VGate = audioSigma * MLXArray(avCaTimestepScaleMultiplier).asType(audioSigma.dtype)
        let scaledV2AGate = videoSigma * MLXArray(avCaTimestepScaleMultiplier).asType(videoSigma.dtype)

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
            (videoAdalnParams, videoEmbedded) = adalnSingle(timestep: scaledVideoSigma, hiddenDType: videoX.dtype)
            (avCaVideoParams, _) = avCaVideoScaleShiftAdalnSingle(timestep: scaledVideoSigma, hiddenDType: videoX.dtype)
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
            (audioAdalnParams, audioEmbedded) = audioAdalnSingle(timestep: scaledAudioSigma, hiddenDType: audioX.dtype)
            (avCaAudioParams, _) = avCaAudioScaleShiftAdalnSingle(timestep: scaledAudioSigma, hiddenDType: audioX.dtype)
        }

        let (avCaA2VGateParams, _) = avCaA2VGateAdalnSingle(timestep: scaledA2VGate, hiddenDType: videoX.dtype)
        let (avCaV2AGateParams, _) = avCaV2AGateAdalnSingle(timestep: scaledV2AGate, hiddenDType: audioX.dtype)
        let (videoPromptParams, _) = promptAdalnSingle(timestep: scaledVideoSigma, hiddenDType: videoX.dtype)
        let (audioPromptParams, _) = audioPromptAdalnSingle(timestep: scaledAudioSigma, hiddenDType: audioX.dtype)
        paritySave(videoAdalnParams, "video_adaln")
        paritySave(audioAdalnParams, "audio_adaln")
        paritySave(avCaVideoParams, "av_video_adaln")
        paritySave(avCaAudioParams, "av_audio_adaln")
        paritySave(avCaA2VGateParams, "a2v_gate_adaln")
        paritySave(avCaV2AGateParams, "v2a_gate_adaln")
        paritySave(videoPromptParams, "video_prompt_adaln")
        paritySave(audioPromptParams, "audio_prompt_adaln")
        paritySave(videoRope.cos, "video_rope_cos")
        paritySave(videoRope.sin, "video_rope_sin")
        paritySave(audioRope.cos, "audio_rope_cos")
        paritySave(audioRope.sin, "audio_rope_sin")
        paritySave(videoCrossRope.cos, "video_cross_rope_cos")
        paritySave(videoCrossRope.sin, "video_cross_rope_sin")
        paritySave(audioCrossRope.cos, "audio_cross_rope_cos")
        paritySave(audioCrossRope.sin, "audio_cross_rope_sin")

        let videoSelfAttentionMask = prepareLTXSelfAttentionMask(
            videoAttentionMask,
            dtype: videoX.dtype
        )
        let blockInputVideo = videoX
        let blockInputAudio = audioX
        let teaCacheDecision: LTXTeaCacheDecision = if let activeTeaCacheController,
                                                       let activeTeaCacheRequest,
                                                       let firstBlock = transformerBlocks.first {
            activeTeaCacheController.decide(
                request: activeTeaCacheRequest,
                gate: firstBlock.teaCacheGateSignal(
                    videoHidden: videoX,
                    videoAdalnParams: videoAdalnParams
                )
            )
        } else {
            .compute
        }

        switch teaCacheDecision {
        case .reuse(let videoResidual, let audioResidual):
            videoX = blockInputVideo + videoResidual
            audioX = blockInputAudio + audioResidual
        case .compute:
            let evalEvery = Int(ProcessInfo.processInfo.environment["LTX2_DIT_EVAL_EVERY"] ?? "8") ?? 8
            for (index, block) in transformerBlocks.enumerated() {
                let skipsVideoSelfAttention = perturbation.skippedVideoSelfAttentionBlocks.contains(index)
                let skipsAudioSelfAttention = perturbation.skippedAudioSelfAttentionBlocks.contains(index)
                let videoTextProjection = activeTextProjectionCache?.video[index]
                let audioTextProjection = activeTextProjectionCache?.audio[index]
                let out = if execution == .compiled, parityIO.outputPrefix == nil {
                    compiledBlockForward(
                        block: block,
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
                        videoTextProjection: videoTextProjection,
                        audioTextProjection: audioTextProjection,
                        videoRope: videoRope,
                        videoSelfAttentionMask: videoSelfAttentionMask,
                        audioRope: audioRope,
                        videoCrossRope: videoCrossRope,
                        audioCrossRope: audioCrossRope,
                        skipsVideoSelfAttention: skipsVideoSelfAttention,
                        skipsAudioSelfAttention: skipsAudioSelfAttention,
                        skipsAudioToVideoCrossAttention: perturbation.skipsAudioToVideoCrossAttention,
                        skipsVideoToAudioCrossAttention: perturbation.skipsVideoToAudioCrossAttention
                    )
                } else {
                    block(
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
                        videoTextProjection: videoTextProjection,
                        audioTextProjection: audioTextProjection,
                        videoRope: videoRope,
                        videoSelfAttentionMask: videoSelfAttentionMask,
                        audioRope: audioRope,
                        videoCrossRope: videoCrossRope,
                        audioCrossRope: audioCrossRope,
                        skipVideoSelfAttention: skipsVideoSelfAttention,
                        skipAudioSelfAttention: skipsAudioSelfAttention,
                        skipAudioToVideoCrossAttention: perturbation.skipsAudioToVideoCrossAttention,
                        skipVideoToAudioCrossAttention: perturbation.skipsVideoToAudioCrossAttention,
                        debugSave: index == 0 ? { array, name in
                            paritySave(array, "block0_\(name)")
                        } : nil
                    )
                }
                videoX = out.video
                audioX = out.audio
                if index == 0 || index == transformerBlocks.count - 1 {
                    paritySave(videoX, "block\(index)_video")
                    paritySave(audioX, "block\(index)_audio")
                }
                if evalEvery > 0 && (index + 1).isMultiple(of: evalEvery) {
                    MLX.eval(videoX, audioX)
                }
            }
            if let activeTeaCacheController, let activeTeaCacheRequest {
                activeTeaCacheController.recordComputedResidual(
                    request: activeTeaCacheRequest,
                    videoResidual: videoX - blockInputVideo,
                    audioResidual: audioX - blockInputAudio
                )
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
        paritySave(videoVelocity, "video_velocity")
        paritySave(audioVelocity, "audio_velocity")
        return (videoVelocity, audioVelocity)
    }

    func forwardVideoOnly(
        videoLatent: MLXArray,
        videoKeyframesMask: MLXArray?,
        videoAttentionMask: MLXArray? = nil,
        timestep: MLXArray,
        videoTimesteps: MLXArray?,
        videoContext: MLXArray,
        videoRope: LTXRope
    ) -> MLXArray {
        let batch = videoLatent.dim(0)
        let tokens = videoLatent.dim(1)
        var video = patchifyProj(videoLatent.asType(.bfloat16))
        if let videoKeyframesMask {
            video = video
                + videoKeyframesMask.asType(video.dtype)
                    * keyframesAbsPosEmbedding.asType(video.dtype)
        }
        let sigma = timestep.asType(video.dtype).reshaped(-1)
        let scaledSigma = sigma * MLXArray(timestepScaleMultiplier).asType(video.dtype)
        let adalnParams: MLXArray
        let embedded: MLXArray
        if let videoTimesteps {
            let scaled = videoTimesteps.asType(video.dtype)
                * MLXArray(timestepScaleMultiplier).asType(video.dtype)
            let values = adalnSingle(timestep: scaled.reshaped(-1), hiddenDType: video.dtype)
            adalnParams = values.0.reshaped(batch, tokens, -1)
            embedded = values.1.reshaped(batch, tokens, -1)
        } else {
            (adalnParams, embedded) = adalnSingle(
                timestep: scaledSigma,
                hiddenDType: video.dtype
            )
        }
        let (promptParams, _) = promptAdalnSingle(
            timestep: scaledSigma,
            hiddenDType: video.dtype
        )
        let videoSelfAttentionMask = prepareLTXSelfAttentionMask(
            videoAttentionMask,
            dtype: video.dtype
        )
        for (index, block) in transformerBlocks.enumerated() {
            video = block.forwardVideoOnly(
                videoHidden: video,
                videoAdalnParams: adalnParams,
                videoPromptAdalnParams: promptParams,
                videoTextEmbeds: videoContext.asType(video.dtype),
                videoRope: videoRope,
                videoSelfAttentionMask: videoSelfAttentionMask
            )
            if (index + 1).isMultiple(of: 8) {
                MLX.eval(video)
            }
        }
        return outputBlock(
            video,
            embeddedTimestep: embedded,
            table: scaleShiftTable,
            norm: normOut,
            projection: projOut,
            dim: videoDim
        )
    }

    private func compiledBlockForward(
        block: LTXUnifiedAVTransformerV2Block,
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
        videoTextProjection: LTXAttentionProjectedContext?,
        audioTextProjection: LTXAttentionProjectedContext?,
        videoRope: LTXRope,
        videoSelfAttentionMask: MLXArray?,
        audioRope: LTXRope,
        videoCrossRope: LTXRope,
        audioCrossRope: LTXRope,
        skipsVideoSelfAttention: Bool,
        skipsAudioSelfAttention: Bool,
        skipsAudioToVideoCrossAttention: Bool,
        skipsVideoToAudioCrossAttention: Bool
    ) -> (video: MLXArray, audio: MLXArray) {
        let variant = LTXV2CompiledBlockVariant(
            hasVideoSelfAttentionMask: videoSelfAttentionMask != nil,
            hasTextProjectionCache: videoTextProjection != nil && audioTextProjection != nil,
            skipsVideoSelfAttention: skipsVideoSelfAttention,
            skipsAudioSelfAttention: skipsAudioSelfAttention,
            skipsAudioToVideoCrossAttention: skipsAudioToVideoCrossAttention,
            skipsVideoToAudioCrossAttention: skipsVideoToAudioCrossAttention
        )
        let runner: LTXUnifiedAVTransformerV2Block
        if let compiledBlockRunner {
            runner = compiledBlockRunner
        } else {
            let created = LTXUnifiedAVTransformerV2Block()
            compiledBlockRunner = created
            runner = created
        }
        runner.update(parameters: block.parameters())

        let forward: LTXV2CompiledBlockForward
        if let compiled = compiledBlockForwards[variant] {
            forward = compiled
        } else {
            let compiled = MLX.compile(inputs: [runner]) { inputs in
                let output = runner(
                    videoHidden: inputs[0],
                    audioHidden: inputs[1],
                    videoAdalnParams: inputs[2],
                    audioAdalnParams: inputs[3],
                    videoPromptAdalnParams: inputs[4],
                    audioPromptAdalnParams: inputs[5],
                    avCaVideoParams: inputs[6],
                    avCaAudioParams: inputs[7],
                    avCaA2VGateParams: inputs[8],
                    avCaV2AGateParams: inputs[9],
                    videoTextEmbeds: inputs[10],
                    audioTextEmbeds: inputs[11],
                    videoTextProjection: variant.hasTextProjectionCache
                        ? LTXAttentionProjectedContext(keys: inputs[20], values: inputs[21])
                        : nil,
                    audioTextProjection: variant.hasTextProjectionCache
                        ? LTXAttentionProjectedContext(keys: inputs[22], values: inputs[23])
                        : nil,
                    videoRope: (cos: inputs[12], sin: inputs[13]),
                    videoSelfAttentionMask: variant.hasVideoSelfAttentionMask
                        ? inputs[variant.hasTextProjectionCache ? 24 : 20]
                        : nil,
                    audioRope: (cos: inputs[14], sin: inputs[15]),
                    videoCrossRope: (cos: inputs[16], sin: inputs[17]),
                    audioCrossRope: (cos: inputs[18], sin: inputs[19]),
                    skipVideoSelfAttention: variant.skipsVideoSelfAttention,
                    skipAudioSelfAttention: variant.skipsAudioSelfAttention,
                    skipAudioToVideoCrossAttention: variant.skipsAudioToVideoCrossAttention,
                    skipVideoToAudioCrossAttention: variant.skipsVideoToAudioCrossAttention
                )
                return [output.video, output.audio]
            }
            compiledBlockForwards[variant] = compiled
            forward = compiled
        }

        var inputs = [
            videoHidden,
            audioHidden,
            videoAdalnParams,
            audioAdalnParams,
            videoPromptAdalnParams,
            audioPromptAdalnParams,
            avCaVideoParams,
            avCaAudioParams,
            avCaA2VGateParams,
            avCaV2AGateParams,
            videoTextEmbeds,
            audioTextEmbeds,
            videoRope.cos,
            videoRope.sin,
            audioRope.cos,
            audioRope.sin,
            videoCrossRope.cos,
            videoCrossRope.sin,
            audioCrossRope.cos,
            audioCrossRope.sin,
        ]
        if let videoTextProjection, let audioTextProjection {
            inputs.append(contentsOf: [
                videoTextProjection.keys,
                videoTextProjection.values,
                audioTextProjection.keys,
                audioTextProjection.values,
            ])
        }
        if let videoSelfAttentionMask {
            inputs.append(videoSelfAttentionMask)
        }
        let outputs = forward(inputs)
        return (outputs[0], outputs[1])
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
        self._ff.wrappedValue = LTXDistilledFeedForward(
            dim: videoDim,
            dimOut: videoDim,
            mult: 4,
            bias: false
        )
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
        videoTextProjection: LTXAttentionProjectedContext? = nil,
        audioTextProjection: LTXAttentionProjectedContext? = nil,
        videoRope: (cos: MLXArray, sin: MLXArray),
        videoSelfAttentionMask: MLXArray?,
        audioRope: (cos: MLXArray, sin: MLXArray),
        videoCrossRope: (cos: MLXArray, sin: MLXArray),
        audioCrossRope: (cos: MLXArray, sin: MLXArray),
        skipVideoSelfAttention: Bool,
        skipAudioSelfAttention: Bool,
        skipAudioToVideoCrossAttention: Bool,
        skipVideoToAudioCrossAttention: Bool,
        debugSave: ((MLXArray, String) -> Void)? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        var video = videoHidden
        var audio = audioHidden

        let vAda = unpackAdaln(videoAdalnParams, table: scaleShiftTable, count: 9, dim: videoDim)
        let aAda = unpackAdaln(audioAdalnParams, table: audioScaleShiftTable, count: 9, dim: audioDim)

        if !skipVideoSelfAttention {
            var videoNorm = rmsNormNoWeight(video)
            videoNorm = videoNorm * (MLXArray(1.0).asType(videoNorm.dtype) + vAda[1]) + vAda[0]
            video = video + attn1(
                videoNorm,
                context: nil,
                mask: videoSelfAttentionMask,
                rope: videoRope
            ) * vAda[2]
        }
        debugSave?(video, "video_after_self_attention")

        if !skipAudioSelfAttention {
            var audioNorm = rmsNormNoWeight(audio)
            audioNorm = audioNorm * (MLXArray(1.0).asType(audioNorm.dtype) + aAda[1]) + aAda[0]
            audio = audio + audioAttn1(audioNorm, context: nil, mask: nil, rope: audioRope) * aAda[2]
        }
        debugSave?(audio, "audio_after_self_attention")

        let vPromptAda = unpackAdaln(videoPromptAdalnParams, table: promptScaleShiftTable, count: 2, dim: videoDim)
        let videoText = videoTextEmbeds * (MLXArray(1.0).asType(videoTextEmbeds.dtype) + vPromptAda[1]) + vPromptAda[0]
        var videoCrossNorm = rmsNormNoWeight(video)
        videoCrossNorm = videoCrossNorm * (MLXArray(1.0).asType(videoCrossNorm.dtype) + vAda[7]) + vAda[6]
        video = video + attn2(
            videoCrossNorm,
            context: videoText,
            mask: nil,
            rope: nil,
            projectedContext: videoTextProjection
        ) * vAda[8]
        debugSave?(video, "video_after_text_attention")

        let aPromptAda = unpackAdaln(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2,
            dim: audioDim
        )
        let audioText = audioTextEmbeds * (MLXArray(1.0).asType(audioTextEmbeds.dtype) + aPromptAda[1]) + aPromptAda[0]
        var audioCrossNorm = rmsNormNoWeight(audio)
        audioCrossNorm = audioCrossNorm * (MLXArray(1.0).asType(audioCrossNorm.dtype) + aAda[7]) + aAda[6]
        audio = audio + audioAttn2(
            audioCrossNorm,
            context: audioText,
            mask: nil,
            rope: nil,
            projectedContext: audioTextProjection
        ) * aAda[8]
        debugSave?(audio, "audio_after_text_attention")

        let videoNorm3 = rmsNormNoWeight(video)
        let audioNorm3 = rmsNormNoWeight(audio)
        let vAV = unpackAdaln(avCaVideoParams, table: scaleShiftTableA2VCAVideo, count: 4, dim: videoDim)
        let aAV = unpackAdaln(avCaAudioParams, table: scaleShiftTableA2VCAAudio, count: 4, dim: audioDim)
        let a2vGate = unpackAVGate(avCaA2VGateParams, table: scaleShiftTableA2VCAVideo, dim: videoDim)
        let v2aGate = unpackAVGate(avCaV2AGateParams, table: scaleShiftTableA2VCAAudio, dim: audioDim)

        if !skipAudioToVideoCrossAttention {
            let videoQA2V = videoNorm3 * (MLXArray(1.0).asType(videoNorm3.dtype) + vAV[0]) + vAV[1]
            let audioKVA2V = audioNorm3 * (MLXArray(1.0).asType(audioNorm3.dtype) + aAV[0]) + aAV[1]
            video = video + audioToVideoAttn(
                videoQA2V,
                context: audioKVA2V,
                mask: nil,
                rope: videoCrossRope,
                keyRope: audioCrossRope
            ) * a2vGate
        }
        debugSave?(video, "video_after_audio_cross_attention")

        if !skipVideoToAudioCrossAttention {
            let audioQV2A = audioNorm3 * (MLXArray(1.0).asType(audioNorm3.dtype) + aAV[2]) + aAV[3]
            let videoKVV2A = videoNorm3 * (MLXArray(1.0).asType(videoNorm3.dtype) + vAV[2]) + vAV[3]
            audio = audio + videoToAudioAttn(
                audioQV2A,
                context: videoKVV2A,
                mask: nil,
                rope: audioCrossRope,
                keyRope: videoCrossRope
            ) * v2aGate
        }
        debugSave?(audio, "audio_after_video_cross_attention")

        var videoFFNorm = rmsNormNoWeight(video)
        videoFFNorm = videoFFNorm * (MLXArray(1.0).asType(videoFFNorm.dtype) + vAda[4]) + vAda[3]
        video = video + ff(videoFFNorm) * vAda[5]
        debugSave?(video, "video_after_feed_forward")

        var audioFFNorm = rmsNormNoWeight(audio)
        audioFFNorm = audioFFNorm * (MLXArray(1.0).asType(audioFFNorm.dtype) + aAda[4]) + aAda[3]
        audio = audio + audioFF(audioFFNorm) * aAda[5]
        debugSave?(audio, "audio_after_feed_forward")

        return (video, audio)
    }

    func projectTextContexts(
        videoTextEmbeds: MLXArray,
        audioTextEmbeds: MLXArray,
        videoPromptAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray
    ) -> (video: LTXAttentionProjectedContext, audio: LTXAttentionProjectedContext) {
        let videoPrompt = unpackAdaln(
            videoPromptAdalnParams,
            table: promptScaleShiftTable,
            count: 2,
            dim: videoDim
        )
        let videoText = videoTextEmbeds
            * (MLXArray(1).asType(videoTextEmbeds.dtype) + videoPrompt[1])
            + videoPrompt[0]
        let audioPrompt = unpackAdaln(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2,
            dim: audioDim
        )
        let audioText = audioTextEmbeds
            * (MLXArray(1).asType(audioTextEmbeds.dtype) + audioPrompt[1])
            + audioPrompt[0]
        return (
            attn2.projectContext(videoText),
            audioAttn2.projectContext(audioText)
        )
    }

    func teaCacheGateSignal(
        videoHidden: MLXArray,
        videoAdalnParams: MLXArray
    ) -> MLXArray {
        let adaln = unpackAdaln(
            videoAdalnParams,
            table: scaleShiftTable,
            count: 9,
            dim: videoDim
        )
        let normalized = rmsNormNoWeight(videoHidden)
        return normalized * (MLXArray(1).asType(normalized.dtype) + adaln[1]) + adaln[0]
    }

    func forwardVideoOnly(
        videoHidden: MLXArray,
        videoAdalnParams: MLXArray,
        videoPromptAdalnParams: MLXArray,
        videoTextEmbeds: MLXArray,
        videoRope: LTXRope,
        videoSelfAttentionMask: MLXArray?
    ) -> MLXArray {
        var video = videoHidden
        let adaln = unpackAdaln(
            videoAdalnParams,
            table: scaleShiftTable,
            count: 9,
            dim: videoDim
        )
        var selfNorm = rmsNormNoWeight(video)
        selfNorm = selfNorm * (MLXArray(1).asType(selfNorm.dtype) + adaln[1]) + adaln[0]
        video = video + attn1(
            selfNorm,
            context: nil,
            mask: videoSelfAttentionMask,
            rope: videoRope
        ) * adaln[2]

        let promptAdaln = unpackAdaln(
            videoPromptAdalnParams,
            table: promptScaleShiftTable,
            count: 2,
            dim: videoDim
        )
        let text = videoTextEmbeds
            * (MLXArray(1).asType(videoTextEmbeds.dtype) + promptAdaln[1])
            + promptAdaln[0]
        var textNorm = rmsNormNoWeight(video)
        textNorm = textNorm * (MLXArray(1).asType(textNorm.dtype) + adaln[7]) + adaln[6]
        video = video + attn2(textNorm, context: text, mask: nil, rope: nil) * adaln[8]

        var feedForwardNorm = rmsNormNoWeight(video)
        feedForwardNorm = feedForwardNorm
            * (MLXArray(1).asType(feedForwardNorm.dtype) + adaln[4])
            + adaln[3]
        return video + ff(feedForwardNorm) * adaln[5]
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

private final class LTXAudioOnlyTransformerV2: Module {
    let audioDim = 2_048
    let timestepScaleMultiplier: Float = 1_000

    @ModuleInfo(key: "audio_patchify_proj") var audioPatchifyProj: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_adaln_single") var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") var audioPromptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTXAudioOnlyTransformerV2Block]
    @ModuleInfo(key: "audio_norm_out") var audioNormOut: LayerNorm

    override init() {
        self._audioPatchifyProj.wrappedValue = Linear(128, audioDim, bias: true)
        self._audioProjOut.wrappedValue = Linear(audioDim, 128, bias: true)
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        self._audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 9
        )
        self._audioPromptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(
            embeddingDim: audioDim,
            embeddingCoefficient: 2
        )
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXAudioOnlyTransformerV2Block()
        }
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: audioDim, eps: 1e-6, affine: false)
        super.init()
    }

    func forward(
        audioLatent: MLXArray,
        timestep: MLXArray,
        audioTimesteps: MLXArray?,
        audioContext: MLXArray,
        audioRope: LTXRope,
        skippedSelfAttentionBlocks: Set<Int> = []
    ) -> MLXArray {
        let batch = audioLatent.dim(0)
        let tokens = audioLatent.dim(1)
        var audio = audioPatchifyProj(audioLatent.asType(.bfloat16))
        let sigma = timestep.asType(audio.dtype).reshaped(-1)
        let scaledSigma = sigma * MLXArray(timestepScaleMultiplier).asType(audio.dtype)
        let adalnParams: MLXArray
        let embedded: MLXArray
        if let audioTimesteps {
            let scaled = audioTimesteps.asType(audio.dtype)
                * MLXArray(timestepScaleMultiplier).asType(audio.dtype)
            let values = audioAdalnSingle(
                timestep: scaled.reshaped(-1),
                hiddenDType: audio.dtype
            )
            adalnParams = values.0.reshaped(batch, tokens, -1)
            embedded = values.1.reshaped(batch, tokens, -1)
        } else {
            (adalnParams, embedded) = audioAdalnSingle(
                timestep: scaledSigma,
                hiddenDType: audio.dtype
            )
        }
        let (promptParams, _) = audioPromptAdalnSingle(
            timestep: scaledSigma,
            hiddenDType: audio.dtype
        )
        for (index, block) in transformerBlocks.enumerated() {
            audio = block(
                audioHidden: audio,
                audioAdalnParams: adalnParams,
                audioPromptAdalnParams: promptParams,
                audioTextEmbeds: audioContext.asType(audio.dtype),
                audioRope: audioRope,
                skipSelfAttention: skippedSelfAttentionBlocks.contains(index)
            )
            if (index + 1).isMultiple(of: 8) {
                MLX.eval(audio)
            }
        }
        let normalizedEmbedding = embedded.ndim == 2
            ? embedded.expandedDimensions(axis: 1)
            : embedded
        let scaleShift = audioScaleShiftTable.reshaped(1, 1, 2, audioDim)
            + normalizedEmbedding.expandedDimensions(axis: 2)
        let shift = scaleShift[0..., 0..., 0, 0...]
        let scale = scaleShift[0..., 0..., 1, 0...]
        var output = audioNormOut(audio)
        output = output * (MLXArray(1).asType(output.dtype) + scale) + shift
        return audioProjOut(output)
    }
}

private final class LTXAudioOnlyTransformerV2Block: Module {
    let audioDim = 2_048

    @ModuleInfo(key: "audio_attn1") var audioAttn1: LTXDistilledAttention
    @ModuleInfo(key: "audio_attn2") var audioAttn2: LTXDistilledAttention
    @ModuleInfo(key: "audio_ff") var audioFF: LTXDistilledFeedForward
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_prompt_scale_shift_table") var audioPromptScaleShiftTable: MLXArray

    override init() {
        self._audioAttn1.wrappedValue = LTXDistilledAttention(
            queryDim: audioDim,
            contextDim: nil,
            heads: 32,
            headDim: 64,
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
        self._audioFF.wrappedValue = LTXDistilledFeedForward(
            dim: audioDim,
            dimOut: audioDim,
            mult: 4
        )
        self._audioScaleShiftTable.wrappedValue = MLX.zeros([9, audioDim], dtype: .float32)
        self._audioPromptScaleShiftTable.wrappedValue = MLX.zeros([2, audioDim], dtype: .float32)
        super.init()
    }

    func callAsFunction(
        audioHidden: MLXArray,
        audioAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray,
        audioTextEmbeds: MLXArray,
        audioRope: LTXRope,
        skipSelfAttention: Bool
    ) -> MLXArray {
        var audio = audioHidden
        let adaln = unpack(
            audioAdalnParams,
            table: audioScaleShiftTable,
            count: 9
        )
        if !skipSelfAttention {
            var selfNorm = rmsNormNoWeight(audio)
            selfNorm = selfNorm * (MLXArray(1).asType(selfNorm.dtype) + adaln[1]) + adaln[0]
            audio = audio
                + audioAttn1(selfNorm, context: nil, mask: nil, rope: audioRope) * adaln[2]
        }

        let promptAdaln = unpack(
            audioPromptAdalnParams,
            table: audioPromptScaleShiftTable,
            count: 2
        )
        let text = audioTextEmbeds
            * (MLXArray(1).asType(audioTextEmbeds.dtype) + promptAdaln[1])
            + promptAdaln[0]
        var textNorm = rmsNormNoWeight(audio)
        textNorm = textNorm * (MLXArray(1).asType(textNorm.dtype) + adaln[7]) + adaln[6]
        audio = audio + audioAttn2(textNorm, context: text, mask: nil, rope: nil) * adaln[8]

        var feedForwardNorm = rmsNormNoWeight(audio)
        feedForwardNorm = feedForwardNorm
            * (MLXArray(1).asType(feedForwardNorm.dtype) + adaln[4])
            + adaln[3]
        return audio + audioFF(feedForwardNorm) * adaln[5]
    }

    private func unpack(
        _ params: MLXArray,
        table: MLXArray,
        count: Int
    ) -> [MLXArray] {
        if params.ndim == 2 {
            let values = params.reshaped(-1, count, audioDim)
                + table[0..<count, 0...].reshaped(1, count, audioDim)
            return (0..<count).map { values[0..., $0, 0...].expandedDimensions(axis: 1) }
        }
        let values = params.reshaped(params.dim(0), params.dim(1), count, audioDim)
            + table[0..<count, 0...].reshaped(1, 1, count, audioDim)
        return (0..<count).map { values[0..., 0..., $0, 0...] }
    }
}

private func denoiseLTX25AudioOnlyLoop(
    audioLatents: MLXArray,
    audioRope: LTXRope,
    positiveContext: MLXArray,
    negativeContext: MLXArray,
    transformer: LTXAudioOnlyTransformerV2,
    sigmas: [Float],
    guidance: LTXTextToAudioGuidance
) -> MLXArray {
    var current = audioLatents
    let batch = current.dim(0)
    let channels = current.dim(1)
    let frames = current.dim(2)
    let melBins = current.dim(3)
    let dtype = current.dtype
    var lastDenoised: MLXArray?

    func denoised(
        context: MLXArray,
        sigma: Float,
        skippedBlocks: Set<Int>
    ) -> MLXArray {
        let flat = current
            .transposed(0, 2, 1, 3)
            .reshaped(batch, frames, channels * melBins)
        let timesteps = MLX.full(
            [batch, frames],
            values: MLXArray(sigma).asType(dtype)
        )
        let global = MLX.full([batch], values: MLXArray(sigma).asType(dtype))
        let velocity = transformer.forward(
            audioLatent: flat,
            timestep: global,
            audioTimesteps: timesteps,
            audioContext: context,
            audioRope: audioRope,
            skippedSelfAttentionBlocks: skippedBlocks
        )
        return toDenoised(
            noisy: current,
            velocity: velocity
                .reshaped(batch, frames, channels, melBins)
                .transposed(0, 2, 1, 3),
            sigma: sigma
        )
    }

    for index in 0..<(sigmas.count - 1) {
        let sigma = sigmas[index]
        let nextSigma = sigmas[index + 1]
        let guided: MLXArray
        if guidance.shouldSkip(step: index), let lastDenoised {
            guided = lastDenoised
        } else {
            let conditioned = denoised(context: positiveContext, sigma: sigma, skippedBlocks: [])
            let negative = guidance.classifierFreeScale == 1
                ? conditioned
                : denoised(context: negativeContext, sigma: sigma, skippedBlocks: [])
            let perturbed = guidance.spatioTemporalScale == 0
                ? conditioned
                : denoised(
                    context: positiveContext,
                    sigma: sigma,
                    skippedBlocks: guidance.spatioTemporalBlocks
                )
            guided = guidance.combine(
                conditioned: conditioned,
                negativeText: negative,
                perturbed: perturbed
            )
            lastDenoised = guided
        }
        if nextSigma == 0 {
            current = guided
        } else {
            let sigmaArray = MLXArray(sigma).asType(dtype)
            let nextArray = MLXArray(nextSigma).asType(dtype)
            current = guided + nextArray * (current - guided) / sigmaArray
        }
        MLX.eval(current)
    }
    return current
}

func mapUnifiedTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard let mapped = mapUnifiedTransformerKey(key) else { return [] }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

func mapUnifiedTransformerKey(_ key: String) -> String? {
    guard key.hasPrefix("model.diffusion_model.") else { return nil }
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
        return nil
    }
    return mapped
}

private func resolvedLTX25TransformerURL(
    resources: LTX25Resources,
    kind: LTX25NativeModelPackKind
) -> URL {
    LTX25NativeModelPack.optimizedURLIfValid(resources: resources, kind: kind)
        ?? (kind == .dev ? resources.devTransformerURL : resources.distilledTransformerURL)
}

func loadLTX25TransformerWeights(
    url: URL,
    model: Module,
    dtype: DType,
    sourceInclude: (String) -> Bool = { $0.hasPrefix("model.diffusion_model.") },
    nativeInclude: (String) -> Bool = { !isLTX25ConnectorTensorKey($0) }
) throws {
    let isNative = LTX25NativeModelPack.isNativePack(url)
    if isNative {
        try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
            url: url,
            to: model,
            verify: .none,
            include: nativeInclude,
            mapper: { key, value in
                let casted = value.dtype.isFloatingPoint && value.dtype != dtype
                    ? value.asType(dtype)
                    : value
                return [(key, casted)]
            },
            batchSize: 24
        )
        return
    }
    try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
        url: url,
        to: model,
        verify: .none,
        include: sourceInclude,
        mapper: { key, value in
            mapUnifiedTransformerWeight(key: key, value: value, dtype: dtype)
        },
        batchSize: 24
    )
}

func ltx25UnifiedTransformerParameterShapes() -> [String: [Int]] {
    Dictionary(
        uniqueKeysWithValues: LTXUnifiedAVTransformerV2().parameters().flattened().map {
            ($0.0, $0.1.shape)
        }
    )
}

func loadLTX25UnifiedTransformerParametersForValidation(
    url: URL,
    dtype: DType
) throws -> [(String, MLXArray)] {
    let model = LTXUnifiedAVTransformerV2()
    try loadLTX25TransformerWeights(url: url, model: model, dtype: dtype)
    return model.parameters().flattened()
}

func mapLTX23UnifiedTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("transformer."),
          let mapped = mapLTX23UnifiedTransformerKey(key) else {
        return []
    }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

func mapLTX23UnifiedTransformerKey(_ key: String) -> String? {
    var mapped: String
    if key.hasPrefix("transformer.") {
        mapped = String(key.dropFirst("transformer.".count))
    } else if key.hasPrefix("diffusion_model.") {
        mapped = String(key.dropFirst("diffusion_model.".count))
    } else {
        return nil
    }
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
        return nil
    }
    return mapped
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

private final class LTXAudioDownsample2d: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init([3, 3]),
            stride: .init([2, 2]),
            padding: .init(0),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Upstream HEIGHT causality pads (left: 0, right: 1, top: 2, bottom: 0)
        // before its stride-2 convolution.
        conv(padded(x, widths: [[0, 0], [2, 0], [0, 1], [0, 0]]))
    }
}

private final class LTXAudioEncoderStage: Module {
    @ModuleInfo(key: "block") var blocks: [LTXAudioResnetBlock2D]
    @ModuleInfo(key: "downsample") var downsample: LTXAudioDownsample2d?

    init(blocks: [LTXAudioResnetBlock2D], downsampleChannels: Int?) {
        self._blocks.wrappedValue = blocks
        if let downsampleChannels {
            self._downsample.wrappedValue = LTXAudioDownsample2d(channels: downsampleChannels)
        } else {
            self._downsample.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        return downsample?(h) ?? h
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

    func normalize(_ x: MLXArray) -> MLXArray {
        let std = stdOfMeans.asType(x.dtype)
        let mean = meanOfMeans.asType(x.dtype)
        return (x - mean) / std
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

private final class LTXAudioEncoder: Module {
    @ModuleInfo(key: "per_channel_statistics") var perChannelStatistics: LTXAudioPerChannelStatistics
    @ModuleInfo(key: "conv_in") var convIn: LTXAudioCausalConv2d
    @ModuleInfo(key: "down") var down: [LTXAudioEncoderStage]
    @ModuleInfo(key: "mid") var mid: LTXAudioMidBlock
    @ModuleInfo(key: "norm_out") var normOut: LTXAudioPixelNorm
    @ModuleInfo(key: "conv_out") var convOut: LTXAudioCausalConv2d

    let patchifier = LTXAudioPatchifier()

    override init() {
        self._perChannelStatistics.wrappedValue = LTXAudioPerChannelStatistics(latentChannels: 128)
        self._convIn.wrappedValue = LTXAudioCausalConv2d(inChannels: 2, outChannels: 128)
        self._down.wrappedValue = [
            LTXAudioEncoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                ],
                downsampleChannels: 128
            ),
            LTXAudioEncoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 256),
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 256),
                ],
                downsampleChannels: 256
            ),
            LTXAudioEncoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 512),
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                ],
                downsampleChannels: nil
            ),
        ]
        self._mid.wrappedValue = LTXAudioMidBlock(channels: 512)
        self._normOut.wrappedValue = LTXAudioPixelNorm()
        self._convOut.wrappedValue = LTXAudioCausalConv2d(inChannels: 512, outChannels: 16)
        super.init()
    }

    /// Encodes `[batch, channels, time, mel]` and returns transformer latents
    /// in `[batch, 8, latent time, 16]` layout.
    func encode(spectrogram: MLXArray) -> MLXArray {
        precondition(
            spectrogram.ndim == 4 && spectrogram.dim(1) == 2 && spectrogram.dim(3) == 64,
            "LTX audio VAE expects [batch, 2, time, 64] log-mels."
        )
        var h = spectrogram.transposed(0, 2, 3, 1)
        h = convIn(h)
        for stage in down {
            h = stage(h)
        }
        h = mid(h)
        h = normOut(h)
        h = silu(h)
        h = convOut(h)

        let means = h[0..., 0..., 0..., 0..<LTXAudioLatentChannels]
        var patched = patchifier.patchify(means)
        patched = perChannelStatistics.normalize(patched)
        let normalized = patchifier.unpatchify(
            patched,
            channels: LTXAudioLatentChannels,
            melBins: means.dim(2)
        )
        return normalized.transposed(0, 3, 1, 2)
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

func mapAudioVaeEncoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("audio_vae.encoder.") {
        mapped = String(key.dropFirst("audio_vae.encoder.".count))
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

func loadLTXPackedBWEVocoderConfig(weightsURL: URL) throws -> LTXBWEVocoderRuntimeConfig? {
    let metadata = try SafetensorsStreamingLoader.fileMetadata(url: weightsURL)
    guard let rawConfig = metadata["config"],
          let data = rawConfig.data(using: .utf8),
          let envelope = try? JSONDecoder().decode(LTXVocoderConfigEnvelope.self, from: data),
          let bwe = envelope.bwe else {
        return nil
    }
    let base = envelope.baseVocoder?.runtimeArchitecture(defaultArchitecture: .defaultBWEBase)
    return bwe.runtimeConfig(baseVocoder: base)
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
