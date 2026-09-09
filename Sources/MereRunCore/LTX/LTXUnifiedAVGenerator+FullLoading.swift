import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
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
        let model = LTXUnifiedAVTransformerV2(
            checkpointKind: isLTX25 ? .ltx25 : .ltx23
        )
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
}
