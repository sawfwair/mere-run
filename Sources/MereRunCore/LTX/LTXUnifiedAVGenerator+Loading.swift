import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
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

    func loadStandalone(
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
            let splitModel = LTXUnifiedAVTransformerV2(checkpointKind: .ltx23)
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
}
