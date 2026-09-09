import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXDistilledLatentGenerator {
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

    func loadEncoderIfNeeded() throws {
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
}
