import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom

/// Loading stays in Core; shared DINOv3 and VAE computation lives in ImageModels.
struct IdolConditioner {
    private struct DinoConfig: Decodable {
        let hidden_size: Int
        let intermediate_size: Int
        let num_hidden_layers: Int
        let num_attention_heads: Int
        let patch_size: Int
        let num_register_tokens: Int
        let layer_norm_eps: Float
        let rope_theta: Float
        let use_gated_mlp: Bool
        let hidden_act: String
        let query_bias: Bool
        let key_bias: Bool
        let value_bias: Bool
        let proj_bias: Bool
        let mlp_bias: Bool
    }
    private struct VAEConfiguration: Decodable {
        let in_channels: Int
        let out_channels: Int
        let latent_channels: Int
        let block_out_channels: [Int]
        let layers_per_block: Int
        let norm_num_groups: Int
        let mid_block_add_attention: Bool
        let use_quant_conv: Bool
        let use_post_quant_conv: Bool
        let batch_norm_eps: Float
    }

    static func dino(_ image: MLXArray, root: URL) throws -> MLXArray {
        let raw = try JSONDecoder().decode(DinoConfig.self, from: Data(contentsOf: root.appending(path: "config.json")))
        guard raw.hidden_size == 1280, raw.num_hidden_layers == 32, raw.num_attention_heads == 20,
              raw.patch_size == 16, raw.num_register_tokens == 4, raw.intermediate_size > 0,
              raw.intermediate_size <= 8192, raw.layer_norm_eps.isFinite, raw.layer_norm_eps > 0,
              raw.rope_theta.isFinite, raw.rope_theta > 0,
              raw.hidden_act == (raw.use_gated_mlp ? "silu" : "gelu") else {
            throw IdolError.invalidConfiguration("Expected DINOv3 ViT-H+/16 configuration.")
        }
        var config = DINOv3Config()
        config.hiddenSize = raw.hidden_size
        config.intermediateSize = raw.intermediate_size
        config.numHiddenLayers = raw.num_hidden_layers
        config.numAttentionHeads = raw.num_attention_heads
        config.imageSize = 1024
        config.layerNormEps = raw.layer_norm_eps
        config.ropeTheta = raw.rope_theta
        config.queryBias = raw.query_bias; config.keyBias = raw.key_bias; config.valueBias = raw.value_bias
        config.projectionBias = raw.proj_bias; config.mlpBias = raw.mlp_bias; config.useGatedMLP = raw.use_gated_mlp
        let model = DINOv3VisionModel(config: config)
        let arrays = try MLX.loadArrays(url: root.appending(path: "model.safetensors"))
        let converted = arrays.mapValues { array in
            (array.ndim == 4 ? array.transposed(0, 2, 3, 1) : array).asType(.bfloat16)
        }
        try model.update(parameters: ModuleParameters.unflattened(converted), verify: .all)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 3, 1, 1)
        let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 3, 1, 1)
        // IDOL applies an additional affine-free norm to HF's learned final norm.
        let result = MLXFast.layerNorm(model.normalizedTokens(((image - mean) / std).asType(.bfloat16)).asType(.float32), eps: 1e-5)
        eval(result)
        return result
    }

    static func vae(_ image: MLXArray, root: URL, posteriorNoise: MLXArray? = nil) throws -> MLXArray {
        let raw = try JSONDecoder().decode(VAEConfiguration.self, from: Data(contentsOf: root.appending(path: "config.json")))
        guard raw.in_channels == 3, raw.out_channels == 3, raw.latent_channels == 32,
              raw.block_out_channels == [128, 256, 512, 512], raw.layers_per_block == 2,
              raw.norm_num_groups == 32, raw.batch_norm_eps.isFinite, raw.batch_norm_eps > 0 else {
            throw IdolError.invalidConfiguration("Expected the original FLUX.2 VAE configuration.")
        }
        let model = AutoencoderKL(configuration: VAEConfig(latentChannels: 32, scalingFactor: 1, shiftFactor: 0,
            blockOutChannels: raw.block_out_channels, midBlockAddAttention: raw.mid_block_add_attention,
            useQuantConv: raw.use_quant_conv, usePostQuantConv: raw.use_post_quant_conv, attentionNormEps: 1e-6))
        var arrays = try MLX.loadArrays(url: root.appending(path: "diffusion_pytorch_model.safetensors"))
        guard let mean = arrays.removeValue(forKey: "bn.running_mean"), let variance = arrays.removeValue(forKey: "bn.running_var"),
              mean.shape == [128], variance.shape == [128] else {
            throw IdolError.invalidWeights("Missing FLUX.2 VAE running statistics.")
        }
        arrays.removeValue(forKey: "bn.num_batches_tracked")
        arrays = arrays.mapValues { array in
            (array.ndim == 4 ? array.transposed(0, 2, 3, 1) : array).asType(.bfloat16)
        }
        try model.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        let moments = model.encode(image.asType(.bfloat16) * 2 - 1)
        let location = moments[0..., 0..<32, 0..., 0...]
        let logVariance = clip(moments[0..., 32..<64, 0..., 0...], min: -30, max: 20)
        let noise = posteriorNoise?.asType(location.dtype) ?? MLXRandom.normal(location.shape, dtype: location.dtype)
        guard noise.shape == location.shape else { throw IdolError.invalidInput("VAE posterior noise has the wrong shape.") }
        let sample = location + exp(logVariance * 0.5) * noise
        let result = try IdolConditioning.packVAE(sample, mean: mean, variance: variance, epsilon: raw.batch_norm_eps)
        eval(result)
        return result
    }
}
