// Adapted from speridlabs/idol-sdk (Apache-2.0), including the upstream notice:
// Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
// See Resources/ModelLicenses/Idol-Code-LICENSE and THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

public struct IdolFlowState {
    public let latent: MLXArray
    public let camera: MLXArray
    public init(latent: MLXArray, camera: MLXArray) { self.latent = latent; self.camera = camera }
}

public struct IdolConditioning {
    public let dino: MLXArray
    public let vae: MLXArray
    public init(dino: MLXArray, vae: MLXArray) { self.dino = dino; self.vae = vae }
    public var zeros: Self { Self(dino: MLX.zeros(like: dino), vae: MLX.zeros(like: vae)) }
}

/// Native joint [latent, image, camera] transformer. Reference: idol-sdk 4702a7f.
public final class IdolDenoiser {
    public let configuration: IdolConfiguration.Denoiser
    private let layers: IdolLayers

    public init(configuration: IdolConfiguration, weights: [String: MLXArray]) throws {
        try configuration.validate()
        self.configuration = configuration.denoiser.args
        layers = IdolLayers(weights: try IdolWeights(weights, shapes: IdolWeightSchema.denoiser(self.configuration)),
                            heads: self.configuration.num_heads)
    }

    public func velocity(_ state: IdolFlowState, time: Float, conditioning: IdolConditioning) throws -> IdolFlowState {
        let config = configuration, weights = layers.weights
        guard state.latent.ndim == 3, state.camera.ndim == 3 else {
            throw IdolError.invalidInput("Denoiser inputs must be rank-three token streams.")
        }
        let batch = state.latent.dim(0)
        guard batch > 0, state.latent.shape == [batch, config.q_token_length, config.in_channels],
              state.camera.shape == [batch, 1, config.cam_channels], conditioning.dino.ndim == 3,
              conditioning.dino.dim(0) == batch, conditioning.dino.dim(2) == config.cond_channels,
              conditioning.vae.shape == [batch, conditioning.dino.dim(1), config.cond2_channels],
              time.isFinite, (0...1000).contains(time) else {
            throw IdolError.invalidInput("Denoiser stream shapes or time do not match the checkpoint.")
        }
        var noise = weights.linear(state.latent, "input_layer").asType(.float16)
        var context = (weights.linear(conditioning.dino, "cond_embedder")
            + weights.linear(conditioning.vae, "cond_embedder2")).asType(.float16)
        let timeEmbedding = layers.scalarEmbedding(ones([batch]) * time, "t_embedder")
        let modulation = weights.linear(silu(timeEmbedding), "adaLN_modulation.1").asType(.float16)
        noise = noise + IdolLayers.positionEmbedding(weights["pos_pe"], width: config.model_channels, decoder: false).asType(.float16)
        for index in 0..<config.num_refiner_blocks {
            let prefix = "noise_refiner.\(index)"
            noise = layers.modulatedBlock(noise, prefix,
                modulation: modulation + weights[prefix + ".shift_table"].asType(.float16),
                angles: layers.rotaryAngles(noise, "noise_repo_layers.\(index)"))
            let contextPrefix = "context_refiner.\(index)"
            let angles = layers.rotaryAngles(context, "context_repo_layers.\(index)")
            context = context + layers.attention(layers.norm(context, contextPrefix + ".norm1"),
                                                contextPrefix + ".attn", angles: angles)
            context = context + layers.ffn(layers.norm(context, contextPrefix + ".norm2"), contextPrefix + ".mlp")
            eval(noise, context)
        }
        var camera = state.camera
        for index in 0..<config.num_refiner_blocks {
            camera = weights.linear(camera, "cam_refiner.mlp.\(2 * index)")
            if index < config.num_refiner_blocks - 1 { camera = geluApproximate(camera) }
        }
        var joint = concatenated([noise, context, camera.asType(.float16)], axis: 1)
        for index in 0..<config.num_blocks {
            let prefix = "blocks.\(index)"
            joint = layers.modulatedBlock(joint, prefix,
                modulation: modulation + weights[prefix + ".shift_table"].asType(.float16),
                angles: layers.rotaryAngles(joint, "repo_layers.\(index)"))
            eval(joint)
        }
        let shifts = split(weights["shift_table"] + timeEmbedding.expandedDimensions(axis: 1), parts: 2, axis: 1)
        func output(_ hidden: MLXArray, _ name: String) -> MLXArray {
            weights.linear(layers.norm(hidden.asType(.float32), eps: 1e-5) * (1 + shifts[1]) + shifts[0], name)
        }
        return IdolFlowState(latent: output(joint[0..., 0..<config.q_token_length, 0...], "out_layer"),
            camera: output(joint[0..., (joint.dim(1) - 1)..., 0...], "cam_out_layer"))
    }

    public func sample(noise: IdolFlowState, conditioning: IdolConditioning, settings: IdolSamplingSettings,
                       progress: ((Int) -> Void)? = nil) throws -> IdolFlowState {
        var state = noise
        let schedule = settings.schedule, negative = conditioning.zeros
        for step in 0..<settings.steps {
            try Task.checkCancellation()
            let positive = try velocity(state, time: schedule[step] * 1000, conditioning: conditioning)
            let velocity: IdolFlowState
            if settings.guidance > 0 {
                let unconditional = try self.velocity(state, time: schedule[step] * 1000, conditioning: negative)
                velocity = IdolFlowState(latent: (1 + settings.guidance) * positive.latent - settings.guidance * unconditional.latent,
                    camera: (1 + settings.guidance) * positive.camera - settings.guidance * unconditional.camera)
            } else { velocity = positive }
            let delta = schedule[step] - schedule[step + 1]
            state = IdolFlowState(latent: state.latent - delta * velocity.latent, camera: state.camera - delta * velocity.camera)
            eval(state.latent, state.camera)
            progress?(step + 1)
        }
        return state
    }
}
