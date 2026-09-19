import Foundation
import MLX

public struct AuKDiTConfiguration: Sendable {
    public var dimension = 1536
    public var heads = 24
    public var doubleLayers = 10
    public var singleLayers = 20
    public var positionGroups = 16
    public init() {}
}

public struct AuKDiT {
    let weights: AuKTensorStore
    let config: AuKDiTConfiguration
    let rotary: AuKRotary
    public init(weights: AuKTensorStore, frequencies: [Float], configuration: AuKDiTConfiguration = .init()) throws {
        guard configuration.dimension > 0, configuration.heads > 0,
              configuration.dimension % configuration.heads == 0,
              frequencies.count * 2 == configuration.dimension / configuration.heads else {
            throw AuKError.invalid("Invalid AuK DiT rotary/head dimensions")
        }
        self.weights = weights
        self.config = configuration
        self.rotary = AuKRotary(frequencies: frequencies)
    }

    func timestep(_ t: MLXArray) throws -> MLXArray {
        let frequencies = exp(MLXArray(0..<128).asType(.float32) * Float(-log(10000.0) / 127))
        let angle = t.reshaped(-1, 1) * 1000 * frequencies
        return try weights.linear(aukSiLU(weights.linear(concatenated([sin(angle), cos(angle)], axis: -1),
            "time_embed.time_mlp.0")), "time_embed.time_mlp.1")
    }

    func embed(_ x: MLXArray) throws -> MLXArray {
        let h = try weights.linear(x, "audio_embed.linear")
        var p = h
        for index in 0..<2 {
            p = try weights.conv(p, "audio_embed.conv_pos_embed.conv1d.\(index)", groups: config.positionGroups)
            p = p * tanh(logAddExp(p, MLXArray(Float(0))))
        }
        return h + p
    }

    func modulation(_ x: MLXArray, _ t: MLXArray, _ key: String) throws -> [MLXArray] {
        let parts = split(try weights.linear(aukSiLU(t), key + ".linear"), parts: 6, axis: -1)
            .map { $0.expandedDimensions(axis: 1) }
        return [aukNorm(x) * (1 + parts[1]) + parts[0], parts[2], parts[3], parts[4], parts[5]]
    }

    func feedForward(_ x: MLXArray, _ key: String) throws -> MLXArray {
        let parts = split(try weights.linear(x, key + ".linear_in"), parts: 2, axis: -1)
        return try weights.linear(aukSiLU(parts[0]) * parts[1], key + ".linear_out")
    }

    func qkv(_ x: MLXArray, _ key: String, context: Bool = false) throws -> [MLXArray] {
        let suffix = context ? "_c" : ""
        let parts = split(try weights.linear(x, key + ".to_qkv" + suffix), parts: 3, axis: -1)
            .map { aukHeads($0, config.heads) }
        let prefix = context ? ".c_" : "."
        return [rotary.apply(try weights.rms(parts[0], key + prefix + "q_norm")),
                rotary.apply(try weights.rms(parts[1], key + prefix + "k_norm")), parts[2]]
    }

    public func velocity(latent: MLXArray, text: MLXArray, time: Float,
                         reference: MLXArray?, guidance: Float) throws -> MLXArray {
        var t = try timestep(MLXArray([time]))
        var c = try weights.rms(weights.linear(text, "txt_proj"), "txt_norm")
        var h = try embed(latent)
        let promptLength = reference?.dim(1) ?? 0
        if let reference, promptLength > 0 { h = concatenated([try embed(reference), h], axis: 1) }
        if guidance > 0 {
            var unconditioned = try embed(latent)
            if let reference, promptLength > 0 {
                unconditioned = concatenated([try embed(zeros(like: reference)), unconditioned], axis: 1)
            }
            h = concatenated([h, unconditioned], axis: 0)
            c = concatenated([c, zeros(like: c)], axis: 0)
            t = concatenated([t, t], axis: 0)
        }
        for index in 0..<config.doubleLayers {
            let key = "transformer_blocks.\(index)"
            let xm = try modulation(h, t, key + ".attn_norm_x")
            let cm = try modulation(c, t, key + ".attn_norm_c")
            let xq = try qkv(xm[0], key + ".attn")
            let cq = try qkv(cm[0], key + ".attn", context: true)
            let q = concatenated([xq[0], cq[0]], axis: 2)
            let k = concatenated([xq[1], cq[1]], axis: 2)
            let v = concatenated([xq[2], cq[2]], axis: 2)
            let attended = aukUnheads(aukAttention(q, k, v))
            let length = h.dim(1)
            h = h + xm[1] * (try weights.linear(attended[0..., ..<length], key + ".attn.to_out.0"))
            c = c + cm[1] * (try weights.linear(attended[0..., length...], key + ".attn.to_out_c"))
            h = h + xm[4] * (try feedForward(aukNorm(h) * (1 + xm[3]) + xm[2], key + ".ff_x"))
            c = c + cm[4] * (try feedForward(aukNorm(c) * (1 + cm[3]) + cm[2], key + ".ff_c"))
        }
        let textLength = c.dim(1)
        h = concatenated([c, h], axis: 1)
        for index in 0..<config.singleLayers {
            let key = "single_transformer_blocks.\(index)"
            let m = try modulation(h, t, key + ".attn_norm")
            let q = try qkv(m[0], key + ".attn")
            h = h + m[1] * (try weights.linear(aukUnheads(aukAttention(q[0], q[1], q[2])), key + ".attn.to_out.0"))
            h = h + m[4] * (try feedForward(aukNorm(h) * (1 + m[3]) + m[2], key + ".ff"))
        }
        h = h[0..., (textLength + promptLength)...]
        let m = split(try weights.linear(aukSiLU(t), "norm_out.linear"), parts: 2, axis: -1)
        h = aukNorm(h) * (1 + m[0].expandedDimensions(axis: 1)) + m[1].expandedDimensions(axis: 1)
        let result = try weights.linear(h, "proj_out")
        return guidance > 0 ? result[0..<1] + guidance * (result[0..<1] - result[1..<2]) : result
    }
}
