import Foundation
import MLX
import MLXFast
import MLXNN

// Swift/MLX adaptation of BFL's Apache-2.0 reference DiT at
// 45e57c9fb06f9751cdced29a65c5d2171003be05; see THIRD_PARTY_NOTICES.md.

/// One sample per call. Reference tokens share a stream; text follows both image streams.
/// Uses the publisher's tensor names directly, without constructing random full-size modules.
public final class Flux3Transformer {
    public let configuration: Flux3Configuration
    private let weights: [String: MLXArray]
    private let rope: Flux2PosEmbed

    public init(configuration: Flux3Configuration = .init(), weights: [String: MLXArray]) throws {
        let shapes = Self.weightShapes(configuration)
        guard Set(shapes.keys) == Set(weights.keys),
              shapes.allSatisfy({ weights[$0.key]?.shape == $0.value }),
              weights.values.allSatisfy({ $0.dtype == .bfloat16 || $0.dtype == .float32 }) else {
            throw Flux3Error.invalidWeights("Expected the original distilled BF16 tensor names and shapes; FP8 and NVFP4 packages are unsupported.")
        }
        self.configuration = configuration
        self.weights = weights
        rope = Flux2PosEmbed(theta: 10000, axesDim: configuration.axesDimensions)
    }

    public static func weightShapes(_ config: Flux3Configuration = .init()) -> [String: [Int]] {
        let hidden = config.hiddenSize, mlp = hidden * config.mlpRatio
        var shapes: [String: [Int]] = [
            "emb_in.image.weight": [hidden, config.imageChannels],
            "emb_in.image_cond.weight": [hidden, config.imageChannels],
            "txt_in.weight": [hidden, config.textChannels],
            "final_layer.image.linear.weight": [config.imageChannels, hidden],
            "final_layer.image.adaLN_modulation.1.weight": [2 * hidden, hidden]
        ]
        for prefix in ["time_in", "ref_time_in"] {
            shapes[prefix + ".in_layer.weight"] = [hidden, 256]
            shapes[prefix + ".out_layer.weight"] = [hidden, hidden]
        }
        for mode in ["image", "image_cond", "txt"] {
            shapes["early_stream_modulations.\(mode).lin.weight"] = [3 * hidden, hidden]
            shapes["single_stream_modulations.\(mode).lin.weight"] = [3 * hidden, hidden]
        }
        let early = ["content_mode_blocks.image", "content_mode_blocks.image_cond", "txt_mode_blocks"]
        let blocks = early.flatMap { prefix in (0..<config.earlyDepth).map { "\(prefix).\($0)" } }
            + (0..<config.singleDepth).map { "single_blocks.\($0)" }
        for prefix in blocks {
            for name in ["q_proj", "k_proj", "v_proj", "attn_out"] {
                shapes[prefix + ".\(name).weight"] = [hidden, hidden]
            }
            shapes[prefix + ".mlp_in.weight"] = [2 * mlp, hidden]
            shapes[prefix + ".mlp_out.weight"] = [hidden, mlp]
            shapes[prefix + ".norm.query_norm.scale"] = [hidden / config.heads]
            shapes[prefix + ".norm.key_norm.scale"] = [hidden / config.heads]
        }
        return shapes
    }

    /// Inputs have shape [tokens, channels], IDs [tokens, 4]. Empty reference streams
    /// are omitted entirely; the upstream phantom token is never visible to attention.
    public func callAsFunction(
        image: MLXArray, imageIDs: MLXArray, text: MLXArray, textIDs: MLXArray,
        references: MLXArray? = nil, referenceIDs: MLXArray? = nil,
        timestep: Float, nextTimestep: Float
    ) throws -> MLXArray {
        let config = configuration
        guard image.ndim == 2, image.dim(0) > 0, image.dim(1) == config.imageChannels,
              text.ndim == 2, text.dim(0) > 0, text.dim(1) == config.textChannels,
              imageIDs.shape == [image.dim(0), 4], textIDs.shape == [text.dim(0), 4],
              timestep.isFinite, nextTimestep.isFinite, (0...1).contains(timestep),
              (0..<timestep).contains(nextTimestep),
              (references == nil) == (referenceIDs == nil) else {
            throw Flux3Error.invalidConfiguration("Invalid stream shapes or flow-map times.")
        }
        if let references, let referenceIDs {
            guard references.ndim == 2, references.dim(0) > 0, references.dim(1) == config.imageChannels,
                  referenceIDs.shape == [references.dim(0), 4] else {
                throw Flux3Error.invalidConfiguration("Reference tokens and IDs must agree.")
            }
        }
        let dtype = weights["txt_in.weight"]!.dtype
        let targetTime = embedTime(nextTimestep, prefix: "ref_time_in", dtype: dtype)
        let imageTime = embedTime(timestep, prefix: "time_in", dtype: dtype) + targetTime
        let conditionTime = embedTime(0, prefix: "time_in", dtype: dtype) + targetTime
        var streams = [linear(image.asType(dtype), "emb_in.image")]
        var ids = [imageIDs]
        var modes = ["image"]
        var vectors = [imageTime]
        var prefixes = ["content_mode_blocks.image"]
        if let references, let referenceIDs {
            streams.append(linear(references.asType(dtype), "emb_in.image_cond"))
            ids.append(referenceIDs)
            modes.append("image_cond")
            vectors.append(conditionTime)
            prefixes.append("content_mode_blocks.image_cond")
        }
        streams.append(linear(text.asType(dtype), "txt_in"))
        ids.append(textIDs)
        modes.append("txt")
        vectors.append(conditionTime)
        prefixes.append("txt_mode_blocks")
        let rotations = ids.map { rope($0) }
        for mode in streams.indices {
            let modulation = modulate(vectors[mode], prefix: "early_stream_modulations.\(modes[mode]).lin")
            for depth in 0..<config.earlyDepth {
                streams[mode] = block(streams[mode], prefix: "\(prefixes[mode]).\(depth)",
                                      rotation: rotations[mode], modulation: modulation)
                eval(streams[mode])
            }
        }
        // Crucial: image, image_cond, text, each with its own time modulation.
        let mods = streams.indices.map { index -> [MLXArray] in
            modulate(vectors[index], prefix: "single_stream_modulations.\(modes[index]).lin")
                .map { broadcast($0, to: [streams[index].dim(0), config.hiddenSize]) }
        }
        let combinedMods = (0..<3).map { item in concatenated(mods.map { $0[item] }, axis: 0) }
        let rotation = rope(concatenated(ids, axis: 0))
        var combined = concatenated(streams, axis: 0)
        for depth in 0..<config.singleDepth {
            combined = block(combined, prefix: "single_blocks.\(depth)", rotation: rotation, modulation: combinedMods)
            eval(combined)
        }
        let output = combined[0..<image.dim(0)]
        let final = MLX.split(linear(silu(imageTime), "final_layer.image.adaLN_modulation.1"), parts: 2, axis: -1)
        return linear(normalize(output) * (1 + final[1]) + final[0], "final_layer.image.linear")
    }

    private func linear(_ input: MLXArray, _ prefix: String) -> MLXArray {
        matmul(input, weights[prefix + ".weight"]!.T)
    }

    private func normalize(_ input: MLXArray) -> MLXArray {
        MLXFast.layerNorm(input, weight: nil, bias: nil, eps: 1e-6)
    }

    private func modulate(_ vector: MLXArray, prefix: String) -> [MLXArray] {
        MLX.split(linear(silu(vector), prefix), parts: 3, axis: -1)
    }

    private func embedTime(_ time: Float, prefix: String, dtype: DType) -> MLXArray {
        // Reference casts the scalar to BF16 *before* the 1000x projection.
        let rounded = (MLXArray([time]).asType(dtype) * 1000).asType(.float32)
        let angles = rounded.reshaped(1, 1)
            * exp(MLXArray(0..<128).asType(.float32) * (-log(Float(10000)) / 128))
        let embedding = concatenated([cos(angles), sin(angles)], axis: -1).asType(dtype)
        return linear(silu(linear(embedding, prefix + ".in_layer")), prefix + ".out_layer")
    }

    private func block(_ input: MLXArray, prefix: String, rotation: (MLXArray, MLXArray),
                       modulation: [MLXArray]) -> MLXArray {
        let config = configuration, length = input.dim(0), headDim = configuration.hiddenSize / configuration.heads
        let normalized = normalize(input) * (1 + modulation[1]) + modulation[0]
        func heads(_ name: String) -> MLXArray {
            linear(normalized, prefix + "." + name).reshaped(1, length, config.heads, headDim).transposed(0, 2, 1, 3)
        }
        func queryKey(_ name: String, norm: String) -> MLXArray {
            let projected = heads(name)
            let normalized = MLXFast.rmsNorm(projected.asType(.float32),
                weight: weights[prefix + ".norm.\(norm)_norm.scale"]!.asType(.float32), eps: 1e-6)
            return Flux2PosEmbed.applyRotaryEmb(normalized, freqs: rotation).asType(projected.dtype)
        }
        let attention = MLXFast.scaledDotProductAttention(
            queries: queryKey("q_proj", norm: "query"), keys: queryKey("k_proj", norm: "key"),
            values: heads("v_proj"), scale: 1 / sqrt(Float(headDim)), mask: nil)
            .transposed(0, 2, 1, 3).reshaped(length, config.hiddenSize)
        let feedForward = MLX.split(linear(normalized, prefix + ".mlp_in"), parts: 2, axis: -1)
        let update = linear(attention, prefix + ".attn_out")
            + linear(silu(feedForward[0]) * feedForward[1], prefix + ".mlp_out")
        return input + modulation[2] * update
    }
}
