import Foundation
import MLX
import MLXFast
import MLXNN

/// Full Gemma 4 vision tower, distinct from the compact Gemma chat patch embedder.
public final class EmbeddingGemma2VisionModel {
    public let config: EmbeddingGemma2VisionConfig
    private let weights: EmbeddingGemma2MediaTensorStore
    private let textHiddenSize: Int

    public init(config: EmbeddingGemma2VisionConfig, textHiddenSize: Int,
                tensors: [String: MLXArray], dtype: DType = .bfloat16) throws {
        try config.validate()
        self.config = config
        self.textHiddenSize = textHiddenSize
        weights = try EmbeddingGemma2MediaTensorStore(
            tensors: tensors, shapes: Self.tensorShapes(config: config, textHiddenSize: textHiddenSize), dtype: dtype)
    }

    public static func tensorShapes(config: EmbeddingGemma2VisionConfig, textHiddenSize: Int) -> [String: [Int]] {
        let h = config.hiddenSize, d = config.headDim
        var shapes = [
            "vision_tower.patch_embedder.input_proj.weight": [h, 3 * config.patchSize * config.patchSize],
            "vision_tower.patch_embedder.position_embedding_table": [2, config.positionEmbeddingSize, h],
            "embed_vision.embedding_projection.weight": [textHiddenSize, h]
        ]
        for index in 0..<config.numHiddenLayers {
            let key = "vision_tower.encoder.layers.\(index)"
            for (name, input, output) in [
                ("self_attn.q_proj", h, config.numAttentionHeads * d),
                ("self_attn.k_proj", h, config.numKeyValueHeads * d),
                ("self_attn.v_proj", h, config.numKeyValueHeads * d),
                ("self_attn.o_proj", config.numAttentionHeads * d, h),
                ("mlp.gate_proj", h, config.intermediateSize), ("mlp.up_proj", h, config.intermediateSize),
                ("mlp.down_proj", config.intermediateSize, h)
            ] { shapes[key + "." + name + ".linear.weight"] = [output, input] }
            for name in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                shapes[key + "." + name + ".weight"] = [h]
            }
            for name in ["q_norm", "k_norm"] { shapes[key + ".self_attn." + name + ".weight"] = [d] }
        }
        return shapes
    }

    /// Returns only real pooled soft tokens, already projected to the text hidden size.
    public func callAsFunction(pixels: MLXArray, positions: [[Int]]) throws -> MLXArray {
        guard pixels.ndim == 3 else { throw EmbeddingGemma2Error.invalidInput("Vision patches must have three dimensions.") }
        let length = pixels.dim(1), patchDim = 3 * config.patchSize * config.patchSize
        let kernel = config.poolingKernelSize
        guard pixels.shape == [1, length, patchDim], positions.count == length,
              length.isMultiple(of: kernel * kernel), positions.allSatisfy({
                  $0.count == 2 && ($0 == [-1, -1] || $0.allSatisfy { (0..<config.positionEmbeddingSize).contains($0) })
              }), positions.contains(where: { $0 != [-1, -1] }) else {
            throw EmbeddingGemma2Error.invalidInput("Invalid vision patches or positions.")
        }
        let real = positions.filter { $0 != [-1, -1] }
        let gridWidth = real.map { $0[0] }.max()! + 1
        let gridHeight = real.map { $0[1] }.max()! + 1
        guard gridWidth.isMultiple(of: kernel), gridHeight.isMultiple(of: kernel),
              real.count == gridWidth * gridHeight, Set(real).count == real.count else {
            throw EmbeddingGemma2Error.invalidInput("Vision positions must describe a complete spatial pooling grid.")
        }
        let ids = MLXArray(positions.flatMap { $0.map(Int32.init) }, [1, length, 2])
        let valid = MLXArray(positions.map { Int32($0 == [-1, -1] ? 0 : 1) }, [1, length])
        let table = weights["vision_tower.patch_embedder.position_embedding_table"]
        let pos = (take(table[0], maximum(ids[0..., 0..., 0], MLXArray(Int32(0))), axis: 0)
                   + take(table[1], maximum(ids[0..., 0..., 1], MLXArray(Int32(0))), axis: 0))
            * valid.asType(weights.dtype).expandedDimensions(axis: -1)
        var hidden = weights.linear((2 * (pixels - MLXArray(Float(0.5)))).asType(weights.dtype), "vision_tower.patch_embedder.input_proj") + pos
        let mask = EmbeddingGemma2TextModel.attentionMask(validTokens: valid, window: nil, dtype: hidden.dtype)
        for index in 0..<config.numHiddenLayers {
            let key = "vision_tower.encoder.layers.\(index)"
            let x = weights.norm(hidden, key + ".input_layernorm", eps: config.rmsNormEps)
            let q = axial(weights.norm(weights.linear(x, key + ".self_attn.q_proj.linear")
                .reshaped(1, length, config.numAttentionHeads, config.headDim), key + ".self_attn.q_norm", eps: config.rmsNormEps), positions: ids)
            let k = axial(weights.norm(weights.linear(x, key + ".self_attn.k_proj.linear")
                .reshaped(1, length, config.numKeyValueHeads, config.headDim), key + ".self_attn.k_norm", eps: config.rmsNormEps), positions: ids)
            let v = EmbeddingGemma2RMSNorm.normalize(weights.linear(x, key + ".self_attn.v_proj.linear")
                .reshaped(1, length, config.numKeyValueHeads, config.headDim), eps: config.rmsNormEps).asType(x.dtype)
            let attended = MLXFast.scaledDotProductAttention(queries: q.transposed(0, 2, 1, 3),
                keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3), scale: 1, mask: .array(mask))
                .transposed(0, 2, 1, 3).reshaped(1, length, -1)
            hidden = hidden + weights.norm(weights.linear(attended, key + ".self_attn.o_proj.linear"), key + ".post_attention_layernorm", eps: config.rmsNormEps)
            let normalized = weights.norm(hidden, key + ".pre_feedforward_layernorm", eps: config.rmsNormEps)
            let mlp = weights.linear(geluApproximate(weights.linear(normalized, key + ".mlp.gate_proj.linear"))
                * weights.linear(normalized, key + ".mlp.up_proj.linear"), key + ".mlp.down_proj.linear")
            hidden = hidden + weights.norm(mlp, key + ".post_feedforward_layernorm", eps: config.rmsNormEps)
        }
        let count = real.count / (kernel * kernel)
        var assignment = [Float](repeating: 0, count: count * length)
        for (index, position) in positions.enumerated() where position != [-1, -1] {
            let group = position[0] / kernel + (gridWidth / kernel) * (position[1] / kernel)
            guard group < count else { throw EmbeddingGemma2Error.invalidInput("Invalid spatial pooling grid.") }
            assignment[group * length + index] = 1 / Float(kernel * kernel)
        }
        let pooled = matmul(MLXArray(assignment, [count, length]), hidden[0].asType(.float32)).asType(hidden.dtype)
        let scaled = (pooled.asType(.float32) * MLXArray(Float(config.hiddenSize).squareRoot())).asType(hidden.dtype)
        return weights.linear(EmbeddingGemma2RMSNorm.normalize(scaled, eps: config.rmsNormEps).asType(hidden.dtype), "embed_vision.embedding_projection")
    }

    private func axial(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let spatial = config.headDim / 2, quarter = spatial / 2
        let inverse = pow(MLXArray(config.ropeParameters.ropeTheta),
                          -MLXArray((0..<quarter).map { Float(2 * $0) / Float(spatial) }))
        return concatenated((0..<2).map { axis in
            let angles = positions[0..., 0..., axis].asType(.float32).expandedDimensions(axis: -1) * inverse
            let c = cos(angles).asType(x.dtype).expandedDimensions(axis: 2)
            let s = sin(angles).asType(x.dtype).expandedDimensions(axis: 2)
            let a = x[0..., 0..., 0..., (axis * spatial)..<(axis * spatial + quarter)]
            let b = x[0..., 0..., 0..., (axis * spatial + quarter)..<((axis + 1) * spatial)]
            return concatenated([a * c - b * s, b * c + a * s], axis: -1)
        }, axis: -1)
    }
}
