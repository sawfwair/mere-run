import Foundation
import MLX
import MLXNN
import MLXRandom

public final class IdolDecoder {
    public let configuration: IdolConfiguration.Decoder
    public let gaussian: Bool
    private let layers: IdolLayers

    public init(configuration: IdolConfiguration, gaussian: Bool, weights: [String: MLXArray]) throws {
        try configuration.validate()
        self.configuration = gaussian ? configuration.decoder_gs.args : configuration.decoder.args
        self.gaussian = gaussian
        layers = IdolLayers(weights: try IdolWeights(weights,
            shapes: IdolWeightSchema.decoder(self.configuration, gaussian: gaussian)), heads: self.configuration.num_heads)
    }

    /// Octree queries are independent; chunking does not change cross-attention context.
    /// Gaussian queries share self-attention and must be evaluated together.
    public func predict(points: MLXArray, latent: MLXArray, resolution: Int = 2) throws -> MLXArray {
        guard points.ndim == 3, points.dim(0) == 1, points.dim(2) == 3, points.dim(1) > 0,
              latent.ndim == 3, latent.dim(0) == 1, latent.dim(2) == configuration.cond_channels else {
            throw IdolError.invalidInput("Decoder expects [1,points,3] queries and [1,tokens,channels] latent.")
        }
        let weights = layers.weights
        let embedded = weights.linear(points, "in_proj")
            + IdolLayers.positionEmbedding(points, width: configuration.model_channels, decoder: true)
        var hidden = weights.linear(embedded, "input_layer").asType(.float16)
        let modulation = gaussian ? nil : weights.linear(silu(layers.scalarEmbedding(MLXArray([Float(resolution)]),
            "l_embedder", period: 1024, twoPi: true)), "adaLN_modulation.1").asType(.float16)
        for index in 0..<configuration.num_blocks {
            let name = "blocks.\(index)"
            if let modulation {
                hidden = layers.modulatedBlock(hidden, name, modulation: modulation, context: latent.asType(.float16))
            } else {
                hidden = hidden + layers.attention(layers.norm(hidden), name + ".self_attn")
                hidden = hidden + layers.attention(layers.norm(hidden, name + ".norm2"), name + ".cross_attn",
                                                   context: latent.asType(.float16))
                hidden = hidden + layers.ffn(layers.norm(hidden), name + ".mlp")
            }
            eval(hidden)
        }
        return weights.linear(layers.norm(hidden.asType(.float32), eps: 1e-5), "out_proj")
    }

    public func anchors(latent: MLXArray, count: Int, level: Int, queryChunkSize: Int = 1024) throws -> MLXArray {
        guard !gaussian, count > 0, count <= 32768, (1...16).contains(level), queryChunkSize > 0 else {
            throw IdolError.invalidInput("Invalid octree sampling request.")
        }
        var coordinates = [[Int]](repeating: [0, 0, 0], count: 1), counts = [count]
        for depth in 1...level {
            try Task.checkCancellation()
            let resolution = 1 << (depth - 1)
            let points = coordinates.flatMap { $0.map { (Float($0) + 0.5) / Float(resolution) } }
            var probabilities: [Float] = []
            for start in stride(from: 0, to: coordinates.count, by: queryChunkSize) {
                let end = min(start + queryChunkSize, coordinates.count)
                let query = MLXArray(Array(points[(start * 3)..<(end * 3)])).reshaped(1, end - start, 3)
                let logits = try predict(points: query, latent: latent, resolution: 1 << depth)
                probabilities += softmax(logits, axis: -1).asArray(Float.self)
            }
            let offsets = MLXRandom.uniform(0..<1, [counts.count]).asArray(Float.self)
            var nextCoordinates: [[Int]] = [], nextCounts: [Int] = []
            for parent in counts.indices {
                let split = try Self.systematicCounts(probabilities: Array(probabilities[(parent * 8)..<(parent * 8 + 8)]),
                                                      count: counts[parent], offset: offsets[parent])
                for child in 0..<8 where split[child] > 0 {
                    nextCoordinates.append([coordinates[parent][0] * 2 + (child & 1),
                        coordinates[parent][1] * 2 + ((child >> 1) & 1), coordinates[parent][2] * 2 + ((child >> 2) & 1)])
                    nextCounts.append(split[child])
                }
            }
            coordinates = nextCoordinates
            counts = nextCounts
        }
        let repeated = coordinates.indices.flatMap { index in
            Array(repeating: coordinates[index].map(Float.init), count: counts[index]).flatMap { $0 }
        }
        return (MLXArray(repeated).reshaped(1, count, 3) + MLXRandom.uniform(0..<1, [1, count, 3])) / Float(1 << level)
    }

    /// offset is uniform in [0,1); the first stratum starts at offset/count.
    public static func systematicCounts(probabilities: [Float], count: Int, offset: Float) throws -> [Int] {
        guard !probabilities.isEmpty, count >= 0, offset.isFinite, (0..<1).contains(offset),
              probabilities.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw IdolError.invalidInput("Invalid systematic resampling distribution.")
        }
        let total = probabilities.reduce(0, +)
        let normalized = total == 0 ? Array(repeating: 1 / Float(probabilities.count), count: probabilities.count)
            : probabilities.map { $0 / max(total, 1) }
        var cdf: [Float] = [], sum: Float = 0
        for probability in normalized { sum += probability; cdf.append(min(sum, 1)) }
        var result = [Int](repeating: 0, count: probabilities.count), bin = 0
        for sample in 0..<count {
            let position = min((offset + Float(sample)) / Float(count), 1)
            while bin < cdf.count - 1 && position > cdf[bin] { bin += 1 }
            result[bin] += 1
        }
        return result
    }

    public func representation(points: MLXArray, features: MLXArray) throws -> IdolGaussians {
        guard gaussian, let config = configuration.representation_config,
              points.ndim == 3, points.dim(0) == 1, points.dim(2) == 3,
              features.shape == [1, points.dim(1), 15 * config.num_gaussians] else {
            throw IdolError.invalidInput("Gaussian feature layout does not match the checkpoint.")
        }
        let perAnchor = config.num_gaussians, anchors = points.dim(1)
        var start = 0
        func take(_ width: Int) -> MLXArray {
            let end = start + perAnchor * width
            let result = features[0..., 0..., start..<end].reshaped(anchors, perAnchor, width)
            start = end
            return result
        }
        let xyz = take(3), dc = take(3) * config.lr._features_dc, scaling = take(3) * config.lr._scaling
        let rotation = take(4) * config.lr._rotation, opacity = take(1) * config.lr._opacity
        let offsetScale = softplus(take(1) + layers.weights["base_offset_scale"])
        let offset = tanh(xyz * config.lr._xyz + layers.weights["points_offset_perturbation"])
            * 0.5 * config.perturbe_size * offsetScale
        return IdolGaussians(rawPositions: (offset + points[0].expandedDimensions(axis: 1)).reshaped(-1, 3),
            featuresDC: dc.reshaped(-1, 3), rawScaling: scaling.reshaped(-1, 3), rawRotation: rotation.reshaped(-1, 4),
            rawOpacity: opacity.reshaped(-1, 1), configuration: config)
    }
}
