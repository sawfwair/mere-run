import Foundation
import MLX
import MLXNN

struct MiniMaxH3ImageMLXFormat: Codable, Sendable {
    static let filename = "h3_image_mlx.json"
    static let cacheFilename = "image_modulations.safetensors"
    static let name = "mere.run.h3-image-q8-g64-v1"
    let format: String
    let bits: Int
    let groupSize: Int
    let sourceRevision: String
    let sourceIdentity: String

    func validate() throws {
        guard format == Self.name, bits == 8, groupSize == 64,
              sourceRevision == MiniMaxH3ImageResources.revision else {
            throw ImageGenerationIssue("h3_image_checkpoint_invalid", "Unsupported native H3-Image Q8 format.")
        }
    }

    static func cacheArrays(_ cache: MiniMaxH3ImageModulationCache) -> [String: MLXArray] {
        var arrays = ["time_embeddings": cache.timeEmbeddings, "final_modulations": cache.finalModulations,
                      "sigmas": MLXArray(MiniMaxH3ImageSampling.sigmas)]
        for (index, value) in cache.blocks.enumerated() { arrays["blocks.\(index)"] = value }
        return arrays
    }

    func loadCache(_ resources: MiniMaxH3ImageResources,
                   configuration: MiniMaxH3TransformerConfiguration) throws -> MiniMaxH3ImageModulationCache {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: resources.rootURL.appending(path: Self.cacheFilename))
        guard metadata["format"] == Self.name, metadata["source_identity"] == sourceIdentity,
              let time = arrays["time_embeddings"], let final = arrays["final_modulations"],
              arrays["sigmas"]?.asArray(Float.self) == MiniMaxH3ImageSampling.sigmas else {
            throw resources.invalid("The exact image timestep cache does not match this checkpoint.")
        }
        let blocks = try (0..<configuration.layerCount).map { index in
            guard let value = arrays["blocks.\(index)"] else { throw resources.invalid("Missing cached block \(index).") }
            return value
        }
        let cache = MiniMaxH3ImageModulationCache(timeEmbeddings: time, blocks: blocks, finalModulations: final)
        guard cache.matches(configuration), arrays.count == configuration.layerCount + 3 else {
            throw resources.invalid("Invalid image timestep cache dimensions.")
        }
        return cache
    }

    static func isCacheWeight(_ key: String) -> Bool {
        key.hasPrefix("time_embedder.") || key.contains(".adaln_proj.")
    }

    /// Native Q8 keys already have global Q/K/V slabs; never deinterleave packed codes again.
    static func apply(_ arrays: [String: MLXArray], to model: Module) throws {
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var replacements = leaves
        var replaced = false
        for key in arrays.keys.sorted() where key.hasSuffix(".scales") {
            let path = String(key.dropLast(".scales".count))
            guard let linear = leaves[path] as? Linear, let codes = arrays[path + ".weight"],
                  let scales = arrays[key], let biases = arrays[path + ".biases"],
                  codes.dtype == .uint32, scales.dtype == .bfloat16, biases.dtype == .bfloat16,
                  codes.shape == [linear.weight.dim(0), linear.weight.dim(1) / 4],
                  scales.shape == [linear.weight.dim(0), linear.weight.dim(1) / 64], biases.shape == scales.shape else {
                throw ImageGenerationIssue("h3_image_checkpoint_invalid", "Invalid Q8 linear tensors at \(path).")
            }
            replacements[path] = QuantizedLinear(weight: codes,
                bias: arrays[path + ".bias"] ?? linear.bias, scales: scales, biases: biases, groupSize: 64, bits: 8)
            replaced = true
        }
        // MLX module-array updates require a complete structure. Preserve the
        // other layers when this shard replaces only a sparse subset of them.
        if replaced { model.update(modules: ModuleChildren.unflattened(replacements.map { ($0.key, $0.value) })) }
        try model.update(parameters: ModuleParameters.unflattened(arrays.map { ($0.key, $0.value) }),
                         verify: [.shapeMismatch, .noUnusedKeys])
    }
}
